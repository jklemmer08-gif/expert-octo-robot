"""Background removal pipeline using RVM (Robust Video Matting).

Pipeline per segment:
1. Extract N frames from input to temp PNGs (via FFmpeg)
2. Process frames in batches through RVM (GPU, batch-based)
3. Save RGBA PNGs, encode to VP9 WebM segment (with alpha)
4. Delete temp frames (recurrent states persist in GPU memory)

VR frames are split per-eye with separate RVM instances per eye.
Output is VP9 WebM with yuva420p pixel format for alpha transparency.
"""

import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # Tests can mock this

from src.config import (
    DEFAULT_RVM_MODEL,
    DEFAULT_DOWNSAMPLE_RATIO,
    DEFAULT_RVM_BATCH_SIZE,
    DEFAULT_VP9_CRF,
    SEGMENT_SIZE,
    TEMP_DIR,
)
from src.pipeline.detector import VRLayout
from src.pipeline.encoder import encode_segment_vp9, concatenate_segments, mux_audio_webm
from src.pipeline.rvm_model import RVMProcessor, RVMError
from src.storage.volume import (
    check_disk_space,
    cleanup_job,
    create_segment_dir,
    estimate_segment_disk_gb,
)
from src.utils.ffmpeg import (
    build_decode_pipe_cmd,
    build_encode_pipe_vp9_cmd,
    build_extract_frames_cmd,
    get_video_metadata,
    run_ffmpeg,
)
from src.utils.streaming import (
    close_process,
    read_frames,
    start_decode_process,
    start_encode_process,
    write_frame,
)

logger = logging.getLogger(__name__)


class BgRemoveError(Exception):
    """Raised when background removal fails after all retries."""
    pass


def _split_vr_frame(frame: np.ndarray, layout: VRLayout) -> Tuple[np.ndarray, np.ndarray]:
    """Split a stereo frame into left/right or top/bottom eyes."""
    h, w = frame.shape[:2]
    if layout == VRLayout.SBS:
        mid = w // 2
        return frame[:, :mid], frame[:, mid:]
    elif layout == VRLayout.OU:
        mid = h // 2
        return frame[:mid, :], frame[mid:, :]
    raise ValueError(f"Cannot split non-stereo layout: {layout}")


def _merge_vr_frame(left: np.ndarray, right: np.ndarray, layout: VRLayout) -> np.ndarray:
    """Merge two eyes back into a stereo frame."""
    if layout == VRLayout.SBS:
        return np.concatenate([left, right], axis=1)
    elif layout == VRLayout.OU:
        return np.concatenate([left, right], axis=0)
    raise ValueError(f"Cannot merge non-stereo layout: {layout}")


def _frames_to_tensor(frames_bgr: List[np.ndarray], device: str) -> "torch.Tensor":
    """Convert list of BGR numpy frames to (B, 3, H, W) float32 RGB tensor on GPU."""
    # BGR → RGB, uint8 → float32 [0, 1], (H, W, 3) → (3, H, W)
    tensors = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        tensors.append(t)
    batch = torch.stack(tensors).to(device)
    return batch


def _tensor_to_frames_bgra(rgba_tensor: "torch.Tensor") -> List[np.ndarray]:
    """Convert (B, 4, H, W) RGBA tensor to list of BGRA numpy arrays (uint8)."""
    frames = []
    batch_np = (rgba_tensor.cpu().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    for i in range(batch_np.shape[0]):
        rgba = batch_np[i]  # (H, W, 4) RGBA
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        frames.append(bgra)
    return frames


def _process_batch_with_oom_retry(
    processor: RVMProcessor,
    frames_bgr: List[np.ndarray],
    device: str,
    batch_size: int,
) -> Tuple[List[np.ndarray], int]:
    """Process a batch of frames, retrying with smaller batch on OOM.

    Returns (list_of_bgra_frames, effective_batch_size).
    """
    current_batch_size = batch_size
    while current_batch_size >= 1:
        results = []
        saved_states = processor.get_recurrent_states()
        try:
            for start in range(0, len(frames_bgr), current_batch_size):
                chunk = frames_bgr[start:start + current_batch_size]
                batch_tensor = _frames_to_tensor(chunk, device)
                rgba_out = processor.process_batch(batch_tensor)
                results.extend(_tensor_to_frames_bgra(rgba_out))
                del batch_tensor, rgba_out
            return results, current_batch_size
        except (RuntimeError,) as e:
            is_oom = "out of memory" in str(e).lower()
            if torch is not None:
                try:
                    is_oom = is_oom or isinstance(e, torch.cuda.OutOfMemoryError)
                except AttributeError:
                    pass
            if not is_oom:
                raise
            new_size = current_batch_size // 2
            logger.warning(
                "OOM with batch_size=%d, retrying with %d",
                current_batch_size, max(new_size, 1),
            )
            if torch is not None:
                torch.cuda.empty_cache()
            # Restore recurrent states to before this failed attempt
            processor.set_recurrent_states(saved_states)
            results.clear()
            if new_size < 1:
                raise BgRemoveError(
                    f"CUDA OOM even with batch_size=1. "
                    "Reduce input resolution or use a GPU with more VRAM."
                ) from e
            current_batch_size = new_size

    raise BgRemoveError("Batch processing failed")


def process_video(
    input_path: str,
    output_path: str,
    model_name: str = DEFAULT_RVM_MODEL,
    downsample_ratio: float = DEFAULT_DOWNSAMPLE_RATIO,
    batch_size: int = DEFAULT_RVM_BATCH_SIZE,
    crf: int = DEFAULT_VP9_CRF,
    layout: VRLayout = VRLayout.FLAT_2D,
    device: str = "cuda:0",
    segment_size: int = SEGMENT_SIZE,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Run the full chunked background removal pipeline on a video.

    Returns a result dict with status, timing, resolution info.
    """
    job_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    segment_clips: List[str] = []

    try:
        # --- Get video metadata ---
        meta = get_video_metadata(input_path)
        total_frames = meta["num_frames"]
        fps = meta["fps"]
        in_w, in_h = meta["width"], meta["height"]

        # Output dimensions = input dimensions (bg removal doesn't change resolution)
        out_w, out_h = in_w, in_h

        logger.info(
            "BG removal: %s (%dx%d, %d frames, %s layout, job=%s)",
            input_path, in_w, in_h, total_frames, layout.value, job_id,
        )

        # --- Estimate disk space (RGBA PNGs are ~33% larger than RGB) ---
        needed_gb = estimate_segment_disk_gb(in_w, in_h, segment_size, scale=1) * 2.0
        if not check_disk_space(needed_gb):
            raise BgRemoveError(
                f"Insufficient disk space: need ~{needed_gb:.1f} GB free for segment processing"
            )

        # --- Load RVM model(s) ---
        is_vr = layout in (VRLayout.SBS, VRLayout.OU)
        if is_vr:
            processor_left = RVMProcessor(model_name, device, downsample_ratio)
            processor_right = RVMProcessor(model_name, device, downsample_ratio)
        else:
            processor = RVMProcessor(model_name, device, downsample_ratio)

        effective_batch = batch_size

        # --- Process segments ---
        num_segments = (total_frames + segment_size - 1) // segment_size
        frames_processed = 0

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_size
            seg_frames = min(segment_size, total_frames - seg_start)
            seg_dir = create_segment_dir(job_id, seg_idx)

            try:
                # 1. Extract frames
                if progress_callback:
                    progress_callback({
                        "stage": "extracting",
                        "segment": seg_idx + 1,
                        "total_segments": num_segments,
                        "frame": frames_processed,
                        "total_frames": total_frames,
                    })

                extract_dir = seg_dir / "input"
                extract_dir.mkdir(exist_ok=True)
                extract_cmd = build_extract_frames_cmd(
                    input_path,
                    str(extract_dir / "frame_%06d.png"),
                    start_frame=seg_start,
                    num_frames=seg_frames,
                    fps=fps,
                )
                run_ffmpeg(extract_cmd)

                # 2. Process frames in batches
                output_dir = seg_dir / "output"
                output_dir.mkdir(exist_ok=True)

                frame_files = sorted(extract_dir.glob("frame_*.png"))
                all_bgr_frames = []
                for frame_file in frame_files:
                    bgr = cv2.imread(str(frame_file))
                    if bgr is None:
                        logger.warning("Failed to read frame: %s", frame_file)
                        continue
                    all_bgr_frames.append(bgr)

                if progress_callback:
                    progress_callback({
                        "stage": "removing_background",
                        "segment": seg_idx + 1,
                        "total_segments": num_segments,
                        "frame": frames_processed,
                        "total_frames": total_frames,
                    })

                if is_vr:
                    # VR: split → process each eye separately → merge RGBA
                    result_frames = _process_vr_batch(
                        all_bgr_frames, layout, processor_left, processor_right,
                        device, effective_batch,
                    )
                else:
                    result_frames, effective_batch = _process_batch_with_oom_retry(
                        processor, all_bgr_frames, device, effective_batch,
                    )

                # Save RGBA frames as PNGs
                for i, bgra_frame in enumerate(result_frames):
                    out_name = f"frame_{i+1:06d}.png"
                    cv2.imwrite(str(output_dir / out_name), bgra_frame)

                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback({
                            "stage": "removing_background",
                            "segment": seg_idx + 1,
                            "total_segments": num_segments,
                            "frame": frames_processed + i + 1,
                            "total_frames": total_frames,
                        })

                frames_processed += len(all_bgr_frames)

                # Free frame memory
                del all_bgr_frames, result_frames

                # 3. Encode segment to VP9 WebM
                if progress_callback:
                    progress_callback({
                        "stage": "encoding",
                        "segment": seg_idx + 1,
                        "total_segments": num_segments,
                        "frame": frames_processed,
                        "total_frames": total_frames,
                    })

                seg_clip_path = str(seg_dir / f"segment_{seg_idx:04d}.webm")
                encode_segment_vp9(
                    frame_pattern=str(output_dir / "frame_%06d.png"),
                    output_path=seg_clip_path,
                    fps=fps,
                    width=out_w,
                    height=out_h,
                    crf=crf,
                )
                segment_clips.append(seg_clip_path)

            finally:
                # 4. Clean up frames (keep segment clip, recurrent states stay in GPU memory)
                for subdir in ["input", "output"]:
                    d = seg_dir / subdir
                    if d.exists():
                        shutil.rmtree(d)
                logger.info("Cleaned frames for segment %d", seg_idx)

        # --- Concatenate segments ---
        if progress_callback:
            progress_callback({
                "stage": "concatenating",
                "frame": total_frames,
                "total_frames": total_frames,
            })

        job_temp_dir = str(Path(segment_clips[0]).parent.parent)
        video_only_path = os.path.join(job_temp_dir, "video_only.webm")

        if len(segment_clips) == 1:
            os.rename(segment_clips[0], video_only_path)
        else:
            concatenate_segments(segment_clips, video_only_path, job_temp_dir)

        # --- Mux audio (Opus for WebM) ---
        if progress_callback:
            progress_callback({
                "stage": "muxing_audio",
                "frame": total_frames,
                "total_frames": total_frames,
            })

        mux_audio_webm(video_only_path, input_path, output_path)

        elapsed = time.time() - start_time

        result = {
            "status": "success",
            "input_path": input_path,
            "output_path": output_path,
            "input_resolution": f"{in_w}x{in_h}",
            "output_resolution": f"{out_w}x{out_h}",
            "total_frames": total_frames,
            "layout": layout.value,
            "model": model_name,
            "batch_size": effective_batch,
            "crf": crf,
            "processing_time_sec": round(elapsed, 1),
            "avg_fps": round(total_frames / elapsed, 2) if elapsed > 0 else 0,
        }

        logger.info("BG removal complete: %s", result)
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("BG removal failed after %.1fs: %s", elapsed, e, exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "input_path": input_path,
            "processing_time_sec": round(elapsed, 1),
        }

    finally:
        cleanup_job(job_id)


def _process_vr_batch(
    frames_bgr: List[np.ndarray],
    layout: VRLayout,
    processor_left: RVMProcessor,
    processor_right: RVMProcessor,
    device: str,
    batch_size: int,
) -> List[np.ndarray]:
    """Process VR frames by splitting into eyes, running RVM separately, merging RGBA."""
    left_frames = []
    right_frames = []
    for bgr in frames_bgr:
        left, right = _split_vr_frame(bgr, layout)
        left_frames.append(left)
        right_frames.append(right)

    left_results, _ = _process_batch_with_oom_retry(
        processor_left, left_frames, device, batch_size,
    )
    right_results, _ = _process_batch_with_oom_retry(
        processor_right, right_frames, device, batch_size,
    )

    merged = []
    for left_bgra, right_bgra in zip(left_results, right_results):
        merged.append(_merge_vr_frame(left_bgra, right_bgra, layout))

    return merged


def process_video_streaming(
    input_path: str,
    output_path: str,
    model_name: str = DEFAULT_RVM_MODEL,
    downsample_ratio: float = DEFAULT_DOWNSAMPLE_RATIO,
    batch_size: int = DEFAULT_RVM_BATCH_SIZE,
    crf: int = DEFAULT_VP9_CRF,
    layout: VRLayout = VRLayout.FLAT_2D,
    device: str = "cuda:0",
    segment_size: int = SEGMENT_SIZE,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Run background removal using FFmpeg pipe streaming (no PNG disk I/O).

    Same interface as process_video() but reads/writes raw frames via pipes.
    Batch-reads frames from decode pipe, processes through RVM, writes BGRA to
    VP9 encode pipe.
    """
    from src.pipeline.encoder import concatenate_segments, mux_audio_webm

    job_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    segment_clips: List[str] = []

    try:
        # --- Get video metadata ---
        meta = get_video_metadata(input_path)
        total_frames = meta["num_frames"]
        fps = meta["fps"]
        in_w, in_h = meta["width"], meta["height"]
        out_w, out_h = in_w, in_h

        logger.info(
            "Streaming BG removal: %s (%dx%d, %d frames, %s layout, job=%s)",
            input_path, in_w, in_h, total_frames, layout.value, job_id,
        )

        # --- Estimate disk space (streaming: only encoded segments) ---
        needed_gb = max(2.0, (total_frames / segment_size) * 1.0)
        if not check_disk_space(needed_gb):
            raise BgRemoveError(f"Insufficient disk space: need ~{needed_gb:.1f} GB free")

        # --- Load RVM model(s) ---
        is_vr = layout in (VRLayout.SBS, VRLayout.OU)
        if is_vr:
            processor_left = RVMProcessor(model_name, device, downsample_ratio)
            processor_right = RVMProcessor(model_name, device, downsample_ratio)
        else:
            processor = RVMProcessor(model_name, device, downsample_ratio)

        effective_batch = batch_size

        # --- Process segments ---
        num_segments = (total_frames + segment_size - 1) // segment_size
        frames_processed = 0
        job_dir = TEMP_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_size
            seg_frames = min(segment_size, total_frames - seg_start)

            seg_clip_path = str(job_dir / f"segment_{seg_idx:04d}.webm")

            if progress_callback:
                progress_callback({
                    "stage": "removing_background",
                    "segment": seg_idx + 1,
                    "total_segments": num_segments,
                    "frame": frames_processed,
                    "total_frames": total_frames,
                })

            # Start decode pipe
            decode_cmd = build_decode_pipe_cmd(
                input_path, seg_start, seg_frames, fps, in_w, in_h,
            )
            decode_proc = start_decode_process(decode_cmd)

            # Start VP9 encode pipe (BGRA input for alpha)
            encode_cmd = build_encode_pipe_vp9_cmd(
                fps, out_w, out_h, crf, seg_clip_path,
            )
            encode_proc = start_encode_process(encode_cmd)

            try:
                remaining = seg_frames
                while remaining > 0:
                    read_count = min(effective_batch, remaining)
                    batch_bgr = read_frames(decode_proc, in_w, in_h, read_count, channels=3)
                    if not batch_bgr:
                        break

                    if is_vr:
                        result_frames = _process_vr_batch(
                            batch_bgr, layout, processor_left, processor_right,
                            device, effective_batch,
                        )
                    else:
                        result_frames, effective_batch = _process_batch_with_oom_retry(
                            processor, batch_bgr, device, effective_batch,
                        )

                    for bgra_frame in result_frames:
                        write_frame(encode_proc, bgra_frame)

                    frames_processed += len(batch_bgr)
                    remaining -= len(batch_bgr)

                    if progress_callback:
                        progress_callback({
                            "stage": "removing_background",
                            "segment": seg_idx + 1,
                            "total_segments": num_segments,
                            "frame": frames_processed,
                            "total_frames": total_frames,
                        })

                    del batch_bgr, result_frames

            finally:
                close_process(decode_proc, "decode")
                close_process(encode_proc, "encode")

            segment_clips.append(seg_clip_path)
            logger.info("Segment %d encoded via pipe", seg_idx)

        # --- Concatenate segments ---
        if progress_callback:
            progress_callback({"stage": "concatenating", "frame": total_frames, "total_frames": total_frames})

        video_only_path = str(job_dir / "video_only.webm")

        if len(segment_clips) == 1:
            os.rename(segment_clips[0], video_only_path)
        else:
            concatenate_segments(segment_clips, video_only_path, str(job_dir))

        # --- Mux audio (Opus for WebM) ---
        if progress_callback:
            progress_callback({"stage": "muxing_audio", "frame": total_frames, "total_frames": total_frames})

        mux_audio_webm(video_only_path, input_path, output_path)

        elapsed = time.time() - start_time

        result = {
            "status": "success",
            "input_path": input_path,
            "output_path": output_path,
            "input_resolution": f"{in_w}x{in_h}",
            "output_resolution": f"{out_w}x{out_h}",
            "total_frames": total_frames,
            "layout": layout.value,
            "model": model_name,
            "batch_size": effective_batch,
            "crf": crf,
            "processing_time_sec": round(elapsed, 1),
            "avg_fps": round(frames_processed / elapsed, 2) if elapsed > 0 else 0,
            "streaming": True,
        }

        logger.info("Streaming BG removal complete: %s", result)
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Streaming BG removal failed after %.1fs: %s", elapsed, e, exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "input_path": input_path,
            "processing_time_sec": round(elapsed, 1),
        }

    finally:
        cleanup_job(job_id)
