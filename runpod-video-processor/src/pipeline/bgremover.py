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
    DEFAULT_HEVC_CRF,
    DEFAULT_RVM_MODEL,
    DEFAULT_DOWNSAMPLE_RATIO,
    DEFAULT_RVM_BATCH_SIZE,
    DEFAULT_VP9_CRF,
    SEGMENT_SIZE,
    TEMP_DIR,
    XALPHA_CRF,
    XALPHA_HEIGHT,
    XALPHA_KEYINT,
)
from src.pipeline.detector import VRLayout
from src.pipeline.encoder import (
    concatenate_segments,
    encode_segment_vp9,
    get_encoder_codec,
    mux_audio_mp4,
    mux_audio_webm,
)
from src.pipeline.metadata import build_metadata_flags, read_vr_metadata
from src.pipeline.rvm_model import RVMProcessor, RVMError
from src.storage.volume import (
    check_disk_space,
    cleanup_job,
    create_segment_dir,
    estimate_segment_disk_gb,
)
from src.utils.ffmpeg import (
    build_decode_pipe_cmd,
    build_encode_pipe_hevc_cmd,
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


def _tensor_to_frames_bgr_premultiplied(
    fgr: "torch.Tensor", pha: "torch.Tensor",
) -> List[np.ndarray]:
    """Convert separate fgr/pha tensors to premultiplied BGR frames (black BG).

    Args:
        fgr: (B, 3, H, W) float32 foreground RGB in [0, 1]
        pha: (B, 1, H, W) float32 alpha in [0, 1]

    Returns:
        List of (H, W, 3) uint8 BGR numpy arrays.
    """
    rgb_out = (fgr * pha).clamp(0, 1)
    batch_np = (rgb_out.cpu() * 255).byte().permute(0, 2, 3, 1).numpy()
    frames = []
    for i in range(batch_np.shape[0]):
        bgr = cv2.cvtColor(batch_np[i], cv2.COLOR_RGB2BGR)
        frames.append(bgr)
    return frames


def _tensor_to_alpha_frames_bgr(
    pha: "torch.Tensor", target_h: int, target_w: int,
) -> List[np.ndarray]:
    """Convert alpha tensor to downscaled BGR grayscale frames for XALPHA output.

    Args:
        pha: (B, 1, H, W) float32 alpha in [0, 1]
        target_h: target height (e.g. 480)
        target_w: target width (computed from aspect ratio)

    Returns:
        List of (target_h, target_w, 3) uint8 BGR where R=G=B=alpha_value.
    """
    small = torch.nn.functional.interpolate(
        pha, size=(target_h, target_w), mode="bilinear", align_corners=False,
    )
    rgb = small.expand(-1, 3, -1, -1)
    batch_np = (rgb.cpu().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    frames = []
    for i in range(batch_np.shape[0]):
        frames.append(batch_np[i].copy())
    return frames


def compute_xalpha_paths(
    input_path: str,
    layout: VRLayout,
    output_dir: Path,
) -> Tuple[str, str]:
    """Compute main and alpha output paths with VR suffixes for HereSphere.

    Returns (main_path, alpha_path).
    """
    stem = Path(input_path).stem

    # Only add VR suffix if not already present in the filename
    stem_upper = stem.upper()
    vr_suffix = ""
    if layout == VRLayout.SBS and not any(
        tag in stem_upper for tag in ("_LR", "_SBS", "_3DH")
    ):
        vr_suffix = "_LR"
    elif layout == VRLayout.OU and not any(
        tag in stem_upper for tag in ("_TB", "_OU", "_3DV")
    ):
        vr_suffix = "_TB"

    main_name = f"{stem}{vr_suffix}.mp4"
    alpha_name = f"{stem}{vr_suffix}_XALPHA.mp4"

    return str(output_dir / main_name), str(output_dir / alpha_name)


def compute_alpha_dimensions(width: int, height: int, alpha_height: int) -> Tuple[int, int]:
    """Compute alpha video dimensions maintaining aspect ratio with even values."""
    alpha_w = int(alpha_height * width / height)
    alpha_w += alpha_w % 2  # ensure even for HEVC
    alpha_h = alpha_height + (alpha_height % 2)
    return alpha_w, alpha_h


def _process_batch_split_with_oom_retry(
    processor: RVMProcessor,
    frames_bgr: List[np.ndarray],
    device: str,
    batch_size: int,
    alpha_h: int,
    alpha_w: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """Process frames returning (main_bgr, alpha_bgr) lists with OOM retry.

    Returns (main_frames, alpha_frames, effective_batch_size).
    """
    current_batch_size = batch_size
    while current_batch_size >= 1:
        main_results = []
        alpha_results = []
        saved_states = processor.get_recurrent_states()
        try:
            for start in range(0, len(frames_bgr), current_batch_size):
                chunk = frames_bgr[start:start + current_batch_size]
                batch_tensor = _frames_to_tensor(chunk, device)
                fgr, pha = processor.process_batch_split(batch_tensor)
                main_results.extend(_tensor_to_frames_bgr_premultiplied(fgr, pha))
                alpha_results.extend(_tensor_to_alpha_frames_bgr(pha, alpha_h, alpha_w))
                del batch_tensor, fgr, pha
            return main_results, alpha_results, current_batch_size
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
                "XALPHA OOM with batch_size=%d, retrying with %d",
                current_batch_size, max(new_size, 1),
            )
            if torch is not None:
                torch.cuda.empty_cache()
            processor.set_recurrent_states(saved_states)
            main_results.clear()
            alpha_results.clear()
            if new_size < 1:
                raise BgRemoveError(
                    "CUDA OOM even with batch_size=1. "
                    "Reduce input resolution or use a GPU with more VRAM."
                ) from e
            current_batch_size = new_size

    raise BgRemoveError("Batch processing failed")


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
                close_process(decode_proc, "decode", tolerant=True)
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


def _process_vr_batch_split(
    frames_bgr: List[np.ndarray],
    layout: VRLayout,
    processor_left: RVMProcessor,
    processor_right: RVMProcessor,
    device: str,
    batch_size: int,
    alpha_h: int,
    alpha_w: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Process VR frames with split output (main BGR + alpha BGR per eye)."""
    left_frames = []
    right_frames = []
    for bgr in frames_bgr:
        left, right = _split_vr_frame(bgr, layout)
        left_frames.append(left)
        right_frames.append(right)

    # Compute per-eye alpha dimensions (half width for SBS, half height for OU)
    if layout == VRLayout.SBS:
        eye_alpha_w = alpha_w // 2
        eye_alpha_h = alpha_h
    else:  # OU
        eye_alpha_w = alpha_w
        eye_alpha_h = alpha_h // 2

    left_main, left_alpha, _ = _process_batch_split_with_oom_retry(
        processor_left, left_frames, device, batch_size, eye_alpha_h, eye_alpha_w,
    )
    right_main, right_alpha, _ = _process_batch_split_with_oom_retry(
        processor_right, right_frames, device, batch_size, eye_alpha_h, eye_alpha_w,
    )

    merged_main = []
    merged_alpha = []
    for lm, rm in zip(left_main, right_main):
        merged_main.append(_merge_vr_frame(lm, rm, layout))
    for la, ra in zip(left_alpha, right_alpha):
        merged_alpha.append(_merge_vr_frame(la, ra, layout))

    return merged_main, merged_alpha


def _apply_vr_metadata(
    file_path: str, input_path: str, layout: VRLayout, job_dir: Path,
) -> str:
    """Apply VR metadata to a file via stream-copy remux. Returns final path."""
    vr_meta = read_vr_metadata(input_path)
    meta_flags = build_metadata_flags(vr_meta, layout)
    if not meta_flags:
        return file_path

    tagged_path = str(job_dir / f"tagged_{Path(file_path).name}")
    cmd = [
        "ffmpeg", "-y", "-i", file_path,
        "-c", "copy", *meta_flags,
        "-movflags", "+faststart",
        tagged_path,
    ]
    run_ffmpeg(cmd)
    os.replace(tagged_path, file_path)
    return file_path


def process_video_streaming_xalpha(
    input_path: str,
    output_path: str,
    alpha_output_path: str,
    model_name: str = DEFAULT_RVM_MODEL,
    downsample_ratio: float = DEFAULT_DOWNSAMPLE_RATIO,
    batch_size: int = DEFAULT_RVM_BATCH_SIZE,
    crf: int = DEFAULT_HEVC_CRF,
    alpha_crf: int = XALPHA_CRF,
    alpha_height: int = XALPHA_HEIGHT,
    keyint: int = XALPHA_KEYINT,
    layout: VRLayout = VRLayout.FLAT_2D,
    device: str = "cuda:0",
    segment_size: int = SEGMENT_SIZE,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Run BG removal with HereSphere External Alpha output (two files).

    Produces:
      output_path: HEVC MP4 (foreground on black, full resolution)
      alpha_output_path: HEVC MP4 (grayscale alpha mask, 480p)

    Both files have matching keyframe spacing for HereSphere seek sync.
    """
    job_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    main_segments: List[str] = []
    alpha_segments: List[str] = []

    try:
        meta = get_video_metadata(input_path)
        total_frames = meta["num_frames"]
        fps = meta["fps"]
        in_w, in_h = meta["width"], meta["height"]
        out_w, out_h = in_w, in_h

        alpha_w, alpha_h = compute_alpha_dimensions(out_w, out_h, alpha_height)
        codec = get_encoder_codec()

        logger.info(
            "XALPHA BG removal: %s (%dx%d → main %dx%d + alpha %dx%d, %d frames, %s, job=%s)",
            input_path, in_w, in_h, out_w, out_h, alpha_w, alpha_h,
            total_frames, layout.value, job_id,
        )

        needed_gb = max(2.0, (total_frames / segment_size) * 2.0)
        if not check_disk_space(needed_gb):
            raise BgRemoveError(f"Insufficient disk space: need ~{needed_gb:.1f} GB free")

        is_vr = layout in (VRLayout.SBS, VRLayout.OU)
        if is_vr:
            processor_left = RVMProcessor(model_name, device, downsample_ratio)
            processor_right = RVMProcessor(model_name, device, downsample_ratio)
        else:
            processor = RVMProcessor(model_name, device, downsample_ratio)

        effective_batch = batch_size
        num_segments = (total_frames + segment_size - 1) // segment_size
        frames_processed = 0
        job_dir = TEMP_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_size
            seg_frames = min(segment_size, total_frames - seg_start)

            seg_main_path = str(job_dir / f"main_{seg_idx:04d}.mp4")
            seg_alpha_path = str(job_dir / f"alpha_{seg_idx:04d}.mp4")

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

            # Start dual encode pipes (main + alpha)
            main_cmd = build_encode_pipe_hevc_cmd(
                fps, out_w, out_h, crf, seg_main_path,
                keyint=keyint, codec=codec,
            )
            alpha_cmd = build_encode_pipe_hevc_cmd(
                fps, alpha_w, alpha_h, alpha_crf, seg_alpha_path,
                keyint=keyint, codec=codec,
            )
            main_enc = start_encode_process(main_cmd)
            alpha_enc = start_encode_process(alpha_cmd)

            try:
                remaining = seg_frames
                while remaining > 0:
                    read_count = min(effective_batch, remaining)
                    batch_bgr = read_frames(decode_proc, in_w, in_h, read_count, channels=3)
                    if not batch_bgr:
                        break

                    if is_vr:
                        main_frames, alpha_frames = _process_vr_batch_split(
                            batch_bgr, layout, processor_left, processor_right,
                            device, effective_batch, alpha_h, alpha_w,
                        )
                    else:
                        main_frames, alpha_frames, effective_batch = (
                            _process_batch_split_with_oom_retry(
                                processor, batch_bgr, device, effective_batch,
                                alpha_h, alpha_w,
                            )
                        )

                    for main_f, alpha_f in zip(main_frames, alpha_frames):
                        write_frame(main_enc, main_f)
                        write_frame(alpha_enc, alpha_f)

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

                    del batch_bgr, main_frames, alpha_frames

            finally:
                close_process(decode_proc, "decode", tolerant=True)
                close_process(main_enc, "main_encode")
                close_process(alpha_enc, "alpha_encode")

            main_segments.append(seg_main_path)
            alpha_segments.append(seg_alpha_path)
            logger.info("XALPHA segment %d encoded", seg_idx)

        # --- Concatenate segments ---
        if progress_callback:
            progress_callback({"stage": "concatenating", "frame": total_frames, "total_frames": total_frames})

        main_video_only = str(job_dir / "main_video_only.mp4")
        alpha_video_only = str(job_dir / "alpha_video_only.mp4")

        if len(main_segments) == 1:
            os.rename(main_segments[0], main_video_only)
            os.rename(alpha_segments[0], alpha_video_only)
        else:
            concatenate_segments(main_segments, main_video_only, str(job_dir))
            concatenate_segments(alpha_segments, alpha_video_only, str(job_dir))

        # --- Mux audio into main video (AAC for MP4) ---
        if progress_callback:
            progress_callback({"stage": "muxing_audio", "frame": total_frames, "total_frames": total_frames})

        mux_audio_mp4(main_video_only, input_path, output_path)

        # Alpha gets no audio — just move to final path
        os.rename(alpha_video_only, alpha_output_path)

        # --- Apply VR metadata to both files ---
        if layout in (VRLayout.SBS, VRLayout.OU):
            if progress_callback:
                progress_callback({"stage": "metadata", "frame": total_frames, "total_frames": total_frames})
            _apply_vr_metadata(output_path, input_path, layout, job_dir)
            _apply_vr_metadata(alpha_output_path, input_path, layout, job_dir)

        elapsed = time.time() - start_time

        result = {
            "status": "success",
            "input_path": input_path,
            "output_path": output_path,
            "alpha_output_path": alpha_output_path,
            "input_resolution": f"{in_w}x{in_h}",
            "output_resolution": f"{out_w}x{out_h}",
            "alpha_resolution": f"{alpha_w}x{alpha_h}",
            "total_frames": total_frames,
            "layout": layout.value,
            "model": model_name,
            "batch_size": effective_batch,
            "crf": crf,
            "alpha_crf": alpha_crf,
            "keyint": keyint,
            "processing_time_sec": round(elapsed, 1),
            "avg_fps": round(frames_processed / elapsed, 2) if elapsed > 0 else 0,
            "output_mode": "xalpha",
        }

        logger.info("XALPHA BG removal complete: %s", result)
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("XALPHA BG removal failed after %.1fs: %s", elapsed, e, exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "input_path": input_path,
            "processing_time_sec": round(elapsed, 1),
        }

    finally:
        cleanup_job(job_id)
