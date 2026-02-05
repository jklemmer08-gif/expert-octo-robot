"""Real-ESRGAN upscaler with chunked segment processing and OOM retry.

Pipeline per segment:
1. Extract N frames from input to temp PNGs (via FFmpeg)
2. Upscale each frame with Real-ESRGAN (GPU, tile-based)
3. Encode upscaled frames to intermediate MKV segment
4. Delete temp frames immediately

VR frames are split per-eye before upscaling and reassembled after.
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
    DEFAULT_MODEL,
    DEFAULT_SCALE,
    DEFAULT_TILE_SIZE,
    MODEL_DIR,
    SEGMENT_SIZE,
    TEMP_DIR,
    TILE_RETRY_SIZES,
    AVAILABLE_MODELS,
)
from src.pipeline.detector import VRLayout
from src.pipeline.encoder import encode_segment, concatenate_segments, mux_audio
from src.pipeline.metadata import read_vr_metadata, build_metadata_flags
from src.storage.volume import (
    check_disk_space,
    cleanup_job,
    cleanup_segment,
    create_segment_dir,
    estimate_segment_disk_gb,
)
from src.utils.ffmpeg import (
    build_decode_pipe_cmd,
    build_encode_pipe_cmd,
    build_extract_frames_cmd,
    get_video_metadata,
    run_ffmpeg,
)
from src.utils.streaming import (
    close_process,
    read_frame,
    start_decode_process,
    start_encode_process,
    write_frame,
)

logger = logging.getLogger(__name__)


class UpscaleError(Exception):
    """Raised when upscaling fails after all retries."""
    pass


def _load_model(model_name: str, scale: int, tile_size: int, device: str):
    """Load a Real-ESRGAN model. Returns the RealESRGANer instance."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model_info = AVAILABLE_MODELS.get(model_name)
    if not model_info:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")

    model_path = str(MODEL_DIR / model_info["file"])
    model_scale = model_info["scale"]

    # Configure network architecture based on model
    if "anime" in model_name.lower():
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=model_scale)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=model_scale)

    gpu_id = int(device.split(":")[-1]) if "cuda" in device else None

    upsampler = RealESRGANer(
        scale=model_scale,
        model_path=model_path,
        model=model,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=True,  # FP16 for speed
        gpu_id=gpu_id,
    )

    logger.info("Loaded model %s (scale=%d, tile=%d, device=%s)", model_name, model_scale, tile_size, device)
    return upsampler


def upscale_frame(
    upsampler,
    frame_bgr: np.ndarray,
    outscale: int = 4,
) -> np.ndarray:
    """Upscale a single BGR frame. Returns upscaled BGR numpy array."""
    output, _ = upsampler.enhance(frame_bgr, outscale=outscale)
    return output


def upscale_frame_with_oom_retry(
    frame_bgr: np.ndarray,
    model_name: str,
    scale: int,
    device: str,
    tile_sizes: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    """Upscale a frame, retrying with smaller tiles on CUDA OOM.

    Returns (upscaled_frame, tile_size_used).
    """
    if tile_sizes is None:
        tile_sizes = list(TILE_RETRY_SIZES)

    for tile_size in tile_sizes:
        try:
            upsampler = _load_model(model_name, scale, tile_size, device)
            output = upscale_frame(upsampler, frame_bgr, outscale=scale)
            return output, tile_size
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM with tile_size=%d, retrying with smaller tiles", tile_size)
            torch.cuda.empty_cache()
            del upsampler
            continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM (RuntimeError) with tile_size=%d, retrying", tile_size)
                torch.cuda.empty_cache()
                continue
            raise

    raise UpscaleError(f"CUDA OOM even with smallest tile size ({tile_sizes[-1]}). Reduce input resolution or use a GPU with more VRAM.")


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


def process_video(
    input_path: str,
    output_path: str,
    model_name: str = DEFAULT_MODEL,
    scale: int = DEFAULT_SCALE,
    tile_size: int = DEFAULT_TILE_SIZE,
    crf: int = 18,
    layout: VRLayout = VRLayout.FLAT_2D,
    device: str = "cuda:0",
    segment_size: int = SEGMENT_SIZE,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Run the full chunked upscaling pipeline on a video.

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

        # Calculate output dimensions
        if layout in (VRLayout.SBS, VRLayout.OU):
            # VR: scale per-eye, then reassemble
            if layout == VRLayout.SBS:
                eye_w, eye_h = in_w // 2, in_h
                out_w = eye_w * scale * 2
                out_h = eye_h * scale
            else:  # OU
                eye_w, eye_h = in_w, in_h // 2
                out_w = eye_w * scale
                out_h = eye_h * scale * 2
        else:
            out_w = in_w * scale
            out_h = in_h * scale

        logger.info(
            "Processing: %s → %dx%d (%d frames, %s layout, job=%s)",
            input_path, out_w, out_h, total_frames, layout.value, job_id,
        )

        # --- Estimate disk space ---
        needed_gb = estimate_segment_disk_gb(in_w, in_h, segment_size, scale) * 1.5
        if not check_disk_space(needed_gb):
            raise UpscaleError(f"Insufficient disk space: need ~{needed_gb:.1f} GB free for segment processing")

        # --- Load model once ---
        upsampler = _load_model(model_name, scale, tile_size, device)
        effective_tile = tile_size

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

                # 2. Upscale frames
                upscaled_dir = seg_dir / "upscaled"
                upscaled_dir.mkdir(exist_ok=True)

                frame_files = sorted(extract_dir.glob("frame_*.png"))

                for i, frame_file in enumerate(frame_files):
                    if progress_callback:
                        progress_callback({
                            "stage": "upscaling",
                            "segment": seg_idx + 1,
                            "total_segments": num_segments,
                            "frame": frames_processed + i + 1,
                            "total_frames": total_frames,
                        })

                    frame_bgr = cv2.imread(str(frame_file))
                    if frame_bgr is None:
                        logger.warning("Failed to read frame: %s", frame_file)
                        continue

                    # VR: split → upscale each eye → merge
                    if layout in (VRLayout.SBS, VRLayout.OU):
                        left_eye, right_eye = _split_vr_frame(frame_bgr, layout)
                        try:
                            left_up = upscale_frame(upsampler, left_eye, outscale=scale)
                            right_up = upscale_frame(upsampler, right_eye, outscale=scale)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                                raise
                            # OOM retry with smaller tiles
                            logger.warning("OOM on VR frame, retrying with smaller tiles")
                            torch.cuda.empty_cache()
                            del upsampler
                            remaining_tiles = [t for t in TILE_RETRY_SIZES if t < effective_tile]
                            if not remaining_tiles:
                                raise UpscaleError("OOM even with smallest tile size")
                            effective_tile = remaining_tiles[0]
                            upsampler = _load_model(model_name, scale, effective_tile, device)
                            left_up = upscale_frame(upsampler, left_eye, outscale=scale)
                            right_up = upscale_frame(upsampler, right_eye, outscale=scale)
                        upscaled = _merge_vr_frame(left_up, right_up, layout)
                    else:
                        try:
                            upscaled = upscale_frame(upsampler, frame_bgr, outscale=scale)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                                raise
                            logger.warning("OOM on frame %d, retrying with smaller tiles", frames_processed + i)
                            torch.cuda.empty_cache()
                            del upsampler
                            remaining_tiles = [t for t in TILE_RETRY_SIZES if t < effective_tile]
                            if not remaining_tiles:
                                raise UpscaleError("OOM even with smallest tile size")
                            effective_tile = remaining_tiles[0]
                            upsampler = _load_model(model_name, scale, effective_tile, device)
                            upscaled = upscale_frame(upsampler, frame_bgr, outscale=scale)

                    # Save upscaled frame
                    out_name = f"frame_{i+1:06d}.png"
                    cv2.imwrite(str(upscaled_dir / out_name), upscaled)

                    # Free original frame from memory
                    del frame_bgr, upscaled

                frames_processed += len(frame_files)

                # 3. Encode segment
                if progress_callback:
                    progress_callback({
                        "stage": "encoding",
                        "segment": seg_idx + 1,
                        "total_segments": num_segments,
                        "frame": frames_processed,
                        "total_frames": total_frames,
                    })

                seg_clip_path = str(seg_dir / f"segment_{seg_idx:04d}.mkv")
                encode_segment(
                    frame_pattern=str(upscaled_dir / "frame_%06d.png"),
                    output_path=seg_clip_path,
                    fps=fps,
                    width=out_w,
                    height=out_h,
                    crf=crf,
                )
                segment_clips.append(seg_clip_path)

            finally:
                # 4. Clean up frames (keep segment clip)
                for subdir in ["input", "upscaled"]:
                    d = seg_dir / subdir
                    if d.exists():
                        shutil.rmtree(d)
                logger.info("Cleaned frames for segment %d", seg_idx)

        # --- Concatenate segments ---
        if progress_callback:
            progress_callback({"stage": "concatenating", "frame": total_frames, "total_frames": total_frames})

        job_temp_dir = str(Path(segment_clips[0]).parent.parent)
        video_only_path = os.path.join(job_temp_dir, "video_only.mkv")

        if len(segment_clips) == 1:
            os.rename(segment_clips[0], video_only_path)
        else:
            concatenate_segments(segment_clips, video_only_path, job_temp_dir)

        # --- Mux audio ---
        if progress_callback:
            progress_callback({"stage": "muxing_audio", "frame": total_frames, "total_frames": total_frames})

        pre_meta_path = os.path.join(job_temp_dir, "pre_metadata.mkv")
        mux_audio(video_only_path, input_path, pre_meta_path)

        # --- Apply VR metadata ---
        vr_meta = read_vr_metadata(input_path)
        meta_flags = build_metadata_flags(vr_meta, layout)

        if meta_flags:
            if progress_callback:
                progress_callback({"stage": "writing_metadata", "frame": total_frames, "total_frames": total_frames})
            # Re-mux with metadata flags
            meta_cmd = [
                "ffmpeg", "-y",
                "-i", pre_meta_path,
                "-c", "copy",
                *meta_flags,
                output_path,
            ]
            run_ffmpeg(meta_cmd)
            os.remove(pre_meta_path)
        else:
            os.rename(pre_meta_path, output_path)

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
            "scale": scale,
            "tile_size": effective_tile,
            "crf": crf,
            "processing_time_sec": round(elapsed, 1),
            "avg_fps": round(total_frames / elapsed, 2) if elapsed > 0 else 0,
        }

        logger.info("Processing complete: %s", result)
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Processing failed after %.1fs: %s", elapsed, e, exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "input_path": input_path,
            "processing_time_sec": round(elapsed, 1),
        }

    finally:
        # Always clean up job temp directory
        cleanup_job(job_id)


def process_video_streaming(
    input_path: str,
    output_path: str,
    model_name: str = DEFAULT_MODEL,
    scale: int = DEFAULT_SCALE,
    tile_size: int = DEFAULT_TILE_SIZE,
    crf: int = 18,
    codec: Optional[str] = None,
    layout: VRLayout = VRLayout.FLAT_2D,
    device: str = "cuda:0",
    segment_size: int = SEGMENT_SIZE,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Run the upscaling pipeline using FFmpeg pipe streaming (no PNG disk I/O).

    Same interface as process_video() but reads/writes raw frames via pipes
    instead of extracting/encoding PNGs on disk. ~20-40x less temp disk usage.
    """
    from src.pipeline.encoder import get_encoder_codec, concatenate_segments, mux_audio

    job_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    segment_clips: List[str] = []

    if codec is None:
        codec = get_encoder_codec()

    try:
        # --- Get video metadata ---
        meta = get_video_metadata(input_path)
        total_frames = meta["num_frames"]
        fps = meta["fps"]
        in_w, in_h = meta["width"], meta["height"]

        # Calculate output dimensions
        if layout in (VRLayout.SBS, VRLayout.OU):
            if layout == VRLayout.SBS:
                eye_w, eye_h = in_w // 2, in_h
                out_w = eye_w * scale * 2
                out_h = eye_h * scale
            else:
                eye_w, eye_h = in_w, in_h // 2
                out_w = eye_w * scale
                out_h = eye_h * scale * 2
        else:
            out_w = in_w * scale
            out_h = in_h * scale

        logger.info(
            "Streaming upscale: %s → %dx%d (%d frames, %s layout, job=%s)",
            input_path, out_w, out_h, total_frames, layout.value, job_id,
        )

        # --- Estimate disk space (streaming: only encoded segments, not PNGs) ---
        # ~0.5 GB per 1000-frame segment clip (vs ~42 GB for PNGs)
        needed_gb = max(2.0, (total_frames / segment_size) * 0.5)
        if not check_disk_space(needed_gb):
            raise UpscaleError(f"Insufficient disk space: need ~{needed_gb:.1f} GB free")

        # --- Load model once ---
        upsampler = _load_model(model_name, scale, tile_size, device)
        effective_tile = tile_size

        # --- Process segments ---
        num_segments = (total_frames + segment_size - 1) // segment_size
        frames_processed = 0
        job_dir = TEMP_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_size
            seg_frames = min(segment_size, total_frames - seg_start)

            seg_clip_path = str(job_dir / f"segment_{seg_idx:04d}.mkv")

            if progress_callback:
                progress_callback({
                    "stage": "upscaling",
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

            # Start encode pipe
            encode_cmd = build_encode_pipe_cmd(
                fps, out_w, out_h, crf, seg_clip_path, codec=codec,
            )
            encode_proc = start_encode_process(encode_cmd)

            try:
                for i in range(seg_frames):
                    frame_bgr = read_frame(decode_proc, in_w, in_h, channels=3)
                    if frame_bgr is None:
                        logger.warning("Decode pipe ended early at frame %d of segment %d", i, seg_idx)
                        break

                    # VR: split → upscale each eye → merge
                    if layout in (VRLayout.SBS, VRLayout.OU):
                        left_eye, right_eye = _split_vr_frame(frame_bgr, layout)
                        try:
                            left_up = upscale_frame(upsampler, left_eye, outscale=scale)
                            right_up = upscale_frame(upsampler, right_eye, outscale=scale)
                        except (RuntimeError,) as e:
                            if "out of memory" not in str(e).lower():
                                raise
                            logger.warning("OOM on VR frame, retrying with smaller tiles")
                            torch.cuda.empty_cache()
                            del upsampler
                            remaining_tiles = [t for t in TILE_RETRY_SIZES if t < effective_tile]
                            if not remaining_tiles:
                                raise UpscaleError("OOM even with smallest tile size")
                            effective_tile = remaining_tiles[0]
                            upsampler = _load_model(model_name, scale, effective_tile, device)
                            left_up = upscale_frame(upsampler, left_eye, outscale=scale)
                            right_up = upscale_frame(upsampler, right_eye, outscale=scale)
                        upscaled = _merge_vr_frame(left_up, right_up, layout)
                    else:
                        try:
                            upscaled = upscale_frame(upsampler, frame_bgr, outscale=scale)
                        except (RuntimeError,) as e:
                            if "out of memory" not in str(e).lower():
                                raise
                            logger.warning("OOM on frame %d, retrying with smaller tiles", frames_processed + i)
                            torch.cuda.empty_cache()
                            del upsampler
                            remaining_tiles = [t for t in TILE_RETRY_SIZES if t < effective_tile]
                            if not remaining_tiles:
                                raise UpscaleError("OOM even with smallest tile size")
                            effective_tile = remaining_tiles[0]
                            upsampler = _load_model(model_name, scale, effective_tile, device)
                            upscaled = upscale_frame(upsampler, frame_bgr, outscale=scale)

                    write_frame(encode_proc, upscaled)

                    frames_processed += 1
                    if progress_callback:
                        progress_callback({
                            "stage": "upscaling",
                            "segment": seg_idx + 1,
                            "total_segments": num_segments,
                            "frame": frames_processed,
                            "total_frames": total_frames,
                        })

                    del frame_bgr, upscaled

            finally:
                close_process(decode_proc, "decode")
                close_process(encode_proc, "encode")

            segment_clips.append(seg_clip_path)
            logger.info("Segment %d encoded via pipe", seg_idx)

        # --- Concatenate segments ---
        if progress_callback:
            progress_callback({"stage": "concatenating", "frame": total_frames, "total_frames": total_frames})

        video_only_path = str(job_dir / "video_only.mkv")

        if len(segment_clips) == 1:
            os.rename(segment_clips[0], video_only_path)
        else:
            concatenate_segments(segment_clips, video_only_path, str(job_dir))

        # --- Mux audio ---
        if progress_callback:
            progress_callback({"stage": "muxing_audio", "frame": total_frames, "total_frames": total_frames})

        pre_meta_path = str(job_dir / "pre_metadata.mkv")
        mux_audio(video_only_path, input_path, pre_meta_path)

        # --- Apply VR metadata ---
        vr_meta = read_vr_metadata(input_path)
        meta_flags = build_metadata_flags(vr_meta, layout)

        if meta_flags:
            if progress_callback:
                progress_callback({"stage": "writing_metadata", "frame": total_frames, "total_frames": total_frames})
            meta_cmd = [
                "ffmpeg", "-y",
                "-i", pre_meta_path,
                "-c", "copy",
                *meta_flags,
                output_path,
            ]
            run_ffmpeg(meta_cmd)
            os.remove(pre_meta_path)
        else:
            os.rename(pre_meta_path, output_path)

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
            "scale": scale,
            "tile_size": effective_tile,
            "crf": crf,
            "processing_time_sec": round(elapsed, 1),
            "avg_fps": round(frames_processed / elapsed, 2) if elapsed > 0 else 0,
            "streaming": True,
        }

        logger.info("Streaming processing complete: %s", result)
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Streaming processing failed after %.1fs: %s", elapsed, e, exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "input_path": input_path,
            "processing_time_sec": round(elapsed, 1),
        }

    finally:
        cleanup_job(job_id)
