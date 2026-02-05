"""FFmpeg video encoding — NVENC with libx265 fallback, audio muxing, concat."""

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.config import DEFAULT_CRF, DEFAULT_CODEC, DEFAULT_PRESET, DEFAULT_VP9_CRF, FALLBACK_CODEC
from src.utils.ffmpeg import (
    build_concat_cmd,
    build_encode_segment_cmd,
    build_encode_segment_vp9_cmd,
    build_mux_audio_cmd,
    build_mux_audio_webm_cmd,
    check_nvenc_available,
    has_audio,
    run_ffmpeg,
)

logger = logging.getLogger(__name__)

# Cache NVENC availability check
_nvenc_available: Optional[bool] = None


def get_encoder_codec() -> str:
    """Return the best available HEVC encoder (NVENC or libx265 fallback)."""
    global _nvenc_available
    if _nvenc_available is None:
        _nvenc_available = check_nvenc_available()
        if _nvenc_available:
            logger.info("NVENC available — using hevc_nvenc")
        else:
            logger.warning("NVENC not available — falling back to libx265 (CPU)")
    return DEFAULT_CODEC if _nvenc_available else FALLBACK_CODEC


def encode_segment(
    frame_pattern: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    crf: int = DEFAULT_CRF,
    codec: Optional[str] = None,
    preset: str = DEFAULT_PRESET,
) -> str:
    """Encode a segment's PNG frames into an MKV clip. Returns output path."""
    if codec is None:
        codec = get_encoder_codec()

    cmd = build_encode_segment_cmd(
        input_pattern=frame_pattern,
        output_path=output_path,
        fps=fps,
        width=width,
        height=height,
        crf=crf,
        codec=codec,
        preset=preset,
    )

    try:
        run_ffmpeg(cmd)
    except RuntimeError:
        # If NVENC fails, retry with libx265
        if codec != FALLBACK_CODEC:
            logger.warning("NVENC encoding failed, retrying with libx265")
            cmd = build_encode_segment_cmd(
                input_pattern=frame_pattern,
                output_path=output_path,
                fps=fps,
                width=width,
                height=height,
                crf=crf,
                codec=FALLBACK_CODEC,
                preset=preset,
            )
            run_ffmpeg(cmd)
        else:
            raise

    logger.info("Encoded segment: %s", output_path)
    return output_path


def concatenate_segments(
    segment_paths: List[str],
    output_path: str,
    temp_dir: str,
) -> str:
    """Concatenate segment MKV clips via FFmpeg concat demuxer."""
    # Write concat list file
    concat_list = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list, "w") as f:
        for seg_path in segment_paths:
            # FFmpeg concat needs absolute paths with escaped single quotes
            f.write(f"file '{os.path.abspath(seg_path)}'\n")

    cmd = build_concat_cmd(concat_list, output_path)
    run_ffmpeg(cmd)
    logger.info("Concatenated %d segments → %s", len(segment_paths), output_path)
    return output_path


def mux_audio(
    video_path: str,
    audio_source_path: str,
    output_path: str,
    extra_flags: Optional[List[str]] = None,
) -> str:
    """Mux audio streams from the original into the upscaled video."""
    if not has_audio(audio_source_path):
        logger.info("No audio streams found in source — skipping audio mux")
        # Just rename/copy the video file
        if video_path != output_path:
            os.rename(video_path, output_path)
        return output_path

    cmd = build_mux_audio_cmd(video_path, audio_source_path, output_path, extra_flags)
    run_ffmpeg(cmd)
    logger.info("Muxed audio from %s → %s", audio_source_path, output_path)
    return output_path


def encode_segment_vp9(
    frame_pattern: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    crf: int = DEFAULT_VP9_CRF,
    speed: int = 4,
) -> str:
    """Encode RGBA PNG frames into a VP9 WebM clip with alpha. Returns output path."""
    cmd = build_encode_segment_vp9_cmd(
        input_pattern=frame_pattern,
        output_path=output_path,
        fps=fps,
        width=width,
        height=height,
        crf=crf,
        speed=speed,
    )
    run_ffmpeg(cmd)
    logger.info("Encoded VP9 segment: %s", output_path)
    return output_path


def mux_audio_webm(
    video_path: str,
    audio_source_path: str,
    output_path: str,
    extra_flags: Optional[List[str]] = None,
) -> str:
    """Mux audio from the original into WebM, transcoding to Opus."""
    if not has_audio(audio_source_path):
        logger.info("No audio streams found in source — skipping audio mux")
        if video_path != output_path:
            os.rename(video_path, output_path)
        return output_path

    cmd = build_mux_audio_webm_cmd(video_path, audio_source_path, output_path, extra_flags)
    run_ffmpeg(cmd)
    logger.info("Muxed audio (Opus) from %s → %s", audio_source_path, output_path)
    return output_path
