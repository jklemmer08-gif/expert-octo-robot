"""FFmpeg / ffprobe helpers for metadata extraction and encoding commands."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def ffprobe_json(path: str, *extra_args: str) -> Dict[str, Any]:
    """Run ffprobe and return parsed JSON output."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        *extra_args,
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def get_video_metadata(path: str) -> Dict[str, Any]:
    """Return a flat dict with key video stream properties."""
    data = ffprobe_json(
        path,
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration,codec_name,nb_frames",
        "-show_entries",
        "format=duration,size",
    )

    stream = data.get("streams", [{}])[0] if data.get("streams") else {}
    fmt = data.get("format", {})

    # Parse frame rate fraction
    fps_str = stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str) if fps_str else 30.0

    # Duration: prefer stream, fallback to format
    duration = float(stream.get("duration") or fmt.get("duration") or 0)

    # Frame count
    nb_frames = stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        num_frames = int(nb_frames)
    else:
        num_frames = int(fps * duration) if duration else 0

    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fps,
        "duration": duration,
        "codec": stream.get("codec_name", ""),
        "num_frames": num_frames,
        "file_size": int(fmt.get("size", 0)),
    }


def get_all_stream_info(path: str) -> Dict[str, Any]:
    """Return full stream and format info (for metadata inspection)."""
    return ffprobe_json(path, "-show_streams", "-show_format")


def get_stream_side_data(path: str) -> List[Dict[str, Any]]:
    """Return side_data_list from the first video stream (VR metadata)."""
    data = ffprobe_json(
        path,
        "-select_streams", "v:0",
        "-show_entries", "stream_side_data",
    )
    streams = data.get("streams", [])
    if streams and streams[0].get("side_data_list"):
        return streams[0]["side_data_list"]
    return []


def has_audio(path: str) -> bool:
    """Check whether the file has at least one audio stream."""
    data = ffprobe_json(path, "-select_streams", "a", "-show_entries", "stream=codec_type")
    return bool(data.get("streams"))


def check_nvenc_available() -> bool:
    """Check if hevc_nvenc encoder is available in the system FFmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "hevc_nvenc" in result.stdout
    except Exception:
        return False


def build_extract_frames_cmd(
    input_path: str,
    output_pattern: str,
    start_frame: int = 0,
    num_frames: int = 1000,
    fps: Optional[float] = None,
) -> List[str]:
    """Build ffmpeg command to extract a segment of frames as PNGs."""
    cmd = ["ffmpeg", "-y"]

    # Seek to start frame (approximate via time for speed)
    if start_frame > 0 and fps:
        start_time = start_frame / fps
        cmd.extend(["-ss", f"{start_time:.4f}"])

    cmd.extend(["-i", str(input_path)])

    if num_frames > 0:
        cmd.extend(["-frames:v", str(num_frames)])

    cmd.extend(["-qscale:v", "2", output_pattern])
    return cmd


def build_encode_segment_cmd(
    input_pattern: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    crf: int = 18,
    codec: str = "hevc_nvenc",
    preset: str = "slow",
) -> List[str]:
    """Build ffmpeg command to encode a segment of frames into an MKV clip."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", codec,
    ]

    if codec == "hevc_nvenc":
        # NVENC uses different preset names and -cq instead of -crf
        cmd.extend(["-preset", "p7", "-rc", "vbr", "-cq", str(crf)])
    else:
        cmd.extend(["-preset", preset, "-crf", str(crf)])

    cmd.extend(["-pix_fmt", "yuv420p", output_path])
    return cmd


def build_concat_cmd(
    concat_list_path: str,
    output_path: str,
) -> List[str]:
    """Build ffmpeg command to concatenate segment clips via concat demuxer."""
    return [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        output_path,
    ]


def build_encode_segment_vp9_cmd(
    input_pattern: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    crf: int = 30,
    speed: int = 4,
) -> List[str]:
    """Build ffmpeg command to encode RGBA PNGs into a VP9 WebM with alpha."""
    return [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-crf", str(crf),
        "-b:v", "0",
        "-auto-alt-ref", "0",
        "-row-mt", "1",
        "-speed", str(speed),
        output_path,
    ]


def build_mux_audio_webm_cmd(
    video_path: str,
    audio_source_path: str,
    output_path: str,
    extra_flags: Optional[List[str]] = None,
) -> List[str]:
    """Mux audio from the original file into a WebM, transcoding to Opus."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_source_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "libopus",
        "-b:a", "128k",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.append(output_path)
    return cmd


def build_mux_audio_cmd(
    video_path: str,
    audio_source_path: str,
    output_path: str,
    extra_flags: Optional[List[str]] = None,
) -> List[str]:
    """Mux audio from the original file into the upscaled video."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_source_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c", "copy",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.append(output_path)
    return cmd


def build_decode_pipe_cmd(
    input_path: str,
    start_frame: int,
    num_frames: int,
    fps: float,
    width: int,
    height: int,
) -> List[str]:
    """Build ffmpeg command to decode frames as raw BGR24 bytes to stdout.

    Output: raw BGR24 bytes to pipe:1 (3 bytes/pixel, width * height * 3 per frame).
    """
    cmd = ["ffmpeg", "-nostdin", "-y"]

    if start_frame > 0 and fps:
        start_time = start_frame / fps
        cmd.extend(["-ss", f"{start_time:.4f}"])

    cmd.extend(["-i", str(input_path)])

    if num_frames > 0:
        cmd.extend(["-frames:v", str(num_frames)])

    cmd.extend([
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "pipe:1",
    ])
    return cmd


def build_encode_pipe_cmd(
    fps: float,
    width: int,
    height: int,
    crf: int,
    output_path: str,
    codec: str = "hevc_nvenc",
    preset: str = "slow",
) -> List[str]:
    """Build ffmpeg command to encode raw BGR24 bytes from stdin to video file.

    Input: raw BGR24 from pipe:0 (3 bytes/pixel).
    """
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", codec,
    ]

    if codec == "hevc_nvenc":
        cmd.extend(["-preset", "p7", "-rc", "vbr", "-cq", str(crf)])
    else:
        cmd.extend(["-preset", preset, "-crf", str(crf)])

    cmd.extend(["-pix_fmt", "yuv420p", "-v", "error", output_path])
    return cmd


def build_encode_pipe_vp9_cmd(
    fps: float,
    width: int,
    height: int,
    crf: int,
    output_path: str,
    speed: int = 4,
) -> List[str]:
    """Build ffmpeg command to encode raw BGRA bytes from stdin to VP9 WebM with alpha.

    Input: raw BGRA from pipe:0 (4 bytes/pixel).
    """
    return [
        "ffmpeg", "-nostdin", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgra",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-crf", str(crf),
        "-b:v", "0",
        "-auto-alt-ref", "0",
        "-row-mt", "1",
        "-speed", str(speed),
        "-v", "error",
        output_path,
    ]


def run_ffmpeg(cmd: List[str], timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run an ffmpeg command, capturing output. Raises on failure."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error("FFmpeg stderr: %s", result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError(f"FFmpeg failed (exit {result.returncode}): {result.stderr[-500:]}")
    return result
