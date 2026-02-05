"""Pipe I/O helpers for FFmpeg streaming pipeline.

Reads/writes raw video frames between Python and FFmpeg subprocess pipes,
eliminating disk I/O for individual frames.
"""

import logging
import subprocess
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def read_frame(pipe: subprocess.Popen, width: int, height: int, channels: int = 3) -> Optional[np.ndarray]:
    """Read one raw frame from FFmpeg decode pipe stdout.

    Returns BGR (3-ch) or BGRA (4-ch) numpy array, or None at end of stream.
    """
    nbytes = width * height * channels
    data = pipe.stdout.read(nbytes)
    if len(data) != nbytes:
        return None
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels)


def read_frames(pipe: subprocess.Popen, width: int, height: int, count: int, channels: int = 3) -> List[np.ndarray]:
    """Read up to `count` raw frames from FFmpeg decode pipe.

    Returns list of numpy arrays. May return fewer than `count` at end of stream.
    """
    frames = []
    for _ in range(count):
        frame = read_frame(pipe, width, height, channels)
        if frame is None:
            break
        frames.append(frame)
    return frames


def write_frame(pipe: subprocess.Popen, frame: np.ndarray) -> None:
    """Write one raw frame to FFmpeg encode pipe stdin."""
    pipe.stdin.write(frame.tobytes())


def start_decode_process(cmd: List[str]) -> subprocess.Popen:
    """Start an FFmpeg decode subprocess with stdout pipe for raw frames."""
    logger.debug("Decode cmd: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def start_encode_process(cmd: List[str]) -> subprocess.Popen:
    """Start an FFmpeg encode subprocess with stdin pipe for raw frames."""
    logger.debug("Encode cmd: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def close_process(proc: subprocess.Popen, name: str = "ffmpeg", timeout: int = 300) -> None:
    """Close a subprocess pipe and wait for completion. Raises on failure."""
    if proc.stdin and not proc.stdin.closed:
        proc.stdin.close()
    if proc.stdout and not proc.stdout.closed:
        proc.stdout.close()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"{name} process timed out after {timeout}s")

    if proc.returncode != 0:
        stderr = ""
        if proc.stderr:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"{name} failed (exit {proc.returncode}): {stderr}")
