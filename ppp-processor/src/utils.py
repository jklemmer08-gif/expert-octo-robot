"""Shared utilities for PPP Processor.

Includes path remapping from batch_process.py and common helpers.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from typing import List, Tuple

# Supported video file extensions
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".webm", ".m4v", ".ts", ".mts", ".m2ts",
}

# Path remapping from Stash Docker mounts to local host paths
# Consolidated from batch_process.py and detect_passthrough.py
PATH_REMAPS: List[Tuple[str, str]] = [
    ("/data/library/", "/home/jtk1234/media-drive1/"),
    ("/data/media/", "/home/jtk1234/media-drive1/"),
    ("/data/recovered/", "/home/jtk1234/media-drive2/"),
    ("/media/library/", "/home/jtk1234/media-drive1/"),
    ("/media/recovered/", "/home/jtk1234/media-drive2/"),
    ("/media/drive3/", "/home/jtk1234/media-drive3/"),
    ("/media/ppp-output/", "/mnt/ppp-work/ppp/output/"),
]


# Windows (Samba) path remapping — Linux host paths to mapped drive letters
WINDOWS_PATH_REMAPS: List[Tuple[str, str]] = [
    ("/home/jtk1234/media-drive1/", "M:/"),
    ("/home/jtk1234/media-drive1", "M:"),
    ("/home/jtk1234/media-drive2/", "N:/"),
    ("/home/jtk1234/media-drive2", "N:"),
]


def remap_path(path: str) -> str:
    """Remap a Stash Docker container path to the host filesystem path.

    Consolidated from batch_process.py:212-217 and detect_passthrough.py:39-44.
    """
    for docker_prefix, local_prefix in PATH_REMAPS:
        if path.startswith(docker_prefix):
            return local_prefix + path[len(docker_prefix):]
    return path


def platform_path(path: str) -> str:
    """Remap Linux paths to Windows drive letters when running on Windows.

    Chains Docker container remapping first, then applies Windows drive mapping.
    E.g. /media/library/foo → /home/jtk1234/media-drive1/foo → M:\\foo
    """
    if platform.system() != "Windows":
        return path
    # First, remap Docker container paths to host paths
    remapped = remap_path(path)
    # Then map host Linux paths to Windows drive letters
    for linux_prefix, win_prefix in WINDOWS_PATH_REMAPS:
        if remapped.startswith(linux_prefix):
            result = win_prefix + remapped[len(linux_prefix):]
            return result.replace("/", "\\")
    return path


def resolution_label(width: int) -> str:
    """Return a human-readable resolution label from pixel width."""
    if width >= 7680:
        return "8K"
    elif width >= 5760:
        return "6K"
    elif width >= 5120:
        return "5K"
    elif width >= 3840:
        return "4K"
    elif width >= 2560:
        return "1440p"
    elif width >= 1920:
        return "1080p"
    elif width >= 1280:
        return "720p"
    elif width >= 854:
        return "480p"
    return "SD"


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 hash of the first 1MB of a file (fast fingerprint)."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        data = f.read(chunk_size)
        sha.update(data)
    return sha.hexdigest()


def is_video_file(path: Path) -> bool:
    """Check if a path has a recognized video extension."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def bitrate_for_resolution(width: int, is_vr: bool = False) -> str:
    """Select an appropriate encoding bitrate string based on output width."""
    if width >= 7680:
        return "150M"
    elif width >= 5760:
        return "100M"
    elif width >= 5120:
        return "80M"
    elif width >= 3840:
        return "50M"
    elif width >= 1920:
        return "15M"
    return "10M"
