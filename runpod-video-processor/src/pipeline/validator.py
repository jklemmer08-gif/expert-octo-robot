"""Input video validation — file existence, codec, size, duration, corruption."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.config import MAX_DURATION_MINUTES, MAX_FILE_SIZE_GB, SUPPORTED_CODECS
from src.utils.ffmpeg import get_video_metadata

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[dict] = None

    def add_error(self, msg: str):
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def validate_input(path: str) -> ValidationResult:
    """Run all validation checks on an input video file."""
    result = ValidationResult()
    p = Path(path)

    # --- File existence ---
    if not p.exists():
        result.add_error(f"File not found: {path}")
        return result

    if not p.is_file():
        result.add_error(f"Not a file: {path}")
        return result

    if not os.access(str(p), os.R_OK):
        result.add_error(f"File not readable: {path}")
        return result

    # --- File size ---
    size_bytes = p.stat().st_size
    size_gb = size_bytes / (1024 ** 3)

    if size_bytes == 0:
        result.add_error("File is empty (0 bytes)")
        return result

    if size_gb > MAX_FILE_SIZE_GB:
        result.add_error(
            f"File too large: {size_gb:.1f} GB (limit: {MAX_FILE_SIZE_GB} GB)"
        )
        return result

    # --- Probe metadata ---
    try:
        meta = get_video_metadata(path)
    except Exception as e:
        result.add_error(f"Cannot read video metadata (file may be corrupt): {e}")
        return result

    result.metadata = meta

    # --- Codec ---
    codec = meta.get("codec", "").lower()
    if not codec:
        result.add_error("No video codec detected")
        return result

    if codec not in SUPPORTED_CODECS:
        result.add_error(
            f"Unsupported codec: {codec}. "
            f"Supported: {', '.join(sorted(SUPPORTED_CODECS))}. "
            f"Try remuxing: ffmpeg -i input -c copy output.mkv"
        )

    # --- Resolution ---
    width = meta.get("width", 0)
    height = meta.get("height", 0)
    if width == 0 or height == 0:
        result.add_error(f"Invalid resolution: {width}x{height}")

    # --- Duration ---
    duration_sec = meta.get("duration", 0)
    duration_min = duration_sec / 60
    if duration_sec <= 0:
        result.add_warning("Could not determine video duration")
    elif duration_min > MAX_DURATION_MINUTES:
        result.add_error(
            f"Video too long: {duration_min:.0f} min (limit: {MAX_DURATION_MINUTES} min)"
        )

    # --- Frame count sanity ---
    num_frames = meta.get("num_frames", 0)
    if num_frames <= 0:
        result.add_warning("Could not determine frame count — will estimate from duration")

    return result


# Avoid circular import: os is needed for access check
import os
