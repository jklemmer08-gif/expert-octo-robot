"""VR stereo layout detection — SBS / OU / Mono from filename, metadata, and aspect ratio."""

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Optional

from src.utils.ffmpeg import get_stream_side_data, get_all_stream_info

logger = logging.getLogger(__name__)


class VRLayout(str, Enum):
    MONO = "mono"
    SBS = "sbs"      # Side-by-Side (left-right)
    OU = "ou"        # Over-Under (top-bottom)
    FLAT_2D = "2d"   # Standard 2D video (not VR)


# --- Filename patterns ---
SBS_PATTERNS = re.compile(
    r"[_\-. ](SBS|LR|3DH|sbs|lr|3dh|HSBS|hsbs|side.?by.?side)[_\-. \.]",
    re.IGNORECASE,
)
OU_PATTERNS = re.compile(
    r"[_\-. ](OU|TB|3DV|ou|tb|3dv|HOU|hou|over.?under|top.?bottom)[_\-. \.]",
    re.IGNORECASE,
)
MONO_PATTERNS = re.compile(
    r"[_\-. ](MONO|mono)[_\-. \.]",
    re.IGNORECASE,
)


def detect_from_filename(filename: str) -> Optional[VRLayout]:
    """Check filename for standard VR naming conventions."""
    # Pad with spaces so patterns anchored to delimiters always match
    padded = f" {filename} "

    if SBS_PATTERNS.search(padded):
        return VRLayout.SBS
    if OU_PATTERNS.search(padded):
        return VRLayout.OU
    if MONO_PATTERNS.search(padded):
        return VRLayout.MONO
    return None


def detect_from_metadata(path: str) -> Optional[VRLayout]:
    """Check container metadata / stream side data for stereo3d tags."""
    try:
        side_data = get_stream_side_data(path)
        for sd in side_data:
            sd_type = sd.get("side_data_type", "")
            if "stereo3d" in sd_type.lower() or "Stereo 3D" in sd_type:
                stereo_type = sd.get("type", "").lower()
                if stereo_type in ("side_by_side", "side by side left first"):
                    return VRLayout.SBS
                if stereo_type in ("top_bottom", "top and bottom"):
                    return VRLayout.OU
    except Exception as e:
        logger.debug("Side data check failed: %s", e)

    # Check format/stream tags
    try:
        info = get_all_stream_info(path)
        # Check format tags
        tags = info.get("format", {}).get("tags", {})
        # Check stream tags
        for stream in info.get("streams", []):
            tags.update(stream.get("tags", {}))

        stereo_mode = tags.get("stereo_mode", "").lower()
        if stereo_mode in ("side_by_side", "left_right"):
            return VRLayout.SBS
        if stereo_mode in ("top_bottom", "bottom_top"):
            return VRLayout.OU
        if stereo_mode == "mono":
            return VRLayout.MONO
    except Exception as e:
        logger.debug("Stream tag check failed: %s", e)

    return None


def detect_from_aspect_ratio(width: int, height: int) -> Optional[VRLayout]:
    """Use aspect ratio heuristics to guess VR layout.

    Heuristics:
    - ~2:1 (e.g. 3840x1920) → SBS 180°
    - ~1:2 (e.g. 1920x3840) → OU 180° (rare)
    - ~4:1 (e.g. 7680x1920) → SBS 360°
    - ~1:1 with high res (e.g. 4096x4096) → likely mono 360°
    """
    if width == 0 or height == 0:
        return None

    ratio = width / height

    if 1.9 <= ratio <= 2.1:
        return VRLayout.SBS
    if 3.8 <= ratio <= 4.2:
        return VRLayout.SBS
    if 0.45 <= ratio <= 0.55:
        return VRLayout.OU

    return None


def detect_layout(path: str, width: int, height: int) -> VRLayout:
    """Detect VR layout using all available signals (priority order).

    Priority:
    1. Filename conventions
    2. Container/stream metadata
    3. Aspect ratio heuristics
    4. Default to FLAT_2D
    """
    filename = Path(path).name

    # 1. Filename
    result = detect_from_filename(filename)
    if result:
        logger.info("Layout detected from filename: %s (%s)", result.value, filename)
        return result

    # 2. Metadata
    result = detect_from_metadata(path)
    if result:
        logger.info("Layout detected from metadata: %s", result.value)
        return result

    # 3. Aspect ratio
    result = detect_from_aspect_ratio(width, height)
    if result:
        logger.info("Layout detected from aspect ratio (%dx%d): %s", width, height, result.value)
        return result

    logger.info("No VR layout detected — treating as 2D")
    return VRLayout.FLAT_2D
