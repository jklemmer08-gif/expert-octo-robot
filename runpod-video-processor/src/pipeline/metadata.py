"""VR metadata read / write / preserve through transcoding."""

import logging
from typing import Any, Dict, List, Optional

from src.pipeline.detector import VRLayout
from src.utils.ffmpeg import get_all_stream_info, get_stream_side_data

logger = logging.getLogger(__name__)


def read_vr_metadata(path: str) -> Dict[str, Any]:
    """Extract all VR-relevant metadata from an input file.

    Returns a dict with:
    - stereo_mode: str or None
    - projection: str or None (equirectangular, cubemap, etc.)
    - spherical_tags: dict of any spherical XML or sv3d tags
    - raw_side_data: list of side_data entries
    - stream_tags: dict of stream-level tags
    """
    meta: Dict[str, Any] = {
        "stereo_mode": None,
        "projection": None,
        "spherical_tags": {},
        "raw_side_data": [],
        "stream_tags": {},
    }

    # Side data (st3d, spherical info)
    try:
        side_data = get_stream_side_data(path)
        meta["raw_side_data"] = side_data
        for sd in side_data:
            sd_type = sd.get("side_data_type", "")
            if "stereo3d" in sd_type.lower() or "Stereo 3D" in sd_type:
                meta["stereo_mode"] = sd.get("type")
            if "spherical" in sd_type.lower():
                meta["projection"] = sd.get("projection", "equirectangular")
                meta["spherical_tags"] = {
                    k: v for k, v in sd.items() if k != "side_data_type"
                }
    except Exception as e:
        logger.debug("Failed to read side data: %s", e)

    # Stream/format tags
    try:
        info = get_all_stream_info(path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                tags = stream.get("tags", {})
                meta["stream_tags"] = tags
                if not meta["stereo_mode"] and "stereo_mode" in tags:
                    meta["stereo_mode"] = tags["stereo_mode"]
                break
    except Exception as e:
        logger.debug("Failed to read stream tags: %s", e)

    return meta


def build_metadata_flags(
    vr_meta: Dict[str, Any],
    layout: VRLayout,
) -> List[str]:
    """Build ffmpeg CLI flags to write VR metadata to the output file.

    Returns a list of strings like ["-metadata:s:v", "stereo_mode=side_by_side", ...].
    """
    flags: List[str] = []

    # Stereo mode
    stereo_mode = vr_meta.get("stereo_mode")
    if not stereo_mode:
        if layout == VRLayout.SBS:
            stereo_mode = "side_by_side"
        elif layout == VRLayout.OU:
            stereo_mode = "top_bottom"

    if stereo_mode:
        flags.extend(["-metadata:s:v", f"stereo_mode={stereo_mode}"])

    # Copy through any stream tags that look VR-related
    vr_tag_keys = {
        "spherical", "stitched", "stitching_software",
        "projection_type", "stereo_mode",
        "initial_heading", "initial_pitch", "initial_roll",
    }
    for key, value in vr_meta.get("stream_tags", {}).items():
        if key.lower() in vr_tag_keys:
            flags.extend(["-metadata:s:v", f"{key}={value}"])

    return flags


def get_metadata_summary(path: str, layout: VRLayout) -> Dict[str, Any]:
    """Return a human-readable summary of VR metadata for the web UI."""
    meta = read_vr_metadata(path)
    return {
        "layout": layout.value,
        "stereo_mode": meta.get("stereo_mode"),
        "projection": meta.get("projection"),
        "has_spherical_tags": bool(meta.get("spherical_tags")),
        "tag_count": len(meta.get("stream_tags", {})),
    }
