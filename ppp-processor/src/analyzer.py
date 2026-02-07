"""Video analysis module for PPP Processor.

Consolidates video probing and VR detection from:
- upscale.py:84-138 (get_video_info / VR detection)
- vr_metadata.py:62-84 (VR_PATTERNS)
- detect_passthrough.py:56-84 (is_vr_scene)

Adds quality scoring for router decisions.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from src.config import Settings
from src.models.schemas import ContentType, VideoInfo, VRMetadata
from src.utils import file_hash, resolution_label

logger = logging.getLogger("ppp.analyzer")


class VideoAnalyzer:
    """Analyzes video files for metadata, VR detection, and quality scoring."""

    # Filename-based VR detection patterns (from vr_metadata.py:62-84)
    VR_PATTERNS = {
        r"_180": {"fov_horizontal": 180, "fov_vertical": 180},
        r"_360": {"fov_horizontal": 360, "fov_vertical": 180},
        r"180x180": {"fov_horizontal": 180, "fov_vertical": 180},
        r"360x180": {"fov_horizontal": 360, "fov_vertical": 180},
        r"_sbs|_lr|_3dh|SBS": {"stereo_mode": "sbs"},
        r"_tb|_ou|_3dv|TB": {"stereo_mode": "tb"},
        r"_mono": {"stereo_mode": "mono"},
        r"_fisheye|_fe": {"projection": "fisheye"},
        r"_equi|_equ": {"projection": "equirectangular"},
        r"MKX200": {"projection": "fisheye", "fov_horizontal": 200},
        r"VRBangers|WankzVR|VRHush|SLR|VirtualReal|POVR|RealJam|VRConk": {
            "is_vr": True, "fov_horizontal": 180, "stereo_mode": "sbs",
        },
    }

    # Known VR resolutions (from vr_metadata.py:87-93)
    VR_RESOLUTIONS = {
        (7680, 3840), (6144, 3072), (5760, 2880),
        (4096, 2048), (3840, 1920),
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.vr_filename_patterns = settings.vr_patterns

    # ------------------------------------------------------------------
    # Video probing (from upscale.py:84-138)
    # ------------------------------------------------------------------
    def probe_video(self, video_path: Path) -> VideoInfo:
        """Extract video metadata using ffprobe and detect VR content."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")

        width = int(video_stream["width"])
        height = int(video_stream["height"])

        # Parse FPS
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den else 30.0
        else:
            fps = float(fps_str)

        duration = float(data.get("format", {}).get("duration", 0))
        codec = video_stream.get("codec_name", "unknown")
        bitrate = int(data.get("format", {}).get("bit_rate", 0))

        # Detect VR
        is_vr, vr_type = self.detect_vr(video_path, width, height, video_stream)

        # Determine content type
        if is_vr:
            if vr_type == "tb":
                content_type = ContentType.VR_TB
            elif vr_type == "sbs":
                content_type = ContentType.VR_SBS
            else:
                content_type = ContentType.VR_MONO
        else:
            content_type = ContentType.FLAT_2D

        # File metadata
        fsize = video_path.stat().st_size if video_path.exists() else None
        fhash = file_hash(video_path) if video_path.exists() else None

        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            codec=codec,
            bitrate=bitrate,
            is_vr=is_vr,
            vr_type=vr_type,
            content_type=content_type,
            file_path=str(video_path),
            file_size=fsize,
            file_hash=fhash,
        )

    # ------------------------------------------------------------------
    # VR detection (consolidated)
    # ------------------------------------------------------------------
    def detect_vr(
        self,
        video_path: Path,
        width: int,
        height: int,
        video_stream: dict | None = None,
    ) -> tuple[bool, str | None]:
        """Consolidated VR detection from filename patterns, resolution, and stream metadata.

        Returns (is_vr, vr_type) where vr_type is 'sbs', 'tb', or None.
        """
        is_vr = False
        vr_type: str | None = None
        filename = video_path.name.lower()

        # 1. Filename patterns (from settings + hardcoded studio patterns)
        for pattern in self.vr_filename_patterns:
            if pattern.lower() in filename:
                is_vr = True
                break

        # 2. Regex-based studio / format patterns (from vr_metadata.py)
        for pattern in self.VR_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                is_vr = True
                attrs = self.VR_PATTERNS[pattern]
                if "stereo_mode" in attrs:
                    vr_type = attrs["stereo_mode"]
                break

        # 3. Resolution check (known VR resolutions)
        if not is_vr and (width, height) in self.VR_RESOLUTIONS:
            is_vr = True

        # 4. Aspect ratio (stereoscopic SBS: width >= 2*height)
        if not is_vr and height > 0 and width >= 2 * height:
            is_vr = True

        # 5. Stream-level spherical metadata
        if video_stream and not is_vr:
            for side_data in video_stream.get("side_data_list", []):
                if side_data.get("side_data_type") == "Spherical Mapping":
                    is_vr = True
                    break

        # Determine VR type if not set
        if is_vr and vr_type is None:
            if "_tb" in filename or "_ou" in filename or "_3dv" in filename:
                vr_type = "tb"
            else:
                vr_type = "sbs"  # default

        return is_vr, vr_type

    # ------------------------------------------------------------------
    # VR metadata extraction (from vr_metadata.py:119)
    # ------------------------------------------------------------------
    def extract_vr_metadata(self, video_path: Path) -> VRMetadata:
        """Extract full VR metadata structure from a video file."""
        info = self.probe_video(video_path)
        meta = VRMetadata(
            is_vr=info.is_vr,
            stereo_mode=info.vr_type or "sbs",
            source_filename=video_path.name,
        )

        # Refine from filename patterns
        filename = video_path.name.lower()
        for pattern, attrs in self.VR_PATTERNS.items():
            if re.search(pattern, filename, re.IGNORECASE):
                for key, val in attrs.items():
                    if hasattr(meta, key):
                        setattr(meta, key, val)

        return meta

    # ------------------------------------------------------------------
    # Quality scoring (new)
    # ------------------------------------------------------------------
    def calculate_quality_score(self, info: VideoInfo) -> float:
        """Calculate a quality score (0-100) based on resolution, bitrate, codec.

        Used by the Router to make processing decisions.
        """
        score = 0.0

        # Resolution contribution (0-40)
        max_dim = max(info.width, info.height)
        if max_dim >= 7680:
            score += 40
        elif max_dim >= 5760:
            score += 35
        elif max_dim >= 3840:
            score += 30
        elif max_dim >= 1920:
            score += 20
        elif max_dim >= 1280:
            score += 10
        else:
            score += 5

        # Bitrate contribution (0-30)
        mbps = info.bitrate / 1_000_000 if info.bitrate else 0
        if mbps >= 50:
            score += 30
        elif mbps >= 25:
            score += 25
        elif mbps >= 10:
            score += 15
        elif mbps >= 5:
            score += 10
        else:
            score += 5

        # Codec contribution (0-20)
        codec_scores = {
            "hevc": 20, "h265": 20, "av1": 20,
            "h264": 15, "avc1": 15,
            "vp9": 15, "mpeg4": 10,
        }
        score += codec_scores.get(info.codec.lower(), 5)

        # VR bonus (0-10)
        if info.is_vr:
            score += 10

        return min(100.0, score)

    def should_upscale(self, info: VideoInfo) -> bool:
        """Determine if a video would benefit from AI upscaling.

        Returns False for already-high-res content or very short clips.
        """
        # Already 8K+ — no point
        if info.width >= 7680:
            return False

        # Very short (< 5 seconds) — not worth it
        if info.duration < 5:
            return False

        # Already high quality 4K+ with good bitrate
        if info.width >= 3840 and info.bitrate > 50_000_000 and not info.is_vr:
            return False

        return True
