"""PPP_VideoAnalyze — Video analysis/probing node for ComfyUI."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the PPP processor src is importable
_PPP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_PPP_ROOT))

from src.analyzer import VideoAnalyzer
from src.config import Settings


class PPP_VideoAnalyze:
    """Probe a video file and extract metadata (resolution, fps, VR detection)."""

    CATEGORY = "PPP/Analysis"
    RETURN_TYPES = ("PPP_VIDEO_INFO", "INT", "INT", "FLOAT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("VIDEO_INFO", "width", "height", "fps", "duration", "is_vr", "vr_type")
    FUNCTION = "analyze"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "config_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def analyze(self, video_path: str, config_path: str = ""):
        config = config_path if config_path else None
        settings = Settings.from_yaml(Path(config) if config else None)
        analyzer = VideoAnalyzer(settings)

        info = analyzer.probe_video(Path(video_path))

        return (
            info,
            info.width,
            info.height,
            info.fps,
            info.duration,
            info.is_vr,
            info.vr_type or "",
        )
