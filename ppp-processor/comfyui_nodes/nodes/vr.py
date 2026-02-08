"""PPP_VRSplit / PPP_VRMerge — VR stereoscopic processing nodes for ComfyUI."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_PPP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_PPP_ROOT))

from src.processor import FrameExtractor, VRProcessor


class PPP_VRSplit:
    """Split a VR stereoscopic video into left/right (SBS) or top/bottom (TB) eye views."""

    CATEGORY = "PPP/VR"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("left_path", "right_path")
    FUNCTION = "split"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "vr_type": (["sbs", "tb"], {"default": "sbs"}),
                "output_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def split(self, video_path: str, vr_type: str = "sbs", output_dir: str = ""):
        vpath = Path(video_path)
        if output_dir:
            base_dir = Path(output_dir)
        else:
            base_dir = Path(tempfile.mkdtemp(prefix="ppp_vr_"))

        frames_dir = base_dir / "frames"
        left_dir = base_dir / "left"
        right_dir = base_dir / "right"

        # Extract frames first
        extractor = FrameExtractor()
        extractor.extract(vpath, frames_dir)

        # Split
        vr = VRProcessor()
        if vr_type == "tb":
            vr.split_tb(frames_dir, left_dir, right_dir)
        else:
            vr.split_sbs(frames_dir, left_dir, right_dir)

        return (str(left_dir), str(right_dir))


class PPP_VRMerge:
    """Merge left/right eye frame directories back into a stereoscopic layout."""

    CATEGORY = "PPP/VR"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_path": ("STRING", {"default": "", "multiline": False}),
                "right_path": ("STRING", {"default": "", "multiline": False}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "vr_type": (["sbs", "tb"], {"default": "sbs"}),
            },
        }

    def merge(self, left_path: str, right_path: str, output_path: str, vr_type: str = "sbs"):
        out_dir = Path(output_path)
        vr = VRProcessor()

        if vr_type == "tb":
            vr.merge_tb(Path(left_path), Path(right_path), out_dir)
        else:
            vr.merge_sbs(Path(left_path), Path(right_path), out_dir)

        return (str(out_dir),)
