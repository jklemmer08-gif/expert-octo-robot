"""PPP_EncodeVideo — Video encoding node for ComfyUI."""

from __future__ import annotations

import sys
from pathlib import Path

_PPP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_PPP_ROOT))

from src.config import Settings
from src.processor import Encoder


class PPP_EncodeVideo:
    """Encode extracted frames to a video file using FFmpeg."""

    CATEGORY = "PPP/Output"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
            },
            "optional": {
                "audio_source": ("STRING", {"default": "", "multiline": False}),
                "bitrate": ("STRING", {"default": "15M"}),
                "encoder": (["hevc_nvenc", "libx265"], {"default": "hevc_nvenc"}),
                "config_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def encode(
        self,
        frames_dir: str,
        output_path: str,
        fps: float = 30.0,
        audio_source: str = "",
        bitrate: str = "15M",
        encoder: str = "hevc_nvenc",
        config_path: str = "",
    ):
        config = config_path if config_path else None
        settings = Settings.from_yaml(Path(config) if config else None)

        enc = Encoder(settings)
        audio_path = Path(audio_source) if audio_source else None

        success = enc.encode(
            Path(frames_dir),
            Path(output_path),
            fps,
            audio_source=audio_path,
            bitrate=bitrate,
            encoder=encoder,
        )

        if not success:
            raise RuntimeError(f"Encoding failed for {output_path}")

        return (output_path,)
