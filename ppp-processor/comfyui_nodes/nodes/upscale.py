"""PPP_UpscaleVideo — AI upscale node for ComfyUI."""

from __future__ import annotations

import sys
from pathlib import Path

_PPP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_PPP_ROOT))

from src.config import Settings
from src.models.schemas import ContentType, ProcessingPlan, VideoInfo
from src.processor import ProcessingPipeline
from src.utils import bitrate_for_resolution


class PPP_UpscaleVideo:
    """Upscale a video using Real-ESRGAN (AI) or Lanczos."""

    CATEGORY = "PPP/Processing"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "VIDEO_INFO": ("PPP_VIDEO_INFO",),
                "model": (["realesr-animevideov3", "realesrgan-x4plus", "lanczos"], {"default": "realesr-animevideov3"}),
                "scale": ("INT", {"default": 2, "min": 1, "max": 4}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "config_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def upscale(
        self,
        video_path: str,
        output_path: str,
        VIDEO_INFO=None,
        model: str = "realesr-animevideov3",
        scale: int = 2,
        tile_size: int = 512,
        config_path: str = "",
    ):
        config = config_path if config_path else None
        settings = Settings.from_yaml(Path(config) if config else None)

        info = VIDEO_INFO
        if info is None:
            from src.analyzer import VideoAnalyzer
            analyzer = VideoAnalyzer(settings)
            info = analyzer.probe_video(Path(video_path))

        bitrate = bitrate_for_resolution(info.width * scale, info.is_vr)

        plan = ProcessingPlan(
            model=model,
            scale=scale,
            worker_type="local",
            bitrate=bitrate,
            encoder=settings.encode.encoder,
            content_type=info.content_type,
            skip_ai=(model == "lanczos"),
            tile_size=tile_size,
            gpu_id=settings.gpu.device_id,
        )

        pipeline = ProcessingPipeline(settings)

        def _progress(stage: str, pct: float):
            try:
                from server import PromptServer
                PromptServer.instance.send_sync("progress", {
                    "node": self.__class__.__name__,
                    "stage": stage,
                    "value": pct,
                    "max": 100,
                })
            except Exception:
                pass

        success = pipeline.run(
            Path(video_path),
            Path(output_path),
            plan,
            info,
            progress_callback=_progress,
        )

        if not success:
            raise RuntimeError(f"Upscale failed for {video_path}")

        return (output_path,)
