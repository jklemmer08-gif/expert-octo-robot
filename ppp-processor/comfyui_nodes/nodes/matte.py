"""PPP_MatteVideo — Background matting node for ComfyUI."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PPP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_PPP_ROOT))

from src.config import Settings
from src.processor import MatteProcessor

# Cached processor instance (GPU model stays loaded across executions)
_cached_processor: MatteProcessor | None = None
_cached_settings_hash: int | None = None


def _get_processor(settings: Settings) -> MatteProcessor:
    """Return a cached MatteProcessor, reusing the GPU model across runs."""
    global _cached_processor, _cached_settings_hash
    h = hash((settings.matte.model_type, settings.matte.downsample_ratio))
    if _cached_processor is None or _cached_settings_hash != h:
        _cached_processor = MatteProcessor(settings)
        _cached_settings_hash = h
    return _cached_processor


class PPP_MatteVideo:
    """Run RVM background matting on a video (streaming FFmpeg pipeline)."""

    CATEGORY = "PPP/Processing"
    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("output_path", "frame_count", "fps_achieved")
    FUNCTION = "matte"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "VIDEO_INFO": ("PPP_VIDEO_INFO",),
                "model_type": (["mobilenetv3", "resnet50"], {"default": "mobilenetv3"}),
                "green_color": ("STRING", {"default": "#00B140"}),
                "downsample_ratio": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05}),
                "use_nvenc": ("BOOLEAN", {"default": True}),
                "use_tensorrt": ("BOOLEAN", {"default": False}),
                "config_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def matte(
        self,
        video_path: str,
        output_path: str,
        VIDEO_INFO=None,
        model_type: str = "mobilenetv3",
        green_color: str = "#00B140",
        downsample_ratio: float = 0.25,
        use_nvenc: bool = True,
        use_tensorrt: bool = False,
        config_path: str = "",
    ):
        config = config_path if config_path else None
        settings = Settings.from_yaml(Path(config) if config else None)

        # Override settings from node inputs
        settings.matte.model_type = model_type
        settings.matte.downsample_ratio = downsample_ratio
        settings.tensorrt.enabled = use_tensorrt

        # Parse hex green color
        gc = green_color.lstrip("#")
        if len(gc) == 6:
            settings.matte.green_color = [int(gc[i:i+2], 16) for i in (0, 2, 4)]

        if not use_nvenc:
            settings.encode.encoder = settings.encode.fallback_encoder

        processor = _get_processor(settings)

        # Build progress tracker
        frame_count = 0
        start_time = time.time()

        def _progress(stage: str, pct: float):
            nonlocal frame_count
            # Push progress to ComfyUI if available
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

        info = VIDEO_INFO
        if info is None:
            from src.analyzer import VideoAnalyzer
            analyzer = VideoAnalyzer(settings)
            info = analyzer.probe_video(Path(video_path))

        success = processor.process_video(
            Path(video_path),
            Path(output_path),
            info,
            progress_callback=_progress,
        )

        elapsed = time.time() - start_time

        # Get output frame count from the file
        if success:
            probe = processor._probe_video_pipe_info(Path(output_path))
            frame_count = probe["total_frames"]
            fps_achieved = frame_count / elapsed if elapsed > 0 else 0.0
        else:
            fps_achieved = 0.0

        return (output_path, frame_count, fps_achieved)
