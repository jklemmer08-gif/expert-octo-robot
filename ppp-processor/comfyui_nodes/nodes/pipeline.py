"""PPP_FullPipeline — End-to-end orchestrator node for ComfyUI."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PPP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_PPP_ROOT))

from src.analyzer import VideoAnalyzer
from src.config import Settings
from src.models.schemas import ProcessingPlan
from src.processor import MatteProcessor, ProcessingPipeline
from src.qa import QACheckpoint
from src.utils import bitrate_for_resolution


class PPP_FullPipeline:
    """Full processing pipeline: analyze -> matte -> (QA) -> (upscale) -> encode."""

    CATEGORY = "PPP/Pipeline"
    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("output_path", "processing_time", "qa_result")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "job_type": (["matte", "upscale", "matte+upscale"], {"default": "matte"}),
                "model_type": (["mobilenetv3", "resnet50"], {"default": "mobilenetv3"}),
                "upscale_model": (["realesr-animevideov3", "realesrgan-x4plus", "lanczos"], {"default": "realesr-animevideov3"}),
                "scale": ("INT", {"default": 2, "min": 1, "max": 4}),
                "encoder": (["hevc_nvenc", "libx265"], {"default": "hevc_nvenc"}),
                "use_tensorrt": ("BOOLEAN", {"default": False}),
                "run_qa": ("BOOLEAN", {"default": True}),
                "config_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def run(
        self,
        video_path: str,
        output_path: str,
        job_type: str = "matte",
        model_type: str = "mobilenetv3",
        upscale_model: str = "realesr-animevideov3",
        scale: int = 2,
        encoder: str = "hevc_nvenc",
        use_tensorrt: bool = False,
        run_qa: bool = True,
        config_path: str = "",
    ):
        config = config_path if config_path else None
        settings = Settings.from_yaml(Path(config) if config else None)

        settings.matte.model_type = model_type
        settings.tensorrt.enabled = use_tensorrt
        settings.encode.encoder = encoder

        start_time = time.time()
        vpath = Path(video_path)
        opath = Path(output_path)
        opath.parent.mkdir(parents=True, exist_ok=True)

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

        # 1. Analyze
        analyzer = VideoAnalyzer(settings)
        info = analyzer.probe_video(vpath)

        qa_summary = ""

        # 2. Matte (if requested)
        if job_type in ("matte", "matte+upscale"):
            matte_output = opath if job_type == "matte" else opath.with_suffix(".matte.mp4")
            processor = MatteProcessor(settings)
            success = processor.process_video(vpath, matte_output, info, progress_callback=_progress)
            if not success:
                raise RuntimeError("Matting failed")

            # QA after matte
            if run_qa:
                qa = QACheckpoint()
                result = qa.run_stage_checkpoint("post-matte", matte_output, check_matte=True)
                qa_summary += result.summary() + "\n"

            # If matte+upscale, the matte output becomes input for upscale
            if job_type == "matte+upscale":
                vpath = matte_output
                info = analyzer.probe_video(vpath)

        # 3. Upscale (if requested)
        if job_type in ("upscale", "matte+upscale"):
            bitrate = bitrate_for_resolution(info.width * scale, info.is_vr)
            plan = ProcessingPlan(
                model=upscale_model,
                scale=scale,
                worker_type="local",
                bitrate=bitrate,
                encoder=encoder,
                content_type=info.content_type,
                skip_ai=(upscale_model == "lanczos"),
                tile_size=settings.gpu.tile_size,
                gpu_id=settings.gpu.device_id,
            )

            pipeline = ProcessingPipeline(settings)
            success = pipeline.run(vpath, opath, plan, info, progress_callback=_progress)
            if not success:
                raise RuntimeError("Upscale pipeline failed")

            # QA after upscale
            if run_qa:
                qa = QACheckpoint()
                result = qa.run_stage_checkpoint(
                    "post-upscale", opath,
                    expected_resolution=(info.width * scale, info.height * scale),
                )
                qa_summary += result.summary() + "\n"

            # Clean up intermediate matte file
            if job_type == "matte+upscale":
                matte_output = opath.with_suffix(".matte.mp4")
                if matte_output.exists():
                    matte_output.unlink()

        elapsed = time.time() - start_time
        return (output_path, elapsed, qa_summary.strip() or "No QA run")
