"""PPP_QACheck — Quality assurance validation node for ComfyUI."""

from __future__ import annotations

import sys
from pathlib import Path

_PPP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_PPP_ROOT))

from src.qa import QACheckpoint, QAThresholds


class PPP_QACheck:
    """Run QA checks on a processed video (frame count, resolution, SSIM/PSNR)."""

    CATEGORY = "PPP/QA"
    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "ssim", "psnr", "summary")
    FUNCTION = "check"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "reference_path": ("STRING", {"default": "", "multiline": False}),
                "expected_frame_count": ("INT", {"default": 0, "min": 0}),
                "expected_width": ("INT", {"default": 0, "min": 0}),
                "expected_height": ("INT", {"default": 0, "min": 0}),
                "check_matte": ("BOOLEAN", {"default": False}),
                "min_ssim": ("FLOAT", {"default": 0.97, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_psnr": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5}),
            },
        }

    def check(
        self,
        output_path: str,
        reference_path: str = "",
        expected_frame_count: int = 0,
        expected_width: int = 0,
        expected_height: int = 0,
        check_matte: bool = False,
        min_ssim: float = 0.97,
        min_psnr: float = 30.0,
    ):
        thresholds = QAThresholds(min_ssim=min_ssim, min_psnr=min_psnr)
        qa = QACheckpoint(thresholds)

        ref = Path(reference_path) if reference_path else None
        exp_res = (expected_width, expected_height) if expected_width > 0 and expected_height > 0 else None
        exp_frames = expected_frame_count if expected_frame_count > 0 else None

        result = qa.run_stage_checkpoint(
            stage="comfyui-qa",
            output_path=Path(output_path),
            reference_path=ref,
            expected_frame_count=exp_frames,
            expected_resolution=exp_res,
            check_matte=check_matte,
        )

        ssim = result.metrics.get("ssim", 0.0)
        psnr = result.metrics.get("psnr", 0.0)

        return (result.passed, ssim, psnr, result.summary())
