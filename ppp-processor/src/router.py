"""Model and worker selection engine for PPP Processor.

Implements the decision tree from the v2.0 plan:
- 720p->1080p 2D: lanczos (skip AI)
- VR target >= 8K: cloud + realesrgan-x4plus
- VR 4K->6K: local + realesr-animevideov3
- 2D scale <= 2: realesr-animevideov3
- 2D scale > 2: realesrgan-x4plus
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import Settings
from src.models.schemas import ContentType, ProcessingPlan, VideoInfo
from src.utils import bitrate_for_resolution, resolution_label

logger = logging.getLogger("ppp.router")


class Router:
    """Generates processing plans based on video analysis."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def plan(self, info: VideoInfo, target_scale: Optional[int] = None) -> ProcessingPlan:
        """Generate a ProcessingPlan for a given video.

        Applies the decision tree to select model, worker, and encoding settings.
        """
        scale = target_scale or self.settings.upscale.scale_factor
        target_width = info.width * scale
        target_height = info.height * scale

        # Decision tree
        model, worker_type, skip_ai = self._select_model_and_worker(
            info, scale, target_width
        )

        bitrate = bitrate_for_resolution(target_width, info.is_vr)

        # Encoder selection
        encoder = self._select_encoder()

        # Estimates
        est_time = self.estimate_time(info, scale, worker_type)
        est_cost = self.estimate_cost(info, scale, worker_type) if worker_type == "cloud" else 0.0

        return ProcessingPlan(
            model=model,
            scale=scale,
            worker_type=worker_type,
            bitrate=bitrate,
            encoder=encoder,
            content_type=info.content_type,
            estimated_time_hours=est_time,
            estimated_cost=est_cost,
            skip_ai=skip_ai,
            tile_size=self.settings.gpu.tile_size,
            gpu_id=self.settings.gpu.device_id,
        )

    def _select_model_and_worker(
        self, info: VideoInfo, scale: int, target_width: int
    ) -> tuple[str, str, bool]:
        """Apply the decision tree to select model and worker.

        Returns (model_name, worker_type, skip_ai).
        """
        # Rule 1: 720p->1080p 2D — use lanczos, skip AI
        # Source is 720p or lower and upscaling to at most 2x
        if (not info.is_vr
                and info.width <= 1280
                and scale <= 2):
            logger.info("720p->1080p 2D: using lanczos (skip AI)")
            return "lanczos", "local", True

        # Rule 2: VR target >= 8K — cloud + x4plus
        if info.is_vr and target_width >= 7680:
            logger.info("VR target >= 8K: cloud + realesrgan-x4plus")
            return "realesrgan-x4plus", "cloud", False

        # Rule 3: VR 4K->6K — local + animevideov3
        if info.is_vr and info.width >= 3840 and target_width >= 5760:
            logger.info("VR 4K->6K: local + realesr-animevideov3")
            return "realesr-animevideov3", "local", False

        # Rule 4: VR — general local
        if info.is_vr:
            logger.info("VR general: local + realesr-animevideov3")
            return "realesr-animevideov3", "local", False

        # Rule 5: 2D scale > 2 — x4plus (higher quality needed)
        if scale > 2:
            logger.info("2D scale > 2: realesrgan-x4plus")
            return "realesrgan-x4plus", "local", False

        # Rule 6: 2D scale <= 2 — animevideov3 (fast)
        logger.info("2D scale <= 2: realesr-animevideov3")
        return "realesr-animevideov3", "local", False

    def _select_encoder(self) -> str:
        """Select the best available encoder with fallback."""
        # Prefer hardware encoder from config, fallback to software
        return self.settings.encode.fallback_encoder

    def estimate_time(
        self, info: VideoInfo, scale: int, worker_type: str
    ) -> float:
        """Estimate processing time in hours.

        Rough heuristics based on observed processing rates:
        - Local GPU: ~10 frames/sec for animevideov3, ~5 fps for x4plus
        - Cloud GPU (4090): ~15 fps for animevideov3, ~8 fps for x4plus
        """
        total_frames = info.fps * info.duration
        if total_frames <= 0:
            return 0.0

        # VR SBS doubles the frame count (L+R eyes)
        if info.is_vr and info.vr_type == "sbs":
            total_frames *= 2

        # Estimate fps based on worker
        if worker_type == "cloud":
            processing_fps = 12.0
        else:
            processing_fps = 8.0

        # x4plus is roughly 2x slower
        if scale > 2:
            processing_fps *= 0.5

        hours = (total_frames / processing_fps) / 3600
        return round(hours, 2)

    def estimate_cost(
        self, info: VideoInfo, scale: int, worker_type: str
    ) -> float:
        """Estimate cloud processing cost in dollars.

        Based on RunPod RTX 4090 pricing (~$0.50/hr).
        """
        if worker_type != "cloud":
            return 0.0

        hours = self.estimate_time(info, scale, worker_type)
        # RTX 4090 ~ $0.50/hr on RunPod
        cost = hours * 0.50
        return round(cost, 2)

    def check_budget(self, estimated_cost: float) -> bool:
        """Check if estimated cost fits within the RunPod budget."""
        budget_total = self.settings.runpod.budget_total
        max_per_job = self.settings.runpod.max_cost_per_job

        if estimated_cost > max_per_job:
            logger.warning(
                "Estimated cost $%.2f exceeds per-job max $%.2f",
                estimated_cost, max_per_job,
            )
            return False

        return estimated_cost <= budget_total
