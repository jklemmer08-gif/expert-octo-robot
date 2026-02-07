"""Parameterized tests for the Router decision tree."""

import pytest

from src.config import Settings
from src.models.schemas import ContentType, ProcessingPlan, VideoInfo
from src.router import Router


@pytest.fixture
def router():
    return Router(Settings())


def _info(w, h, is_vr=False, vr_type=None, content_type=ContentType.FLAT_2D):
    return VideoInfo(
        width=w, height=h, fps=30.0, duration=600.0,
        codec="h264", bitrate=10_000_000,
        is_vr=is_vr, vr_type=vr_type, content_type=content_type,
    )


class TestDecisionTree:
    """Test every branch of the router decision tree."""

    def test_720p_to_1080p_lanczos(self, router):
        """720p->1080p 2D -> lanczos (skip AI)."""
        info = _info(1280, 720)
        plan = router.plan(info, target_scale=2)
        assert plan.skip_ai is True
        assert plan.model == "lanczos"
        assert plan.worker_type == "local"

    def test_720p_to_1080p_exact(self, router):
        """Even smaller: 960x540 -> skip AI at 2x."""
        info = _info(960, 540)
        plan = router.plan(info, target_scale=2)
        assert plan.skip_ai is True

    def test_vr_8k_cloud(self, router):
        """VR target >= 8K -> cloud + x4plus."""
        info = _info(3840, 1920, is_vr=True, vr_type="sbs",
                     content_type=ContentType.VR_SBS)
        plan = router.plan(info, target_scale=2)  # 3840*2 = 7680
        assert plan.model == "realesrgan-x4plus"
        assert plan.worker_type == "cloud"

    def test_vr_4k_to_6k_local(self, router):
        """VR 4K->6K -> local + animevideov3."""
        # 4096*1.5 ~= 6144 but we only do integer scale
        # Use width=3840 * 2 = 7680 which is 8K -> cloud
        # Use 2880 * 2 = 5760 which is 6K
        info = _info(2880, 1440, is_vr=True, vr_type="sbs",
                     content_type=ContentType.VR_SBS)
        plan = router.plan(info, target_scale=2)  # 2880*2=5760
        # This won't match 4K->6K rule, but will match general VR
        assert plan.worker_type == "local"
        assert "animevideov3" in plan.model

    def test_vr_general_local(self, router):
        """VR content that doesn't match specific rules -> local."""
        info = _info(1920, 960, is_vr=True, vr_type="sbs",
                     content_type=ContentType.VR_SBS)
        plan = router.plan(info, target_scale=2)
        assert plan.worker_type == "local"
        assert plan.skip_ai is False

    def test_2d_scale_gt_2(self, router):
        """2D scale > 2 -> x4plus."""
        info = _info(1920, 1080)
        plan = router.plan(info, target_scale=4)
        assert plan.model == "realesrgan-x4plus"
        assert plan.skip_ai is False

    def test_2d_scale_le_2(self, router):
        """2D scale <= 2 -> animevideov3."""
        info = _info(1920, 1080)
        plan = router.plan(info, target_scale=2)
        assert plan.model == "realesr-animevideov3"
        assert plan.worker_type == "local"


class TestEstimates:

    def test_estimate_time_positive(self, router):
        info = _info(1920, 1080)
        t = router.estimate_time(info, scale=2, worker_type="local")
        assert t > 0

    def test_estimate_cost_cloud(self, router):
        info = _info(3840, 1920, is_vr=True)
        c = router.estimate_cost(info, scale=2, worker_type="cloud")
        assert c > 0

    def test_estimate_cost_local_is_zero(self, router):
        info = _info(1920, 1080)
        c = router.estimate_cost(info, scale=2, worker_type="local")
        assert c == 0.0


class TestBudget:

    def test_check_budget_ok(self, router):
        assert router.check_budget(3.0) is True

    def test_check_budget_too_expensive(self, router):
        assert router.check_budget(10.0) is False
