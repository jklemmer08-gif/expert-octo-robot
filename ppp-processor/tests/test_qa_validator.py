"""Tests for QA validation and auto-approval logic."""

import pytest

from src.config import Settings
from src.database import JobDatabase
from src.qa_validator import QAValidator


@pytest.fixture
def qa(test_settings, test_db):
    return QAValidator(test_settings, test_db)


class TestAutoApprove:

    def test_approve_high_quality(self, qa):
        """SSIM >= 0.85 and PSNR >= 28 -> auto-approve."""
        result = qa.auto_approve(0.92, 32.5)
        assert result is True

    def test_reject_low_quality(self, qa):
        """Well below thresholds -> reject."""
        result = qa.auto_approve(0.60, 20.0)
        assert result is False

    def test_human_review_borderline(self, qa):
        """Within tolerance of thresholds -> None (human review)."""
        # SSIM threshold 0.85 * 0.9 = 0.765, PSNR threshold 28 * 0.9 = 25.2
        result = qa.auto_approve(0.80, 26.0)
        assert result is None

    def test_none_when_no_metrics(self, qa):
        """No metrics available -> None."""
        result = qa.auto_approve(None, None)
        assert result is None

    def test_ssim_ok_psnr_bad(self, qa):
        """One metric OK, other bad -> depends on tolerance."""
        result = qa.auto_approve(0.90, 20.0)
        assert result is False

    def test_exact_threshold(self, qa):
        """Exactly at threshold -> approve."""
        result = qa.auto_approve(0.85, 28.0)
        assert result is True

    def test_just_below_threshold(self, qa):
        """Just below threshold but within tolerance -> human review."""
        result = qa.auto_approve(0.84, 27.5)
        assert result is None


class TestComputeSharpness:

    def test_sharpness_flat(self, qa):
        """A flat image has low sharpness."""
        import numpy as np
        flat = np.ones((100, 100), dtype=np.float64) * 128
        sharpness = qa._compute_sharpness(flat)
        assert sharpness == 0.0

    def test_sharpness_noisy(self, qa):
        """A noisy image has high sharpness."""
        import numpy as np
        rng = np.random.default_rng(42)
        noisy = rng.random((100, 100)) * 255
        sharpness = qa._compute_sharpness(noisy)
        assert sharpness > 1000
