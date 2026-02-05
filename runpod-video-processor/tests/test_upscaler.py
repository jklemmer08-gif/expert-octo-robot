"""Tests for Real-ESRGAN upscaler — single frame, VR split/merge, OOM retry.

Tests marked 'integration' require a GPU and Real-ESRGAN models.
Non-integration tests use mocks.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from src.pipeline.upscaler import (
    _split_vr_frame,
    _merge_vr_frame,
    upscale_frame_with_oom_retry,
    UpscaleError,
)
from src.pipeline.detector import VRLayout


class TestVRSplitMerge:
    """VR frame splitting and merging (pure numpy, no GPU)."""

    def test_sbs_split(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :100] = 255  # left eye white
        left, right = _split_vr_frame(frame, VRLayout.SBS)
        assert left.shape == (100, 100, 3)
        assert right.shape == (100, 100, 3)
        assert left.mean() == 255
        assert right.mean() == 0

    def test_ou_split(self):
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[:100, :] = 255  # top eye white
        top, bottom = _split_vr_frame(frame, VRLayout.OU)
        assert top.shape == (100, 100, 3)
        assert bottom.shape == (100, 100, 3)
        assert top.mean() == 255
        assert bottom.mean() == 0

    def test_sbs_merge(self):
        left = np.ones((100, 100, 3), dtype=np.uint8) * 128
        right = np.ones((100, 100, 3), dtype=np.uint8) * 64
        merged = _merge_vr_frame(left, right, VRLayout.SBS)
        assert merged.shape == (100, 200, 3)
        assert merged[:, :100].mean() == 128
        assert merged[:, 100:].mean() == 64

    def test_ou_merge(self):
        top = np.ones((50, 100, 3), dtype=np.uint8) * 200
        bottom = np.ones((50, 100, 3), dtype=np.uint8) * 100
        merged = _merge_vr_frame(top, bottom, VRLayout.OU)
        assert merged.shape == (100, 100, 3)
        assert merged[:50].mean() == 200
        assert merged[50:].mean() == 100

    def test_split_merge_roundtrip_sbs(self):
        original = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        left, right = _split_vr_frame(original, VRLayout.SBS)
        merged = _merge_vr_frame(left, right, VRLayout.SBS)
        np.testing.assert_array_equal(original, merged)

    def test_split_merge_roundtrip_ou(self):
        original = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        top, bottom = _split_vr_frame(original, VRLayout.OU)
        merged = _merge_vr_frame(top, bottom, VRLayout.OU)
        np.testing.assert_array_equal(original, merged)

    def test_split_invalid_layout(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            _split_vr_frame(frame, VRLayout.FLAT_2D)


class TestOOMRetry:
    """OOM retry logic with mocked model loading."""

    def _make_mock_torch(self):
        """Create a mock torch module with cuda.OutOfMemoryError."""
        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        mock_torch.cuda.empty_cache = MagicMock()
        return mock_torch

    def test_success_first_try(self):
        mock_upsampler = MagicMock()
        expected_output = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_upsampler.enhance.return_value = (expected_output, None)

        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch("src.pipeline.upscaler._load_model", return_value=mock_upsampler), \
             patch("src.pipeline.upscaler.torch", self._make_mock_torch(), create=True):
            result, tile = upscale_frame_with_oom_retry(
                frame, "RealESRGAN_x4plus", 4, "cuda:0", tile_sizes=[512, 256, 128]
            )

        assert tile == 512
        np.testing.assert_array_equal(result, expected_output)

    def test_oom_retries_with_smaller_tiles(self):
        call_count = 0
        expected_output = np.zeros((256, 256, 3), dtype=np.uint8)

        def mock_load(model_name, scale, tile_size, device):
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            if tile_size > 256:
                mock.enhance.side_effect = RuntimeError("CUDA out of memory")
            else:
                mock.enhance.return_value = (expected_output, None)
            return mock

        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch("src.pipeline.upscaler._load_model", side_effect=mock_load), \
             patch("src.pipeline.upscaler.torch", self._make_mock_torch(), create=True):
            result, tile = upscale_frame_with_oom_retry(
                frame, "RealESRGAN_x4plus", 4, "cuda:0", tile_sizes=[512, 384, 256]
            )

        assert tile == 256
        assert call_count == 3

    def test_all_tiles_oom_raises(self):
        def mock_load(model_name, scale, tile_size, device):
            mock = MagicMock()
            mock.enhance.side_effect = RuntimeError("CUDA out of memory")
            return mock

        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch("src.pipeline.upscaler._load_model", side_effect=mock_load), \
             patch("src.pipeline.upscaler.torch", self._make_mock_torch(), create=True), \
             pytest.raises(UpscaleError, match="smallest tile size"):
            upscale_frame_with_oom_retry(
                frame, "RealESRGAN_x4plus", 4, "cuda:0", tile_sizes=[256, 128]
            )


@pytest.mark.integration
class TestUpscalerIntegration:
    """Integration tests requiring GPU + Real-ESRGAN models."""

    def test_single_frame_upscale(self):
        """Upscale a 64x64 test frame → verify 256x256 output."""
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result, tile = upscale_frame_with_oom_retry(
            frame, "RealESRGAN_x4plus", 4, "cuda:0"
        )
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8
