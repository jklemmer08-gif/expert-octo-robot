"""Tests for TensorRT inference engine module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.trt_engine import ResolutionBucket, RVMTensorRTEngine


class TestResolutionBucket:

    def test_matches_within_range(self):
        bucket = ResolutionBucket("1080p", 720, 1280, 1080, 1920, 1152, 2048)
        assert bucket.matches(1080, 1920) is True
        assert bucket.matches(720, 1280) is True
        assert bucket.matches(1152, 2048) is True

    def test_matches_outside_range(self):
        bucket = ResolutionBucket("1080p", 720, 1280, 1080, 1920, 1152, 2048)
        assert bucket.matches(2160, 3840) is False
        assert bucket.matches(480, 640) is False

    def test_matches_height_in_width_out(self):
        bucket = ResolutionBucket("1080p", 720, 1280, 1080, 1920, 1152, 2048)
        assert bucket.matches(1080, 4000) is False


class TestRVMTensorRTEngine:

    @pytest.fixture
    def trt_settings(self, tmp_path):
        return Settings(
            tensorrt=Settings.model_fields["tensorrt"].default_factory().__class__(
                enabled=True,
                cache_dir=str(tmp_path / "trt_cache"),
                fp16=True,
                max_workspace_mb=2048,
            ),
            paths=Settings.model_fields["paths"].default_factory().__class__(
                models_dir=str(tmp_path / "models"),
            ),
        )

    def test_init_creates_cache_dir(self, trt_settings, tmp_path):
        engine = RVMTensorRTEngine(trt_settings)
        assert engine.cache_dir.exists()

    def test_select_bucket_exact_match(self, trt_settings):
        engine = RVMTensorRTEngine(trt_settings)
        bucket = engine.select_bucket(1080, 1920)
        assert bucket is not None
        assert bucket.name == "1080p"

    def test_select_bucket_4k(self, trt_settings):
        engine = RVMTensorRTEngine(trt_settings)
        bucket = engine.select_bucket(2160, 3840)
        assert bucket is not None
        assert bucket.name == "4k"

    def test_select_bucket_vr(self, trt_settings):
        engine = RVMTensorRTEngine(trt_settings)
        bucket = engine.select_bucket(1920, 1920)
        assert bucket is not None
        assert bucket.name == "vr"

    def test_select_bucket_closest_fallback(self, trt_settings):
        """When no exact match, should find closest bucket."""
        engine = RVMTensorRTEngine(trt_settings)
        bucket = engine.select_bucket(500, 900)
        assert bucket is not None  # Should find 720p as closest

    def test_default_buckets_loaded(self, trt_settings):
        engine = RVMTensorRTEngine(trt_settings)
        assert len(engine.buckets) == 5
        names = [b.name for b in engine.buckets]
        assert "720p" in names
        assert "4k" in names
        assert "vr" in names

    def test_reset_recurrent_state(self, trt_settings):
        engine = RVMTensorRTEngine(trt_settings)

        mock_torch = MagicMock()
        mock_torch.zeros.return_value = MagicMock()
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        with patch("src.trt_engine._import_torch", return_value=mock_torch):
            engine.reset_recurrent_state()

        assert len(engine._rec_states) == 4
        assert mock_torch.zeros.call_count == 4

    def test_cleanup(self, trt_settings):
        engine = RVMTensorRTEngine(trt_settings)
        engine._engine = MagicMock()
        engine._context = MagicMock()
        engine._stream = MagicMock()
        engine._current_bucket = MagicMock()

        engine.cleanup()

        assert engine._engine is None
        assert engine._context is None
        assert engine._stream is None
        assert engine._current_bucket is None

    def test_infer_without_prepare_raises(self, trt_settings):
        engine = RVMTensorRTEngine(trt_settings)
        mock_torch = MagicMock()
        mock_np = MagicMock()

        with patch("src.trt_engine._import_trt"), \
             patch("src.trt_engine._import_torch", return_value=mock_torch), \
             patch("src.trt_engine._import_numpy", return_value=mock_np):
            with pytest.raises(RuntimeError, match="TRT engine not loaded"):
                engine.infer(MagicMock())
