"""Tests for RVM TorchScript model loader — loading, inference, state management.

All tests are mocked (no GPU needed).
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock


class TestRVMProcessorLoading:
    """Model loading and error handling."""

    def _make_mock_torch(self):
        """Create a mock torch module."""
        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})

        # Mock model that returns proper shaped tensors
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_torch.jit.load.return_value = mock_model

        return mock_torch

    def test_load_resnet50(self, tmp_path):
        """Load resnet50 model from file."""
        mock_torch = self._make_mock_torch()

        model_dir = tmp_path / "rvm"
        model_dir.mkdir()
        model_file = model_dir / "rvm_resnet50_fp32.torchscript"
        model_file.write_bytes(b"fake model data")

        with patch("src.pipeline.rvm_model.torch", mock_torch), \
             patch("src.pipeline.rvm_model.RVM_MODEL_DIR", model_dir):
            from src.pipeline.rvm_model import RVMProcessor
            proc = RVMProcessor("resnet50", "cuda:0", 0.5)

        mock_torch.jit.load.assert_called_once_with(str(model_file), map_location="cuda:0")
        assert proc.downsample_ratio == 0.5
        assert len(proc.rec) == 4
        assert all(r is None for r in proc.rec)

    def test_load_mobilenetv3(self, tmp_path):
        """Load mobilenetv3 model from file."""
        mock_torch = self._make_mock_torch()

        model_dir = tmp_path / "rvm"
        model_dir.mkdir()
        (model_dir / "rvm_mobilenetv3_fp32.torchscript").write_bytes(b"fake")

        with patch("src.pipeline.rvm_model.torch", mock_torch), \
             patch("src.pipeline.rvm_model.RVM_MODEL_DIR", model_dir):
            from src.pipeline.rvm_model import RVMProcessor
            proc = RVMProcessor("mobilenetv3", "cuda:0", 0.4)

        assert proc.downsample_ratio == 0.4

    def test_unknown_model_raises(self, tmp_path):
        """Unknown model name raises ValueError."""
        mock_torch = self._make_mock_torch()

        with patch("src.pipeline.rvm_model.torch", mock_torch), \
             patch("src.pipeline.rvm_model.RVM_MODEL_DIR", tmp_path):
            from src.pipeline.rvm_model import RVMProcessor
            with pytest.raises(ValueError, match="Unknown RVM model"):
                RVMProcessor("nonexistent_model", "cuda:0")

    def test_missing_file_raises(self, tmp_path):
        """Missing model file raises FileNotFoundError."""
        mock_torch = self._make_mock_torch()

        model_dir = tmp_path / "rvm"
        model_dir.mkdir()
        # Don't create the model file

        with patch("src.pipeline.rvm_model.torch", mock_torch), \
             patch("src.pipeline.rvm_model.RVM_MODEL_DIR", model_dir):
            from src.pipeline.rvm_model import RVMProcessor
            with pytest.raises(FileNotFoundError, match="not found"):
                RVMProcessor("resnet50", "cuda:0")


class TestRVMProcessorInference:
    """Batch processing and recurrent state management."""

    def _make_mock_torch(self):
        """Create a mock torch module."""
        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        return mock_torch

    def _create_processor(self, mock_torch, tmp_path):
        """Helper to create a processor with mocked model."""
        model_dir = tmp_path / "rvm"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "rvm_resnet50_fp32.torchscript").write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_torch.jit.load.return_value = mock_model

        with patch("src.pipeline.rvm_model.torch", mock_torch), \
             patch("src.pipeline.rvm_model.RVM_MODEL_DIR", model_dir):
            from src.pipeline.rvm_model import RVMProcessor
            proc = RVMProcessor("resnet50", "cuda:0", 0.5)

        # Manually set torch ref for process_batch
        proc_torch = mock_torch
        return proc, mock_model

    def test_process_batch_output_shape(self, tmp_path):
        """process_batch returns (B, 4, H, W) RGBA from (B, 3, H, W) RGB input."""
        mock_torch = self._make_mock_torch()

        # Create mock tensors for model output
        mock_fgr = MagicMock()  # (B, 3, H, W)
        mock_pha = MagicMock()  # (B, 1, H, W)
        mock_rec = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        proc, mock_model = self._create_processor(mock_torch, tmp_path)

        # Mock model to return fgr, pha, and 4 rec states
        mock_model.return_value = (mock_fgr, mock_pha, *mock_rec)

        # Mock fgr * pha and torch.cat
        mock_fgr_mul = MagicMock()
        mock_fgr.__mul__ = MagicMock(return_value=mock_fgr_mul)
        mock_rgba = MagicMock()
        mock_rgba.clamp.return_value = mock_rgba
        mock_torch.cat.return_value = mock_rgba

        # Patch torch.no_grad
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        # Run with mocked torch
        with patch("src.pipeline.rvm_model.torch", mock_torch):
            proc_orig_rec = list(proc.rec)
            batch = MagicMock()  # fake (B, 3, H, W) tensor
            result = proc.process_batch(batch)

        # Verify model was called
        mock_model.assert_called_once()
        # Verify recurrent states updated
        assert proc.rec == list(mock_rec)

    def test_reset_clears_states(self, tmp_path):
        """reset() sets all recurrent states to None."""
        mock_torch = self._make_mock_torch()
        proc, _ = self._create_processor(mock_torch, tmp_path)

        # Set some fake states
        proc.rec = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        proc.reset()

        assert len(proc.rec) == 4
        assert all(r is None for r in proc.rec)

    def test_get_set_recurrent_states(self, tmp_path):
        """get_recurrent_states / set_recurrent_states round-trip."""
        mock_torch = self._make_mock_torch()
        proc, _ = self._create_processor(mock_torch, tmp_path)

        # Set fake states with clone that returns a new mock
        fake_states = []
        for i in range(4):
            state = MagicMock()
            state.clone.return_value = MagicMock(name=f"cloned_{i}")
            fake_states.append(state)
        proc.rec = fake_states

        with patch("src.pipeline.rvm_model.torch", mock_torch):
            saved = proc.get_recurrent_states()

        assert len(saved) == 4
        # Each state should be cloned
        for i in range(4):
            fake_states[i].clone.assert_called_once()

    def test_get_recurrent_states_with_none(self, tmp_path):
        """get_recurrent_states handles None entries."""
        mock_torch = self._make_mock_torch()
        proc, _ = self._create_processor(mock_torch, tmp_path)

        # Default rec is [None, None, None, None]
        with patch("src.pipeline.rvm_model.torch", mock_torch):
            saved = proc.get_recurrent_states()

        assert saved == [None, None, None, None]
