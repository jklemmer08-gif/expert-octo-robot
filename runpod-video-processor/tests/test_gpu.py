"""Tests for GPU detection and profile selection (no GPU required — all mocked)."""

import pytest
from unittest.mock import patch, MagicMock
from src.gpu import detect_gpu, select_profile, get_gpu_profile, get_vram_usage, PROFILES


class TestSelectProfile:
    """Profile selection based on VRAM and GPU name."""

    def test_a100_80gb_by_name_and_vram(self):
        p = select_profile(80.0, "NVIDIA A100-SXM4-80GB")
        assert p.name == "A100-80GB"
        assert p.tile_size == 1024

    def test_a100_40gb_by_name(self):
        p = select_profile(40.0, "NVIDIA A100-PCIE-40GB")
        assert p.name == "A100-40GB"
        assert p.tile_size == 768

    def test_l40s_by_name(self):
        p = select_profile(48.0, "NVIDIA L40S")
        assert p.name == "L40S-48GB"
        assert p.tile_size == 768

    def test_24gb_rtx_4090(self):
        p = select_profile(24.0, "NVIDIA GeForce RTX 4090")
        assert p.name == "24GB"
        assert p.tile_size == 512
        assert p.batch_size == 4

    def test_24gb_rtx_3090(self):
        p = select_profile(24.0, "NVIDIA GeForce RTX 3090")
        assert p.name == "24GB"

    def test_24gb_a40(self):
        p = select_profile(48.0, "NVIDIA A40")
        assert p.name == "L40S-48GB"  # 48GB VRAM, no L40S in name, uses VRAM tier

    def test_16gb_gpu(self):
        p = select_profile(16.0, "NVIDIA Tesla T4")
        assert p.name == "16GB"
        assert p.tile_size == 384

    def test_fallback_low_vram(self):
        p = select_profile(8.0, "NVIDIA GeForce GTX 1080")
        assert p.name == "fallback"
        assert p.tile_size == 256
        assert p.batch_size == 1

    def test_fallback_no_gpu(self):
        p = select_profile(0, None)
        assert p.name == "fallback"

    def test_vram_based_no_name(self):
        """When GPU name is not known, use pure VRAM thresholds."""
        p = select_profile(80.0, None)
        assert p.tile_size == 1024

        p = select_profile(48.0, None)
        assert p.tile_size == 768

        p = select_profile(24.0, None)
        assert p.tile_size == 512


class TestDetectGPU:
    """GPU detection with mocked torch.cuda."""

    def test_no_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = detect_gpu()
        assert info["name"] is None
        assert info["vram_gb"] == 0

    def test_cuda_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_props = MagicMock()
        mock_props.total_mem = 24 * (1024 ** 3)  # 24 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.device_count.return_value = 1
        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = detect_gpu()
        assert info["name"] == "NVIDIA RTX 4090"
        assert info["vram_gb"] == 24.0
        assert info["count"] == 1

    def test_import_error(self):
        with patch.dict("sys.modules", {"torch": None}):
            # detect_gpu should handle ImportError gracefully
            info = detect_gpu()
        assert info["vram_gb"] == 0


class TestGetGPUProfile:
    """Integration: detect_gpu -> select_profile."""

    def test_returns_profile(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100-SXM4-80GB"
        mock_props = MagicMock()
        mock_props.total_mem = 80 * (1024 ** 3)
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.device_count.return_value = 1
        with patch.dict("sys.modules", {"torch": mock_torch}):
            profile = get_gpu_profile()
        assert profile.name == "A100-80GB"
        assert profile.segment_size == 2000


class TestGetVRAMUsage:
    """VRAM usage stats with mocked torch.cuda."""

    def test_no_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            usage = get_vram_usage()
        assert usage["total_gb"] == 0

    def test_with_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_mem = 24 * (1024 ** 3)
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_reserved.return_value = 4 * (1024 ** 3)
        mock_torch.cuda.memory_allocated.return_value = 2 * (1024 ** 3)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            usage = get_vram_usage()
        assert usage["total_gb"] == 24.0
        assert usage["used_gb"] == 2.0
        assert usage["free_gb"] == 20.0
