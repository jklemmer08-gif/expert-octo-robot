"""Tests for TensorRT engine export, build, and TRTUpscaler inference (all mocked)."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open


# ---------------------------------------------------------------------------
# Helpers to build mock torch / tensorrt modules
# ---------------------------------------------------------------------------

def _make_mock_torch(cuda_available=True, major=12, minor=0):
    """Create a mock torch module with cuda properties."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = cuda_available
    props = MagicMock()
    props.major = major
    props.minor = minor
    mock.cuda.get_device_properties.return_value = props
    mock.cuda.current_stream.return_value.cuda_stream = 0
    mock.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    return mock


def _make_mock_trt():
    """Create a mock tensorrt module."""
    mock = MagicMock()
    mock.Logger.WARNING = 2
    mock.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
    mock.BuilderFlag.FP16 = 0
    mock.BuilderFlag.BF16 = 1
    mock.MemoryPoolType.WORKSPACE = 0
    return mock


# ===========================================================================
# TestIsAvailable
# ===========================================================================

class TestIsAvailable:
    """Test is_available() under various import scenarios."""

    def test_available_when_both_present(self):
        mock_torch = _make_mock_torch()
        mock_trt = _make_mock_trt()
        with patch("src.pipeline.trt_engine.trt", mock_trt), \
             patch("src.pipeline.trt_engine.torch", mock_torch):
            from src.pipeline.trt_engine import is_available
            assert is_available() is True

    def test_unavailable_when_trt_missing(self):
        mock_torch = _make_mock_torch()
        with patch("src.pipeline.trt_engine.trt", None), \
             patch("src.pipeline.trt_engine.torch", mock_torch):
            from src.pipeline.trt_engine import is_available
            assert is_available() is False

    def test_unavailable_when_torch_missing(self):
        mock_trt = _make_mock_trt()
        with patch("src.pipeline.trt_engine.trt", mock_trt), \
             patch("src.pipeline.trt_engine.torch", None):
            from src.pipeline.trt_engine import is_available
            assert is_available() is False

    def test_unavailable_when_no_cuda(self):
        mock_torch = _make_mock_torch(cuda_available=False)
        mock_trt = _make_mock_trt()
        with patch("src.pipeline.trt_engine.trt", mock_trt), \
             patch("src.pipeline.trt_engine.torch", mock_torch):
            from src.pipeline.trt_engine import is_available
            assert is_available() is False


# ===========================================================================
# TestGetGpuArch
# ===========================================================================

class TestGetGpuArch:

    def test_returns_correct_arch(self):
        mock_torch = _make_mock_torch(major=12, minor=0)
        with patch("src.pipeline.trt_engine.torch", mock_torch):
            from src.pipeline.trt_engine import get_gpu_arch
            assert get_gpu_arch() == "sm_120"

    def test_ampere_arch(self):
        mock_torch = _make_mock_torch(major=8, minor=6)
        with patch("src.pipeline.trt_engine.torch", mock_torch):
            from src.pipeline.trt_engine import get_gpu_arch
            assert get_gpu_arch() == "sm_86"


# ===========================================================================
# TestGetEnginePath
# ===========================================================================

class TestGetEnginePath:

    def test_path_includes_arch_and_model(self):
        mock_torch = _make_mock_torch(major=12, minor=0)
        with patch("src.pipeline.trt_engine.torch", mock_torch), \
             patch("src.pipeline.trt_engine.TRT_ENGINE_DIR", Path("/cache/engines")):
            from src.pipeline.trt_engine import get_engine_path
            p = get_engine_path("RealESRGAN_x2plus", scale=2, tile_size=1024, tile_pad=10)
            assert "sm_120" in p.name
            assert "RealESRGAN_x2plus" in p.name
            assert "1044x1044" in p.name
            assert p.parent == Path("/cache/engines")

    def test_different_tile_size(self):
        mock_torch = _make_mock_torch(major=8, minor=9)
        with patch("src.pipeline.trt_engine.torch", mock_torch), \
             patch("src.pipeline.trt_engine.TRT_ENGINE_DIR", Path("/cache/engines")):
            from src.pipeline.trt_engine import get_engine_path
            p = get_engine_path("RealESRGAN_x4plus", scale=4, tile_size=512, tile_pad=10)
            assert "532x532" in p.name
            assert "sm_89" in p.name


# ===========================================================================
# TestExportOnnx
# ===========================================================================

class TestExportOnnx:

    def test_skips_when_exists(self, tmp_path):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_text("dummy")

        with patch("src.pipeline.trt_engine._get_onnx_path", return_value=onnx_file):
            from src.pipeline.trt_engine import export_onnx
            result = export_onnx("RealESRGAN_x2plus", scale=2)
            assert result == onnx_file

    def test_exports_when_missing(self, tmp_path):
        onnx_file = tmp_path / "onnx" / "model.onnx"

        mock_torch = _make_mock_torch()
        mock_net = MagicMock()
        mock_state = {"params_ema": {}}

        with patch("src.pipeline.trt_engine._get_onnx_path", return_value=onnx_file), \
             patch("src.pipeline.trt_engine._build_rrdbnet", return_value=mock_net), \
             patch("src.pipeline.trt_engine.torch", mock_torch), \
             patch("src.pipeline.trt_engine.AVAILABLE_MODELS", {"RealESRGAN_x2plus": {"file": "RealESRGAN_x2plus.pth", "scale": 2}}), \
             patch("src.pipeline.trt_engine.MODEL_DIR", tmp_path):
            mock_torch.load.return_value = mock_state
            mock_torch.randn.return_value = MagicMock()

            from src.pipeline.trt_engine import export_onnx
            result = export_onnx("RealESRGAN_x2plus", scale=2)

            assert result == onnx_file
            mock_torch.onnx.export.assert_called_once()
            mock_net.load_state_dict.assert_called_once()
            mock_net.eval.assert_called_once()


# ===========================================================================
# TestBuildEngine
# ===========================================================================

class TestBuildEngine:

    def test_skips_when_cached(self, tmp_path):
        engine_file = tmp_path / "engine.engine"
        engine_file.write_text("cached")

        from src.pipeline.trt_engine import build_engine
        result = build_engine(Path("dummy.onnx"), engine_file)
        assert result == engine_file

    def test_builds_engine(self, tmp_path):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"onnx_data")
        engine_file = tmp_path / "engines" / "model.engine"

        mock_trt = _make_mock_trt()
        mock_builder = MagicMock()
        mock_network = MagicMock()
        mock_parser = MagicMock()
        mock_config = MagicMock()
        mock_serialized = b"engine_bytes"

        mock_trt.Builder.return_value = mock_builder
        mock_builder.create_network.return_value = mock_network
        mock_trt.OnnxParser.return_value = mock_parser
        mock_parser.parse.return_value = True
        mock_builder.create_builder_config.return_value = mock_config
        mock_builder.build_serialized_network.return_value = mock_serialized

        with patch("src.pipeline.trt_engine.trt", mock_trt):
            from src.pipeline.trt_engine import build_engine
            result = build_engine(onnx_file, engine_file)

        assert result == engine_file
        assert engine_file.exists()
        assert engine_file.read_bytes() == b"engine_bytes"
        mock_parser.parse.assert_called_once()
        mock_config.set_flag.assert_any_call(mock_trt.BuilderFlag.FP16)
        mock_config.set_flag.assert_any_call(mock_trt.BuilderFlag.BF16)

    def test_raises_on_parse_failure(self, tmp_path):
        onnx_file = tmp_path / "bad.onnx"
        onnx_file.write_bytes(b"bad_data")
        engine_file = tmp_path / "engine.engine"

        mock_trt = _make_mock_trt()
        mock_parser = MagicMock()
        mock_parser.parse.return_value = False
        mock_parser.num_errors = 1
        mock_parser.get_error.return_value = "parse error"
        mock_trt.OnnxParser.return_value = mock_parser
        mock_trt.Builder.return_value.create_network.return_value = MagicMock()

        with patch("src.pipeline.trt_engine.trt", mock_trt), \
             pytest.raises(RuntimeError, match="ONNX parse failed"):
            from src.pipeline.trt_engine import build_engine
            build_engine(onnx_file, engine_file)

    def test_raises_on_build_failure(self, tmp_path):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"onnx_data")
        engine_file = tmp_path / "engine.engine"

        mock_trt = _make_mock_trt()
        mock_builder = MagicMock()
        mock_network = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse.return_value = True

        mock_trt.Builder.return_value = mock_builder
        mock_builder.create_network.return_value = mock_network
        mock_trt.OnnxParser.return_value = mock_parser
        mock_builder.create_builder_config.return_value = MagicMock()
        mock_builder.build_serialized_network.return_value = None

        with patch("src.pipeline.trt_engine.trt", mock_trt), \
             pytest.raises(RuntimeError, match="build failed"):
            from src.pipeline.trt_engine import build_engine
            build_engine(onnx_file, engine_file)


# ===========================================================================
# TestTRTUpscalerEnhance
# ===========================================================================

class TestTRTUpscalerEnhance:
    """Test TRTUpscaler.enhance() tiling logic with mocked _infer_tile."""

    def _make_upscaler(self, scale=2, tile_size=64, tile_pad=4):
        """Create a TRTUpscaler with all heavy init mocked out."""
        mock_torch = _make_mock_torch()
        mock_trt = _make_mock_trt()

        # Mock the engine/context
        mock_runtime = MagicMock()
        mock_engine = MagicMock()
        mock_context = MagicMock()
        mock_trt.Runtime.return_value = mock_runtime
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine
        mock_engine.create_execution_context.return_value = mock_context

        # Mock torch tensor buffers
        mock_torch.empty.return_value = MagicMock()

        with patch("src.pipeline.trt_engine.trt", mock_trt), \
             patch("src.pipeline.trt_engine.torch", mock_torch), \
             patch("src.pipeline.trt_engine.export_onnx", return_value=Path("/tmp/m.onnx")), \
             patch("src.pipeline.trt_engine.get_engine_path", return_value=Path("/tmp/m.engine")), \
             patch("src.pipeline.trt_engine.build_engine", return_value=Path("/tmp/m.engine")), \
             patch("builtins.open", mock_open(read_data=b"engine")), \
             patch("src.pipeline.trt_engine.AVAILABLE_MODELS", {
                 "RealESRGAN_x2plus": {"file": "RealESRGAN_x2plus.pth", "scale": 2},
             }):
            from src.pipeline.trt_engine import TRTUpscaler
            upscaler = TRTUpscaler(
                "RealESRGAN_x2plus", scale=scale,
                tile_size=tile_size, tile_pad=tile_pad,
                device="cuda:0",
            )

        return upscaler

    def _mock_infer_tile(self, upscaler):
        """Replace _infer_tile with a simple scale-up function for testing."""
        scale = upscaler.scale

        def fake_infer(tile_chw):
            c, h, w = tile_chw.shape
            # Fill with a pattern so we can verify tiling works
            out = np.zeros((c, h * scale, w * scale), dtype=np.float32)
            # Simple nearest-neighbor-like upscale for content
            for y in range(h):
                for x in range(w):
                    out[:, y*scale:(y+1)*scale, x*scale:(x+1)*scale] = tile_chw[:, y, x].reshape(c, 1, 1)
            return out

        upscaler._infer_tile = fake_infer

    def test_output_shape_small_image(self):
        """Image smaller than tile_size → single tile."""
        upscaler = self._make_upscaler(scale=2, tile_size=64, tile_pad=4)
        self._mock_infer_tile(upscaler)

        img = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
        out, _ = upscaler.enhance(img)
        assert out.shape == (64, 96, 3)
        assert out.dtype == np.uint8

    def test_output_shape_exact_tile(self):
        """Image exactly tile_size → single tile."""
        upscaler = self._make_upscaler(scale=2, tile_size=64, tile_pad=4)
        self._mock_infer_tile(upscaler)

        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out, _ = upscaler.enhance(img)
        assert out.shape == (128, 128, 3)

    def test_output_shape_multi_tile(self):
        """Image requiring multiple tiles."""
        upscaler = self._make_upscaler(scale=2, tile_size=32, tile_pad=4)
        self._mock_infer_tile(upscaler)

        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        out, _ = upscaler.enhance(img)
        assert out.shape == (200, 160, 3)

    def test_non_divisible_dimensions(self):
        """Image dimensions not divisible by tile_size (edge tiles)."""
        upscaler = self._make_upscaler(scale=2, tile_size=64, tile_pad=4)
        self._mock_infer_tile(upscaler)

        img = np.random.randint(0, 255, (100, 70, 3), dtype=np.uint8)
        out, _ = upscaler.enhance(img)
        assert out.shape == (200, 140, 3)

    def test_vr_eye_size(self):
        """Typical VR SBS eye size (scaled down for test speed)."""
        upscaler = self._make_upscaler(scale=2, tile_size=128, tile_pad=4)
        self._mock_infer_tile(upscaler)

        # Use 270x270 as a scaled proxy for 2700x2700
        img = np.random.randint(0, 255, (270, 270, 3), dtype=np.uint8)
        out, _ = upscaler.enhance(img)
        assert out.shape == (540, 540, 3)

    def test_outscale_different_from_model_scale(self):
        """outscale != model scale → resize after upscaling."""
        upscaler = self._make_upscaler(scale=2, tile_size=64, tile_pad=4)
        self._mock_infer_tile(upscaler)

        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out, _ = upscaler.enhance(img, outscale=4)
        assert out.shape == (256, 256, 3)

    def test_returns_bgr_uint8(self):
        """Output must be BGR uint8."""
        upscaler = self._make_upscaler(scale=2, tile_size=64, tile_pad=4)
        self._mock_infer_tile(upscaler)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out, extra = upscaler.enhance(img)
        assert out.dtype == np.uint8
        assert extra is None

    def test_content_not_all_zero(self):
        """Non-black input → non-black output."""
        upscaler = self._make_upscaler(scale=2, tile_size=64, tile_pad=4)
        self._mock_infer_tile(upscaler)

        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        out, _ = upscaler.enhance(img)
        assert out.mean() > 0

    def test_unknown_model_raises(self):
        """Unknown model name raises ValueError."""
        mock_trt = _make_mock_trt()
        mock_torch = _make_mock_torch()

        with patch("src.pipeline.trt_engine.trt", mock_trt), \
             patch("src.pipeline.trt_engine.torch", mock_torch), \
             patch("src.pipeline.trt_engine.AVAILABLE_MODELS", {}), \
             pytest.raises(ValueError, match="Unknown model"):
            from src.pipeline.trt_engine import TRTUpscaler
            TRTUpscaler("NonExistentModel", scale=2)
