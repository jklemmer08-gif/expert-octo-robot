"""TensorRT acceleration for Real-ESRGAN upscaling.

Exports RRDBNet to ONNX, builds a TensorRT engine (bf16+fp16), caches it
per GPU architecture on the network volume, and provides TRTUpscaler as a
drop-in replacement for RealESRGANer.

Requires: tensorrt>=10.0, onnx>=1.14, torch with CUDA.
Falls back gracefully when TRT is unavailable.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorrt as trt
except ImportError:
    trt = None

from src.config import (
    AVAILABLE_MODELS,
    MODEL_DIR,
    TRT_ENGINE_DIR,
    TRT_ONNX_DIR,
    TRT_TILE_PAD,
    TRT_TILE_SIZE,
)

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if TensorRT acceleration is usable (tensorrt + CUDA)."""
    if trt is None or torch is None:
        return False
    if not torch.cuda.is_available():
        return False
    return True


def get_gpu_arch() -> str:
    """Return GPU SM architecture string, e.g. 'sm_120'."""
    props = torch.cuda.get_device_properties(0)
    return f"sm_{props.major}{props.minor}"


def get_engine_path(
    model_name: str,
    scale: int,
    tile_size: int = TRT_TILE_SIZE,
    tile_pad: int = TRT_TILE_PAD,
) -> Path:
    """Deterministic engine path keyed by GPU arch, model, and tile geometry."""
    arch = get_gpu_arch()
    # Engine input size = tile_size + 2*tile_pad
    engine_h = tile_size + 2 * tile_pad
    name = f"{model_name}_{arch}_bf16fp16_{engine_h}x{engine_h}.engine"
    return TRT_ENGINE_DIR / name


def _get_onnx_path(model_name: str, scale: int, tile_size: int, tile_pad: int) -> Path:
    """Deterministic ONNX path for export."""
    engine_h = tile_size + 2 * tile_pad
    return TRT_ONNX_DIR / f"{model_name}_{engine_h}x{engine_h}.onnx"


def _build_rrdbnet(model_name: str, scale: int):
    """Instantiate the RRDBNet architecture for a given model."""
    from basicsr.archs.rrdbnet_arch import RRDBNet

    if "anime" in model_name.lower():
        return RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=6, num_grow_ch=32, scale=scale,
        )
    return RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=scale,
    )


def export_onnx(
    model_name: str,
    scale: int,
    tile_size: int = TRT_TILE_SIZE,
    tile_pad: int = TRT_TILE_PAD,
) -> Path:
    """Export RRDBNet to ONNX. Skips if file already exists. Returns ONNX path."""
    onnx_path = _get_onnx_path(model_name, scale, tile_size, tile_pad)
    if onnx_path.exists():
        logger.info("ONNX already exists: %s", onnx_path)
        return onnx_path

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model_info = AVAILABLE_MODELS[model_name]
    weights_path = MODEL_DIR / model_info["file"]

    net = _build_rrdbnet(model_name, scale)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    net.load_state_dict(state, strict=True)
    net.eval()

    engine_h = tile_size + 2 * tile_pad
    dummy = torch.randn(1, 3, engine_h, engine_h)

    logger.info("Exporting ONNX: %s (%dx%d input)", model_name, engine_h, engine_h)
    torch.onnx.export(
        net, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,  # fixed size for best TRT optimization
    )
    logger.info("ONNX exported: %s", onnx_path)
    return onnx_path


def build_engine(onnx_path: Path, engine_path: Path) -> Path:
    """Build a TensorRT engine from ONNX. Skips if engine already exists."""
    if engine_path.exists():
        logger.info("TRT engine already cached: %s", engine_path)
        return engine_path

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building TRT engine (this may take 10-30 min)...")
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (returned None)")

    with open(str(engine_path), "wb") as f:
        f.write(serialized)

    logger.info("TRT engine saved: %s (%.1f MB)", engine_path, engine_path.stat().st_size / 1e6)
    return engine_path


class TRTUpscaler:
    """Drop-in replacement for RealESRGANer using a TensorRT engine.

    Matches the `enhance(img_bgr, outscale=None) -> (output_bgr, None)` API
    so existing upscale_frame() calls work unchanged.
    """

    def __init__(
        self,
        model_name: str,
        scale: int,
        tile_size: int = TRT_TILE_SIZE,
        tile_pad: int = TRT_TILE_PAD,
        device: str = "cuda:0",
    ):
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.device = device

        model_info = AVAILABLE_MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")

        # Engine input/output spatial size
        self.engine_h = tile_size + 2 * tile_pad
        self.engine_w = self.engine_h
        self.out_engine_h = self.engine_h * scale
        self.out_engine_w = self.engine_w * scale

        # Export ONNX + build engine if needed
        onnx_path = export_onnx(model_name, scale, tile_size, tile_pad)
        engine_path = get_engine_path(model_name, scale, tile_size, tile_pad)
        build_engine(onnx_path, engine_path)

        # Load engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(str(engine_path), "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Pre-allocate GPU buffers
        self.input_buf = torch.empty(
            (1, 3, self.engine_h, self.engine_w),
            dtype=torch.float16, device=self.device,
        )
        self.output_buf = torch.empty(
            (1, 3, self.out_engine_h, self.out_engine_w),
            dtype=torch.float16, device=self.device,
        )

        # Bind buffers to context
        self.context.set_tensor_address("input", self.input_buf.data_ptr())
        self.context.set_tensor_address("output", self.output_buf.data_ptr())

        logger.info(
            "TRTUpscaler ready: %s, tile=%d, pad=%d, engine=%dx%d",
            model_name, tile_size, tile_pad, self.engine_h, self.engine_h,
        )

    def _infer_tile(self, tile_chw: np.ndarray) -> np.ndarray:
        """Run a single CHW float32 tile through TRT. Returns CHW float32 output."""
        h, w = tile_chw.shape[1], tile_chw.shape[2]

        # Zero-pad to engine size if tile is smaller (edge tiles)
        padded = np.zeros((3, self.engine_h, self.engine_w), dtype=np.float32)
        padded[:, :h, :w] = tile_chw

        # Copy to GPU buffer (fp16)
        self.input_buf.copy_(
            torch.from_numpy(padded).unsqueeze(0).half().to(self.device)
        )

        # Execute
        self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream)
        torch.cuda.current_stream(self.device).synchronize()

        # Copy back, crop to actual output size
        out = self.output_buf[0].float().cpu().numpy()
        out_h = h * self.scale
        out_w = w * self.scale
        return out[:, :out_h, :out_w]

    def enhance(self, img: np.ndarray, outscale: Optional[int] = None) -> Tuple[np.ndarray, None]:
        """Upscale a BGR uint8 image using tiled TRT inference.

        Matches RealESRGANer.enhance() signature: returns (output_bgr, None).
        """
        if outscale is None:
            outscale = self.scale

        # Pre-process: BGR uint8 → RGB float32 [0,1] → CHW
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_f = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_f, (2, 0, 1))  # (3, H, W)

        h, w = img_chw.shape[1], img_chw.shape[2]
        out_h = h * self.scale
        out_w = w * self.scale

        # Reflection-pad by tile_pad on all sides
        img_padded = np.pad(
            img_chw,
            ((0, 0), (self.tile_pad, self.tile_pad), (self.tile_pad, self.tile_pad)),
            mode="reflect",
        )
        ph, pw = img_padded.shape[1], img_padded.shape[2]

        # Output accumulator
        output = np.zeros((3, out_h, out_w), dtype=np.float32)

        # Tile grid: stride = tile_size, extract tiles of (tile_size + 2*tile_pad)
        tile_stride = self.tile_size
        pad = self.tile_pad
        scale = self.scale

        tiles_y = list(range(0, h, tile_stride))
        tiles_x = list(range(0, w, tile_stride))

        for iy in tiles_y:
            for ix in tiles_x:
                # Input tile region in padded image (includes padding border)
                in_y_start = iy  # iy is already offset 0 in original, +pad in padded
                in_x_start = ix
                in_y_end = min(iy + tile_stride + 2 * pad, ph)
                in_x_end = min(ix + tile_stride + 2 * pad, pw)

                tile = img_padded[:, in_y_start:in_y_end, in_x_start:in_x_end]
                tile_h, tile_w = tile.shape[1], tile.shape[2]

                # Infer
                out_tile = self._infer_tile(tile)

                # Crop padding from output tile
                out_tile_h = out_tile.shape[1]
                out_tile_w = out_tile.shape[2]

                # The padding border in the output
                crop_top = pad * scale
                crop_left = pad * scale
                crop_bottom = out_tile_h - pad * scale
                crop_right = out_tile_w - pad * scale

                # Handle edge tiles: don't crop past tile boundary
                # Actual content pixels in this tile
                content_h = min(tile_stride, h - iy)
                content_w = min(tile_stride, w - ix)

                crop_bottom = min(crop_top + content_h * scale, out_tile_h)
                crop_right = min(crop_left + content_w * scale, out_tile_w)

                cropped = out_tile[:, crop_top:crop_bottom, crop_left:crop_right]

                # Place in output
                out_iy = iy * scale
                out_ix = ix * scale
                ch = cropped.shape[1]
                cw = cropped.shape[2]
                output[:, out_iy:out_iy + ch, out_ix:out_ix + cw] = cropped

        # Post-process: clamp, scale, CHW→HWC, RGB→BGR
        output = np.clip(output, 0.0, 1.0)
        output = (output * 255.0).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))  # HWC
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Handle outscale != model scale
        if outscale != self.scale:
            final_h = int(h * outscale)
            final_w = int(w * outscale)
            output_bgr = cv2.resize(output_bgr, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)

        return output_bgr, None
