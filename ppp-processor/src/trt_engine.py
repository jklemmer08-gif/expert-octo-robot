"""TensorRT inference engine for RobustVideoMatting.

Handles ONNX export, TRT engine compilation with optimization profiles
(resolution buckets), cached engine loading, and recurrent state management.

Requires: tensorrt, torch, onnxruntime (optional for validation).
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import Settings

logger = logging.getLogger("ppp.trt_engine")

# Lazy imports — these are heavy and may not be installed
_trt = None
_torch = None
_np = None
_onnx = None


def _import_trt():
    global _trt
    if _trt is None:
        import tensorrt as _trt
    return _trt


def _import_torch():
    global _torch
    if _torch is None:
        import torch as _torch
    return _torch


def _import_numpy():
    global _np
    if _np is None:
        import numpy as _np
    return _np


class ResolutionBucket:
    """A resolution bucket for TRT optimization profiles."""

    def __init__(self, name: str, min_h: int, min_w: int,
                 opt_h: int, opt_w: int, max_h: int, max_w: int):
        self.name = name
        self.min_h = min_h
        self.min_w = min_w
        self.opt_h = opt_h
        self.opt_w = opt_w
        self.max_h = max_h
        self.max_w = max_w

    def matches(self, h: int, w: int) -> bool:
        """Check if a resolution fits within this bucket's range."""
        return (self.min_h <= h <= self.max_h and
                self.min_w <= w <= self.max_w)


class RVMTensorRTEngine:
    """TensorRT engine for RobustVideoMatting with recurrent state.

    RVM forward signature:
        (src, r1, r2, r3, r4, downsample_ratio) -> (fgr, pha, r1o, r2o, r3o, r4o)

    Recurrent states r1-r4 are initialized as [1,1,1,1] tensors (zeros)
    and grow to resolution-dependent shapes after the first frame.
    Within a single video, shapes are fixed after frame 1.
    """

    ONNX_FILENAME = "rvm_{model_type}.onnx"
    ENGINE_FILENAME = "rvm_{model_type}_{bucket}.engine"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.trt_config = settings.tensorrt
        self.cache_dir = Path(self.trt_config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.buckets = [
            ResolutionBucket(**b) for b in self.trt_config.resolution_buckets
        ]

        self._engine = None
        self._context = None
        self._current_bucket = None
        self._rec_states: List[Optional[object]] = [None] * 4
        self._stream = None
        self._bindings: Dict[str, object] = {}

    def export_onnx(
        self,
        pytorch_model,
        model_type: str,
        device,
        downsample_ratio: float = 0.25,
    ) -> Path:
        """Export PyTorch RVM model to ONNX format with dynamic axes.

        Only needs to run once per model type.
        """
        torch = _import_torch()
        onnx_path = self.cache_dir / self.ONNX_FILENAME.format(model_type=model_type)

        if onnx_path.exists():
            logger.info("ONNX model already exists: %s", onnx_path)
            return onnx_path

        logger.info("Exporting RVM to ONNX: %s", onnx_path)

        pytorch_model.eval()
        # Use float32 for ONNX export regardless of runtime precision
        model_f32 = pytorch_model.float()

        # Dummy inputs for tracing
        dummy_src = torch.randn(1, 3, 1080, 1920, device=device, dtype=torch.float32)
        dummy_r1 = torch.zeros(1, 1, 1, 1, device=device, dtype=torch.float32)
        dummy_r2 = torch.zeros(1, 1, 1, 1, device=device, dtype=torch.float32)
        dummy_r3 = torch.zeros(1, 1, 1, 1, device=device, dtype=torch.float32)
        dummy_r4 = torch.zeros(1, 1, 1, 1, device=device, dtype=torch.float32)
        dummy_dr = torch.tensor([downsample_ratio], device=device, dtype=torch.float32)

        input_names = ["src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio"]
        output_names = ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"]

        # Dynamic axes for spatial dimensions and recurrent state
        dynamic_axes = {
            "src": {0: "batch", 2: "height", 3: "width"},
            "r1i": {0: "batch", 1: "r1c", 2: "r1h", 3: "r1w"},
            "r2i": {0: "batch", 1: "r2c", 2: "r2h", 3: "r2w"},
            "r3i": {0: "batch", 1: "r3c", 2: "r3h", 3: "r3w"},
            "r4i": {0: "batch", 1: "r4c", 2: "r4h", 3: "r4w"},
            "fgr": {0: "batch", 2: "height", 3: "width"},
            "pha": {0: "batch", 2: "height", 3: "width"},
            "r1o": {0: "batch", 1: "r1c", 2: "r1h", 3: "r1w"},
            "r2o": {0: "batch", 1: "r2c", 2: "r2h", 3: "r2w"},
            "r3o": {0: "batch", 1: "r3c", 2: "r3h", 3: "r3w"},
            "r4o": {0: "batch", 1: "r4c", 2: "r4h", 3: "r4w"},
        }

        torch.onnx.export(
            model_f32,
            (dummy_src, dummy_r1, dummy_r2, dummy_r3, dummy_r4, dummy_dr),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )

        logger.info("ONNX export complete: %s (%.1f MB)",
                     onnx_path.name, onnx_path.stat().st_size / 1024 / 1024)
        return onnx_path

    def build_engine(self, onnx_path: Path, bucket: ResolutionBucket,
                     model_type: str) -> Path:
        """Build a TRT engine for a specific resolution bucket.

        Uses optimization profiles with min/opt/max shapes.
        Caches to disk for fast reload (~2s vs ~30s compile).
        """
        trt = _import_trt()
        torch = _import_torch()

        engine_path = self.cache_dir / self.ENGINE_FILENAME.format(
            model_type=model_type, bucket=bucket.name,
        )

        if engine_path.exists():
            logger.info("TRT engine cached: %s", engine_path)
            return engine_path

        logger.info("Building TRT engine for bucket '%s' (%dx%d opt)...",
                     bucket.name, bucket.opt_w, bucket.opt_h)
        start = time.time()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("ONNX parse error: %s", parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.trt_config.max_workspace_mb * 1024 * 1024,
        )

        if self.trt_config.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("TRT FP16 enabled")

        # Optimization profile for this bucket
        profile = builder.create_optimization_profile()

        # src input: [1, 3, H, W]
        profile.set_shape("src",
                          (1, 3, bucket.min_h, bucket.min_w),
                          (1, 3, bucket.opt_h, bucket.opt_w),
                          (1, 3, bucket.max_h, bucket.max_w))

        # Recurrent state inputs: dynamic shapes
        # First frame: [1,1,1,1], subsequent: resolution-dependent
        # We use generous ranges to accommodate both
        for ri in ["r1i", "r2i", "r3i", "r4i"]:
            profile.set_shape(ri,
                              (1, 1, 1, 1),
                              (1, 64, bucket.opt_h // 4, bucket.opt_w // 4),
                              (1, 128, bucket.max_h // 2, bucket.max_w // 2))

        # downsample_ratio: scalar tensor
        profile.set_shape("downsample_ratio", (1,), (1,), (1,))

        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine build failed")

        with open(engine_path, "wb") as f:
            f.write(serialized)

        elapsed = time.time() - start
        logger.info("TRT engine built in %.1fs: %s (%.1f MB)",
                     elapsed, engine_path.name,
                     engine_path.stat().st_size / 1024 / 1024)
        return engine_path

    def load_engine(self, engine_path: Path):
        """Load a compiled TRT engine from disk."""
        trt = _import_trt()
        torch = _import_torch()

        logger.info("Loading TRT engine: %s", engine_path.name)
        start = time.time()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")

        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream()

        elapsed = time.time() - start
        logger.info("TRT engine loaded in %.2fs", elapsed)

    def select_bucket(self, height: int, width: int) -> Optional[ResolutionBucket]:
        """Pick the correct resolution bucket for given dimensions."""
        for bucket in self.buckets:
            if bucket.matches(height, width):
                return bucket

        # If no exact match, find the closest bucket by opt resolution
        best = None
        best_dist = float("inf")
        for bucket in self.buckets:
            dist = abs(bucket.opt_h - height) + abs(bucket.opt_w - width)
            if dist < best_dist:
                best_dist = dist
                best = bucket

        if best:
            logger.warning(
                "No exact bucket for %dx%d, using closest: %s (%dx%d)",
                width, height, best.name, best.opt_w, best.opt_h,
            )
        return best

    def prepare(self, height: int, width: int, pytorch_model,
                model_type: str, device) -> bool:
        """Prepare engine for a specific resolution: export ONNX, build/load engine.

        Returns True if TRT is ready, False if it should fall back to PyTorch.
        """
        try:
            bucket = self.select_bucket(height, width)
            if bucket is None:
                logger.warning("No suitable TRT bucket for %dx%d", width, height)
                return False

            if self._current_bucket and self._current_bucket.name == bucket.name:
                logger.info("TRT engine already loaded for bucket '%s'", bucket.name)
                return True

            # Export ONNX if needed
            onnx_path = self.export_onnx(pytorch_model, model_type, device)

            # Build or load engine
            engine_path = self.build_engine(onnx_path, bucket, model_type)
            self.load_engine(engine_path)

            self._current_bucket = bucket
            self.reset_recurrent_state()
            return True

        except Exception as e:
            logger.error("TRT preparation failed, falling back to PyTorch: %s", e)
            return False

    def reset_recurrent_state(self):
        """Clear recurrent state — needed for VR eye switch or new video."""
        torch = _import_torch()
        self._rec_states = [
            torch.zeros(1, 1, 1, 1, dtype=torch.float16 if self.trt_config.fp16 else torch.float32,
                        device="cuda")
            for _ in range(4)
        ]

    def infer(self, src_tensor, downsample_ratio: float = 0.25):
        """Run TRT inference with recurrent state management.

        Args:
            src_tensor: [1, 3, H, W] input tensor on CUDA
            downsample_ratio: RVM downsample ratio

        Returns:
            (fgr, pha, r1, r2, r3, r4) tensors on CUDA
        """
        trt = _import_trt()
        torch = _import_torch()
        np = _import_numpy()

        if self._context is None:
            raise RuntimeError("TRT engine not loaded — call prepare() first")

        dtype = torch.float16 if self.trt_config.fp16 else torch.float32
        src = src_tensor.to(dtype)
        _, _, h, w = src.shape

        # Set input shapes for this frame
        self._context.set_input_shape("src", src.shape)
        for i, ri in enumerate(["r1i", "r2i", "r3i", "r4i"]):
            self._context.set_input_shape(ri, tuple(self._rec_states[i].shape))
        self._context.set_input_shape("downsample_ratio", (1,))

        # Prepare downsample_ratio tensor
        dr_tensor = torch.tensor([downsample_ratio], dtype=dtype, device="cuda")

        # Allocate output buffers
        # fgr and pha have same spatial dims as src
        fgr = torch.empty(1, 3, h, w, dtype=dtype, device="cuda")
        pha = torch.empty(1, 1, h, w, dtype=dtype, device="cuda")

        # Recurrent output shapes match input shapes (or grow on first frame)
        rec_out = [torch.empty_like(r) for r in self._rec_states]

        # Bind inputs
        self._context.set_tensor_address("src", src.data_ptr())
        for i, ri in enumerate(["r1i", "r2i", "r3i", "r4i"]):
            self._context.set_tensor_address(ri, self._rec_states[i].data_ptr())
        self._context.set_tensor_address("downsample_ratio", dr_tensor.data_ptr())

        # Bind outputs
        self._context.set_tensor_address("fgr", fgr.data_ptr())
        self._context.set_tensor_address("pha", pha.data_ptr())
        for i, ro in enumerate(["r1o", "r2o", "r3o", "r4o"]):
            self._context.set_tensor_address(ro, rec_out[i].data_ptr())

        # Execute
        with torch.cuda.stream(self._stream):
            self._context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        # Get actual output shapes and re-read recurrent states
        for i, ro in enumerate(["r1o", "r2o", "r3o", "r4o"]):
            out_shape = self._context.get_tensor_shape(ro)
            if tuple(out_shape) != tuple(rec_out[i].shape):
                # Shape changed (first frame) — reallocate
                rec_out[i] = torch.empty(tuple(out_shape), dtype=dtype, device="cuda")
                self._context.set_tensor_address(ro, rec_out[i].data_ptr())
                # Re-execute to get correct data (only on first frame)
                with torch.cuda.stream(self._stream):
                    self._context.execute_async_v3(self._stream.cuda_stream)
                self._stream.synchronize()
                break

        # Update recurrent state for next frame
        self._rec_states = rec_out

        return fgr, pha, *rec_out

    def cleanup(self):
        """Release TRT resources."""
        self._context = None
        self._engine = None
        self._stream = None
        self._rec_states = [None] * 4
        self._current_bucket = None
