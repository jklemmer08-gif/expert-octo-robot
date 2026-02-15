"""OpenVINO inference engine for RobustVideoMatting.

Drop-in replacement for ort_engine.py on Intel platforms.
Loads ONNX model directly via OpenVINO (no PyTorch required).
Targets Intel Arc GPUs via Level Zero backend.

Same interface: prepare(), infer(), reset_recurrent_state(), cleanup().

Requires: openvino>=2024.6.0, numpy
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("ppp.openvino_engine")

# Lazy import
_ov = None


def _import_ov():
    global _ov
    if _ov is None:
        import openvino as _ov
    return _ov


class RVMOpenVINOEngine:
    """OpenVINO engine for RobustVideoMatting with recurrent state.

    RVM forward signature:
        (src, r1, r2, r3, r4) -> (fgr, pha, r1o, r2o, r3o, r4o)

    Recurrent states r1-r4 are initialized as zero tensors on first frame,
    then fed back as inputs each subsequent frame. OpenVINO handles FP16
    optimization internally when targeting GPU.
    """

    # RVM resnet50 recurrent state metadata:
    # channels per decoder stage, and stride divisors from original resolution
    REC_CHANNELS = [16, 32, 64, 128]    # r1, r2, r3, r4
    REC_STRIDES = [8, 16, 32, 64]       # spatial dims = ceil(H/stride), ceil(W/stride)

    def __init__(self):
        self._compiled_model = None
        self._infer_request = None
        self._rec_states = [None] * 4  # NumPy arrays
        self._rec_shapes = None  # Computed from input resolution
        self._input_height = 0
        self._input_width = 0
        self._device = "GPU"
        self._model_path: Optional[Path] = None

    def prepare(
        self,
        height: int,
        width: int,
        onnx_path: Path,
        device: str = "GPU",
    ) -> bool:
        """Load ONNX model and compile for target device.

        Args:
            height: Input frame height
            width: Input frame width
            onnx_path: Path to rvm_resnet50.onnx (FP32, opset 17)
            device: OpenVINO device string ("GPU", "CPU", etc.)

        Returns:
            True if engine is ready, False on failure.
        """
        try:
            ov = _import_ov()
            self._device = device

            if not onnx_path.exists():
                logger.error("ONNX model not found: %s", onnx_path)
                return False

            logger.info("Loading ONNX model: %s", onnx_path)
            core = ov.Core()

            # Check device availability
            available = core.available_devices
            logger.info("OpenVINO available devices: %s", available)
            if device not in available:
                logger.warning(
                    "Device %s not available (have: %s), falling back to CPU",
                    device, available,
                )
                self._device = "CPU"

            # Read and compile model
            start = time.time()
            model = core.read_model(str(onnx_path))
            logger.info("Model read in %.2fs, compiling for %s...", time.time() - start, self._device)

            # Set performance hints for throughput
            config = {}
            if self._device == "GPU":
                config["PERFORMANCE_HINT"] = "LATENCY"

            start = time.time()
            self._compiled_model = core.compile_model(model, self._device, config)
            logger.info("Model compiled in %.2fs", time.time() - start)

            # Create persistent infer request for reuse
            self._infer_request = self._compiled_model.create_infer_request()

            # Compute correct recurrent state shapes from input resolution.
            # RVM's decoder stages produce rec states at specific spatial scales
            # relative to the original input (not the internally-downsampled input).
            self._input_height = height
            self._input_width = width
            self._rec_shapes = self._compute_rec_shapes(height, width)
            logger.info(
                "Rec shapes for %dx%d: %s",
                width, height, [list(s) for s in self._rec_shapes],
            )

            # Initialize with correct shapes
            self._rec_states = [
                np.zeros(s, dtype=np.float32) for s in self._rec_shapes
            ]

            # Warm-up inference to trigger JIT compilation
            logger.info("Running warm-up inference at %dx%d...", width, height)
            dummy = np.random.rand(1, 3, height, width).astype(np.float32)
            start = time.time()
            self._run_inference(dummy)
            logger.info("Warm-up complete in %.2fs", time.time() - start)

            # Reset state after warm-up (don't carry dummy state into real video)
            self.reset_recurrent_state()

            self._model_path = onnx_path
            logger.info("OpenVINO engine ready for %dx%d on %s", width, height, self._device)
            return True

        except Exception as e:
            logger.error("OpenVINO preparation failed: %s", e)
            import traceback
            logger.debug(traceback.format_exc())
            return False

    @classmethod
    def _compute_rec_shapes(cls, height: int, width: int) -> list:
        """Compute recurrent state shapes from input resolution.

        RVM's decoder stages produce states at spatial scales relative to the
        original input: r1=/8, r2=/16, r3=/32, r4=/64 with ceil rounding.
        """
        return [
            (1, ch, math.ceil(height / stride), math.ceil(width / stride))
            for ch, stride in zip(cls.REC_CHANNELS, cls.REC_STRIDES)
        ]

    def _run_inference(self, src: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Run a single inference pass, returning all 6 outputs."""
        input_dict = {
            "src": src,
            "r1i": self._rec_states[0],
            "r2i": self._rec_states[1],
            "r3i": self._rec_states[2],
            "r4i": self._rec_states[3],
        }

        self._infer_request.infer(input_dict)

        # Extract outputs
        fgr = self._infer_request.get_output_tensor(0).data.copy()
        pha = self._infer_request.get_output_tensor(1).data.copy()
        r1o = self._infer_request.get_output_tensor(2).data.copy()
        r2o = self._infer_request.get_output_tensor(3).data.copy()
        r3o = self._infer_request.get_output_tensor(4).data.copy()
        r4o = self._infer_request.get_output_tensor(5).data.copy()

        # Update recurrent states for next frame
        self._rec_states = [r1o, r2o, r3o, r4o]

        # Capture shapes after first real inference (for reset)
        if self._rec_shapes is None:
            self._rec_shapes = [r.shape for r in self._rec_states]
            logger.info(
                "Recurrent state shapes: r1=%s r2=%s r3=%s r4=%s",
                *[list(s) for s in self._rec_shapes],
            )

        return fgr, pha, r1o, r2o, r3o, r4o

    def infer(self, src: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a single frame.

        Args:
            src: [1, 3, H, W] float32 NumPy array (normalized 0-1)

        Returns:
            (fgr, pha) — both [1, C, H, W] float32 NumPy arrays
        """
        if self._compiled_model is None:
            raise RuntimeError("OpenVINO model not loaded — call prepare() first")

        fgr, pha, *_ = self._run_inference(src)
        return fgr, pha

    def reset_recurrent_state(self):
        """Reset recurrent states for new video or eye switch."""
        if self._rec_shapes is not None:
            self._rec_states = [
                np.zeros(s, dtype=np.float32) for s in self._rec_shapes
            ]
        else:
            # Fallback: compute from stored dimensions
            if self._input_height > 0 and self._input_width > 0:
                self._rec_shapes = self._compute_rec_shapes(
                    self._input_height, self._input_width,
                )
                self._rec_states = [
                    np.zeros(s, dtype=np.float32) for s in self._rec_shapes
                ]
            else:
                self._rec_states = [
                    np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4)
                ]
        logger.debug("Recurrent state reset")

    def cleanup(self):
        """Release OpenVINO resources."""
        self._infer_request = None
        self._compiled_model = None
        self._rec_states = [None] * 4
        self._rec_shapes = None
        logger.info("OpenVINO engine resources released")
