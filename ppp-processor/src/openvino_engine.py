"""OpenVINO inference engine for RobustVideoMatting.

Drop-in replacement for ort_engine.py on Intel platforms.
Loads ONNX model directly via OpenVINO (no PyTorch required).
Targets Intel Arc GPUs via Level Zero backend.

Same interface: prepare(), infer(), reset_recurrent_state(), cleanup().
Extended interface: infer_raw() accepts uint8 NHWC input (preprocessing on GPU).

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

    Two inference modes:
        infer(src)     — [1, 3, H, W] float32 normalized input (legacy)
        infer_raw(src) — [1, H, W, 3] uint8 input (preprocessing done on GPU)
    """

    # RVM resnet50 recurrent state metadata:
    # channels per decoder stage, and stride divisors from original resolution
    REC_CHANNELS = [16, 32, 64, 128]    # r1, r2, r3, r4
    REC_STRIDES = [8, 16, 32, 64]       # spatial dims = ceil(H/stride), ceil(W/stride)

    def __init__(self):
        self._compiled_model = None
        self._compiled_model_ppp = None  # Model with PrePostProcessor (uint8 NHWC input)
        self._infer_request = None
        self._infer_request_ppp = None   # Infer request for PPP model
        self._rec_states = [None] * 4    # NumPy arrays
        self._rec_shapes = None          # Computed from input resolution
        self._input_height = 0
        self._input_width = 0
        self._device = "GPU"
        self._model_path: Optional[Path] = None
        self._has_ppp = False            # Whether PrePostProcessor model is available

    def prepare(
        self,
        height: int,
        width: int,
        onnx_path: Path,
        device: str = "GPU",
    ) -> bool:
        """Load model (ONNX or OpenVINO IR) and compile for target device.

        Args:
            height: Input frame height
            width: Input frame width
            onnx_path: Path to model (.onnx or .xml OpenVINO IR format)
            device: OpenVINO device string ("GPU", "CPU", etc.)

        Returns:
            True if engine is ready, False on failure.
        """
        try:
            ov = _import_ov()
            self._device = device

            # Auto-detect FP16 IR model alongside ONNX
            model_path = onnx_path
            if onnx_path.suffix == ".onnx":
                fp16_ir = onnx_path.with_name(onnx_path.stem + "_fp16_ov.xml")
                if fp16_ir.exists():
                    logger.info("Found FP16 IR model, using: %s", fp16_ir)
                    model_path = fp16_ir

            if not model_path.exists():
                logger.error("Model not found: %s", model_path)
                return False

            logger.info("Loading model: %s", model_path)
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

            # Read model
            start = time.time()
            model = core.read_model(str(model_path))
            logger.info("Model read in %.2fs, compiling for %s...", time.time() - start, self._device)

            # Performance config
            config = {}
            if self._device == "GPU":
                config["PERFORMANCE_HINT"] = "LATENCY"

            # --- Standard model (float32 NCHW input) ---
            start = time.time()
            self._compiled_model = core.compile_model(model, self._device, config)
            logger.info("Standard model compiled in %.2fs", time.time() - start)
            self._infer_request = self._compiled_model.create_infer_request()

            # --- PrePostProcessor model (uint8 NHWC input, normalize on GPU) ---
            try:
                self._build_ppp_model(core, model_path, config)
            except Exception as e:
                logger.warning("PrePostProcessor model failed, raw input disabled: %s", e)
                self._has_ppp = False

            # Compute recurrent state shapes
            self._input_height = height
            self._input_width = width
            self._rec_shapes = self._compute_rec_shapes(height, width)
            logger.info(
                "Rec shapes for %dx%d: %s",
                width, height, [list(s) for s in self._rec_shapes],
            )

            # Initialize recurrent states
            self._rec_states = [
                np.zeros(s, dtype=np.float32) for s in self._rec_shapes
            ]

            # Warm-up both models
            logger.info("Running warm-up inference at %dx%d...", width, height)
            dummy_f32 = np.random.rand(1, 3, height, width).astype(np.float32)
            start = time.time()
            self._run_inference(dummy_f32)
            logger.info("Standard warm-up: %.2fs", time.time() - start)

            self.reset_recurrent_state()

            if self._has_ppp:
                dummy_u8 = np.random.randint(0, 255, (1, height, width, 3), dtype=np.uint8)
                start = time.time()
                self._run_inference_ppp(dummy_u8)
                logger.info("PPP warm-up: %.2fs", time.time() - start)
                self.reset_recurrent_state()

            self._model_path = model_path
            logger.info(
                "OpenVINO engine ready for %dx%d on %s (raw_input=%s)",
                width, height, self._device, self._has_ppp,
            )
            return True

        except Exception as e:
            logger.error("OpenVINO preparation failed: %s", e)
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _build_ppp_model(self, core, onnx_path: Path, config: dict):
        """Build a second compiled model with PrePostProcessor for uint8 NHWC input.

        This moves normalize (uint8→float32 /255) and layout transpose (HWC→NCHW)
        into the OpenVINO graph so they execute on GPU, saving ~80ms/frame at 4K.
        """
        ov = _import_ov()
        from openvino.preprocess import PrePostProcessor

        model = core.read_model(str(onnx_path))

        ppp = PrePostProcessor(model)

        # Configure 'src' input: expect uint8 NHWC, model wants float32 NCHW
        ppp_input = ppp.input("src")
        ppp_input.tensor() \
            .set_element_type(ov.Type.u8) \
            .set_layout(ov.Layout("NHWC"))
        ppp_input.preprocess() \
            .convert_element_type(ov.Type.f32) \
            .mean([0.0, 0.0, 0.0]) \
            .scale([255.0, 255.0, 255.0])
        ppp_input.model().set_layout(ov.Layout("NCHW"))

        # Recurrent state inputs (r1i-r4i) stay as float32 NCHW — no preprocessing
        # They don't need layout/type changes since they're already in the right format

        model_ppp = ppp.build()

        start = time.time()
        self._compiled_model_ppp = core.compile_model(model_ppp, self._device, config)
        logger.info("PPP model compiled in %.2fs", time.time() - start)
        self._infer_request_ppp = self._compiled_model_ppp.create_infer_request()
        self._has_ppp = True

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
        """Run a single inference pass with standard model, returning all 6 outputs."""
        input_dict = {
            "src": src,
            "r1i": self._rec_states[0],
            "r2i": self._rec_states[1],
            "r3i": self._rec_states[2],
            "r4i": self._rec_states[3],
        }

        self._infer_request.infer(input_dict)
        return self._extract_outputs(self._infer_request)

    def _run_inference_ppp(self, src_uint8_nhwc: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Run inference with PrePostProcessor model (uint8 NHWC input)."""
        input_dict = {
            "src": src_uint8_nhwc,
            "r1i": self._rec_states[0],
            "r2i": self._rec_states[1],
            "r3i": self._rec_states[2],
            "r4i": self._rec_states[3],
        }

        self._infer_request_ppp.infer(input_dict)
        return self._extract_outputs(self._infer_request_ppp)

    def _extract_outputs(self, request, skip_fgr: bool = False) -> Tuple[np.ndarray, ...]:
        """Extract outputs from an infer request and update recurrent state.

        Args:
            skip_fgr: If True, return None for fgr to avoid copying the large
                      [1, 3, H, W] tensor (~192 MB at 4K). Used by alpha-only path.
        """
        fgr = None if skip_fgr else request.get_output_tensor(0).data.copy()
        pha = request.get_output_tensor(1).data.copy()
        r1o = request.get_output_tensor(2).data.copy()
        r2o = request.get_output_tensor(3).data.copy()
        r3o = request.get_output_tensor(4).data.copy()
        r4o = request.get_output_tensor(5).data.copy()

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
        """Run inference on a single frame (standard float32 NCHW input).

        Args:
            src: [1, 3, H, W] float32 NumPy array (normalized 0-1)

        Returns:
            (fgr, pha) — both [1, C, H, W] float32 NumPy arrays
        """
        if self._compiled_model is None:
            raise RuntimeError("OpenVINO model not loaded — call prepare() first")

        fgr, pha, *_ = self._run_inference(src)
        return fgr, pha

    def infer_raw(self, src_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a raw uint8 frame (preprocessing done on GPU).

        Requires PrePostProcessor model (built during prepare()).
        Falls back to CPU normalize + standard model if PPP unavailable.

        Args:
            src_uint8: [1, H, W, 3] uint8 NumPy array (raw RGB, 0-255)

        Returns:
            (fgr, pha) — both [1, C, H, W] float32 NumPy arrays
        """
        if self._has_ppp:
            fgr, pha, *_ = self._run_inference_ppp(src_uint8)
            return fgr, pha
        else:
            # Fallback: CPU normalize + standard model
            src_f32 = src_uint8.astype(np.float32) / 255.0
            src_nchw = np.ascontiguousarray(np.transpose(src_f32, (0, 3, 1, 2)))
            return self.infer(src_nchw)

    def infer_alpha_raw(self, src_uint8: np.ndarray) -> np.ndarray:
        """Run inference and return only the alpha matte (skip fgr copy).

        Optimized for alpha-packing pipeline: skips the large fgr tensor copy
        (~192 MB at 4K), saving ~15ms/frame.

        Args:
            src_uint8: [1, H, W, 3] uint8 NumPy array (raw RGB, 0-255)

        Returns:
            pha — [1, 1, H, W] float32 NumPy array
        """
        if self._has_ppp:
            input_dict = {
                "src": src_uint8,
                "r1i": self._rec_states[0],
                "r2i": self._rec_states[1],
                "r3i": self._rec_states[2],
                "r4i": self._rec_states[3],
            }
            self._infer_request_ppp.infer(input_dict)
            _, pha, *_ = self._extract_outputs(self._infer_request_ppp, skip_fgr=True)
            return pha
        else:
            # Fallback
            src_f32 = src_uint8.astype(np.float32) / 255.0
            src_nchw = np.ascontiguousarray(np.transpose(src_f32, (0, 3, 1, 2)))
            _, pha = self.infer(src_nchw)
            return pha

    @property
    def has_raw_input(self) -> bool:
        """Whether infer_raw() uses GPU preprocessing (True) or CPU fallback (False)."""
        return self._has_ppp

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
