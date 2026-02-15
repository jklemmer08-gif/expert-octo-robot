#!/usr/bin/env python3
"""Smoke test for OpenVINO RVM inference engine.

Loads ONNX model, runs inference on a dummy frame,
verifies output shapes and recurrent state cycling.

Usage:
    python scripts/test_openvino.py [--onnx PATH] [--device GPU|CPU] [--resolution 4000x4000]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Test OpenVINO RVM inference")
    parser.add_argument("--onnx", type=str, default="./trt_engines/rvm_resnet50.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--device", type=str, default="GPU",
                        help="OpenVINO device (GPU, CPU)")
    parser.add_argument("--resolution", type=str, default="1920x1080",
                        help="Test resolution WxH")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of test frames to run")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found: {onnx_path}")
        sys.exit(1)

    w, h = [int(x) for x in args.resolution.split("x")]
    print(f"Resolution: {w}x{h}")
    print(f"ONNX model: {onnx_path}")
    print(f"Device: {args.device}")

    # Check OpenVINO availability
    try:
        import openvino as ov
        print(f"OpenVINO version: {ov.__version__}")
        core = ov.Core()
        print(f"Available devices: {core.available_devices}")
    except ImportError:
        print("ERROR: openvino not installed")
        sys.exit(1)

    # Add project root to path for src imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.openvino_engine import RVMOpenVINOEngine

    engine = RVMOpenVINOEngine()

    # Prepare
    print(f"\n--- Preparing engine for {w}x{h} ---")
    t0 = time.time()
    success = engine.prepare(h, w, onnx_path, device=args.device)
    print(f"Prepare: {'OK' if success else 'FAILED'} ({time.time() - t0:.2f}s)")
    if not success:
        sys.exit(1)

    # Run inference on dummy frames
    print(f"\n--- Running {args.frames} frames ---")
    times = []
    for i in range(args.frames):
        dummy = np.random.rand(1, 3, h, w).astype(np.float32)
        t0 = time.time()
        fgr, pha = engine.infer(dummy)
        elapsed = time.time() - t0
        times.append(elapsed)

        if i == 0:
            print(f"  Frame {i}: fgr={fgr.shape} pha={pha.shape} ({elapsed:.3f}s)")
            assert fgr.shape == (1, 3, h, w), f"fgr shape mismatch: {fgr.shape}"
            assert pha.shape == (1, 1, h, w), f"pha shape mismatch: {pha.shape}"
            assert fgr.dtype == np.float32, f"fgr dtype: {fgr.dtype}"
            assert pha.dtype == np.float32, f"pha dtype: {pha.dtype}"
            print("  Shape/dtype checks: PASSED")
        elif i == args.frames - 1:
            print(f"  Frame {i}: ({elapsed:.3f}s)")

    # Performance summary
    avg_ms = np.mean(times[1:]) * 1000  # Skip first frame (warm-up)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    print(f"\n--- Performance ---")
    print(f"  Avg: {avg_ms:.1f}ms/frame ({fps:.1f} fps) [excluding warm-up]")
    print(f"  Min: {min(times[1:]) * 1000:.1f}ms  Max: {max(times[1:]) * 1000:.1f}ms")

    # Test recurrent state reset
    print("\n--- Testing recurrent state reset ---")
    engine.reset_recurrent_state()
    fgr2, pha2 = engine.infer(np.random.rand(1, 3, h, w).astype(np.float32))
    assert fgr2.shape == (1, 3, h, w), "Post-reset fgr shape mismatch"
    assert pha2.shape == (1, 1, h, w), "Post-reset pha shape mismatch"
    print("  Reset + re-inference: PASSED")

    # Cleanup
    engine.cleanup()
    print("\n--- All tests PASSED ---")


if __name__ == "__main__":
    main()
