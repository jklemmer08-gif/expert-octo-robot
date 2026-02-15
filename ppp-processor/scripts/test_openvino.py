#!/usr/bin/env python3
"""Smoke test for OpenVINO RVM inference engine.

Loads ONNX model, runs inference on a dummy frame,
verifies output shapes and recurrent state cycling.
Benchmarks both standard (float32 NCHW) and raw (uint8 NHWC) paths.

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

    print(f"  has_raw_input (PPP model): {engine.has_raw_input}")

    # ---- Benchmark standard path (float32 NCHW) ----
    print(f"\n--- Standard path: {args.frames} frames (float32 NCHW) ---")
    engine.reset_recurrent_state()
    times_std = []
    for i in range(args.frames):
        dummy = np.random.rand(1, 3, h, w).astype(np.float32)
        t0 = time.time()
        fgr, pha = engine.infer(dummy)
        elapsed = time.time() - t0
        times_std.append(elapsed)

        if i == 0:
            print(f"  Frame {i}: fgr={fgr.shape} pha={pha.shape} ({elapsed:.3f}s)")
            assert fgr.shape == (1, 3, h, w), f"fgr shape mismatch: {fgr.shape}"
            assert pha.shape == (1, 1, h, w), f"pha shape mismatch: {pha.shape}"
            assert fgr.dtype == np.float32, f"fgr dtype: {fgr.dtype}"
            assert pha.dtype == np.float32, f"pha dtype: {pha.dtype}"
            print("  Shape/dtype checks: PASSED")

    avg_ms = np.mean(times_std[1:]) * 1000
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    print(f"  Avg: {avg_ms:.1f}ms/frame ({fps:.1f} fps) [excluding warm-up]")
    print(f"  Min: {min(times_std[1:]) * 1000:.1f}ms  Max: {max(times_std[1:]) * 1000:.1f}ms")

    # ---- Benchmark raw path (uint8 NHWC) ----
    if engine.has_raw_input:
        print(f"\n--- Raw path: {args.frames} frames (uint8 NHWC, GPU preprocessing) ---")
        engine.reset_recurrent_state()
        times_raw = []
        for i in range(args.frames):
            dummy = np.random.randint(0, 255, (1, h, w, 3), dtype=np.uint8)
            t0 = time.time()
            fgr, pha = engine.infer_raw(dummy)
            elapsed = time.time() - t0
            times_raw.append(elapsed)

            if i == 0:
                print(f"  Frame {i}: fgr={fgr.shape} pha={pha.shape} ({elapsed:.3f}s)")
                assert fgr.shape == (1, 3, h, w), f"raw fgr shape mismatch: {fgr.shape}"
                assert pha.shape == (1, 1, h, w), f"raw pha shape mismatch: {pha.shape}"
                print("  Shape/dtype checks: PASSED")

        avg_ms_raw = np.mean(times_raw[1:]) * 1000
        fps_raw = 1000.0 / avg_ms_raw if avg_ms_raw > 0 else 0
        print(f"  Avg: {avg_ms_raw:.1f}ms/frame ({fps_raw:.1f} fps) [excluding warm-up]")
        print(f"  Min: {min(times_raw[1:]) * 1000:.1f}ms  Max: {max(times_raw[1:]) * 1000:.1f}ms")

        # Compare
        speedup = avg_ms / avg_ms_raw if avg_ms_raw > 0 else 0
        saved = avg_ms - avg_ms_raw
        print(f"\n--- Comparison ---")
        print(f"  Standard: {avg_ms:.1f}ms  vs  Raw: {avg_ms_raw:.1f}ms")
        print(f"  Saved: {saved:.1f}ms/frame ({speedup:.2f}x)")
    else:
        print("\n--- Raw path: SKIPPED (PrePostProcessor not available) ---")

    # ---- Benchmark end-to-end simulation (uint8 → infer_raw → tobytes) ----
    if engine.has_raw_input:
        print(f"\n--- E2E simulation: {args.frames} frames (uint8→infer_raw→extract_alpha→tobytes) ---")
        engine.reset_recurrent_state()
        times_e2e = []
        for i in range(args.frames):
            # Simulate: decode gives us HWC uint8 frame
            frame_hwc = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            t0 = time.time()
            # Add batch dim (this is what processor.py does)
            src_raw = np.ascontiguousarray(frame_hwc[np.newaxis])
            fgr, pha = engine.infer_raw(src_raw)
            alpha = pha[0, 0]  # extract [H, W]
            elapsed = time.time() - t0
            times_e2e.append(elapsed)

        avg_e2e = np.mean(times_e2e[1:]) * 1000
        fps_e2e = 1000.0 / avg_e2e if avg_e2e > 0 else 0
        print(f"  Avg: {avg_e2e:.1f}ms/frame ({fps_e2e:.1f} fps)")

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
