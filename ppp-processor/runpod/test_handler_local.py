#!/usr/bin/env python3
"""Local test for RunPod handler — bypasses S3 and RunPod, tests matting directly.

Usage (from repo root):
  python runpod/test_handler_local.py "N:/ppp-output/test/vr_sbs_test_10s.mp4"
  python runpod/test_handler_local.py "N:/ppp-output/test/vr_sbs_test_10s.mp4" --vr-type 2d
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Patch handler paths before importing it
# Handler expects /app/RobustVideoMatting/ and /app/models/ (Docker layout)
# Locally these are in bin/
REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure ffmpeg/ffprobe are on PATH
bin_dir = str(REPO_ROOT / "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

# Add repo root to sys.path so handler can import
sys.path.insert(0, str(REPO_ROOT / "runpod"))

# Now patch the handler module paths before load_model runs
import handler

# Override Docker paths to local paths
handler.load_model_orig = handler.load_model

def load_model_local(model_type="resnet50"):
    """Patched load_model that uses local paths."""
    import torch

    if handler._model is not None:
        return

    rvm_path = REPO_ROOT / "bin" / "RobustVideoMatting"
    if rvm_path.exists() and str(rvm_path) not in sys.path:
        sys.path.insert(0, str(rvm_path))

    from model import MattingNetwork  # type: ignore

    handler._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {handler._device}")

    model_path = REPO_ROOT / "bin" / "models" / f"rvm_{model_type}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    handler._model = MattingNetwork(model_type).eval()
    handler._model.load_state_dict(torch.load(model_path, map_location=handler._device, weights_only=True))
    handler._model = handler._model.to(handler._device)

    handler._use_fp16 = handler._device.type == "cuda"
    if handler._use_fp16:
        handler._model = handler._model.half()
        print("FP16 enabled")

    # Skip torch.compile on Windows (Dynamo dtype conflict with FP16)
    import platform
    if handler._device.type == "cuda" and platform.system() != "Windows":
        try:
            handler._model = torch.compile(handler._model, mode="reduce-overhead")
            print("torch.compile() enabled (reduce-overhead)")
        except Exception:
            try:
                handler._model = torch.compile(handler._model, backend="eager")
                print("torch.compile() enabled (eager)")
            except Exception as e:
                print(f"torch.compile() unavailable: {e}")
    else:
        print("Skipping torch.compile() on Windows")

    print("Model loaded")

handler.load_model = load_model_local


def main():
    parser = argparse.ArgumentParser(description="Test RunPod handler locally")
    parser.add_argument("input", help="Path to test video")
    parser.add_argument("--output", help="Output path (default: input_matted.mp4 next to input)")
    parser.add_argument("--vr-type", default="auto", choices=["auto", "sbs", "tb", "2d"],
                        help="VR type (default: auto-detect)")
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "mobilenetv3"])
    parser.add_argument("--downsample", type=float, default=0.4)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_matted.mp4"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # 1. Probe
    print("--- Probing video ---")
    probe = handler.probe_video(input_path)
    print(f"  {probe['width']}x{probe['height']} @ {probe['fps']:.2f} fps")
    print(f"  {probe['total_frames']} frames, {probe['duration']:.1f}s, audio={probe['has_audio']}")

    # 2. Detect VR type
    vr_type = args.vr_type
    if vr_type == "auto":
        vr_type = handler.detect_vr_type(input_path, probe["width"], probe["height"])
    print(f"  VR type: {vr_type}")
    print()

    # 3. Load model
    print("--- Loading model ---")
    handler.load_model(args.model)
    print()

    # 4. Process
    green_color = [0, 177, 64]
    t_start = time.time()

    if vr_type == "sbs":
        print("--- Processing VR SBS ---")
        success = handler.process_vr_sbs(
            input_path, output_path, probe,
            green_color=green_color,
            downsample_ratio=args.downsample,
            bitrate="50M",
        )
    else:
        print("--- Processing 2D ---")
        success = handler.process_video_streaming(
            input_path, output_path, probe,
            green_color=green_color,
            downsample_ratio=args.downsample,
            bitrate="15M",
        )

    elapsed = time.time() - t_start
    print()

    if success:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"SUCCESS: {output_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Speed: {probe['total_frames'] / max(elapsed, 0.001):.1f} fps")
    else:
        print("FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
