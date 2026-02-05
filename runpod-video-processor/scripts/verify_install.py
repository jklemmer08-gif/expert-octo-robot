#!/usr/bin/env python3
"""Verify all dependencies are installed correctly."""
import subprocess
import pathlib

print("=" * 60)
print("DOCKER IMAGE VERIFICATION")
print("=" * 60)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")

import cv2
print(f"OpenCV: {cv2.__version__}")

import flask
print(f"Flask: {flask.__version__}")

r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
print(f"FFmpeg: {r.stdout.splitlines()[0]}")

r2 = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
nvenc = "hevc_nvenc" in r2.stdout
print(f"NVENC (hevc_nvenc): {'Available' if nvenc else 'Not available - will use libx265'}")

# Import RRDBNet directly to avoid basicsr's top-level data imports
# which reference deprecated torchvision.transforms.functional_tensor
from basicsr.archs.rrdbnet_arch import RRDBNet
print("basicsr RRDBNet: OK")
from realesrgan import RealESRGANer
print("Real-ESRGAN: OK")

models = list(pathlib.Path("/app/models").glob("*.pth"))
print(f"Models: {len(models)} found")
for m in models:
    print(f"  - {m.name} ({m.stat().st_size / 1e6:.1f} MB)")

print("=" * 60)
print("ALL CHECKS PASSED")
