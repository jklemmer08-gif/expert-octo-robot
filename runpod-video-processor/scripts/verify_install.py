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

models = list(pathlib.Path("/workspace/app/models").glob("*.pth"))
print(f"Real-ESRGAN Models: {len(models)} found")
for m in models:
    print(f"  - {m.name} ({m.stat().st_size / 1e6:.1f} MB)")

# Verify RVM TorchScript models
rvm_dir = pathlib.Path("/workspace/app/models/rvm")
rvm_models = list(rvm_dir.glob("*.torchscript"))
print(f"RVM Models: {len(rvm_models)} found")
for m in rvm_models:
    print(f"  - {m.name} ({m.stat().st_size / 1e6:.1f} MB)")
    # Verify TorchScript loads on CPU
    loaded = torch.jit.load(str(m), map_location="cpu")
    print(f"    TorchScript load: OK")

print("=" * 60)
print("ALL CHECKS PASSED")
