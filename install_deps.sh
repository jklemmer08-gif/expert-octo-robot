#!/bin/bash

echo "========================================================================"
echo "Installing dependencies with --break-system-packages"
echo "========================================================================"

pip install torchcodec torchaudio imageio numpy Pillow --break-system-packages

echo ""
echo "========================================================================"
echo "Verifying installations..."
echo "========================================================================"

python3 << 'VERIFY'
import torch
import torchcodec
import torchaudio
import imageio

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
print(f"✓ GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"✓ TorchCodec: {torchcodec.__version__}")
print(f"✓ TorchAudio: {torchaudio.__version__}")
print(f"✓ imageio: {imageio.__version__}")
print("")
print("========================================================================"
print("✓ All dependencies installed successfully!")
print("========================================================================")
VERIFY
