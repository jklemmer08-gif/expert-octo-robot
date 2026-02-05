# GPU-Accelerated VR Video Processing Docker Image
# PyTorch 2.4 + CUDA 12.4 + all dependencies + code

FROM pytorch/pytorch:2.4.0-cuda12.4-runtime-ubuntu22.04

LABEL maintainer="jklemmer08-gif@github.com"
LABEL description="GPU-accelerated VR video processing with background removal"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.torch

# Update system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages \
    torchcodec \
    torchaudio \
    imageio \
    numpy \
    Pillow

# Clone the repository
RUN git clone https://github.com/jklemmer08-gif/expert-octo-robot.git /workspace/app

# Set working directory to app
WORKDIR /workspace/app

# Create directories for input/output
RUN mkdir -p /workspace/input_gpu0 /workspace/input_gpu1 /workspace/output

# Verify installations
RUN python3 << 'VERIFY'
import torch
import torchcodec
import torchaudio
import imageio

print("=" * 60)
print("DOCKER IMAGE VERIFICATION")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"TorchCodec: {torchcodec.__version__}")
print(f"TorchAudio: {torchaudio.__version__}")
print(f"imageio: {imageio.__version__}")
print("=" * 60)
VERIFY

# Set default command
CMD ["bash"]
