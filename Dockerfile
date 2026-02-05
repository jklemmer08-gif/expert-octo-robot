# GPU-Accelerated VR Video Processing Docker Image
# PyTorch 2.6 + CUDA 12.4 + TorchCodec + all dependencies
# For RunPod Custom Template deployment
# Uses TorchCodec for GPU-accelerated video decoding (NVDEC)

# Use NVIDIA CUDA devel image for full compatibility
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

LABEL maintainer="jklemmer08-gif@github.com"
LABEL description="GPU-accelerated VR video processing with background removal"
LABEL version="2.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.torch
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies including FFmpeg dev libraries for TorchCodec
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    # FFmpeg and all development libraries required by TorchCodec
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    libpostproc-dev \
    # Additional video codec support
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libaom-dev \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch 2.6 with CUDA 12.4 support (matching base image)
RUN pip3 install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install TorchCodec (requires PyTorch 2.6)
RUN pip3 install --no-cache-dir torchcodec

# Install remaining Python dependencies
RUN pip3 install --no-cache-dir \
    imageio \
    imageio-ffmpeg \
    numpy \
    Pillow

# Clone the repository
RUN git clone https://github.com/jklemmer08-gif/expert-octo-robot.git /workspace/app

# Set working directory to app
WORKDIR /workspace/app

# Create directories for input/output
RUN mkdir -p /workspace/input_gpu0 /workspace/input_gpu1 /workspace/output

# Verify all installations
RUN python3 -c "\
import torch; \
import torchcodec; \
import torchaudio; \
import imageio; \
import subprocess; \
print('=' * 60); \
print('DOCKER IMAGE VERIFICATION'); \
print('=' * 60); \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA Available: {torch.cuda.is_available()}'); \
print(f'CUDA Version: {torch.version.cuda}'); \
print(f'TorchCodec: {torchcodec.__version__}'); \
print(f'TorchAudio: {torchaudio.__version__}'); \
print(f'imageio: {imageio.__version__}'); \
result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True); \
print(f'FFmpeg: {result.stdout.split(chr(10))[0]}'); \
print('=' * 60); \
print('All dependencies installed successfully!'); \
print('=' * 60); \
"

# Expose port for potential web UI
EXPOSE 8888

# Set default command
CMD ["bash"]
