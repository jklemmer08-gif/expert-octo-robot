# GPU-Accelerated VR Video Processing Docker Image
# PyTorch 2.6 + CUDA 12.4 + TorchCodec + all dependencies
# For RunPod Custom Template deployment

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL maintainer="jklemmer08-gif@github.com"
LABEL description="GPU-accelerated VR video processing with background removal"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.torch
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Update system packages and install FFmpeg with dev libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libswresample-dev \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir \
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

# Verify installations (runs during build to catch errors early)
RUN python3 -c "\
import torch; \
import torchcodec; \
import torchaudio; \
import imageio; \
print('=' * 60); \
print('DOCKER IMAGE VERIFICATION'); \
print('=' * 60); \
print(f'PyTorch: {torch.__version__}'); \
print(f'TorchCodec: {torchcodec.__version__}'); \
print(f'TorchAudio: {torchaudio.__version__}'); \
print(f'imageio: {imageio.__version__}'); \
print('=' * 60); \
"

# Expose port for potential web UI
EXPOSE 8888

# Set default command
CMD ["bash"]
