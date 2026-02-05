# RunPod Video Processor — GPU-Accelerated Upscaling
# Base: RunPod PyTorch with CUDA 12.1 (pre-built, known working on RunPod)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

LABEL maintainer="runpod-video-processor"
LABEL description="GPU-accelerated video upscaling with Real-ESRGAN"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# ============================================================
# System dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Python dependencies
# ============================================================
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 && \
    # Fix basicsr compatibility with torchvision >= 0.17 \
    # (functional_tensor was removed, use functional instead) \
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py

# ============================================================
# Real-ESRGAN model weights (baked in — no runtime downloads)
# ============================================================
RUN mkdir -p /app/models && \
    wget -q -O /app/models/RealESRGAN_x4plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" && \
    wget -q -O /app/models/RealESRGAN_x4plus_anime_6B.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" && \
    wget -q -O /app/models/RealESRGAN_x2plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" && \
    echo "Real-ESRGAN models downloaded:" && ls -lh /app/models/

# ============================================================
# RVM (Robust Video Matting) TorchScript models (baked in)
# ============================================================
RUN mkdir -p /app/models/rvm && \
    wget -q -O /app/models/rvm/rvm_resnet50_fp32.torchscript \
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.torchscript" && \
    wget -q -O /app/models/rvm/rvm_mobilenetv3_fp32.torchscript \
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.torchscript" && \
    echo "RVM models downloaded:" && ls -lh /app/models/rvm/

# ============================================================
# Application code
# ============================================================
COPY src/ /app/src/

# ============================================================
# Workspace directories (created on volume mount or at runtime)
# ============================================================
RUN mkdir -p /workspace/input /workspace/output /workspace/temp

EXPOSE 8080

COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh

# ============================================================
# Verify installation
# ============================================================
RUN python3 /app/scripts/verify_install.py

CMD ["python3", "-u", "/app/src/app.py"]
