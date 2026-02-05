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
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

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
    echo "Models downloaded:" && ls -lh /app/models/

# ============================================================
# Application code
# ============================================================
COPY src/ /app/src/

# ============================================================
# Workspace directories (created on volume mount or at runtime)
# ============================================================
RUN mkdir -p /workspace/input /workspace/output /workspace/temp

# ============================================================
# Verify installation
# ============================================================
RUN python3 -c "\
import torch; \
import cv2; \
import flask; \
print('=' * 60); \
print('DOCKER IMAGE VERIFICATION'); \
print('=' * 60); \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA Available: {torch.cuda.is_available()}'); \
print(f'CUDA Version: {torch.version.cuda}'); \
print(f'OpenCV: {cv2.__version__}'); \
print(f'Flask: {flask.__version__}'); \
import subprocess; \
r = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True); \
print(f'FFmpeg: {r.stdout.splitlines()[0]}'); \
# Check NVENC \
r2 = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True); \
nvenc = 'hevc_nvenc' in r2.stdout; \
print(f'NVENC (hevc_nvenc): {\"Available\" if nvenc else \"Not available — will use libx265\"}'); \
# Verify Real-ESRGAN can be imported \
from basicsr.archs.rrdbnet_arch import RRDBNet; \
from realesrgan import RealESRGANer; \
print('Real-ESRGAN: OK'); \
import pathlib; \
models = list(pathlib.Path('/app/models').glob('*.pth')); \
print(f'Models: {len(models)} found'); \
for m in models: print(f'  - {m.name} ({m.stat().st_size / 1e6:.1f} MB)'); \
print('=' * 60); \
"

EXPOSE 8080

COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh

CMD ["python3", "-u", "/app/src/app.py"]
