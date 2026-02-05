# Docker Setup for GPU-Accelerated VR Video Processing

Complete guide for building and running the Docker image on RunPod with PyTorch 2.4.

## Files

- **Dockerfile** - Complete image definition with all dependencies
- **docker-compose.yml** - Orchestration config for GPU access and volume mounts
- **.dockerignore** - Files to exclude from image
- **DOCKER_SETUP.md** - This file

## Quick Start

### Option 1: Using Docker Compose (Recommended)

#### Step 1: Prepare Local Directory Structure

```bash
# On your local machine, create the project structure
mkdir -p vr-processor
cd vr-processor

# Copy these files from the repository
# - Dockerfile
# - docker-compose.yml
# - .dockerignore
# - All .py and .md files from expert-octo-robot
```

#### Step 2: Transfer to RunPod

```bash
# From your local machine
scp -r -P 20938 -i ~/.ssh/id_ed25519 vr-processor/ root@195.26.233.87:/workspace/

# Or if already on RunPod:
cd /workspace
git clone https://github.com/jklemmer08-gif/expert-octo-robot.git vr-processor
cd vr-processor
```

#### Step 3: Prepare Input Directories

```bash
# On RunPod, create input directories
mkdir -p /workspace/vr-processor/input_gpu0
mkdir -p /workspace/vr-processor/input_gpu1
mkdir -p /workspace/vr-processor/output

# Copy your video files (from wherever they are stored)
# cp /path/to/videos/* /workspace/vr-processor/input_gpu0/
```

#### Step 4: Build Docker Image

```bash
cd /workspace/vr-processor

# Build the image (this takes 5-10 minutes)
docker-compose build

# This will:
# ✓ Pull PyTorch 2.4 base image
# ✓ Install system packages (ffmpeg, git, etc.)
# ✓ Install Python packages (torchcodec, torchaudio, imageio)
# ✓ Clone the GitHub repository
# ✓ Verify all installations
```

#### Step 5: Run Container

```bash
# Start the container
docker-compose up -d

# Verify it's running
docker-compose ps

# Expected output:
# NAME                COMMAND             STATUS              PORTS
# vr-processor-gpu    "bash"              Up X seconds        (healthy)
```

#### Step 6: Access Container Shell

```bash
# Enter the running container
docker-compose exec vr-processor bash

# You should now be in /workspace/app with all code and dependencies ready
```

#### Step 7: Run Phase 1 Verification

```bash
# Inside the container
python3 << 'VERIFY'
import torch
import torchcodec
import torchaudio
import imageio

print("✓ PyTorch:", torch.__version__)
print("✓ CUDA:", torch.cuda.is_available())
print("✓ GPUs:", torch.cuda.device_count())
print("✓ TorchCodec:", torchcodec.__version__)
print("✓ TorchAudio:", torchaudio.__version__)
print("✓ imageio:", imageio.__version__)
VERIFY
```

---

### Option 2: Manual Docker Commands

If you prefer not to use docker-compose:

```bash
# Build the image
docker build -t vr-processor:latest .

# Run the container with GPU support
docker run -it --gpus all \
  -v /workspace/input_gpu0:/workspace/input_gpu0:ro \
  -v /workspace/input_gpu1:/workspace/input_gpu1:ro \
  -v /workspace/output:/workspace/output:rw \
  --workdir /workspace/app \
  vr-processor:latest bash
```

---

## Next: Phase 1 Setup (Inside Container)

Once you're inside the container (`docker-compose exec vr-processor bash`):

### Create Test Clips

```bash
# Find video files
ls -lh /workspace/input_gpu0/

# Create 10-second test clips
ffmpeg -i /workspace/input_gpu0/[FIRST_VIDEO].mp4 -t 10 -c copy test_av1.mp4 -y
ffmpeg -i /workspace/input_gpu0/[SECOND_VIDEO].mp4 -t 10 -c copy test_h264.mp4 -y

# Verify
ls -lh test_*.mp4
```

### Run Phase 2: Unit Tests

```bash
# Test decoder
python3 test_decoder.py

# Test processor
python3 test_background_processor.py

# Test encoder
python3 test_encoder.py
```

All tests should pass with green checkmarks ✓

---

## Container Management

### View Logs

```bash
docker-compose logs -f vr-processor
```

### Stop Container

```bash
docker-compose down
```

### Remove Image

```bash
docker-compose down
docker rmi vr-processor:latest
```

### Clean Up Everything

```bash
docker-compose down -v  # Removes volumes too
```

---

## Troubleshooting

### Issue: "docker: command not found"

Docker isn't installed. On RunPod with Ubuntu:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### Issue: "permission denied while trying to connect to Docker daemon"

```bash
# Add user to docker group (if not root)
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: "could not select device driver"

GPU support not enabled. Ensure:
- NVIDIA Docker Runtime installed: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
- RunPod has GPU enabled (check `/proc/driver/nvidia/gpus/`)

### Issue: Container exits immediately

Check logs:
```bash
docker-compose logs vr-processor
```

Look for dependency errors and rebuild if needed.

---

## Environment Inside Container

Path structure:
```
/workspace/
├── app/                    # Your code (from GitHub)
│   ├── video_decoder.py
│   ├── background_processor.py
│   ├── video_processor.py
│   ├── batch_processor.py
│   ├── test_*.py
│   └── (all .py and .md files)
├── input_gpu0/            # Input videos (read-only)
├── input_gpu1/            # Input videos (read-only)
├── output/                # Output files (read-write)
└── .torch/                # PyTorch cache (persistent)
```

Volumes are mounted from the host:
- Input read-only: prevents accidental modification
- Output read-write: processes save here
- Cache persistent: faster model loading on restart

---

## Next Steps

1. ✅ Build Docker image
2. ✅ Run container with GPU support
3. ✅ Verify dependencies inside container
4. ✅ Create test clips
5. ✓ Run Phase 2 unit tests
6. ✓ Run Phase 3 integration tests
7. ✓ Run Phase 4 batch tests
8. ✓ Run Phase 5 production batch

---

## References

- Docker Documentation: https://docs.docker.com/
- Docker Compose: https://docs.docker.com/compose/
- NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker
- PyTorch Docker: https://hub.docker.com/r/pytorch/pytorch

---

**Status**: Docker setup ready for deployment
**Version**: 1.0
