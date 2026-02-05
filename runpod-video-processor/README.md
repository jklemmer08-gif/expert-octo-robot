# RunPod Video Processor

GPU-accelerated video upscaling for 2D and VR content, running on RunPod GPU pods with a web UI.

## Features

- **Real-ESRGAN upscaling** — 1080p to 4K, VR 4K to 6K/8K
- **VR-aware processing** — auto-detects SBS/OU layouts, splits per-eye for upscaling
- **Chunked pipeline** — handles 20GB+ files without exhausting disk space
- **Adaptive GPU settings** — auto-detects VRAM and adjusts tile/batch sizes
- **NVENC encoding** — GPU-accelerated H.265 with libx265 CPU fallback
- **Web UI** — file browser, job config, real-time progress via SSE

## Quick Start (RunPod)

1. Create a RunPod GPU pod with the Docker image
2. Set the network volume mount to `/workspace`
3. Place input videos in `/workspace/input/`
4. Access the web UI at `https://{pod-id}-8080.proxy.runpod.net`
5. Select a file, configure settings, click "Start Upscaling"

## Local Development

```bash
# Generate test fixtures
./scripts/generate_test_fixtures.sh

# Run tests (no GPU needed for unit tests)
cd runpod-video-processor
pytest tests/ -k "not integration" -v

# Build Docker image
./scripts/build.sh

# Run locally with docker-compose
mkdir -p test_input test_output test_temp
cp tests/fixtures/*.mp4 test_input/
docker-compose up
```

## Project Structure

```
src/
  app.py              — Flask web UI (port 8080)
  config.py           — Environment variables and defaults
  gpu.py              — VRAM detection and adaptive profiles
  pipeline/
    validator.py      — Input file validation
    detector.py       — VR layout detection (SBS/OU/mono/2D)
    upscaler.py       — Real-ESRGAN chunked upscaling pipeline
    encoder.py        — FFmpeg encoding (NVENC/libx265)
    metadata.py       — VR metadata preservation
  storage/
    volume.py         — /workspace file operations, disk space
  utils/
    ffmpeg.py         — FFmpeg/ffprobe command helpers
    logging.py        — Structured JSON logging
    progress.py       — Progress tracking + SSE
  templates/
    index.html        — Web UI (vanilla JS + TailwindCSS CDN)
```

## GPU Profiles

| GPU | VRAM | Tile Size | Batch Size |
|-----|------|-----------|------------|
| A100-80GB | 80 GB | 1024 | 16 |
| A100-40GB | 40 GB | 768 | 8 |
| L40S | 48 GB | 768 | 8 |
| RTX 4090/A40 | 24 GB | 512 | 4 |
| T4 | 16 GB | 384 | 2 |
| Fallback | <12 GB | 256 | 1 |

## Upscaling Pipeline

For large files, processing is segmented to manage disk space:

1. Extract N frames (default 1000) to temp PNGs
2. Upscale each frame with Real-ESRGAN (per-eye for VR)
3. Encode segment to intermediate MKV
4. Delete temp frames immediately
5. Repeat for all segments
6. Concatenate segments, mux audio, write VR metadata

## Models

| Model | Use Case | Scale |
|-------|----------|-------|
| RealESRGAN_x4plus | General live-action | 4x |
| RealESRGAN_x4plus_anime_6B | Animation/anime | 4x |
| RealESRGAN_x2plus | Conservative 2x upscale | 2x |
