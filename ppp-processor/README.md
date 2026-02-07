# PPP Processor - Video Upscaling Pipeline

Automated video upscaling pipeline for VR and 2D content with Real-ESRGAN.
Optimized for Heresphere/DeoVR playback on Meta Quest 3S.

## Features

- **Local upscaling** using Real-ESRGAN on Intel Arc B580
- **VR SBS handling** - automatic split/upscale/merge for side-by-side content
- **Background matting** (RVM) for passthrough experience
- **Batch processing** with SQLite tracking and resume capability
- **RunPod integration** for cloud processing with $75 budget management
- **Heresphere-optimized** HEVC encoding

## Quick Start

### 1. Install Dependencies

```bash
cd ppp-processor

# Python packages
pip install -r requirements.txt

# Check Real-ESRGAN (already installed in bin/)
./bin/realesrgan-ncnn-vulkan --help
```

### 2. Process a Single Video

```bash
# Basic upscale (2x)
python scripts/upscale.py /path/to/video.mp4

# VR content with specific model
python scripts/upscale.py /path/to/vr_180_sbs.mp4 -m realesrgan-x4plus -s 2

# Process 15-second sample first
python scripts/upscale.py /path/to/video.mp4 --sample
```

### 3. Batch Processing

```bash
# Import jobs from analysis CSVs
python scripts/batch_process.py --import

# Check queue status
python scripts/batch_process.py --status

# Start processing (runs until interrupted)
python scripts/batch_process.py --run

# Process only 5 jobs
python scripts/batch_process.py --run --limit 5
```

### 4. Cloud Processing (RunPod)

```bash
# Set API key
export RUNPOD_API_KEY='your-key-here'

# Check available GPUs
python scripts/runpod_worker.py --gpus

# Estimate cost
python scripts/runpod_worker.py /path/to/video.mp4 --estimate --target 8K

# Process on cloud
python scripts/runpod_worker.py /path/to/video.mp4 -o output_8k.mp4 --target 8K

# Check budget
python scripts/runpod_worker.py --budget
```

### 5. Background Matting (Passthrough)

```bash
# Setup RVM (first time only)
python scripts/matte.py --setup

# Process video with green screen output
python scripts/matte.py /path/to/video.mp4 -o matted.mp4

# Process VR SBS content
python scripts/matte.py /path/to/vr_sbs.mp4 -o matted_sbs.mp4 --vr
```

## Directory Structure

```
ppp-processor/
├── scripts/
│   ├── upscale.py         # Single video upscaling
│   ├── batch_process.py   # Queue-based batch processing
│   ├── matte.py           # Background removal (RVM)
│   └── runpod_worker.py   # Cloud processing
├── config/
│   └── settings.yaml      # Configuration
├── bin/
│   ├── realesrgan-ncnn-vulkan
│   └── models/
├── output/                # Processed videos
├── temp/                  # Temporary frames
├── logs/                  # Processing logs
├── queue/                 # Job queue (filesystem)
└── jobs.db                # SQLite job database
```

## Model Selection

| Source | Target | Model | Speed | Quality |
|--------|--------|-------|-------|---------|
| 4K | 6K VR | realesr-animevideov3 | Fast | ⭐⭐⭐⭐ |
| 4K | 8K VR | realesrgan-x4plus | Slow | ⭐⭐⭐⭐⭐ |
| 1080p | 4K | realesr-animevideov3 | Fast | ⭐⭐⭐⭐ |
| 720p | 1080p | (use FFmpeg lanczos) | Instant | ⭐⭐⭐ |

## Heresphere Chroma Key Setup

For passthrough content with green screen background:

1. Open video in Heresphere
2. Go to **Video Settings** → **Chroma Key**
3. Enable chroma key
4. Set **Key Color**: RGB(0, 177, 64)
5. Adjust **Similarity**: 0.4
6. Adjust **Smoothness**: 0.1

## Processing Times (Estimates)

### Local (Intel Arc B580)

| Content | Time per 30-min |
|---------|-----------------|
| 4K → 6K VR | 5-6 hours |
| 1080p → 4K | 2-3 hours |
| With matting | +50% time |

### Cloud (RTX 4090)

| Content | Time | Cost |
|---------|------|------|
| 4K → 6K VR | 1-1.5 hrs | ~$1.00 |
| 4K → 8K VR | 3-4 hrs | ~$2.50 |

## Budget Management ($75)

Recommended allocation:
- **Top 15 → 8K**: ~$45 (showcase content)
- **Next 25 → 6K**: ~$30 (high-quality VR)

Check spending:
```bash
python scripts/runpod_worker.py --budget
```

## Troubleshooting

### Out of Memory
Reduce tile size in `config/settings.yaml`:
```yaml
gpu:
  tile_size: 256  # Default 512, reduce if OOM
```

### VR Not Detected
Ensure filename contains VR identifiers:
- `_180`, `_360`, `_vr`, `_sbs`, `_LR`, etc.

### Encoding Issues
Check FFmpeg codec support:
```bash
ffmpeg -encoders | grep hevc
```

## License

Personal use only. Not for redistribution.
