# PPP Processor - Project Context

## Overview
Automated video upscaling pipeline for VR and 2D adult content.
Processes from Stash-managed library to Jellyfin for Meta Quest 3S streaming.

## Architecture
- **Backend**: Python scripts with SQLite for job tracking
- **Processing**: Real-ESRGAN-ncnn-vulkan (Vulkan GPU acceleration)
- **Encoding**: FFmpeg with HEVC/libx265
- **Cloud**: RunPod integration for priority content

## Key Paths
- Library analysis: `../ppp_analysis/`
- Output: `./output/`
- Models: `./bin/models/`
- Logs: `./logs/`
- Temp frames: `./temp/`

## GPU Configuration
- Linux (Arc B580): Use Vulkan backend, tile size 512
- Windows (RTX 3060 Ti): Use NCNN Vulkan, tile size 400
- Cloud (A100/4090): Use PyTorch CUDA, tile size 800

## VR Detection Rules
File is VR if ANY of these conditions:
- Filename contains: `_180`, `_vr`, `_sbs`, `_LR`, `_TB`, `_6K`, `_8K`
- Resolution is stereoscopic (width > 2x height)

## Model Selection Rules
- 4K→6K (VR): `realesr-animevideov3` (fast, good quality)
- 4K→8K (VR): `realesrgan-x4plus` (best quality, slow)
- 1080p→4K: `realesr-animevideov3`
- 720p→1080p: Use FFmpeg lanczos (AI overkill)

## Commands
```bash
# Single video
python scripts/upscale.py input.mp4 -m realesr-animevideov3

# Batch processing
python scripts/batch_process.py --import
python scripts/batch_process.py --run

# Background matting
python scripts/matte.py input.mp4 -o output.mp4

# Cloud processing
python scripts/runpod_worker.py input.mp4 --target 8K
```

## Safety Rules
- NEVER delete source files
- Always verify output before marking complete
- Keep temp files until job completes successfully
- Log all processing decisions

## Budget Constraints
- Total RunPod budget: $75
- Max per job: $5
- Preferred GPU: RTX 4090 (~$0.50/hr)

## Testing
- Use `--sample` flag to process 15-second preview
- Verify VR metadata preserved after processing
- Check output plays correctly in Heresphere/DeoVR

## Processing Tiers
1. **Tier 1** (Top 40): Best quality model, cloud priority
2. **Tier 2** (70 VR): Cloud 6K→8K
3. **Tier 3** (2065): Local bulk processing
