# GPU-Accelerated VR Video Processing - Deployment Guide

Complete implementation for processing 10 VR videos (65GB) with AI background removal on RunPod using 2x RTX 6000 Ada GPUs.

## Architecture Overview

```
Input Video (any codec: H.264, HEVC, AV1, VP9)
    ↓
VideoDecoder (TorchCodec NVDEC - GPU accelerated)
    ↓
GPU Tensor Batch (NCHW format, on GPU)
    ↓
BackgroundProcessor (RVM ResNet50 - GPU)
    ↓
RGBA Tensor Batch (with alpha channel)
    ↓
VideoEncoder (VP9 WebM with yuva420p)
    ↓
Output WebM (transparent background)
```

## Implementation Files

### Core Components (Phase 2)
- **`video_decoder.py`**: TorchCodec-based GPU video decoder
  - Supports: H.264, HEVC, AV1, VP9
  - Output: GPU tensors (NCHW format)
  - No hanging on any codec

- **`background_processor.py`**: RVM model for background removal
  - Input: RGB frames [0, 255]
  - Output: RGBA frames [0, 1] with alpha channel
  - Maintains recurrent state for temporal consistency

- **`video_encoder.py`**: VP9 WebM encoder with alpha channel
  - Input: RGBA GPU tensors
  - Output: WebM file with yuva420p pixelformat
  - Supports context manager for safe resource cleanup

### Integration & Orchestration (Phase 3-4)
- **`video_processor.py`**: End-to-end pipeline
  - Decodes → Processes → Encodes
  - Auto-adjusts batch size by resolution
  - Progress tracking and logging
  - Error handling with rollback

- **`batch_processor.py`**: Multi-GPU batch orchestration
  - Round-robin video distribution to GPUs
  - Parallel processing (one GPU per process)
  - Result queue-based communication
  - Summary statistics and logging

### Quality Assurance (Phase 2-4)
- **`test_decoder.py`**: Unit tests for video decoder
  - Tests: AV1, H.264, multi-batch decoding, no hanging
  
- **`test_background_processor.py`**: Unit tests for RVM processor
  - Tests: Initialization, batch processing, recurrent states, resolutions
  
- **`test_encoder.py`**: Unit tests for video encoder
  - Tests: Initialization, batch encoding, multiple batches, resolutions
  
- **`test_integration.py`**: Integration tests for complete pipeline
  - Tests: AV1 E2E, H.264 E2E, GPU utilization
  
- **`test_batch.py`**: Batch processing tests
  - Tests: Dual GPU parallel processing

## Setup Instructions

### Phase 1: Environment Setup (RunPod)

```bash
# SSH to RunPod
ssh root@195.26.233.87 -p 40029

# Install dependencies
pip install torchcodec torchaudio imageio

# Verify installations
python3 -c "import torchcodec; print('TorchCodec:', torchcodec.__version__)"
python3 -c "import torchaudio; print('TorchAudio:', torchaudio.__version__)"
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count())"

# Verify GPUs
nvidia-smi
```

**Expected output**: 2x NVIDIA RTX 6000 Ada with 48GB VRAM each

### Phase 2: Create Test Clips

```bash
# Extract 10-second test clips from input videos
ffmpeg -i input_gpu0/av1_video.mp4 -t 10 -c copy test_av1.mp4
ffmpeg -i input_gpu0/h264_video.mp4 -t 10 -c copy test_h264.mp4

# Verify clips
ls -lh test_*.mp4
```

### Phase 2: Run Unit Tests

```bash
# Test decoder
python3 test_decoder.py

# Test background processor  
python3 test_background_processor.py

# Test encoder
python3 test_encoder.py

# All tests should pass: ✅
```

### Phase 3: Integration Testing

```bash
# Test complete pipeline with test clips
python3 test_integration.py

# Validate output
ffprobe -select_streams v:0 -show_entries stream=pix_fmt output_av1_e2e.webm
# Expected: pix_fmt=yuva420p (confirms alpha channel)
```

### Phase 4: Batch Processing Test

```bash
# Test dual GPU processing with 2 videos
python3 test_batch.py

# Monitor GPU utilization in separate terminal
watch -n 2 nvidia-smi
```

**Expected**: Both GPUs at 60-90% utilization, videos processing simultaneously

### Phase 5: Production Batch Processing

#### GPU 0 Batch (5 videos, 28GB)

```bash
# Edit batch_processor.py or run directly
python3 -c "
from pathlib import Path
from batch_processor import BatchProcessor

processor = BatchProcessor(model='resnet50')
results = processor.process_batch(
    videos=sorted(Path('input_gpu0').glob('*.mp4')),
    output_dir=Path('output_run1_gpu0'),
    num_gpus=2
)
"
```

**Monitoring** (in separate terminals):
```bash
# Terminal 1: GPU usage
watch -n 2 nvidia-smi

# Terminal 2: Disk usage
watch -n 10 df -h /

# Terminal 3: Logs
tail -f /var/log/batch_processing.log
```

#### GPU 1 Batch (2 videos, 37GB)

After GPU 0 batch completes:

```bash
python3 -c "
from pathlib import Path
from batch_processor import BatchProcessor

processor = BatchProcessor(model='resnet50')
results = processor.process_batch(
    videos=sorted(Path('input_gpu1').glob('*.mp4')),
    output_dir=Path('output_run1_gpu1'),
    num_gpus=2
)
"
```

## Quality Gates

### Gate 1: Environment Setup
- [ ] TorchCodec installed and importable
- [ ] TorchAudio installed and importable
- [ ] Both GPUs visible and accessible
- [ ] Test clips created (< 500MB each)

### Gate 2: Component Testing
- [ ] VideoDecoder tests pass (AV1 + H.264 without hanging)
- [ ] BackgroundProcessor tests pass (all resolutions)
- [ ] VideoEncoder tests pass (context manager works)
- [ ] All unit tests: 100% pass rate

### Gate 3: Integration Testing
- [ ] AV1 E2E test passes
- [ ] H.264 E2E test passes
- [ ] GPU utilization 60-90%
- [ ] Output files have alpha channel (ffprobe confirms yuva420p)

### Gate 4: Batch Testing
- [ ] Dual GPU processing works
- [ ] No resource conflicts or OOM errors
- [ ] Both videos process simultaneously
- [ ] Progress logs are clear and informative

### Gate 5: Production
- [ ] All 7 videos processed successfully
- [ ] Output files created with correct size
- [ ] No disk space issues
- [ ] Processing time under 8 hours total

## Performance Expectations

### Phase 3 (Test Clips - 10 seconds each)
- **FPS**: 5-15 fps for 1080p
- **Time**: 1-2 seconds per clip
- **GPU Util**: 70-85%

### Phase 5 (Production - Full videos)
- **4K (3840x2160)**: 2-5 fps
- **HD (1920x1080)**: 5-10 fps
- **Total time**: 4-8 hours for 7 videos (5 on GPU 0, 2 on GPU 1)
- **GPU Util**: 60-90%
- **Output size**: ~4-5 GB per 4K video

## Troubleshooting

### Issue: "No module named 'torchcodec'"
```bash
# Reinstall torchcodec
pip install --force-reinstall torchcodec
```

### Issue: "CUDA out of memory"
- Reduce batch_size in VideoProcessor initialization
- Reduce downsample_ratio in BackgroundProcessor
- Process one GPU at a time (set num_gpus=1)

### Issue: "ffmpeg not found" during test clip creation
```bash
apt-get update && apt-get install -y ffmpeg
```

### Issue: "Video hangs indefinitely"
- Check that TorchCodec was installed correctly
- Verify GPU drivers: `nvidia-smi`
- Try with smaller test clip: `ffmpeg -i input.mp4 -t 1 -c copy test_tiny.mp4`

### Issue: "Low GPU utilization"
- Check batch_size: smaller resolutions can use larger batches
- Monitor with `nvidia-smi dmon` to see per-process utilization
- Check for CPU bottlenecks: `top` command

## Storage Management

### Disk Usage Tracking
```bash
# Monitor in real-time
watch -n 10 'du -sh /workspace/* && df -h /'

# Before batch processing
du -sh input_gpu0 input_gpu1
# Expected: ~65GB total

# Expected output directory size
# - 5 videos @ 4-5 GB each = ~25 GB for GPU 0 batch
# - 2 videos @ 4-5 GB each = ~10 GB for GPU 1 batch
# Total: ~35 GB (pod volume + container disk)
```

### Cleanup Strategy
```bash
# After successful batch processing
rm -rf output_run1_gpu0  # Move to external storage first!
# Or compress: tar -czf output_run1_gpu0.tar.gz output_run1_gpu0

# This frees pod volume for next batch
```

## Deployment Checklist

### Pre-Deployment
- [ ] All code files transferred to RunPod
- [ ] Dependencies installed
- [ ] GPU access verified
- [ ] Test clips created
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Batch test passes

### Deployment
- [ ] Input video directories prepared
- [ ] Output directories have write permissions
- [ ] Disk space verified (70GB pod + 220GB container)
- [ ] Logging configured
- [ ] GPU monitoring tools ready

### Post-Deployment
- [ ] All videos processed
- [ ] Output quality verified (spot check 3 videos)
- [ ] Results moved to safe storage
- [ ] Cleanup completed
- [ ] Documentation updated

## Success Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| GPU Utilization | 60-90% | __ |
| Processing Speed (4K) | 2-5 fps | __ |
| Success Rate | 95%+ | __ |
| Total Time | < 8 hours | __ |
| Storage Efficiency | < 10GB temp | __ |
| No Hangs | 100% | __ |

## References

- **TorchCodec**: https://github.com/pytorch/pytorch/tree/main/torch/csrc/api/include/torch/csrc/jit/tensorexpr
- **RVM Model**: https://github.com/PeterL1n/RobustVideoMatting
- **VP9 WebM**: https://www.webmproject.org/
- **imageio**: https://imageio.readthedocs.io/

## Support

If tests fail at any stage:
1. Check logs: `tail -100 /var/log/batch_processing.log`
2. Verify GPU: `nvidia-smi`
3. Test individual component
4. Review error message for root cause
5. Fix issue and re-run tests from that phase

Only proceed to next phase after all stage gates pass.
