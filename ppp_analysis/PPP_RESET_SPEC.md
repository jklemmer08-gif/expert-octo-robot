# PPP (Porno Processing Pipeline) Reset Specification

## Document Control
- **Version:** 2.0.0
- **Created:** 2026-02-01
- **Status:** READY FOR IMPLEMENTATION

---

## Executive Summary

This document provides a complete reset specification for the PPP project, including a new cloud-ready Processing Agent module. The architecture is designed for:
- Autonomous operation via Claude Code agent
- Docker containerization (deployable on Windows or Linux)
- Intelligent model selection based on content type
- 15-second QA sampling before full job execution
- Library monitoring for new content discovery

---

## Existing Tech Stack (Preserved)

| Component | Purpose | Location |
|-----------|---------|----------|
| **Stash** | Metadata database + scene management | http://localhost:9999/graphql |
| **Jellyfin** | Media server for Quest 3S streaming | http://localhost:8096 |
| **FFmpeg** | Frame extraction, encoding (QSV/VAAPI) | System |
| **Real-ESRGAN-ncnn-vulkan** | AI upscaling (Vulkan backend) | `/home/jtk1234/tools/` |
| **Intel Arc B580** | Primary GPU (Linux server, 12GB VRAM) | Linux |
| **RTX 3060 Ti** | Secondary GPU (Windows, 8GB VRAM) | Windows |

---

## New Architecture: Processing Agent Module

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PPP PROCESSING AGENT v2.0                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   WATCHER       │────▶│   ANALYZER      │────▶│   ROUTER        │
│                 │     │                 │     │                 │
│ • Library scan  │     │ • Resolution    │     │ • Model select  │
│ • New content   │     │ • VR detection  │     │ • Local/Cloud   │
│ • Queue jobs    │     │ • Quality score │     │ • Cost estimate │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   QA VALIDATOR  │◀────│   PROCESSOR     │◀────│   DISPATCHER    │
│                 │     │                 │     │                 │
│ • 15-sec sample │     │ • Frame extract │     │ • Job queue     │
│ • Quality check │     │ • AI upscale    │     │ • Worker assign │
│ • Human review  │     │ • Encode output │     │ • Progress track│
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              WORKER POOL                                        │
├────────────────────┬────────────────────┬────────────────────────────────────────┤
│   LOCAL LINUX      │   LOCAL WINDOWS    │   CLOUD (RunPod)                      │
│   Arc B580         │   RTX 3060 Ti      │   A100/4090                           │
│                    │                    │                                        │
│ • 4K→6K VR         │ • 720p→1080p 2D    │ • 4K→8K (high quality)                │
│ • 1080p→4K 2D      │ • Light jobs       │ • 6K→8K VR                            │
│ • Cost: $0         │ • Cost: $0         │ • Cost: $0.50-2.00/hr                 │
└────────────────────┴────────────────────┴────────────────────────────────────────┘
```

---

## AI Model Selection Matrix

### Model Recommendations by Use Case

| Source Resolution | Target | Content Type | Recommended Model | GPU Tier | Notes |
|-------------------|--------|--------------|-------------------|----------|-------|
| 720p | 1080p | 2D Flat | **Lanczos (FFmpeg)** | Any | AI overkill for 1.5x |
| 720p | 1440p | 2D Flat | Real-ESRGAN x2 | Local | Good ROI |
| 720p | 4K | 2D Flat | Real-ESRGAN x4 | Local | Significant improvement |
| 1080p | 4K | 2D Flat | Real-ESRGAN x2 | Local | Sweet spot |
| 1080p | 4K | VR SBS | Real-ESRGAN x2 | Local | Split/process/merge |
| 4K | 6K | VR SBS | **RealESRGAN-animevideo-v3** | Local | Best quality/speed balance |
| 4K | 8K | VR SBS | **ESRGAN x4plus** | Cloud A100 | VRAM hungry, slow locally |
| 5K/6K | 8K | VR SBS | **ESRGAN x4plus** | Cloud A100 | Definitely cloud-tier |

### Model Files Required

```bash
# Core models (download to /models/)
RealESRGAN_x4plus.pth           # General purpose 4x (best quality)
RealESRGAN_x4plus_anime_6B.pth  # Anime-optimized 4x
realesr-animevideov3.pth        # Video-optimized (faster)
realesr-general-x4v3.pth        # General video 4x
RealESRGAN_x2plus.pth           # 2x upscale (faster, less VRAM)
```

### Decision Tree for Model Selection

```
IF target_resolution == "1080p" AND source_resolution == "720p":
    USE lanczos (skip AI)
    
ELIF content_type == "VR":
    IF target >= "8K":
        ROUTE_TO cloud (A100/4090)
        USE RealESRGAN_x4plus
    ELSE:
        USE realesr-animevideov3 (faster, good quality)
        PROCESS locally
        
ELIF content_type == "2D":
    IF scale_factor <= 2:
        USE RealESRGAN_x2plus
    ELSE:
        USE RealESRGAN_x4plus
```

---

## Cloud Processing Recommendations

### When to Use Cloud (RunPod/Vast.ai)

| Scenario | Recommendation | Est. Cost | Est. Time |
|----------|----------------|-----------|-----------|
| 4K→8K VR (30-min) | **Use Cloud A100** | $3-6 | 2-3 hours |
| 6K→8K VR (30-min) | **Use Cloud A100** | $4-8 | 3-4 hours |
| Batch 10+ 8K jobs | **Use Cloud 4090** | $0.50/hr | Overnight |
| 4K→6K VR (any) | **Local Arc B580** | $0 | 4-6 hours |
| 1080p→4K 2D (any) | **Local Arc B580** | $0 | 1-2 hours |

### RunPod Template (Pre-configured)

```dockerfile
# PPP-Worker-RunPod
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ffmpeg python3-pip git vulkan-tools

RUN pip3 install torch torchvision realesrgan celery redis

COPY worker.py /app/
COPY models/ /app/models/

ENV CELERY_BROKER_URL=redis://your-redis-server:6379
WORKDIR /app
CMD ["celery", "-A", "worker", "worker", "--loglevel=info"]
```

---

## QA Workflow: 15-Second Sample Process

### Sample Extraction Logic

```python
def extract_qa_sample(input_path: str, output_dir: str) -> str:
    """
    Extract 15-second sample from middle of video for QA.
    Returns path to sample file.
    """
    # Get video duration
    duration = get_duration(input_path)
    
    # Start sample at 40% mark (avoid intros/outros)
    start_time = duration * 0.4
    
    # Extract 15 seconds
    sample_path = f"{output_dir}/qa_sample_{uuid4()}.mp4"
    
    ffmpeg_cmd = [
        'ffmpeg', '-ss', str(start_time),
        '-i', input_path,
        '-t', '15',
        '-c:v', 'copy', '-c:a', 'copy',
        sample_path
    ]
    
    return sample_path
```

### QA Validation States

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PENDING   │────▶│  SAMPLING   │────▶│  UPSCALING  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  REJECTED   │◀────│   REVIEW    │◀────│  QA_READY   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                    │
      │                    ▼
      │             ┌─────────────┐
      │             │  APPROVED   │───▶ FULL PROCESSING
      │             └─────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  RETRY with different model/settings                │
│  OR skip file (mark as unsuitable)                  │
└─────────────────────────────────────────────────────┘
```

### Quality Metrics for Auto-Approval

```python
QA_THRESHOLDS = {
    'min_ssim': 0.85,          # Structural similarity
    'min_psnr': 28.0,          # Peak signal-to-noise ratio
    'max_artifact_score': 0.3, # AI artifact detection
    'min_sharpness': 0.6,      # Laplacian variance normalized
}

def auto_approve(sample_original: str, sample_upscaled: str) -> bool:
    """
    Automatically approve if quality metrics pass.
    Returns False if human review needed.
    """
    metrics = calculate_metrics(sample_original, sample_upscaled)
    
    if metrics['ssim'] >= QA_THRESHOLDS['min_ssim'] and \
       metrics['psnr'] >= QA_THRESHOLDS['min_psnr'] and \
       metrics['artifact_score'] <= QA_THRESHOLDS['max_artifact_score']:
        return True
    
    return False  # Route to human review
```

---

## Docker Container Architecture

### Directory Structure

```
ppp-processor/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── CLAUDE.md                    # Agent instructions
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   ├── watcher.py               # Library scanner
│   ├── analyzer.py              # Content analysis
│   ├── router.py                # Model/worker selection
│   ├── processor.py             # Upscaling pipeline
│   ├── qa_validator.py          # QA sampling/approval
│   └── workers/
│       ├── local_worker.py      # Local GPU worker
│       └── cloud_worker.py      # RunPod integration
├── models/                      # AI models
├── config/
│   └── settings.yaml
├── tests/
│   └── test_pipeline.py
└── scripts/
    └── install_models.sh
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  ppp-agent:
    build: .
    container_name: ppp-processor
    restart: unless-stopped
    environment:
      - LIBRARY_PATH=/media
      - OUTPUT_PATH=/output
      - STASH_URL=http://host.docker.internal:9999/graphql
      - REDIS_URL=redis://redis:6379
      - GPU_DEVICE=0
    volumes:
      - /home/jtk1234/media-drive1/Media:/media:ro
      - /home/jtk1234/media-output:/output
      - ./models:/app/models
      - ./config:/app/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis

  redis:
    image: redis:alpine
    container_name: ppp-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data

  dashboard:
    build: ./dashboard
    container_name: ppp-dashboard
    ports:
      - "3000:3000"
    environment:
      - API_URL=http://ppp-agent:8000

volumes:
  redis_data:
```

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3-pip \
    python3-dev \
    git \
    vulkan-tools \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Real-ESRGAN NCNN Vulkan (precompiled binary)
RUN wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip \
    && unzip realesrgan-ncnn-vulkan*.zip -d /app/bin/ \
    && chmod +x /app/bin/realesrgan-ncnn-vulkan \
    && rm realesrgan-ncnn-vulkan*.zip

# Application code
COPY src/ /app/src/
COPY config/ /app/config/
COPY CLAUDE.md /app/

WORKDIR /app
ENV PATH="/app/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "src/main.py"]
```

---

## Claude Code Implementation Prompts

### CLAUDE.md (Project Context File)

```markdown
# PPP Processor - Claude Code Context

## Project Overview
Automated video upscaling pipeline for VR and 2D adult content. 
Processes content from Stash-managed library to Jellyfin for Meta Quest 3S streaming.

## Architecture
- **Backend**: FastAPI + Celery + Redis
- **Processing**: Real-ESRGAN-ncnn-vulkan (Vulkan GPU acceleration)
- **Database**: SQLite for job tracking, Stash GraphQL for metadata
- **Frontend**: React dashboard (optional)

## Key Paths
- Library: /media (mounted from host)
- Output: /output (mounted from host)
- Models: /app/models/
- Logs: /app/logs/

## GPU Configuration
- Linux (Arc B580): Use Vulkan backend, tile size 512
- Windows (RTX 3060 Ti): Use NCNN Vulkan, tile size 400
- Cloud (A100/4090): Use PyTorch CUDA, tile size 800

## VR Detection Rules
File is VR if ANY of these conditions:
- Filename contains: _180, _vr, _sbs, _LR, _TB, _6K, _8K
- Stash has VR tag (ID: 299)
- Resolution is stereoscopic (width > 2x height)

## Model Selection Rules
- 720p→1080p (2D): Use lanczos (skip AI)
- 720p→4K (2D): Real-ESRGAN x4plus
- 1080p→4K (2D/VR): Real-ESRGAN x2plus
- 4K→6K (VR): realesr-animevideov3
- 4K→8K or 6K→8K (VR): Route to cloud worker

## Safety Rules
- NEVER delete source files
- Always verify output before marking complete
- Keep 15-second QA samples for 7 days
- Log all processing decisions

## Commands
- Start processor: `python3 src/main.py`
- Run single job: `python3 src/main.py --job-id <id>`
- Scan library: `python3 src/watcher.py --scan`
- Check status: `curl http://localhost:8000/status`

## Testing
- Use 30-second test clips before full videos
- Verify VR metadata preserved after processing
- Check output plays correctly in Heresphere/DeoVR
```

---

### Phase 1: Core Infrastructure

**Claude Code Prompt:**

```
/plan Build the PPP Processor core infrastructure with the following requirements:

1. Create FastAPI backend with these endpoints:
   - POST /jobs - Submit new processing job
   - GET /jobs/{id} - Get job status
   - GET /jobs - List all jobs with filtering
   - GET /health - Health check
   - GET /status - System status (GPU, queue depth)

2. Set up Celery with Redis for job queue:
   - Worker registration/heartbeat
   - Job priority levels (high/medium/low)
   - Retry logic with exponential backoff
   - Dead letter queue for failed jobs

3. Create SQLite database schema:
   - jobs: id, source_path, target_path, status, model, created_at, updated_at
   - qa_samples: id, job_id, sample_path, metrics, approved, reviewed_at
   - workers: id, name, type (local/cloud), status, last_heartbeat

4. Implement basic logging to /app/logs/ with rotation

Requirements:
- Use Python 3.11+
- Type hints everywhere
- Pydantic models for validation
- Async where beneficial
- Docker-compatible (no hardcoded paths)

Do NOT implement the actual video processing yet - just the job management infrastructure.
After each component, run basic tests to verify functionality.
```

---

### Phase 2: Library Watcher & Analyzer

**Claude Code Prompt:**

```
/plan Build the Library Watcher and Content Analyzer modules:

1. Library Watcher (src/watcher.py):
   - Scan /media directory recursively for video files
   - Supported extensions: .mp4, .mkv, .avi, .mov, .webm
   - Skip hidden files and macOS metadata (._*)
   - Track file hashes to detect new/modified content
   - Run on schedule (configurable, default: every 6 hours)
   - Emit events for: NEW_FILE, MODIFIED_FILE, DELETED_FILE

2. Content Analyzer (src/analyzer.py):
   - Extract video metadata using ffprobe:
     * Resolution (width x height)
     * Duration
     * Codec
     * Bitrate
     * Frame rate
   - VR detection logic:
     * Check filename patterns (_180, _vr, _sbs, etc.)
     * Check aspect ratio (width > 2x height = likely VR)
     * Query Stash GraphQL for VR tag
   - Calculate quality score (0-100):
     * Based on: resolution, bitrate, codec age
   - Determine if upscaling beneficial:
     * Score < 60 AND resolution < target = recommend upscale

3. Integration:
   - Watcher detects new file → Analyzer processes → Creates job if needed
   - Store analysis results in database
   - Expose endpoint: GET /library/scan (manual trigger)
   - Expose endpoint: GET /library/stats (file counts by type/quality)

Test with small subset of files before full library scan.
```

---

### Phase 3: Router & Model Selection

**Claude Code Prompt:**

```
/plan Build the intelligent Router module for model and worker selection:

1. Router Logic (src/router.py):
   - Input: analyzed file metadata
   - Output: processing plan (model, worker, settings, cost estimate)

2. Model Selection Rules (implement as config-driven):
   ```yaml
   rules:
     - name: "Skip AI for small upscales"
       condition: "scale_factor <= 1.5 AND content_type == '2D'"
       action: "use_lanczos"
       
     - name: "4K to 8K VR - Cloud required"
       condition: "source_res >= 4K AND target_res == 8K AND content_type == 'VR'"
       action: "route_cloud"
       model: "RealESRGAN_x4plus"
       
     - name: "4K to 6K VR - Local preferred"
       condition: "source_res >= 4K AND target_res == 6K AND content_type == 'VR'"
       action: "route_local"
       model: "realesr-animevideov3"
   ```

3. Worker Selection:
   - Check local GPU availability (query nvidia-smi / vulkaninfo)
   - Check cloud worker status (if configured)
   - Estimate processing time based on:
     * File duration
     * Source/target resolution
     * Model complexity
     * GPU performance (stored benchmarks)
   - Return cost estimate for cloud jobs

4. VR Processing Plan:
   - For SBS: plan split → process_left → process_right → merge
   - For TB: similar but top/bottom split
   - Preserve audio track (copy, don't re-encode)
   - Preserve VR metadata (projection type, stereo mode)

5. Expose endpoint: POST /route (preview routing decision without creating job)

Include comprehensive unit tests for edge cases.
```

---

### Phase 4: QA Sampling System

**Claude Code Prompt:**

```
/plan Build the QA Sampling and Validation system:

1. Sample Extraction (src/qa_validator.py):
   - Extract 15-second sample from video at 40% mark
   - Use stream copy (no re-encoding) for speed
   - Store samples in /output/qa_samples/
   - Naming: {job_id}_original.mp4, {job_id}_upscaled.mp4

2. Quality Metrics Calculation:
   - SSIM (Structural Similarity Index) - target >= 0.85
   - PSNR (Peak Signal-to-Noise Ratio) - target >= 28 dB
   - Sharpness (Laplacian variance) - should increase
   - Artifact detection (optional, ML-based)
   
3. Auto-Approval Logic:
   - If all metrics pass thresholds → auto-approve
   - If borderline (within 10% of thresholds) → flag for review
   - If clearly failing → auto-reject with reason

4. Human Review Interface:
   - Endpoint: GET /qa/pending - list samples awaiting review
   - Endpoint: POST /qa/{job_id}/approve - approve sample
   - Endpoint: POST /qa/{job_id}/reject - reject with reason
   - Endpoint: GET /qa/{job_id}/compare - get side-by-side paths

5. Workflow Integration:
   - Job status: SAMPLING → SAMPLE_READY → (APPROVED|REJECTED|REVIEWING)
   - On approve → continue to full processing
   - On reject → option to retry with different model/settings

6. Cleanup:
   - Auto-delete approved samples after 7 days
   - Keep rejected samples for 30 days (debugging)

Create test with actual video sample to verify metrics calculation.
```

---

### Phase 5: Processing Pipeline

**Claude Code Prompt:**

```
/plan Build the core video processing pipeline:

1. Frame Extraction (using FFmpeg):
   ```python
   def extract_frames(input_path: str, output_dir: str, fps: int = None):
       """
       Extract frames as PNG for AI processing.
       If fps not specified, use source fps.
       For VR: split SBS/TB first, process separately.
       """
   ```

2. AI Upscaling:
   - Wrapper for realesrgan-ncnn-vulkan binary
   - Support for all model variants
   - Configurable tile size based on GPU VRAM
   - Progress tracking (frames processed / total frames)
   - Error handling for OOM (reduce tile size, retry)

3. Frame Encoding:
   - Reassemble frames to video using FFmpeg
   - Use hardware encoding:
     * Intel: QSV (hevc_qsv)
     * NVIDIA: NVENC (hevc_nvenc)
   - Target bitrate based on resolution:
     * 6K VR: 80-100 Mbps
     * 8K VR: 120-150 Mbps
     * 4K 2D: 25-35 Mbps
   - Preserve audio (copy, not re-encode)

4. VR-Specific Processing:
   ```python
   def process_vr_sbs(input_path: str, output_path: str, model: str):
       """
       1. Extract left eye (crop left half)
       2. Extract right eye (crop right half)
       3. Upscale left with AI model
       4. Upscale right with AI model
       5. Stitch back to SBS format
       6. Encode with proper VR metadata
       """
   ```

5. Metadata Preservation:
   - Copy over VR projection metadata
   - Preserve chapter markers if present
   - Update resolution in container metadata

6. Progress Reporting:
   - WebSocket endpoint for real-time progress
   - Update job status at each stage
   - Estimate remaining time based on progress

7. Output Verification:
   - Verify output file exists and is valid
   - Check duration matches source
   - Verify resolution is as expected
   - Run quick playback test (first 5 frames)

Start with 2D processing, then add VR support.
Test with 30-second clip before full implementation.
```

---

### Phase 6: Cloud Worker Integration

**Claude Code Prompt:**

```
/plan Build RunPod cloud worker integration:

1. RunPod API Client (src/workers/cloud_worker.py):
   - Create pod from template
   - Monitor pod status
   - Execute processing job
   - Retrieve results
   - Terminate pod when done

2. File Transfer:
   - Upload source file to pod (via SCP or S3-compatible)
   - Download result from pod
   - Verify integrity with checksums

3. Cost Tracking:
   - Query RunPod API for pricing
   - Track GPU hours per job
   - Generate monthly cost reports

4. Job Execution Flow:
   ```
   1. Spin up pod with PPP worker image
   2. Wait for pod ready (healthcheck)
   3. Upload source file
   4. Trigger processing via API
   5. Poll for completion
   6. Download result
   7. Verify output
   8. Terminate pod
   9. Update job status
   ```

5. Fallback Logic:
   - If cloud unavailable → queue for local processing
   - If upload fails → retry with exponential backoff
   - If pod crashes → auto-restart with fresh pod

6. Configuration:
   ```yaml
   runpod:
     api_key: ${RUNPOD_API_KEY}
     template_id: "ppp-worker-v2"
     preferred_gpu: "NVIDIA A100"
     fallback_gpu: "NVIDIA RTX 4090"
     max_cost_per_job: 10.00
     timeout_minutes: 180
   ```

7. Endpoints:
   - GET /workers/cloud/status - Cloud worker availability
   - POST /workers/cloud/test - Run test job on cloud

Mock the RunPod API for testing, then test with real API using short job.
```

---

### Phase 7: Dashboard & Agent Interface

**Claude Code Prompt:**

```
/plan Build the monitoring dashboard and agent interface:

1. React Dashboard (dashboard/):
   - Job queue view (pending, processing, completed, failed)
   - Real-time progress for active jobs
   - QA review interface (side-by-side comparison)
   - Worker status (local GPUs, cloud pods)
   - Cost tracking and estimates
   - Library statistics

2. Agent API Endpoints:
   - POST /agent/decide - Submit file for routing decision
   - POST /agent/process - Submit file for full processing
   - GET /agent/recommendations - Get processing recommendations for library
   - POST /agent/batch - Submit batch job

3. Webhook Notifications:
   - Job completed
   - Job failed
   - QA review needed
   - Cost threshold exceeded

4. CLI Interface:
   ```bash
   ppp scan              # Scan library for new content
   ppp status            # Show system status
   ppp queue             # Show job queue
   ppp process <file>    # Process single file
   ppp batch <dir>       # Process directory
   ppp qa                # Show pending QA items
   ```

5. Scheduling:
   - Configurable processing windows (e.g., overnight only)
   - Power-aware (pause if system busy)
   - Queue prioritization

Use Vite + React + Tailwind for dashboard.
API should work headless (dashboard optional).
```

---

## Git Workflow

### Repository Structure

```
github.com/jtk1234/ppp-processor/
├── .github/
│   └── workflows/
│       ├── test.yml        # Run tests on PR
│       └── build.yml       # Build Docker image
├── src/                    # Application code
├── dashboard/              # React frontend
├── tests/                  # Test suite
├── config/                 # Configuration files
├── docs/                   # Documentation
├── CLAUDE.md               # Agent context
├── README.md
├── docker-compose.yml
└── Dockerfile
```

### Branch Strategy

```
main          ← Production-ready code
  └── develop ← Integration branch
        ├── feature/core-infrastructure
        ├── feature/library-watcher
        ├── feature/qa-system
        └── feature/cloud-worker
```

### Commit Convention

```
feat(watcher): add library scanning with hash tracking
fix(processor): handle OOM by reducing tile size
docs(readme): update installation instructions
test(qa): add metrics calculation tests
```

---

## Getting Started

### 1. Initialize Repository

```bash
mkdir ppp-processor && cd ppp-processor
git init
git remote add origin git@github.com:jtk1234/ppp-processor.git
```

### 2. Start Claude Code Session

```bash
cd ppp-processor
claude
/init
```

### 3. Copy CLAUDE.md Content

Paste the CLAUDE.md content from this document into the initialized file.

### 4. Begin Phase 1

```
Please review CLAUDE.md and begin implementing Phase 1: Core Infrastructure.
Start with the FastAPI backend and basic endpoint structure.
After completing each component, run tests to verify.
Commit working code to feature/core-infrastructure branch.
```

---

## Success Criteria

The PPP Processor is complete when:

1. ✅ New content in library auto-detected within 6 hours
2. ✅ Correct model selected based on content type/resolution
3. ✅ 15-second QA sample generated before full processing
4. ✅ Auto-approval works for quality content
5. ✅ Cloud routing works for 8K jobs
6. ✅ VR metadata preserved after processing
7. ✅ Output plays correctly in Heresphere on Quest 3S
8. ✅ Processing can run autonomously overnight
9. ✅ Dashboard shows real-time status
10. ✅ Cost tracking accurate within 10%

---

## Appendix: Quick Reference Commands

```bash
# Docker commands
docker-compose up -d                    # Start all services
docker-compose logs -f ppp-agent        # Watch agent logs
docker-compose exec ppp-agent bash      # Shell into container

# API commands
curl http://localhost:8000/health       # Health check
curl http://localhost:8000/status       # System status
curl -X POST http://localhost:8000/library/scan  # Trigger scan

# Debug commands
nvidia-smi                              # GPU status
vulkaninfo | grep -i gpu               # Vulkan GPU info
docker stats                            # Container resources
```
