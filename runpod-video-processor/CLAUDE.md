# RunPod Video Processor

## Project Context
GPU-accelerated video processing app running on RunPod pods.
Two pipelines: upscaling (Real-ESRGAN) and background removal (RVM).
Web UI on port 8080. Files live on network volume at /workspace.

## Storage
- /workspace/input/   — source videos (20-30GB+, NEVER copy these)
- /workspace/output/  — processed results
- /workspace/temp/    — intermediate frames (auto-cleaned, can be 200GB+)
- All file I/O goes through src/storage/volume.py
- Always check disk space before starting any job

## GPU
- GPU type varies (4090/A40/A100/L40S) — ALWAYS detect at startup
- VRAM-adaptive settings in src/gpu.py
- OOM retry: reduce tile size 512→256→128 before failing

## Key Decisions
- Chunked processing: 1000 frames per segment, adaptive to disk space
- No S3 — everything via network volume
- MKV container for all outputs (best metadata + multi-audio support)
- H.265 NVENC encoding, fallback to libx265
- Web UI: Flask + vanilla JS + SSE, no build step
- SQLite for job queue (Phase 3)

## Testing
- `pytest` for unit tests (no GPU needed)
- `pytest -m integration` for GPU tests
- Test fixtures in tests/fixtures/ (generated via script)
- Mock GPU tests use --mock-gpu flag

## Common Pitfalls
- Never extract all frames at once — use chunked pipeline
- Always clean temp files in finally blocks
- VR SBS frames must be split per-eye before upscaling
- ffprobe is the source of truth for video metadata
- NVENC may not be in base image — check during Docker build
