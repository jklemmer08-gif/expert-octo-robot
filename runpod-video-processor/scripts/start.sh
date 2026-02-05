#!/usr/bin/env bash
# Entry point for RunPod pod — ensures workspace dirs exist, cleans orphans, starts app.
set -euo pipefail

echo "=== RunPod Video Processor ==="

# Ensure workspace directories
mkdir -p /workspace/input /workspace/output /workspace/temp

# Show GPU info
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_mem / 1e9:.1f} GB VRAM)')
else:
    print('WARNING: No CUDA GPUs detected')
" 2>/dev/null || echo "GPU detection failed"

# Show disk space
echo "Disk space:"
df -h /workspace 2>/dev/null || echo "  /workspace not mounted"

echo ""
echo "Starting web UI on port 8080..."
exec python3 -u /app/src/app.py
