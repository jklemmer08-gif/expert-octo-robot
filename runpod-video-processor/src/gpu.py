"""GPU/VRAM detection and adaptive processing profiles."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUProfile:
    """Processing parameters tuned for a specific VRAM tier."""
    name: str
    vram_gb: float
    tile_size: int
    batch_size: int
    segment_size: int  # frames per chunked segment
    rvm_batch_size: int = 8
    rvm_downsample_ratio: float = 0.5


# Profiles ordered by VRAM tier
PROFILES = {
    "A100-80GB": GPUProfile("A100-80GB", 80, 1024, 16, 2000, rvm_batch_size=16),
    "A100-40GB": GPUProfile("A100-40GB", 40, 768, 8, 1500, rvm_batch_size=12),
    "L40S-48GB": GPUProfile("L40S-48GB", 48, 768, 8, 1500, rvm_batch_size=12),
    "24GB":      GPUProfile("24GB", 24, 512, 4, 1000, rvm_batch_size=8),
    "16GB":      GPUProfile("16GB", 16, 384, 2, 500, rvm_batch_size=4, rvm_downsample_ratio=0.4),
    "fallback":  GPUProfile("fallback", 0, 256, 1, 300, rvm_batch_size=2, rvm_downsample_ratio=0.25),
}


def detect_gpu() -> dict:
    """Return GPU name and total VRAM in GB. Returns None values if no GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"name": None, "vram_gb": 0, "count": 0}
        name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_mem
        vram_gb = round(vram_bytes / (1024 ** 3), 1)
        count = torch.cuda.device_count()
        return {"name": name, "vram_gb": vram_gb, "count": count}
    except Exception as e:
        logger.warning("GPU detection failed: %s", e)
        return {"name": None, "vram_gb": 0, "count": 0}


def select_profile(vram_gb: float, gpu_name: Optional[str] = None) -> GPUProfile:
    """Pick the best processing profile for the available VRAM."""
    if gpu_name:
        # Check for exact name match first
        if "A100" in gpu_name and vram_gb >= 70:
            return PROFILES["A100-80GB"]
        if "A100" in gpu_name:
            return PROFILES["A100-40GB"]
        if "L40S" in gpu_name or "L40" in gpu_name:
            return PROFILES["L40S-48GB"]

    # Fall back to VRAM-based selection
    if vram_gb >= 70:
        return PROFILES["A100-80GB"]
    if vram_gb >= 40:
        return PROFILES["L40S-48GB"]
    if vram_gb >= 20:
        return PROFILES["24GB"]
    if vram_gb >= 12:
        return PROFILES["16GB"]
    return PROFILES["fallback"]


def get_gpu_profile() -> GPUProfile:
    """Detect GPU and return the appropriate processing profile."""
    info = detect_gpu()
    profile = select_profile(info["vram_gb"], info["name"])
    logger.info(
        "GPU: %s | VRAM: %.1f GB | Profile: %s (tile=%d, batch=%d, segment=%d)",
        info["name"] or "none",
        info["vram_gb"],
        profile.name,
        profile.tile_size,
        profile.batch_size,
        profile.segment_size,
    )
    return profile


def get_vram_usage() -> dict:
    """Return current VRAM usage stats (for the system info panel)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0}
        total = torch.cuda.get_device_properties(0).total_mem
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        return {
            "total_gb": round(total / (1024 ** 3), 1),
            "used_gb": round(allocated / (1024 ** 3), 1),
            "reserved_gb": round(reserved / (1024 ** 3), 1),
            "free_gb": round((total - reserved) / (1024 ** 3), 1),
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0}
