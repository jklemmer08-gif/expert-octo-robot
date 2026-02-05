"""Network volume file operations and disk space management for /workspace."""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from src.config import INPUT_DIR, OUTPUT_DIR, TEMP_DIR

logger = logging.getLogger(__name__)


def ensure_dirs():
    """Create the standard workspace directories if they don't exist."""
    for d in (INPUT_DIR, OUTPUT_DIR, TEMP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_disk_space(path: Optional[Path] = None) -> dict:
    """Return total/used/free disk space in GB for the given path."""
    target = str(path or TEMP_DIR)
    try:
        usage = shutil.disk_usage(target)
        return {
            "total_gb": round(usage.total / (1024 ** 3), 1),
            "used_gb": round(usage.used / (1024 ** 3), 1),
            "free_gb": round(usage.free / (1024 ** 3), 1),
        }
    except OSError as e:
        logger.error("disk_usage failed for %s: %s", target, e)
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0}


def check_disk_space(needed_gb: float, path: Optional[Path] = None) -> bool:
    """Return True if at least `needed_gb` is available."""
    space = get_disk_space(path)
    ok = space["free_gb"] >= needed_gb
    if not ok:
        logger.warning(
            "Insufficient disk space: need %.1f GB, have %.1f GB free",
            needed_gb,
            space["free_gb"],
        )
    return ok


def estimate_segment_disk_gb(
    width: int, height: int, num_frames: int, scale: int = 4
) -> float:
    """Estimate GB needed for one segment's frames on disk (input + upscaled PNGs)."""
    # PNG size ~= width * height * 3 bytes (uncompressed RGB) * ~0.7 compression
    input_frame_bytes = width * height * 3 * 0.7
    output_frame_bytes = (width * scale) * (height * scale) * 3 * 0.7
    total_bytes = (input_frame_bytes + output_frame_bytes) * num_frames
    return total_bytes / (1024 ** 3)


def list_input_files() -> List[dict]:
    """List video files in the input directory with basic metadata."""
    extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m4v"}
    files = []
    if not INPUT_DIR.exists():
        return files
    for entry in sorted(INPUT_DIR.iterdir()):
        if entry.is_file() and entry.suffix.lower() in extensions:
            stat = entry.stat()
            files.append({
                "name": entry.name,
                "path": str(entry),
                "size_bytes": stat.st_size,
                "size_gb": round(stat.st_size / (1024 ** 3), 2),
            })
    return files


def list_output_files() -> List[dict]:
    """List files in the output directory."""
    if not OUTPUT_DIR.exists():
        return []
    files = []
    for entry in sorted(OUTPUT_DIR.iterdir()):
        if entry.is_file():
            stat = entry.stat()
            files.append({
                "name": entry.name,
                "path": str(entry),
                "size_bytes": stat.st_size,
                "size_gb": round(stat.st_size / (1024 ** 3), 2),
            })
    return files


def create_segment_dir(job_id: str, segment_index: int) -> Path:
    """Create and return a temp directory for a processing segment."""
    seg_dir = TEMP_DIR / job_id / f"segment_{segment_index:04d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    return seg_dir


def cleanup_segment(segment_dir: Path):
    """Remove a segment's temp directory and all contents."""
    try:
        if segment_dir.exists():
            shutil.rmtree(segment_dir)
            logger.info("Cleaned up segment dir: %s", segment_dir)
    except OSError as e:
        logger.error("Failed to clean up %s: %s", segment_dir, e)


def cleanup_job(job_id: str):
    """Remove all temp files for a job."""
    job_dir = TEMP_DIR / job_id
    try:
        if job_dir.exists():
            shutil.rmtree(job_dir)
            logger.info("Cleaned up job dir: %s", job_dir)
    except OSError as e:
        logger.error("Failed to clean up job %s: %s", job_id, e)


def cleanup_orphaned_temp():
    """Remove any leftover temp directories from interrupted jobs (called at startup)."""
    if not TEMP_DIR.exists():
        return
    for entry in TEMP_DIR.iterdir():
        if entry.is_dir():
            logger.info("Removing orphaned temp directory: %s", entry)
            try:
                shutil.rmtree(entry)
            except OSError as e:
                logger.error("Failed to remove orphaned dir %s: %s", entry, e)
