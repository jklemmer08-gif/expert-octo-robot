"""Configuration loaded from environment variables with sensible defaults."""

import os
from pathlib import Path

# --- Workspace Paths ---
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
INPUT_DIR = Path(os.getenv("INPUT_DIR", str(WORKSPACE_DIR / "input")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(WORKSPACE_DIR / "output")))
TEMP_DIR = Path(os.getenv("TEMP_DIR", str(WORKSPACE_DIR / "temp")))

# --- Processing Limits ---
MAX_DURATION_MINUTES = int(os.getenv("MAX_DURATION_MINUTES", "180"))
MAX_FILE_SIZE_GB = float(os.getenv("MAX_FILE_SIZE_GB", "50"))

# --- Encoding Defaults ---
DEFAULT_CRF = int(os.getenv("DEFAULT_CRF", "18"))
DEFAULT_CODEC = os.getenv("DEFAULT_CODEC", "hevc_nvenc")
DEFAULT_PRESET = os.getenv("DEFAULT_PRESET", "slow")
FALLBACK_CODEC = "libx265"

# --- Upscaler ---
DEFAULT_SCALE = int(os.getenv("DEFAULT_SCALE", "4"))
DEFAULT_TILE_SIZE = int(os.getenv("DEFAULT_TILE_SIZE", "512"))
TILE_RETRY_SIZES = [512, 384, 256, 128]
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "RealESRGAN_x4plus")
SEGMENT_SIZE = int(os.getenv("SEGMENT_SIZE", "1000"))

# --- Device ---
DEVICE = os.getenv("DEVICE", "cuda:0")
GPU_ID = int(os.getenv("GPU_ID", "0"))

# --- Web UI ---
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))

# --- Supported Codecs ---
SUPPORTED_CODECS = {"h264", "hevc", "h265", "av1", "vp9"}

# --- Supported Models ---
AVAILABLE_MODELS = {
    "RealESRGAN_x4plus": {
        "file": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "description": "General content (live action)",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "description": "Animation / anime",
    },
    "RealESRGAN_x2plus": {
        "file": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "description": "2x upscale (general)",
    },
}
