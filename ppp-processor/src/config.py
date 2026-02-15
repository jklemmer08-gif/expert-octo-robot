"""Pydantic Settings configuration for PPP Processor.

Parses the existing config/settings.yaml and adds new sections for
Redis, watcher, and QA settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env from project root (next to config/)
load_dotenv(Path(__file__).parent.parent / ".env")


class PathsConfig(BaseSettings):
    stash_url: str = "http://localhost:9999/graphql"
    stash_api_key: str = ""
    library_root: str = "/home/jtk1234/media-drive1/Media"
    output_dir: str = "/mnt/ppp-work/ppp/output"
    temp_dir: str = "/mnt/ppp-work/ppp/temp"
    models_dir: str = "./bin/models"
    realesrgan_bin: str = "./bin/realesrgan-ncnn-vulkan"
    analysis_dir: str = "../ppp_analysis"
    local_cache_dir: str = ""  # Local SSD cache for network files (empty = disabled)


class GPUConfig(BaseSettings):
    device_id: int = 0
    tile_size: int = 512


class UpscaleConfig(BaseSettings):
    default_model: str = "realesr-animevideov3"
    priority_model: str = "realesrgan-x4plus"
    scale_factor: int = 2


class EncodeConfig(BaseSettings):
    codec: str = "hevc"
    encoder: str = "hevc_nvenc"
    fallback_encoder: str = "libx265"
    bitrates: Dict[str, str] = Field(default_factory=lambda: {
        "8K": "150M", "6K": "100M", "5K": "80M", "4K": "50M", "1080p": "15M",
    })
    crf: int = 20
    preset: str = "medium"
    # NVENC-specific params
    qp: int = 20
    nvenc_preset: str = "p5"
    tune: str = "hq"
    rc_mode: str = "constqp"
    spatial_aq: bool = True
    temporal_aq: bool = True
    rc_lookahead: int = 32
    profile: str = "main10"
    # VAAPI (Intel) params
    vaapi_device: str = "/dev/dri/renderD128"
    vaapi_qp: int = 20


class TensorRTConfig(BaseSettings):
    """TensorRT inference settings for RVM."""
    enabled: bool = False
    cache_dir: str = "./trt_engines"
    fp16: bool = True
    max_workspace_mb: int = 4096
    resolution_buckets: List[Dict[str, Any]] = Field(default_factory=lambda: [
        {"name": "720p", "min_h": 360, "min_w": 640, "opt_h": 720, "opt_w": 1280, "max_h": 768, "max_w": 1408},
        {"name": "1080p", "min_h": 720, "min_w": 1280, "opt_h": 1080, "opt_w": 1920, "max_h": 1152, "max_w": 2048},
        {"name": "1440p", "min_h": 1080, "min_w": 1920, "opt_h": 1440, "opt_w": 2560, "max_h": 1536, "max_w": 2816},
        {"name": "4k", "min_h": 1440, "min_w": 2560, "opt_h": 2160, "opt_w": 3840, "max_h": 2304, "max_w": 4096},
        {"name": "vr", "min_h": 1440, "min_w": 1920, "opt_h": 2160, "opt_w": 2160, "max_h": 2880, "max_w": 2880},
    ])


class StashConfig(BaseSettings):
    url: str = "http://localhost:9999/graphql"
    api_key: str = ""
    tags: List[str] = Field(default_factory=lambda: ["Upscaled", "PPP-Processed"])


class JellyfinConfig(BaseSettings):
    url: str = "http://localhost:8096"
    api_key: str = ""
    library_path: str = ""
    organize_by: Dict[str, str] = Field(default_factory=lambda: {
        "vr": "VR/", "studio": "Studios/", "default": "Unsorted/",
    })


class S3Config(BaseSettings):
    """S3 storage for cloud worker file staging."""
    access_key_id: str = ""
    secret_access_key: str = ""
    bucket: str = "ppp-processor-media"
    region: str = "us-east-2"


class CloudConfig(BaseSettings):
    """Cloud GPU processing settings."""
    primary_provider: str = "runpod"  # "runpod" or "vastai"
    docker_image: str = "ppp-processor/cloud-worker:latest"
    setup_timeout: int = 600  # seconds to wait for pod ready
    processing_timeout: int = 14400  # 4 hours max per job
    budget_total: float = 50.00


class RunPodConfig(BaseSettings):
    api_key: str = ""
    template_id: str = ""
    preferred_gpu: str = "NVIDIA RTX 4090"
    fallback_gpu: str = "NVIDIA RTX 3090"
    max_cost_per_job: float = 5.00
    budget_total: float = 75.00
    serverless_endpoint_id: str = ""
    s3_bucket: str = "ppp-processor-media"
    s3_region: str = "us-east-2"


class RedisConfig(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class WatcherConfig(BaseSettings):
    enabled: bool = False
    scan_interval_hours: float = 6.0
    scan_paths: List[str] = Field(default_factory=list)


class MatteConfig(BaseSettings):
    """Background matting / chroma key settings."""
    model_type: str = "mobilenetv3"  # "mobilenetv3" or "resnet50"
    downsample_ratio: float = 0.25  # Lower = faster, less accurate edges
    green_color: List[int] = Field(default_factory=lambda: [0, 177, 64])  # Heresphere default
    output_type: str = "green_screen"  # "green_screen", "alpha_matte", or "alpha_pack"
    use_streaming: bool = True       # FFmpeg pipe streaming (False = legacy disk path)
    fp16: bool = True                # FP16 mixed precision on CUDA
    seq_chunk: int = 1               # Frames per forward pass (future batching)
    encode_bitrate: str = "15M"      # 2D matting output bitrate
    vr_encode_bitrate: str = "50M"   # VR matting output bitrate
    progress_interval: int = 100     # Report progress every N frames
    refine_alpha: bool = False       # Edge refinement via guided filter + morph close
    despill: bool = False            # Green spill removal on foreground edges
    despill_strength: float = 0.8    # Despill intensity (0.0-1.0)
    alpha_sharpness: str = "fine"    # Refine mode: "fine" (multi-scale) or "soft" (legacy)
    output_scale: float = 1.0       # Downscale output (0.75 = 75% res, faster; 1.0 = full res)
    channels_last: bool = True       # NHWC memory format for tensor core acceleration
    cuda_graphs_pytorch: bool = True # CUDA Graphs for PyTorch fallback (Windows)
    # Intel/OpenVINO settings
    openvino_device: str = "GPU"             # OpenVINO target device ("GPU", "CPU")
    onnx_model_path: str = ""                # Explicit ONNX path (empty = auto-detect)
    alpha_pack_scale: float = 0.4            # Matte downscale factor for corner packing
    vaapi_device: str = "/dev/dri/renderD128"  # VAAPI render node for decode/encode


class QAConfig(BaseSettings):
    sample_duration: int = 15
    sample_start_percent: float = 0.4
    ssim_threshold: float = 0.85
    psnr_threshold: float = 28.0
    auto_approve_tolerance: float = 0.10
    cleanup_approved_days: int = 7
    cleanup_rejected_days: int = 30


class Settings(BaseSettings):
    """Root settings class that aggregates all configuration."""

    model_config = {"extra": "ignore"}

    paths: PathsConfig = Field(default_factory=PathsConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    upscale: UpscaleConfig = Field(default_factory=UpscaleConfig)
    encode: EncodeConfig = Field(default_factory=EncodeConfig)
    vr_patterns: List[str] = Field(default_factory=lambda: [
        "_180", "_360", "_vr", "_VR", "_sbs", "_SBS",
        "_LR", "_TB", "_3dh", "_6k", "_6K", "_7k",
        "_8k", "_8K", "_fisheye", "POVR", "VRBangers",
        "WankzVR", "SLR", "VirtualReal",
    ])
    stash: StashConfig = Field(default_factory=StashConfig)
    jellyfin: JellyfinConfig = Field(default_factory=JellyfinConfig)
    runpod: RunPodConfig = Field(default_factory=RunPodConfig)
    s3: S3Config = Field(default_factory=S3Config)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    matte: MatteConfig = Field(default_factory=MatteConfig)
    tensorrt: TensorRTConfig = Field(default_factory=TensorRTConfig)
    tiers: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from a YAML file, falling back to defaults."""
        if config_path is None:
            import os
            env_path = os.environ.get("PPP_CONFIG")
            if env_path:
                config_path = Path(env_path)
            else:
                config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        return cls(**raw)


# Module-level singleton (lazy-loaded)
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[Path] = None) -> Settings:
    """Get or create the global Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml(config_path)
    return _settings
