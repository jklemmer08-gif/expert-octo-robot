"""Tests for configuration loading."""

from pathlib import Path

from src.config import Settings


def test_default_settings():
    """Settings can be created with defaults."""
    s = Settings()
    assert s.gpu.device_id == 0
    assert s.upscale.default_model == "realesr-animevideov3"
    assert s.redis.port == 6379
    assert s.qa.ssim_threshold == 0.85


def test_from_yaml():
    """Settings.from_yaml parses the real settings.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if not config_path.exists():
        return  # Skip if not in repo
    s = Settings.from_yaml(config_path)
    assert s.gpu.tile_size == 512
    assert s.encode.codec == "hevc"
    assert "Upscaled" in s.stash.tags
    assert s.runpod.budget_total == 75.0


def test_from_yaml_missing():
    """Settings.from_yaml with missing file returns defaults."""
    s = Settings.from_yaml(Path("/nonexistent/config.yaml"))
    assert s.gpu.device_id == 0


def test_redis_url():
    """Redis URL is generated correctly."""
    s = Settings()
    assert s.redis.url == "redis://localhost:6379/0"


def test_redis_url_with_password():
    from src.config import RedisConfig
    rc = RedisConfig(password="secret")
    assert rc.url == "redis://:secret@localhost:6379/0"


def test_vr_patterns():
    """VR patterns are loaded from config."""
    s = Settings()
    assert "_180" in s.vr_patterns
    assert "VRBangers" in s.vr_patterns


def test_qa_config_defaults():
    s = Settings()
    assert s.qa.sample_duration == 15
    assert s.qa.cleanup_approved_days == 7


def test_encode_nvenc_defaults():
    """NVENC-specific encode config fields have correct defaults."""
    s = Settings()
    assert s.encode.encoder == "hevc_nvenc"
    assert s.encode.fallback_encoder == "libx265"
    assert s.encode.qp == 20
    assert s.encode.nvenc_preset == "p5"
    assert s.encode.tune == "hq"
    assert s.encode.rc_mode == "constqp"
    assert s.encode.spatial_aq is True
    assert s.encode.temporal_aq is True
    assert s.encode.rc_lookahead == 32
    assert s.encode.profile == "main10"


def test_tensorrt_config_defaults():
    """TensorRT config fields have correct defaults."""
    s = Settings()
    assert s.tensorrt.enabled is False
    assert s.tensorrt.fp16 is True
    assert s.tensorrt.max_workspace_mb == 4096
    assert len(s.tensorrt.resolution_buckets) == 5
    bucket_names = [b["name"] for b in s.tensorrt.resolution_buckets]
    assert "720p" in bucket_names
    assert "1080p" in bucket_names
    assert "4k" in bucket_names
    assert "vr" in bucket_names


def test_tensorrt_config_in_settings():
    """TensorRT config is accessible from root Settings."""
    s = Settings()
    assert hasattr(s, "tensorrt")
    assert s.tensorrt.cache_dir == "./trt_engines"
