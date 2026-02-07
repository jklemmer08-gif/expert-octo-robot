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
