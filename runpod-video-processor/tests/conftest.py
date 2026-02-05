"""Shared test fixtures and configuration."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: requires GPU and/or real FFmpeg")


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_metadata():
    """Standard video metadata dict for testing."""
    return {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "duration": 600.0,
        "codec": "h264",
        "num_frames": 18000,
        "file_size": 5 * 1024 ** 3,
    }


@pytest.fixture
def vr_sbs_metadata():
    """VR Side-by-Side video metadata."""
    return {
        "width": 3840,
        "height": 1920,
        "fps": 60.0,
        "duration": 300.0,
        "codec": "hevc",
        "num_frames": 18000,
        "file_size": 20 * 1024 ** 3,
    }


@pytest.fixture
def vr_ou_metadata():
    """VR Over-Under video metadata."""
    return {
        "width": 1920,
        "height": 3840,
        "fps": 60.0,
        "duration": 300.0,
        "codec": "hevc",
        "num_frames": 18000,
        "file_size": 20 * 1024 ** 3,
    }


@pytest.fixture
def mock_workspace(tmp_path):
    """Set up a temporary workspace structure."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir()
    output_dir.mkdir()
    temp_dir.mkdir()

    with patch("src.config.WORKSPACE_DIR", tmp_path), \
         patch("src.config.INPUT_DIR", input_dir), \
         patch("src.config.OUTPUT_DIR", output_dir), \
         patch("src.config.TEMP_DIR", temp_dir), \
         patch("src.storage.volume.INPUT_DIR", input_dir), \
         patch("src.storage.volume.OUTPUT_DIR", output_dir), \
         patch("src.storage.volume.TEMP_DIR", temp_dir):
        yield {
            "root": tmp_path,
            "input": input_dir,
            "output": output_dir,
            "temp": temp_dir,
        }
