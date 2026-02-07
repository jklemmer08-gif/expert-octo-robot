"""Shared fixtures for PPP Processor tests."""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.config import Settings
from src.database import JobDatabase


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def test_settings(tmp_path):
    """Create Settings with test-friendly paths."""
    return Settings(
        paths=Settings.model_fields["paths"].default_factory().__class__(
            output_dir=str(tmp_path / "output"),
            temp_dir=str(tmp_path / "temp"),
            library_root=str(tmp_path / "library"),
            models_dir=str(tmp_path / "models"),
            realesrgan_bin="/usr/bin/false",  # won't be called in unit tests
            analysis_dir=str(tmp_path / "analysis"),
        ),
        gpu=Settings.model_fields["gpu"].default_factory().__class__(
            device_id=0, tile_size=256,
        ),
    )


@pytest.fixture
def test_db(tmp_path):
    """Create an in-memory-style SQLite database for tests."""
    db_path = tmp_path / "test_jobs.db"
    db = JobDatabase(db_path)
    yield db
    db.close()


@pytest.fixture
def sample_job():
    """A minimal job dict for testing."""
    return {
        "id": "test-job-001",
        "scene_id": "scene-1",
        "title": "Test Video",
        "source_path": "/tmp/test_input.mp4",
        "output_path": "/tmp/test_output.mp4",
        "tier": "tier3",
        "model": "realesr-animevideov3",
        "scale": 2,
        "is_vr": False,
        "status": "pending",
        "priority": 10,
    }


@pytest.fixture
def sample_vr_job():
    """A VR job dict for testing."""
    return {
        "id": "test-job-vr-001",
        "scene_id": "scene-vr-1",
        "title": "VR Test Video 180 SBS",
        "source_path": "/tmp/test_vr_180_sbs.mp4",
        "output_path": "/tmp/test_vr_output.mp4",
        "tier": "tier2",
        "model": "realesr-animevideov3",
        "scale": 2,
        "is_vr": True,
        "status": "pending",
        "priority": 50,
    }


@pytest.fixture
def sample_matte_job():
    """A matte/chroma-key job dict for testing."""
    return {
        "id": "test-job-matte-001",
        "scene_id": "scene-pov-1",
        "title": "POV Test Video",
        "source_path": "/tmp/test_pov.mp4",
        "output_path": "/tmp/test_pov_matted.mp4",
        "tier": "matte",
        "model": None,
        "scale": None,
        "is_vr": False,
        "matte": True,
        "status": "pending",
        "priority": 5,
    }
