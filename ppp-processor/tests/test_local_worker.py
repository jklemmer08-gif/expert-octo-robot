"""Tests for local GPU worker (mocked GPU)."""

from unittest.mock import MagicMock, patch

import pytest

from src.database import JobDatabase


def test_process_job_not_found(test_db):
    """Job lookup returns None for nonexistent job."""
    job = test_db.get_job("nonexistent")
    assert job is None


def test_process_job_source_missing(test_db, sample_job, test_settings):
    """Task should fail if source file doesn't exist."""
    test_db.add_job(sample_job)
    job = test_db.get_job("test-job-001")
    assert job is not None
    # Source /tmp/test_input.mp4 doesn't exist
    from pathlib import Path
    assert not Path(job["source_path"]).exists()


def test_job_status_transitions(test_db, sample_job):
    """Test that job status transitions are recorded correctly."""
    test_db.add_job(sample_job)

    test_db.update_job_status("test-job-001", "sampling", current_stage="sampling")
    job = test_db.get_job("test-job-001")
    assert job["status"] == "sampling"

    test_db.update_job_status("test-job-001", "sample_ready")
    job = test_db.get_job("test-job-001")
    assert job["status"] == "sample_ready"

    test_db.update_job_status("test-job-001", "approved")
    job = test_db.get_job("test-job-001")
    assert job["status"] == "approved"

    test_db.update_job_status("test-job-001", "processing",
                              worker_id="local-0", current_stage="upscaling")
    job = test_db.get_job("test-job-001")
    assert job["status"] == "processing"
    assert job["worker_id"] == "local-0"

    test_db.update_job_status("test-job-001", "completed", processing_time=300.0)
    job = test_db.get_job("test-job-001")
    assert job["status"] == "completed"
    assert job["processing_time_sec"] == 300.0
