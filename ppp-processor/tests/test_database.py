"""Tests for database CRUD operations."""

import pytest
from datetime import datetime

from src.database import JobDatabase


def test_add_and_get_job(test_db, sample_job):
    assert test_db.add_job(sample_job) is True
    job = test_db.get_job("test-job-001")
    assert job is not None
    assert job["title"] == "Test Video"
    assert job["status"] == "pending"


def test_add_duplicate_job(test_db, sample_job):
    test_db.add_job(sample_job)
    # Second insert should be ignored (INSERT OR IGNORE)
    result = test_db.add_job(sample_job)
    assert result is False


def test_get_next_job(test_db, sample_job, sample_vr_job):
    test_db.add_job(sample_job)
    test_db.add_job(sample_vr_job)
    # VR job has higher priority (50 vs 10)
    next_job = test_db.get_next_job()
    assert next_job["id"] == "test-job-vr-001"


def test_update_job_status(test_db, sample_job):
    test_db.add_job(sample_job)
    test_db.update_job_status("test-job-001", "processing", worker_id="w1")
    job = test_db.get_job("test-job-001")
    assert job["status"] == "processing"
    assert job["started_at"] is not None
    assert job["worker_id"] == "w1"


def test_update_job_completed(test_db, sample_job):
    test_db.add_job(sample_job)
    test_db.update_job_status("test-job-001", "completed", processing_time=120.5)
    job = test_db.get_job("test-job-001")
    assert job["status"] == "completed"
    assert job["completed_at"] is not None
    assert job["processing_time_sec"] == 120.5


def test_update_job_failed(test_db, sample_job):
    test_db.add_job(sample_job)
    test_db.update_job_status("test-job-001", "failed", error="Out of memory")
    job = test_db.get_job("test-job-001")
    assert job["status"] == "failed"
    assert job["error_message"] == "Out of memory"


def test_get_jobs_with_filter(test_db, sample_job, sample_vr_job):
    test_db.add_job(sample_job)
    test_db.add_job(sample_vr_job)
    test_db.update_job_status("test-job-001", "completed")

    pending = test_db.get_jobs(status="pending")
    assert len(pending) == 1
    assert pending[0]["id"] == "test-job-vr-001"

    completed = test_db.get_jobs(status="completed")
    assert len(completed) == 1


def test_get_stats(test_db, sample_job, sample_vr_job):
    test_db.add_job(sample_job)
    test_db.add_job(sample_vr_job)
    test_db.update_job_status("test-job-001", "completed", processing_time=60.0)

    stats = test_db.get_stats()
    assert stats["pending"] == 1
    assert stats["completed"] == 1
    assert stats["total"] == 2
    assert stats["avg_time_sec"] == 60.0


def test_reset_stuck_jobs(test_db, sample_job):
    test_db.add_job(sample_job)
    test_db.update_job_status("test-job-001", "processing")
    test_db.reset_stuck_jobs()
    job = test_db.get_job("test-job-001")
    assert job["status"] == "pending"
    assert job["worker_id"] is None


def test_job_exists_for_path(test_db, sample_job):
    test_db.add_job(sample_job)
    assert test_db.job_exists_for_path("/tmp/test_input.mp4") is True
    assert test_db.job_exists_for_path("/tmp/other.mp4") is False


def test_job_exists_for_hash(test_db):
    test_db.add_job({
        "id": "hash-test",
        "source_path": "/tmp/vid.mp4",
        "file_hash": "abc123",
    })
    assert test_db.job_exists_for_hash("abc123") is True
    assert test_db.job_exists_for_hash("def456") is False


# QA sample tests
def test_add_and_get_qa_sample(test_db, sample_job):
    test_db.add_job(sample_job)
    sid = test_db.add_qa_sample({
        "job_id": "test-job-001",
        "ssim": 0.92,
        "psnr": 32.5,
        "sharpness": 150.0,
        "auto_approved": True,
    })
    sample = test_db.get_qa_sample(sid)
    assert sample is not None
    assert sample["ssim"] == 0.92
    assert sample["auto_approved"] == 1  # SQLite stores as int


def test_get_qa_samples_for_job(test_db, sample_job):
    test_db.add_job(sample_job)
    test_db.add_qa_sample({"id": "s1", "job_id": "test-job-001"})
    test_db.add_qa_sample({"id": "s2", "job_id": "test-job-001"})
    samples = test_db.get_qa_samples_for_job("test-job-001")
    assert len(samples) == 2


def test_pending_qa_samples(test_db, sample_job):
    test_db.add_job(sample_job)
    test_db.add_qa_sample({
        "id": "need-review",
        "job_id": "test-job-001",
        "auto_approved": None,
    })
    pending = test_db.get_pending_qa_samples()
    assert len(pending) == 1


def test_update_qa_sample(test_db, sample_job):
    test_db.add_job(sample_job)
    sid = test_db.add_qa_sample({"job_id": "test-job-001"})
    test_db.update_qa_sample(sid, human_approved=True, reviewer_notes="Looks good")
    sample = test_db.get_qa_sample(sid)
    assert sample["human_approved"] == 1
    assert sample["reviewer_notes"] == "Looks good"


# Worker tests
def test_upsert_worker(test_db):
    test_db.upsert_worker({
        "id": "local-1",
        "worker_type": "local_gpu",
        "status": "idle",
        "gpu_name": "Arc B580",
    })
    workers = test_db.get_workers()
    assert len(workers) == 1
    assert workers[0]["gpu_name"] == "Arc B580"

    # Update
    test_db.upsert_worker({
        "id": "local-1",
        "worker_type": "local_gpu",
        "status": "busy",
        "gpu_name": "Arc B580",
    })
    w = test_db.get_worker("local-1")
    assert w["status"] == "busy"


# Cost tracking tests
def test_log_and_get_cost(test_db, sample_job):
    test_db.add_job(sample_job)
    test_db.log_cost("test-job-001", "cloud-1", "RTX 4090", 1.50, 3.0)
    assert test_db.get_total_cost() == 1.50

    summary = test_db.get_cost_summary()
    assert summary["total_cost"] == 1.50
    assert "RTX 4090" in summary["by_gpu"]
