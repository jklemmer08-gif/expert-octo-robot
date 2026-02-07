"""Tests for FastAPI endpoints using TestClient."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(test_db, test_settings):
    """Create a test client with mocked database."""
    # Reset the settings singleton to use our test settings
    import src.config as config_module
    config_module._settings = test_settings

    import src.main as main_module
    main_module._db = test_db

    from src.main import app
    with TestClient(app) as c:
        yield c

    main_module._db = None
    config_module._settings = None


class TestHealthEndpoints:

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"

    def test_status(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "queue" in data
        assert "cost" in data


class TestJobEndpoints:

    @patch("src.main.local_task", create=True)
    def test_create_job(self, mock_task, client):
        resp = client.post("/jobs", json={"source_path": "/tmp/test.mp4"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_path"] == "/tmp/test.mp4"
        assert data["status"] == "pending"
        assert data["id"] is not None

    def test_list_jobs_empty(self, client):
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_jobs_with_data(self, client, test_db, sample_job):
        test_db.add_job(sample_job)
        resp = client.get("/jobs")
        assert resp.status_code == 200
        jobs = resp.json()
        assert len(jobs) == 1
        assert jobs[0]["id"] == "test-job-001"

    def test_get_job(self, client, test_db, sample_job):
        test_db.add_job(sample_job)
        resp = client.get("/jobs/test-job-001")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Test Video"

    def test_get_job_not_found(self, client):
        resp = client.get("/jobs/nonexistent")
        assert resp.status_code == 404

    def test_list_jobs_filter(self, client, test_db, sample_job):
        test_db.add_job(sample_job)
        test_db.update_job_status("test-job-001", "completed")

        resp = client.get("/jobs?status=completed")
        assert len(resp.json()) == 1

        resp = client.get("/jobs?status=pending")
        assert len(resp.json()) == 0


class TestQAEndpoints:

    def test_qa_pending_empty(self, client):
        resp = client.get("/qa/pending")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_qa_approve_not_found(self, client):
        resp = client.post("/qa/nonexistent/approve")
        assert resp.status_code == 404

    def test_qa_reject_not_found(self, client):
        resp = client.post("/qa/nonexistent/reject")
        assert resp.status_code == 404

    def test_qa_approve(self, client, test_db, sample_job):
        test_db.add_job(sample_job)
        sid = test_db.add_qa_sample({
            "job_id": "test-job-001",
            "ssim": 0.80,
        })
        resp = client.post(f"/qa/{sid}/approve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    def test_qa_reject(self, client, test_db, sample_job):
        test_db.add_job(sample_job)
        sid = test_db.add_qa_sample({
            "job_id": "test-job-001",
            "ssim": 0.50,
        })
        resp = client.post(f"/qa/{sid}/reject")
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"


class TestWorkerEndpoints:

    def test_list_workers_empty(self, client):
        resp = client.get("/workers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_workers(self, client, test_db):
        test_db.upsert_worker({
            "id": "local-1",
            "worker_type": "local_gpu",
            "status": "idle",
        })
        resp = client.get("/workers")
        assert resp.status_code == 200
        assert len(resp.json()) == 1


class TestMatteEndpoints:

    @patch("src.main.local_task", create=True)
    def test_create_matte_job(self, mock_task, client):
        resp = client.post("/jobs", json={
            "source_path": "/tmp/pov_scene.mp4",
            "matte": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["matte"] is True
        assert "_matted" in data["output_path"]

    @patch("src.main.local_task", create=True)
    def test_create_normal_job_no_matte(self, mock_task, client):
        resp = client.post("/jobs", json={
            "source_path": "/tmp/normal.mp4",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["matte"] is False
        assert "_matted" not in data["output_path"]
