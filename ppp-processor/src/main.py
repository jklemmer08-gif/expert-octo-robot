"""FastAPI application for PPP Processor v2.0.

Provides REST endpoints for job management, QA review, worker status,
library scanning, and health checks.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.database import JobDatabase
from src.logging_config import setup_logging
from src.models.schemas import (
    JobCreate,
    JobResponse,
    JobStatus,
    QASample,
    WorkerStatus,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PPP Processor",
    description="Automated video upscaling pipeline API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database — lazily initialised on first request
_db: Optional[JobDatabase] = None


def get_db() -> JobDatabase:
    global _db
    if _db is None:
        s = get_settings()
        db_path = Path(s.paths.temp_dir).parent / "jobs.db"
        _db = JobDatabase(db_path)
    return _db


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/status")
def status():
    db = get_db()
    stats = db.get_stats()
    cost = db.get_cost_summary()
    return {"queue": stats, "cost": cost}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
@app.post("/jobs", response_model=JobResponse)
def create_job(job: JobCreate):
    db = get_db()
    s = get_settings()
    job_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    output_path = job.output_path
    if not output_path:
        source = Path(job.source_path)
        output_path = str(
            Path(s.paths.output_dir) / job.tier
            / f"{source.stem}_upscaled_{job.scale or s.upscale.scale_factor}x.mp4"
        )

    job_dict: Dict[str, Any] = {
        "id": job_id,
        "scene_id": job.scene_id,
        "title": job.title or Path(job.source_path).stem,
        "source_path": job.source_path,
        "output_path": output_path,
        "tier": job.tier,
        "model": job.model or s.upscale.default_model,
        "scale": job.scale or s.upscale.scale_factor,
        "is_vr": bool(job.force_vr) if job.force_vr is not None else False,
        "status": "pending",
        "priority": job.priority,
        "created_at": now,
    }

    db.add_job(job_dict)

    # Dispatch to Celery (lazy import to avoid circular deps if Celery not running)
    try:
        from src.workers.local_worker import process_job as local_task
        local_task.delay(job_id)
    except Exception:
        pass  # Celery not available — job stays pending for manual pickup

    return JobResponse(**job_dict)


@app.get("/jobs", response_model=List[JobResponse])
def list_jobs(
    status: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
):
    db = get_db()
    rows = db.get_jobs(status=status, limit=limit, offset=offset)
    return [JobResponse(**r) for r in rows]


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    db = get_db()
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**row)


# ---------------------------------------------------------------------------
# Library scanning
# ---------------------------------------------------------------------------
@app.post("/library/scan")
def library_scan():
    """Trigger a library scan via Celery (or synchronous fallback)."""
    try:
        from src.watcher import LibraryScanner
        s = get_settings()
        scanner = LibraryScanner(s, get_db())
        count = scanner.scan_directory(Path(s.paths.library_root))
        return {"status": "completed", "new_jobs": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/library/stats")
def library_stats():
    db = get_db()
    stats = db.get_stats()
    return stats


# ---------------------------------------------------------------------------
# QA endpoints
# ---------------------------------------------------------------------------
@app.get("/qa/pending")
def qa_pending():
    db = get_db()
    samples = db.get_pending_qa_samples()
    return samples


@app.post("/qa/{sample_id}/approve")
def qa_approve(sample_id: str, notes: Optional[str] = None):
    db = get_db()
    sample = db.get_qa_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    db.update_qa_sample(
        sample_id,
        human_approved=True,
        reviewer_notes=notes,
        reviewed_at=datetime.now().isoformat(),
    )

    # Move job to approved status
    db.update_job_status(sample["job_id"], JobStatus.APPROVED.value)

    # Dispatch full processing
    try:
        from src.workers.local_worker import process_job as local_task
        local_task.delay(sample["job_id"])
    except Exception:
        pass

    return {"status": "approved", "job_id": sample["job_id"]}


@app.post("/qa/{sample_id}/reject")
def qa_reject(sample_id: str, notes: Optional[str] = None):
    db = get_db()
    sample = db.get_qa_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    db.update_qa_sample(
        sample_id,
        human_approved=False,
        reviewer_notes=notes,
        reviewed_at=datetime.now().isoformat(),
    )

    db.update_job_status(sample["job_id"], JobStatus.REJECTED.value)
    return {"status": "rejected", "job_id": sample["job_id"]}


@app.get("/qa/{sample_id}/compare")
def qa_compare(sample_id: str):
    db = get_db()
    sample = db.get_qa_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return {
        "sample": sample,
        "original_path": sample.get("original_sample_path"),
        "upscaled_path": sample.get("sample_path"),
    }


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
@app.get("/workers", response_model=List[WorkerStatus])
def list_workers():
    db = get_db()
    rows = db.get_workers()
    return [WorkerStatus(**r) for r in rows]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@app.post("/route")
def route_job(source_path: str, force_vr: Optional[bool] = None):
    """Get a processing plan for a source file without creating a job."""
    try:
        from src.analyzer import VideoAnalyzer
        from src.router import Router

        s = get_settings()
        analyzer = VideoAnalyzer(s)
        info = analyzer.probe_video(Path(source_path))
        router = Router(s)
        plan = router.plan(info)
        return plan.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
