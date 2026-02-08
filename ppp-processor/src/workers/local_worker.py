"""Local GPU Celery worker for PPP Processor.

Handles QA sampling, auto-approval, full pipeline processing,
output verification, and post-processing (Stash/Jellyfin).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from celery import Task

from src.celery_app import app
from src.config import get_settings
from src.database import JobDatabase

logger = logging.getLogger("ppp.worker.local")


class LocalGPUTask(Task):
    """Base task class with lazy-initialized shared resources."""

    _db = None
    _settings = None

    @property
    def settings(self):
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def db(self):
        if self._db is None:
            db_path = Path(self.settings.paths.temp_dir).parent / "jobs.db"
            self._db = JobDatabase(db_path)
        return self._db


@app.task(base=LocalGPUTask, bind=True, name="src.workers.local_worker.process_job")
def process_job(self: LocalGPUTask, job_id: str):
    """Main Celery task: QA sample -> auto-approve -> full pipeline -> verify -> post-process."""
    from src.analyzer import VideoAnalyzer
    from src.models.schemas import JobStatus
    from src.processor import ProcessingPipeline
    from src.qa_validator import QAValidator
    from src.router import Router

    from src.utils import platform_path

    job = self.db.get_job(job_id)
    if not job:
        logger.error("Job %s not found", job_id)
        return {"status": "error", "message": "Job not found"}

    source_path = Path(platform_path(job["source_path"]))
    if not source_path.exists():
        self.db.update_job_status(job_id, "failed", error=f"Source file not found: {source_path}")
        return {"status": "failed", "message": f"Source file not found: {source_path}"}

    worker_id = f"local-{self.request.hostname}" if self.request.hostname else "local-0"
    start_time = time.time()

    def progress_callback(stage: str, percent: float):
        self.update_state(
            state="PROGRESS",
            meta={"stage": stage, "percent": percent, "job_id": job_id},
        )
        self.db.update_job_status(
            job_id, "processing",
            progress=percent, current_stage=stage, worker_id=worker_id,
        )

    try:
        # Analyze video
        settings = self.settings
        analyzer = VideoAnalyzer(settings)
        info = analyzer.probe_video(source_path)

        # Override VR detection from job if specified
        if job.get("is_vr") and not info.is_vr:
            info.is_vr = True
            info.vr_type = "sbs"

        # --- Matte-only job path ---
        if job.get("matte"):
            from src.processor import MatteProcessor

            self.db.update_job_status(
                job_id, JobStatus.PROCESSING.value,
                worker_id=worker_id, current_stage="matting",
            )

            output_path = Path(platform_path(job["output_path"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)

            matte_proc = MatteProcessor(settings)
            if info.is_vr and info.vr_type == "sbs":
                success = matte_proc.process_vr_sbs(
                    source_path, output_path, info, progress_callback,
                )
            else:
                success = matte_proc.process_video(
                    source_path, output_path, info, progress_callback,
                )

            processing_time = time.time() - start_time

            if success and output_path.exists() and output_path.stat().st_size > 1024:
                self.db.update_job_status(
                    job_id, JobStatus.COMPLETED.value,
                    processing_time=processing_time,
                    progress=100, current_stage="complete",
                )
                logger.info("Matte job %s completed in %.1f min", job_id, processing_time / 60)
                return {"status": "completed", "processing_time": processing_time}
            else:
                self.db.update_job_status(
                    job_id, JobStatus.FAILED.value,
                    error="Matting pipeline failed",
                    processing_time=processing_time,
                )
                return {"status": "failed"}

        # Generate processing plan
        router = Router(settings)
        plan = router.plan(info, target_scale=job.get("scale"))

        # Override model from job if specified
        if job.get("model"):
            plan.model = job["model"]

        # QA Sampling phase
        self.db.update_job_status(
            job_id, JobStatus.SAMPLING.value,
            worker_id=worker_id, current_stage="sampling",
        )

        qa = QAValidator(settings, self.db)
        sample = qa.process_sample(source_path, job_id, plan, info)

        if sample.auto_approved is True:
            # Auto-approved — proceed to full processing
            logger.info("Job %s auto-approved, proceeding", job_id)
            self.db.update_job_status(job_id, JobStatus.APPROVED.value)
        elif sample.auto_approved is False:
            # Auto-rejected
            logger.warning("Job %s auto-rejected (SSIM=%.3f PSNR=%.1f)",
                          job_id, sample.ssim or 0, sample.psnr or 0)
            self.db.update_job_status(
                job_id, JobStatus.REJECTED.value,
                error=f"QA rejected: SSIM={sample.ssim}, PSNR={sample.psnr}",
            )
            return {"status": "rejected", "sample_id": sample.id}
        else:
            # Needs human review
            logger.info("Job %s needs human QA review", job_id)
            self.db.update_job_status(job_id, JobStatus.SAMPLE_READY.value)
            return {"status": "sample_ready", "sample_id": sample.id}

        # Full processing
        self.db.update_job_status(
            job_id, JobStatus.PROCESSING.value,
            worker_id=worker_id, current_stage="processing",
        )

        output_path = Path(platform_path(job["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline = ProcessingPipeline(settings)
        success = pipeline.run(source_path, output_path, plan, info, progress_callback)

        processing_time = time.time() - start_time

        if not success:
            self.db.update_job_status(
                job_id, JobStatus.FAILED.value,
                error="Processing pipeline failed",
                processing_time=processing_time,
            )
            return {"status": "failed"}

        # Output verification
        if not output_path.exists() or output_path.stat().st_size < 1024:
            self.db.update_job_status(
                job_id, JobStatus.FAILED.value,
                error="Output file missing or too small",
                processing_time=processing_time,
            )
            return {"status": "failed"}

        # Post-processing (Stash tags + Jellyfin)
        try:
            _post_process(settings, job, output_path, info)
        except Exception as e:
            logger.warning("Post-processing failed (non-fatal): %s", e)

        # Mark completed
        self.db.update_job_status(
            job_id, JobStatus.COMPLETED.value,
            processing_time=processing_time,
            progress=100, current_stage="complete",
        )

        logger.info(
            "Job %s completed in %.1f min",
            job_id, processing_time / 60,
        )
        return {"status": "completed", "processing_time": processing_time}

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)[:500]
        logger.error("Job %s failed: %s", job_id, error_msg)
        self.db.update_job_status(
            job_id, JobStatus.FAILED.value,
            error=error_msg, processing_time=processing_time,
        )
        return {"status": "failed", "error": error_msg}


def _post_process(settings, job: dict, output_path: Path, info):
    """Run Stash and Jellyfin integrations after successful processing."""
    from src.integrations.stash import StashClient
    from src.integrations.jellyfin import JellyfinClient
    from src.utils import resolution_label

    scene_id = job.get("scene_id")
    res_label = resolution_label(info.width * (job.get("scale") or 2))

    # Stash tagging
    if scene_id and settings.stash.api_key:
        try:
            stash = StashClient(settings.stash.url, settings.stash.api_key)
            stash.update_scene_after_upscale(scene_id, output_path, res_label)
        except Exception as e:
            logger.warning("Stash update failed: %s", e)

    # Jellyfin organization
    if settings.jellyfin.library_path:
        try:
            jf = JellyfinClient(settings.jellyfin.url, settings.jellyfin.api_key,
                               settings.jellyfin.library_path)
            jf.organize_and_copy(output_path, is_vr=info.is_vr)
            jf.trigger_library_scan()
        except Exception as e:
            logger.warning("Jellyfin integration failed: %s", e)
