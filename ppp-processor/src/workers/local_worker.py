"""Local GPU Celery worker for PPP Processor.

Handles QA sampling, auto-approval, full pipeline processing,
output verification, and post-processing (Stash/Jellyfin).
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from celery import Task

from src.celery_app import app
from src.config import get_settings
from src.database import JobDatabase

logger = logging.getLogger("ppp.worker.local")


def _cache_to_local(source_path: Path, cache_dir: str) -> tuple[Path, bool]:
    """Copy a network file to local SSD cache for faster I/O + NVDEC.

    Returns (local_path, was_cached). If cache_dir is empty or source is
    already local, returns the original path with was_cached=False.
    """
    if not cache_dir:
        return source_path, False

    # Skip if already on a local drive (not a UNC path or mapped network drive)
    drive = source_path.drive
    if drive and drive[0] not in ("M", "N", "\\"):
        return source_path, False

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    local_path = cache / source_path.name

    if local_path.exists() and local_path.stat().st_size == source_path.stat().st_size:
        logger.info("Cache hit: %s", local_path)
        return local_path, True

    size_mb = source_path.stat().st_size / 1024 / 1024
    logger.info("Caching %.0f MB to local SSD: %s -> %s", size_mb, source_path.name, local_path)
    shutil.copy2(source_path, local_path)
    logger.info("Cache copy complete: %s", local_path.name)
    return local_path, True


def _cleanup_cache(local_path: Path, was_cached: bool):
    """Remove a cached local copy after processing."""
    if was_cached and local_path.exists():
        try:
            local_path.unlink()
            logger.info("Cleaned cache: %s", local_path.name)
        except PermissionError:
            logger.warning("Cache file still locked, will be cleaned next run: %s", local_path.name)


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

    # Cache network file to local SSD for faster I/O + NVDEC support
    settings = self.settings
    local_source, was_cached = _cache_to_local(source_path, settings.paths.local_cache_dir)

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
        # Analyze video (use local cached copy for I/O speed)
        analyzer = VideoAnalyzer(settings)
        info = analyzer.probe_video(local_source)

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
            # Alpha pack output: append _FISHEYE190_alpha suffix
            if settings.matte.output_type == "alpha_pack" and info.is_vr:
                stem = output_path.stem
                if "_FISHEYE190_alpha" not in stem:
                    output_path = output_path.with_stem(stem + "_FISHEYE190_alpha")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            matte_proc = MatteProcessor(settings)
            try:
                if info.is_vr and info.vr_type == "sbs":
                    success = matte_proc.process_vr_sbs(
                        local_source, output_path, info, progress_callback,
                    )
                else:
                    success = matte_proc.process_video(
                        local_source, output_path, info, progress_callback,
                    )
            finally:
                # Release inference engine resources before next job to prevent
                # context corruption in Celery's solo pool.
                if matte_proc._openvino_engine is not None:
                    matte_proc._openvino_engine.cleanup()
                    matte_proc._openvino_engine = None
                if matte_proc._ort_engine is not None:
                    matte_proc._ort_engine.cleanup()
                    matte_proc._ort_engine = None
                if matte_proc._trt_engine is not None:
                    matte_proc._trt_engine.cleanup()
                    matte_proc._trt_engine = None

            processing_time = time.time() - start_time

            if success and output_path.exists() and output_path.stat().st_size > 1024:
                # Tag original scene in Stash so user knows matte is available
                scene_id = job.get("scene_id")
                if scene_id:
                    try:
                        from src.integrations.stash import StashClient
                        stash = StashClient(settings.paths.stash_url)
                        stash.add_tag_to_scene(scene_id, "Green Screen Available")
                        stash.add_tag_to_scene(scene_id, "PPP-Processed")
                        stash.add_tag_to_scene(scene_id, "Passthrough_simple")
                        logger.info("Tagged scene %s with 'Green Screen Available' + 'Passthrough_simple'", scene_id)
                    except Exception as e:
                        logger.warning("Stash tagging failed (non-fatal): %s", e)

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

        # --- Upscale-only job path ---
        if job.get("upscale"):
            from src.processor import Encoder, ProcessingPipeline, UpscaleEngine
            from src.models.schemas import ContentType, ProcessingPlan

            self.db.update_job_status(
                job_id, JobStatus.PROCESSING.value,
                worker_id=worker_id, current_stage="upscaling",
            )

            output_path = Path(platform_path(job["output_path"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)

            scale = job.get("scale") or settings.upscale.scale_factor
            model = job.get("model") or settings.upscale.default_model
            bitrate = settings.encode.bitrates.get("4K", "50M")

            plan = ProcessingPlan(
                model=model,
                scale=scale,
                worker_type="local",
                bitrate=bitrate,
                encoder=settings.encode.encoder if settings.encode.encoder != "hevc_vaapi" else settings.encode.fallback_encoder,
                content_type=info.content_type,
                tile_size=settings.gpu.tile_size,
                gpu_id=settings.gpu.device_id,
            )

            pipeline = ProcessingPipeline(settings)

            if info.is_vr and info.vr_type == "sbs":
                success = pipeline.run(local_source, output_path, plan, info, progress_callback)
            else:
                success = pipeline.run(local_source, output_path, plan, info, progress_callback)

            processing_time = time.time() - start_time

            if success and output_path.exists() and output_path.stat().st_size > 1024:
                self.db.update_job_status(
                    job_id, JobStatus.COMPLETED.value,
                    processing_time=processing_time,
                    progress=100, current_stage="complete",
                )
                logger.info("Upscale job %s completed in %.1f min", job_id, processing_time / 60)
                return {"status": "completed", "processing_time": processing_time}
            else:
                self.db.update_job_status(
                    job_id, JobStatus.FAILED.value,
                    error="Upscale pipeline failed",
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
        sample = qa.process_sample(local_source, job_id, plan, info)

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
        success = pipeline.run(local_source, output_path, plan, info, progress_callback)

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

    finally:
        _cleanup_cache(local_source, was_cached)


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
