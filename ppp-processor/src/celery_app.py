"""Celery application configuration for PPP Processor.

Broker: Redis.  Routes local GPU tasks to 'local_gpu' queue and
cloud tasks to 'cloud' queue.  task_acks_late=True for crash recovery.
"""

from __future__ import annotations

from celery import Celery

from src.config import get_settings

app = Celery(
    "ppp_processor",
    include=[
        "src.workers.local_worker",
        "src.workers.cloud_worker",
        "src.workers.cloud_worker_v2",
    ],
)


def _configure_celery():
    """Apply settings to Celery. Called lazily to avoid import-time YAML parse."""
    import os
    import platform
    settings = get_settings()

    # Ensure FFmpeg binaries are on PATH (bin/ lives next to models_dir)
    if platform.system() == "Windows":
        from pathlib import Path
        bin_dir = str(Path(settings.paths.models_dir).parent)
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    app.conf.update(
        broker_url=settings.redis.url,
        result_backend=settings.redis.url,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_routes={
            "src.workers.local_worker.*": {"queue": "local_gpu"},
            "src.workers.cloud_worker.*": {"queue": "cloud"},
            "src.workers.cloud_worker_v2.*": {"queue": "cloud"},
        },
        # Cloud tasks get 4-hour hard limit
        task_time_limit=14400,
        result_expires=86400,
        task_track_started=True,
    )


# Configure when Celery is actually used (not at import time in tests)
try:
    _configure_celery()
except Exception:
    pass  # Allow import to succeed even without valid config/Redis

# Auto-discover tasks in worker modules
app.autodiscover_tasks(["src.workers"])
