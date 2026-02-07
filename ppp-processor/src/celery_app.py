"""Celery application configuration for PPP Processor.

Broker: Redis.  Routes local GPU tasks to 'local_gpu' queue and
cloud tasks to 'cloud' queue.  task_acks_late=True for crash recovery.
"""

from __future__ import annotations

from celery import Celery

from src.config import get_settings

app = Celery("ppp_processor")


def _configure_celery():
    """Apply settings to Celery. Called lazily to avoid import-time YAML parse."""
    settings = get_settings()
    app.conf.update(
        broker_url=settings.redis.url,
        result_backend=settings.redis.url,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_routes={
            "src.workers.local_worker.*": {"queue": "local_gpu"},
            "src.workers.cloud_worker.*": {"queue": "cloud"},
        },
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
