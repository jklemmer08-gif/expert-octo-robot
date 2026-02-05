"""Progress tracking and Server-Sent Events (SSE) broadcasting."""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class JobProgress:
    """Track progress for a single processing job."""
    job_id: str
    stage: str = "queued"
    frame: int = 0
    total_frames: int = 0
    segment: int = 0
    total_segments: int = 0
    started_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def percent(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return round(self.frame / self.total_frames * 100, 1)

    @property
    def elapsed_sec(self) -> float:
        return round(time.time() - self.started_at, 1)

    @property
    def eta_sec(self) -> Optional[float]:
        if self.frame == 0 or self.total_frames == 0:
            return None
        elapsed = time.time() - self.started_at
        rate = self.frame / elapsed
        remaining = self.total_frames - self.frame
        return round(remaining / rate, 1) if rate > 0 else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "frame": self.frame,
            "total_frames": self.total_frames,
            "segment": self.segment,
            "total_segments": self.total_segments,
            "percent": self.percent,
            "elapsed_sec": self.elapsed_sec,
            "eta_sec": self.eta_sec,
            "error": self.error,
        }


class ProgressManager:
    """Thread-safe progress tracking with SSE subscriber support."""

    def __init__(self):
        self._jobs: Dict[str, JobProgress] = {}
        self._subscribers: List = []
        self._lock = threading.Lock()

    def create_job(self, job_id: str, total_frames: int = 0) -> JobProgress:
        with self._lock:
            progress = JobProgress(job_id=job_id, total_frames=total_frames)
            self._jobs[job_id] = progress
            self._broadcast(progress)
            return progress

    def update(self, job_id: str, data: Dict[str, Any]):
        with self._lock:
            progress = self._jobs.get(job_id)
            if not progress:
                return
            for key in ("stage", "frame", "total_frames", "segment", "total_segments", "error"):
                if key in data:
                    setattr(progress, key, data[key])
            self._broadcast(progress)

    def get_progress(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            progress = self._jobs.get(job_id)
            return progress.to_dict() if progress else None

    def get_current_job(self) -> Optional[Dict]:
        """Return the most recent active job's progress."""
        with self._lock:
            for job_id in reversed(list(self._jobs.keys())):
                p = self._jobs[job_id]
                if p.stage not in ("completed", "failed"):
                    return p.to_dict()
            # Return most recent job even if completed
            if self._jobs:
                last = list(self._jobs.values())[-1]
                return last.to_dict()
            return None

    def subscribe(self):
        """Return a generator that yields SSE events. Used by Flask route."""
        import queue
        q = queue.Queue()
        with self._lock:
            self._subscribers.append(q)

        try:
            # Send current state immediately
            current = self.get_current_job()
            if current:
                yield f"data: {json.dumps(current)}\n\n"

            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
        finally:
            with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    def _broadcast(self, progress: JobProgress):
        """Send update to all SSE subscribers (called with lock held)."""
        data = progress.to_dict()
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(data)
            except Exception:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)


# Singleton instance
progress_manager = ProgressManager()
