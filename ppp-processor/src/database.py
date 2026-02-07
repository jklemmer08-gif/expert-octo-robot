"""Extended SQLite database for PPP Processor v2.0.

Refactored from batch_process.py:JobDatabase with additional tables
for QA samples, workers, and cost tracking.  Uses WAL mode for
concurrent read access.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class JobDatabase:
    """SQLite-based persistence for jobs, QA samples, workers, and costs."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL mode for concurrent reads
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                scene_id TEXT,
                title TEXT,
                source_path TEXT,
                output_path TEXT,
                tier TEXT,
                model TEXT,
                scale INTEGER,
                is_vr INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                processing_time_sec REAL,
                progress REAL,
                current_stage TEXT,
                worker_id TEXT,
                estimated_cost REAL,
                actual_cost REAL,
                file_hash TEXT
            );

            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS qa_samples (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                sample_path TEXT,
                original_sample_path TEXT,
                ssim REAL,
                psnr REAL,
                sharpness REAL,
                auto_approved INTEGER,
                human_approved INTEGER,
                reviewer_notes TEXT,
                created_at TEXT,
                reviewed_at TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE TABLE IF NOT EXISTS workers (
                id TEXT PRIMARY KEY,
                worker_type TEXT,
                status TEXT DEFAULT 'idle',
                current_job_id TEXT,
                gpu_name TEXT,
                gpu_utilization REAL,
                memory_utilization REAL,
                jobs_completed INTEGER DEFAULT 0,
                total_processing_hours REAL DEFAULT 0.0,
                total_cost REAL DEFAULT 0.0,
                last_heartbeat TEXT,
                pod_id TEXT,
                extra TEXT
            );

            CREATE TABLE IF NOT EXISTS cost_tracking (
                id TEXT PRIMARY KEY,
                job_id TEXT,
                worker_id TEXT,
                gpu_name TEXT,
                cost REAL,
                duration_hours REAL,
                timestamp TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority DESC);
            CREATE INDEX IF NOT EXISTS idx_qa_job_id ON qa_samples(job_id);
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Job CRUD (preserved from batch_process.py + extensions)
    # ------------------------------------------------------------------
    def add_job(self, job: Dict[str, Any]) -> bool:
        """Add a job if it doesn't already exist. Returns True on insert."""
        job_id = job.get("id") or str(uuid.uuid4())
        try:
            cursor = self.conn.execute("""
                INSERT OR IGNORE INTO jobs
                (id, scene_id, title, source_path, output_path, tier, model,
                 scale, is_vr, status, priority, created_at, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                job.get("scene_id"),
                job.get("title"),
                job.get("source_path"),
                job.get("output_path"),
                job.get("tier", "tier3"),
                job.get("model"),
                job.get("scale"),
                int(job.get("is_vr", False)),
                job.get("status", "pending"),
                job.get("priority", 0),
                job.get("created_at") or datetime.now().isoformat(),
                job.get("file_hash"),
            ))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.IntegrityError:
            return False

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID."""
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """Get the next pending job by priority."""
        row = self.conn.execute("""
            SELECT * FROM jobs
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
        """).fetchone()
        return dict(row) if row else None

    def get_jobs(self, status: Optional[str] = None, limit: int = 100,
                 offset: int = 0) -> List[Dict[str, Any]]:
        """Get jobs with optional status filter."""
        if status:
            rows = self.conn.execute("""
                SELECT * FROM jobs WHERE status = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT ? OFFSET ?
            """, (status, limit, offset)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT * FROM jobs
                ORDER BY priority DESC, created_at ASC
                LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def update_job_status(self, job_id: str, status: str,
                          error: Optional[str] = None,
                          processing_time: Optional[float] = None,
                          progress: Optional[float] = None,
                          current_stage: Optional[str] = None,
                          worker_id: Optional[str] = None,
                          actual_cost: Optional[float] = None):
        """Update job status and optional fields."""
        now = datetime.now().isoformat()

        if status == "processing":
            self.conn.execute("""
                UPDATE jobs SET status=?, started_at=?, worker_id=?,
                               progress=?, current_stage=?
                WHERE id=?
            """, (status, now, worker_id, progress, current_stage, job_id))
        elif status in ("completed", "failed"):
            self.conn.execute("""
                UPDATE jobs SET status=?, completed_at=?, error_message=?,
                               processing_time_sec=?, actual_cost=?,
                               progress=?, current_stage=?
                WHERE id=?
            """, (status, now, error, processing_time, actual_cost,
                  progress, current_stage, job_id))
        else:
            self.conn.execute("""
                UPDATE jobs SET status=?, progress=?, current_stage=?
                WHERE id=?
            """, (status, progress, current_stage, job_id))

        self.conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats: Dict[str, Any] = {}
        for s in ["pending", "processing", "completed", "failed", "skipped",
                   "sampling", "sample_ready", "approved", "rejected", "encoding"]:
            count = self.conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?", (s,)
            ).fetchone()[0]
            stats[s] = count

        stats["total"] = sum(stats.values())

        avg = self.conn.execute("""
            SELECT AVG(processing_time_sec) FROM jobs
            WHERE status = 'completed' AND processing_time_sec > 0
        """).fetchone()[0]
        stats["avg_time_sec"] = round(avg, 1) if avg else 0

        return stats

    def reset_stuck_jobs(self):
        """Reset jobs stuck in processing (crash recovery)."""
        self.conn.execute("""
            UPDATE jobs SET status='pending', started_at=NULL, worker_id=NULL
            WHERE status='processing'
        """)
        self.conn.commit()

    def job_exists_for_path(self, source_path: str) -> bool:
        """Check if a job already exists for a given source path."""
        row = self.conn.execute(
            "SELECT 1 FROM jobs WHERE source_path = ? LIMIT 1", (source_path,)
        ).fetchone()
        return row is not None

    def job_exists_for_hash(self, file_hash: str) -> bool:
        """Check if a job already exists for a given file hash."""
        row = self.conn.execute(
            "SELECT 1 FROM jobs WHERE file_hash = ? LIMIT 1", (file_hash,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # QA samples
    # ------------------------------------------------------------------
    def add_qa_sample(self, sample: Dict[str, Any]) -> str:
        """Insert a QA sample record. Returns the sample ID."""
        sample_id = sample.get("id") or str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO qa_samples
            (id, job_id, sample_path, original_sample_path, ssim, psnr,
             sharpness, auto_approved, human_approved, reviewer_notes,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_id,
            sample["job_id"],
            sample.get("sample_path"),
            sample.get("original_sample_path"),
            sample.get("ssim"),
            sample.get("psnr"),
            sample.get("sharpness"),
            sample.get("auto_approved"),
            sample.get("human_approved"),
            sample.get("reviewer_notes"),
            sample.get("created_at") or datetime.now().isoformat(),
        ))
        self.conn.commit()
        return sample_id

    def get_qa_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM qa_samples WHERE id = ?", (sample_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_qa_samples_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM qa_samples WHERE job_id = ? ORDER BY created_at DESC",
            (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_qa_samples(self) -> List[Dict[str, Any]]:
        """Get samples needing human review (auto_approved IS NULL or False, not yet reviewed)."""
        rows = self.conn.execute("""
            SELECT qs.*, j.title, j.source_path FROM qa_samples qs
            JOIN jobs j ON qs.job_id = j.id
            WHERE qs.human_approved IS NULL
              AND (qs.auto_approved IS NULL OR qs.auto_approved = 0)
            ORDER BY qs.created_at ASC
        """).fetchall()
        return [dict(r) for r in rows]

    def update_qa_sample(self, sample_id: str, **kwargs):
        """Update QA sample fields."""
        allowed = {"human_approved", "reviewer_notes", "reviewed_at",
                    "ssim", "psnr", "sharpness", "auto_approved",
                    "sample_path", "original_sample_path"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [sample_id]
        self.conn.execute(
            f"UPDATE qa_samples SET {set_clause} WHERE id=?", values
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------
    def upsert_worker(self, worker: Dict[str, Any]):
        """Insert or update a worker record."""
        self.conn.execute("""
            INSERT INTO workers
            (id, worker_type, status, current_job_id, gpu_name,
             gpu_utilization, memory_utilization, jobs_completed,
             total_processing_hours, total_cost, last_heartbeat, pod_id, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                current_job_id=excluded.current_job_id,
                gpu_name=excluded.gpu_name,
                gpu_utilization=excluded.gpu_utilization,
                memory_utilization=excluded.memory_utilization,
                jobs_completed=excluded.jobs_completed,
                total_processing_hours=excluded.total_processing_hours,
                total_cost=excluded.total_cost,
                last_heartbeat=excluded.last_heartbeat,
                pod_id=excluded.pod_id,
                extra=excluded.extra
        """, (
            worker["id"],
            worker.get("worker_type", "local_gpu"),
            worker.get("status", "idle"),
            worker.get("current_job_id"),
            worker.get("gpu_name"),
            worker.get("gpu_utilization"),
            worker.get("memory_utilization"),
            worker.get("jobs_completed", 0),
            worker.get("total_processing_hours", 0.0),
            worker.get("total_cost", 0.0),
            worker.get("last_heartbeat") or datetime.now().isoformat(),
            worker.get("pod_id"),
            worker.get("extra"),
        ))
        self.conn.commit()

    @staticmethod
    def _normalize_worker(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        if d.get("extra") is None:
            d["extra"] = {}
        return d

    def get_workers(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM workers ORDER BY worker_type, id"
        ).fetchall()
        return [self._normalize_worker(r) for r in rows]

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM workers WHERE id = ?", (worker_id,)
        ).fetchone()
        return self._normalize_worker(row) if row else None

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------
    def log_cost(self, job_id: str, worker_id: str, gpu_name: str,
                 cost: float, duration_hours: float):
        cost_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO cost_tracking
            (id, job_id, worker_id, gpu_name, cost, duration_hours, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (cost_id, job_id, worker_id, gpu_name, cost, duration_hours,
              datetime.now().isoformat()))
        self.conn.commit()

    def get_total_cost(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost), 0) FROM cost_tracking"
        ).fetchone()
        return float(row[0])

    def get_cost_summary(self) -> Dict[str, Any]:
        total = self.get_total_cost()
        by_gpu = {}
        for row in self.conn.execute(
            "SELECT gpu_name, SUM(cost), SUM(duration_hours), COUNT(*) "
            "FROM cost_tracking GROUP BY gpu_name"
        ).fetchall():
            by_gpu[row[0]] = {
                "cost": round(row[1], 2),
                "hours": round(row[2], 2),
                "jobs": row[3],
            }
        return {"total_cost": round(total, 2), "by_gpu": by_gpu}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self):
        self.conn.close()
