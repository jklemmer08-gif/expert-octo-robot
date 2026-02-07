#!/usr/bin/env python3
"""
PPP Batch Processor - Queue-based video processing with progress tracking
Reads from analysis CSVs and processes videos with resume capability
"""

import csv
import sqlite3
import os
import sys
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.upscale import PPPUpscaler, UpscaleJob

@dataclass
class Job:
    id: str
    scene_id: str
    title: str
    source_path: str
    output_path: str
    tier: str
    model: str
    scale: int
    is_vr: bool
    status: str  # pending, processing, completed, failed, skipped
    priority: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_sec: Optional[float] = None

class JobDatabase:
    """SQLite-based job tracking"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                scene_id TEXT,
                title TEXT,
                source_path TEXT,
                output_path TEXT,
                tier TEXT,
                model TEXT,
                scale INTEGER,
                is_vr INTEGER,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                processing_time_sec REAL
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        self.conn.commit()
    
    def add_job(self, job: Job) -> bool:
        """Add a job if it doesn't already exist"""
        try:
            self.conn.execute("""
                INSERT OR IGNORE INTO jobs 
                (id, scene_id, title, source_path, output_path, tier, model, scale, 
                 is_vr, status, priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id, job.scene_id, job.title, job.source_path, job.output_path,
                job.tier, job.model, job.scale, int(job.is_vr), job.status,
                job.priority, job.created_at
            ))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def get_next_job(self) -> Optional[Job]:
        """Get the next pending job by priority"""
        row = self.conn.execute("""
            SELECT * FROM jobs 
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
        """).fetchone()
        
        if row:
            return Job(**dict(row))
        return None
    
    def update_job_status(self, job_id: str, status: str, 
                          error: Optional[str] = None,
                          processing_time: Optional[float] = None):
        """Update job status"""
        now = datetime.now().isoformat()
        
        if status == 'processing':
            self.conn.execute("""
                UPDATE jobs SET status = ?, started_at = ?
                WHERE id = ?
            """, (status, now, job_id))
        elif status in ('completed', 'failed'):
            self.conn.execute("""
                UPDATE jobs SET status = ?, completed_at = ?, 
                               error_message = ?, processing_time_sec = ?
                WHERE id = ?
            """, (status, now, error, processing_time, job_id))
        else:
            self.conn.execute("""
                UPDATE jobs SET status = ?
                WHERE id = ?
            """, (status, job_id))
        
        self.conn.commit()
    
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        stats = {}
        
        for status in ['pending', 'processing', 'completed', 'failed', 'skipped']:
            count = self.conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)
            ).fetchone()[0]
            stats[status] = count
        
        stats['total'] = sum(stats.values())
        
        # Average processing time
        avg = self.conn.execute("""
            SELECT AVG(processing_time_sec) FROM jobs 
            WHERE status = 'completed' AND processing_time_sec > 0
        """).fetchone()[0]
        stats['avg_time_sec'] = round(avg, 1) if avg else 0
        
        return stats
    
    def get_jobs_by_status(self, status: str, limit: int = 100) -> List[Job]:
        """Get jobs by status"""
        rows = self.conn.execute("""
            SELECT * FROM jobs WHERE status = ?
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
        """, (status, limit)).fetchall()
        
        return [Job(**dict(row)) for row in rows]
    
    def reset_stuck_jobs(self):
        """Reset jobs that were processing but never completed (crashed)"""
        self.conn.execute("""
            UPDATE jobs SET status = 'pending', started_at = NULL
            WHERE status = 'processing'
        """)
        self.conn.commit()
    
    def close(self):
        self.conn.close()


class BatchProcessor:
    """Batch processing manager"""
    
    # Path remapping from Stash Docker mounts to local mounts
    PATH_REMAPS = [
        ("/data/library/", "/home/jtk1234/media-drive1/"),
        ("/data/media/", "/home/jtk1234/media-drive1/"),
        ("/data/recovered/", "/home/jtk1234/media-drive2/"),
    ]

    def __init__(self, config_path: Optional[Path] = None):
        self.base_dir = Path(__file__).parent.parent
        self.analysis_dir = self.base_dir.parent / "ppp_analysis"

        self.upscaler = PPPUpscaler(config_path)
        self.output_dir = self.upscaler.output_dir
        self.db_path = self.upscaler.temp_dir.parent / "jobs.db"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db = JobDatabase(self.db_path)
        
        self.running = True
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        print("\n\nInterrupt received, finishing current job...")
        self.running = False

    def _remap_path(self, path: str) -> str:
        """Remap Stash Docker paths to local mount points"""
        for docker_prefix, local_prefix in self.PATH_REMAPS:
            if path.startswith(docker_prefix):
                return local_prefix + path[len(docker_prefix):]
        return path
    
    def import_from_csv(self, csv_path: Path, tier: str, 
                        limit: Optional[int] = None) -> int:
        """Import jobs from analysis CSV"""
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            return 0
        
        # Determine model and priority based on tier
        tier_config = {
            'tier1': {'model': 'realesrgan-x4plus', 'priority': 100, 'scale': 2},
            'tier2': {'model': 'realesr-animevideov3', 'priority': 50, 'scale': 2},
            'tier3': {'model': 'realesr-animevideov3', 'priority': 10, 'scale': 2},
        }
        
        config = tier_config.get(tier, tier_config['tier3'])
        
        imported = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                
                source_path = self._remap_path(row.get('path', ''))
                if not source_path or not Path(source_path).exists():
                    continue
                
                scene_id = row.get('id', str(i))
                title = row.get('title', Path(source_path).stem)[:80]
                is_vr = row.get('is_vr', '').lower() == 'true'
                
                # Generate output filename (preserve VR naming)
                source_name = Path(source_path).stem
                output_name = f"{source_name}_upscaled_{config['scale']}x.mp4"
                output_path = self.output_dir / tier / output_name
                
                # Use quality score for priority boost
                quality_score = float(row.get('quality_score', 0))
                priority = config['priority'] + int(quality_score / 10)
                
                job = Job(
                    id=f"{tier}_{scene_id}",
                    scene_id=scene_id,
                    title=title,
                    source_path=source_path,
                    output_path=str(output_path),
                    tier=tier,
                    model=config['model'],
                    scale=config['scale'],
                    is_vr=is_vr,
                    status='pending',
                    priority=priority,
                    created_at=datetime.now().isoformat()
                )
                
                if self.db.add_job(job):
                    imported += 1
        
        print(f"Imported {imported} jobs from {csv_path.name}")
        return imported
    
    def import_all_tiers(self):
        """Import from all tier CSVs"""
        csvs = {
            'tier1': 'tier1_topaz_candidates.csv',
            'tier2': 'tier2_runpod_vr.csv', 
            'tier3': 'tier3_local_bulk.csv',
        }
        
        total = 0
        for tier, filename in csvs.items():
            csv_path = self.analysis_dir / filename
            if csv_path.exists():
                total += self.import_from_csv(csv_path, tier)
        
        print(f"\nTotal imported: {total} jobs")
        return total
    
    def process_job(self, job: Job) -> bool:
        """Process a single job"""
        print(f"\n{'='*60}")
        print(f"Processing: {job.title}")
        print(f"Tier: {job.tier}, Model: {job.model}, Scale: {job.scale}x")
        print(f"{'='*60}")
        
        # Update status to processing
        self.db.update_job_status(job.id, 'processing')
        
        start_time = time.time()
        
        try:
            # Create output directory
            output_path = Path(job.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if output already exists
            if output_path.exists():
                print(f"Output already exists, skipping: {output_path}")
                self.db.update_job_status(job.id, 'skipped')
                return True
            
            # Create upscale job (pass VR flag from CSV to override detection)
            upscale_job = UpscaleJob(
                input_path=Path(job.source_path),
                output_path=output_path,
                model=job.model,
                scale=job.scale,
                tile_size=512,
                gpu_id=0,
                force_vr=job.is_vr
            )
            
            # Run upscaling
            success = self.upscaler.process_video(upscale_job)
            
            processing_time = time.time() - start_time
            
            if success:
                self.db.update_job_status(
                    job.id, 'completed', 
                    processing_time=processing_time
                )
                print(f"\n✓ Completed in {processing_time/60:.1f} minutes")
                return True
            else:
                self.db.update_job_status(
                    job.id, 'failed',
                    error='Upscaling failed',
                    processing_time=processing_time
                )
                return False
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)[:500]
            print(f"\n✗ Error: {error_msg}")
            self.db.update_job_status(
                job.id, 'failed',
                error=error_msg,
                processing_time=processing_time
            )
            return False
    
    def run(self, max_jobs: Optional[int] = None):
        """Main processing loop"""
        print("\n" + "="*60)
        print("PPP Batch Processor")
        print("="*60)
        
        # Reset any stuck jobs from previous run
        self.db.reset_stuck_jobs()
        
        stats = self.db.get_stats()
        print(f"\nQueue Status:")
        print(f"  Pending:   {stats['pending']}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Failed:    {stats['failed']}")
        
        if stats['pending'] == 0:
            print("\nNo pending jobs. Import jobs first with --import")
            return
        
        processed = 0
        
        while self.running:
            job = self.db.get_next_job()
            
            if not job:
                print("\nNo more pending jobs.")
                break
            
            self.process_job(job)
            processed += 1
            
            if max_jobs and processed >= max_jobs:
                print(f"\nReached max jobs limit ({max_jobs})")
                break
            
            # Print progress
            stats = self.db.get_stats()
            remaining = stats['pending']
            if remaining > 0 and stats['avg_time_sec'] > 0:
                eta_hours = (remaining * stats['avg_time_sec']) / 3600
                print(f"\nProgress: {stats['completed']}/{stats['total']} complete, ~{eta_hours:.1f}h remaining")
        
        # Final summary
        stats = self.db.get_stats()
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"  Processed this session: {processed}")
        print(f"  Total completed: {stats['completed']}")
        print(f"  Total failed: {stats['failed']}")
        print(f"  Remaining: {stats['pending']}")
    
    def show_status(self):
        """Show current queue status"""
        stats = self.db.get_stats()
        
        print("\n" + "="*60)
        print("QUEUE STATUS")
        print("="*60)
        print(f"  Total jobs:   {stats['total']}")
        print(f"  Pending:      {stats['pending']}")
        print(f"  Processing:   {stats['processing']}")
        print(f"  Completed:    {stats['completed']}")
        print(f"  Failed:       {stats['failed']}")
        print(f"  Skipped:      {stats['skipped']}")
        
        if stats['avg_time_sec'] > 0:
            print(f"\n  Avg processing time: {stats['avg_time_sec']/60:.1f} minutes")
            
            if stats['pending'] > 0:
                eta_hours = (stats['pending'] * stats['avg_time_sec']) / 3600
                eta_days = eta_hours / 24
                print(f"  Est. time remaining: {eta_hours:.1f} hours ({eta_days:.1f} days)")
        
        # Show recent failures
        failed = self.db.get_jobs_by_status('failed', limit=5)
        if failed:
            print("\n  Recent failures:")
            for job in failed:
                print(f"    - {job.title[:40]}: {job.error_message[:50] if job.error_message else 'Unknown'}")
    
    def close(self):
        self.db.close()


def main():
    parser = argparse.ArgumentParser(description="PPP Batch Processor")
    parser.add_argument("--import", dest="import_jobs", action="store_true",
                        help="Import jobs from analysis CSVs")
    parser.add_argument("--import-csv", type=Path,
                        help="Import from specific CSV file")
    parser.add_argument("--tier", default="tier3",
                        choices=["tier1", "tier2", "tier3"],
                        help="Tier for imported jobs")
    parser.add_argument("--limit", type=int,
                        help="Limit number of jobs to import or process")
    parser.add_argument("--status", action="store_true",
                        help="Show queue status")
    parser.add_argument("--run", action="store_true",
                        help="Start processing queue")
    parser.add_argument("--reset-failed", action="store_true",
                        help="Reset failed jobs to pending")
    
    args = parser.parse_args()
    
    processor = BatchProcessor()
    
    try:
        if args.status:
            processor.show_status()
        
        elif args.import_jobs:
            processor.import_all_tiers()
        
        elif args.import_csv:
            processor.import_from_csv(args.import_csv, args.tier, args.limit)
        
        elif args.reset_failed:
            processor.db.conn.execute(
                "UPDATE jobs SET status = 'pending' WHERE status = 'failed'"
            )
            processor.db.conn.commit()
            print("Reset all failed jobs to pending")
        
        elif args.run:
            processor.run(max_jobs=args.limit)
        
        else:
            parser.print_help()
            print("\nQuick start:")
            print("  1. Import jobs:  python batch_process.py --import")
            print("  2. Check status: python batch_process.py --status")
            print("  3. Start:        python batch_process.py --run")
    
    finally:
        processor.close()


if __name__ == "__main__":
    main()
