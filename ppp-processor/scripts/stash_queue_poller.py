#!/usr/bin/env python3
"""Stash tag-based job queue poller.

Polls Stash every N minutes for scenes tagged "PPP-Queue", creates matte
jobs in the database, dispatches them to the local_gpu_priority Celery
queue, and swaps the tag to "PPP-Queued" + "PPP-Processing" so they
aren't picked up again.

Usage:
    python scripts/stash_queue_poller.py                  # default 15 min
    python scripts/stash_queue_poller.py --interval 60    # 1 min (testing)
    python scripts/stash_queue_poller.py --once            # single pass
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.database import JobDatabase
from src.integrations.stash import StashClient
from src.utils import platform_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ppp.poller")

INPUT_TAG = "PPP-Queue"
QUEUED_TAG = "PPP-Queued"
PROCESSING_TAG = "PPP-Processing"
PROCESSED_TAG = "PPP-Processed"
PRIORITY = 200
QUEUE = "local_gpu_priority"


def poll_once(stash: StashClient, db: JobDatabase, settings) -> int:
    """Check Stash for tagged scenes and create jobs. Returns count created."""
    scenes = stash.find_scenes_by_tag(INPUT_TAG)
    if not scenes:
        logger.info("No scenes tagged '%s'", INPUT_TAG)
        return 0

    logger.info("Found %d scene(s) tagged '%s'", len(scenes), INPUT_TAG)
    created = 0

    for scene in scenes:
        scene_id = scene["id"]
        title = scene.get("title") or f"scene-{scene_id}"

        # Skip scenes already processed
        scene_tags = [t.get("name", "") for t in scene.get("tags", [])]
        if PROCESSED_TAG in scene_tags:
            logger.info("Scene %s already tagged '%s', skipping", scene_id, PROCESSED_TAG)
            stash.remove_tag_from_scene(scene_id, INPUT_TAG)
            continue

        files = scene.get("files", [])
        if not files:
            logger.warning("Scene %s has no files, skipping", scene_id)
            stash.remove_tag_from_scene(scene_id, INPUT_TAG)
            continue

        source_docker = files[0].get("path", "")
        if not source_docker:
            logger.warning("Scene %s file has no path, skipping", scene_id)
            stash.remove_tag_from_scene(scene_id, INPUT_TAG)
            continue

        source_local = platform_path(source_docker)

        # Check if source file exists
        if not Path(source_local).exists():
            logger.warning("Source not found: %s (scene %s), skipping", source_local, scene_id)
            stash.remove_tag_from_scene(scene_id, INPUT_TAG)
            continue

        # Skip if a job already exists for this scene
        if db.job_exists_for_path(source_docker):
            logger.info("Job already exists for scene %s, skipping", scene_id)
            stash.remove_tag_from_scene(scene_id, INPUT_TAG)
            stash.add_tag_to_scene(scene_id, QUEUED_TAG)
            continue

        # Build output path (same pattern as main.py matte dispatch)
        source = Path(source_docker)
        output_path = str(
            Path(settings.paths.output_dir) / "matted"
            / f"{source.stem}_matted.mp4"
        )

        job_id = str(uuid.uuid4())
        job_dict = {
            "id": job_id,
            "scene_id": scene_id,
            "title": title[:80],
            "source_path": source_docker,
            "output_path": output_path,
            "tier": "matte",
            "model": None,
            "scale": None,
            "is_vr": False,
            "matte": True,
            "status": "pending",
            "priority": PRIORITY,
            "created_at": datetime.now().isoformat(),
        }

        if not db.add_job(job_dict):
            logger.info("Job insert skipped (duplicate?) for scene %s", scene_id)
            stash.remove_tag_from_scene(scene_id, INPUT_TAG)
            stash.add_tag_to_scene(scene_id, QUEUED_TAG)
            continue

        # Dispatch to priority queue
        try:
            from src.celery_app import app as celery_app
            celery_app.send_task(
                "src.workers.local_worker.process_job",
                args=[job_id],
                queue=QUEUE,
            )
            logger.info("Dispatched job %s to %s (scene %s: %s)", job_id, QUEUE, scene_id, title)
        except Exception as e:
            logger.warning("Celery dispatch failed (job stays pending): %s", e)

        # Swap tags: remove input tag, add queued + processing
        stash.remove_tag_from_scene(scene_id, INPUT_TAG)
        stash.add_tag_to_scene(scene_id, QUEUED_TAG)
        stash.add_tag_to_scene(scene_id, PROCESSING_TAG)
        created += 1

    return created


def main():
    parser = argparse.ArgumentParser(description="Stash tag-based job queue poller")
    parser.add_argument("--interval", type=int, default=900,
                        help="Poll interval in seconds (default: 900 = 15 min)")
    parser.add_argument("--once", action="store_true",
                        help="Run a single poll cycle and exit")
    args = parser.parse_args()

    settings = get_settings()
    stash = StashClient(settings.paths.stash_url)
    db_path = Path(settings.paths.temp_dir).parent / "jobs.db"
    db = JobDatabase(db_path)

    logger.info("Stash queue poller started (interval=%ds, tag='%s', queue='%s')",
                args.interval, INPUT_TAG, QUEUE)

    if args.once:
        count = poll_once(stash, db, settings)
        logger.info("Single pass complete: %d job(s) created", count)
        db.close()
        return

    try:
        while True:
            count = poll_once(stash, db, settings)
            if count:
                logger.info("Created %d job(s) this cycle", count)
            logger.info("Sleeping %d seconds until next poll...", args.interval)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Poller stopped by user")
    finally:
        db.close()


if __name__ == "__main__":
    main()
