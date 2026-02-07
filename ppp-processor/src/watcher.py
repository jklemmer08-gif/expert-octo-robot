"""Library scanner and watcher for PPP Processor.

Walks directory trees to find new videos, checks against the database,
and creates jobs for unprocessed content.  Supports CSV import from
the existing batch_process.py workflow.
"""

from __future__ import annotations

import csv
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import Settings
from src.database import JobDatabase
from src.utils import file_hash, is_video_file, remap_path

logger = logging.getLogger("ppp.watcher")


class LibraryScanner:
    """Scans library directories for new videos and creates processing jobs."""

    def __init__(self, settings: Settings, db: JobDatabase):
        self.settings = settings
        self.db = db

    # ------------------------------------------------------------------
    # Directory scanning
    # ------------------------------------------------------------------
    def scan_directory(self, root: Path, tier: str = "tier3") -> int:
        """Walk a directory tree, find videos, check against DB, create jobs.

        Returns the number of new jobs created.
        """
        if not root.exists():
            logger.warning("Scan path does not exist: %s", root)
            return 0

        created = 0
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if not is_video_file(path):
                continue

            # Skip output directory
            output_dir = Path(self.settings.paths.output_dir)
            try:
                path.relative_to(output_dir)
                continue  # Inside output dir — skip
            except ValueError:
                pass

            # Skip temp directory
            temp_dir = Path(self.settings.paths.temp_dir)
            try:
                path.relative_to(temp_dir)
                continue
            except ValueError:
                pass

            # Check if already in DB by path
            if self.db.job_exists_for_path(str(path)):
                continue

            # Fast file fingerprint
            fhash = file_hash(path)
            if self.db.job_exists_for_hash(fhash):
                continue

            # Create job
            job_id = str(uuid.uuid4())
            output_name = f"{path.stem}_upscaled_{self.settings.upscale.scale_factor}x.mp4"
            output_path = Path(self.settings.paths.output_dir) / tier / output_name

            self.db.add_job({
                "id": job_id,
                "title": path.stem[:80],
                "source_path": str(path),
                "output_path": str(output_path),
                "tier": tier,
                "model": self.settings.upscale.default_model,
                "scale": self.settings.upscale.scale_factor,
                "status": "pending",
                "priority": 0,
                "file_hash": fhash,
            })
            created += 1

        logger.info("Scan complete: %d new jobs from %s", created, root)
        return created

    # ------------------------------------------------------------------
    # CSV import (from batch_process.py:219-279)
    # ------------------------------------------------------------------
    def import_from_csv(
        self, csv_path: Path, tier: str = "tier3",
        limit: Optional[int] = None,
    ) -> int:
        """Import jobs from an analysis CSV file.

        Preserves the tier-based model/priority configuration from
        the original batch_process.py.
        """
        if not csv_path.exists():
            logger.warning("CSV not found: %s", csv_path)
            return 0

        tier_config = {
            "tier1": {"model": "realesrgan-x4plus", "priority": 100, "scale": 2},
            "tier2": {"model": "realesr-animevideov3", "priority": 50, "scale": 2},
            "tier3": {"model": "realesr-animevideov3", "priority": 10, "scale": 2},
        }
        config = tier_config.get(tier, tier_config["tier3"])

        imported = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break

                source_path = remap_path(row.get("path", ""))
                if not source_path or not Path(source_path).exists():
                    continue

                scene_id = row.get("id", str(i))
                title = row.get("title", Path(source_path).stem)[:80]
                is_vr = row.get("is_vr", "").lower() == "true"

                source_name = Path(source_path).stem
                output_name = f"{source_name}_upscaled_{config['scale']}x.mp4"
                output_path = Path(self.settings.paths.output_dir) / tier / output_name

                quality_score = float(row.get("quality_score", 0))
                priority = config["priority"] + int(quality_score / 10)

                job = {
                    "id": f"{tier}_{scene_id}",
                    "scene_id": scene_id,
                    "title": title,
                    "source_path": source_path,
                    "output_path": str(output_path),
                    "tier": tier,
                    "model": config["model"],
                    "scale": config["scale"],
                    "is_vr": is_vr,
                    "status": "pending",
                    "priority": priority,
                }

                if self.db.add_job(job):
                    imported += 1

        logger.info("Imported %d jobs from %s", imported, csv_path.name)
        return imported

    def import_all_tiers(self) -> int:
        """Import from all tier CSVs in the analysis directory."""
        analysis_dir = Path(self.settings.paths.analysis_dir)
        csvs = {
            "tier1": "tier1_topaz_candidates.csv",
            "tier2": "tier2_runpod_vr.csv",
            "tier3": "tier3_local_bulk.csv",
        }

        total = 0
        for tier, filename in csvs.items():
            csv_path = analysis_dir / filename
            if csv_path.exists():
                total += self.import_from_csv(csv_path, tier)

        logger.info("Total imported: %d jobs", total)
        return total
