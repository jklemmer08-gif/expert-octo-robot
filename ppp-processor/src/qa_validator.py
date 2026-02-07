"""QA validation system for PPP Processor v2.0.

Extracts 15-second samples, upscales them, computes quality metrics
(SSIM, PSNR, sharpness), and applies auto-approval logic.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.config import Settings
from src.database import JobDatabase
from src.models.schemas import ProcessingPlan, QASample, VideoInfo
from src.processor import FrameExtractor, ProcessingPipeline

logger = logging.getLogger("ppp.qa")


class QAValidator:
    """QA validation with sample extraction, quality metrics, and auto-approval."""

    def __init__(self, settings: Settings, db: JobDatabase):
        self.settings = settings
        self.db = db
        self.extractor = FrameExtractor()
        self.samples_dir = Path(settings.paths.temp_dir) / "qa_samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Sample extraction (from upscale.py:410-443)
    # ------------------------------------------------------------------
    def extract_sample(self, source_path: Path, job_id: str) -> Path:
        """Extract a short sample from the source video."""
        sample_path = self.samples_dir / f"{job_id}_original_sample.mp4"
        self.extractor.extract_sample(
            source_path, sample_path,
            duration=self.settings.qa.sample_duration,
            start_percent=self.settings.qa.sample_start_percent,
        )
        return sample_path

    def process_sample(
        self,
        source_path: Path,
        job_id: str,
        plan: ProcessingPlan,
        info: VideoInfo,
    ) -> QASample:
        """Extract sample, upscale it, compute metrics, store in DB."""
        sample_id = str(uuid.uuid4())

        # Extract original sample
        original_sample = self.extract_sample(source_path, job_id)

        # Upscale sample
        upscaled_sample = self.samples_dir / f"{job_id}_upscaled_sample.mp4"
        pipeline = ProcessingPipeline(self.settings)
        pipeline.run(original_sample, upscaled_sample, plan, info)

        # Compute metrics
        ssim_val, psnr_val, sharpness_val = self.compute_metrics(
            original_sample, upscaled_sample,
        )

        # Auto-approve decision
        auto = self.auto_approve(ssim_val, psnr_val)

        sample = QASample(
            id=sample_id,
            job_id=job_id,
            sample_path=str(upscaled_sample),
            original_sample_path=str(original_sample),
            ssim=ssim_val,
            psnr=psnr_val,
            sharpness=sharpness_val,
            auto_approved=auto,
            created_at=datetime.now(),
        )

        # Persist
        self.db.add_qa_sample(sample.model_dump())

        logger.info(
            "QA sample %s: SSIM=%.3f PSNR=%.1f sharpness=%.1f auto=%s",
            sample_id, ssim_val or 0, psnr_val or 0,
            sharpness_val or 0, auto,
        )
        return sample

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------
    def compute_metrics(
        self, original_path: Path, upscaled_path: Path,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Compute SSIM, PSNR, and sharpness between original and upscaled.

        Uses scikit-image for SSIM/PSNR and Laplacian variance for sharpness.
        Extracts a single frame from each video for comparison.
        """
        try:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        except ImportError:
            logger.warning("scikit-image not available, skipping metrics")
            return None, None, None

        try:
            orig_frame = self._extract_comparison_frame(original_path, "orig")
            upsc_frame = self._extract_comparison_frame(upscaled_path, "upsc")

            if orig_frame is None or upsc_frame is None:
                return None, None, None

            # Resize original to match upscaled dimensions for comparison
            from PIL import Image
            orig_img = Image.open(orig_frame)
            upsc_img = Image.open(upsc_frame)

            # Resize original up to match upscaled dimensions
            orig_resized = orig_img.resize(upsc_img.size, Image.LANCZOS)

            orig_arr = np.array(orig_resized).astype(np.float64)
            upsc_arr = np.array(upsc_img).astype(np.float64)

            orig_img.close()
            upsc_img.close()

            # SSIM
            min_dim = min(orig_arr.shape[0], orig_arr.shape[1])
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            if win_size < 3:
                win_size = 3
            ssim_val = structural_similarity(
                orig_arr, upsc_arr, channel_axis=2,
                data_range=255, win_size=win_size,
            )

            # PSNR
            psnr_val = peak_signal_noise_ratio(orig_arr, upsc_arr, data_range=255)

            # Sharpness (Laplacian variance of upscaled frame)
            sharpness_val = self._compute_sharpness(upsc_arr)

            # Cleanup
            orig_frame.unlink(missing_ok=True)
            upsc_frame.unlink(missing_ok=True)

            return (
                round(float(ssim_val), 4),
                round(float(psnr_val), 2),
                round(float(sharpness_val), 2),
            )

        except Exception as e:
            logger.error("Metrics computation failed: %s", e)
            return None, None, None

    def _extract_comparison_frame(self, video_path: Path, tag: str) -> Optional[Path]:
        """Extract a single frame from the middle of a video for comparison."""
        import subprocess, json

        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))

        frame_path = self.samples_dir / f"_compare_{tag}.png"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(duration / 2),
            "-i", str(video_path),
            "-frames:v", "1",
            str(frame_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return frame_path if frame_path.exists() else None

    def _compute_sharpness(self, img_array: np.ndarray) -> float:
        """Compute sharpness via Laplacian variance (higher = sharper)."""
        # Convert to grayscale
        if img_array.ndim == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Laplacian kernel
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

        from scipy.signal import convolve2d
        lap = convolve2d(gray, laplacian, mode="valid")
        return float(np.var(lap))

    # ------------------------------------------------------------------
    # Auto-approval logic
    # ------------------------------------------------------------------
    def auto_approve(
        self, ssim: Optional[float], psnr: Optional[float],
    ) -> Optional[bool]:
        """Auto-approve logic:
        - SSIM >= threshold AND PSNR >= threshold -> approve (True)
        - Within tolerance % of thresholds -> needs human review (None)
        - Below thresholds -> reject (False)
        """
        if ssim is None or psnr is None:
            return None  # Can't decide without metrics

        ssim_thresh = self.settings.qa.ssim_threshold
        psnr_thresh = self.settings.qa.psnr_threshold
        tolerance = self.settings.qa.auto_approve_tolerance

        ssim_ok = ssim >= ssim_thresh
        psnr_ok = psnr >= psnr_thresh

        if ssim_ok and psnr_ok:
            return True  # Auto-approve

        # Check if within tolerance (needs human review)
        ssim_close = ssim >= ssim_thresh * (1 - tolerance)
        psnr_close = psnr >= psnr_thresh * (1 - tolerance)

        if ssim_close and psnr_close:
            return None  # Human review

        return False  # Reject

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup_old_samples(self):
        """Remove old QA samples based on approval status and age."""
        now = datetime.now()
        approved_cutoff = now - timedelta(days=self.settings.qa.cleanup_approved_days)
        rejected_cutoff = now - timedelta(days=self.settings.qa.cleanup_rejected_days)

        # Get all samples
        rows = self.db.conn.execute("""
            SELECT id, sample_path, original_sample_path, auto_approved,
                   human_approved, created_at
            FROM qa_samples
        """).fetchall()

        cleaned = 0
        for row in rows:
            row = dict(row)
            created = datetime.fromisoformat(row["created_at"]) if row["created_at"] else now

            approved = row["human_approved"] or row["auto_approved"]
            if approved and created < approved_cutoff:
                self._remove_sample_files(row)
                cleaned += 1
            elif approved is False and created < rejected_cutoff:
                self._remove_sample_files(row)
                cleaned += 1

        logger.info("Cleaned up %d old QA samples", cleaned)

    def _remove_sample_files(self, sample: dict):
        """Remove sample video files from disk."""
        for key in ("sample_path", "original_sample_path"):
            path = sample.get(key)
            if path:
                p = Path(path)
                if p.exists():
                    p.unlink()
