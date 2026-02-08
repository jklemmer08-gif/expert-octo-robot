"""QA checkpoint framework for PPP Processor pipeline.

Provides reusable quality checks at each pipeline stage:
post-matte, post-encode, post-upscale, post-VR-merge.

Uses FFmpeg filter-based metrics (SSIM, PSNR) and frame-level
integrity checks to catch regressions early.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ppp.qa")


@dataclass
class QAResult:
    """Result of a QA checkpoint run."""
    stage: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = [f"[{status}] {self.stage}"]
        for k, v in self.metrics.items():
            parts.append(f"  {k}: {v:.4f}")
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN: {w}")
        return "\n".join(parts)


@dataclass
class QAThresholds:
    """Configurable thresholds for QA checks."""
    min_ssim: float = 0.97
    min_psnr: float = 30.0
    max_frame_count_drift: int = 2
    min_file_size_bytes: int = 1024
    min_alpha_coverage: float = 0.01
    max_alpha_coverage: float = 0.99
    min_edge_sharpness: float = 5.0


class QACheckpoint:
    """Reusable QA checkpoint runner for pipeline stages."""

    def __init__(self, thresholds: Optional[QAThresholds] = None):
        self.thresholds = thresholds or QAThresholds()

    def extract_frame(self, video_path: Path, output_path: Path,
                      timestamp: float = 0.0, frame_index: Optional[int] = None) -> Optional[Path]:
        """Extract a specific frame as PNG for visual inspection.

        Args:
            video_path: Input video file
            output_path: Where to save the PNG frame
            timestamp: Time in seconds to extract from
            frame_index: If set, extract Nth frame (overrides timestamp)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if frame_index is not None:
            cmd.extend(["-i", str(video_path)])
            cmd.extend(["-vf", f"select=eq(n\\,{frame_index})"])
            cmd.extend(["-frames:v", "1", "-vsync", "vfr"])
        else:
            cmd.extend(["-ss", str(timestamp)])
            cmd.extend(["-i", str(video_path)])
            cmd.extend(["-frames:v", "1"])
        cmd.append(str(output_path))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists():
            return output_path
        logger.warning("Frame extraction failed: %s", result.stderr[:200])
        return None

    def compute_ssim(self, reference: Path, distorted: Path) -> Optional[float]:
        """Compute SSIM between two videos using FFmpeg's ssim filter.

        Returns the average SSIM score, or None on failure.
        """
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(distorted),
            "-i", str(reference),
            "-filter_complex", "ssim",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse SSIM from stderr: "SSIM All:0.987654 (19.123456)"
        for line in result.stderr.split("\n"):
            if "SSIM" in line and "All:" in line:
                try:
                    ssim_str = line.split("All:")[1].split("(")[0].strip()
                    return float(ssim_str)
                except (IndexError, ValueError):
                    pass
        return None

    def compute_psnr(self, reference: Path, distorted: Path) -> Optional[float]:
        """Compute PSNR between two videos using FFmpeg's psnr filter.

        Returns the average PSNR in dB, or None on failure.
        """
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(distorted),
            "-i", str(reference),
            "-filter_complex", "psnr",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse PSNR from stderr: "PSNR average:38.1234 ..."
        for line in result.stderr.split("\n"):
            if "PSNR" in line and "average:" in line:
                try:
                    psnr_str = line.split("average:")[1].split()[0]
                    return float(psnr_str)
                except (IndexError, ValueError):
                    pass
        return None

    def verify_frame_count(self, video_path: Path, expected: int) -> Tuple[bool, int]:
        """Verify actual frame count matches expected within tolerance.

        Returns (passed, actual_count).
        """
        cmd = [
            "ffprobe", "-v", "quiet", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-print_format", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        try:
            data = json.loads(result.stdout)
            actual = int(data["streams"][0]["nb_read_frames"])
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            return False, 0

        drift = abs(actual - expected)
        passed = drift <= self.thresholds.max_frame_count_drift
        return passed, actual

    def verify_resolution(self, video_path: Path,
                          expected_w: int, expected_h: int) -> Tuple[bool, int, int]:
        """Verify video resolution matches expected.

        Returns (passed, actual_w, actual_h).
        """
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-print_format", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        try:
            data = json.loads(result.stdout)
            actual_w = int(data["streams"][0]["width"])
            actual_h = int(data["streams"][0]["height"])
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            return False, 0, 0

        passed = actual_w == expected_w and actual_h == expected_h
        return passed, actual_w, actual_h

    def check_alpha_matte_quality(self, frame_path: Path) -> Dict[str, float]:
        """Analyze alpha matte quality from a green-screen frame.

        Checks:
        - coverage: fraction of non-green pixels (should be between thresholds)
        - edge_sharpness: Laplacian variance of the green channel mask
        - noise_level: variance in supposedly uniform green areas

        Returns dict with metric values.
        """
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            logger.warning("PIL/numpy not available for matte quality check")
            return {}

        img = Image.open(frame_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)

        # Detect green screen pixels: high G, low R, low B
        green_mask = (arr[:, :, 1] > 100) & (arr[:, :, 0] < 100) & (arr[:, :, 2] < 100)
        total_pixels = green_mask.size
        green_pixels = green_mask.sum()
        coverage = 1.0 - (green_pixels / total_pixels)

        # Edge sharpness via Laplacian on the green mask
        mask_float = green_mask.astype(np.float32)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        try:
            from scipy.signal import convolve2d
            edges = convolve2d(mask_float, laplacian, mode="valid")
            edge_sharpness = float(np.var(edges))
        except ImportError:
            edge_sharpness = 0.0

        # Noise in green areas
        if green_pixels > 0:
            green_values = arr[green_mask]
            noise_level = float(np.std(green_values))
        else:
            noise_level = 0.0

        img.close()

        return {
            "coverage": round(coverage, 4),
            "edge_sharpness": round(edge_sharpness, 4),
            "noise_level": round(noise_level, 4),
        }

    def run_stage_checkpoint(
        self,
        stage: str,
        output_path: Path,
        reference_path: Optional[Path] = None,
        expected_frame_count: Optional[int] = None,
        expected_resolution: Optional[Tuple[int, int]] = None,
        check_matte: bool = False,
    ) -> QAResult:
        """Run a full QA pass for a pipeline stage.

        Args:
            stage: Name of the stage (e.g., "post-matte", "post-encode")
            output_path: The output file to check
            reference_path: Optional reference for SSIM/PSNR comparison
            expected_frame_count: Expected number of frames
            expected_resolution: Expected (width, height)
            check_matte: Whether to run alpha matte quality checks
        """
        result = QAResult(stage=stage, passed=True)

        # File existence and minimum size
        if not output_path.exists():
            result.passed = False
            result.errors.append(f"Output file missing: {output_path}")
            return result

        size = output_path.stat().st_size
        if size < self.thresholds.min_file_size_bytes:
            result.passed = False
            result.errors.append(f"File too small: {size} bytes")
            return result
        result.metrics["file_size_mb"] = size / 1024 / 1024

        # Frame count verification
        if expected_frame_count is not None:
            passed, actual = self.verify_frame_count(output_path, expected_frame_count)
            result.metrics["frame_count"] = actual
            result.metrics["frame_count_expected"] = expected_frame_count
            if not passed:
                result.passed = False
                result.errors.append(
                    f"Frame count mismatch: expected {expected_frame_count}, got {actual}"
                )

        # Resolution verification
        if expected_resolution is not None:
            exp_w, exp_h = expected_resolution
            passed, actual_w, actual_h = self.verify_resolution(output_path, exp_w, exp_h)
            result.metrics["width"] = actual_w
            result.metrics["height"] = actual_h
            if not passed:
                result.passed = False
                result.errors.append(
                    f"Resolution mismatch: expected {exp_w}x{exp_h}, got {actual_w}x{actual_h}"
                )

        # SSIM / PSNR comparison
        if reference_path and reference_path.exists():
            ssim = self.compute_ssim(reference_path, output_path)
            if ssim is not None:
                result.metrics["ssim"] = ssim
                if ssim < self.thresholds.min_ssim:
                    result.warnings.append(f"SSIM below threshold: {ssim:.4f} < {self.thresholds.min_ssim}")

            psnr = self.compute_psnr(reference_path, output_path)
            if psnr is not None:
                result.metrics["psnr"] = psnr
                if psnr < self.thresholds.min_psnr:
                    result.warnings.append(f"PSNR below threshold: {psnr:.2f} < {self.thresholds.min_psnr}")

        # Alpha matte quality
        if check_matte:
            frame_path = output_path.parent / f"_qa_{stage}_frame.png"
            extracted = self.extract_frame(output_path, frame_path, timestamp=1.0)
            if extracted:
                matte_metrics = self.check_alpha_matte_quality(extracted)
                result.metrics.update(matte_metrics)

                if matte_metrics.get("coverage", 0) < self.thresholds.min_alpha_coverage:
                    result.warnings.append("Matte coverage too low — may be all green")
                if matte_metrics.get("coverage", 1) > self.thresholds.max_alpha_coverage:
                    result.warnings.append("Matte coverage too high — may have failed")
                if matte_metrics.get("edge_sharpness", 999) < self.thresholds.min_edge_sharpness:
                    result.warnings.append("Matte edges too soft")

                # Clean up extracted frame
                extracted.unlink(missing_ok=True)

        logger.info("QA checkpoint: %s", result.summary())
        return result
