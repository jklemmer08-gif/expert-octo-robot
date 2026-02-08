"""Tests for QA checkpoint framework."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.qa import QACheckpoint, QAResult, QAThresholds


class TestQAThresholds:

    def test_default_thresholds(self):
        t = QAThresholds()
        assert t.min_ssim == 0.97
        assert t.min_psnr == 30.0
        assert t.max_frame_count_drift == 2
        assert t.min_file_size_bytes == 1024

    def test_custom_thresholds(self):
        t = QAThresholds(min_ssim=0.95, min_psnr=25.0)
        assert t.min_ssim == 0.95
        assert t.min_psnr == 25.0


class TestQAResult:

    def test_passed_result(self):
        r = QAResult(stage="test", passed=True, metrics={"ssim": 0.99})
        assert r.passed is True
        assert "PASS" in r.summary()

    def test_failed_result(self):
        r = QAResult(stage="test", passed=False, errors=["File missing"])
        assert r.passed is False
        assert "FAIL" in r.summary()
        assert "File missing" in r.summary()


class TestQACheckpoint:

    @patch("src.qa.subprocess.run")
    def test_extract_frame(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        output = tmp_path / "frame.png"
        output.write_bytes(b"\x89PNG")  # Fake PNG

        qa = QACheckpoint()
        result = qa.extract_frame(Path("/fake/video.mp4"), output, timestamp=5.0)
        assert result == output
        assert mock_run.called

    @patch("src.qa.subprocess.run")
    def test_extract_frame_failure(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        qa = QACheckpoint()
        output = tmp_path / "frame.png"
        result = qa.extract_frame(Path("/fake/video.mp4"), output)
        assert result is None

    @patch("src.qa.subprocess.run")
    def test_compute_ssim(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="[Parsed_ssim_0 @ 0x1234] SSIM All:0.987654 (18.123456)\n",
        )
        qa = QACheckpoint()
        ssim = qa.compute_ssim(Path("/ref.mp4"), Path("/dist.mp4"))
        assert ssim is not None
        assert abs(ssim - 0.987654) < 0.001

    @patch("src.qa.subprocess.run")
    def test_compute_psnr(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="[Parsed_psnr_0 @ 0x1234] PSNR average:38.1234 min:35.0\n",
        )
        qa = QACheckpoint()
        psnr = qa.compute_psnr(Path("/ref.mp4"), Path("/dist.mp4"))
        assert psnr is not None
        assert abs(psnr - 38.1234) < 0.01

    @patch("src.qa.subprocess.run")
    def test_verify_frame_count_match(self, mock_run):
        import json
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"nb_read_frames": "900"}]}),
        )
        qa = QACheckpoint()
        passed, actual = qa.verify_frame_count(Path("/fake.mp4"), expected=900)
        assert passed is True
        assert actual == 900

    @patch("src.qa.subprocess.run")
    def test_verify_frame_count_drift(self, mock_run):
        import json
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"nb_read_frames": "910"}]}),
        )
        qa = QACheckpoint()
        passed, actual = qa.verify_frame_count(Path("/fake.mp4"), expected=900)
        assert passed is False
        assert actual == 910

    @patch("src.qa.subprocess.run")
    def test_verify_resolution(self, mock_run):
        import json
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"width": 3840, "height": 2160}]}),
        )
        qa = QACheckpoint()
        passed, w, h = qa.verify_resolution(Path("/fake.mp4"), 3840, 2160)
        assert passed is True
        assert w == 3840
        assert h == 2160

    def test_run_stage_checkpoint_missing_file(self, tmp_path):
        qa = QACheckpoint()
        result = qa.run_stage_checkpoint(
            "post-encode",
            tmp_path / "nonexistent.mp4",
        )
        assert result.passed is False
        assert any("missing" in e.lower() for e in result.errors)

    def test_run_stage_checkpoint_small_file(self, tmp_path):
        small_file = tmp_path / "tiny.mp4"
        small_file.write_bytes(b"\x00" * 100)

        qa = QACheckpoint()
        result = qa.run_stage_checkpoint("post-encode", small_file)
        assert result.passed is False
        assert any("too small" in e.lower() for e in result.errors)

    def test_run_stage_checkpoint_valid_file(self, tmp_path):
        valid_file = tmp_path / "output.mp4"
        valid_file.write_bytes(b"\x00" * 2048)

        qa = QACheckpoint()
        result = qa.run_stage_checkpoint("post-encode", valid_file)
        assert result.passed is True
        assert result.metrics["file_size_mb"] > 0
