"""Tests for input validation (all mocked — no GPU or ffprobe needed)."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.pipeline.validator import validate_input, ValidationResult


@pytest.fixture
def mock_metadata():
    """Standard valid video metadata."""
    return {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "duration": 600.0,  # 10 minutes
        "codec": "h264",
        "num_frames": 18000,
        "file_size": 5 * 1024 ** 3,  # 5 GB
    }


class TestFileExistence:
    def test_missing_file(self, tmp_path):
        result = validate_input(str(tmp_path / "nonexistent.mp4"))
        assert not result.valid
        assert "not found" in result.errors[0].lower()

    def test_directory_not_file(self, tmp_path):
        result = validate_input(str(tmp_path))
        assert not result.valid
        assert "not a file" in result.errors[0].lower()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.mp4"
        f.write_bytes(b"")
        result = validate_input(str(f))
        assert not result.valid
        assert "empty" in result.errors[0].lower()


class TestFileSize:
    def test_file_too_large(self, tmp_path, mock_metadata):
        f = tmp_path / "big.mp4"
        f.write_bytes(b"x" * 1024)
        # Set the limit absurdly low so our small file exceeds it
        with patch("src.pipeline.validator.MAX_FILE_SIZE_GB", 0.000000001):
            result = validate_input(str(f))
        assert not result.valid
        assert "too large" in result.errors[0].lower()


class TestCodecValidation:
    def test_supported_codecs(self, tmp_path, mock_metadata):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"x" * 1024)

        for codec in ["h264", "hevc", "h265", "av1", "vp9"]:
            meta = {**mock_metadata, "codec": codec}
            with patch("src.pipeline.validator.get_video_metadata", return_value=meta):
                result = validate_input(str(f))
            assert result.valid, f"Codec {codec} should be valid"

    def test_unsupported_codec(self, tmp_path, mock_metadata):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"x" * 1024)
        meta = {**mock_metadata, "codec": "mpeg2video"}
        with patch("src.pipeline.validator.get_video_metadata", return_value=meta):
            result = validate_input(str(f))
        assert not result.valid
        assert "unsupported codec" in result.errors[0].lower()
        assert "remux" in result.errors[0].lower()

    def test_no_codec_detected(self, tmp_path, mock_metadata):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"x" * 1024)
        meta = {**mock_metadata, "codec": ""}
        with patch("src.pipeline.validator.get_video_metadata", return_value=meta):
            result = validate_input(str(f))
        assert not result.valid
        assert "no video codec" in result.errors[0].lower()


class TestDuration:
    def test_too_long(self, tmp_path, mock_metadata):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"x" * 1024)
        meta = {**mock_metadata, "duration": 200 * 60}  # 200 minutes
        with patch("src.pipeline.validator.get_video_metadata", return_value=meta):
            result = validate_input(str(f))
        assert not result.valid
        assert "too long" in result.errors[0].lower()

    def test_zero_duration_warning(self, tmp_path, mock_metadata):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"x" * 1024)
        meta = {**mock_metadata, "duration": 0}
        with patch("src.pipeline.validator.get_video_metadata", return_value=meta):
            result = validate_input(str(f))
        assert result.valid
        assert any("duration" in w.lower() for w in result.warnings)


class TestCorruptFile:
    def test_ffprobe_failure(self, tmp_path):
        f = tmp_path / "corrupt.mp4"
        f.write_bytes(b"x" * 1024)
        with patch(
            "src.pipeline.validator.get_video_metadata",
            side_effect=RuntimeError("ffprobe failed"),
        ):
            result = validate_input(str(f))
        assert not result.valid
        assert "corrupt" in result.errors[0].lower()


class TestResolution:
    def test_zero_resolution(self, tmp_path, mock_metadata):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"x" * 1024)
        meta = {**mock_metadata, "width": 0, "height": 0}
        with patch("src.pipeline.validator.get_video_metadata", return_value=meta):
            result = validate_input(str(f))
        assert not result.valid
        assert "resolution" in result.errors[0].lower()


class TestValidFile:
    def test_valid_file(self, tmp_path, mock_metadata):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"x" * 1024)
        with patch("src.pipeline.validator.get_video_metadata", return_value=mock_metadata):
            result = validate_input(str(f))
        assert result.valid
        assert result.metadata is not None
        assert result.metadata["width"] == 1920
