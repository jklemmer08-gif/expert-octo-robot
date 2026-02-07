"""Tests for LibraryScanner."""

from pathlib import Path

import pytest

from src.config import Settings
from src.database import JobDatabase
from src.watcher import LibraryScanner


@pytest.fixture
def scanner(test_settings, test_db):
    return LibraryScanner(test_settings, test_db)


def test_scan_empty_dir(scanner, tmp_path):
    """Scanning an empty directory creates no jobs."""
    (tmp_path / "library").mkdir()
    count = scanner.scan_directory(tmp_path / "library")
    assert count == 0


def test_scan_with_videos(scanner, test_db, tmp_path):
    """Scanning finds video files and creates jobs."""
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "video1.mp4").write_bytes(b"\x00" * 2048)
    (lib / "video2.mkv").write_bytes(b"\x01" * 2048)
    (lib / "readme.txt").write_text("not a video")

    count = scanner.scan_directory(lib)
    assert count == 2

    jobs = test_db.get_jobs()
    assert len(jobs) == 2


def test_scan_skip_duplicates(scanner, test_db, tmp_path):
    """Second scan of same directory doesn't create duplicates."""
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "video.mp4").write_bytes(b"\x00" * 2048)

    count1 = scanner.scan_directory(lib)
    assert count1 == 1

    count2 = scanner.scan_directory(lib)
    assert count2 == 0


def test_scan_skips_output_dir(scanner, test_settings, tmp_path):
    """Videos in the output directory are skipped."""
    output = Path(test_settings.paths.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "processed.mp4").write_bytes(b"\x00" * 2048)

    count = scanner.scan_directory(output)
    assert count == 0


def test_scan_nonexistent(scanner):
    """Scanning a nonexistent path returns 0."""
    count = scanner.scan_directory(Path("/nonexistent/path"))
    assert count == 0


def test_scan_nested(scanner, test_db, tmp_path):
    """Scanning recurses into subdirectories."""
    lib = tmp_path / "library"
    sub = lib / "subdir"
    sub.mkdir(parents=True)
    (sub / "deep.mp4").write_bytes(b"\x00" * 2048)

    count = scanner.scan_directory(lib)
    assert count == 1
