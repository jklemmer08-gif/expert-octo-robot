"""Tests for the processing pipeline components."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.models.schemas import ContentType, ProcessingPlan, VideoInfo
from src.processor import Encoder, FrameExtractor, ProcessingPipeline, UpscaleEngine, VRProcessor


class TestFrameExtractor:

    @patch("src.processor.subprocess.run")
    def test_extract_calls_ffmpeg(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        ext = FrameExtractor()
        output = tmp_path / "frames"
        # Create a fake frame to simulate extraction
        output.mkdir()
        (output / "frame_00000001.png").write_bytes(b"\x00")

        count = ext.extract(Path("/fake/video.mp4"), output, fps=30.0)
        assert mock_run.called
        assert count == 1

    @patch("src.processor.subprocess.run")
    def test_extract_failure_raises(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")
        ext = FrameExtractor()
        with pytest.raises(RuntimeError, match="Frame extraction failed"):
            ext.extract(Path("/fake/video.mp4"), tmp_path / "frames")


class TestVRProcessor:

    def test_split_and_merge_sbs(self, tmp_path):
        """Split a fake SBS frame and merge it back."""
        from PIL import Image

        frames = tmp_path / "frames"
        frames.mkdir()

        # Create a 200x100 "SBS" image (left half blue, right half red)
        img = Image.new("RGB", (200, 100), (0, 0, 255))
        for x in range(100, 200):
            for y in range(100):
                img.putpixel((x, y), (255, 0, 0))
        img.save(frames / "frame_00000001.png")
        img.close()

        left = tmp_path / "left"
        right = tmp_path / "right"

        vr = VRProcessor()
        vr.split_sbs(frames, left, right)

        assert (left / "frame_00000001.png").exists()
        assert (right / "frame_00000001.png").exists()

        # Check dimensions
        l_img = Image.open(left / "frame_00000001.png")
        assert l_img.size == (100, 100)
        l_img.close()

        # Merge back
        merged = tmp_path / "merged"
        vr.merge_sbs(left, right, merged)

        m_img = Image.open(merged / "frame_00000001.png")
        assert m_img.size == (200, 100)
        m_img.close()

    def test_split_and_merge_tb(self, tmp_path):
        """Split a fake TB frame and merge it back."""
        from PIL import Image

        frames = tmp_path / "frames"
        frames.mkdir()

        img = Image.new("RGB", (100, 200), (0, 255, 0))
        img.save(frames / "frame_00000001.png")
        img.close()

        top = tmp_path / "top"
        bottom = tmp_path / "bottom"

        vr = VRProcessor()
        vr.split_tb(frames, top, bottom)

        t_img = Image.open(top / "frame_00000001.png")
        assert t_img.size == (100, 100)
        t_img.close()

        merged = tmp_path / "merged"
        vr.merge_tb(top, bottom, merged)

        m_img = Image.open(merged / "frame_00000001.png")
        assert m_img.size == (100, 200)
        m_img.close()


class TestEncoder:

    @patch("src.processor.subprocess.run")
    def test_encode_calls_ffmpeg(self, mock_run, test_settings, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        enc = Encoder(test_settings)

        frames = tmp_path / "frames"
        frames.mkdir()
        output = tmp_path / "output.mp4"

        result = enc.encode(frames, output, fps=30.0, bitrate="100M")
        assert result is True
        assert mock_run.called

    @patch("src.processor.subprocess.run")
    def test_encode_fallback(self, mock_run, test_settings, tmp_path):
        """If first encoder fails, fallback should be tried."""
        # First call fails, second succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr="encoder not found"),
            MagicMock(returncode=0),
        ]
        enc = Encoder(test_settings)
        frames = tmp_path / "frames"
        frames.mkdir()
        output = tmp_path / "output.mp4"

        result = enc.encode(frames, output, fps=30.0, encoder="hevc_vaapi")
        assert result is True
        assert mock_run.call_count == 2
