"""Tests for VideoAnalyzer (VR detection, quality scoring)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analyzer import VideoAnalyzer
from src.config import Settings
from src.models.schemas import ContentType, VideoInfo


@pytest.fixture
def analyzer(test_settings):
    return VideoAnalyzer(test_settings)


class TestVRDetection:
    """Test VR detection from various filename patterns."""

    def test_180_sbs(self, analyzer):
        is_vr, vr_type = analyzer.detect_vr(Path("video_180_sbs.mp4"), 3840, 1920)
        assert is_vr is True
        assert vr_type == "sbs"

    def test_360_tb(self, analyzer):
        is_vr, vr_type = analyzer.detect_vr(Path("video_360_tb.mp4"), 3840, 1920)
        assert is_vr is True
        assert vr_type == "tb"

    def test_vr_tag(self, analyzer):
        is_vr, vr_type = analyzer.detect_vr(Path("video_VR.mp4"), 1920, 1080)
        assert is_vr is True

    def test_studio_pattern(self, analyzer):
        is_vr, vr_type = analyzer.detect_vr(Path("VRBangers_scene.mp4"), 3840, 1920)
        assert is_vr is True
        assert vr_type == "sbs"

    def test_aspect_ratio(self, analyzer):
        is_vr, vr_type = analyzer.detect_vr(Path("unknown_video.mp4"), 3840, 1920)
        assert is_vr is True  # 2:1 aspect ratio

    def test_known_resolution(self, analyzer):
        is_vr, vr_type = analyzer.detect_vr(Path("scene.mp4"), 7680, 3840)
        assert is_vr is True

    def test_not_vr(self, analyzer):
        is_vr, vr_type = analyzer.detect_vr(Path("movie.mp4"), 1920, 1080)
        assert is_vr is False
        assert vr_type is None

    def test_fisheye(self, analyzer):
        is_vr, vr_type = analyzer.detect_vr(Path("scene_fisheye.mp4"), 3840, 1920)
        assert is_vr is True


class TestQualityScoring:

    def test_high_quality_4k(self, analyzer):
        info = VideoInfo(
            width=3840, height=2160, fps=60, duration=600,
            codec="hevc", bitrate=50_000_000, is_vr=False,
        )
        score = analyzer.calculate_quality_score(info)
        assert score >= 60

    def test_low_quality_480p(self, analyzer):
        info = VideoInfo(
            width=854, height=480, fps=24, duration=600,
            codec="mpeg4", bitrate=2_000_000, is_vr=False,
        )
        score = analyzer.calculate_quality_score(info)
        assert score < 30

    def test_vr_bonus(self, analyzer):
        base = VideoInfo(
            width=1920, height=1080, fps=30, duration=600,
            codec="h264", bitrate=10_000_000, is_vr=False,
        )
        vr = VideoInfo(
            width=1920, height=1080, fps=30, duration=600,
            codec="h264", bitrate=10_000_000, is_vr=True,
        )
        assert analyzer.calculate_quality_score(vr) > analyzer.calculate_quality_score(base)


class TestShouldUpscale:

    def test_already_8k(self, analyzer):
        info = VideoInfo(
            width=7680, height=4320, fps=30, duration=600,
            codec="hevc", bitrate=100_000_000,
        )
        assert analyzer.should_upscale(info) is False

    def test_too_short(self, analyzer):
        info = VideoInfo(
            width=1920, height=1080, fps=30, duration=3,
            codec="h264", bitrate=10_000_000,
        )
        assert analyzer.should_upscale(info) is False

    def test_good_candidate(self, analyzer):
        info = VideoInfo(
            width=1920, height=1080, fps=30, duration=600,
            codec="h264", bitrate=10_000_000,
        )
        assert analyzer.should_upscale(info) is True
