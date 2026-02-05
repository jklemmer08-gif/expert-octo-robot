"""Tests for VR layout detection (filename, metadata, aspect ratio — all mocked)."""

import pytest
from unittest.mock import patch

from src.pipeline.detector import (
    VRLayout,
    detect_from_filename,
    detect_from_aspect_ratio,
    detect_from_metadata,
    detect_layout,
)


class TestFilenameDetection:
    """Filename-based layout detection."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("video_SBS_180.mp4", VRLayout.SBS),
            ("video_LR.mp4", VRLayout.SBS),
            ("video_3DH.mp4", VRLayout.SBS),
            ("video_HSBS.mp4", VRLayout.SBS),
            ("video-sbs-180x180.mp4", VRLayout.SBS),
            ("video side-by-side test.mp4", VRLayout.SBS),
            ("video_OU_180.mp4", VRLayout.OU),
            ("video_TB.mp4", VRLayout.OU),
            ("video_3DV.mp4", VRLayout.OU),
            ("video_HOU.mp4", VRLayout.OU),
            ("video-ou-180.mp4", VRLayout.OU),
            ("video over-under test.mp4", VRLayout.OU),
            ("video_MONO.mp4", VRLayout.MONO),
        ],
    )
    def test_known_patterns(self, filename, expected):
        assert detect_from_filename(filename) == expected

    def test_no_match(self):
        assert detect_from_filename("regular_video.mp4") is None

    def test_substring_no_false_positive(self):
        # "submission" contains "sbs" but shouldn't match
        assert detect_from_filename("submission_video.mp4") is None


class TestAspectRatioDetection:
    """Aspect ratio heuristic detection."""

    def test_sbs_180(self):
        # 2:1 aspect ratio → SBS 180°
        assert detect_from_aspect_ratio(3840, 1920) == VRLayout.SBS
        assert detect_from_aspect_ratio(5760, 2880) == VRLayout.SBS

    def test_sbs_360(self):
        # 4:1 aspect ratio → SBS 360°
        assert detect_from_aspect_ratio(7680, 1920) == VRLayout.SBS

    def test_ou(self):
        # 1:2 aspect ratio → OU
        assert detect_from_aspect_ratio(1920, 3840) == VRLayout.OU

    def test_standard_16_9(self):
        # 16:9 is just normal 2D
        assert detect_from_aspect_ratio(1920, 1080) is None

    def test_zero_dimensions(self):
        assert detect_from_aspect_ratio(0, 0) is None
        assert detect_from_aspect_ratio(1920, 0) is None


class TestMetadataDetection:
    """Container metadata / side data detection (mocked ffprobe)."""

    def test_sbs_side_data(self):
        side_data = [
            {"side_data_type": "Stereo 3D", "type": "side_by_side"}
        ]
        with patch("src.pipeline.detector.get_stream_side_data", return_value=side_data), \
             patch("src.pipeline.detector.get_all_stream_info", return_value={}):
            assert detect_from_metadata("test.mp4") == VRLayout.SBS

    def test_ou_side_data(self):
        side_data = [
            {"side_data_type": "Stereo 3D", "type": "top_bottom"}
        ]
        with patch("src.pipeline.detector.get_stream_side_data", return_value=side_data), \
             patch("src.pipeline.detector.get_all_stream_info", return_value={}):
            assert detect_from_metadata("test.mp4") == VRLayout.OU

    def test_stereo_mode_stream_tag(self):
        info = {
            "streams": [
                {
                    "codec_type": "video",
                    "tags": {"stereo_mode": "side_by_side"},
                }
            ],
            "format": {},
        }
        with patch("src.pipeline.detector.get_stream_side_data", return_value=[]), \
             patch("src.pipeline.detector.get_all_stream_info", return_value=info):
            assert detect_from_metadata("test.mp4") == VRLayout.SBS

    def test_no_metadata(self):
        with patch("src.pipeline.detector.get_stream_side_data", return_value=[]), \
             patch("src.pipeline.detector.get_all_stream_info", return_value={"streams": [], "format": {}}):
            assert detect_from_metadata("test.mp4") is None


class TestDetectLayout:
    """Full detection priority chain."""

    def test_filename_takes_priority(self):
        with patch("src.pipeline.detector.detect_from_metadata", return_value=VRLayout.OU):
            # Filename says SBS, metadata says OU → filename wins
            result = detect_layout("video_SBS.mp4", 3840, 1920)
        assert result == VRLayout.SBS

    def test_metadata_over_aspect_ratio(self):
        with patch("src.pipeline.detector.detect_from_metadata", return_value=VRLayout.OU):
            # Aspect ratio says SBS (2:1) but metadata says OU → metadata wins
            result = detect_layout("regular.mp4", 3840, 1920)
        assert result == VRLayout.OU

    def test_falls_through_to_aspect_ratio(self):
        with patch("src.pipeline.detector.detect_from_metadata", return_value=None):
            result = detect_layout("regular.mp4", 3840, 1920)
        assert result == VRLayout.SBS

    def test_defaults_to_2d(self):
        with patch("src.pipeline.detector.detect_from_metadata", return_value=None):
            result = detect_layout("regular.mp4", 1920, 1080)
        assert result == VRLayout.FLAT_2D
