"""Tests for VR metadata read/write (mocked ffprobe)."""

import pytest
from unittest.mock import patch

from src.pipeline.metadata import read_vr_metadata, build_metadata_flags, get_metadata_summary
from src.pipeline.detector import VRLayout


class TestReadVRMetadata:
    def test_reads_stereo_side_data(self):
        side_data = [
            {"side_data_type": "Stereo 3D", "type": "side_by_side"}
        ]
        info = {"streams": [{"codec_type": "video", "tags": {}}], "format": {}}

        with patch("src.pipeline.metadata.get_stream_side_data", return_value=side_data), \
             patch("src.pipeline.metadata.get_all_stream_info", return_value=info):
            meta = read_vr_metadata("test.mp4")

        assert meta["stereo_mode"] == "side_by_side"

    def test_reads_spherical_side_data(self):
        side_data = [
            {
                "side_data_type": "Spherical Mapping",
                "projection": "equirectangular",
                "yaw": 0,
                "pitch": 0,
                "roll": 0,
            }
        ]
        info = {"streams": [{"codec_type": "video", "tags": {}}], "format": {}}

        with patch("src.pipeline.metadata.get_stream_side_data", return_value=side_data), \
             patch("src.pipeline.metadata.get_all_stream_info", return_value=info):
            meta = read_vr_metadata("test.mp4")

        assert meta["projection"] == "equirectangular"
        assert "yaw" in meta["spherical_tags"]

    def test_reads_stream_tags(self):
        info = {
            "streams": [
                {
                    "codec_type": "video",
                    "tags": {"stereo_mode": "top_bottom", "projection_type": "equirectangular"},
                }
            ],
            "format": {},
        }

        with patch("src.pipeline.metadata.get_stream_side_data", return_value=[]), \
             patch("src.pipeline.metadata.get_all_stream_info", return_value=info):
            meta = read_vr_metadata("test.mp4")

        assert meta["stereo_mode"] == "top_bottom"
        assert meta["stream_tags"]["projection_type"] == "equirectangular"

    def test_empty_metadata(self):
        with patch("src.pipeline.metadata.get_stream_side_data", return_value=[]), \
             patch("src.pipeline.metadata.get_all_stream_info", return_value={"streams": [], "format": {}}):
            meta = read_vr_metadata("test.mp4")

        assert meta["stereo_mode"] is None
        assert meta["projection"] is None


class TestBuildMetadataFlags:
    def test_sbs_flags(self):
        meta = {"stereo_mode": "side_by_side", "stream_tags": {}}
        flags = build_metadata_flags(meta, VRLayout.SBS)
        assert "-metadata:s:v" in flags
        assert "stereo_mode=side_by_side" in flags

    def test_ou_flags(self):
        meta = {"stereo_mode": "top_bottom", "stream_tags": {}}
        flags = build_metadata_flags(meta, VRLayout.OU)
        assert "stereo_mode=top_bottom" in flags

    def test_infers_stereo_mode_from_layout(self):
        meta = {"stereo_mode": None, "stream_tags": {}}
        flags = build_metadata_flags(meta, VRLayout.SBS)
        assert "stereo_mode=side_by_side" in flags

    def test_preserves_vr_tags(self):
        meta = {
            "stereo_mode": "side_by_side",
            "stream_tags": {
                "projection_type": "equirectangular",
                "stereo_mode": "side_by_side",
                "title": "My Video",  # non-VR tag, should be excluded
            },
        }
        flags = build_metadata_flags(meta, VRLayout.SBS)
        assert "projection_type=equirectangular" in flags
        # "title" is not a VR tag, should not be included
        assert not any("title=" in f for f in flags)

    def test_flat_2d_no_flags(self):
        meta = {"stereo_mode": None, "stream_tags": {}}
        flags = build_metadata_flags(meta, VRLayout.FLAT_2D)
        assert flags == []


class TestGetMetadataSummary:
    def test_summary(self):
        meta = {
            "stereo_mode": "side_by_side",
            "projection": "equirectangular",
            "spherical_tags": {"yaw": 0},
            "raw_side_data": [],
            "stream_tags": {"stereo_mode": "side_by_side"},
        }
        with patch("src.pipeline.metadata.read_vr_metadata", return_value=meta):
            summary = get_metadata_summary("test.mp4", VRLayout.SBS)

        assert summary["layout"] == "sbs"
        assert summary["stereo_mode"] == "side_by_side"
        assert summary["has_spherical_tags"] is True
