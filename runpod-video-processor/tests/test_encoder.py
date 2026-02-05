"""Tests for FFmpeg encoding — command generation, codec selection, NVENC fallback."""

import pytest
from unittest.mock import patch, MagicMock

from src.pipeline.encoder import get_encoder_codec, encode_segment, concatenate_segments, mux_audio
from src.utils.ffmpeg import build_encode_segment_cmd, build_concat_cmd, build_mux_audio_cmd


class TestCodecSelection:
    def setup_method(self):
        # Reset cached NVENC state
        import src.pipeline.encoder as enc
        enc._nvenc_available = None

    def test_nvenc_available(self):
        with patch("src.pipeline.encoder.check_nvenc_available", return_value=True):
            codec = get_encoder_codec()
        assert codec == "hevc_nvenc"

    def test_nvenc_unavailable_fallback(self):
        with patch("src.pipeline.encoder.check_nvenc_available", return_value=False):
            codec = get_encoder_codec()
        assert codec == "libx265"


class TestBuildEncodeCmd:
    def test_nvenc_cmd(self):
        cmd = build_encode_segment_cmd(
            "frames/frame_%06d.png", "out.mkv", 30.0, 3840, 2160,
            crf=18, codec="hevc_nvenc",
        )
        assert "hevc_nvenc" in cmd
        assert "-cq" in cmd  # NVENC uses -cq, not -crf
        assert "18" in cmd

    def test_libx265_cmd(self):
        cmd = build_encode_segment_cmd(
            "frames/frame_%06d.png", "out.mkv", 30.0, 3840, 2160,
            crf=18, codec="libx265", preset="slow",
        )
        assert "libx265" in cmd
        assert "-crf" in cmd
        assert "-preset" in cmd
        assert "slow" in cmd

    def test_fps_in_cmd(self):
        cmd = build_encode_segment_cmd(
            "frames/frame_%06d.png", "out.mkv", 59.94, 1920, 1080,
        )
        assert "59.94" in cmd

    def test_output_format(self):
        cmd = build_encode_segment_cmd(
            "frames/frame_%06d.png", "output.mkv", 30.0, 1920, 1080,
        )
        assert cmd[-1] == "output.mkv"


class TestBuildConcatCmd:
    def test_concat_cmd(self):
        cmd = build_concat_cmd("concat_list.txt", "output.mkv")
        assert "-f" in cmd
        assert "concat" in cmd
        assert "-safe" in cmd
        assert "concat_list.txt" in cmd

    def test_stream_copy(self):
        cmd = build_concat_cmd("list.txt", "out.mkv")
        assert "-c" in cmd
        assert "copy" in cmd


class TestBuildMuxAudioCmd:
    def test_basic_mux(self):
        cmd = build_mux_audio_cmd("video.mkv", "original.mp4", "final.mkv")
        assert "-map" in cmd
        assert "0:v" in cmd
        assert "1:a" in cmd
        assert cmd[-1] == "final.mkv"

    def test_extra_flags(self):
        cmd = build_mux_audio_cmd(
            "video.mkv", "original.mp4", "final.mkv",
            extra_flags=["-metadata:s:v", "stereo_mode=side_by_side"],
        )
        assert "stereo_mode=side_by_side" in cmd


class TestCRFValidation:
    """Verify CRF values are passed correctly."""

    @pytest.mark.parametrize("crf", [15, 18, 23, 28])
    def test_crf_range(self, crf):
        cmd = build_encode_segment_cmd(
            "f/%06d.png", "out.mkv", 30.0, 1920, 1080,
            crf=crf, codec="libx265",
        )
        crf_idx = cmd.index("-crf")
        assert cmd[crf_idx + 1] == str(crf)


class TestEncodeFunctions:
    """Test encode_segment, concatenate, mux_audio with mocked FFmpeg."""

    def test_encode_segment_success(self, tmp_path):
        with patch("src.pipeline.encoder.get_encoder_codec", return_value="libx265"), \
             patch("src.pipeline.encoder.run_ffmpeg") as mock_run:
            result = encode_segment(
                str(tmp_path / "frame_%06d.png"),
                str(tmp_path / "segment.mkv"),
                fps=30.0, width=1920, height=1080,
            )
        assert result == str(tmp_path / "segment.mkv")
        mock_run.assert_called_once()

    def test_encode_segment_nvenc_fallback(self, tmp_path):
        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and "hevc_nvenc" in cmd:
                raise RuntimeError("NVENC init failed")
            return MagicMock()

        with patch("src.pipeline.encoder.get_encoder_codec", return_value="hevc_nvenc"), \
             patch("src.pipeline.encoder.run_ffmpeg", side_effect=side_effect):
            result = encode_segment(
                str(tmp_path / "frame_%06d.png"),
                str(tmp_path / "segment.mkv"),
                fps=30.0, width=1920, height=1080,
            )
        assert call_count == 2  # first NVENC, then libx265

    def test_mux_audio_no_audio(self, tmp_path):
        video = tmp_path / "video.mkv"
        video.touch()
        output = tmp_path / "final.mkv"

        with patch("src.pipeline.encoder.has_audio", return_value=False):
            result = mux_audio(str(video), "original.mp4", str(output))
        assert result == str(output)

    def test_concatenate_segments(self, tmp_path):
        seg1 = tmp_path / "seg1.mkv"
        seg2 = tmp_path / "seg2.mkv"
        seg1.touch()
        seg2.touch()

        with patch("src.pipeline.encoder.run_ffmpeg"):
            result = concatenate_segments(
                [str(seg1), str(seg2)],
                str(tmp_path / "output.mkv"),
                str(tmp_path),
            )
        assert result == str(tmp_path / "output.mkv")
        # Verify concat list was written
        concat_list = tmp_path / "concat_list.txt"
        assert concat_list.exists()
        content = concat_list.read_text()
        assert "seg1.mkv" in content
        assert "seg2.mkv" in content
