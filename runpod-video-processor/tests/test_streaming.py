"""Tests for streaming pipeline — pipe helpers, command builders, streaming functions.

All tests are mocked (no GPU or FFmpeg needed).
"""

import io
import subprocess
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

from src.utils.streaming import read_frame, read_frames, write_frame, close_process
from src.utils.ffmpeg import (
    build_decode_pipe_cmd,
    build_encode_pipe_cmd,
    build_encode_pipe_vp9_cmd,
)


class TestBuildDecodePipeCmd:
    """Test FFmpeg decode pipe command builder."""

    def test_basic_cmd(self):
        cmd = build_decode_pipe_cmd("/input.mp4", 0, 100, 30.0, 1920, 1080)
        assert cmd[0] == "ffmpeg"
        assert "-f" in cmd
        assert "rawvideo" in cmd
        assert "-pix_fmt" in cmd
        assert "bgr24" in cmd
        assert "pipe:1" in cmd
        assert "-frames:v" in cmd
        assert "100" in cmd

    def test_seek_with_start_frame(self):
        cmd = build_decode_pipe_cmd("/input.mp4", 300, 100, 30.0, 1920, 1080)
        assert "-ss" in cmd
        ss_idx = cmd.index("-ss")
        assert float(cmd[ss_idx + 1]) == pytest.approx(10.0, abs=0.01)

    def test_no_seek_at_start(self):
        cmd = build_decode_pipe_cmd("/input.mp4", 0, 100, 30.0, 1920, 1080)
        assert "-ss" not in cmd

    def test_input_path_in_cmd(self):
        cmd = build_decode_pipe_cmd("/path/to/video.mp4", 0, 50, 24.0, 640, 480)
        assert "/path/to/video.mp4" in cmd


class TestBuildEncodePipeCmd:
    """Test FFmpeg encode pipe command builders."""

    def test_hevc_nvenc_cmd(self):
        cmd = build_encode_pipe_cmd(30.0, 3840, 2160, 18, "/out.mkv", codec="hevc_nvenc")
        assert "hevc_nvenc" in cmd
        assert "-cq" in cmd
        assert "pipe:0" in cmd
        assert "bgr24" in cmd
        assert "3840x2160" in cmd

    def test_libx265_cmd(self):
        cmd = build_encode_pipe_cmd(30.0, 1920, 1080, 23, "/out.mkv", codec="libx265")
        assert "libx265" in cmd
        assert "-crf" in cmd
        assert "23" in cmd

    def test_vp9_cmd(self):
        cmd = build_encode_pipe_vp9_cmd(30.0, 1920, 1080, 30, "/out.webm")
        assert "libvpx-vp9" in cmd
        assert "bgra" in cmd
        assert "yuva420p" in cmd
        assert "pipe:0" in cmd
        assert "1920x1080" in cmd

    def test_vp9_speed_param(self):
        cmd = build_encode_pipe_vp9_cmd(30.0, 640, 480, 30, "/out.webm", speed=2)
        speed_idx = cmd.index("-speed")
        assert cmd[speed_idx + 1] == "2"

    def test_fps_in_encode_cmd(self):
        cmd = build_encode_pipe_cmd(59.94, 1920, 1080, 18, "/out.mkv")
        assert "59.94" in cmd


class TestReadFrame:
    """Test read_frame pipe helper."""

    def test_read_bgr_frame(self):
        """Read a 4x3 BGR frame from pipe."""
        width, height, channels = 4, 3, 3
        frame_data = np.arange(width * height * channels, dtype=np.uint8).tobytes()

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = frame_data

        result = read_frame(mock_proc, width, height, channels)

        assert result is not None
        assert result.shape == (height, width, channels)
        assert result.dtype == np.uint8
        mock_proc.stdout.read.assert_called_once_with(width * height * channels)

    def test_read_bgra_frame(self):
        """Read a 4-channel BGRA frame."""
        width, height, channels = 4, 3, 4
        frame_data = np.zeros(width * height * channels, dtype=np.uint8).tobytes()

        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = frame_data

        result = read_frame(mock_proc, width, height, channels)

        assert result is not None
        assert result.shape == (height, width, channels)

    def test_read_frame_end_of_stream(self):
        """Return None when pipe has no more data."""
        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b""

        result = read_frame(mock_proc, 4, 3, 3)
        assert result is None

    def test_read_frame_partial_data(self):
        """Return None on incomplete frame (truncated stream)."""
        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b"\x00" * 10  # too short

        result = read_frame(mock_proc, 100, 100, 3)
        assert result is None


class TestReadFrames:
    """Test batch read_frames helper."""

    def test_read_multiple_frames(self):
        width, height, channels = 4, 3, 3
        nbytes = width * height * channels
        frame_data = np.zeros(nbytes, dtype=np.uint8).tobytes()

        mock_proc = MagicMock()
        mock_proc.stdout.read.side_effect = [frame_data, frame_data, frame_data]

        result = read_frames(mock_proc, width, height, 3, channels)
        assert len(result) == 3

    def test_read_frames_early_end(self):
        width, height, channels = 4, 3, 3
        nbytes = width * height * channels
        frame_data = np.zeros(nbytes, dtype=np.uint8).tobytes()

        mock_proc = MagicMock()
        mock_proc.stdout.read.side_effect = [frame_data, b""]

        result = read_frames(mock_proc, width, height, 5, channels)
        assert len(result) == 1


class TestWriteFrame:
    """Test write_frame pipe helper."""

    def test_write_frame(self):
        frame = np.zeros((3, 4, 3), dtype=np.uint8)
        mock_proc = MagicMock()

        write_frame(mock_proc, frame)

        mock_proc.stdin.write.assert_called_once()
        written = mock_proc.stdin.write.call_args[0][0]
        assert len(written) == 3 * 4 * 3

    def test_write_bgra_frame(self):
        frame = np.zeros((3, 4, 4), dtype=np.uint8)
        mock_proc = MagicMock()

        write_frame(mock_proc, frame)

        written = mock_proc.stdin.write.call_args[0][0]
        assert len(written) == 3 * 4 * 4


class TestCloseProcess:
    """Test process cleanup helper."""

    def test_close_success(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdin.closed = False
        mock_proc.stdout.closed = False

        close_process(mock_proc, "test")

        mock_proc.stdin.close.assert_called_once()
        mock_proc.stdout.close.assert_called_once()
        mock_proc.wait.assert_called_once()

    def test_close_failure_raises(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdin = None
        mock_proc.stdout = None
        mock_proc.stderr.read.return_value = b"some error"

        with pytest.raises(RuntimeError, match="test failed"):
            close_process(mock_proc, "test")

    def test_close_timeout_kills(self):
        mock_proc = MagicMock()
        mock_proc.stdin = None
        mock_proc.stdout = None
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("ffmpeg", 300)

        with pytest.raises(RuntimeError, match="timed out"):
            close_process(mock_proc, "test")
        mock_proc.kill.assert_called_once()


class TestUpscalerStreaming:
    """Test process_video_streaming for the upscaler pipeline (mocked)."""

    def _make_mock_torch(self):
        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        mock_torch.cuda.empty_cache = MagicMock()
        return mock_torch

    def test_streaming_progress_stages(self, tmp_path):
        from src.pipeline.upscaler import process_video_streaming

        mock_torch = self._make_mock_torch()
        stages_seen = []

        def track_progress(update):
            stages_seen.append(update.get("stage"))

        mock_meta = {
            "width": 64, "height": 64, "fps": 30.0,
            "duration": 0.1, "num_frames": 3,
            "codec": "h264", "file_size": 1000,
        }

        # Mock upsampler
        mock_upsampler = MagicMock()
        upscaled_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_upsampler.enhance.return_value = (upscaled_frame, None)

        # Mock decode pipe returning 3 frames then None
        frame_data = np.zeros((64, 64, 3), dtype=np.uint8)
        frame_bytes = frame_data.tobytes()

        mock_decode_proc = MagicMock()
        mock_decode_proc.stdout.read.side_effect = [frame_bytes, frame_bytes, frame_bytes, b""]
        mock_decode_proc.returncode = 0
        mock_decode_proc.stdin = None
        mock_decode_proc.stderr = None

        mock_encode_proc = MagicMock()
        mock_encode_proc.returncode = 0
        mock_encode_proc.stdout = None
        mock_encode_proc.stderr = None
        mock_encode_proc.stdin.closed = False

        with patch("src.pipeline.upscaler.torch", mock_torch), \
             patch("src.pipeline.upscaler.get_video_metadata", return_value=mock_meta), \
             patch("src.pipeline.upscaler.check_disk_space", return_value=True), \
             patch("src.pipeline.upscaler._load_model", return_value=mock_upsampler), \
             patch("src.pipeline.upscaler.start_decode_process", return_value=mock_decode_proc), \
             patch("src.pipeline.upscaler.start_encode_process", return_value=mock_encode_proc), \
             patch("src.pipeline.upscaler.close_process"), \
             patch("src.pipeline.upscaler.read_vr_metadata", return_value={}), \
             patch("src.pipeline.upscaler.build_metadata_flags", return_value=[]), \
             patch("src.pipeline.upscaler.cleanup_job"), \
             patch("src.pipeline.encoder.concatenate_segments"), \
             patch("src.pipeline.encoder.mux_audio"), \
             patch("os.rename"):

            result = process_video_streaming(
                input_path="/fake/input.mp4",
                output_path=str(tmp_path / "output.mkv"),
                segment_size=1000,
                codec="libx265",
                progress_callback=track_progress,
            )

        assert result["status"] == "success"
        assert result["streaming"] is True
        assert "upscaling" in stages_seen
        assert "muxing_audio" in stages_seen

    def test_streaming_disk_space_error(self, tmp_path):
        from src.pipeline.upscaler import process_video_streaming

        mock_torch = self._make_mock_torch()
        mock_meta = {
            "width": 64, "height": 64, "fps": 30.0,
            "duration": 1.0, "num_frames": 10,
            "codec": "h264", "file_size": 1000,
        }

        with patch("src.pipeline.upscaler.torch", mock_torch), \
             patch("src.pipeline.upscaler.get_video_metadata", return_value=mock_meta), \
             patch("src.pipeline.upscaler.check_disk_space", return_value=False), \
             patch("src.pipeline.upscaler.cleanup_job"):
            result = process_video_streaming(
                input_path="/fake/input.mp4",
                output_path=str(tmp_path / "output.mkv"),
                codec="libx265",
            )

        assert result["status"] == "failed"
        assert "disk space" in result["error"].lower()

    def test_streaming_vr_sbs(self, tmp_path):
        """Streaming pipeline correctly processes VR SBS frames."""
        from src.pipeline.upscaler import process_video_streaming
        from src.pipeline.detector import VRLayout

        mock_torch = self._make_mock_torch()

        mock_meta = {
            "width": 128, "height": 64, "fps": 30.0,
            "duration": 0.1, "num_frames": 1,
            "codec": "h264", "file_size": 1000,
        }

        # SBS: each eye is 64x64, upscaled 2x = 128x128, merged = 256x128
        mock_upsampler = MagicMock()
        upscaled_eye = np.zeros((128, 128, 3), dtype=np.uint8)
        mock_upsampler.enhance.return_value = (upscaled_eye, None)

        frame_data = np.zeros((64, 128, 3), dtype=np.uint8).tobytes()

        mock_decode_proc = MagicMock()
        mock_decode_proc.stdout.read.side_effect = [frame_data, b""]
        mock_decode_proc.returncode = 0
        mock_decode_proc.stdin = None
        mock_decode_proc.stderr = None

        mock_encode_proc = MagicMock()
        mock_encode_proc.returncode = 0
        mock_encode_proc.stdout = None
        mock_encode_proc.stderr = None
        mock_encode_proc.stdin.closed = False

        with patch("src.pipeline.upscaler.torch", mock_torch), \
             patch("src.pipeline.upscaler.get_video_metadata", return_value=mock_meta), \
             patch("src.pipeline.upscaler.check_disk_space", return_value=True), \
             patch("src.pipeline.upscaler._load_model", return_value=mock_upsampler), \
             patch("src.pipeline.upscaler.start_decode_process", return_value=mock_decode_proc), \
             patch("src.pipeline.upscaler.start_encode_process", return_value=mock_encode_proc), \
             patch("src.pipeline.upscaler.close_process"), \
             patch("src.pipeline.upscaler.read_vr_metadata", return_value={}), \
             patch("src.pipeline.upscaler.build_metadata_flags", return_value=[]), \
             patch("src.pipeline.upscaler.cleanup_job"), \
             patch("src.pipeline.encoder.concatenate_segments"), \
             patch("src.pipeline.encoder.mux_audio"), \
             patch("os.rename"):

            result = process_video_streaming(
                input_path="/fake/input.mp4",
                output_path=str(tmp_path / "output.mkv"),
                scale=2,
                codec="libx265",
                layout=VRLayout.SBS,
            )

        assert result["status"] == "success"
        # Each eye upscaled: enhance called twice (left + right)
        assert mock_upsampler.enhance.call_count == 2


class TestBgRemoverStreaming:
    """Test process_video_streaming for the bgremover pipeline (mocked)."""

    def _make_mock_torch(self):
        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        mock_torch.cuda.empty_cache = MagicMock()
        return mock_torch

    def test_streaming_progress_stages(self, tmp_path):
        from src.pipeline.bgremover import process_video_streaming

        mock_torch = self._make_mock_torch()
        stages_seen = []

        def track_progress(update):
            stages_seen.append(update.get("stage"))

        mock_meta = {
            "width": 64, "height": 64, "fps": 30.0,
            "duration": 0.1, "num_frames": 2,
            "codec": "h264", "file_size": 1000,
        }

        mock_proc = MagicMock()
        mock_proc.get_recurrent_states.return_value = [None] * 4
        mock_proc.process_batch.return_value = MagicMock()

        # Mock decode pipe returning 2 frames
        frame_data = np.zeros((64, 64, 3), dtype=np.uint8).tobytes()

        mock_decode_proc = MagicMock()
        mock_decode_proc.stdout.read.side_effect = [frame_data, frame_data, b""]
        mock_decode_proc.returncode = 0
        mock_decode_proc.stdin = None
        mock_decode_proc.stderr = None

        mock_encode_proc = MagicMock()
        mock_encode_proc.returncode = 0
        mock_encode_proc.stdout = None
        mock_encode_proc.stderr = None
        mock_encode_proc.stdin.closed = False

        fake_bgra = [np.zeros((64, 64, 4), dtype=np.uint8) for _ in range(2)]

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover.get_video_metadata", return_value=mock_meta), \
             patch("src.pipeline.bgremover.check_disk_space", return_value=True), \
             patch("src.pipeline.bgremover.RVMProcessor", return_value=mock_proc), \
             patch("src.pipeline.bgremover.start_decode_process", return_value=mock_decode_proc), \
             patch("src.pipeline.bgremover.start_encode_process", return_value=mock_encode_proc), \
             patch("src.pipeline.bgremover.close_process"), \
             patch("src.pipeline.bgremover._frames_to_tensor"), \
             patch("src.pipeline.bgremover._tensor_to_frames_bgra", return_value=fake_bgra), \
             patch("src.pipeline.bgremover.cleanup_job"), \
             patch("src.pipeline.encoder.concatenate_segments"), \
             patch("src.pipeline.encoder.mux_audio_webm"), \
             patch("os.rename"):

            result = process_video_streaming(
                input_path="/fake/input.mp4",
                output_path=str(tmp_path / "output.webm"),
                segment_size=1000,
                progress_callback=track_progress,
            )

        assert result["status"] == "success"
        assert result["streaming"] is True
        assert "removing_background" in stages_seen
        assert "muxing_audio" in stages_seen

    def test_streaming_disk_space_error(self, tmp_path):
        from src.pipeline.bgremover import process_video_streaming

        mock_torch = self._make_mock_torch()
        mock_meta = {
            "width": 64, "height": 64, "fps": 30.0,
            "duration": 1.0, "num_frames": 10,
            "codec": "h264", "file_size": 1000,
        }

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover.get_video_metadata", return_value=mock_meta), \
             patch("src.pipeline.bgremover.check_disk_space", return_value=False), \
             patch("src.pipeline.bgremover.cleanup_job"):
            result = process_video_streaming(
                input_path="/fake/input.mp4",
                output_path=str(tmp_path / "output.webm"),
            )

        assert result["status"] == "failed"
        assert "disk space" in result["error"].lower()


class TestDiskEstimate:
    """Test that streaming disk estimates are much smaller than old PNG-based ones."""

    def test_estimate_smaller_than_before(self):
        from src.storage.volume import estimate_segment_disk_gb

        # 1000 frames of 1920x1080 at 4x scale
        gb = estimate_segment_disk_gb(1920, 1080, 1000, scale=4)
        # Old estimate would be ~30-40 GB. New should be well under 5 GB.
        assert gb < 5.0
        assert gb > 0

    def test_estimate_scales_with_resolution(self):
        from src.storage.volume import estimate_segment_disk_gb

        small = estimate_segment_disk_gb(640, 480, 100, scale=2)
        large = estimate_segment_disk_gb(3840, 2160, 100, scale=2)
        assert large > small
