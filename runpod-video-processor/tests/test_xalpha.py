"""Tests for HereSphere External Alpha (_XALPHA) pipeline.

Covers: filename conventions, alpha dimensions, tensor conversions,
FFmpeg command builders (HEVC + keyint), OOM retry for split mode,
and the full streaming xalpha function (mocked).
"""

import io
import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.pipeline.detector import VRLayout
from src.utils.ffmpeg import build_encode_pipe_hevc_cmd, build_mux_audio_mp4_cmd


# ---------------------------------------------------------------------------
# Filename conventions
# ---------------------------------------------------------------------------
class TestComputeXalphaPaths:
    """compute_xalpha_paths: VR suffix logic, no double-suffixing."""

    def test_sbs_adds_lr_suffix(self, tmp_path):
        from src.pipeline.bgremover import compute_xalpha_paths
        main, alpha = compute_xalpha_paths(
            "/input/scene_12345.mp4", VRLayout.SBS, tmp_path,
        )
        assert main == str(tmp_path / "scene_12345_LR.mp4")
        assert alpha == str(tmp_path / "scene_12345_LR_XALPHA.mp4")

    def test_ou_adds_tb_suffix(self, tmp_path):
        from src.pipeline.bgremover import compute_xalpha_paths
        main, alpha = compute_xalpha_paths(
            "/input/video_180.mp4", VRLayout.OU, tmp_path,
        )
        assert main == str(tmp_path / "video_180_TB.mp4")
        assert alpha == str(tmp_path / "video_180_TB_XALPHA.mp4")

    def test_2d_no_vr_suffix(self, tmp_path):
        from src.pipeline.bgremover import compute_xalpha_paths
        main, alpha = compute_xalpha_paths(
            "/input/flat_video.mp4", VRLayout.FLAT_2D, tmp_path,
        )
        assert main == str(tmp_path / "flat_video.mp4")
        assert alpha == str(tmp_path / "flat_video_XALPHA.mp4")

    def test_no_double_suffix_lr(self, tmp_path):
        from src.pipeline.bgremover import compute_xalpha_paths
        main, alpha = compute_xalpha_paths(
            "/input/scene_FISHEYE190_LR.mp4", VRLayout.SBS, tmp_path,
        )
        assert "_LR_LR" not in main
        assert main == str(tmp_path / "scene_FISHEYE190_LR.mp4")
        assert alpha == str(tmp_path / "scene_FISHEYE190_LR_XALPHA.mp4")

    def test_no_double_suffix_sbs(self, tmp_path):
        from src.pipeline.bgremover import compute_xalpha_paths
        main, alpha = compute_xalpha_paths(
            "/input/video_SBS.mp4", VRLayout.SBS, tmp_path,
        )
        assert "_LR" not in main
        assert main == str(tmp_path / "video_SBS.mp4")

    def test_no_double_suffix_3dh(self, tmp_path):
        from src.pipeline.bgremover import compute_xalpha_paths
        main, alpha = compute_xalpha_paths(
            "/input/video_3DH.mp4", VRLayout.SBS, tmp_path,
        )
        assert "_LR" not in main

    def test_no_double_suffix_tb(self, tmp_path):
        from src.pipeline.bgremover import compute_xalpha_paths
        main, alpha = compute_xalpha_paths(
            "/input/video_TB.mp4", VRLayout.OU, tmp_path,
        )
        assert "_TB_TB" not in main

    def test_no_double_suffix_ou(self, tmp_path):
        from src.pipeline.bgremover import compute_xalpha_paths
        main, alpha = compute_xalpha_paths(
            "/input/video_OU.mp4", VRLayout.OU, tmp_path,
        )
        assert "_TB" not in main


# ---------------------------------------------------------------------------
# Alpha dimension calculations
# ---------------------------------------------------------------------------
class TestComputeAlphaDimensions:
    """compute_alpha_dimensions: aspect ratio + even values."""

    def test_16_9_aspect(self):
        from src.pipeline.bgremover import compute_alpha_dimensions
        w, h = compute_alpha_dimensions(1920, 1080, 480)
        assert h == 480
        assert w % 2 == 0
        assert abs(w / h - 1920 / 1080) < 0.02

    def test_sbs_double_wide(self):
        from src.pipeline.bgremover import compute_alpha_dimensions
        w, h = compute_alpha_dimensions(3840, 1080, 480)
        assert h == 480
        assert w % 2 == 0
        assert w > 480  # wider than tall

    def test_square(self):
        from src.pipeline.bgremover import compute_alpha_dimensions
        w, h = compute_alpha_dimensions(1000, 1000, 480)
        assert h == 480
        assert w == 480

    def test_even_enforcement(self):
        """Odd alpha_height gets rounded up to even."""
        from src.pipeline.bgremover import compute_alpha_dimensions
        w, h = compute_alpha_dimensions(1920, 1080, 479)
        assert h % 2 == 0
        assert w % 2 == 0


# ---------------------------------------------------------------------------
# FFmpeg command builders
# ---------------------------------------------------------------------------
class TestBuildEncodeHevcCmd:
    """build_encode_pipe_hevc_cmd: keyint, codec selection, resolution."""

    def test_hevc_nvenc_with_keyint(self):
        cmd = build_encode_pipe_hevc_cmd(
            30.0, 1920, 1080, 20, "/out.mp4", keyint=60, codec="hevc_nvenc",
        )
        assert "hevc_nvenc" in cmd
        assert "-g" in cmd
        g_idx = cmd.index("-g")
        assert cmd[g_idx + 1] == "60"
        assert "-keyint_min" in cmd
        kmin_idx = cmd.index("-keyint_min")
        assert cmd[kmin_idx + 1] == "60"
        assert "bgr24" in cmd
        assert "1920x1080" in cmd
        assert "-movflags" in cmd
        assert "+faststart" in cmd

    def test_libx265_fallback(self):
        cmd = build_encode_pipe_hevc_cmd(
            30.0, 1920, 1080, 20, "/out.mp4", keyint=60, codec="libx265",
        )
        assert "libx265" in cmd
        assert "-crf" in cmd
        assert "-g" in cmd

    def test_crf_value(self):
        cmd = build_encode_pipe_hevc_cmd(
            30.0, 640, 480, 23, "/out.mp4", keyint=60,
        )
        # For NVENC: -cq; for libx265: -crf
        if "hevc_nvenc" in cmd:
            cq_idx = cmd.index("-cq")
            assert cmd[cq_idx + 1] == "23"
        else:
            crf_idx = cmd.index("-crf")
            assert cmd[crf_idx + 1] == "23"

    def test_output_path(self):
        cmd = build_encode_pipe_hevc_cmd(30.0, 640, 480, 20, "/path/out.mp4")
        assert cmd[-1] == "/path/out.mp4"

    def test_pipe_input(self):
        cmd = build_encode_pipe_hevc_cmd(30.0, 640, 480, 20, "/out.mp4")
        assert "pipe:0" in cmd


class TestBuildMuxAudioMp4Cmd:
    """build_mux_audio_mp4_cmd: AAC codec, faststart."""

    def test_basic_mux(self):
        cmd = build_mux_audio_mp4_cmd("video.mp4", "original.mp4", "final.mp4")
        assert "-c:v" in cmd
        assert "copy" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-b:a" in cmd
        assert "192k" in cmd
        assert "+faststart" in cmd
        assert cmd[-1] == "final.mp4"

    def test_extra_flags(self):
        cmd = build_mux_audio_mp4_cmd(
            "video.mp4", "original.mp4", "final.mp4",
            extra_flags=["-metadata:s:v", "stereo_mode=left_right"],
        )
        assert "stereo_mode=left_right" in cmd

    def test_map_streams(self):
        cmd = build_mux_audio_mp4_cmd("v.mp4", "a.mp4", "out.mp4")
        assert "0:v" in cmd
        assert "1:a" in cmd


# ---------------------------------------------------------------------------
# Tensor conversions (require torch)
# ---------------------------------------------------------------------------
try:
    import torch as _real_torch
    _has_torch = True
except ImportError:
    _has_torch = False


@pytest.mark.skipif(not _has_torch, reason="torch not installed")
class TestTensorConversions:
    """Premultiplied BGR and alpha downscale conversions."""

    def test_premultiplied_bgr_full_alpha(self):
        """Full alpha (1.0) → foreground colors preserved."""
        import torch
        from src.pipeline.bgremover import _tensor_to_frames_bgr_premultiplied

        fgr = torch.zeros(1, 3, 2, 2)
        fgr[0, 0] = 1.0   # R = 1
        fgr[0, 1] = 0.5   # G = 0.5
        fgr[0, 2] = 0.0   # B = 0

        pha = torch.ones(1, 1, 2, 2)  # full alpha

        frames = _tensor_to_frames_bgr_premultiplied(fgr, pha)
        assert len(frames) == 1
        assert frames[0].shape == (2, 2, 3)
        assert frames[0].dtype == np.uint8
        # BGR order: B=0, G≈128, R=255
        assert frames[0][0, 0, 2] == 255  # R
        assert frames[0][0, 0, 0] == 0    # B

    def test_premultiplied_bgr_zero_alpha(self):
        """Zero alpha → black output (premultiplied)."""
        import torch
        from src.pipeline.bgremover import _tensor_to_frames_bgr_premultiplied

        fgr = torch.ones(1, 3, 4, 4)
        pha = torch.zeros(1, 1, 4, 4)

        frames = _tensor_to_frames_bgr_premultiplied(fgr, pha)
        assert frames[0].max() == 0  # all black

    def test_premultiplied_bgr_half_alpha(self):
        """50% alpha → colors halved."""
        import torch
        from src.pipeline.bgremover import _tensor_to_frames_bgr_premultiplied

        fgr = torch.ones(1, 3, 4, 4)  # all 1.0
        pha = torch.full((1, 1, 4, 4), 0.5)

        frames = _tensor_to_frames_bgr_premultiplied(fgr, pha)
        assert frames[0][0, 0, 0] == pytest.approx(128, abs=2)

    def test_alpha_frames_downscale(self):
        """Alpha downscaled to target resolution, 3-channel grayscale."""
        import torch
        from src.pipeline.bgremover import _tensor_to_alpha_frames_bgr

        pha = torch.ones(2, 1, 480, 640)  # 2 frames, full alpha

        frames = _tensor_to_alpha_frames_bgr(pha, 120, 160)
        assert len(frames) == 2
        assert frames[0].shape == (120, 160, 3)
        assert frames[0].dtype == np.uint8
        # All channels should be 255 (white = opaque)
        assert frames[0][0, 0, 0] == 255
        assert frames[0][0, 0, 1] == 255
        assert frames[0][0, 0, 2] == 255

    def test_alpha_frames_zero(self):
        """Zero alpha downscaled → all black."""
        import torch
        from src.pipeline.bgremover import _tensor_to_alpha_frames_bgr

        pha = torch.zeros(1, 1, 480, 640)
        frames = _tensor_to_alpha_frames_bgr(pha, 120, 160)
        assert frames[0].max() == 0

    def test_alpha_frames_batch_size(self):
        """Batch dimension is preserved."""
        import torch
        from src.pipeline.bgremover import _tensor_to_alpha_frames_bgr

        pha = torch.ones(5, 1, 100, 100)
        frames = _tensor_to_alpha_frames_bgr(pha, 50, 50)
        assert len(frames) == 5


# ---------------------------------------------------------------------------
# OOM retry for split mode
# ---------------------------------------------------------------------------
class TestSplitOOMRetry:
    """_process_batch_split_with_oom_retry: OOM halving, state restore."""

    def _make_mock_torch(self):
        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        mock_torch.cuda.empty_cache = MagicMock()
        return mock_torch

    def test_success_first_try(self):
        from src.pipeline.bgremover import _process_batch_split_with_oom_retry

        mock_torch = self._make_mock_torch()
        mock_proc = MagicMock()
        mock_proc.get_recurrent_states.return_value = [None] * 4

        mock_fgr = MagicMock()
        mock_pha = MagicMock()
        mock_proc.process_batch_split.return_value = (mock_fgr, mock_pha)

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
        fake_main = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
        fake_alpha = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(4)]

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover._frames_to_tensor"), \
             patch("src.pipeline.bgremover._tensor_to_frames_bgr_premultiplied", return_value=fake_main), \
             patch("src.pipeline.bgremover._tensor_to_alpha_frames_bgr", return_value=fake_alpha):
            main, alpha, bs = _process_batch_split_with_oom_retry(
                mock_proc, frames, "cuda:0", 4, 30, 40,
            )

        assert bs == 4
        assert len(main) == 4
        assert len(alpha) == 4

    def test_oom_halves_batch(self):
        from src.pipeline.bgremover import _process_batch_split_with_oom_retry

        mock_torch = self._make_mock_torch()
        mock_proc = MagicMock()
        mock_proc.get_recurrent_states.return_value = [None] * 4

        call_count = 0

        def mock_split(tensor):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("CUDA out of memory")
            return (MagicMock(), MagicMock())

        mock_proc.process_batch_split.side_effect = mock_split

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
        fake_main = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        fake_alpha = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(2)]

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover._frames_to_tensor"), \
             patch("src.pipeline.bgremover._tensor_to_frames_bgr_premultiplied", return_value=fake_main), \
             patch("src.pipeline.bgremover._tensor_to_alpha_frames_bgr", return_value=fake_alpha):
            main, alpha, bs = _process_batch_split_with_oom_retry(
                mock_proc, frames, "cuda:0", 4, 30, 40,
            )

        assert bs == 2
        mock_torch.cuda.empty_cache.assert_called()

    def test_oom_all_sizes_raises(self):
        from src.pipeline.bgremover import _process_batch_split_with_oom_retry, BgRemoveError

        mock_torch = self._make_mock_torch()
        mock_proc = MagicMock()
        mock_proc.get_recurrent_states.return_value = [None] * 4
        mock_proc.process_batch_split.side_effect = RuntimeError("CUDA out of memory")

        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover._frames_to_tensor"), \
             pytest.raises(BgRemoveError, match="batch_size=1"):
            _process_batch_split_with_oom_retry(
                mock_proc, frames, "cuda:0", 1, 30, 40,
            )


# ---------------------------------------------------------------------------
# Encoder: mux_audio_mp4
# ---------------------------------------------------------------------------
class TestMuxAudioMp4:
    """mux_audio_mp4 encoder wrapper."""

    def test_mux_with_audio(self, tmp_path):
        from src.pipeline.encoder import mux_audio_mp4

        video = tmp_path / "video.mp4"
        video.touch()

        with patch("src.pipeline.encoder.has_audio", return_value=True), \
             patch("src.pipeline.encoder.run_ffmpeg") as mock_run:
            result = mux_audio_mp4(str(video), "original.mp4", str(tmp_path / "final.mp4"))

        assert result == str(tmp_path / "final.mp4")
        mock_run.assert_called_once()
        # Verify AAC codec is in the command
        cmd = mock_run.call_args[0][0]
        assert "aac" in cmd

    def test_mux_no_audio(self, tmp_path):
        from src.pipeline.encoder import mux_audio_mp4

        video = tmp_path / "video.mp4"
        video.touch()
        output = tmp_path / "final.mp4"

        with patch("src.pipeline.encoder.has_audio", return_value=False):
            result = mux_audio_mp4(str(video), "original.mp4", str(output))

        assert result == str(output)


# ---------------------------------------------------------------------------
# Streaming XALPHA pipeline (mocked)
# ---------------------------------------------------------------------------
class TestStreamingXalpha:
    """process_video_streaming_xalpha: dual encode pipes, metadata flow."""

    def _make_mock_torch(self):
        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        mock_torch.cuda.empty_cache = MagicMock()
        return mock_torch

    def test_xalpha_progress_stages(self, tmp_path):
        from src.pipeline.bgremover import process_video_streaming_xalpha

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
        mock_proc.process_batch_split.return_value = (MagicMock(), MagicMock())

        frame_bytes = np.zeros((64, 64, 3), dtype=np.uint8).tobytes()
        mock_decode_proc = MagicMock()
        mock_decode_proc.stdout = io.BytesIO(frame_bytes * 2)
        mock_decode_proc.returncode = 0
        mock_decode_proc.stdin = None
        mock_decode_proc.stderr = None

        mock_main_enc = MagicMock()
        mock_main_enc.returncode = 0
        mock_main_enc.stdout = None
        mock_main_enc.stderr = None
        mock_main_enc.stdin.closed = False

        mock_alpha_enc = MagicMock()
        mock_alpha_enc.returncode = 0
        mock_alpha_enc.stdout = None
        mock_alpha_enc.stderr = None
        mock_alpha_enc.stdin.closed = False

        fake_main = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        fake_alpha = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(2)]

        output_path = str(tmp_path / "main.mp4")
        alpha_path = str(tmp_path / "alpha.mp4")

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover.get_video_metadata", return_value=mock_meta), \
             patch("src.pipeline.bgremover.check_disk_space", return_value=True), \
             patch("src.pipeline.bgremover.get_encoder_codec", return_value="libx265"), \
             patch("src.pipeline.bgremover.RVMProcessor", return_value=mock_proc), \
             patch("src.pipeline.bgremover.start_decode_process", return_value=mock_decode_proc), \
             patch("src.pipeline.bgremover.start_encode_process", side_effect=[mock_main_enc, mock_alpha_enc]), \
             patch("src.pipeline.bgremover.close_process"), \
             patch("src.pipeline.bgremover._frames_to_tensor"), \
             patch("src.pipeline.bgremover._tensor_to_frames_bgr_premultiplied", return_value=fake_main), \
             patch("src.pipeline.bgremover._tensor_to_alpha_frames_bgr", return_value=fake_alpha), \
             patch("src.pipeline.bgremover.cleanup_job"), \
             patch("src.pipeline.bgremover.concatenate_segments"), \
             patch("src.pipeline.bgremover.mux_audio_mp4"), \
             patch("os.rename"):

            result = process_video_streaming_xalpha(
                input_path="/fake/input.mp4",
                output_path=output_path,
                alpha_output_path=alpha_path,
                segment_size=1000,
                progress_callback=track_progress,
            )

        assert result["status"] == "success"
        assert result["output_mode"] == "xalpha"
        assert "removing_background" in stages_seen
        assert "muxing_audio" in stages_seen

    def test_xalpha_dual_encode_processes(self, tmp_path):
        """Verify that two encode processes are started (main + alpha)."""
        from src.pipeline.bgremover import process_video_streaming_xalpha

        mock_torch = self._make_mock_torch()
        mock_meta = {
            "width": 64, "height": 64, "fps": 30.0,
            "duration": 0.1, "num_frames": 2,
            "codec": "h264", "file_size": 1000,
        }

        mock_proc = MagicMock()
        mock_proc.get_recurrent_states.return_value = [None] * 4
        mock_proc.process_batch_split.return_value = (MagicMock(), MagicMock())

        frame_bytes = np.zeros((64, 64, 3), dtype=np.uint8).tobytes()
        mock_decode_proc = MagicMock()
        mock_decode_proc.stdout = io.BytesIO(frame_bytes * 2)
        mock_decode_proc.returncode = 0
        mock_decode_proc.stdin = None
        mock_decode_proc.stderr = None

        mock_enc = MagicMock()
        mock_enc.returncode = 0
        mock_enc.stdout = None
        mock_enc.stderr = None
        mock_enc.stdin.closed = False

        fake_main = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        fake_alpha = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(2)]

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover.get_video_metadata", return_value=mock_meta), \
             patch("src.pipeline.bgremover.check_disk_space", return_value=True), \
             patch("src.pipeline.bgremover.get_encoder_codec", return_value="libx265"), \
             patch("src.pipeline.bgremover.RVMProcessor", return_value=mock_proc), \
             patch("src.pipeline.bgremover.start_decode_process", return_value=mock_decode_proc), \
             patch("src.pipeline.bgremover.start_encode_process", return_value=mock_enc) as mock_start, \
             patch("src.pipeline.bgremover.close_process"), \
             patch("src.pipeline.bgremover._frames_to_tensor"), \
             patch("src.pipeline.bgremover._tensor_to_frames_bgr_premultiplied", return_value=fake_main), \
             patch("src.pipeline.bgremover._tensor_to_alpha_frames_bgr", return_value=fake_alpha), \
             patch("src.pipeline.bgremover.cleanup_job"), \
             patch("src.pipeline.bgremover.concatenate_segments"), \
             patch("src.pipeline.bgremover.mux_audio_mp4"), \
             patch("os.rename"):

            result = process_video_streaming_xalpha(
                input_path="/fake/input.mp4",
                output_path=str(tmp_path / "main.mp4"),
                alpha_output_path=str(tmp_path / "alpha.mp4"),
                segment_size=1000,
            )

        assert result["status"] == "success"
        # Two encode processes per segment (main + alpha)
        assert mock_start.call_count == 2

    def test_xalpha_disk_space_error(self, tmp_path):
        from src.pipeline.bgremover import process_video_streaming_xalpha

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
            result = process_video_streaming_xalpha(
                input_path="/fake/input.mp4",
                output_path=str(tmp_path / "main.mp4"),
                alpha_output_path=str(tmp_path / "alpha.mp4"),
            )

        assert result["status"] == "failed"
        assert "disk space" in result["error"].lower()

    def test_xalpha_result_fields(self, tmp_path):
        """Result dict contains expected XALPHA-specific fields."""
        from src.pipeline.bgremover import process_video_streaming_xalpha

        mock_torch = self._make_mock_torch()
        mock_meta = {
            "width": 1920, "height": 1080, "fps": 30.0,
            "duration": 0.1, "num_frames": 2,
            "codec": "h264", "file_size": 1000,
        }

        mock_proc = MagicMock()
        mock_proc.get_recurrent_states.return_value = [None] * 4
        mock_proc.process_batch_split.return_value = (MagicMock(), MagicMock())

        frame_bytes = np.zeros((1080, 1920, 3), dtype=np.uint8).tobytes()
        mock_decode_proc = MagicMock()
        mock_decode_proc.stdout = io.BytesIO(frame_bytes * 2)
        mock_decode_proc.returncode = 0
        mock_decode_proc.stdin = None
        mock_decode_proc.stderr = None

        mock_enc = MagicMock()
        mock_enc.returncode = 0
        mock_enc.stdout = None
        mock_enc.stderr = None
        mock_enc.stdin.closed = False

        fake_main = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(2)]
        fake_alpha = [np.zeros((480, 854, 3), dtype=np.uint8) for _ in range(2)]

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover.get_video_metadata", return_value=mock_meta), \
             patch("src.pipeline.bgremover.check_disk_space", return_value=True), \
             patch("src.pipeline.bgremover.get_encoder_codec", return_value="libx265"), \
             patch("src.pipeline.bgremover.RVMProcessor", return_value=mock_proc), \
             patch("src.pipeline.bgremover.start_decode_process", return_value=mock_decode_proc), \
             patch("src.pipeline.bgremover.start_encode_process", return_value=mock_enc), \
             patch("src.pipeline.bgremover.close_process"), \
             patch("src.pipeline.bgremover._frames_to_tensor"), \
             patch("src.pipeline.bgremover._tensor_to_frames_bgr_premultiplied", return_value=fake_main), \
             patch("src.pipeline.bgremover._tensor_to_alpha_frames_bgr", return_value=fake_alpha), \
             patch("src.pipeline.bgremover.cleanup_job"), \
             patch("src.pipeline.bgremover.concatenate_segments"), \
             patch("src.pipeline.bgremover.mux_audio_mp4"), \
             patch("os.rename"):

            result = process_video_streaming_xalpha(
                input_path="/fake/input.mp4",
                output_path=str(tmp_path / "main.mp4"),
                alpha_output_path=str(tmp_path / "alpha.mp4"),
                segment_size=1000,
            )

        assert result["status"] == "success"
        assert "alpha_output_path" in result
        assert "alpha_resolution" in result
        assert "keyint" in result
        assert result["keyint"] == 60
        assert result["output_mode"] == "xalpha"
        assert result["output_resolution"] == "1920x1080"
