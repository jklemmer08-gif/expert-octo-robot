"""Tests for the processing pipeline components."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.models.schemas import ContentType, ProcessingPlan, VideoInfo
from src.processor import Encoder, FrameExtractor, MatteProcessor, ProcessingPipeline, UpscaleEngine, VRProcessor


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


class TestMatteProcessor:

    def _make_processor(self, settings):
        """Create a MatteProcessor bypassing heavy __init__ imports."""
        proc = object.__new__(MatteProcessor)
        proc.settings = settings
        proc.models_dir = Path(settings.paths.models_dir)
        proc.model = None
        proc.device = None
        proc._use_fp16 = False
        return proc

    @patch("src.processor.subprocess.run")
    def test_probe_video_pipe_info(self, mock_run, test_settings):
        """_probe_video_pipe_info parses ffprobe JSON correctly."""
        import json as _json

        probe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30000/1001",
                    "nb_frames": "54000",
                },
                {"codec_type": "audio"},
            ],
            "format": {"duration": "1800.0"},
        }
        mock_run.return_value = MagicMock(
            returncode=0, stdout=_json.dumps(probe_output),
        )

        proc = self._make_processor(test_settings)
        info = proc._probe_video_pipe_info(Path("/fake/video.mp4"))

        assert info["width"] == 1920
        assert info["height"] == 1080
        assert abs(info["fps"] - 29.97) < 0.01
        assert info["fps_str"] == "30000/1001"
        assert info["total_frames"] == 54000
        assert info["has_audio"] is True
        assert info["duration"] == 1800.0

    @patch("src.processor.subprocess.run")
    def test_probe_no_video_stream_raises(self, mock_run, test_settings):
        """_probe_video_pipe_info raises if no video stream found."""
        import json as _json

        probe_output = {"streams": [{"codec_type": "audio"}], "format": {}}
        mock_run.return_value = MagicMock(
            returncode=0, stdout=_json.dumps(probe_output),
        )

        proc = self._make_processor(test_settings)
        with pytest.raises(RuntimeError, match="No video stream"):
            proc._probe_video_pipe_info(Path("/fake/audio_only.mp4"))

    def test_streaming_dispatch_2d(self, test_settings):
        """process_video dispatches to streaming path when use_streaming=True."""
        test_settings.matte.use_streaming = True
        proc = self._make_processor(test_settings)

        with patch.object(proc, "_process_video_streaming_2d", return_value=True) as mock_stream:
            info = VideoInfo(width=1920, height=1080, fps=30, duration=60,
                             codec="h264", bitrate=15000000)
            result = proc.process_video(Path("/fake/in.mp4"), Path("/fake/out.mp4"), info)
            assert result is True
            mock_stream.assert_called_once()

    def test_legacy_dispatch_2d(self, test_settings):
        """process_video dispatches to legacy path when use_streaming=False."""
        test_settings.matte.use_streaming = False
        proc = self._make_processor(test_settings)

        with patch.object(proc, "_process_video_legacy", return_value=True) as mock_legacy:
            info = VideoInfo(width=1920, height=1080, fps=30, duration=60,
                             codec="h264", bitrate=15000000)
            result = proc.process_video(Path("/fake/in.mp4"), Path("/fake/out.mp4"), info)
            assert result is True
            mock_legacy.assert_called_once()

    def test_streaming_dispatch_vr(self, test_settings):
        """process_vr_sbs dispatches to streaming path when use_streaming=True."""
        test_settings.matte.use_streaming = True
        proc = self._make_processor(test_settings)

        with patch.object(proc, "_process_vr_sbs_streaming", return_value=True) as mock_stream:
            info = VideoInfo(width=3840, height=1920, fps=30, duration=60,
                             codec="h264", bitrate=50000000, is_vr=True,
                             content_type=ContentType.VR_SBS)
            result = proc.process_vr_sbs(Path("/fake/vr.mp4"), Path("/fake/vr_out.mp4"), info)
            assert result is True
            mock_stream.assert_called_once()

    def test_legacy_dispatch_vr(self, test_settings):
        """process_vr_sbs dispatches to legacy path when use_streaming=False."""
        test_settings.matte.use_streaming = False
        proc = self._make_processor(test_settings)

        with patch.object(proc, "_process_vr_sbs_legacy", return_value=True) as mock_legacy:
            info = VideoInfo(width=3840, height=1920, fps=30, duration=60,
                             codec="h264", bitrate=50000000, is_vr=True,
                             content_type=ContentType.VR_SBS)
            result = proc.process_vr_sbs(Path("/fake/vr.mp4"), Path("/fake/vr_out.mp4"), info)
            assert result is True
            mock_legacy.assert_called_once()

    def test_fp16_enabled_on_cuda(self, test_settings):
        """FP16 flag is set when config fp16=True and device is CUDA."""
        test_settings.matte.fp16 = True
        proc = self._make_processor(test_settings)

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.half.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = MagicMock(type="cuda")
        mock_torch.load.return_value = {}

        with patch("src.processor._import_torch", return_value=(mock_torch, MagicMock(), MagicMock())), \
             patch.object(proc, "_ensure_model", return_value=Path("/fake/model.pth")), \
             patch.dict("sys.modules", {"model": MagicMock(MattingNetwork=MagicMock(return_value=mock_model))}):
            proc.device = mock_torch.device("cuda")
            proc.device.type = "cuda"
            proc._load_model("mobilenetv3")

        assert proc._use_fp16 is True
        mock_model.half.assert_called_once()

    def test_fp16_disabled_on_cpu(self, test_settings):
        """FP16 flag is not set when device is CPU even if config fp16=True."""
        test_settings.matte.fp16 = True
        proc = self._make_processor(test_settings)

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device.return_value = MagicMock(type="cpu")
        mock_torch.load.return_value = {}

        with patch("src.processor._import_torch", return_value=(mock_torch, MagicMock(), MagicMock())), \
             patch.object(proc, "_ensure_model", return_value=Path("/fake/model.pth")), \
             patch.dict("sys.modules", {"model": MagicMock(MattingNetwork=MagicMock(return_value=mock_model))}):
            proc.device = mock_torch.device("cpu")
            proc.device.type = "cpu"
            proc._load_model("mobilenetv3")

        assert proc._use_fp16 is False
        mock_model.half.assert_not_called()
