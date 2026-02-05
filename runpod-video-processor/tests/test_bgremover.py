"""Tests for background removal pipeline — VR split/merge, batch processing, OOM retry.

All tests are mocked (no GPU needed).
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.pipeline.detector import VRLayout


class TestVRSplitMergeRGBA:
    """VR frame splitting and merging with 4-channel RGBA/BGRA frames."""

    def test_sbs_split_rgba(self):
        """Split BGRA SBS frame into two eyes."""
        from src.pipeline.bgremover import _split_vr_frame
        frame = np.zeros((100, 200, 4), dtype=np.uint8)
        frame[:, :100] = [255, 128, 64, 255]  # left eye BGRA
        left, right = _split_vr_frame(frame, VRLayout.SBS)
        assert left.shape == (100, 100, 4)
        assert right.shape == (100, 100, 4)
        assert left[0, 0, 0] == 255  # B channel
        assert right[0, 0, 0] == 0

    def test_ou_split_rgba(self):
        """Split BGRA OU frame into two eyes."""
        from src.pipeline.bgremover import _split_vr_frame
        frame = np.zeros((200, 100, 4), dtype=np.uint8)
        frame[:100, :] = [255, 128, 64, 255]
        top, bottom = _split_vr_frame(frame, VRLayout.OU)
        assert top.shape == (100, 100, 4)
        assert bottom.shape == (100, 100, 4)

    def test_sbs_merge_rgba(self):
        """Merge two BGRA eyes into SBS frame."""
        from src.pipeline.bgremover import _merge_vr_frame
        left = np.ones((100, 100, 4), dtype=np.uint8) * 200
        right = np.ones((100, 100, 4), dtype=np.uint8) * 100
        merged = _merge_vr_frame(left, right, VRLayout.SBS)
        assert merged.shape == (100, 200, 4)
        assert merged[:, :100].mean() == 200
        assert merged[:, 100:].mean() == 100

    def test_ou_merge_rgba(self):
        """Merge two BGRA eyes into OU frame."""
        from src.pipeline.bgremover import _merge_vr_frame
        top = np.ones((50, 100, 4), dtype=np.uint8) * 180
        bottom = np.ones((50, 100, 4), dtype=np.uint8) * 90
        merged = _merge_vr_frame(top, bottom, VRLayout.OU)
        assert merged.shape == (100, 100, 4)

    def test_split_merge_roundtrip_sbs_rgba(self):
        """SBS BGRA split then merge is lossless."""
        from src.pipeline.bgremover import _split_vr_frame, _merge_vr_frame
        original = np.random.randint(0, 255, (100, 200, 4), dtype=np.uint8)
        left, right = _split_vr_frame(original, VRLayout.SBS)
        merged = _merge_vr_frame(left, right, VRLayout.SBS)
        np.testing.assert_array_equal(original, merged)

    def test_split_merge_roundtrip_ou_rgba(self):
        """OU BGRA split then merge is lossless."""
        from src.pipeline.bgremover import _split_vr_frame, _merge_vr_frame
        original = np.random.randint(0, 255, (200, 100, 4), dtype=np.uint8)
        top, bottom = _split_vr_frame(original, VRLayout.OU)
        merged = _merge_vr_frame(top, bottom, VRLayout.OU)
        np.testing.assert_array_equal(original, merged)

    def test_split_invalid_layout(self):
        """Splitting non-stereo layout raises ValueError."""
        from src.pipeline.bgremover import _split_vr_frame
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            _split_vr_frame(frame, VRLayout.FLAT_2D)

    def test_split_3channel_bgr(self):
        """Split also works with 3-channel BGR frames (input frames)."""
        from src.pipeline.bgremover import _split_vr_frame
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        left, right = _split_vr_frame(frame, VRLayout.SBS)
        assert left.shape == (100, 100, 3)
        assert right.shape == (100, 100, 3)


try:
    import torch as _real_torch
    _has_torch = True
except ImportError:
    _has_torch = False


@pytest.mark.skipif(not _has_torch, reason="torch not installed")
class TestFrameConversion:
    """Test tensor conversion utilities (requires torch)."""

    def test_frames_to_tensor_converts_bgr_to_rgb(self):
        """_frames_to_tensor converts BGR numpy to RGB GPU tensor."""
        from src.pipeline.bgremover import _frames_to_tensor

        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        frames[0][:, :, 0] = 255  # Blue channel in BGR

        tensor = _frames_to_tensor(frames, "cpu")
        assert tensor.shape == (1, 3, 64, 64)
        # After BGR→RGB, blue (channel 0 in BGR) becomes channel 2 in RGB
        assert tensor[0, 2, 0, 0].item() == pytest.approx(1.0, abs=0.01)
        assert tensor[0, 0, 0, 0].item() == pytest.approx(0.0, abs=0.01)

    def test_tensor_to_frames_bgra(self):
        """_tensor_to_frames_bgra converts RGBA tensor to BGRA numpy."""
        import torch as real_torch
        from src.pipeline.bgremover import _tensor_to_frames_bgra

        # Create RGBA tensor: R=1, G=0.5, B=0, A=1
        tensor = real_torch.zeros(1, 4, 2, 2)
        tensor[0, 0] = 1.0   # R
        tensor[0, 1] = 0.5   # G
        tensor[0, 2] = 0.0   # B
        tensor[0, 3] = 1.0   # A

        frames = _tensor_to_frames_bgra(tensor)

        assert len(frames) == 1
        assert frames[0].shape == (2, 2, 4)
        assert frames[0].dtype == np.uint8
        # BGRA order: B=0, G=128, R=255, A=255
        assert frames[0][0, 0, 2] == 255  # R
        assert frames[0][0, 0, 0] == 0    # B
        assert frames[0][0, 0, 3] == 255  # A


class TestOOMRetry:
    """OOM retry with batch size reduction."""

    def _make_mock_torch(self):
        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        mock_torch.cuda.empty_cache = MagicMock()
        return mock_torch

    def test_success_first_try(self):
        """Batch processes successfully on first try."""
        from src.pipeline.bgremover import _process_batch_with_oom_retry

        mock_torch = self._make_mock_torch()
        mock_proc = MagicMock()
        mock_proc.get_recurrent_states.return_value = [None] * 4

        # Mock the tensor pipeline
        mock_rgba = MagicMock()
        mock_proc.process_batch.return_value = mock_rgba

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
        fake_bgra = [np.zeros((64, 64, 4), dtype=np.uint8) for _ in range(4)]

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover._frames_to_tensor") as mock_f2t, \
             patch("src.pipeline.bgremover._tensor_to_frames_bgra", return_value=fake_bgra):
            result, bs = _process_batch_with_oom_retry(mock_proc, frames, "cuda:0", 4)

        assert bs == 4
        assert len(result) == 4

    def test_oom_halves_batch_size(self):
        """OOM triggers batch size halving."""
        from src.pipeline.bgremover import _process_batch_with_oom_retry

        mock_torch = self._make_mock_torch()
        mock_proc = MagicMock()
        mock_proc.get_recurrent_states.return_value = [None] * 4

        call_count = 0

        def mock_process_batch(tensor):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("CUDA out of memory")
            return MagicMock()

        mock_proc.process_batch.side_effect = mock_process_batch

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
        fake_bgra = [np.zeros((64, 64, 4), dtype=np.uint8) for _ in range(2)]

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover._frames_to_tensor") as mock_f2t, \
             patch("src.pipeline.bgremover._tensor_to_frames_bgra", return_value=fake_bgra):
            result, bs = _process_batch_with_oom_retry(mock_proc, frames, "cuda:0", 4)

        assert bs == 2
        mock_torch.cuda.empty_cache.assert_called()

    def test_oom_all_sizes_raises(self):
        """OOM on all batch sizes raises BgRemoveError."""
        from src.pipeline.bgremover import _process_batch_with_oom_retry, BgRemoveError

        mock_torch = self._make_mock_torch()
        mock_proc = MagicMock()
        mock_proc.get_recurrent_states.return_value = [None] * 4
        mock_proc.process_batch.side_effect = RuntimeError("CUDA out of memory")

        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover._frames_to_tensor"), \
             pytest.raises(BgRemoveError, match="batch_size=1"):
            _process_batch_with_oom_retry(mock_proc, frames, "cuda:0", 1)


class TestProcessVideoCallback:
    """Progress callback stages during process_video."""

    def _make_mock_torch(self):
        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        mock_torch.cuda.empty_cache = MagicMock()
        return mock_torch

    def test_progress_stages(self, tmp_path):
        """process_video emits expected progress stages."""
        from src.pipeline.bgremover import process_video

        mock_torch = self._make_mock_torch()
        stages_seen = []

        def track_progress(update):
            stages_seen.append(update.get("stage"))

        # Mock all the heavy dependencies
        mock_meta = {
            "width": 64, "height": 64, "fps": 30.0,
            "duration": 1.0, "num_frames": 10,
            "codec": "h264", "file_size": 1000,
        }

        mock_proc = MagicMock()
        mock_proc.get_recurrent_states.return_value = [None] * 4
        mock_proc.process_batch.return_value = MagicMock()

        # Create fake frame files
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        fake_bgra = [np.zeros((64, 64, 4), dtype=np.uint8)]

        with patch("src.pipeline.bgremover.torch", mock_torch), \
             patch("src.pipeline.bgremover.get_video_metadata", return_value=mock_meta), \
             patch("src.pipeline.bgremover.check_disk_space", return_value=True), \
             patch("src.pipeline.bgremover.RVMProcessor", return_value=mock_proc), \
             patch("src.pipeline.bgremover.create_segment_dir") as mock_seg_dir, \
             patch("src.pipeline.bgremover.run_ffmpeg"), \
             patch("src.pipeline.bgremover.encode_segment_vp9", return_value="seg.webm"), \
             patch("src.pipeline.bgremover.concatenate_segments"), \
             patch("src.pipeline.bgremover.mux_audio_webm"), \
             patch("src.pipeline.bgremover.cleanup_job"), \
             patch("src.pipeline.bgremover.cv2") as mock_cv2, \
             patch("src.pipeline.bgremover._frames_to_tensor"), \
             patch("src.pipeline.bgremover._tensor_to_frames_bgra", return_value=fake_bgra), \
             patch("src.pipeline.bgremover.build_extract_frames_cmd", return_value=["ffmpeg"]), \
             patch("os.rename"), \
             patch("shutil.rmtree"):

            # Set up segment dir with subdirs
            seg_dir = tmp_path / "seg"
            seg_dir.mkdir()
            (seg_dir / "input").mkdir()
            (seg_dir / "output").mkdir()
            mock_seg_dir.return_value = seg_dir

            # Mock glob to return one fake frame
            fake_frame = seg_dir / "input" / "frame_000001.png"
            fake_frame.write_bytes(b"fake png")

            mock_cv2.imread.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            mock_cv2.imwrite.return_value = True

            result = process_video(
                input_path="/fake/input.mp4",
                output_path=str(tmp_path / "output.webm"),
                segment_size=1000,
                progress_callback=track_progress,
            )

        assert "extracting" in stages_seen
        assert "removing_background" in stages_seen
        assert "encoding" in stages_seen
        assert result["status"] == "success"

    def test_disk_space_error(self, tmp_path):
        """process_video fails gracefully when disk space is insufficient."""
        from src.pipeline.bgremover import process_video

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
            result = process_video(
                input_path="/fake/input.mp4",
                output_path=str(tmp_path / "output.webm"),
            )

        assert result["status"] == "failed"
        assert "disk space" in result["error"].lower()
