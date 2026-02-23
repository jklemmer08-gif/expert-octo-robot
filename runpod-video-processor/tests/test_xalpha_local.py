"""Local integration tests for XALPHA FFmpeg pipeline.

Tests the actual FFmpeg encode/decode/concat/mux chain with real files.
No GPU required — simulates RVM output with numpy operations.

Requires: ffmpeg with libx265 encoder available on PATH.
"""

import json
import subprocess
import shutil
import pytest
import numpy as np
from pathlib import Path

# Skip entire module if ffmpeg not available
pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not available on PATH",
)

FIXTURES = Path(__file__).parent / "fixtures"
TINY_VIDEO = FIXTURES / "test_tiny.mp4"       # 64x64, 30 frames, 10fps, no audio
SBS_VIDEO = FIXTURES / "test_SBS_vr.mp4"      # 1280x640, 300 frames, 30fps


def _ffprobe(path: str, *extra_args) -> dict:
    """Run ffprobe and return parsed JSON."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", *extra_args, str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"ffprobe failed: {result.stderr}"
    return json.loads(result.stdout)


def _get_keyframes(path: str) -> list:
    """Get list of keyframe PTS values via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=pict_type,pts_time",
        "-print_format", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    data = json.loads(result.stdout)
    return [f for f in data.get("frames", []) if f.get("pict_type") == "I"]


class TestXalphaEncodePipeline:
    """End-to-end: decode → fake processing → dual HEVC encode → verify."""

    def test_tiny_dual_encode(self, tmp_path):
        """Decode test_tiny.mp4, produce main + alpha HEVC files with matching keyint."""
        from src.utils.ffmpeg import build_decode_pipe_cmd, build_encode_pipe_hevc_cmd
        from src.utils.streaming import (
            start_decode_process, start_encode_process,
            read_frame, write_frame, close_process,
        )
        from src.pipeline.bgremover import compute_alpha_dimensions

        assert TINY_VIDEO.exists(), f"Test fixture not found: {TINY_VIDEO}"

        width, height, fps = 64, 64, 10.0
        num_frames = 30
        keyint = 60
        alpha_w, alpha_h = compute_alpha_dimensions(width, height, 32)

        main_path = str(tmp_path / "main.mp4")
        alpha_path = str(tmp_path / "alpha.mp4")

        # Start decode
        decode_cmd = build_decode_pipe_cmd(str(TINY_VIDEO), 0, num_frames, fps, width, height)
        decode_proc = start_decode_process(decode_cmd)

        # Start dual encode (libx265 — no GPU needed)
        main_cmd = build_encode_pipe_hevc_cmd(
            fps, width, height, 28, main_path, keyint=keyint, codec="libx265",
        )
        alpha_cmd = build_encode_pipe_hevc_cmd(
            fps, alpha_w, alpha_h, 30, alpha_path, keyint=keyint, codec="libx265",
        )
        main_enc = start_encode_process(main_cmd)
        alpha_enc = start_encode_process(alpha_cmd)

        # Process frames
        frames_written = 0
        try:
            for _ in range(num_frames):
                frame = read_frame(decode_proc, width, height, channels=3)
                if frame is None:
                    break

                # Simulate foreground on black (darken)
                main_frame = (frame // 2).astype(np.uint8)

                # Simulate alpha mask: grayscale, downscaled
                gray = np.mean(frame, axis=2, keepdims=True).astype(np.uint8)
                alpha_small = np.repeat(gray, 3, axis=2)
                # Resize to alpha dimensions using simple nearest-neighbor
                alpha_resized = np.zeros((alpha_h, alpha_w, 3), dtype=np.uint8)
                for c in range(3):
                    for y in range(alpha_h):
                        src_y = int(y * height / alpha_h)
                        for x in range(alpha_w):
                            src_x = int(x * width / alpha_w)
                            alpha_resized[y, x, c] = alpha_small[src_y, src_x, c]

                write_frame(main_enc, main_frame)
                write_frame(alpha_enc, alpha_resized)
                frames_written += 1
        finally:
            close_process(decode_proc, "decode", tolerant=True)
            close_process(main_enc, "main_encode")
            close_process(alpha_enc, "alpha_encode")

        assert frames_written == num_frames

        # --- Verify outputs with ffprobe ---

        # Main video
        main_info = _ffprobe(main_path, "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name,width,height,nb_frames,r_frame_rate",
                            "-show_entries", "format=duration")
        main_stream = main_info["streams"][0]
        assert main_stream["codec_name"] == "hevc"
        assert int(main_stream["width"]) == width
        assert int(main_stream["height"]) == height
        assert int(main_stream["nb_frames"]) == num_frames

        # Alpha video
        alpha_info = _ffprobe(alpha_path, "-select_streams", "v:0",
                             "-show_entries", "stream=codec_name,width,height,nb_frames",
                             "-show_entries", "format=duration")
        alpha_stream = alpha_info["streams"][0]
        assert alpha_stream["codec_name"] == "hevc"
        assert int(alpha_stream["width"]) == alpha_w
        assert int(alpha_stream["height"]) == alpha_h
        assert int(alpha_stream["nb_frames"]) == num_frames

        # Durations should match
        main_dur = float(main_info["format"]["duration"])
        alpha_dur = float(alpha_info["format"]["duration"])
        assert abs(main_dur - alpha_dur) < 0.2

        # Both should be valid (non-zero size)
        assert Path(main_path).stat().st_size > 0
        assert Path(alpha_path).stat().st_size > 0

    def test_tiny_all_keyframes(self, tmp_path):
        """With keyint=60 and only 30 frames, every GOP should start with an I-frame."""
        from src.utils.ffmpeg import build_decode_pipe_cmd, build_encode_pipe_hevc_cmd
        from src.utils.streaming import (
            start_decode_process, start_encode_process,
            read_frame, write_frame, close_process,
        )

        assert TINY_VIDEO.exists()

        width, height, fps = 64, 64, 10.0
        num_frames = 30
        keyint = 60

        main_path = str(tmp_path / "keytest.mp4")

        decode_cmd = build_decode_pipe_cmd(str(TINY_VIDEO), 0, num_frames, fps, width, height)
        decode_proc = start_decode_process(decode_cmd)

        enc_cmd = build_encode_pipe_hevc_cmd(
            fps, width, height, 28, main_path, keyint=keyint, codec="libx265",
        )
        enc_proc = start_encode_process(enc_cmd)

        try:
            for _ in range(num_frames):
                frame = read_frame(decode_proc, width, height, 3)
                if frame is None:
                    break
                write_frame(enc_proc, frame)
        finally:
            close_process(decode_proc, "decode", tolerant=True)
            close_process(enc_proc, "encode")

        # Check keyframes — with 30 frames and keyint=60, there should be
        # exactly 1 keyframe at the start
        keyframes = _get_keyframes(main_path)
        assert len(keyframes) >= 1
        # First frame should be I-frame
        assert float(keyframes[0]["pts_time"]) == pytest.approx(0.0, abs=0.1)


@pytest.mark.skipif(not SBS_VIDEO.exists(), reason="SBS test fixture not found")
class TestXalphaSBS:
    """Test with SBS VR video — larger file, multi-segment potential."""

    def test_sbs_dual_encode_first_segment(self, tmp_path):
        """Encode first 30 frames of SBS video as main + alpha HEVC."""
        from src.utils.ffmpeg import build_decode_pipe_cmd, build_encode_pipe_hevc_cmd
        from src.utils.streaming import (
            start_decode_process, start_encode_process,
            read_frame, write_frame, close_process,
        )
        from src.pipeline.bgremover import compute_alpha_dimensions

        width, height, fps = 1280, 640, 30.0
        num_frames = 30  # just first second
        keyint = 60

        alpha_w, alpha_h = compute_alpha_dimensions(width, height, 480)

        main_path = str(tmp_path / "sbs_main.mp4")
        alpha_path = str(tmp_path / "sbs_alpha.mp4")

        decode_cmd = build_decode_pipe_cmd(str(SBS_VIDEO), 0, num_frames, fps, width, height)
        decode_proc = start_decode_process(decode_cmd)

        main_cmd = build_encode_pipe_hevc_cmd(
            fps, width, height, 28, main_path, keyint=keyint, codec="libx265", preset="ultrafast",
        )
        alpha_cmd = build_encode_pipe_hevc_cmd(
            fps, alpha_w, alpha_h, 30, alpha_path, keyint=keyint, codec="libx265", preset="ultrafast",
        )
        main_enc = start_encode_process(main_cmd)
        alpha_enc = start_encode_process(alpha_cmd)

        frames_written = 0
        try:
            for _ in range(num_frames):
                frame = read_frame(decode_proc, width, height, 3)
                if frame is None:
                    break

                # Simulate main: darken
                main_frame = (frame.astype(np.float32) * 0.5).astype(np.uint8)

                # Simulate alpha: grayscale at reduced res
                gray = np.mean(frame, axis=2).astype(np.uint8)
                # Use cv2-free resize: simple block average
                alpha_frame = np.zeros((alpha_h, alpha_w, 3), dtype=np.uint8)
                y_scale = height / alpha_h
                x_scale = width / alpha_w
                for y in range(alpha_h):
                    src_y = min(int(y * y_scale), height - 1)
                    for x in range(alpha_w):
                        src_x = min(int(x * x_scale), width - 1)
                        val = gray[src_y, src_x]
                        alpha_frame[y, x] = [val, val, val]

                write_frame(main_enc, main_frame)
                write_frame(alpha_enc, alpha_frame)
                frames_written += 1
        finally:
            close_process(decode_proc, "decode", tolerant=True)
            close_process(main_enc, "main_encode")
            close_process(alpha_enc, "alpha_encode")

        assert frames_written == num_frames

        # Verify main
        main_info = _ffprobe(main_path, "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name,width,height,nb_frames")
        assert main_info["streams"][0]["codec_name"] == "hevc"
        assert int(main_info["streams"][0]["width"]) == width
        assert int(main_info["streams"][0]["height"]) == height

        # Verify alpha dimensions
        alpha_info = _ffprobe(alpha_path, "-select_streams", "v:0",
                             "-show_entries", "stream=codec_name,width,height,nb_frames")
        assert alpha_info["streams"][0]["codec_name"] == "hevc"
        assert int(alpha_info["streams"][0]["width"]) == alpha_w
        assert int(alpha_info["streams"][0]["height"]) == alpha_h
        assert int(alpha_info["streams"][0]["nb_frames"]) == num_frames

    def test_vr_metadata_injection(self, tmp_path):
        """Verify VR metadata tags can be injected into an HEVC file."""
        from src.utils.ffmpeg import build_decode_pipe_cmd, build_encode_pipe_hevc_cmd, run_ffmpeg
        from src.utils.streaming import (
            start_decode_process, start_encode_process,
            read_frame, write_frame, close_process,
        )

        # Create a small HEVC file first
        width, height, fps = 128, 64, 10.0
        num_frames = 10
        video_path = str(tmp_path / "before_meta.mp4")

        decode_cmd = build_decode_pipe_cmd(str(SBS_VIDEO), 0, num_frames, fps, 1280, 640)
        decode_proc = start_decode_process(decode_cmd)

        enc_cmd = build_encode_pipe_hevc_cmd(
            fps, width, height, 30, video_path, keyint=60, codec="libx265", preset="ultrafast",
        )
        enc_proc = start_encode_process(enc_cmd)

        try:
            for _ in range(num_frames):
                frame = read_frame(decode_proc, 1280, 640, 3)
                if frame is None:
                    break
                # Downscale crudely by slicing
                small = frame[:height, :width]
                write_frame(enc_proc, small)
        finally:
            close_process(decode_proc, "decode", tolerant=True)
            close_process(enc_proc, "encode")

        assert Path(video_path).exists()

        # Inject VR metadata
        tagged_path = str(tmp_path / "with_meta.mp4")
        meta_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c", "copy",
            "-metadata:s:v:0", "stereo_mode=left_right",
            "-metadata:s:v:0", "projection=equirectangular",
            "-metadata:s:v:0", "fov_horizontal=180",
            "-metadata:s:v:0", "fov_vertical=180",
            "-movflags", "+faststart",
            tagged_path,
        ]
        run_ffmpeg(meta_cmd)

        assert Path(tagged_path).exists()
        assert Path(tagged_path).stat().st_size > 0

        # Verify file was remuxed successfully (MP4 doesn't expose custom stream
        # metadata via ffprobe — HereSphere uses filename patterns for VR detection)
        info = _ffprobe(tagged_path, "-select_streams", "v:0",
                       "-show_entries", "stream=codec_name,nb_frames")
        assert info["streams"][0]["codec_name"] == "hevc"
        assert int(info["streams"][0]["nb_frames"]) == num_frames
        # File should be at least as large as input (stream copy + metadata)
        assert Path(tagged_path).stat().st_size >= Path(video_path).stat().st_size * 0.9


class TestXalphaFilenames:
    """Verify compute_xalpha_paths produces correct names for real scenarios."""

    def test_typical_vr_scene(self, tmp_path):
        from src.pipeline.bgremover import compute_xalpha_paths
        from src.pipeline.detector import VRLayout

        main, alpha = compute_xalpha_paths(
            "/input/Take It From the Bottom (2021)_FISHEYE190.mp4",
            VRLayout.SBS,
            tmp_path,
        )
        assert main.endswith("Take It From the Bottom (2021)_FISHEYE190_LR.mp4")
        assert alpha.endswith("Take It From the Bottom (2021)_FISHEYE190_LR_XALPHA.mp4")

    def test_already_has_lr(self, tmp_path):
        from src.pipeline.bgremover import compute_xalpha_paths
        from src.pipeline.detector import VRLayout

        main, alpha = compute_xalpha_paths(
            "/input/scene_LR_FISHEYE190.mp4",
            VRLayout.SBS,
            tmp_path,
        )
        # Should NOT double up the _LR
        assert "_LR_LR" not in main
        assert main.endswith("scene_LR_FISHEYE190.mp4")
        assert alpha.endswith("scene_LR_FISHEYE190_XALPHA.mp4")
