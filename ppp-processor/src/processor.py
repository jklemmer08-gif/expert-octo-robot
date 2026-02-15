"""Core processing pipeline for PPP Processor v2.0.

Modular classes extracted from PPPUpscaler (upscale.py):
- FrameExtractor — extract_frames()
- VRProcessor — split/merge SBS and TB
- UpscaleEngine — Real-ESRGAN with OOM retry
- Encoder — FFmpeg HEVC with hardware encoder fallback
- VRMetadataPreserver — metadata preservation from vr_metadata.py
- ProcessingPipeline — orchestrator
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from src.config import Settings
from src.models.schemas import ContentType, ProcessingPlan, VideoInfo

# Lazy torch import — only needed for matting
_torch = None
_transforms = None
_np = None


def _import_torch():
    """Lazy import of PyTorch + torchvision + numpy for matting."""
    global _torch, _transforms, _np
    if _torch is None:
        import torch as _torch
        import torchvision.transforms as _transforms
        import numpy as _np
    return _torch, _transforms, _np

logger = logging.getLogger("ppp.processor")

ProgressCallback = Optional[Callable[[str, float], None]]


# ---------------------------------------------------------------------------
# Frame Extractor (from upscale.py:140-163)
# ---------------------------------------------------------------------------
class FrameExtractor:
    """Extract video frames as PNG images using FFmpeg."""

    def extract(
        self,
        video_path: Path,
        output_dir: Path,
        fps: Optional[float] = None,
    ) -> int:
        """Extract frames from video. Returns frame count."""
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["ffmpeg", "-y", "-i", str(video_path)]
        if fps:
            cmd.extend(["-vf", f"fps={fps}"])
        cmd.extend(["-pix_fmt", "rgb24", str(output_dir / "frame_%08d.png")])

        logger.info("Extracting frames from %s", video_path.name)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {result.stderr}")

        frame_count = len(list(output_dir.glob("frame_*.png")))
        logger.info("Extracted %d frames", frame_count)
        return frame_count

    def extract_sample(
        self,
        video_path: Path,
        output_path: Path,
        duration: int = 15,
        start_percent: float = 0.4,
    ) -> Path:
        """Extract a short sample clip (from upscale.py:410-428)."""
        # Get duration via ffprobe
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        total_duration = float(data.get("format", {}).get("duration", 0))
        start_time = total_duration * start_percent

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(video_path),
            "-t", str(duration),
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path


# ---------------------------------------------------------------------------
# VR Processor (from upscale.py:165-217, extended with TB support)
# ---------------------------------------------------------------------------
class VRProcessor:
    """Split and merge stereoscopic VR frames (SBS and TB)."""

    def split_sbs(self, frames_dir: Path, left_dir: Path, right_dir: Path):
        """Split SBS frames into left and right eye (upscale.py:165-187)."""
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(frames_dir.glob("frame_*.png"))
        logger.info("Splitting %d SBS frames into L/R", len(frames))

        for i, frame in enumerate(frames):
            img = Image.open(frame)
            w, h = img.size
            half = w // 2
            img.crop((0, 0, half, h)).save(left_dir / frame.name)
            img.crop((half, 0, w, h)).save(right_dir / frame.name)
            img.close()

            if (i + 1) % 500 == 0:
                logger.info("Split %d/%d frames", i + 1, len(frames))

    def merge_sbs(self, left_dir: Path, right_dir: Path, output_dir: Path):
        """Merge left/right frames back to SBS (upscale.py:189-217)."""
        output_dir.mkdir(parents=True, exist_ok=True)

        left_frames = sorted(left_dir.glob("frame_*.png"))
        logger.info("Merging %d upscaled frames back to SBS", len(left_frames))

        for i, left_frame in enumerate(left_frames):
            right_frame = right_dir / left_frame.name
            left_img = Image.open(left_frame)
            right_img = Image.open(right_frame)
            lw, lh = left_img.size
            rw, rh = right_img.size

            merged = Image.new("RGB", (lw + rw, max(lh, rh)))
            merged.paste(left_img, (0, 0))
            merged.paste(right_img, (lw, 0))
            merged.save(output_dir / left_frame.name)

            left_img.close()
            right_img.close()
            merged.close()

            if (i + 1) % 500 == 0:
                logger.info("Merged %d/%d frames", i + 1, len(left_frames))

    def split_tb(self, frames_dir: Path, top_dir: Path, bottom_dir: Path):
        """Split Top-Bottom frames into top and bottom eye."""
        top_dir.mkdir(parents=True, exist_ok=True)
        bottom_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(frames_dir.glob("frame_*.png"))
        logger.info("Splitting %d TB frames into top/bottom", len(frames))

        for frame in frames:
            img = Image.open(frame)
            w, h = img.size
            half = h // 2
            img.crop((0, 0, w, half)).save(top_dir / frame.name)
            img.crop((0, half, w, h)).save(bottom_dir / frame.name)
            img.close()

    def merge_tb(self, top_dir: Path, bottom_dir: Path, output_dir: Path):
        """Merge top/bottom frames back to TB."""
        output_dir.mkdir(parents=True, exist_ok=True)

        top_frames = sorted(top_dir.glob("frame_*.png"))
        logger.info("Merging %d upscaled frames back to TB", len(top_frames))

        for top_frame in top_frames:
            bottom_frame = bottom_dir / top_frame.name
            top_img = Image.open(top_frame)
            bot_img = Image.open(bottom_frame)
            tw, th = top_img.size
            bw, bh = bot_img.size

            merged = Image.new("RGB", (max(tw, bw), th + bh))
            merged.paste(top_img, (0, 0))
            merged.paste(bot_img, (0, th))
            merged.save(output_dir / top_frame.name)

            top_img.close()
            bot_img.close()
            merged.close()


# ---------------------------------------------------------------------------
# Upscale Engine (from upscale.py:219-259, with OOM retry)
# ---------------------------------------------------------------------------
class UpscaleEngine:
    """Run Real-ESRGAN on extracted frames with OOM retry."""

    def __init__(self, settings: Settings):
        base_dir = Path(settings.paths.realesrgan_bin).parent.parent
        self.realesrgan_bin = Path(settings.paths.realesrgan_bin)
        if not self.realesrgan_bin.is_absolute():
            self.realesrgan_bin = base_dir / settings.paths.realesrgan_bin

    def upscale(
        self,
        input_dir: Path,
        output_dir: Path,
        model: str,
        scale: int,
        tile_size: int = 512,
        gpu_id: int = 0,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Run Real-ESRGAN with automatic tile_size reduction on OOM."""
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_count = len(list(input_dir.glob("frame_*.png")))

        current_tile = tile_size
        max_retries = 3

        for attempt in range(max_retries):
            cmd = [
                str(self.realesrgan_bin),
                "-i", str(input_dir),
                "-o", str(output_dir),
                "-n", model,
                "-s", str(scale),
                "-t", str(current_tile),
                "-g", str(gpu_id),
                "-f", "png",
            ]

            logger.info(
                "Upscaling %d frames: %s %dx (tile=%d, attempt %d)",
                frame_count, model, scale, current_tile, attempt + 1,
            )

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )

            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.debug("  %s", line)
                    # Parse progress if available
                    if progress_callback and "%" in line:
                        try:
                            pct = float(line.split("%")[0].split()[-1])
                            progress_callback("upscaling", pct)
                        except (ValueError, IndexError):
                            pass

            process.wait()

            if process.returncode == 0:
                output_count = len(list(output_dir.glob("frame_*.png")))
                logger.info("Upscaled %d/%d frames", output_count, frame_count)
                return output_count == frame_count

            # Check for OOM and retry with smaller tile
            if current_tile > 64:
                current_tile //= 2
                logger.warning(
                    "Upscale failed (OOM?), retrying with tile_size=%d",
                    current_tile,
                )
                # Clear partial output
                for f in output_dir.glob("frame_*.png"):
                    f.unlink()
                continue

            break

        logger.error("Upscaling failed after %d attempts", max_retries)
        return False


# ---------------------------------------------------------------------------
# Encoder (from upscale.py:261-302, with hw encoder selection + fallback)
# ---------------------------------------------------------------------------
class Encoder:
    """Encode frames to video using FFmpeg with hardware fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def encode(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float,
        audio_source: Optional[Path] = None,
        bitrate: str = "100M",
        encoder: Optional[str] = None,
    ) -> bool:
        """Encode frames back to video with HEVC."""
        enc = encoder or self.settings.encode.fallback_encoder
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try preferred encoder first, fall back if it fails
        encoders_to_try = [enc]
        if enc != self.settings.encode.fallback_encoder:
            encoders_to_try.append(self.settings.encode.fallback_encoder)

        for current_encoder in encoders_to_try:
            success = self._try_encode(
                frames_dir, output_path, fps, audio_source,
                bitrate, current_encoder,
            )
            if success:
                return True
            logger.warning("Encoder %s failed, trying fallback", current_encoder)

        return False

    def _try_encode(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float,
        audio_source: Optional[Path],
        bitrate: str,
        encoder: str,
    ) -> bool:
        ec = self.settings.encode
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%08d.png"),
        ]

        if audio_source and audio_source.exists():
            cmd.extend(["-i", str(audio_source)])

        # Encoder-specific params
        if encoder == "hevc_nvenc":
            cmd.extend([
                "-c:v", encoder,
                "-preset", ec.nvenc_preset,
                "-tune", ec.tune,
                "-rc", ec.rc_mode,
                "-qp", str(ec.qp),
                "-profile:v", ec.profile,
                "-rc-lookahead", str(ec.rc_lookahead),
                "-pix_fmt", "p010le",
                "-tag:v", "hvc1",
            ])
            if ec.spatial_aq:
                cmd.extend(["-spatial-aq", "1"])
            if ec.temporal_aq:
                cmd.extend(["-temporal-aq", "1"])
        else:
            bufsize = str(int(bitrate.rstrip("M")) * 2) + "M"
            cmd.extend([
                "-c:v", encoder,
                "-preset", ec.preset,
                "-crf", str(ec.crf),
                "-maxrate", bitrate,
                "-bufsize", bufsize,
                "-pix_fmt", "yuv420p",
                "-tag:v", "hvc1",
            ])

        if audio_source and audio_source.exists():
            cmd.extend(["-c:a", "copy", "-map", "0:v", "-map", "1:a"])

        cmd.extend(["-movflags", "+faststart", str(output_path)])

        logger.info("Encoding with %s to %s", encoder, output_path.name)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("Encoding failed: %s", result.stderr[-500:])
            return False

        logger.info("Encoded successfully: %s", output_path)
        return True


# ---------------------------------------------------------------------------
# VR Metadata Preserver (from vr_metadata.py)
# ---------------------------------------------------------------------------
class VRMetadataPreserver:
    """Preserve and inject VR metadata during processing."""

    def preserve_and_transfer(self, source_path: Path, dest_path: Path) -> Path:
        """Extract VR metadata from source and inject into dest."""
        # Use ffprobe to check for spherical metadata
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(source_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        try:
            probe_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return dest_path

        # Check for VR indicators in source
        video_stream = next(
            (s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            return dest_path

        is_vr = False
        stereo_mode = "sbs"
        fov = 180

        # Check stream side data
        for side_data in video_stream.get("side_data_list", []):
            if side_data.get("side_data_type") == "Spherical Mapping":
                is_vr = True
                break

        # Check filename patterns
        filename = source_path.name.lower()
        if any(p in filename for p in ["_180", "_360", "_vr", "_sbs", "_tb"]):
            is_vr = True

        if "_tb" in filename or "_ou" in filename:
            stereo_mode = "tb"
        if "_360" in filename:
            fov = 360

        # Aspect ratio check
        w = int(video_stream.get("width", 0))
        h = int(video_stream.get("height", 0))
        if h > 0 and w >= 2 * h:
            is_vr = True

        if not is_vr:
            return dest_path

        # Inject metadata via ffmpeg (copy, no re-encode)
        return self._inject_metadata(dest_path, stereo_mode, fov)

    def _inject_metadata(
        self, video_path: Path, stereo_mode: str, fov: int,
    ) -> Path:
        """Inject VR metadata tags into video file."""
        temp_path = video_path.with_suffix(".tmp.mp4")

        meta_tags = []
        if fov == 360:
            meta_tags.extend(["-metadata:s:v:0", "spherical=true"])
            meta_tags.extend(["-metadata:s:v:0", "stitched=true"])

        stereo_tag = {"sbs": "left_right", "tb": "top_bottom", "mono": "mono"}.get(
            stereo_mode, "left_right"
        )
        meta_tags.extend(["-metadata:s:v:0", f"stereo_mode={stereo_tag}"])
        meta_tags.extend(["-metadata:s:v:0", "projection=equirectangular"])
        meta_tags.extend(["-metadata:s:v:0", f"fov_horizontal={fov}"])
        meta_tags.extend(["-metadata:s:v:0", f"fov_vertical={fov}"])

        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-c", "copy", *meta_tags,
            "-movflags", "+faststart",
            str(temp_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            temp_path.replace(video_path)
        else:
            logger.warning("VR metadata injection failed, keeping original")
            if temp_path.exists():
                temp_path.unlink()

        return video_path

    def ensure_vr_filename(self, video_path: Path, is_vr: bool,
                            stereo_mode: str = "sbs", fov: int = 180) -> Path:
        """Ensure filename contains VR identifiers for player detection."""
        if not is_vr:
            return video_path

        filename = video_path.stem.lower()
        has_fov = any(p in filename for p in ["_180", "_360", "180x", "360x"])
        has_stereo = any(p in filename for p in ["_sbs", "_tb", "_lr", "_ou", "_mono"])

        if has_fov and has_stereo:
            return video_path

        new_stem = video_path.stem
        if not has_fov:
            new_stem += f"_{fov}"
        if not has_stereo:
            new_stem += f"_{stereo_mode}"

        new_path = video_path.with_stem(new_stem)
        if new_path != video_path:
            video_path.rename(new_path)
        return new_path


# ---------------------------------------------------------------------------
# Matte Processor — background removal via RobustVideoMatting
# ---------------------------------------------------------------------------
class MatteProcessor:
    """Background removal using RobustVideoMatting (refactored from matte.py).

    Produces green-screen output suitable for Heresphere chroma key passthrough.
    Requires PyTorch + torchvision. Works best with CUDA (RTX 3060 Ti / 4090).
    """

    RVM_WEIGHTS = {
        "mobilenetv3": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth",
        "resnet50": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth",
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.models_dir = Path(settings.paths.models_dir)
        self.model = None
        self.device = None
        self._trt_engine = None  # TensorRT engine (Phase 2)
        self._ort_engine = None  # ONNX Runtime engine
        self._openvino_engine = None  # OpenVINO engine (Intel)
        self._nvenc_available = None  # Cached NVENC probe result
        self._vaapi_available = None  # Cached VAAPI probe result
        self._platform = self._detect_platform()  # "nvidia", "intel", or "cpu"
        # CUDA Graphs state (Windows PyTorch fallback)
        self._cuda_graph = None
        self._cuda_graph_input = None
        self._cuda_graph_output_fgr = None
        self._cuda_graph_output_pha = None
        self._cuda_graph_rec_in = None
        self._cuda_graph_rec_out = None
        self._cuda_graph_frame_count = 0
        # ORT benchmark state — auto-fallback to PyTorch if ORT is too slow
        self._ort_frame_count = 0
        self._ort_bench_start = None

    def _probe_nvenc(self) -> bool:
        """Test NVENC availability with a minimal encode. Caches result."""
        if self._nvenc_available is not None:
            return self._nvenc_available

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-y",
                    "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                    "-c:v", "hevc_nvenc", "-f", "null", "-",
                ],
                capture_output=True, text=True, timeout=10,
            )
            self._nvenc_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._nvenc_available = False

        if self._nvenc_available:
            logger.info("NVENC hardware encoder available")
        else:
            logger.info("NVENC unavailable, will use %s", self.settings.encode.fallback_encoder)
        return self._nvenc_available

    @staticmethod
    def _detect_platform() -> str:
        """Detect GPU platform: 'nvidia', 'intel', or 'cpu'."""
        import platform as _platform

        if _platform.system() == "Windows":
            return "nvidia"

        # Linux: check for OpenVINO GPU support (Intel Arc)
        if _platform.system() == "Linux":
            try:
                import openvino as ov
                devices = ov.Core().available_devices
                if "GPU" in devices:
                    logger.info("Platform detected: intel (OpenVINO GPU available)")
                    return "intel"
            except ImportError:
                pass

            # Check for NVIDIA on Linux
            try:
                import torch
                if torch.cuda.is_available():
                    return "nvidia"
            except ImportError:
                pass

        return "cpu"

    def _probe_vaapi(self) -> bool:
        """Test VAAPI encoder availability with a minimal encode. Caches result."""
        if self._vaapi_available is not None:
            return self._vaapi_available

        vaapi_device = self.settings.matte.vaapi_device
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-y",
                    "-vaapi_device", vaapi_device,
                    "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                    "-vf", "format=nv12,hwupload",
                    "-c:v", "hevc_vaapi", "-f", "null", "-",
                ],
                capture_output=True, text=True, timeout=10,
            )
            self._vaapi_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._vaapi_available = False

        if self._vaapi_available:
            logger.info("VAAPI hardware encoder available (%s)", vaapi_device)
        else:
            logger.info("VAAPI unavailable, will use %s", self.settings.encode.fallback_encoder)
        return self._vaapi_available

    def _build_encode_cmd_vaapi(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps_str: str,
        audio_source: Optional[Path] = None,
        has_audio: bool = False,
    ) -> list:
        """Build FFmpeg VAAPI encode command for Intel GPUs.

        Uses rawvideo RGB24 pipe input → VAAPI hwupload → hevc_vaapi.
        Falls back to libx265 if VAAPI is unavailable.
        """
        ec = self.settings.encode
        mc = self.settings.matte
        vaapi_device = mc.vaapi_device

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]

        if self._probe_vaapi():
            cmd.extend(["-vaapi_device", vaapi_device])
            cmd.extend([
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", fps_str,
                "-i", "pipe:0",
            ])
            if has_audio and audio_source:
                cmd.extend(["-i", str(audio_source)])
            cmd.extend([
                "-vf", "format=nv12,hwupload",
                "-c:v", "hevc_vaapi",
                "-rc_mode", "CQP",
                "-global_quality", str(ec.vaapi_qp),
                "-profile:v", "main",
                "-tag:v", "hvc1",
            ])
            if has_audio and audio_source:
                cmd.extend(["-map", "0:v", "-map", "1:a", "-c:a", "copy"])
        else:
            # Fallback to software encoder
            cmd.extend([
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", fps_str,
                "-i", "pipe:0",
            ])
            if has_audio and audio_source:
                cmd.extend(["-i", str(audio_source)])
            cmd.extend([
                "-c:v", ec.fallback_encoder,
                "-preset", ec.preset,
                "-crf", str(ec.crf),
                "-pix_fmt", "yuv420p",
                "-tag:v", "hvc1",
            ])
            if has_audio and audio_source:
                cmd.extend(["-map", "0:v", "-map", "1:a", "-c:a", "copy"])

        cmd.extend(["-movflags", "+faststart", str(output_path)])
        return cmd

    def _check_pynvvideocodec(self) -> bool:
        """Check if PyNvVideoCodec is available for GPU-resident decode."""
        if hasattr(self, "_pynvc_available"):
            return self._pynvc_available
        try:
            import os as _os
            torch, _, _ = _import_torch()
            torch_lib = _os.path.join(_os.path.dirname(torch.__file__), "lib")
            _os.add_dll_directory(torch_lib)
            import PyNvVideoCodec  # noqa: F401
            self._pynvc_available = True
            logger.info("PyNvVideoCodec available — GPU-resident pipeline enabled")
        except (ImportError, OSError) as e:
            self._pynvc_available = False
            logger.info("PyNvVideoCodec unavailable — using FFmpeg pipes: %s", e)
        return self._pynvc_available

    @staticmethod
    def _rgb_chw_to_nv12_gpu(rgb):
        """Convert CHW RGB uint8 CUDA tensor to NV12 on GPU (BT.601).

        Returns a flat uint8 CUDA tensor: Y plane (H*W) + interleaved UV (H/2*W).
        Total size = H * W * 1.5 bytes.
        """
        torch, _, _ = _import_torch()
        r = rgb[0].float()
        g = rgb[1].float()
        b = rgb[2].float()
        # Y plane (full resolution)
        y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 255).to(torch.uint8)
        # UV plane (subsampled 2x2)
        r_sub = r[0::2, 0::2]
        g_sub = g[0::2, 0::2]
        b_sub = b[0::2, 0::2]
        u = (-0.169 * r_sub - 0.331 * g_sub + 0.500 * b_sub + 128).clamp(0, 255).to(torch.uint8)
        v = (0.500 * r_sub - 0.419 * g_sub - 0.081 * b_sub + 128).clamp(0, 255).to(torch.uint8)
        # Interleave U and V for NV12
        h2, w2 = u.shape
        uv = torch.empty(h2, w2 * 2, dtype=torch.uint8, device=rgb.device)
        uv[:, 0::2] = u
        uv[:, 1::2] = v
        return torch.cat([y.reshape(-1), uv.reshape(-1)])

    def _build_encode_cmd(
        self,
        input_args: list,
        output_path: Path,
        bitrate: str,
        extra_input_args: Optional[list] = None,
        extra_output_args: Optional[list] = None,
    ) -> list:
        """Build FFmpeg encode command with NVENC or fallback encoder.

        Returns the full ffmpeg command list. Uses NVENC with constqp rate control
        when available, falls back to libx265 CRF otherwise.
        """
        ec = self.settings.encode

        if self._probe_nvenc():
            encoder = ec.encoder  # hevc_nvenc
            encode_params = [
                "-c:v", encoder,
                "-preset", ec.nvenc_preset,
                "-tune", ec.tune,
                "-rc", ec.rc_mode,
                "-qp", str(ec.qp),
                "-profile:v", ec.profile,
                "-rc-lookahead", str(ec.rc_lookahead),
                "-pix_fmt", "p010le",
                "-tag:v", "hvc1",
            ]
            if ec.spatial_aq:
                encode_params.extend(["-spatial-aq", "1"])
            if ec.temporal_aq:
                encode_params.extend(["-temporal-aq", "1"])
        else:
            encoder = ec.fallback_encoder  # libx265
            bufsize_str = str(int(bitrate.rstrip("M")) * 2) + "M"
            encode_params = [
                "-c:v", encoder,
                "-preset", ec.preset,
                "-crf", str(ec.crf),
                "-maxrate", bitrate, "-bufsize", bufsize_str,
                "-pix_fmt", "yuv420p",
                "-tag:v", "hvc1",
            ]

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        cmd.extend(input_args)
        if extra_input_args:
            cmd.extend(extra_input_args)
        cmd.extend(encode_params)
        if extra_output_args:
            cmd.extend(extra_output_args)
        cmd.extend(["-movflags", "+faststart", str(output_path)])
        return cmd

    def _ensure_model(self, model_type: str = "mobilenetv3") -> Path:
        """Download RVM weights if not present."""
        model_path = self.models_dir / f"rvm_{model_type}.pth"
        if model_path.exists():
            return model_path

        url = self.RVM_WEIGHTS.get(model_type)
        if not url:
            raise ValueError(f"Unknown RVM model type: {model_type}")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading RVM %s model from %s ...", model_type, url)
        import urllib.request
        urllib.request.urlretrieve(url, str(model_path))
        logger.info("Downloaded RVM %s model to %s", model_type, model_path)
        return model_path

    def _probe_video_pipe_info(self, video_path: Path) -> dict:
        """Probe video for dimensions, FPS, frame count, and audio presence."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            raise RuntimeError(f"No video stream found in {video_path}")

        has_audio = any(
            s.get("codec_type") == "audio" for s in data.get("streams", [])
        )

        # Parse FPS — prefer avg_frame_rate (actual content rate) over
        # r_frame_rate (which can report field rate / timebase, e.g. 59.94
        # for 29.97fps H.264 content, causing 2x speed output).
        duration = float(data.get("format", {}).get("duration", 0))
        nb_frames = video_stream.get("nb_frames")

        avg_fps_str = video_stream.get("avg_frame_rate", "0/0")
        r_fps_str = video_stream.get("r_frame_rate", "30/1")

        def _parse_fps(s):
            try:
                n, d = s.split("/")
                return float(n) / float(d) if float(d) != 0 else 0.0
            except (ValueError, ZeroDivisionError):
                return 0.0

        avg_fps = _parse_fps(avg_fps_str)
        r_fps = _parse_fps(r_fps_str)

        # Use avg_frame_rate when it's valid (non-zero, reasonable)
        if avg_fps > 1.0:
            fps = avg_fps
            fps_str = avg_fps_str
        else:
            fps = r_fps if r_fps > 0 else 30.0
            fps_str = r_fps_str

        total_frames = int(nb_frames) if nb_frames else int(duration * fps)

        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "fps_str": fps_str,
            "total_frames": total_frames,
            "has_audio": has_audio,
            "duration": duration,
        }

    def _load_model(self, model_type: str = "mobilenetv3"):
        """Load RVM model onto best available device, optionally in FP16."""
        torch, transforms, np = _import_torch()

        model_path = self._ensure_model(model_type)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info("Loading RVM on device: %s", self.device)

        # Import the MattingNetwork from RVM repo (must be cloned)
        import sys as _sys
        rvm_path = self.models_dir.parent / "RobustVideoMatting"
        if rvm_path.exists() and str(rvm_path) not in _sys.path:
            _sys.path.insert(0, str(rvm_path))

        from model import MattingNetwork  # type: ignore[import-untyped]

        self.model = MattingNetwork(model_type).eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)

        # FP16 on CUDA for ~2x tensor core throughput
        mc = self.settings.matte
        self._use_fp16 = mc.fp16 and self.device.type == "cuda"
        if self._use_fp16:
            self.model = self.model.half()
            logger.info("FP16 enabled — model converted to half precision")

        # channels_last (NHWC) memory format for ~15-20% tensor core speedup on Ampere+
        if mc.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
            logger.info("channels_last memory format enabled")

        # torch.compile() for fused kernels
        # reduce-overhead mode requires Triton which is unavailable on Windows;
        # the compile() call itself succeeds but the first forward pass fails.
        # On Windows with FP16, torch.compile(eager) causes dtype mismatch
        # (bias float32 vs input float16) during Dynamo tracing — skip it.
        import platform as _platform
        if self.device.type == "cuda":
            if _platform.system() != "Windows":
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("torch.compile() enabled (reduce-overhead mode)")
                except Exception:
                    try:
                        self.model = torch.compile(self.model, backend="eager")
                        logger.info("torch.compile() enabled (eager backend)")
                    except Exception as e:
                        logger.info("torch.compile() unavailable: %s", e)
            elif not self._use_fp16:
                try:
                    self.model = torch.compile(self.model, backend="eager")
                    logger.info("torch.compile() enabled (eager backend — Windows)")
                except Exception as e:
                    logger.info("torch.compile() unavailable on Windows: %s", e)
            else:
                logger.info("Skipping torch.compile() on Windows with FP16 (Dynamo dtype conflict)")

    def _resolve_onnx_path(self) -> Optional[Path]:
        """Find the ONNX model for OpenVINO inference.

        Checks explicit config path first, then common locations.
        """
        mc = self.settings.matte

        # Explicit path from config
        if mc.onnx_model_path:
            p = Path(mc.onnx_model_path)
            if p.exists():
                return p

        # Check TRT cache dir (where ORT exports ONNX)
        trt_dir = Path(self.settings.tensorrt.cache_dir)
        candidates = [
            trt_dir / f"rvm_{mc.model_type}.onnx",
            trt_dir / f"rvm_{mc.model_type}_dr040.onnx",
            trt_dir / f"rvm_{mc.model_type}_dr025.onnx",
        ]
        for c in candidates:
            if c.exists():
                return c

        # Check models dir
        models_onnx = self.models_dir / f"rvm_{mc.model_type}.onnx"
        if models_onnx.exists():
            return models_onnx

        return None

    def _prepare_inference_engine(self, height: int, width: int):
        """Prepare accelerated inference engine for the given resolution.

        Priority:
        - Intel: OpenVINO GPU (no PyTorch needed)
        - NVIDIA: ORT (TRT EP → CUDA EP) → raw TRT → PyTorch fallback
        """
        mc = self.settings.matte

        # 0. OpenVINO (Intel path — no PyTorch needed)
        if self._platform == "intel" and self._openvino_engine is None:
            try:
                from src.openvino_engine import RVMOpenVINOEngine
                engine = RVMOpenVINOEngine()

                # Resolve ONNX model path
                onnx_path = self._resolve_onnx_path()
                if onnx_path and onnx_path.exists():
                    success = engine.prepare(
                        height, width, onnx_path,
                        device=mc.openvino_device,
                    )
                    if success:
                        self._openvino_engine = engine
                        logger.info("OpenVINO engine ready for %dx%d", width, height)
                        return
                    else:
                        logger.warning("OpenVINO prepare failed, falling back")
                else:
                    logger.warning("ONNX model not found at %s", onnx_path)
            except ImportError:
                logger.info("OpenVINO not installed, skipping")
            except Exception as e:
                logger.warning("OpenVINO init error: %s", e)

        # NVIDIA path needs PyTorch
        torch, _, _ = _import_torch()

        if self.model is None:
            self._load_model(mc.model_type)

        # 1. Try ONNX Runtime first (manages CUDA context internally)
        if self._ort_engine is None:
            try:
                from src.ort_engine import RVMOrtEngine
                self._ort_engine = RVMOrtEngine()
            except ImportError:
                logger.info("onnxruntime not installed, skipping ORT")
                self._ort_engine = None

        if self._ort_engine is not None:
            try:
                cache_dir = Path(self.settings.tensorrt.cache_dir)
                success = self._ort_engine.prepare(
                    height, width, self.model,
                    mc.model_type, self.device,
                    downsample_ratio=mc.downsample_ratio,
                    cache_dir=cache_dir,
                )
                # ONNX export may have changed model precision — always restore
                if self._use_fp16 and self.model is not None:
                    self.model = self.model.half()
                    if mc.channels_last and self.device.type == "cuda":
                        self.model = self.model.to(memory_format=torch.channels_last)
                    logger.info("Restored model to FP16 after ORT preparation")

                if success:
                    logger.info("ORT engine ready for %dx%d", width, height)
                    return
                else:
                    self._ort_engine = None
            except Exception as e:
                logger.warning("ORT init error: %s", e)
                self._ort_engine = None

        # 2. Fall back to raw TensorRT
        trt_cfg = self.settings.tensorrt
        if trt_cfg.enabled:
            try:
                from src.trt_engine import RVMTensorRTEngine
                if self._trt_engine is None:
                    self._trt_engine = RVMTensorRTEngine(self.settings)

                success = self._trt_engine.prepare(
                    height, width, self.model,
                    mc.model_type, self.device,
                )
                if success:
                    logger.info("TensorRT engine ready for %dx%d", width, height)
                else:
                    logger.info("TRT preparation failed, using PyTorch")
                    self._trt_engine = None
            except ImportError:
                logger.info("TensorRT not installed, using PyTorch inference")
                self._trt_engine = None
            except Exception as e:
                logger.warning("TRT init error, falling back to PyTorch: %s", e)
                self._trt_engine = None

        # ONNX export calls model.float() which mutates in-place — restore FP16
        if self._use_fp16 and self.model is not None:
            self.model = self.model.half()
            if mc.channels_last and self.device.type == "cuda":
                self.model = self.model.to(memory_format=torch.channels_last)
            logger.info("Restored model to FP16 after engine preparation")

    def _infer_frame(self, gpu_input, rec, downsample_ratio):
        """Dispatch inference to OpenVINO → ORT → TRT → PyTorch.

        Args:
            gpu_input: [1, 3, H, W] tensor on device (torch) or NumPy array
            rec: list of 4 recurrent state tensors (or Nones)
            downsample_ratio: float

        Returns:
            (fgr, pha, *rec_new) — same signature as RVM forward
        """
        # 0. OpenVINO (Intel path — input is NumPy, no PyTorch)
        if self._openvino_engine is not None:
            try:
                # gpu_input may be NumPy array on Intel path
                if hasattr(gpu_input, 'numpy'):
                    src_np = gpu_input.numpy() if not hasattr(gpu_input, 'cpu') else gpu_input.cpu().numpy()
                else:
                    src_np = gpu_input
                fgr, pha = self._openvino_engine.infer(src_np)
                return fgr, pha, *rec
            except Exception as e:
                logger.warning("OpenVINO inference failed, falling back: %s", e)
                self._openvino_engine = None

        torch, _, _ = _import_torch()

        # 1. ONNX Runtime (manages CUDA context — safe with PyNvVideoCodec)
        if self._ort_engine is not None:
            try:
                import time as _time
                if self._ort_bench_start is None:
                    self._ort_bench_start = _time.time()

                fgr, pha = self._ort_engine.infer(gpu_input)
                self._ort_frame_count += 1

                # After 20 warm-up frames, check if ORT is fast enough.
                # If below 5 fps at this point, PyTorch+CUDA Graphs will be faster.
                if self._ort_frame_count == 20:
                    elapsed = _time.time() - self._ort_bench_start
                    ort_fps = 20 / elapsed if elapsed > 0 else 0
                    logger.info("ORT benchmark: %.1f fps over 20 frames", ort_fps)
                    if ort_fps < 5.0:
                        logger.info(
                            "ORT too slow (%.1f fps < 5.0) — switching to PyTorch+CUDA Graphs",
                            ort_fps,
                        )
                        self._ort_engine.cleanup()
                        self._ort_engine = None
                        # Don't return — fall through to PyTorch path below

                if self._ort_engine is not None:
                    return fgr, pha, *rec
            except Exception as e:
                logger.warning("ORT inference failed, falling back: %s", e)
                self._ort_engine = None

        # 2. Raw TensorRT
        if self._trt_engine is not None:
            try:
                return self._trt_engine.infer(gpu_input, downsample_ratio)
            except Exception as e:
                logger.warning("TRT inference failed, falling back to PyTorch: %s", e)
                self._trt_engine = None

        # 3. PyTorch fallback (with optional CUDA Graphs on Windows)
        import platform as _platform
        mc = self.settings.matte
        use_cuda_graphs = (
            mc.cuda_graphs_pytorch
            and self.device.type == "cuda"
            and _platform.system() == "Windows"
        )

        if use_cuda_graphs:
            return self._infer_pytorch_cuda_graph(gpu_input, rec, downsample_ratio)

        with torch.no_grad():
            fgr, pha, *rec_new = self.model(gpu_input, *rec, downsample_ratio)
        return fgr, pha, *rec_new

    def _infer_pytorch_cuda_graph(self, gpu_input, rec, downsample_ratio):
        """PyTorch inference with CUDA Graph replay for reduced CPU overhead.

        After frame 1 (when recurrent states have resolved to real shapes),
        captures the inference as a CUDA Graph. Subsequent frames replay the
        graph with zero CPU-side kernel launch overhead (~10-20% speedup).
        """
        torch, _, _ = _import_torch()

        self._cuda_graph_frame_count += 1

        # Frame 1: warm-up run to resolve recurrent state shapes
        if self._cuda_graph_frame_count == 1:
            with torch.no_grad():
                fgr, pha, *rec_new = self.model(gpu_input, *rec, downsample_ratio)
            return fgr, pha, *rec_new

        # Frame 2: capture the graph (rec shapes are now fixed)
        if self._cuda_graph is None:
            # Run once to populate shapes
            with torch.no_grad():
                fgr, pha, *rec_new = self.model(gpu_input, *rec, downsample_ratio)

            # Allocate static buffers
            self._cuda_graph_input = gpu_input.clone()
            self._cuda_graph_rec_in = [r.clone() for r in rec_new]

            # Warm-up for graph capture (3 iters recommended by PyTorch)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.model(
                            self._cuda_graph_input,
                            *self._cuda_graph_rec_in,
                            downsample_ratio,
                        )
            torch.cuda.current_stream().wait_stream(s)

            # Capture
            self._cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._cuda_graph):
                with torch.no_grad():
                    g_fgr, g_pha, *g_rec = self.model(
                        self._cuda_graph_input,
                        *self._cuda_graph_rec_in,
                        downsample_ratio,
                    )

            self._cuda_graph_output_fgr = g_fgr
            self._cuda_graph_output_pha = g_pha
            self._cuda_graph_rec_out = g_rec

            logger.info("CUDA Graph captured for PyTorch inference")

            # Return the initial result (pre-capture run)
            return fgr, pha, *rec_new

        # Frame 3+: replay the captured graph
        self._cuda_graph_input.copy_(gpu_input)
        for i, r in enumerate(rec):
            if r is not None and self._cuda_graph_rec_in[i] is not None:
                self._cuda_graph_rec_in[i].copy_(r)

        self._cuda_graph.replay()

        # Copy outputs (graph output tensors are static buffers)
        fgr = self._cuda_graph_output_fgr.clone()
        pha = self._cuda_graph_output_pha.clone()
        rec_new = [r.clone() for r in self._cuda_graph_rec_out]

        # Feed rec_out back to rec_in for next frame
        for i in range(len(self._cuda_graph_rec_in)):
            self._cuda_graph_rec_in[i].copy_(self._cuda_graph_rec_out[i])

        return fgr, pha, *rec_new

    def _refine_alpha(self, pha, src):
        """Refine raw alpha matte with guided filter, thresholding, and morphological close.

        Args:
            pha: [1, 1, H, W] float tensor on GPU — raw alpha matte
            src: [1, 3, H, W] float tensor on GPU — source frame (guide for edge-aware filter)

        Returns:
            Refined pha tensor, same shape/dtype/device.
        """
        torch, _, _ = _import_torch()
        F = torch.nn.functional
        orig_dtype = pha.dtype

        # Upcast to FP32 — guided filter math overflows in FP16
        pha = pha.float()
        src_f = src.float()

        # --- Guided filter (edge-aware smoothing) ---
        # Grayscale guide from source: I = 0.299R + 0.587G + 0.114B
        I = src_f[:, 0:1] * 0.299 + src_f[:, 1:2] * 0.587 + src_f[:, 2:3] * 0.114

        r = 8  # filter radius
        eps = 1e-4
        k = 2 * r + 1  # kernel size = 17

        # Box filter via avg_pool2d (reflect-pad to avoid border shrinkage)
        def box_filter(x):
            return F.avg_pool2d(F.pad(x, [r, r, r, r], mode="reflect"), k, stride=1)

        mean_I = box_filter(I)
        mean_p = box_filter(pha)
        corr_Ip = box_filter(I * pha)
        var_I = box_filter(I * I) - mean_I * mean_I

        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        pha = box_filter(a) * I + box_filter(b)

        # --- Threshold clamping (suppress bleed, solidify foreground) ---
        pha = pha.clamp(0.0, 1.0)
        pha[pha < 0.05] = 0.0
        pha[pha > 0.95] = 1.0

        # --- Morphological close (fill small holes): dilate then erode ---
        # Dilation: max_pool2d with 3x3 kernel
        pha = F.max_pool2d(F.pad(pha, [1, 1, 1, 1], mode="reflect"), 3, stride=1)
        # Erosion: negate, max_pool, negate
        pha = -F.max_pool2d(F.pad(-pha, [1, 1, 1, 1], mode="reflect"), 3, stride=1)

        return pha.to(orig_dtype)

    def matte_frames(
        self,
        input_dir: Path,
        matted_dir: Path,
        green_color: tuple = (0, 177, 64),
        model_type: str = "mobilenetv3",
        downsample_ratio: float = 0.25,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Run RVM on all frames in input_dir → green-screen composites in matted_dir."""
        torch, transforms, np = _import_torch()

        matted_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            self._load_model(model_type)

        transform = transforms.Compose([transforms.ToTensor()])
        frames = sorted(input_dir.glob("frame_*.png"))
        total = len(frames)
        logger.info("Matting %d frames with RVM (%s)", total, model_type)

        rec = [None] * 4

        with torch.no_grad():
            for i, frame_path in enumerate(frames):
                img = Image.open(frame_path).convert("RGB")
                src = transform(img).unsqueeze(0).to(self.device)

                # channels_last for tensor core acceleration
                mc = self.settings.matte
                if mc.channels_last and self.device.type == "cuda":
                    src = src.contiguous(memory_format=torch.channels_last)

                fgr, pha, *rec = self.model(src, *rec, downsample_ratio)

                if mc.refine_alpha:
                    pha = self._refine_alpha(pha, src)

                # Build green-screen composite
                alpha_np = (pha[0, 0].cpu().numpy() * 255).astype(np.uint8)
                alpha_img = Image.fromarray(alpha_np, mode="L")

                frame_rgba = img.convert("RGBA")
                frame_rgba.putalpha(alpha_img)
                green_bg = Image.new("RGBA", img.size, (*green_color, 255))
                result = Image.alpha_composite(green_bg, frame_rgba)
                result.convert("RGB").save(matted_dir / frame_path.name)

                if progress_callback and (i + 1) % 50 == 0:
                    pct = (i + 1) / total * 100
                    progress_callback("matting", pct)

                if (i + 1) % 100 == 0:
                    logger.info("  Matted %d/%d frames", i + 1, total)

        logger.info("Matting complete: %d frames", total)
        return True

    def _process_video_gpu_resident(
        self,
        input_path: Path,
        output_path: Path,
        probe: dict,
        crop_region: Optional[tuple] = None,
        bitrate: str = "15M",
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """GPU-resident matte: PyNvVideoCodec decode → GPU infer → FFmpeg NVENC.

        Eliminates FFmpeg pipe decode bottleneck by decoding directly to GPU
        memory (zero-copy via DLPack). Encode uses FFmpeg with NV12 pipe input
        (12 MB/frame at 4K vs 25 MB with RGB24).

        Returns True on success, False on failure.
        """
        import os as _os
        import queue
        import threading

        torch, transforms, np = _import_torch()

        # Import PyNvVideoCodec (add DLL path for cudart64_12.dll)
        torch_lib = _os.path.join(_os.path.dirname(torch.__file__), "lib")
        if _os.path.isdir(torch_lib):
            _os.add_dll_directory(torch_lib)
        import PyNvVideoCodec as nvc

        mc = self.settings.matte
        src_w, src_h = probe["width"], probe["height"]
        fps_str = probe["fps_str"]
        total_frames = probe["total_frames"]

        if self.model is None:
            self._load_model(mc.model_type)

        # Determine processing dimensions
        if crop_region:
            cx, cy, cw, ch = crop_region
            proc_w, proc_h = cw, ch
        else:
            cx, cy = 0, 0
            proc_w, proc_h = src_w, src_h

        # Prepare inference engine — ORT manages its own CUDA context so it's
        # safe alongside PyNvVideoCodec. Raw TRT MUST NOT be used here because
        # TRT's CUDA context conflicts with PyNvVideoCodec (silently returns
        # fgr=0/pha=1 → black video).
        self._prepare_inference_engine(proc_h, proc_w)
        # If ORT didn't engage, disable raw TRT for GPU-resident path
        if self._ort_engine is None and self._trt_engine is not None:
            logger.info("Disabling raw TRT in GPU-resident path (CUDA context conflict)")
            self._trt_engine = None

        dtype = torch.float16 if getattr(self, "_use_fp16", False) else torch.float32

        # Pre-allocate GPU buffers
        green = mc.green_color
        green_tensor = torch.tensor(
            [[[green[0] / 255.0]], [[green[1] / 255.0]], [[green[2] / 255.0]]],
            dtype=dtype, device=self.device,
        )
        gpu_input = torch.empty((1, 3, proc_h, proc_w), dtype=dtype, device=self.device)
        out_gpu = torch.empty((1, 3, proc_h, proc_w), dtype=dtype, device=self.device)

        # NV12 pinned output buffer for fast GPU→CPU transfer
        nv12_size = proc_w * proc_h * 3 // 2
        nv12_pin = torch.empty(nv12_size, dtype=torch.uint8).pin_memory()

        # --- FFmpeg encode pipe (NV12 input → NVENC) ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipe_input_args = [
            "-f", "rawvideo", "-pix_fmt", "nv12",
            "-s", f"{proc_w}x{proc_h}", "-r", fps_str,
            "-i", "pipe:0",
        ]
        audio_args = None
        if probe["has_audio"]:
            audio_args = [
                "-i", str(input_path),
                "-map", "0:v", "-map", "1:a", "-c:a", "copy",
            ]
        encode_cmd = self._build_encode_cmd(
            pipe_input_args, output_path, bitrate,
            extra_input_args=audio_args,
        )

        pipe_bufsize = nv12_size * 8

        # --- Async encode thread ---
        SENTINEL = None
        encode_q: queue.Queue = queue.Queue(maxsize=4)
        encode_error = [None]

        encoder = subprocess.Popen(
            encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=pipe_bufsize,
        )

        def encode_thread():
            """Pull NV12 frame bytes from queue and write to FFmpeg pipe."""
            try:
                while True:
                    data = encode_q.get()
                    if data is SENTINEL:
                        break
                    encoder.stdin.write(data)
            except Exception as e:
                encode_error[0] = e

        rec = [None] * 4
        frame_idx = 0
        t_start = time.time()

        try:
            # Start GPU decoder (RGBP = planar RGB → CHW uint8 on GPU)
            # cuda_context=0 uses the primary CUDA context (shared with PyTorch/TRT)
            decoder = nvc.SimpleDecoder(
                str(input_path), gpu_id=0,
                cuda_context=0, cuda_stream=0,
                output_color_type=nvc.OutputColorType.RGBP,
            )

            # Start encode thread
            t_encode = threading.Thread(target=encode_thread, daemon=True)
            t_encode.start()

            for frame in decoder:
                # DLPack zero-copy GPU decode → torch CUDA tensor (CHW uint8)
                frame_tensor = torch.from_dlpack(frame)

                # Apply crop on GPU (tensor slice — effectively free)
                if crop_region:
                    frame_tensor = frame_tensor[:, cy:cy + ch, cx:cx + cw].contiguous()

                # Convert to model input format (normalized float16/32)
                gpu_input[0].copy_(frame_tensor.to(dtype))
                gpu_input.div_(255.0)

                # channels_last for tensor core acceleration
                if mc.channels_last:
                    gpu_input = gpu_input.contiguous(memory_format=torch.channels_last)

                # GPU inference (ORT or PyTorch)
                fgr, pha, *rec = self._infer_frame(gpu_input, rec, mc.downsample_ratio)

                if mc.refine_alpha:
                    pha = self._refine_alpha(pha, gpu_input)

                # GPU compositing: out = green + pha * (fgr - green)
                torch.addcmul(
                    green_tensor.expand_as(fgr), pha,
                    fgr - green_tensor, out=out_gpu,
                )
                out_gpu.clamp_(0.0, 1.0)

                # Convert composite to uint8 CHW then GPU NV12
                frame_out = out_gpu[0].mul(255).to(torch.uint8)
                nv12_gpu = self._rgb_chw_to_nv12_gpu(frame_out)

                # GPU→CPU via pinned memory
                nv12_pin.copy_(nv12_gpu)
                encode_q.put(nv12_pin.numpy().tobytes())

                frame_idx += 1
                if frame_idx % mc.progress_interval == 0:
                    elapsed = time.time() - t_start
                    fps_actual = frame_idx / max(elapsed, 0.1)
                    if progress_callback:
                        pct = min(frame_idx / max(total_frames, 1) * 100, 99)
                        progress_callback("matting", pct)
                    logger.info(
                        "  GPU-resident: %d/%d frames (%.1f fps)",
                        frame_idx, total_frames, fps_actual,
                    )

            # Signal encode thread to finish
            encode_q.put(SENTINEL)
            t_encode.join(timeout=30)

            encoder.stdin.close()
            encoder.wait()

            if encode_error[0]:
                logger.error("Encode thread error: %s", encode_error[0])

            if encoder.returncode != 0:
                err = encoder.stderr.read().decode() if encoder.stderr else ""
                logger.error("GPU-resident encode failed: %s", err[-500:])
                return False

            elapsed = time.time() - t_start
            fps_avg = frame_idx / max(elapsed, 0.1)
            logger.info(
                "GPU-resident matting complete: %d frames in %.1fs (%.1f fps avg)",
                frame_idx, elapsed, fps_avg,
            )
            return True

        except Exception as e:
            logger.error("GPU-resident pipeline error: %s", e)
            import traceback
            logger.error(traceback.format_exc())
            return False

        finally:
            if encoder and encoder.poll() is None:
                encoder.kill()
            if self.device and self.device.type == "cuda":
                torch.cuda.empty_cache()

    # Files that crashed PyNvVideoCodec (segfault-prone) — skip GPU-resident for these.
    # Persisted as a simple file so it survives worker restarts.
    _NVDEC_BLOCKLIST_FILE = Path(__file__).parent.parent / "cache" / "nvdec_blocklist.txt"

    def _is_nvdec_blocked(self, input_path: Path) -> bool:
        """Check if a file previously caused a PyNvVideoCodec segfault."""
        try:
            if self._NVDEC_BLOCKLIST_FILE.exists():
                blocked = self._NVDEC_BLOCKLIST_FILE.read_text().strip().splitlines()
                return input_path.name in blocked
        except OSError:
            pass
        return False

    def _mark_nvdec_blocked(self, input_path: Path):
        """Mark a file as unsafe for PyNvVideoCodec (called before GPU-resident attempt)."""
        try:
            self._NVDEC_BLOCKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self._NVDEC_BLOCKLIST_FILE, "a") as f:
                f.write(input_path.name + "\n")
            logger.info("Marked %s in NVDEC blocklist", input_path.name)
        except OSError as e:
            logger.warning("Could not write NVDEC blocklist: %s", e)

    def _unmark_nvdec_blocked(self, input_path: Path):
        """Remove a file from the blocklist after successful GPU-resident processing."""
        try:
            if self._NVDEC_BLOCKLIST_FILE.exists():
                lines = self._NVDEC_BLOCKLIST_FILE.read_text().strip().splitlines()
                lines = [l for l in lines if l != input_path.name]
                self._NVDEC_BLOCKLIST_FILE.write_text("\n".join(lines) + "\n" if lines else "")
        except OSError:
            pass

    def _process_video_with_best_backend(
        self,
        input_path: Path,
        output_path: Path,
        probe: dict,
        crop_region: Optional[tuple] = None,
        bitrate: str = "15M",
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Try GPU-resident pipeline in-process, fall back to streaming.

        GPU-resident uses PyNvVideoCodec for zero-copy GPU decode, which is
        10-20x faster than the streaming FFmpeg-pipe path. Falls back to
        streaming if PyNvVideoCodec is unavailable or encounters an error.

        Files that previously caused a PyNvVideoCodec segfault are blocklisted
        and sent directly to the streaming path.
        """
        torch, _, _ = _import_torch()

        if self._is_nvdec_blocked(input_path):
            logger.warning("File %s is NVDEC-blocklisted, using streaming path", input_path.name)
        elif self._check_pynvvideocodec():
            # Mark before attempt — if we segfault, the file is already blocklisted
            # for the next worker restart. Removed on success.
            self._mark_nvdec_blocked(input_path)
            try:
                logger.info("GPU-resident pipeline: %s", input_path.name)
                success = self._process_video_gpu_resident(
                    input_path, output_path, probe,
                    crop_region=crop_region, bitrate=bitrate,
                    progress_callback=progress_callback,
                )
                if success:
                    self._unmark_nvdec_blocked(input_path)
                    return True
                logger.warning("GPU-resident returned False, falling back to streaming")
            except Exception as e:
                logger.warning("GPU-resident failed (%s), falling back to streaming", e)
                # Clean up CUDA state before fallback
                torch.cuda.empty_cache()

            # Clean up partial output so streaming path starts fresh
            try:
                if output_path.exists():
                    output_path.unlink()
                    logger.info("Deleted partial output: %s", output_path)
            except OSError as e:
                logger.warning("Could not delete partial output: %s", e)

        return self._process_video_streaming(
            input_path, output_path, probe,
            crop_region=crop_region, bitrate=bitrate,
            progress_callback=progress_callback,
        )

    def _process_video_streaming(
        self,
        input_path: Path,
        output_path: Path,
        probe: dict,
        crop_region: Optional[tuple] = None,
        bitrate: str = "15M",
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Streaming matting engine: FFmpeg pipe decode → GPU inference → pipe encode.

        Optimized with:
        - NVDEC hardware decode (falls back to CPU if unavailable)
        - Pre-allocated pinned memory + GPU tensors (zero per-frame allocation)
        - Async 3-stage pipeline: decode / infer / encode overlap via threading
        - torch.compile() on the model (applied in _load_model)
        """
        import threading
        import queue

        torch, transforms, np = _import_torch()
        mc = self.settings.matte

        if self.model is None:
            self._load_model(mc.model_type)

        src_w, src_h = probe["width"], probe["height"]
        fps_str = probe["fps_str"]
        total_frames = probe["total_frames"]

        # Determine decode/process dimensions
        if crop_region:
            cx, cy, cw, ch = crop_region
            vf_decode = f"crop={cw}:{ch}:{cx}:{cy}"
            proc_w, proc_h = cw, ch
        else:
            vf_decode = None
            proc_w, proc_h = src_w, src_h

        # Try to prepare TensorRT engine for this resolution
        self._prepare_inference_engine(proc_h, proc_w)

        frame_bytes = proc_w * proc_h * 3  # RGB24
        pipe_bufsize = frame_bytes * 8  # 8 frames of buffer

        # --- Pre-allocate reusable tensors ---
        dtype = torch.float16 if getattr(self, "_use_fp16", False) else torch.float32
        # Pinned host buffer for fast CPU→GPU transfer
        pin_buffer = torch.empty((proc_h, proc_w, 3), dtype=torch.uint8).pin_memory()
        # Pre-allocated GPU input tensor
        gpu_input = torch.empty((1, 3, proc_h, proc_w), dtype=dtype, device=self.device)

        # Pre-allocate green background tensor on GPU (stays resident)
        green = mc.green_color
        green_tensor = torch.tensor(
            [[[green[0] / 255.0]], [[green[1] / 255.0]], [[green[2] / 255.0]]],
            dtype=dtype, device=self.device,
        )

        # Pre-allocated output buffer on GPU
        out_gpu = torch.empty((1, 3, proc_h, proc_w), dtype=dtype, device=self.device)
        # Pre-allocated output host buffer (pinned for fast GPU→CPU)
        out_pin = torch.empty((proc_h, proc_w, 3), dtype=torch.uint8).pin_memory()

        # CUDA stream for async GPU↔CPU transfers
        transfer_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

        # --- FFmpeg decode pipe (try NVDEC hardware accel) ---
        decode_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if self.device.type == "cuda":
            decode_cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
        decode_cmd.extend(["-i", str(input_path)])
        if vf_decode:
            # hwaccel requires scale_cuda for filtering; fall back to vf for crop
            if self.device.type == "cuda":
                decode_cmd.extend(["-vf", f"hwdownload,format=nv12,{vf_decode}"])
            else:
                decode_cmd.extend(["-vf", vf_decode])
        elif self.device.type == "cuda":
            decode_cmd.extend(["-vf", "hwdownload,format=nv12"])
        decode_cmd.extend(["-pix_fmt", "rgb24", "-f", "rawvideo", "-v", "error", "pipe:1"])

        # --- FFmpeg encode pipe ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipe_input_args = [
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{proc_w}x{proc_h}", "-r", fps_str,
            "-i", "pipe:0",
        ]
        audio_args = None
        if probe["has_audio"]:
            audio_args = ["-i", str(input_path), "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
        encode_cmd = self._build_encode_cmd(
            pipe_input_args, output_path, bitrate,
            extra_input_args=audio_args,
        )

        def _read_with_timeout(pipe, nbytes, timeout=5):
            """Read from pipe with a timeout to avoid hanging on network I/O."""
            result = [None]
            def _read():
                result[0] = pipe.read(nbytes)
            t = threading.Thread(target=_read, daemon=True)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                return None  # Timed out
            return result[0]

        # If NVDEC decode fails, retry without hardware accel
        decoder = None
        nvdec_failed = False
        try:
            decoder = subprocess.Popen(
                decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=pipe_bufsize,
            )
            # Test first frame (with timeout to avoid hanging on network files)
            test_raw = _read_with_timeout(decoder.stdout, frame_bytes, timeout=5)
            if not test_raw or len(test_raw) < frame_bytes:
                decoder.kill()
                decoder.wait()
                nvdec_failed = True
        except Exception:
            if decoder:
                decoder.kill()
                decoder.wait()
            nvdec_failed = True

        if nvdec_failed:
            logger.info("NVDEC unavailable, falling back to CPU decode")
            decode_cmd_cpu = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(input_path),
            ]
            if vf_decode:
                decode_cmd_cpu.extend(["-vf", vf_decode])
            decode_cmd_cpu.extend(["-pix_fmt", "rgb24", "-f", "rawvideo", "-v", "error", "pipe:1"])
            decoder = subprocess.Popen(
                decode_cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=pipe_bufsize,
            )
            test_raw = _read_with_timeout(decoder.stdout, frame_bytes, timeout=5)
            if not test_raw or len(test_raw) < frame_bytes:
                logger.error("CPU decode also failed")
                decoder.kill()
                return False

        # --- Async 3-stage pipeline ---
        # Stage 1 (thread): decode frames from ffmpeg pipe → decode_q
        # Stage 2 (main):   GPU inference on each frame
        # Stage 3 (thread): write matted frames to ffmpeg encode pipe

        SENTINEL = None
        decode_q = queue.Queue(maxsize=4)
        encode_q = queue.Queue(maxsize=4)
        decode_error = [None]
        encode_error = [None]

        encoder = subprocess.Popen(
            encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=pipe_bufsize,
        )

        def decode_thread():
            """Read raw frames from ffmpeg and push to decode_q."""
            try:
                # First frame was already read for NVDEC test
                decode_q.put(test_raw)
                while True:
                    raw = decoder.stdout.read(frame_bytes)
                    if not raw or len(raw) < frame_bytes:
                        break
                    decode_q.put(raw)
            except Exception as e:
                decode_error[0] = e
            finally:
                decode_q.put(SENTINEL)

        def encode_thread():
            """Pull matted frame bytes from encode_q and write to ffmpeg."""
            try:
                while True:
                    data = encode_q.get()
                    if data is SENTINEL:
                        break
                    encoder.stdin.write(data)
            except Exception as e:
                encode_error[0] = e

        rec = [None] * 4

        try:
            t_decode = threading.Thread(target=decode_thread, daemon=True)
            t_encode = threading.Thread(target=encode_thread, daemon=True)
            t_decode.start()
            t_encode.start()

            frame_idx = 0
            while True:
                raw = decode_q.get()
                if raw is SENTINEL:
                    break

                # --- Fast CPU→GPU path using pinned memory ---
                frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, 3)
                pin_buffer.copy_(torch.from_numpy(frame_np.copy()))

                if transfer_stream:
                    with torch.cuda.stream(transfer_stream):
                        gpu_input.copy_(
                            pin_buffer.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype, device=self.device, non_blocking=True)
                        )
                        gpu_input.div_(255.0)
                    transfer_stream.synchronize()
                else:
                    gpu_input.copy_(
                        pin_buffer.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype, device=self.device)
                    )
                    gpu_input.div_(255.0)

                # channels_last for tensor core acceleration
                if mc.channels_last:
                    gpu_input = gpu_input.contiguous(memory_format=torch.channels_last)

                # --- GPU inference (ORT or PyTorch) ---
                fgr, pha, *rec = self._infer_frame(gpu_input, rec, mc.downsample_ratio)

                if mc.refine_alpha:
                    pha = self._refine_alpha(pha, gpu_input)

                # --- GPU compositing: out = green + pha * (fgr - green) ---
                torch.addcmul(green_tensor.expand_as(fgr), pha, fgr - green_tensor, out=out_gpu)
                out_gpu.clamp_(0.0, 1.0)

                # --- Fast GPU→CPU path ---
                frame_out = out_gpu[0].permute(1, 2, 0).mul(255).to(torch.uint8)
                out_pin.copy_(frame_out.cpu())
                encode_q.put(out_pin.numpy().tobytes())

                frame_idx += 1
                if frame_idx % mc.progress_interval == 0:
                    if progress_callback:
                        pct = min(frame_idx / max(total_frames, 1) * 100, 99)
                        progress_callback("matting", pct)
                    logger.info("  Streamed %d/%d frames", frame_idx, total_frames)

            # Signal encode thread to finish
            encode_q.put(SENTINEL)
            t_encode.join(timeout=30)
            t_decode.join(timeout=10)

            encoder.stdin.close()
            encoder.wait()
            decoder.wait()

            if decode_error[0]:
                logger.error("Decode thread error: %s", decode_error[0])
            if encode_error[0]:
                logger.error("Encode thread error: %s", encode_error[0])

            if encoder.returncode != 0:
                err = encoder.stderr.read().decode() if encoder.stderr else ""
                logger.error("Encode pipe failed: %s", err[-500:])
                return False

            logger.info("Streaming matting complete: %d frames", frame_idx)
            return True

        finally:
            if decoder and decoder.poll() is None:
                decoder.kill()
            if encoder and encoder.poll() is None:
                encoder.kill()
            if self.device and self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _merge_sbs_videos(
        self,
        left_path: Path,
        right_path: Path,
        output_path: Path,
        audio_source: Path,
        has_audio: bool = True,
    ) -> bool:
        """Merge two matted eye videos side-by-side using FFmpeg hstack."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_args = [
            "-i", str(left_path),
            "-i", str(right_path),
        ]
        if has_audio:
            input_args.extend(["-i", str(audio_source)])

        filter_args = [
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]",
        ]
        if has_audio:
            filter_args.extend(["-map", "2:a", "-c:a", "copy"])

        bitrate = self.settings.matte.vr_encode_bitrate
        cmd = self._build_encode_cmd(
            input_args, output_path, bitrate,
            extra_input_args=filter_args,
        )

        logger.info("Merging L/R matted videos to SBS: %s", output_path.name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("SBS merge failed: %s", result.stderr[-500:])
            return False
        return True

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        info: VideoInfo,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Full matting pipeline for a 2D video.

        Dispatches to Intel (OpenVINO + NumPy), streaming (FFmpeg pipes),
        or legacy (disk-based) path.
        """
        mc = self.settings.matte
        # Intel path: OpenVINO + NumPy green screen with despill/refine
        if self._platform == "intel":
            return self._process_video_intel(
                input_path, output_path, progress_callback,
            )
        if mc.use_streaming:
            return self._process_video_streaming_2d(
                input_path, output_path, progress_callback,
            )
        return self._process_video_legacy(
            input_path, output_path, info, progress_callback,
        )

    def _process_video_streaming_2d(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Streaming matting for a 2D video."""
        logger.info("Matting video (streaming): %s", input_path.name)
        mc = self.settings.matte

        if progress_callback:
            progress_callback("probing", 1)

        probe = self._probe_video_pipe_info(input_path)
        success = self._process_video_with_best_backend(
            input_path, output_path, probe,
            bitrate=mc.encode_bitrate,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback("complete" if success else "failed", 100 if success else 0)
        return success

    def _process_video_intel(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """2D green screen matting on Intel (OpenVINO + NumPy).

        Single-pass pipeline: decode → infer → despill → refine → composite → encode.
        No PyTorch required — uses OpenVINO for inference and NumPy/OpenCV
        for post-processing.
        """
        import os
        import queue
        import threading

        import cv2
        import numpy as np

        mc = self.settings.matte
        logger.info("Matting video (Intel): %s", input_path.name)

        if progress_callback:
            progress_callback("probing", 1)

        probe = self._probe_video_pipe_info(input_path)
        orig_w, orig_h = probe["width"], probe["height"]
        fps_str = probe["fps_str"]
        total_frames = probe["total_frames"]

        # Smart output downscale: scale down only when source exceeds the display's
        # perceptual limit (min_output_width, default 1920).
        # Never upscale, never go below min_output_width.
        min_w = mc.min_output_width
        out_scale = min(1.0, max(mc.output_scale, min_w / orig_w)) if orig_w > min_w else 1.0
        if out_scale < 1.0:
            src_w = int(orig_w * out_scale)
            src_h = int(orig_h * out_scale)
            # Ensure even dimensions for encoder
            src_w += src_w % 2
            src_h += src_h % 2
        else:
            src_w, src_h = orig_w, orig_h

        # Inference at downsample_ratio of the output resolution
        infer_h = int(src_h * mc.downsample_ratio)
        infer_w = int(src_w * mc.downsample_ratio)
        infer_h += infer_h % 2
        infer_w += infer_w % 2

        logger.info(
            "Intel 2D: %dx%d%s → infer at %dx%d (%.0f%%), despill=%s, refine=%s",
            src_w, src_h,
            f" (scale={out_scale:.2f} from {orig_w}x{orig_h})" if out_scale < 1.0 else "",
            infer_w, infer_h, mc.downsample_ratio * 100,
            mc.despill, mc.refine_alpha,
        )

        # Prepare OpenVINO engine at inference resolution
        self._prepare_inference_engine(infer_h, infer_w)
        if self._openvino_engine is None:
            logger.error("OpenVINO engine not available for Intel 2D matting")
            return False

        # Pre-allocate green background (float32 for OpenCV blend)
        green_bg = np.full((src_h, src_w, 3), mc.green_color, dtype=np.float32)

        # --- Decode pipe (with optional scale filter) ---
        frame_bytes = src_w * src_h * 3
        pipe_bufsize = frame_bytes * 4

        decode_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-threads", "0",
            "-i", str(input_path),
        ]
        if out_scale < 1.0:
            decode_cmd.extend(["-vf", f"scale={src_w}:{src_h}"])
        decode_cmd.extend([
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-v", "error", "pipe:1",
        ])

        # --- Encode pipe ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_cmd = self._build_encode_cmd_vaapi(
            output_path, src_w, src_h, fps_str,
            audio_source=input_path if probe["has_audio"] else None,
            has_audio=probe["has_audio"],
        )

        # --- Async pipeline ---
        SENTINEL = None
        decode_q: queue.Queue = queue.Queue(maxsize=4)
        encode_q: queue.Queue = queue.Queue(maxsize=4)
        decode_error = [None]
        encode_error = [None]

        decoder = subprocess.Popen(
            decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=pipe_bufsize,
        )
        encoder = subprocess.Popen(
            encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=pipe_bufsize,
        )

        def decode_thread():
            try:
                buf = bytearray(frame_bytes)
                mv = memoryview(buf)
                while True:
                    offset = 0
                    while offset < frame_bytes:
                        n = decoder.stdout.readinto(mv[offset:])
                        if n == 0 or n is None:
                            break
                        offset += n
                    if offset < frame_bytes:
                        break
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(src_h, src_w, 3).copy()
                    decode_q.put(frame)
            except Exception as e:
                decode_error[0] = e
            finally:
                decode_q.put(SENTINEL)

        def encode_thread():
            try:
                fd = encoder.stdin.fileno()
                while True:
                    data = encode_q.get()
                    if data is SENTINEL:
                        break
                    if isinstance(data, np.ndarray):
                        mv = memoryview(data)
                        total = mv.nbytes
                        written = 0
                        while written < total:
                            written += os.write(fd, mv[written:])
                    else:
                        encoder.stdin.write(data)
            except Exception as e:
                encode_error[0] = e

        try:
            t_decode = threading.Thread(target=decode_thread, daemon=True)
            t_encode = threading.Thread(target=encode_thread, daemon=True)
            t_decode.start()
            t_encode.start()

            if progress_callback:
                progress_callback("matting", 5)

            frame_idx = 0
            t_start = time.time()

            while True:
                frame = decode_q.get()
                if frame is SENTINEL:
                    break

                # Downscale for inference
                frame_small = cv2.resize(
                    frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA,
                )
                src_raw = np.ascontiguousarray(frame_small[np.newaxis])  # [1, H, W, 3] uint8

                # OpenVINO inference → alpha only (skip fgr copy for speed)
                pha_raw = self._openvino_engine.infer_alpha_raw(src_raw)
                # pha_raw: [1, 1, infer_H, infer_W] float32
                pha = pha_raw[0, 0]  # [H, W]

                # Optional alpha refinement (at inference resolution)
                if mc.refine_alpha:
                    pha = self._refine_alpha_numpy(
                        pha, mc.alpha_sharpness,
                        threshold_low=mc.refine_threshold_low,
                        threshold_high=mc.refine_threshold_high,
                        laplacian_strength=mc.refine_laplacian_strength,
                        morph_kernel_size=mc.refine_morph_kernel,
                    )

                # Upscale alpha to full resolution as uint8 for fast compositing
                pha_u8 = np.clip(pha * 255.0, 0.0, 255.0).astype(np.uint8)
                if (infer_w, infer_h) != (src_w, src_h):
                    pha_u8 = cv2.resize(pha_u8, (src_w, src_h), interpolation=cv2.INTER_LINEAR)

                # Optional despill on original source at full resolution (uint8 path)
                if mc.despill:
                    frame = self._despill_numpy_u8(
                        frame, pha_u8, mc.despill_strength,
                        dilation_kernel=mc.despill_dilation_kernel,
                        dilation_iters=mc.despill_dilation_iters,
                    )

                # Green screen composite using OpenCV (multi-threaded C++, bypasses GIL).
                # out = src * (alpha/255) + green * (1 - alpha/255)
                alpha_f32 = pha_u8.astype(np.float32) / 255.0
                alpha_3 = cv2.merge([alpha_f32, alpha_f32, alpha_f32])
                frame_out = cv2.convertScaleAbs(
                    frame.astype(np.float32) * alpha_3
                    + green_bg * (1.0 - alpha_3)
                )

                encode_q.put(frame_out)

                frame_idx += 1
                if frame_idx % mc.progress_interval == 0:
                    elapsed = time.time() - t_start
                    fps_actual = frame_idx / max(elapsed, 0.1)
                    pct = min(frame_idx / max(total_frames, 1) * 100, 99)
                    logger.info(
                        "  Intel 2D: %d/%d frames (%.1f fps)",
                        frame_idx, total_frames, fps_actual,
                    )
                    if progress_callback:
                        progress_callback("matting", int(5 + pct * 0.9))

            # Signal encode done
            encode_q.put(SENTINEL)
            t_encode.join(timeout=30)
            t_decode.join(timeout=10)

            encoder.stdin.close()
            encoder.wait()
            decoder.wait()

            if decode_error[0]:
                logger.error("Decode thread error: %s", decode_error[0])
            if encode_error[0]:
                logger.error("Encode thread error: %s", encode_error[0])

            elapsed = time.time() - t_start
            fps_actual = frame_idx / max(elapsed, 0.1)
            logger.info(
                "Intel 2D matting complete: %d frames in %.1fs (%.1f fps)",
                frame_idx, elapsed, fps_actual,
            )

            if progress_callback:
                progress_callback("complete", 100)

            return encoder.returncode == 0

        finally:
            if decoder and decoder.poll() is None:
                decoder.kill()
            if encoder and encoder.poll() is None:
                encoder.kill()

    @staticmethod
    def _despill_numpy(
        fgr: "np.ndarray",
        pha: "np.ndarray",
        green_color: list,
        strength: float = 0.8,
    ) -> "np.ndarray":
        """Remove green spill from foreground edges using NumPy.

        Args:
            fgr: [H, W, 3] float32 foreground (0-1)
            pha: [H, W] float32 alpha (0-1)
            green_color: [R, G, B] int 0-255
            strength: despill intensity (0-1)

        Returns:
            Cleaned fgr with green spill suppressed at edges.
        """
        import cv2
        import numpy as np

        # Edge mask: transition zone pixels where chroma key struggles
        edge_mask = ((pha > 0.02) & (pha < 0.98)).astype(np.float32)
        # Dilate to catch adjacent spill
        kernel = np.ones((5, 5), dtype=np.uint8)
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)

        r, g, b = fgr[:, :, 0], fgr[:, :, 1], fgr[:, :, 2]
        # Classic despill: clamp green to max(red, blue)
        g_clamped = np.minimum(g, np.maximum(r, b))
        # Blend: only affect edge regions
        blend = edge_mask * strength
        fgr[:, :, 1] = g * (1.0 - blend) + g_clamped * blend
        return fgr

    @staticmethod
    def _despill_numpy_u8(
        frame: "np.ndarray",
        pha_u8: "np.ndarray",
        strength: float = 0.8,
        dilation_kernel: int = 7,
        dilation_iters: int = 2,
    ) -> "np.ndarray":
        """Remove green spill from edges using uint8 data. Modifies frame in-place.

        Args:
            frame: [H, W, 3] uint8 source frame
            pha_u8: [H, W] uint8 alpha (0-255)
            strength: despill intensity (0-1)
            dilation_kernel: size of dilation kernel (px)
            dilation_iters: number of dilation iterations

        Returns:
            frame with green channel clamped at edge transitions.
        """
        import cv2
        import numpy as np

        # Edge mask: transition zone (roughly 5-250 in uint8 = 0.02-0.98)
        edge_mask = ((pha_u8 > 5) & (pha_u8 < 250)).astype(np.uint8)
        kernel = np.ones((dilation_kernel, dilation_kernel), dtype=np.uint8)
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=dilation_iters)

        # Only process pixels in the edge mask for speed
        mask_idx = edge_mask > 0
        if not np.any(mask_idx):
            return frame

        r = frame[:, :, 0][mask_idx].astype(np.int16)
        g = frame[:, :, 1][mask_idx].astype(np.int16)
        b = frame[:, :, 2][mask_idx].astype(np.int16)
        g_clamped = np.minimum(g, np.maximum(r, b))
        # Blend with integer math: g_new = g + (g_clamped - g) * strength
        g_new = g + ((g_clamped - g) * int(strength * 256) >> 8)
        frame[:, :, 1][mask_idx] = np.clip(g_new, 0, 255).astype(np.uint8)
        return frame

    @staticmethod
    def _refine_alpha_numpy(
        pha: "np.ndarray",
        sharpness: str = "fine",
        threshold_low: float = 0.01,
        threshold_high: float = 0.99,
        laplacian_strength: float = 0.3,
        morph_kernel_size: int = 5,
    ) -> "np.ndarray":
        """Refine alpha matte using OpenCV box-filter guided filter.

        Args:
            pha: [H, W] float32 alpha (0-1)
            sharpness: "fine" for multi-scale + sharpening, "soft" for legacy
            threshold_low: alpha below this → 0 (black separation)
            threshold_high: alpha above this → 1 (white separation)
            laplacian_strength: edge sharpening intensity
            morph_kernel_size: kernel size for morphological open/close (px)

        Returns:
            Refined alpha [H, W] float32.
        """
        import cv2
        import numpy as np

        def _guided_filter_gray(p, radius, eps=1e-4):
            """Self-guided filter: smooths alpha while preserving edges."""
            k = 2 * radius + 1
            mean_p = cv2.blur(p, (k, k))
            mean_pp = cv2.blur(p * p, (k, k))
            var_p = mean_pp - mean_p * mean_p
            a = var_p / (var_p + eps)
            b = mean_p - a * mean_p
            mean_a = cv2.blur(a, (k, k))
            mean_b = cv2.blur(b, (k, k))
            return mean_a * p + mean_b

        kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)

        if sharpness == "soft":
            # Single-scale guided filter
            pha = _guided_filter_gray(pha, radius=8)
            pha = np.clip(pha, 0.0, 1.0)
            pha[pha < 0.05] = 0.0
            pha[pha > 0.95] = 1.0
            # Morphological open (remove specks) then close (fill gaps)
            pha_u8 = (pha * 255).astype(np.uint8)
            pha_u8 = cv2.morphologyEx(pha_u8, cv2.MORPH_OPEN, kernel)
            pha_u8 = cv2.morphologyEx(pha_u8, cv2.MORPH_CLOSE, kernel)
            pha = pha_u8.astype(np.float32) / 255.0
        else:
            # "fine": multi-scale guided filter + sharpening
            pha_fine = _guided_filter_gray(pha, radius=4)
            pha_bulk = _guided_filter_gray(pha, radius=12)

            # Blend by local variance — high variance uses fine filter
            local_mean = cv2.blur(pha, (9, 9))
            local_var = cv2.blur((pha - local_mean) ** 2, (9, 9))
            blend_w = np.clip(local_var * 50.0, 0.0, 1.0)
            pha = pha_bulk * (1.0 - blend_w) + pha_fine * blend_w

            # Threshold separation
            pha = np.clip(pha, 0.0, 1.0)
            lo, hi = threshold_low, threshold_high
            mask_lo = (pha < lo).astype(np.float32)
            mask_hi = (pha > hi).astype(np.float32)
            mask_mid = 1.0 - mask_lo - mask_hi
            pha_mid = np.clip((pha - lo) / (hi - lo), 0.0, 1.0)
            pha = mask_hi + mask_mid * pha_mid

            # Laplacian edge sharpening
            laplacian = cv2.Laplacian(pha, cv2.CV_32F, ksize=3)
            pha = np.clip(pha + laplacian_strength * laplacian, 0.0, 1.0)

            # Morphological open (remove floating alpha specks) then close (fill edge gaps)
            pha_u8 = (pha * 255).astype(np.uint8)
            pha_u8 = cv2.morphologyEx(pha_u8, cv2.MORPH_OPEN, kernel)
            pha_u8 = cv2.morphologyEx(pha_u8, cv2.MORPH_CLOSE, kernel)
            pha = pha_u8.astype(np.float32) / 255.0

        return pha

    def _process_video_legacy(
        self,
        input_path: Path,
        output_path: Path,
        info: VideoInfo,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Legacy disk-based matting for a 2D video."""
        logger.info("Matting video (legacy): %s", input_path.name)
        mc = self.settings.matte

        temp_dir = Path(self.settings.paths.temp_dir)
        work_dir = temp_dir / f"matte_{input_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if progress_callback:
                progress_callback("extracting", 5)

            frames_dir = work_dir / "frames"
            extractor = FrameExtractor()
            extractor.extract(input_path, frames_dir, fps=info.fps)

            if progress_callback:
                progress_callback("matting", 10)

            matted_dir = work_dir / "matted"
            green = tuple(mc.green_color)
            success = self.matte_frames(
                frames_dir, matted_dir,
                green_color=green,
                model_type=mc.model_type,
                downsample_ratio=mc.downsample_ratio,
                progress_callback=progress_callback,
            )

            if not success:
                return False

            if progress_callback:
                progress_callback("encoding", 90)

            encoder = Encoder(self.settings)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = encoder.encode(
                matted_dir, output_path, info.fps,
                audio_source=input_path,
                bitrate=mc.encode_bitrate,
            )

            if progress_callback:
                progress_callback("complete" if success else "failed", 100 if success else 0)

            return success

        finally:
            if work_dir.exists():
                logger.info("Cleaning up matte temp files: %s", work_dir)
                shutil.rmtree(work_dir)

    def process_vr_sbs(
        self,
        input_path: Path,
        output_path: Path,
        info: VideoInfo,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Matte VR SBS video — each eye independently for temporal consistency.

        Dispatches to alpha_pack (Intel), streaming, or legacy path.
        """
        mc = self.settings.matte

        # Alpha pack path: single-pass pipeline for Intel/VAAPI
        if mc.output_type == "alpha_pack":
            return self._process_vr_sbs_alpha_pack(
                input_path, output_path, progress_callback,
            )

        if mc.use_streaming:
            return self._process_vr_sbs_streaming(
                input_path, output_path, progress_callback,
            )
        return self._process_vr_sbs_legacy(
            input_path, output_path, info, progress_callback,
        )

    def _process_vr_sbs_alpha_pack(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Single-pass VR SBS alpha packing pipeline.

        Decodes full SBS frame, runs inference on left eye only,
        packs alpha into corner dead zones of both eyes, encodes output.
        Designed for Intel Arc + OpenVINO but works on any platform.

        Flow:
            FFmpeg decode → crop left eye → OpenVINO inference → AlphaPacker → FFmpeg encode
        """
        import queue
        import threading

        import cv2
        import numpy as np
        from src.alpha_packer import AlphaPacker

        mc = self.settings.matte
        logger.info("Alpha pack VR SBS: %s", input_path.name)

        if progress_callback:
            progress_callback("probing", 1)

        probe = self._probe_video_pipe_info(input_path)
        full_w, full_h = probe["width"], probe["height"]
        eye_w = full_w // 2
        fps_str = probe["fps_str"]
        total_frames = probe["total_frames"]

        # Alpha packer
        packer = AlphaPacker(scale=mc.alpha_pack_scale)

        # For alpha packing, we only need the matte at the pack scale.
        # Downscaling the eye BEFORE inference saves ~4x compute at 4K.
        infer_h = int(full_h * mc.alpha_pack_scale)
        infer_w = int(eye_w * mc.alpha_pack_scale)
        use_downscaled_infer = (
            self._openvino_engine is not None
            or (self._platform == "intel")
        )
        if use_downscaled_infer:
            logger.info(
                "Alpha pack: inferring at %dx%d (%.0f%% of %dx%d eye)",
                infer_w, infer_h, mc.alpha_pack_scale * 100, eye_w, full_h,
            )

        # Prepare inference engine at the resolution we'll actually infer at
        if use_downscaled_infer:
            self._prepare_inference_engine(infer_h, infer_w)
        else:
            self._prepare_inference_engine(full_h, eye_w)

        # --- FFmpeg decode pipe (full SBS frame, RGB24) ---
        frame_bytes = full_w * full_h * 3
        pipe_bufsize = frame_bytes * 4

        # Software decode is faster than VAAPI for pipe output because
        # VAAPI hwdownload adds GPU→CPU transfer overhead. Use multi-threaded
        # SW decode which is ~30% faster for 8K HEVC when output is CPU memory.
        decode_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-threads", "0",
            "-i", str(input_path),
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-v", "error", "pipe:1",
        ]

        # --- FFmpeg encode pipe ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self._platform == "intel":
            encode_cmd = self._build_encode_cmd_vaapi(
                output_path, full_w, full_h, fps_str,
                audio_source=input_path if probe["has_audio"] else None,
                has_audio=probe["has_audio"],
            )
        else:
            pipe_input_args = [
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{full_w}x{full_h}", "-r", fps_str,
                "-i", "pipe:0",
            ]
            audio_args = None
            if probe["has_audio"]:
                audio_args = ["-i", str(input_path), "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
            encode_cmd = self._build_encode_cmd(
                pipe_input_args, output_path, mc.vr_encode_bitrate,
                extra_input_args=audio_args,
            )

        # --- Async 3-stage pipeline ---
        SENTINEL = None
        decode_q: queue.Queue = queue.Queue(maxsize=4)
        encode_q: queue.Queue = queue.Queue(maxsize=4)
        decode_error = [None]
        encode_error = [None]

        decoder = subprocess.Popen(
            decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=pipe_bufsize,
        )
        encoder = subprocess.Popen(
            encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=pipe_bufsize,
        )

        def decode_thread():
            try:
                # Use readinto with pre-allocated bytearray for ~40% faster decode
                buf = bytearray(frame_bytes)
                mv = memoryview(buf)
                while True:
                    offset = 0
                    while offset < frame_bytes:
                        n = decoder.stdout.readinto(mv[offset:])
                        if n == 0 or n is None:
                            break
                        offset += n
                    if offset < frame_bytes:
                        break
                    # Create writable numpy array from bytearray (zero-copy view)
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(full_h, full_w, 3).copy()
                    decode_q.put(frame)
            except Exception as e:
                decode_error[0] = e
            finally:
                decode_q.put(SENTINEL)

        def encode_thread():
            import os
            try:
                fd = encoder.stdin.fileno()
                while True:
                    data = encode_q.get()
                    if data is SENTINEL:
                        break
                    # Use os.write with memoryview for zero-copy pipe write.
                    # Avoids the 33ms tobytes() copy for 96MB 8K frames.
                    if isinstance(data, np.ndarray):
                        mv = memoryview(data)
                        total = mv.nbytes
                        written = 0
                        while written < total:
                            written += os.write(fd, mv[written:])
                    else:
                        encoder.stdin.write(data)
            except Exception as e:
                encode_error[0] = e

        rec = [None] * 4

        try:
            t_decode = threading.Thread(target=decode_thread, daemon=True)
            t_encode = threading.Thread(target=encode_thread, daemon=True)
            t_decode.start()
            t_encode.start()

            if progress_callback:
                progress_callback("matting", 5)

            frame_idx = 0
            t_start = time.time()

            while True:
                frame = decode_q.get()
                if frame is SENTINEL:
                    break

                # Crop left eye (NumPy slice — zero copy for read)
                left_eye = frame[:, :eye_w, :]

                # Inference — prefer downscaled raw uint8 path for alpha packing
                if use_downscaled_infer and self._openvino_engine is not None and self._openvino_engine.has_raw_input:
                    # Downscale eye to pack resolution before inference (~4x fewer pixels)
                    eye_small = cv2.resize(left_eye, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                    src_raw = np.ascontiguousarray(eye_small[np.newaxis])  # [1, H, W, 3] uint8
                    pha = self._openvino_engine.infer_alpha_raw(src_raw)
                    alpha = pha[0, 0]  # [infer_H, infer_W] float32 — already at pack scale
                    packer.pack(frame, alpha, eye_width=eye_w, presized=True)
                else:
                    # Full-resolution path (ORT/PyTorch/OpenVINO fallback)
                    src = left_eye.astype(np.float32) / 255.0
                    src = np.transpose(src, (2, 0, 1))  # HWC → CHW
                    src = np.expand_dims(src, 0)  # CHW → NCHW
                    fgr, pha, *rec = self._infer_frame(src, rec, mc.downsample_ratio)
                    if hasattr(pha, 'cpu'):
                        alpha = pha[0, 0].cpu().numpy()
                    elif isinstance(pha, np.ndarray):
                        alpha = pha[0, 0] if pha.ndim == 4 else pha
                    else:
                        alpha = np.array(pha[0, 0])
                    packer.pack(frame, alpha, eye_width=eye_w)

                # Pass frame array to encode thread — tobytes runs there, overlapping
                # with next frame's GPU inference. frame is freshly allocated each
                # iteration (from frombuffer().copy()), so no aliasing risk.
                encode_q.put(frame)

                frame_idx += 1
                if frame_idx % mc.progress_interval == 0:
                    elapsed = time.time() - t_start
                    fps_actual = frame_idx / max(elapsed, 0.1)
                    if progress_callback:
                        pct = min(frame_idx / max(total_frames, 1) * 100, 99)
                        progress_callback("matting", pct)
                    logger.info(
                        "  Alpha pack: %d/%d frames (%.1f fps)",
                        frame_idx, total_frames, fps_actual,
                    )

            # Signal encode thread to finish
            encode_q.put(SENTINEL)
            t_encode.join(timeout=30)
            t_decode.join(timeout=10)

            encoder.stdin.close()
            encoder.wait()
            decoder.wait()

            if decode_error[0]:
                logger.error("Decode thread error: %s", decode_error[0])
            if encode_error[0]:
                logger.error("Encode thread error: %s", encode_error[0])

            if encoder.returncode != 0:
                err = encoder.stderr.read().decode() if encoder.stderr else ""
                logger.error("Alpha pack encode failed: %s", err[-500:])
                return False

            elapsed = time.time() - t_start
            fps_avg = frame_idx / max(elapsed, 0.1)
            logger.info(
                "Alpha pack complete: %d frames in %.1fs (%.1f fps avg)",
                frame_idx, elapsed, fps_avg,
            )

            if progress_callback:
                progress_callback("complete", 100)
            return True

        except Exception as e:
            logger.error("Alpha pack pipeline error: %s", e)
            import traceback
            logger.error(traceback.format_exc())
            return False

        finally:
            if decoder and decoder.poll() is None:
                decoder.kill()
            if encoder and encoder.poll() is None:
                encoder.kill()

    def _process_vr_sbs_streaming(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Streaming VR SBS: crop L/R eyes from pipe, matte separately, hstack merge."""
        logger.info("Matting VR SBS (streaming): %s", input_path.name)
        mc = self.settings.matte

        temp_dir = Path(self.settings.paths.temp_dir)
        work_dir = temp_dir / f"matte_vr_{input_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if progress_callback:
                progress_callback("probing", 1)

            probe = self._probe_video_pipe_info(input_path)
            half_w = probe["width"] // 2
            h = probe["height"]

            left_tmp = work_dir / "left_matted.mp4"
            right_tmp = work_dir / "right_matted.mp4"

            # Matte left eye (crop from left half)
            if progress_callback:
                progress_callback("matting_left", 5)

            success = self._process_video_with_best_backend(
                input_path, left_tmp, probe,
                crop_region=(0, 0, half_w, h),
                bitrate=mc.vr_encode_bitrate,
                progress_callback=progress_callback,
            )
            if not success:
                return False

            # Reset recurrent state for right eye (preserves loaded model/engines)
            if self._ort_engine is not None:
                dtype_str = "float16" if getattr(self, "_use_fp16", False) else "float32"
                self._ort_engine.reset_recurrent_state(dtype_str)
            elif self._trt_engine is not None:
                self._trt_engine.reset_recurrent_state()
            else:
                # PyTorch path: need to reload model to clear recurrent state
                self.model = None
            # Reset CUDA graph (rec state shapes may differ between eyes)
            self._cuda_graph = None
            self._cuda_graph_frame_count = 0

            if progress_callback:
                progress_callback("matting_right", 50)

            success = self._process_video_with_best_backend(
                input_path, right_tmp, probe,
                crop_region=(half_w, 0, half_w, h),
                bitrate=mc.vr_encode_bitrate,
                progress_callback=progress_callback,
            )
            if not success:
                return False

            # Merge L/R into SBS output
            if progress_callback:
                progress_callback("merging", 95)

            success = self._merge_sbs_videos(
                left_tmp, right_tmp, output_path,
                audio_source=input_path,
                has_audio=probe["has_audio"],
            )

            if progress_callback:
                progress_callback("complete" if success else "failed", 100 if success else 0)

            return success

        finally:
            if work_dir.exists():
                logger.info("Cleaning up VR matte temp files: %s", work_dir)
                shutil.rmtree(work_dir)

    def _process_vr_sbs_legacy(
        self,
        input_path: Path,
        output_path: Path,
        info: VideoInfo,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Legacy disk-based VR SBS matting."""
        logger.info("Matting VR SBS (legacy): %s", input_path.name)
        mc = self.settings.matte

        temp_dir = Path(self.settings.paths.temp_dir)
        work_dir = temp_dir / f"matte_vr_{input_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            extractor = FrameExtractor()
            vr_proc = VRProcessor()

            if progress_callback:
                progress_callback("extracting", 5)

            frames_dir = work_dir / "frames"
            extractor.extract(input_path, frames_dir, fps=info.fps)

            if progress_callback:
                progress_callback("splitting", 10)

            left_dir = work_dir / "left"
            right_dir = work_dir / "right"
            vr_proc.split_sbs(frames_dir, left_dir, right_dir)

            # Matte left eye
            if progress_callback:
                progress_callback("matting_left", 15)

            left_matted = work_dir / "left_matted"
            green = tuple(mc.green_color)
            self.matte_frames(
                left_dir, left_matted,
                green_color=green,
                model_type=mc.model_type,
                downsample_ratio=mc.downsample_ratio,
            )

            # Reset recurrent state for right eye (preserves loaded model/engines)
            if self._ort_engine is not None:
                dtype_str = "float16" if getattr(self, "_use_fp16", False) else "float32"
                self._ort_engine.reset_recurrent_state(dtype_str)
            elif self._trt_engine is not None:
                self._trt_engine.reset_recurrent_state()
            else:
                self.model = None
            # Reset CUDA graph (rec state shapes may differ between eyes)
            self._cuda_graph = None
            self._cuda_graph_frame_count = 0

            if progress_callback:
                progress_callback("matting_right", 50)

            right_matted = work_dir / "right_matted"
            self.matte_frames(
                right_dir, right_matted,
                green_color=green,
                model_type=mc.model_type,
                downsample_ratio=mc.downsample_ratio,
            )

            # Merge back to SBS
            if progress_callback:
                progress_callback("merging", 85)

            merged_dir = work_dir / "merged"
            vr_proc.merge_sbs(left_matted, right_matted, merged_dir)

            if progress_callback:
                progress_callback("encoding", 90)

            encoder = Encoder(self.settings)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = encoder.encode(
                merged_dir, output_path, info.fps,
                audio_source=input_path,
                bitrate=mc.vr_encode_bitrate,
            )

            if progress_callback:
                progress_callback("complete" if success else "failed", 100 if success else 0)

            return success

        finally:
            if work_dir.exists():
                logger.info("Cleaning up VR matte temp files: %s", work_dir)
                shutil.rmtree(work_dir)


# ---------------------------------------------------------------------------
# Processing Pipeline — orchestrator (from upscale.py:304-408)
# ---------------------------------------------------------------------------
class ProcessingPipeline:
    """Orchestrates the full video processing workflow."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.extractor = FrameExtractor()
        self.vr_processor = VRProcessor()
        self.upscale_engine = UpscaleEngine(settings)
        self.encoder = Encoder(settings)
        self.vr_metadata = VRMetadataPreserver()

    def run(
        self,
        input_path: Path,
        output_path: Path,
        plan: ProcessingPlan,
        info: VideoInfo,
        progress_callback: ProgressCallback = None,
    ) -> bool:
        """Execute the full processing pipeline.

        Returns True on success.
        """
        logger.info("Processing: %s", input_path.name)
        logger.info(
            "Plan: model=%s, scale=%d, worker=%s, bitrate=%s",
            plan.model, plan.scale, plan.worker_type, plan.bitrate,
        )

        if progress_callback:
            progress_callback("starting", 0)

        # Handle lanczos-only (skip AI)
        if plan.skip_ai:
            return self._run_lanczos(input_path, output_path, plan, info)

        temp_dir = Path(self.settings.paths.temp_dir)
        work_dir = temp_dir / f"job_{input_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract frames
            if progress_callback:
                progress_callback("extracting", 5)

            frames_dir = work_dir / "frames_original"
            self.extractor.extract(input_path, frames_dir, fps=info.fps)

            if info.content_type == ContentType.VR_SBS:
                success = self._run_vr_sbs(
                    work_dir, frames_dir, output_path, plan, info, progress_callback,
                )
            elif info.content_type == ContentType.VR_TB:
                success = self._run_vr_tb(
                    work_dir, frames_dir, output_path, plan, info, progress_callback,
                )
            else:
                success = self._run_flat(
                    work_dir, frames_dir, output_path, plan, info, progress_callback,
                )

            # Preserve VR metadata
            if success and info.is_vr:
                if progress_callback:
                    progress_callback("metadata", 95)
                self.vr_metadata.preserve_and_transfer(input_path, output_path)
                self.vr_metadata.ensure_vr_filename(
                    output_path, info.is_vr, info.vr_type or "sbs",
                )

            # Verify output
            if success and output_path.exists():
                out_size = output_path.stat().st_size
                if out_size < 1024:
                    logger.error("Output file too small (%d bytes)", out_size)
                    success = False

            if progress_callback:
                progress_callback("complete" if success else "failed", 100 if success else 0)

            return success

        finally:
            if work_dir.exists():
                logger.info("Cleaning up temp files: %s", work_dir)
                shutil.rmtree(work_dir)

    def _run_flat(
        self, work_dir: Path, frames_dir: Path, output_path: Path,
        plan: ProcessingPlan, info: VideoInfo, cb: ProgressCallback,
    ) -> bool:
        """Standard 2D workflow."""
        upscaled_dir = work_dir / "frames_upscaled"

        if cb:
            cb("upscaling", 20)

        if not self.upscale_engine.upscale(
            frames_dir, upscaled_dir, plan.model, plan.scale,
            plan.tile_size, plan.gpu_id, cb,
        ):
            return False

        if cb:
            cb("encoding", 85)

        return self.encoder.encode(
            upscaled_dir, output_path, info.fps,
            audio_source=Path(info.file_path) if info.file_path else None,
            bitrate=plan.bitrate, encoder=plan.encoder,
        )

    def _run_vr_sbs(
        self, work_dir: Path, frames_dir: Path, output_path: Path,
        plan: ProcessingPlan, info: VideoInfo, cb: ProgressCallback,
    ) -> bool:
        """VR SBS workflow: split -> upscale L/R -> merge -> encode."""
        left_orig = work_dir / "frames_left_orig"
        right_orig = work_dir / "frames_right_orig"
        left_up = work_dir / "frames_left_upscaled"
        right_up = work_dir / "frames_right_upscaled"
        merged_dir = work_dir / "frames_merged"

        if cb:
            cb("splitting", 10)
        self.vr_processor.split_sbs(frames_dir, left_orig, right_orig)

        if cb:
            cb("upscaling_left", 20)
        if not self.upscale_engine.upscale(
            left_orig, left_up, plan.model, plan.scale,
            plan.tile_size, plan.gpu_id, cb,
        ):
            return False

        if cb:
            cb("upscaling_right", 50)
        if not self.upscale_engine.upscale(
            right_orig, right_up, plan.model, plan.scale,
            plan.tile_size, plan.gpu_id, cb,
        ):
            return False

        if cb:
            cb("merging", 80)
        self.vr_processor.merge_sbs(left_up, right_up, merged_dir)

        if cb:
            cb("encoding", 85)
        return self.encoder.encode(
            merged_dir, output_path, info.fps,
            audio_source=Path(info.file_path) if info.file_path else None,
            bitrate=plan.bitrate, encoder=plan.encoder,
        )

    def _run_vr_tb(
        self, work_dir: Path, frames_dir: Path, output_path: Path,
        plan: ProcessingPlan, info: VideoInfo, cb: ProgressCallback,
    ) -> bool:
        """VR TB workflow: split -> upscale top/bottom -> merge -> encode."""
        top_orig = work_dir / "frames_top_orig"
        bot_orig = work_dir / "frames_bottom_orig"
        top_up = work_dir / "frames_top_upscaled"
        bot_up = work_dir / "frames_bottom_upscaled"
        merged_dir = work_dir / "frames_merged"

        if cb:
            cb("splitting", 10)
        self.vr_processor.split_tb(frames_dir, top_orig, bot_orig)

        if cb:
            cb("upscaling_top", 20)
        if not self.upscale_engine.upscale(
            top_orig, top_up, plan.model, plan.scale,
            plan.tile_size, plan.gpu_id, cb,
        ):
            return False

        if cb:
            cb("upscaling_bottom", 50)
        if not self.upscale_engine.upscale(
            bot_orig, bot_up, plan.model, plan.scale,
            plan.tile_size, plan.gpu_id, cb,
        ):
            return False

        if cb:
            cb("merging", 80)
        self.vr_processor.merge_tb(top_up, bot_up, merged_dir)

        if cb:
            cb("encoding", 85)
        return self.encoder.encode(
            merged_dir, output_path, info.fps,
            audio_source=Path(info.file_path) if info.file_path else None,
            bitrate=plan.bitrate, encoder=plan.encoder,
        )

    def _run_lanczos(
        self, input_path: Path, output_path: Path,
        plan: ProcessingPlan, info: VideoInfo,
    ) -> bool:
        """FFmpeg lanczos scaling for 720p->1080p (skip AI)."""
        target_width = info.width * plan.scale
        target_height = info.height * plan.scale
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
            "-c:v", plan.encoder,
            "-preset", self.settings.encode.preset,
            "-crf", str(self.settings.encode.crf),
            "-maxrate", plan.bitrate,
            "-bufsize", str(int(plan.bitrate.rstrip("M")) * 2) + "M",
            "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1",
            "-c:a", "copy",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
