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
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from src.config import Settings
from src.models.schemas import ContentType, ProcessingPlan, VideoInfo

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
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%08d.png"),
        ]

        if audio_source and audio_source.exists():
            cmd.extend(["-i", str(audio_source)])

        # HEVC encoding settings for Heresphere/DeoVR
        bufsize = str(int(bitrate.rstrip("M")) * 2) + "M"
        cmd.extend([
            "-c:v", encoder,
            "-preset", self.settings.encode.preset,
            "-crf", str(self.settings.encode.crf),
            "-maxrate", bitrate,
            "-bufsize", bufsize,
            "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1",
        ])

        if audio_source and audio_source.exists():
            cmd.extend(["-c:a", "copy", "-map", "0:v", "-map", "1:a"])

        cmd.append(str(output_path))

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
