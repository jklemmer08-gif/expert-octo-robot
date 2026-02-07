#!/usr/bin/env python3
"""
PPP Upscaler - Real-ESRGAN video upscaling with VR SBS support
"""

import subprocess
import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import argparse
from PIL import Image

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import VR metadata handler
try:
    from scripts.vr_metadata import VRMetadataHandler
    VR_METADATA_AVAILABLE = True
except ImportError:
    VR_METADATA_AVAILABLE = False

@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    bitrate: int
    is_vr: bool
    vr_type: Optional[str] = None  # 'sbs', 'tb', or None

@dataclass
class UpscaleJob:
    input_path: Path
    output_path: Path
    model: str = "realesr-animevideov3"
    scale: int = 2
    tile_size: int = 512
    gpu_id: int = 0
    force_vr: Optional[bool] = None  # Override VR detection from batch CSV

class PPPUpscaler:
    """Video upscaler using Real-ESRGAN-ncnn-vulkan"""
    
    VR_PATTERNS = [
        "_180", "_360", "_vr", "_VR", "_sbs", "_SBS", 
        "_LR", "_TB", "_3dh", "_6k", "_8k", "_fisheye"
    ]
    
    def __init__(self, config_path: Optional[Path] = None):
        self.base_dir = Path(__file__).parent.parent
        self.realesrgan_bin = self.base_dir / "bin" / "realesrgan-ncnn-vulkan"
        self.models_dir = self.base_dir / "bin" / "models"

        # Load config for paths if available
        config_file = config_path or self.base_dir / "config" / "settings.yaml"
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            paths = cfg.get("paths", {})
            self.temp_dir = Path(paths.get("temp_dir", self.base_dir / "temp"))
            self.output_dir = Path(paths.get("output_dir", self.base_dir / "output"))
        else:
            self.temp_dir = self.base_dir / "temp"
            self.output_dir = self.base_dir / "output"
        
        # VR metadata handler for preserving projection/stereo info
        if VR_METADATA_AVAILABLE:
            self.vr_handler = VRMetadataHandler()
        else:
            self.vr_handler = None
        
        if not self.realesrgan_bin.exists():
            raise FileNotFoundError(f"Real-ESRGAN not found at {self.realesrgan_bin}")
    
    def get_video_info(self, video_path: Path) -> VideoInfo:
        """Extract video metadata using ffprobe"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        video_stream = next(
            (s for s in data["streams"] if s["codec_type"] == "video"), 
            None
        )
        
        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")
        
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        
        # Parse FPS (handle fractional like "30000/1001")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den else 30.0
        else:
            fps = float(fps_str)
        
        duration = float(data["format"].get("duration", 0))
        codec = video_stream.get("codec_name", "unknown")
        bitrate = int(data["format"].get("bit_rate", 0))
        
        # Detect VR content by filename patterns
        filename = video_path.name.lower()
        is_vr = any(p.lower() in filename for p in self.VR_PATTERNS)

        # Also detect by aspect ratio: width >= 2*height is stereoscopic SBS
        if not is_vr and width >= 2 * height:
            is_vr = True

        # Detect VR type (SBS vs TB)
        vr_type = None
        if is_vr:
            if "_tb" in filename or "_ou" in filename:
                vr_type = "tb"  # Top-Bottom / Over-Under
            else:
                vr_type = "sbs"  # Side-by-Side (default for VR)
        
        return VideoInfo(
            width=width, height=height, fps=fps, duration=duration,
            codec=codec, bitrate=bitrate, is_vr=is_vr, vr_type=vr_type
        )
    
    def extract_frames(self, video_path: Path, output_dir: Path, 
                       fps: Optional[float] = None) -> int:
        """Extract frames from video as PNG"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = ["ffmpeg", "-y", "-i", str(video_path)]
        
        if fps:
            cmd.extend(["-vf", f"fps={fps}"])
        
        cmd.extend([
            "-pix_fmt", "rgb24",
            str(output_dir / "frame_%08d.png")
        ])
        
        print(f"Extracting frames from {video_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {result.stderr}")
        
        frame_count = len(list(output_dir.glob("frame_*.png")))
        print(f"  Extracted {frame_count} frames")
        return frame_count
    
    def split_vr_sbs(self, frames_dir: Path, left_dir: Path, right_dir: Path):
        """Split SBS frames into left and right eye"""
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(frames_dir.glob("frame_*.png"))
        print(f"Splitting {len(frames)} SBS frames into L/R...")

        for i, frame in enumerate(frames):
            if (i + 1) % 100 == 0:
                print(f"  Split {i + 1}/{len(frames)} frames...")

            img = Image.open(frame)
            width, height = img.size
            half_width = width // 2

            # Crop left half and save
            img.crop((0, 0, half_width, height)).save(left_dir / frame.name)
            # Crop right half and save
            img.crop((half_width, 0, width, height)).save(right_dir / frame.name)
            img.close()

        print(f"  Split complete: {len(frames)} frames")
    
    def merge_vr_sbs(self, left_dir: Path, right_dir: Path, output_dir: Path):
        """Merge left and right eye frames back to SBS"""
        output_dir.mkdir(parents=True, exist_ok=True)

        left_frames = sorted(left_dir.glob("frame_*.png"))
        print(f"Merging {len(left_frames)} upscaled frames back to SBS...")

        for i, left_frame in enumerate(left_frames):
            if (i + 1) % 100 == 0:
                print(f"  Merged {i + 1}/{len(left_frames)} frames...")

            right_frame = right_dir / left_frame.name
            output_frame = output_dir / left_frame.name

            left_img = Image.open(left_frame)
            right_img = Image.open(right_frame)
            lw, lh = left_img.size
            rw, rh = right_img.size

            merged = Image.new("RGB", (lw + rw, max(lh, rh)))
            merged.paste(left_img, (0, 0))
            merged.paste(right_img, (lw, 0))
            merged.save(output_frame)

            left_img.close()
            right_img.close()
            merged.close()
        
        print(f"  Merge complete: {len(left_frames)} frames")
    
    def upscale_frames(self, input_dir: Path, output_dir: Path, 
                       model: str, scale: int, tile_size: int, gpu_id: int) -> bool:
        """Run Real-ESRGAN on extracted frames"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(self.realesrgan_bin),
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-n", model,
            "-s", str(scale),
            "-t", str(tile_size),
            "-g", str(gpu_id),
            "-f", "png"
        ]
        
        frame_count = len(list(input_dir.glob("frame_*.png")))
        print(f"Upscaling {frame_count} frames with {model} ({scale}x)...")
        print(f"  Command: {' '.join(cmd)}")
        
        # Run with output streaming
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, bufsize=1
        )
        
        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"  {line}")
        
        process.wait()
        
        if process.returncode != 0:
            print(f"  WARNING: Real-ESRGAN exited with code {process.returncode}")
            return False
        
        output_count = len(list(output_dir.glob("frame_*.png")))
        print(f"  Upscaled {output_count}/{frame_count} frames")
        
        return output_count == frame_count
    
    def encode_video(self, frames_dir: Path, output_path: Path, 
                     fps: float, audio_source: Optional[Path] = None,
                     bitrate: str = "100M") -> bool:
        """Encode frames back to video with HEVC"""
        print(f"Encoding to {output_path.name}...")
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%08d.png"),
        ]
        
        # Add audio from source if available
        if audio_source and audio_source.exists():
            cmd.extend(["-i", str(audio_source)])
        
        # HEVC encoding settings for Heresphere/DeoVR
        cmd.extend([
            "-c:v", "libx265",  # Use software encoding for reliability
            "-preset", "medium",
            "-crf", "18",
            "-maxrate", bitrate,
            "-bufsize", str(int(bitrate.rstrip('M')) * 2) + "M",
            "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1",  # Required for Apple/Quest compatibility
        ])
        
        # Map audio if present
        if audio_source and audio_source.exists():
            cmd.extend(["-c:a", "copy", "-map", "0:v", "-map", "1:a"])
        
        cmd.append(str(output_path))
        
        print(f"  Command: {' '.join(cmd[:20])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  Encoding failed: {result.stderr[-500:]}")
            return False
        
        print(f"  Encoded successfully: {output_path}")
        return True
    
    def process_video(self, job: UpscaleJob) -> bool:
        """Full video upscaling pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing: {job.input_path.name}")
        print(f"{'='*60}")
        
        # Get video info
        info = self.get_video_info(job.input_path)

        # Allow batch processor to override VR detection
        if job.force_vr is not None and job.force_vr and not info.is_vr:
            info.is_vr = True
            info.vr_type = "sbs"  # Default to SBS when forced

        print(f"  Source: {info.width}x{info.height} @ {info.fps:.2f}fps")
        print(f"  Duration: {info.duration:.1f}s, VR: {info.is_vr} ({info.vr_type})")
        
        # Calculate target resolution
        target_width = info.width * job.scale
        target_height = info.height * job.scale
        print(f"  Target: {target_width}x{target_height}")
        
        # Determine bitrate based on target resolution
        if target_width >= 7680:
            bitrate = "150M"
        elif target_width >= 5760:
            bitrate = "100M"
        elif target_width >= 3840:
            bitrate = "50M"
        else:
            bitrate = "25M"
        
        # Create temp working directory
        work_dir = self.temp_dir / f"job_{job.input_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract frames
            frames_dir = work_dir / "frames_original"
            self.extract_frames(job.input_path, frames_dir, fps=info.fps)
            
            if info.is_vr and info.vr_type == "sbs":
                # VR SBS workflow: split → upscale L/R → merge
                left_orig = work_dir / "frames_left_orig"
                right_orig = work_dir / "frames_right_orig"
                left_up = work_dir / "frames_left_upscaled"
                right_up = work_dir / "frames_right_upscaled"
                merged_dir = work_dir / "frames_merged"
                
                # Split
                self.split_vr_sbs(frames_dir, left_orig, right_orig)
                
                # Upscale left eye
                print("\nUpscaling LEFT eye...")
                if not self.upscale_frames(
                    left_orig, left_up, job.model, job.scale, job.tile_size, job.gpu_id
                ):
                    return False
                
                # Upscale right eye
                print("\nUpscaling RIGHT eye...")
                if not self.upscale_frames(
                    right_orig, right_up, job.model, job.scale, job.tile_size, job.gpu_id
                ):
                    return False
                
                # Merge back to SBS
                self.merge_vr_sbs(left_up, right_up, merged_dir)
                
                # Encode
                success = self.encode_video(
                    merged_dir, job.output_path, info.fps, 
                    job.input_path, bitrate
                )
            else:
                # Standard 2D workflow
                upscaled_dir = work_dir / "frames_upscaled"
                
                if not self.upscale_frames(
                    frames_dir, upscaled_dir, job.model, job.scale, job.tile_size, job.gpu_id
                ):
                    return False
                
                success = self.encode_video(
                    upscaled_dir, job.output_path, info.fps,
                    job.input_path, bitrate
                )
            
            # Preserve VR metadata from source file
            if success and info.is_vr and self.vr_handler:
                print("Preserving VR metadata...")
                self.vr_handler.preserve_and_transfer(job.input_path, job.output_path)
                # Ensure output filename has VR identifiers for player detection
                source_meta = self.vr_handler.extract_metadata(job.input_path)
                final_path = self.vr_handler.ensure_vr_filename(job.output_path, source_meta)
                if final_path != job.output_path:
                    print(f"  Renamed to: {final_path.name}")
            
            return success
            
        finally:
            # Cleanup temp files
            if work_dir.exists():
                print(f"Cleaning up temp files...")
                shutil.rmtree(work_dir)
    
    def process_sample(self, input_path: Path, output_path: Path,
                       duration: int = 15, start_percent: float = 0.4,
                       **kwargs) -> bool:
        """Process a short sample for QA preview"""
        info = self.get_video_info(input_path)
        start_time = info.duration * start_percent
        
        # Extract sample
        sample_path = self.temp_dir / f"sample_{input_path.stem}.mp4"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c", "copy",
            str(sample_path)
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Process sample
        job = UpscaleJob(
            input_path=sample_path,
            output_path=output_path,
            **kwargs
        )
        
        try:
            return self.process_video(job)
        finally:
            if sample_path.exists():
                sample_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="PPP Video Upscaler")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("-m", "--model", default="realesr-animevideov3",
                        choices=["realesr-animevideov3", "realesrgan-x4plus", "realesrgan-x4plus-anime"],
                        help="Upscaling model")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor")
    parser.add_argument("-t", "--tile-size", type=int, default=512, help="Tile size")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--sample", action="store_true", help="Process 15-second sample only")
    parser.add_argument("--sample-duration", type=int, default=15, help="Sample duration in seconds")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    upscaler = PPPUpscaler()

    # Generate output path if not specified
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "_upscaled" if not args.sample else "_sample"
        output_path = upscaler.output_dir / f"{input_path.stem}{suffix}_{args.scale}x.mp4"
    
    if args.sample:
        success = upscaler.process_sample(
            input_path, output_path,
            duration=args.sample_duration,
            model=args.model,
            scale=args.scale,
            tile_size=args.tile_size,
            gpu_id=args.gpu
        )
    else:
        job = UpscaleJob(
            input_path=input_path,
            output_path=output_path,
            model=args.model,
            scale=args.scale,
            tile_size=args.tile_size,
            gpu_id=args.gpu
        )
        success = upscaler.process_video(job)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
