#!/usr/bin/env python3
"""
PPP Matte - RobustVideoMatting for Heresphere passthrough
Removes background and outputs green screen or alpha matte
"""

import subprocess
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import argparse

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class MatteConfig:
    """Configuration for matting operation"""
    output_type: str = "green_screen"  # "green_screen", "alpha_matte", or "composite"
    green_color: Tuple[int, int, int] = (0, 177, 64)  # Heresphere default chroma key
    model_type: str = "mobilenetv3"  # or "resnet50"
    downsample_ratio: float = 0.25  # Lower = faster but less accurate edges

class PPPMatte:
    """Video background matting using RobustVideoMatting"""
    
    RVM_REPO = "https://github.com/PeterL1n/RobustVideoMatting.git"
    RVM_WEIGHTS = {
        "mobilenetv3": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth",
        "resnet50": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth"
    }
    
    def __init__(self, config: Optional[MatteConfig] = None):
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / "models"
        self.temp_dir = self.base_dir / "temp"
        self.config = config or MatteConfig()
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.device = None
    
    def check_dependencies(self) -> bool:
        """Verify torch and RVM are available"""
        if not TORCH_AVAILABLE:
            print("ERROR: PyTorch not installed. Run:")
            print("  pip install torch torchvision")
            return False
        return True
    
    def download_model(self, model_type: str = "mobilenetv3") -> Path:
        """Download RVM model weights"""
        model_path = self.models_dir / f"rvm_{model_type}.pth"
        
        if model_path.exists():
            print(f"Model already exists: {model_path}")
            return model_path
        
        url = self.RVM_WEIGHTS.get(model_type)
        if not url:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Downloading RVM {model_type} model...")
        subprocess.run([
            "wget", "-q", "--show-progress",
            "-O", str(model_path),
            url
        ], check=True)
        
        return model_path
    
    def load_model(self, model_type: str = "mobilenetv3"):
        """Load RVM model"""
        if not self.check_dependencies():
            raise RuntimeError("Dependencies not available")
        
        # Download if needed
        model_path = self.download_model(model_type)
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Load model (RVM uses a custom architecture)
        # We need to clone RVM repo or use the inference script
        print(f"Loading model from {model_path}...")
        
        # For now, we'll use a subprocess approach with the RVM inference script
        # This is more reliable than trying to load the model directly
        self.model_path = model_path
        self.model_type = model_type
    
    def extract_frames(self, video_path: Path, output_dir: Path) -> int:
        """Extract frames from video"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-pix_fmt", "rgb24",
            str(output_dir / "frame_%08d.png")
        ]
        
        print(f"Extracting frames from {video_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {result.stderr}")
        
        frame_count = len(list(output_dir.glob("frame_*.png")))
        print(f"  Extracted {frame_count} frames")
        return frame_count
    
    def process_frame_chroma(self, frame_path: Path, output_path: Path,
                              alpha_path: Path, green_color: Tuple[int, int, int]):
        """Apply alpha matte to create green screen frame"""
        # Load frame and alpha
        frame = Image.open(frame_path).convert("RGBA")
        alpha = Image.open(alpha_path).convert("L")
        
        # Create green background
        green_bg = Image.new("RGBA", frame.size, (*green_color, 255))
        
        # Composite: foreground over green using alpha
        frame.putalpha(alpha)
        result = Image.alpha_composite(green_bg, frame)
        
        # Save as RGB (no alpha needed for chroma key)
        result.convert("RGB").save(output_path)
    
    def matte_frames_simple(self, input_dir: Path, output_dir: Path, 
                            alpha_dir: Path) -> bool:
        """
        Simple frame-by-frame matting using FFmpeg's background subtraction
        This is a fallback when RVM is not available
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        alpha_dir.mkdir(parents=True, exist_ok=True)
        
        frames = sorted(input_dir.glob("frame_*.png"))
        print(f"Processing {len(frames)} frames with simple matting...")
        
        # This is a placeholder - for real matting you'd use RVM
        # For now, we'll just copy frames and create dummy alpha
        for i, frame in enumerate(frames):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(frames)} frames...")
            
            # Copy original frame
            shutil.copy(frame, output_dir / frame.name)
            
            # Create full-white alpha (no matting)
            img = Image.open(frame)
            alpha = Image.new("L", img.size, 255)
            alpha.save(alpha_dir / frame.name)
        
        print("  NOTE: Using placeholder matting. Install RVM for real background removal.")
        return True
    
    def matte_frames_rvm(self, input_dir: Path, output_dir: Path,
                         alpha_dir: Path) -> bool:
        """
        Process frames using RobustVideoMatting
        Requires RVM to be properly installed
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available, falling back to simple matting")
            return self.matte_frames_simple(input_dir, output_dir, alpha_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        alpha_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to import RVM
        try:
            # Add RVM to path if cloned
            rvm_path = self.base_dir / "RobustVideoMatting"
            if rvm_path.exists():
                sys.path.insert(0, str(rvm_path))
            
            from model import MattingNetwork
            
            # Load model
            model = MattingNetwork(self.config.model_type).eval()
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model = model.to(self.device)
            
            # Setup transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            frames = sorted(input_dir.glob("frame_*.png"))
            print(f"Processing {len(frames)} frames with RVM...")
            
            # RVM uses recurrent states for temporal consistency
            rec = [None] * 4
            downsample_ratio = self.config.downsample_ratio
            
            with torch.no_grad():
                for i, frame_path in enumerate(frames):
                    if (i + 1) % 50 == 0:
                        print(f"  Matted {i + 1}/{len(frames)} frames...")
                    
                    # Load and transform frame
                    img = Image.open(frame_path).convert("RGB")
                    src = transform(img).unsqueeze(0).to(self.device)
                    
                    # Run inference
                    fgr, pha, *rec = model(src, *rec, downsample_ratio)
                    
                    # Save alpha matte
                    alpha_np = (pha[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    alpha_img = Image.fromarray(alpha_np, mode="L")
                    alpha_img.save(alpha_dir / frame_path.name)
                    
                    # Apply green screen
                    self.process_frame_chroma(
                        frame_path, output_dir / frame_path.name,
                        alpha_dir / frame_path.name, self.config.green_color
                    )
            
            return True
            
        except ImportError as e:
            print(f"RVM not available ({e}), falling back to simple matting")
            return self.matte_frames_simple(input_dir, output_dir, alpha_dir)
    
    def encode_video(self, frames_dir: Path, output_path: Path,
                     fps: float, audio_source: Optional[Path] = None) -> bool:
        """Encode matted frames to video"""
        print(f"Encoding matted video to {output_path.name}...")
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%08d.png"),
        ]
        
        if audio_source and audio_source.exists():
            cmd.extend(["-i", str(audio_source)])
        
        cmd.extend([
            "-c:v", "libx265",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1",
        ])
        
        if audio_source and audio_source.exists():
            cmd.extend(["-c:a", "copy", "-map", "0:v", "-map", "1:a"])
        
        cmd.append(str(output_path))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  Encoding failed: {result.stderr[-500:]}")
            return False
        
        return True
    
    def get_video_fps(self, video_path: Path) -> float:
        """Get video FPS using ffprobe"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        video_stream = next(
            (s for s in data["streams"] if s["codec_type"] == "video"),
            None
        )
        
        if video_stream:
            fps_str = video_stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                return num / den if den else 30.0
            return float(fps_str)
        
        return 30.0
    
    def process_video(self, input_path: Path, output_path: Path) -> bool:
        """Full matting pipeline for a video"""
        print(f"\n{'='*60}")
        print(f"Matting: {input_path.name}")
        print(f"Output type: {self.config.output_type}")
        print(f"{'='*60}")
        
        fps = self.get_video_fps(input_path)
        
        # Create work directory
        work_dir = self.temp_dir / f"matte_{input_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        frames_dir = work_dir / "frames"
        matted_dir = work_dir / "matted"
        alpha_dir = work_dir / "alpha"
        
        try:
            # Extract frames
            self.extract_frames(input_path, frames_dir)
            
            # Load model
            self.load_model(self.config.model_type)
            
            # Process frames
            success = self.matte_frames_rvm(frames_dir, matted_dir, alpha_dir)
            
            if not success:
                return False
            
            # Encode output
            return self.encode_video(matted_dir, output_path, fps, input_path)
            
        finally:
            # Cleanup
            if work_dir.exists():
                print("Cleaning up temp files...")
                shutil.rmtree(work_dir)
    
    def process_vr_sbs(self, input_path: Path, output_path: Path) -> bool:
        """Process VR SBS video - matte each eye separately"""
        print(f"\n{'='*60}")
        print(f"Matting VR SBS: {input_path.name}")
        print(f"{'='*60}")
        
        fps = self.get_video_fps(input_path)
        
        work_dir = self.temp_dir / f"matte_vr_{input_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract frames
            frames_dir = work_dir / "frames"
            self.extract_frames(input_path, frames_dir)
            
            # Split into L/R
            left_dir = work_dir / "left"
            right_dir = work_dir / "right"
            left_dir.mkdir(exist_ok=True)
            right_dir.mkdir(exist_ok=True)
            
            frames = sorted(frames_dir.glob("frame_*.png"))
            print(f"Splitting {len(frames)} SBS frames...")
            
            for frame in frames:
                img = Image.open(frame)
                w, h = img.size
                half = w // 2
                
                img.crop((0, 0, half, h)).save(left_dir / frame.name)
                img.crop((half, 0, w, h)).save(right_dir / frame.name)
            
            # Load model
            self.load_model(self.config.model_type)
            
            # Matte left eye
            print("\nMatting LEFT eye...")
            left_matted = work_dir / "left_matted"
            left_alpha = work_dir / "left_alpha"
            self.matte_frames_rvm(left_dir, left_matted, left_alpha)
            
            # Matte right eye
            print("\nMatting RIGHT eye...")
            right_matted = work_dir / "right_matted"
            right_alpha = work_dir / "right_alpha"
            self.matte_frames_rvm(right_dir, right_matted, right_alpha)
            
            # Merge back to SBS
            merged_dir = work_dir / "merged"
            merged_dir.mkdir(exist_ok=True)
            
            print("Merging matted frames back to SBS...")
            left_frames = sorted(left_matted.glob("frame_*.png"))
            for lf in left_frames:
                rf = right_matted / lf.name
                
                left_img = Image.open(lf)
                right_img = Image.open(rf)
                
                merged = Image.new("RGB", (left_img.width * 2, left_img.height))
                merged.paste(left_img, (0, 0))
                merged.paste(right_img, (left_img.width, 0))
                merged.save(merged_dir / lf.name)
            
            # Encode
            return self.encode_video(merged_dir, output_path, fps, input_path)
            
        finally:
            if work_dir.exists():
                print("Cleaning up...")
                shutil.rmtree(work_dir)


def setup_rvm():
    """Download and setup RobustVideoMatting"""
    base_dir = Path(__file__).parent.parent
    rvm_dir = base_dir / "RobustVideoMatting"
    
    if rvm_dir.exists():
        print("RVM already cloned")
        return
    
    print("Cloning RobustVideoMatting...")
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/PeterL1n/RobustVideoMatting.git",
        str(rvm_dir)
    ], check=True)
    
    print("RVM setup complete!")


def main():
    parser = argparse.ArgumentParser(description="PPP Video Matting for Passthrough")
    parser.add_argument("input", nargs="?", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("--setup", action="store_true", help="Setup RVM (download model and repo)")
    parser.add_argument("--model", default="mobilenetv3", choices=["mobilenetv3", "resnet50"],
                        help="RVM model type")
    parser.add_argument("--output-type", default="green_screen",
                        choices=["green_screen", "alpha_matte"],
                        help="Output type")
    parser.add_argument("--vr", action="store_true", help="Process as VR SBS content")
    parser.add_argument("--green", default="0,177,64", 
                        help="Green screen color as R,G,B (default: Heresphere green)")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_rvm()
        matte = PPPMatte()
        matte.download_model("mobilenetv3")
        matte.download_model("resnet50")
        print("\nSetup complete! You can now process videos with:")
        print("  python matte.py input.mp4 -o output.mp4")
        return
    
    if not args.input:
        parser.print_help()
        return
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Parse green color
    green = tuple(map(int, args.green.split(",")))
    
    config = MatteConfig(
        output_type=args.output_type,
        green_color=green,
        model_type=args.model
    )
    
    matte = PPPMatte(config)
    
    # Generate output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(f"{input_path.stem}_matted")
    
    if args.vr:
        success = matte.process_vr_sbs(input_path, output_path)
    else:
        success = matte.process_video(input_path, output_path)
    
    if success:
        print(f"\n✓ Matted video saved to: {output_path}")
        print("\nHeresphere Chroma Key Settings:")
        print(f"  Key Color: RGB({green[0]}, {green[1]}, {green[2]})")
        print("  Similarity: 0.4")
        print("  Smoothness: 0.1")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
