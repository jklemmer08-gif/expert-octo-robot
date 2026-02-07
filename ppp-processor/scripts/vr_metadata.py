#!/usr/bin/env python3
"""
PPP VR Metadata Handler
Preserves and injects VR metadata for Heresphere/DeoVR compatibility

VR Metadata Standards:
- Google/YouTube Spherical Metadata (sv3d, st3d atoms)
- Filename conventions (_180_sbs, _360_tb, etc.)
- EXIF/XMP tags

References:
- https://github.com/google/spatial-media
- Heresphere documentation
"""

import subprocess
import json
import re
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, asdict
import argparse

@dataclass
class VRMetadata:
    """VR video metadata structure"""
    is_vr: bool = False
    projection: str = "equirectangular"  # equirectangular, fisheye, equi180
    stereo_mode: str = "sbs"  # sbs (side-by-side), tb (top-bottom), mono
    fov_horizontal: int = 180  # 180 or 360
    fov_vertical: int = 180
    source_filename: str = ""
    
    def to_filename_suffix(self) -> str:
        """Generate filename suffix for VR identification"""
        if not self.is_vr:
            return ""
        
        parts = []
        
        # FOV
        if self.fov_horizontal == 360:
            parts.append("360")
        else:
            parts.append("180")
        
        # Stereo mode
        if self.stereo_mode == "sbs":
            parts.append("sbs")
        elif self.stereo_mode == "tb":
            parts.append("tb")
        else:
            parts.append("mono")
        
        return "_" + "_".join(parts)


class VRMetadataHandler:
    """Handle VR metadata extraction, preservation, and injection"""
    
    # Filename patterns for VR detection
    VR_PATTERNS = {
        # FOV patterns
        r'_180': {'fov_horizontal': 180, 'fov_vertical': 180},
        r'_360': {'fov_horizontal': 360, 'fov_vertical': 180},
        r'180x180': {'fov_horizontal': 180, 'fov_vertical': 180},
        r'360x180': {'fov_horizontal': 360, 'fov_vertical': 180},
        
        # Stereo patterns
        r'_sbs|_lr|_3dh|SBS': {'stereo_mode': 'sbs'},
        r'_tb|_ou|_3dv|TB': {'stereo_mode': 'tb'},
        r'_mono': {'stereo_mode': 'mono'},
        
        # Projection patterns
        r'_fisheye|_fe': {'projection': 'fisheye'},
        r'_equi|_equ': {'projection': 'equirectangular'},
        r'MKX200': {'projection': 'fisheye', 'fov_horizontal': 200},
        
        # Studio patterns (usually 180 SBS)
        r'VRBangers|WankzVR|VRHush|SLR|VirtualReal|POVR|RealJam|VRConk': {
            'is_vr': True, 'fov_horizontal': 180, 'stereo_mode': 'sbs'
        },
    }
    
    # Resolution hints for VR
    VR_RESOLUTIONS = {
        (7680, 3840): {'fov_horizontal': 180, 'stereo_mode': 'sbs'},  # 8K SBS
        (6144, 3072): {'fov_horizontal': 180, 'stereo_mode': 'sbs'},  # 6K SBS
        (5760, 2880): {'fov_horizontal': 180, 'stereo_mode': 'sbs'},  # 5.7K SBS
        (4096, 2048): {'fov_horizontal': 180, 'stereo_mode': 'sbs'},  # 4K SBS
        (3840, 1920): {'fov_horizontal': 180, 'stereo_mode': 'sbs'},  # 4K SBS
    }
    
    def __init__(self):
        self.spatial_media_path = self._find_spatial_media()
    
    def _find_spatial_media(self) -> Optional[Path]:
        """Find spatial-media tool if installed"""
        # Check for google spatial-media tool
        result = subprocess.run(
            ["which", "spatial-media"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
        
        # Check in common locations
        locations = [
            Path.home() / "tools" / "spatial-media" / "spatialmedia",
            Path("/usr/local/bin/spatial-media"),
        ]
        for loc in locations:
            if loc.exists():
                return loc
        
        return None
    
    def extract_metadata(self, video_path: Path) -> VRMetadata:
        """Extract VR metadata from video file"""
        metadata = VRMetadata(source_filename=video_path.name)
        
        # Get video info with ffprobe
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        try:
            probe_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            probe_data = {}
        
        # Check video stream
        video_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        
        if video_stream:
            width = video_stream.get("width", 0)
            height = video_stream.get("height", 0)
            
            # Check side_data for spherical metadata
            for side_data in video_stream.get("side_data_list", []):
                if side_data.get("side_data_type") == "Spherical Mapping":
                    metadata.is_vr = True
                    metadata.projection = side_data.get("projection", "equirectangular")
                    
                    # Parse stereo mode
                    if "stereo" in side_data:
                        stereo = side_data["stereo"]
                        if "left-right" in stereo.lower() or "sbs" in stereo.lower():
                            metadata.stereo_mode = "sbs"
                        elif "top-bottom" in stereo.lower():
                            metadata.stereo_mode = "tb"
            
            # Resolution-based VR detection
            res_key = (width, height)
            if res_key in self.VR_RESOLUTIONS:
                metadata.is_vr = True
                for key, val in self.VR_RESOLUTIONS[res_key].items():
                    setattr(metadata, key, val)
            
            # Aspect ratio check (SBS VR is typically 2:1 or wider)
            if height > 0 and width / height >= 1.9:
                metadata.is_vr = True
                metadata.stereo_mode = "sbs"
        
        # Filename-based detection (overrides other methods as it's explicit)
        filename = video_path.name.lower()
        
        for pattern, attrs in self.VR_PATTERNS.items():
            if re.search(pattern, filename, re.IGNORECASE):
                metadata.is_vr = True
                for key, val in attrs.items():
                    setattr(metadata, key, val)
        
        return metadata
    
    def inject_metadata(self, video_path: Path, metadata: VRMetadata, 
                        output_path: Optional[Path] = None) -> Path:
        """Inject VR metadata into video file"""
        if output_path is None:
            output_path = video_path.with_stem(f"{video_path.stem}_meta")
        
        if not metadata.is_vr:
            # Just copy if not VR
            shutil.copy(video_path, output_path)
            return output_path
        
        # Method 1: Use spatial-media tool if available
        if self.spatial_media_path and metadata.projection == "equirectangular":
            return self._inject_with_spatial_media(video_path, metadata, output_path)
        
        # Method 2: Use FFmpeg with custom metadata
        return self._inject_with_ffmpeg(video_path, metadata, output_path)
    
    def _inject_with_spatial_media(self, video_path: Path, metadata: VRMetadata,
                                    output_path: Path) -> Path:
        """Inject using Google's spatial-media tool"""
        cmd = [
            "python3", str(self.spatial_media_path),
            "-i", str(video_path),
            "-o", str(output_path),
        ]
        
        # Add stereo mode
        if metadata.stereo_mode == "sbs":
            cmd.append("--stereo=left-right")
        elif metadata.stereo_mode == "tb":
            cmd.append("--stereo=top-bottom")
        
        subprocess.run(cmd, check=True)
        return output_path
    
    def _inject_with_ffmpeg(self, video_path: Path, metadata: VRMetadata,
                            output_path: Path) -> Path:
        """Inject VR metadata using FFmpeg"""
        # Build metadata string for Heresphere/DeoVR compatibility
        # These tags help players auto-detect settings
        
        meta_tags = []
        
        # Spherical video metadata
        if metadata.fov_horizontal == 360:
            meta_tags.append("-metadata:s:v:0")
            meta_tags.append("spherical=true")
            meta_tags.append("-metadata:s:v:0")
            meta_tags.append("stitched=true")
        
        # Stereo mode tag
        stereo_tag = {
            "sbs": "left_right",
            "tb": "top_bottom", 
            "mono": "mono"
        }.get(metadata.stereo_mode, "left_right")
        
        meta_tags.append("-metadata:s:v:0")
        meta_tags.append(f"stereo_mode={stereo_tag}")
        
        # Projection tag
        meta_tags.append("-metadata:s:v:0")
        meta_tags.append(f"projection={metadata.projection}")
        
        # FOV tags (custom, for players that support it)
        meta_tags.append("-metadata:s:v:0")
        meta_tags.append(f"fov_horizontal={metadata.fov_horizontal}")
        meta_tags.append("-metadata:s:v:0")
        meta_tags.append(f"fov_vertical={metadata.fov_vertical}")
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-c", "copy",  # No re-encoding
            *meta_tags,
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        print(f"Injecting VR metadata: {metadata.stereo_mode}, {metadata.fov_horizontal}°")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: FFmpeg metadata injection failed: {result.stderr[-200:]}")
            # Fall back to copy
            shutil.copy(video_path, output_path)
        
        return output_path
    
    def preserve_and_transfer(self, source_path: Path, dest_path: Path) -> Path:
        """Extract metadata from source and inject into destination"""
        metadata = self.extract_metadata(source_path)
        
        if metadata.is_vr:
            print(f"Preserving VR metadata: {metadata.projection}, "
                  f"{metadata.stereo_mode}, {metadata.fov_horizontal}°")
            return self.inject_metadata(dest_path, metadata)
        
        return dest_path
    
    def ensure_vr_filename(self, video_path: Path, metadata: VRMetadata) -> Path:
        """Ensure filename contains VR identifiers for player detection"""
        if not metadata.is_vr:
            return video_path
        
        filename = video_path.stem.lower()
        
        # Check if already has VR identifiers
        has_fov = any(p in filename for p in ['_180', '_360', '180x', '360x'])
        has_stereo = any(p in filename for p in ['_sbs', '_tb', '_lr', '_ou', '_mono'])
        
        if has_fov and has_stereo:
            return video_path  # Already has identifiers
        
        # Build new filename
        new_stem = video_path.stem
        
        if not has_fov:
            new_stem += f"_{metadata.fov_horizontal}"
        
        if not has_stereo:
            new_stem += f"_{metadata.stereo_mode}"
        
        new_path = video_path.with_stem(new_stem)
        
        if new_path != video_path:
            print(f"Renaming for VR detection: {video_path.name} → {new_path.name}")
            video_path.rename(new_path)
        
        return new_path


def setup_spatial_media():
    """Download and setup Google's spatial-media tool"""
    import os
    
    tools_dir = Path.home() / "tools"
    tools_dir.mkdir(exist_ok=True)
    
    sm_dir = tools_dir / "spatial-media"
    
    if sm_dir.exists():
        print("spatial-media already installed")
        return
    
    print("Installing spatial-media tool...")
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/google/spatial-media.git",
        str(sm_dir)
    ], check=True)
    
    print(f"Installed to: {sm_dir}")
    print("Usage: python3 ~/tools/spatial-media/spatialmedia -i input.mp4 -o output.mp4 --stereo=left-right")


def main():
    parser = argparse.ArgumentParser(description="PPP VR Metadata Handler")
    parser.add_argument("input", nargs="?", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("--extract", action="store_true", help="Extract and show metadata")
    parser.add_argument("--inject", action="store_true", help="Inject VR metadata")
    parser.add_argument("--fix-filename", action="store_true", help="Ensure VR identifiers in filename")
    parser.add_argument("--setup", action="store_true", help="Setup spatial-media tool")
    
    # Manual metadata options
    parser.add_argument("--projection", default="equirectangular",
                        choices=["equirectangular", "fisheye"],
                        help="VR projection type")
    parser.add_argument("--stereo", default="sbs",
                        choices=["sbs", "tb", "mono"],
                        help="Stereo mode")
    parser.add_argument("--fov", type=int, default=180,
                        choices=[180, 360],
                        help="Field of view (degrees)")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_spatial_media()
        return
    
    if not args.input:
        parser.print_help()
        return
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return
    
    handler = VRMetadataHandler()
    
    if args.extract:
        metadata = handler.extract_metadata(input_path)
        print("\nExtracted VR Metadata:")
        print(f"  Is VR: {metadata.is_vr}")
        print(f"  Projection: {metadata.projection}")
        print(f"  Stereo Mode: {metadata.stereo_mode}")
        print(f"  FOV: {metadata.fov_horizontal}° x {metadata.fov_vertical}°")
        print(f"  Filename suffix: {metadata.to_filename_suffix()}")
        return
    
    if args.inject:
        metadata = VRMetadata(
            is_vr=True,
            projection=args.projection,
            stereo_mode=args.stereo,
            fov_horizontal=args.fov,
            fov_vertical=180
        )
        
        output_path = Path(args.output) if args.output else input_path.with_stem(
            f"{input_path.stem}_vr"
        )
        
        handler.inject_metadata(input_path, metadata, output_path)
        print(f"✓ Metadata injected: {output_path}")
        return
    
    if args.fix_filename:
        metadata = handler.extract_metadata(input_path)
        new_path = handler.ensure_vr_filename(input_path, metadata)
        print(f"✓ Filename: {new_path.name}")
        return
    
    # Default: extract and show
    metadata = handler.extract_metadata(input_path)
    print(f"\n{input_path.name}:")
    print(f"  VR: {metadata.is_vr}, {metadata.stereo_mode}, {metadata.fov_horizontal}°")


if __name__ == "__main__":
    main()
