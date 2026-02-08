#!/usr/bin/env python3
"""
PPP Passthrough Detector - Auto-detect VR scenes with solid-color backgrounds
for chroma key passthrough on Quest 3S via Heresphere.

Analyzes outer border regions of VR SBS frames for uniform solid colors
(green screen, etc.) and tags candidates in Stash with 'Passthrough'.
"""

import subprocess
import json
import re
import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import numpy as np
from PIL import Image

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.integrations import StashIntegration, StashConfig

# ---------------------------------------------------------------------------
# Path remapping: Stash Docker container paths → host paths
# (from docker-compose.yml volume mounts)
# ---------------------------------------------------------------------------
PATH_REMAPS = {
    "/media/library/":    "/home/jtk1234/media-drive1/",
    "/media/recovered/":  "/home/jtk1234/media-drive2/",
    "/media/drive3/":     "/home/jtk1234/media-drive3/",
    "/media/ppp-output/": "/mnt/ppp-work/ppp/output/",
}


def remap_path(docker_path: str) -> str:
    """Remap a Stash Docker container path to the host filesystem path."""
    for docker_prefix, host_prefix in PATH_REMAPS.items():
        if docker_path.startswith(docker_prefix):
            return host_prefix + docker_path[len(docker_prefix):]
    return docker_path


# ---------------------------------------------------------------------------
# VR detection helpers
# ---------------------------------------------------------------------------
VR_FILENAME_PATTERNS = [
    "_180", "_360", "_vr", "_VR", "_sbs", "_SBS",
    "_LR", "_TB", "_3dh", "_6k", "_8k", "_fisheye",
]


def is_vr_scene(scene: Dict) -> bool:
    """Determine if a Stash scene is VR content.

    Checks:
    - Tags containing 'VR' or known VR identifiers
    - Filename patterns (_180, _sbs, etc.)
    - Stereoscopic aspect ratio (width >= 2 * height)
    """
    # Tag check
    tag_names = [t["name"].lower() for t in scene.get("tags", [])]
    if any("vr" in t for t in tag_names):
        return True

    # File-level checks
    for f in scene.get("files", []):
        basename = f.get("basename", "")
        width = f.get("width", 0)
        height = f.get("height", 0)

        # Filename pattern check
        name_lower = basename.lower()
        if any(p.lower() in name_lower for p in VR_FILENAME_PATTERNS):
            return True

        # Stereoscopic aspect ratio
        if height > 0 and width >= 2 * height:
            return True

    return False


# ---------------------------------------------------------------------------
# FFmpeg / ffprobe helpers
# ---------------------------------------------------------------------------
def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except (json.JSONDecodeError, ValueError):
        return 0.0


def extract_sample_frames(video_path: str, num_frames: int = 5,
                          output_dir: Optional[str] = None) -> List[str]:
    """Extract sample frames via fast keyframe seeks.

    Returns list of paths to extracted PNG frames.
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        return []

    if output_dir is None:
        base = Path(__file__).parent.parent
        output_dir = str(base / "temp" / "passthrough_frames")

    os.makedirs(output_dir, exist_ok=True)

    # Space timestamps evenly, avoiding first/last 5%
    start = duration * 0.05
    end = duration * 0.95
    if num_frames == 1:
        timestamps = [duration / 2]
    else:
        step = (end - start) / (num_frames - 1)
        timestamps = [start + i * step for i in range(num_frames)]

    frame_paths = []
    for i, ts in enumerate(timestamps):
        out_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts:.2f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(out_path):
            frame_paths.append(out_path)

    return frame_paths


# ---------------------------------------------------------------------------
# SBS frame splitting
# ---------------------------------------------------------------------------
def split_sbs_frame(frame_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side frame into left and right eye numpy arrays."""
    img = Image.open(frame_path)
    arr = np.array(img)
    img.close()
    h, w = arr.shape[:2]
    half = w // 2
    return arr[:, :half], arr[:, half:]


# ---------------------------------------------------------------------------
# Border analysis
# ---------------------------------------------------------------------------
@dataclass
class BorderResult:
    """Analysis result for a single border region."""
    region: str          # e.g. "left_eye_right_edge"
    mean_rgb: Tuple[int, int, int] = (0, 0, 0)
    variance: float = 0.0
    uniformity: float = 0.0   # % of pixels near mean color
    is_solid: bool = False


def analyze_border_region(pixels: np.ndarray, region_name: str,
                          distance_threshold: int = 30,
                          variance_threshold: float = 500,
                          uniformity_threshold: float = 0.70) -> BorderResult:
    """Analyze a border region for solid color.

    Args:
        pixels: (N, 3) RGB pixel array from the border region
        region_name: descriptive name for this region
        distance_threshold: max per-channel distance from mean to be "near"
        variance_threshold: max total variance to qualify as solid
        uniformity_threshold: min fraction of pixels near mean to qualify
    """
    if pixels.size == 0:
        return BorderResult(region=region_name)

    # Reshape to (N, 3) if needed
    if pixels.ndim == 3:
        pixels = pixels.reshape(-1, 3)

    pixels = pixels.astype(np.float64)

    mean_rgb = pixels.mean(axis=0)
    # Per-channel variance, summed
    variance = pixels.var(axis=0).sum()

    # Fraction of pixels within distance_threshold of mean on all channels
    diffs = np.abs(pixels - mean_rgb)
    near_mask = np.all(diffs <= distance_threshold, axis=1)
    uniformity = near_mask.mean()

    is_solid = variance < variance_threshold and uniformity > uniformity_threshold

    return BorderResult(
        region=region_name,
        mean_rgb=tuple(int(round(c)) for c in mean_rgb),
        variance=round(variance, 1),
        uniformity=round(uniformity, 4),
        is_solid=is_solid,
    )


def extract_border_pixels(eye: np.ndarray, edge: str,
                          border_frac: float = 0.12) -> np.ndarray:
    """Extract pixels from a border region of an eye view.

    Args:
        eye: (H, W, 3) numpy array for one eye
        edge: one of "left", "right", "top", "bottom"
        border_frac: fraction of dimension to use as border width
    """
    h, w = eye.shape[:2]
    bw = max(1, int(w * border_frac))
    bh = max(1, int(h * border_frac))

    if edge == "left":
        return eye[:, :bw]
    elif edge == "right":
        return eye[:, w - bw:]
    elif edge == "top":
        return eye[:bh, :]
    elif edge == "bottom":
        return eye[h - bh:, :]
    else:
        raise ValueError(f"Unknown edge: {edge}")


# ---------------------------------------------------------------------------
# Color safety checks
# ---------------------------------------------------------------------------
def is_skin_tone(r: int, g: int, b: int) -> bool:
    """Check if an RGB color falls within common skin tone ranges.

    Uses a broad set of hue/saturation rules to cover light through dark
    skin tones.  Returns True if the color is likely a skin color and
    therefore unsafe to use as a chroma key.
    """
    # Very dark colors (near-black) — common hair/eye color.
    # But allow pure black (< 10) as a valid chroma key background.
    if r < 50 and g < 50 and b < 50:
        if r < 10 and g < 10 and b < 10:
            return False  # Pure black is a valid chroma bg
        return True

    # Skin-tone heuristic: R > G > B with warm bias
    # Covers a wide range from pale to deep skin tones
    if r > 60 and g > 30 and b > 10:
        # Warm ratio: R should dominate, B should be lowest
        if r > g > b:
            # Check it's not too saturated (which would be orange/red, not skin)
            spread = r - b
            if spread < 150:
                return True

        # Medium/olive skin: R and G close, B lower
        if r > b and g > b and abs(r - g) < 40:
            rg_avg = (r + g) / 2
            if b < rg_avg * 0.85 and rg_avg > 80:
                return True

    # Blonde/brown hair: warm browns and tans
    if 80 < r < 220 and 50 < g < 180 and 20 < b < 120:
        if r > g > b and (r - b) > 30 and (r - b) < 130:
            return True

    return False


def compute_center_conflict(eye: np.ndarray, border_color: Tuple[int, int, int],
                            distance_threshold: int = 40) -> float:
    """Measure what fraction of center-region pixels match the border color.

    Samples the central 50% of the eye view (where the performer is) and
    returns the fraction of pixels that are within distance_threshold of
    the detected border color on all channels.

    A high conflict ratio means the chroma key would eat into the performer.
    """
    h, w = eye.shape[:2]
    # Central 50% crop
    y0 = h // 4
    y1 = h - h // 4
    x0 = w // 4
    x1 = w - w // 4
    center = eye[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)

    if center.size == 0:
        return 0.0

    color = np.array(border_color, dtype=np.float64)
    diffs = np.abs(center - color)
    near_mask = np.all(diffs <= distance_threshold, axis=1)
    return float(near_mask.mean())


# ---------------------------------------------------------------------------
# Optimized chroma color selection
# ---------------------------------------------------------------------------
def _sample_center_pixels(eye: np.ndarray, max_samples: int = 5000) -> np.ndarray:
    """Subsample center-region pixels from an eye view for color analysis."""
    h, w = eye.shape[:2]
    center = eye[h // 4:h - h // 4, w // 4:w - w // 4].reshape(-1, 3)
    if len(center) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(center), max_samples, replace=False)
        center = center[idx]
    return center.astype(np.float64)


def optimize_chroma_color(border_pixels: np.ndarray,
                          center_pixels: np.ndarray,
                          base_color: Tuple[int, int, int],
                          search_radius: int = 25,
                          step: int = 5) -> Tuple[Tuple[int, int, int], float]:
    """Find the border color shade that maximizes distance from performer pixels.

    Starting from the detected mean border color, searches nearby shades
    within the border pixel distribution and picks the one whose minimum
    Euclidean distance to any center (performer) pixel is largest.

    Args:
        border_pixels: (N, 3) flat array of all solid-border pixels
        center_pixels: (M, 3) flat array of performer center pixels
        base_color: the naive mean border color
        search_radius: max per-channel deviation from base to search
        step: grid step size for search

    Returns:
        (optimal_rgb, min_distance) — the best chroma color and its
        minimum Euclidean distance to any performer pixel.
    """
    if center_pixels.size == 0 or border_pixels.size == 0:
        return base_color, 0.0

    border_pixels = border_pixels.astype(np.float64)
    center_pixels = center_pixels.astype(np.float64)

    # Build candidate grid around base_color, only keeping colors that
    # actually appear in the border region (within distance 15 of a
    # border pixel, so we don't hallucinate colors not in the background)
    base = np.array(base_color, dtype=np.float64)
    offsets = range(-search_radius, search_radius + 1, step)
    candidates = []
    for dr in offsets:
        for dg in offsets:
            for db in offsets:
                c = base + np.array([dr, dg, db])
                c = np.clip(c, 0, 255)
                # Verify this candidate is plausible: at least 30% of border
                # pixels are within 20 of it
                diffs = np.abs(border_pixels - c)
                near = np.all(diffs <= 20, axis=1).mean()
                if near >= 0.30:
                    candidates.append(c)

    if not candidates:
        candidates = [base]

    candidates = np.array(candidates)  # (K, 3)

    # For each candidate, find minimum Euclidean distance to any center pixel.
    # Use chunked computation to avoid OOM on large pixel sets.
    best_color = base_color
    best_min_dist = 0.0

    for cand in candidates:
        dists = np.sqrt(((center_pixels - cand) ** 2).sum(axis=1))
        # Use 2nd percentile instead of absolute min to be robust to outliers
        min_dist = float(np.percentile(dists, 2))
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_color = tuple(int(round(v)) for v in cand)

    return best_color, best_min_dist


def compute_recommended_tolerance(min_distance: float) -> float:
    """Convert minimum color distance to a Heresphere tolerance value.

    Heresphere's chroma key tolerance is roughly 0.0-1.0. We recommend
    a value that keys the background but leaves a safety margin so
    performer pixels aren't eaten.

    Rule: tolerance = (min_distance * 0.6) / 255, clamped to [0.05, 0.50].
    The 0.6 factor keeps us well inside the safe zone.
    """
    tol = (min_distance * 0.6) / 255.0
    return round(max(0.05, min(0.50, tol)), 3)


# ---------------------------------------------------------------------------
# Scene-level scoring
# ---------------------------------------------------------------------------
@dataclass
class SceneScore:
    """Scoring result for a single scene."""
    scene_id: str
    title: str = ""
    confidence: float = 0.0
    dominant_color: Tuple[int, int, int] = (0, 0, 0)
    optimized_color: Tuple[int, int, int] = (0, 0, 0)  # best chroma shade
    min_performer_distance: float = 0.0  # Euclidean distance to nearest performer px
    recommended_tolerance: float = 0.0   # suggested Heresphere tolerance
    num_frames_analyzed: int = 0
    frame_scores: List[float] = field(default_factory=list)
    region_details: List[Dict] = field(default_factory=list)
    center_conflict: float = 0.0       # avg fraction of center matching border color
    skin_tone_rejected: bool = False    # True if border color is a skin/hair tone


def score_scene_frames(scene_id: str, frame_paths: List[str],
                       title: str = "", verbose: bool = False) -> SceneScore:
    """Score a scene's frames for passthrough candidacy.

    For VR 180 SBS:
    - Left eye RIGHT edge and Right eye LEFT edge are outer periphery (3x weight)
    - Top/bottom edges of both eyes are supplementary (1x weight)
    """
    all_frame_scores = []
    all_region_details = []
    color_accumulator = []

    for frame_path in frame_paths:
        try:
            left_eye, right_eye = split_sbs_frame(frame_path)
        except Exception as e:
            if verbose:
                print(f"    Failed to split frame {frame_path}: {e}")
            continue

        # Analyze border regions with weights
        regions = [
            # Outer periphery (high weight)
            (left_eye, "right", "left_eye_right_edge", 3.0),
            (right_eye, "left", "right_eye_left_edge", 3.0),
            # Top/bottom supplementary (low weight)
            (left_eye, "top", "left_eye_top", 1.0),
            (left_eye, "bottom", "left_eye_bottom", 1.0),
            (right_eye, "top", "right_eye_top", 1.0),
            (right_eye, "bottom", "right_eye_bottom", 1.0),
        ]

        weighted_sum = 0.0
        weight_total = 0.0
        frame_details = []

        for eye, edge, name, weight in regions:
            pixels = extract_border_pixels(eye, edge)
            result = analyze_border_region(pixels, name)

            if verbose:
                solid_mark = "SOLID" if result.is_solid else "     "
                print(f"    {solid_mark} {name:30s}  "
                      f"rgb=({result.mean_rgb[0]:3d},{result.mean_rgb[1]:3d},{result.mean_rgb[2]:3d})  "
                      f"var={result.variance:8.1f}  unif={result.uniformity:.3f}")

            frame_details.append({
                "region": result.region,
                "mean_rgb": result.mean_rgb,
                "variance": result.variance,
                "uniformity": result.uniformity,
                "is_solid": result.is_solid,
                "weight": weight,
            })

            if result.is_solid:
                weighted_sum += weight
                color_accumulator.append(result.mean_rgb)

            weight_total += weight

        frame_score = (weighted_sum / weight_total * 100) if weight_total > 0 else 0
        all_frame_scores.append(frame_score)
        all_region_details.extend(frame_details)

        if verbose:
            print(f"    Frame score: {frame_score:.1f}")

    # Overall confidence: average across frames
    confidence = sum(all_frame_scores) / len(all_frame_scores) if all_frame_scores else 0

    # Dominant color: average of all solid region colors
    if color_accumulator:
        avg_color = tuple(
            int(round(sum(c[i] for c in color_accumulator) / len(color_accumulator)))
            for i in range(3)
        )
    else:
        avg_color = (0, 0, 0)

    # --- Safety checks ---

    # 1. Skin-tone rejection: if the detected border color looks like skin,
    #    hair, or eyes, it's unsafe for chroma key
    skin_rejected = False
    if confidence > 0 and is_skin_tone(*avg_color):
        skin_rejected = True
        if verbose:
            print(f"    SKIN-TONE REJECT: rgb({avg_color[0]},{avg_color[1]},{avg_color[2]}) "
                  f"looks like skin/hair — forcing confidence to 0")
        confidence = 0

    # 2. Center-region conflict: check how much of the performer area
    #    matches the detected border color.  Penalize proportionally.
    avg_conflict = 0.0
    if confidence > 0 and avg_color != (0, 0, 0):
        conflict_scores = []
        for frame_path in frame_paths:
            try:
                left_eye, right_eye = split_sbs_frame(frame_path)
                c_left = compute_center_conflict(left_eye, avg_color)
                c_right = compute_center_conflict(right_eye, avg_color)
                conflict_scores.append((c_left + c_right) / 2)
            except Exception:
                continue

        if conflict_scores:
            avg_conflict = sum(conflict_scores) / len(conflict_scores)

            if verbose:
                print(f"    Center conflict: {avg_conflict:.3f} "
                      f"({avg_conflict*100:.1f}% of performer pixels match border color)")

            # Penalty: if >5% of center pixels match, start reducing confidence
            # At 15%+ conflict, confidence drops to 0
            if avg_conflict > 0.05:
                penalty = min(1.0, (avg_conflict - 0.05) / 0.10)
                old_conf = confidence
                confidence *= (1.0 - penalty)
                if verbose:
                    print(f"    Conflict penalty: {old_conf:.1f} -> {confidence:.1f} "
                          f"(penalty={penalty:.2f})")

    # 3. Optimized chroma color: pick the border shade with maximum
    #    separation from performer center pixels
    opt_color = avg_color
    min_dist = 0.0
    rec_tolerance = 0.0

    if confidence > 0 and avg_color != (0, 0, 0):
        # Gather border and center pixels across all frames
        all_border_px = []
        all_center_px = []
        for frame_path in frame_paths:
            try:
                left_eye, right_eye = split_sbs_frame(frame_path)
            except Exception:
                continue

            # Border pixels from outer-periphery edges (the important ones)
            for eye, edge in [(left_eye, "right"), (right_eye, "left")]:
                bp = extract_border_pixels(eye, edge).reshape(-1, 3)
                all_border_px.append(bp)

            # Center performer pixels
            all_center_px.append(_sample_center_pixels(left_eye))
            all_center_px.append(_sample_center_pixels(right_eye))

        if all_border_px and all_center_px:
            border_combined = np.concatenate(all_border_px)
            center_combined = np.concatenate(all_center_px)

            opt_color, min_dist = optimize_chroma_color(
                border_combined, center_combined, avg_color
            )
            rec_tolerance = compute_recommended_tolerance(min_dist)

            if verbose:
                print(f"    Optimized chroma: rgb({opt_color[0]},{opt_color[1]},{opt_color[2]})  "
                      f"(was rgb({avg_color[0]},{avg_color[1]},{avg_color[2]}))")
                print(f"    Min performer distance: {min_dist:.1f}  "
                      f"-> recommended tolerance: {rec_tolerance:.3f}")

    return SceneScore(
        scene_id=scene_id,
        title=title,
        confidence=round(confidence, 1),
        dominant_color=avg_color,
        optimized_color=opt_color,
        min_performer_distance=round(min_dist, 1),
        recommended_tolerance=rec_tolerance,
        num_frames_analyzed=len(all_frame_scores),
        frame_scores=all_frame_scores,
        region_details=all_region_details,
        center_conflict=round(avg_conflict, 4),
        skin_tone_rejected=skin_rejected,
    )


# ---------------------------------------------------------------------------
# Color conversion
# ---------------------------------------------------------------------------
def rgb_to_heresphere(r: int, g: int, b: int) -> Dict[str, float]:
    """Convert RGB 0-255 to Heresphere 0.0-1.0 float format."""
    return {
        "r": round(r / 255.0, 4),
        "g": round(g / 255.0, 4),
        "b": round(b / 255.0, 4),
    }


# ---------------------------------------------------------------------------
# PassthroughDetector class
# ---------------------------------------------------------------------------
class PassthroughDetector:
    """Main detector: queries Stash, analyzes frames, tags candidates."""

    GRAPHQL_FIND_SCENES = """
    query FindVRScenes($filter: FindFilterType!, $scene_filter: SceneFilterType!) {
        findScenes(filter: $filter, scene_filter: $scene_filter) {
            count
            scenes {
                id title details
                tags { id name }
                files { path width height basename duration }
            }
        }
    }
    """

    def __init__(self, stash: StashIntegration, threshold: float = 70,
                 num_frames: int = 5, verbose: bool = False,
                 source_tag: str = "export_deovr"):
        self.stash = stash
        self.threshold = threshold
        self.num_frames = num_frames
        self.verbose = verbose
        self.source_tag = source_tag

        base = Path(__file__).parent.parent
        self.temp_dir = base / "temp" / "passthrough_frames"
        self.output_dir = base / "output"

    def find_candidate_scenes(self, limit: Optional[int] = None) -> List[Dict]:
        """Query Stash for VR scenes, excluding those already tagged Passthrough.

        Uses paginated GraphQL queries. Filters by source_tag (default
        'export_deovr', can be overridden to 'VR' etc.).
        Client-side filters out scenes already tagged Passthrough and
        non-VR scenes.
        """
        # Get the source tag ID
        export_tag_id = self.stash.get_or_create_tag(self.source_tag)
        if not export_tag_id:
            print(f"Error: Could not find/create '{self.source_tag}' tag")
            return []

        candidates = []
        page = 1
        per_page = 100

        while True:
            variables = {
                "filter": {
                    "page": page,
                    "per_page": per_page,
                    "sort": "title",
                    "direction": "ASC",
                },
                "scene_filter": {
                    "tags": {
                        "value": [export_tag_id],
                        "modifier": "INCLUDES_ALL",
                    }
                },
            }

            result = self.stash._query(self.GRAPHQL_FIND_SCENES, variables)
            data = result.get("findScenes", {})
            scenes = data.get("scenes", [])
            total = data.get("count", 0)

            if not scenes:
                break

            for scene in scenes:
                # Skip if already tagged Passthrough
                tag_names = [t["name"] for t in scene.get("tags", [])]
                if "Passthrough" in tag_names:
                    continue

                # Must be VR
                if not is_vr_scene(scene):
                    continue

                candidates.append(scene)

                if limit and len(candidates) >= limit:
                    return candidates

            if page * per_page >= total:
                break
            page += 1

        return candidates

    def analyze_scene(self, scene: Dict) -> Optional[SceneScore]:
        """Extract frames from a scene, split SBS, analyze borders, score."""
        scene_id = scene["id"]
        title = scene.get("title", "") or ""
        files = scene.get("files", [])

        if not files:
            if self.verbose:
                print(f"  Scene {scene_id}: no files, skipping")
            return None

        # Use first file
        file_info = files[0]
        docker_path = file_info.get("path", "")
        local_path = remap_path(docker_path)

        if not os.path.exists(local_path):
            print(f"  Scene {scene_id}: file not found at {local_path}")
            return None

        if self.verbose:
            print(f"  Analyzing: {os.path.basename(local_path)}")

        # Create scene-specific temp dir
        scene_temp = str(self.temp_dir / f"scene_{scene_id}")

        try:
            frame_paths = extract_sample_frames(
                local_path, num_frames=self.num_frames, output_dir=scene_temp
            )

            if not frame_paths:
                print(f"  Scene {scene_id}: could not extract frames")
                return None

            score = score_scene_frames(
                scene_id, frame_paths, title=title, verbose=self.verbose
            )
            return score

        finally:
            # Clean up temp frames
            import shutil
            if os.path.exists(scene_temp):
                shutil.rmtree(scene_temp, ignore_errors=True)

    def tag_scene(self, scene: Dict, score: SceneScore) -> bool:
        """Tag a scene with Passthrough and store chroma color in details."""
        scene_id = scene["id"]

        # Add Passthrough tag
        self.stash.add_tag_to_scene(scene_id, "Passthrough")

        # Store optimized color + tolerance in scene details
        r, g, b = score.optimized_color
        chroma_line = (f"[PPP-Chroma: R={r}, G={g}, B={b}]"
                       f" [PPP-Tolerance: {score.recommended_tolerance:.3f}]")

        current = scene.get("details", "") or ""

        # Don't duplicate — remove old PPP tags before inserting
        if "[PPP-Chroma:" in current or "[PPP-Tolerance:" in current:
            current = re.sub(r"\[PPP-Chroma:.*?\](\s*\[PPP-Tolerance:.*?\])?", "", current)
            current = current.strip()

        current = f"{current}\n\n{chroma_line}" if current else chroma_line

        mutation = """
        mutation UpdateScene($input: SceneUpdateInput!) {
            sceneUpdate(input: $input) { id }
        }
        """
        self.stash._query(mutation, {
            "input": {
                "id": scene_id,
                "details": current.strip(),
            }
        })

        return True

    def run(self, dry_run: bool = False, scene_id: Optional[str] = None,
            limit: Optional[int] = None) -> List[Dict]:
        """Main pipeline: find scenes, analyze, tag, report.

        Returns list of result dicts for report generation.
        """
        results = []

        # Single scene mode
        if scene_id:
            scene = self.stash.find_scene_by_id(scene_id)
            if not scene:
                print(f"Scene {scene_id} not found in Stash")
                return results

            # Fetch full scene data with files
            scene = self._fetch_full_scene(scene_id)
            if not scene:
                return results

            scenes = [scene]
            print(f"Analyzing single scene: {scene.get('title', scene_id)}")
        else:
            print("Finding candidate VR scenes...")
            scenes = self.find_candidate_scenes(limit=limit)
            print(f"Found {len(scenes)} candidates to analyze")

        if not scenes:
            print("No scenes to analyze.")
            return results

        tagged_count = 0

        for i, scene in enumerate(scenes, 1):
            sid = scene["id"]
            title = scene.get("title", "") or os.path.basename(
                scene.get("files", [{}])[0].get("path", sid)
            )
            print(f"\n[{i}/{len(scenes)}] {title}")

            score = self.analyze_scene(scene)
            if score is None:
                continue

            hs_naive = rgb_to_heresphere(*score.dominant_color)
            hs_optimized = rgb_to_heresphere(*score.optimized_color)

            result = {
                "scene_id": sid,
                "title": title,
                "confidence": score.confidence,
                "dominant_color_rgb": score.dominant_color,
                "dominant_color_heresphere": hs_naive,
                "optimized_color_rgb": score.optimized_color,
                "optimized_color_heresphere": hs_optimized,
                "min_performer_distance": score.min_performer_distance,
                "recommended_tolerance": score.recommended_tolerance,
                "num_frames": score.num_frames_analyzed,
                "center_conflict": score.center_conflict,
                "skin_tone_rejected": score.skin_tone_rejected,
                "tagged": False,
            }

            if score.confidence >= self.threshold:
                r, g, b = score.optimized_color
                print(f"  -> CANDIDATE  confidence={score.confidence:.1f}%  "
                      f"chroma=rgb({r},{g},{b})  tolerance={score.recommended_tolerance:.3f}")

                if not dry_run:
                    self.tag_scene(scene, score)
                    result["tagged"] = True
                    tagged_count += 1
                    print(f"  -> Tagged with 'Passthrough'")
                else:
                    print(f"  -> (dry run, not tagging)")
            else:
                print(f"  -> SKIP  confidence={score.confidence:.1f}% (below {self.threshold}%)")

            results.append(result)

        print(f"\n{'='*60}")
        print(f"Summary: {len(results)} analyzed, "
              f"{sum(1 for r in results if r['confidence'] >= self.threshold)} candidates, "
              f"{tagged_count} tagged")

        return results

    def _fetch_full_scene(self, scene_id: str) -> Optional[Dict]:
        """Fetch a scene with full file info (width, height, etc.)."""
        query = """
        query FindScene($id: ID!) {
            findScene(id: $id) {
                id title details
                tags { id name }
                files { path width height basename duration }
            }
        }
        """
        result = self.stash._query(query, {"id": scene_id})
        return result.get("findScene")


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------
def save_report(results: List[Dict], output_path: str):
    """Save JSON report of detection results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = {
        "generated": __import__("datetime").datetime.now().isoformat(),
        "total_analyzed": len(results),
        "candidates": sum(1 for r in results if r.get("tagged") or r["confidence"] >= 70),
        "tagged": sum(1 for r in results if r.get("tagged")),
        "scenes": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved: {output_path}")


# ---------------------------------------------------------------------------
# Survey mode: find diverse test scenes by background color
# ---------------------------------------------------------------------------
def color_luminance(r: int, g: int, b: int) -> float:
    """Perceived luminance (ITU-R BT.601) in 0-255 range."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def run_survey(detector: PassthroughDetector, pick_count: int = 10,
               scan_limit: Optional[int] = None) -> List[Dict]:
    """Scan VR scenes and select a diverse set sorted by background luminance.

    Analyzes all candidate scenes (or up to scan_limit), then picks
    pick_count scenes evenly spaced from darkest to lightest background.
    Only scenes with confidence > 0 (i.e. detected solid background) are
    considered.
    """
    # Run analysis in dry-run mode to collect data
    results = detector.run(dry_run=True, limit=scan_limit)

    # Keep only scenes where a solid background was actually detected
    with_bg = [r for r in results if r["confidence"] > 0]

    if not with_bg:
        print("\nSurvey: No scenes with solid backgrounds detected.")
        return []

    # Sort by luminance of the optimized chroma color
    for r in with_bg:
        c = r.get("optimized_color_rgb", r["dominant_color_rgb"])
        r["_luminance"] = color_luminance(*c)

    with_bg.sort(key=lambda r: r["_luminance"])

    # Pick evenly spaced entries
    if len(with_bg) <= pick_count:
        picks = with_bg
    else:
        step = (len(with_bg) - 1) / (pick_count - 1)
        indices = [int(round(i * step)) for i in range(pick_count)]
        picks = [with_bg[i] for i in indices]

    # Display results
    print(f"\n{'='*72}")
    print(f"SURVEY: {pick_count} test scenes selected (dark -> light)")
    print(f"{'='*72}")
    print(f"{'#':>3}  {'Lum':>5}  {'Confidence':>10}  {'Chroma RGB':>14}  "
          f"{'Tolerance':>9}  {'Title'}")
    print(f"{'-'*3}  {'-'*5}  {'-'*10}  {'-'*14}  {'-'*9}  {'-'*30}")

    for i, r in enumerate(picks, 1):
        c = r.get("optimized_color_rgb", r["dominant_color_rgb"])
        tol = r.get("recommended_tolerance", 0)
        lum = r["_luminance"]
        title = r["title"][:40] if r["title"] else f"Scene {r['scene_id']}"
        print(f"{i:3d}  {lum:5.1f}  {r['confidence']:9.1f}%  "
              f"({c[0]:3d},{c[1]:3d},{c[2]:3d})  "
              f"{tol:9.3f}  {title}")

    # Clean up internal key
    for r in picks:
        r.pop("_luminance", None)

    return picks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Detect VR scenes with solid-color backgrounds for passthrough"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze only, don't tag scenes in Stash")
    parser.add_argument("--threshold", type=float, default=70,
                        help="Confidence threshold for tagging (0-100, default 70)")
    parser.add_argument("--scene-id", type=str, default=None,
                        help="Test a single scene by Stash scene ID")
    parser.add_argument("--num-frames", type=int, default=5,
                        help="Number of sample frames to extract (default 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of scenes to process")
    parser.add_argument("--survey", type=int, default=None, metavar="N",
                        help="Survey mode: scan scenes and pick N with diverse "
                             "background colors (dark to light)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-region analysis stats")
    parser.add_argument("--tag", default="export_deovr",
                        help="Source tag to filter scenes (default: export_deovr)")
    parser.add_argument("--stash-url", default="http://localhost:9999/graphql",
                        help="Stash GraphQL endpoint")
    parser.add_argument("--stash-key", default="",
                        help="Stash API key")

    args = parser.parse_args()

    # Initialize Stash connection
    stash = StashIntegration(StashConfig(
        url=args.stash_url,
        api_key=args.stash_key,
    ))

    # Test connection
    try:
        stash._query("{ systemStatus { databaseSchema } }")
        print("Connected to Stash")
    except Exception as e:
        print(f"Error: Cannot connect to Stash at {args.stash_url}: {e}")
        sys.exit(1)

    detector = PassthroughDetector(
        stash=stash,
        threshold=args.threshold,
        num_frames=args.num_frames,
        verbose=args.verbose,
        source_tag=args.tag,
    )

    # Survey mode
    if args.survey:
        results = run_survey(detector, pick_count=args.survey,
                             scan_limit=args.limit)
        if results:
            base = Path(__file__).parent.parent
            report_path = str(base / "output" / "passthrough_survey.json")
            save_report(results, report_path)
        return

    # Normal detection mode
    results = detector.run(
        dry_run=args.dry_run,
        scene_id=args.scene_id,
        limit=args.limit,
    )

    # Save report
    if results:
        base = Path(__file__).parent.parent
        report_path = str(base / "output" / "passthrough_scenes.json")
        save_report(results, report_path)


if __name__ == "__main__":
    main()
