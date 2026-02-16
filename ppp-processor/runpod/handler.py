"""RunPod Serverless Handler for VR Video Matting.

Self-contained matting handler adapted from src/processor.py.
No Settings/Celery/TRT dependency chain — uses FFmpeg streaming pipes,
boto3 for S3, and module-level model loading for warm reuse.

Job input schema:
    {
        "source_key": "ppp-matte/input/scene_12345.mp4",
        "output_key": "ppp-matte/output/scene_12345_matted.mp4",
        "s3_bucket": "ppp-processor-media",
        "vr_type": "sbs",                # "sbs", "tb", or "2d"
        "model_type": "resnet50",         # "resnet50" or "mobilenetv3"
        "downsample_ratio": 0.4,
        "green_color": [0, 177, 64],
        "vr_encode_bitrate": "50M",
        "encode_bitrate": "15M",
        "refine_alpha": false,
        "despill": false,                 # green spill removal
        "despill_strength": 0.8,          # 0-1 intensity
        "alpha_sharpness": "fine",        # "fine" or "soft"
        "output_mode": "green_screen",    # "green_screen", "alpha_xalpha", "prores4444"
        "alpha_output_key": "",           # auto-generated if empty (_XALPHA suffix)
        "upscale": false,                 # Real-ESRGAN 4x upscaling
        "upscale_model": "RealESRGAN_x4plus",
        "upscale_scale": 4,
        "upscale_tile_size": 512,
        "refine_threshold_low": 0.01,     # alpha below this → 0
        "refine_threshold_high": 0.99,    # alpha above this → 1
        "refine_laplacian_strength": 0.3, # edge sharpening intensity
        "refine_morph_kernel": 5,         # morph open/close kernel (px)
        "despill_dilation_kernel": 7,     # edge mask dilation kernel (px)
        "despill_dilation_iters": 2       # edge mask dilation passes
    }
"""

from __future__ import annotations

import json
import logging
import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import runpod
import torch
import torchvision.transforms as transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("runpod.matte")

# ---------------------------------------------------------------------------
# Module-level state (persists across warm invocations)
# ---------------------------------------------------------------------------
_model = None
_device = None
_use_fp16 = False
_s3_client = None
_ort_engine = None  # ONNX Runtime engine (shared across warm invocations)

# VR filename detection patterns (subset of analyzer.py)
VR_PATTERNS = {
    r"_180": {},
    r"_360": {},
    r"_sbs|_lr|_3dh|SBS": {"stereo_mode": "sbs"},
    r"_tb|_ou|_3dv|TB": {"stereo_mode": "tb"},
    r"VRBangers|WankzVR|VRHush|SLR|VirtualReal|POVR|RealJam|VRConk": {
        "stereo_mode": "sbs",
    },
}

VR_FILENAME_PATTERNS = [
    "_180", "_360", "_vr", "_VR", "_sbs", "_SBS",
    "_LR", "_TB", "_3dh", "_6k", "_6K", "_7k",
    "_8k", "_8K", "_fisheye", "POVR", "VRBangers",
    "WankzVR", "SLR", "VirtualReal",
]

# CUVID decoder mapping: codec_name → cuvid decoder name
CUVID_DECODERS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid",
}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=os.environ.get("S3_REGION", "us-east-2"),
        )
    return _s3_client


def s3_download(bucket: str, key: str, local_path: Path):
    """Download a file from S3."""
    logger.info("Downloading s3://%s/%s → %s", bucket, key, local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    get_s3_client().download_file(bucket, key, str(local_path))
    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info("Downloaded %.1f MB", size_mb)


def s3_upload(local_path: Path, bucket: str, key: str):
    """Upload a file to S3."""
    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info("Uploading %s (%.1f MB) → s3://%s/%s", local_path.name, size_mb, bucket, key)
    get_s3_client().upload_file(str(local_path), bucket, key)
    logger.info("Upload complete")


# ---------------------------------------------------------------------------
# Model loading (cold start once, reuse across warm invocations)
# ---------------------------------------------------------------------------
def load_model(model_type: str = "resnet50"):
    """Load RVM model onto GPU with FP16 and torch.compile."""
    global _model, _device, _use_fp16

    if _model is not None:
        return

    # RVM model code is baked into the Docker image at /app/RobustVideoMatting/
    rvm_path = Path("/app/RobustVideoMatting")
    if rvm_path.exists() and str(rvm_path) not in sys.path:
        sys.path.insert(0, str(rvm_path))

    from model import MattingNetwork  # type: ignore[import-untyped]

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading RVM %s on device: %s", model_type, _device)

    model_path = Path(f"/app/models/rvm_{model_type}.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    _model = MattingNetwork(model_type).eval()
    _model.load_state_dict(torch.load(model_path, map_location=_device, weights_only=True))
    _model = _model.to(_device)

    # FP16 on CUDA
    _use_fp16 = _device.type == "cuda"
    if _use_fp16:
        _model = _model.half()
        logger.info("FP16 enabled")

    # channels_last (NHWC) memory format for tensor core acceleration
    if _device.type == "cuda":
        _model = _model.to(memory_format=torch.channels_last)
        logger.info("channels_last memory format enabled")

    # torch.compile — skip on serverless cold starts, but allow on batch pods
    # RUNPOD_POD_ID="batch" is set by batch_runner; "serverless" means skip
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    skip_compile = pod_id and pod_id != "batch"
    if _device.type == "cuda" and platform.system() != "Windows" and not skip_compile:
        # Suppress errors so dynamo falls back to eager on failure
        _dynamo = __import__("torch._dynamo", fromlist=["config"])
        _dynamo.config.suppress_errors = True
        try:
            _model = torch.compile(_model, mode="reduce-overhead")
            logger.info("torch.compile() enabled (reduce-overhead)")
        except Exception:
            try:
                _model = torch.compile(_model, backend="eager")
                logger.info("torch.compile() enabled (eager)")
            except Exception as e:
                logger.info("torch.compile() unavailable: %s", e)
    else:
        logger.info("torch.compile() skipped (serverless environment)")

    logger.info("Model loaded successfully")

    # Try to initialize ORT engine for accelerated inference (with timeout)
    _init_ort_engine_safe(model_type)


def _init_ort_engine_safe(model_type: str = "resnet50"):
    """Initialize ORT engine with a timeout guard to prevent startup hangs."""
    result = [None]
    error = [None]

    def _init():
        try:
            _init_ort_engine(model_type)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_init, daemon=True)
    t.start()
    t.join(timeout=120)  # 2 minute max for ORT init
    if t.is_alive():
        logger.warning("ORT engine init timed out after 120s, using PyTorch")
        global _ort_engine
        _ort_engine = None
    elif error[0]:
        logger.warning("ORT engine init error: %s", error[0])


def _init_ort_engine(model_type: str = "resnet50"):
    """Initialize ORT engine for accelerated inference (optional)."""
    global _ort_engine, _model, _device, _use_fp16

    if _ort_engine is not None:
        return

    try:
        # Import from shared module (copied into Docker image)
        try:
            from ort_engine import RVMOrtEngine
        except ImportError:
            from src.ort_engine import RVMOrtEngine

        _ort_engine = RVMOrtEngine()
        cache_dir = Path("/app/trt_engines")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Height/width used only for ONNX export dummy input; actual inference
        # uses dynamic axes so any resolution works at runtime.
        success = _ort_engine.prepare(
            1080, 1920, _model, model_type, _device,
            downsample_ratio=0.4,
            cache_dir=cache_dir,
        )
        if success:
            logger.info("ORT engine initialized")
            # Restore model to FP16 after ONNX export
            if _use_fp16 and _model is not None:
                _model = _model.half()
                _model = _model.to(memory_format=torch.channels_last)
        else:
            _ort_engine = None
            logger.info("ORT engine init failed, using PyTorch")
    except ImportError:
        logger.info("onnxruntime not available, using PyTorch inference")
        _ort_engine = None
    except Exception as e:
        logger.warning("ORT engine init error: %s", e)
        _ort_engine = None


# ---------------------------------------------------------------------------
# Video probing (from processor.py:672)
# ---------------------------------------------------------------------------
def probe_video(video_path: Path) -> dict:
    """Probe video for dimensions, FPS, frame count, and audio presence."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_ffmpeg_env())
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

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
        "codec_name": video_stream.get("codec_name", ""),
        "fps": fps,
        "fps_str": fps_str,
        "total_frames": total_frames,
        "has_audio": has_audio,
        "duration": duration,
    }


# ---------------------------------------------------------------------------
# VR type detection (simplified from analyzer.py)
# ---------------------------------------------------------------------------
def detect_vr_type(video_path: Path, width: int, height: int) -> str:
    """Detect VR type from filename patterns and aspect ratio.

    Returns "sbs", "tb", or "2d".
    """
    filename = video_path.name.lower()

    # Filename pattern matching
    is_vr = False
    vr_type: Optional[str] = None

    for pattern in VR_FILENAME_PATTERNS:
        if pattern.lower() in filename:
            is_vr = True
            break

    for pattern, attrs in VR_PATTERNS.items():
        if re.search(pattern, filename, re.IGNORECASE):
            is_vr = True
            if "stereo_mode" in attrs:
                vr_type = attrs["stereo_mode"]
            break

    # Aspect ratio heuristic: width >= 2 * height → SBS
    if not is_vr and height > 0 and width >= 2 * height:
        is_vr = True

    if is_vr and vr_type is None:
        if "_tb" in filename or "_ou" in filename or "_3dv" in filename:
            vr_type = "tb"
        else:
            vr_type = "sbs"

    return vr_type if is_vr else "2d"


def compute_target_resolution(
    src_w: int, src_h: int, vr_type: str,
) -> Optional[tuple]:
    """Compute upscale target resolution based on source dimensions.

    2D rules (by height):
      ≤1080p → min(2x height, 1440p), scale width proportionally
      >1080p → no upscale

    VR SBS rules (by full frame width):
      <6144  → 6144 wide (6K)
      <8192  → 8192 wide (8K)
      ≥8192  → no upscale

    Returns (target_w, target_h) or None if no upscale needed.
    """
    if vr_type == "sbs":
        if src_w < 6144:
            target_w = 6144
        elif src_w < 8192:
            target_w = 8192
        else:
            return None
        # Scale height proportionally, ensure even
        target_h = int(src_h * target_w / src_w)
        target_h = target_h + (target_h % 2)
        target_w = target_w + (target_w % 2)
        return (target_w, target_h)

    elif vr_type == "2d" or vr_type == "tb":
        if src_h > 1080:
            return None
        target_h = min(src_h * 2, 1440)
        # Ensure even
        target_h = target_h + (target_h % 2)
        target_w = int(src_w * target_h / src_h)
        target_w = target_w + (target_w % 2)
        return (target_w, target_h)

    return None


def rescale_video(
    input_path: Path,
    output_path: Path,
    target_w: int,
    target_h: int,
    bitrate: str = "15M",
) -> bool:
    """Rescale a video to target resolution using FFmpeg lanczos."""
    temp_path = input_path.with_suffix(".tmp.mp4")
    input_path.rename(temp_path)

    cmd = build_encode_cmd(
        ["-i", str(temp_path)],
        output_path,
        bitrate,
        extra_input_args=[
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            "-map", "0:v",
        ],
        extra_output_args=["-map", "0:a?", "-c:a", "copy"],
    )

    logger.info("Rescaling %s → %dx%d (lanczos)", input_path.name, target_w, target_h)
    result = subprocess.run(cmd, capture_output=True, text=True, env=_ffmpeg_env())
    temp_path.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error("Rescale failed: %s", result.stderr[-500:])
        return False
    return True


# ---------------------------------------------------------------------------
# NVENC probe (from processor.py:534)
# ---------------------------------------------------------------------------
_nvenc_available: Optional[bool] = None


def despill_green(fgr, pha, strength=0.8, dilation_kernel=7, dilation_iters=2):
    """Suppress green spill in edge regions where 0 < alpha < 1.

    Args:
        fgr: [1, 3, H, W] float tensor — foreground RGB
        pha: [1, 1, H, W] float tensor — alpha matte
        strength: float 0-1, despill intensity
        dilation_kernel: edge mask dilation kernel size (px)
        dilation_iters: number of dilation passes

    Returns:
        Cleaned fgr tensor with green spill suppressed in transition zones.
    """
    F = torch.nn.functional

    # Edge mask: transition zone pixels
    edge_mask = ((pha > 0.01) & (pha < 0.99)).float()
    # Dilate edge mask to catch adjacent spill
    pad_size = dilation_kernel // 2
    for _ in range(dilation_iters):
        edge_mask = F.max_pool2d(
            F.pad(edge_mask, [pad_size] * 4, mode="reflect"),
            dilation_kernel, stride=1,
        )

    r, g, b = fgr[:, 0:1], fgr[:, 1:2], fgr[:, 2:3]
    # Classic despill: clamp green to max(red, blue)
    g_clamped = torch.min(g, torch.max(r, b))
    # Blend: only affect edge regions
    g_new = torch.lerp(g, g_clamped, edge_mask * strength)
    fgr = torch.cat([r, g_new, b], dim=1).clamp_(0, 1)
    return fgr


def refine_alpha(
    pha, src, sharpness="fine",
    threshold_low=0.01, threshold_high=0.99,
    laplacian_strength=0.3, morph_kernel_size=5,
):
    """Refine raw alpha matte with guided filter, thresholding, and morphological ops.

    Args:
        pha: [1, 1, H, W] float tensor on GPU — raw alpha matte
        src: [1, 3, H, W] float tensor on GPU — source frame (guide for edge-aware filter)
        sharpness: "fine" for crisp edges (multi-scale + Laplacian), "soft" for legacy behavior
        threshold_low: alpha below this → 0 (default 0.01)
        threshold_high: alpha above this → 1 (default 0.99)
        laplacian_strength: edge sharpening intensity (default 0.3)
        morph_kernel_size: kernel size for morphological open/close (default 5)

    Returns:
        Refined pha tensor, same shape/dtype/device.
    """
    F = torch.nn.functional
    orig_dtype = pha.dtype

    pha = pha.float()
    src_f = src.float()

    # Luminance guide
    I = src_f[:, 0:1] * 0.299 + src_f[:, 1:2] * 0.587 + src_f[:, 2:3] * 0.114

    if sharpness == "soft":
        # Legacy single-scale guided filter
        r = 8
        eps = 1e-4
        k = 2 * r + 1

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

        # Legacy thresholds
        pha = pha.clamp(0.0, 1.0)
        pha[pha < 0.05] = 0.0
        pha[pha > 0.95] = 1.0

        # Legacy morphological close (3px)
        pha = F.max_pool2d(F.pad(pha, [1, 1, 1, 1], mode="reflect"), 3, stride=1)
        pha = -F.max_pool2d(F.pad(-pha, [1, 1, 1, 1], mode="reflect"), 3, stride=1)

    else:
        # "fine" mode: multi-scale guided filter + Laplacian sharpening
        eps = 1e-4

        def _guided_filter(p, guide, radius):
            k = 2 * radius + 1
            pad = radius
            def bf(x):
                return F.avg_pool2d(F.pad(x, [pad]*4, mode="reflect"), k, stride=1)
            mean_g = bf(guide)
            mean_p = bf(p)
            corr_gp = bf(guide * p)
            var_g = bf(guide * guide) - mean_g * mean_g
            cov_gp = corr_gp - mean_g * mean_p
            a = cov_gp / (var_g + eps)
            b = mean_p - a * mean_g
            return bf(a) * guide + bf(b)

        # Fine detail pass (r=4) and bulk pass (r=12)
        pha_fine = _guided_filter(pha, I, radius=4)
        pha_bulk = _guided_filter(pha, I, radius=12)

        # Blend by local variance — high variance regions use fine filter
        local_mean = F.avg_pool2d(F.pad(pha, [4]*4, mode="reflect"), 9, stride=1)
        local_var = F.avg_pool2d(F.pad((pha - local_mean)**2, [4]*4, mode="reflect"), 9, stride=1)
        # Normalize variance to [0, 1] blend weight
        blend_w = (local_var * 50.0).clamp(0.0, 1.0)
        pha = torch.lerp(pha_bulk, pha_fine, blend_w)

        # Soft sigmoid thresholds
        pha = pha.clamp(0.0, 1.0)
        lo, hi = threshold_low, threshold_high
        mask_lo = (pha < lo).float()
        mask_hi = (pha > hi).float()
        mask_mid = 1.0 - mask_lo - mask_hi
        # Rescale midrange to [0, 1]
        pha_mid = ((pha - lo) / (hi - lo)).clamp(0.0, 1.0)
        pha = mask_hi + mask_mid * pha_mid

        # Laplacian edge sharpening (subtle)
        lap_kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            dtype=torch.float32, device=pha.device,
        ).reshape(1, 1, 3, 3)
        pha_padded = F.pad(pha, [1, 1, 1, 1], mode="reflect")
        laplacian = F.conv2d(pha_padded, lap_kernel)
        pha = (pha + laplacian_strength * laplacian).clamp(0.0, 1.0)

        # Morphological open (remove small noise): erode then dilate
        morph_pad = morph_kernel_size // 2
        pha = -F.max_pool2d(F.pad(-pha, [morph_pad] * 4, mode="reflect"), morph_kernel_size, stride=1)
        pha = F.max_pool2d(F.pad(pha, [morph_pad] * 4, mode="reflect"), morph_kernel_size, stride=1)

        # Morphological close (fill small holes): dilate then erode
        pha = F.max_pool2d(F.pad(pha, [morph_pad] * 4, mode="reflect"), morph_kernel_size, stride=1)
        pha = -F.max_pool2d(F.pad(-pha, [morph_pad] * 4, mode="reflect"), morph_kernel_size, stride=1)

    return pha.to(orig_dtype)


def _ffmpeg_env():
    """Return env dict with LD_LIBRARY_PATH set for NVIDIA libraries."""
    env = os.environ.copy()
    nvidia_paths = (
        "/usr/local/nvidia/lib64:"    # NVIDIA Container Toolkit driver mount
        "/usr/local/nvidia/lib:"
        "/usr/local/cuda/lib64:"
        "/usr/local/lib:"             # FFmpeg shared libraries
        "/usr/lib/x86_64-linux-gnu"
    )
    env["LD_LIBRARY_PATH"] = nvidia_paths + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def probe_nvenc() -> bool:
    """Test NVENC availability. Cached after first call."""
    global _nvenc_available
    if _nvenc_available is not None:
        return _nvenc_available

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-y",
                "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                "-c:v", "hevc_nvenc", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=10,
            env=_ffmpeg_env(),
        )
        _nvenc_available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        _nvenc_available = False

    logger.info("NVENC available: %s", _nvenc_available)
    return _nvenc_available


# ---------------------------------------------------------------------------
# FFmpeg encode command builder (from processor.py:601)
# ---------------------------------------------------------------------------
def build_encode_cmd(
    input_args: list,
    output_path: Path,
    bitrate: str,
    extra_input_args: Optional[list] = None,
    extra_output_args: Optional[list] = None,
) -> list:
    """Build FFmpeg HEVC encode command (NVENC or libx265 fallback)."""
    if probe_nvenc():
        encode_params = [
            "-c:v", "hevc_nvenc",
            "-preset", "p5",
            "-tune", "hq",
            "-rc", "constqp",
            "-qp", "20",
            "-profile:v", "main10",
            "-rc-lookahead", "32",
            "-pix_fmt", "p010le",
            "-tag:v", "hvc1",
            "-spatial-aq", "1",
            "-temporal-aq", "1",
        ]
    else:
        bufsize_str = str(int(bitrate.rstrip("M")) * 2) + "M"
        encode_params = [
            "-c:v", "libx265",
            "-preset", "medium",
            "-crf", "18",
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


# ---------------------------------------------------------------------------
# Core streaming matting (from processor.py:1207)
# ---------------------------------------------------------------------------
def process_video_streaming(
    input_path: Path,
    output_path: Path,
    probe: dict,
    green_color: list,
    downsample_ratio: float = 0.4,
    bitrate: str = "15M",
    crop_region: Optional[tuple] = None,
    use_refine_alpha: bool = False,
    use_despill: bool = False,
    despill_strength: float = 0.8,
    alpha_sharpness: str = "fine",
    output_mode: str = "green_screen",
    alpha_output_path: Optional[Path] = None,
    upscaler_instance=None,
    fast_decode: bool = True,
    refine_threshold_low: float = 0.01,
    refine_threshold_high: float = 0.99,
    refine_laplacian_strength: float = 0.3,
    refine_morph_kernel: int = 5,
    despill_dilation_kernel: int = 7,
    despill_dilation_iters: int = 2,
) -> bool:
    """Streaming matting: FFmpeg decode pipe → GPU inference → FFmpeg encode pipe.

    3-stage async pipeline with pre-allocated pinned memory.

    When fast_decode=True (default), FFmpeg scales frames to the model's
    processing resolution BEFORE piping — dramatically reducing CPU→GPU
    data transfer for high-res VR content. The model runs at
    downsample_ratio=1.0 on the pre-scaled frames. Encode resolution
    matches the pre-scaled size (model output). For VR, this yields
    identical alpha quality since the model only processes at the
    downsampled resolution anyway.
    """
    global _model, _device, _use_fp16

    # Ensure model is loaded (may have been reset between L/R eyes)
    if _model is None:
        load_model()

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

    # --- Fast decode: scale in FFmpeg BEFORE piping to reduce data volume ---
    # The model internally downsamples to downsample_ratio anyway, so pre-scaling
    # in FFmpeg gives identical quality with vastly less CPU→GPU data transfer.
    actual_downsample_ratio = downsample_ratio  # ratio passed to the model
    original_proc_w, original_proc_h = proc_w, proc_h
    if fast_decode and downsample_ratio < 1.0:
        # Scale to model resolution in FFmpeg
        scaled_w = int(proc_w * downsample_ratio)
        scaled_h = int(proc_h * downsample_ratio)
        # Ensure even dimensions for HEVC encoding
        scaled_w = scaled_w + (scaled_w % 2)
        scaled_h = scaled_h + (scaled_h % 2)
        scale_filter = f"scale={scaled_w}:{scaled_h}"
        if vf_decode:
            vf_decode = f"{vf_decode},{scale_filter}"
        else:
            vf_decode = scale_filter
        proc_w, proc_h = scaled_w, scaled_h
        actual_downsample_ratio = 1.0  # already at model scale
        logger.info(
            "Fast decode: %dx%d → %dx%d (%.0f%% less pipe data)",
            original_proc_w, original_proc_h, proc_w, proc_h,
            (1 - (proc_w * proc_h) / (original_proc_w * original_proc_h)) * 100,
        )

    frame_bytes = proc_w * proc_h * 3  # RGB24
    pipe_bufsize = frame_bytes * 8

    # Determine encode dimensions (may differ from proc if upscaling)
    F = torch.nn.functional
    upscale_scale = getattr(upscaler_instance, "scale", 1) if upscaler_instance else 1
    encode_w = proc_w * upscale_scale
    encode_h = proc_h * upscale_scale

    # Pre-allocate reusable tensors
    dtype = torch.float16 if _use_fp16 else torch.float32
    pin_buffer = torch.empty((proc_h, proc_w, 3), dtype=torch.uint8).pin_memory()
    gpu_input = torch.empty((1, 3, proc_h, proc_w), dtype=dtype, device=_device)

    green_tensor = torch.tensor(
        [[[green_color[0] / 255.0]], [[green_color[1] / 255.0]], [[green_color[2] / 255.0]]],
        dtype=dtype, device=_device,
    )

    out_gpu = torch.empty((1, 3, encode_h, encode_w), dtype=dtype, device=_device)
    out_pin = torch.empty((encode_h, encode_w, 3), dtype=torch.uint8).pin_memory()

    transfer_stream = torch.cuda.Stream() if _device.type == "cuda" else None

    # --- FFmpeg decode pipe (multiple strategies, fastest first) ---
    # Strategy 1: CUVID decoder with -crop + -resize (all on GPU decoder hardware)
    # Strategy 2: NVDEC hwaccel + scale_npp + CPU crop (NPP lib required)
    # Strategy 3: NVDEC hwaccel + hwdownload + CPU filters
    # Strategy 4: CPU decode (always available)
    decode_strategies = []

    codec_name = probe.get("codec_name", "")
    cuvid_decoder = CUVID_DECODERS.get(codec_name)

    if _device.type == "cuda":
        # Strategy 1: CUVID decoder — crop and scale on GPU decoder chip
        # -crop T:B:L:R = pixels to REMOVE from each edge (not x,y,w,h)
        # -resize WxH = scale output to target size
        # Both execute on NVDEC hardware — zero CPU involvement
        if cuvid_decoder:
            cuvid_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-c:v", cuvid_decoder,
            ]
            if crop_region:
                cx, cy, cw, ch = crop_region
                crop_top = cy
                crop_bottom = src_h - (cy + ch)
                crop_left = cx
                crop_right = src_w - (cx + cw)
                cuvid_cmd.extend([
                    "-crop", f"{crop_top}:{crop_bottom}:{crop_left}:{crop_right}",
                ])
            if fast_decode and downsample_ratio < 1.0:
                cuvid_cmd.extend(["-resize", f"{proc_w}x{proc_h}"])
            cuvid_cmd.extend([
                "-i", str(input_path),
                "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
            ])
            decode_strategies.append({
                "name": f"CUVID ({cuvid_decoder})",
                "cmd": cuvid_cmd,
            })

        # Strategy 2: NVDEC + scale_npp (GPU) + CPU crop
        if crop_region and fast_decode and downsample_ratio < 1.0:
            cx, cy, cw, ch = crop_region
            full_scaled_w = int(src_w * downsample_ratio)
            full_scaled_h = int(src_h * downsample_ratio)
            full_scaled_w += full_scaled_w % 2
            full_scaled_h += full_scaled_h % 2
            scaled_cx = int(cx * downsample_ratio)
            scaled_cy = int(cy * downsample_ratio)
            npp_vf = (
                f"scale_npp=w={full_scaled_w}:h={full_scaled_h},"
                f"hwdownload,format=nv12,"
                f"crop=w={proc_w}:h={proc_h}:x={scaled_cx}:y={scaled_cy}"
            )
            decode_strategies.append({
                "name": "NVDEC+NPP scale+CPU crop",
                "cmd": [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                    "-i", str(input_path),
                    "-vf", npp_vf,
                    "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
                ],
            })

        # Strategy 3: NVDEC + hwdownload + CPU crop/scale
        if vf_decode:
            cpu_vf = f"hwdownload,format=nv12,{vf_decode}"
        else:
            cpu_vf = "hwdownload,format=nv12"
        decode_strategies.append({
            "name": "NVDEC+CPU filters",
            "cmd": [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                "-i", str(input_path),
                "-vf", cpu_vf,
                "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
            ],
        })

    # Strategy 4: Pure CPU decode (always available)
    cpu_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]
    if vf_decode:
        cpu_cmd.extend(["-vf", vf_decode])
    cpu_cmd.extend(["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    decode_strategies.append({"name": "CPU decode", "cmd": cpu_cmd})

    # --- FFmpeg encode pipe (RGB) ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipe_input_args = [
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{encode_w}x{encode_h}", "-r", fps_str,
        "-i", "pipe:0",
    ]
    audio_args = None
    if probe["has_audio"]:
        audio_args = ["-i", str(input_path), "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
    encode_cmd = build_encode_cmd(
        pipe_input_args, output_path, bitrate,
        extra_input_args=audio_args,
    )

    # --- FFmpeg alpha encode pipe (for alpha_xalpha mode) ---
    alpha_encoder = None
    alpha_encode_q = None
    alpha_encode_error = [None]
    alpha_encode_cmd = None
    alpha_w, alpha_h = 0, 0
    alpha_out_pin = None
    if output_mode == "alpha_xalpha" and alpha_output_path:
        alpha_output_path.parent.mkdir(parents=True, exist_ok=True)
        alpha_h = 480
        alpha_w = int(480 * encode_w / encode_h)
        alpha_w = alpha_w + (alpha_w % 2)  # Ensure even for HEVC
        alpha_h = alpha_h + (alpha_h % 2)
        alpha_pipe_args = [
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{alpha_w}x{alpha_h}", "-r", fps_str,
            "-i", "pipe:0",
        ]
        alpha_encode_cmd = build_encode_cmd(
            alpha_pipe_args, alpha_output_path, "2M",
        )
        alpha_out_pin = torch.empty((alpha_h, alpha_w, 3), dtype=torch.uint8).pin_memory()
        alpha_encode_q = queue.Queue(maxsize=4)

    def _read_with_timeout(pipe, nbytes, timeout=10):
        result_data = [None]
        def _read():
            result_data[0] = pipe.read(nbytes)
        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            return None
        return result_data[0]

    # Try decode strategies in order (fastest first)
    decoder = None
    test_raw = None
    for strat in decode_strategies:
        try:
            logger.info("Trying decode: %s", strat["name"])
            decoder = subprocess.Popen(
                strat["cmd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=pipe_bufsize, env=_ffmpeg_env(),
            )
            test_raw = _read_with_timeout(decoder.stdout, frame_bytes, timeout=15)
            if test_raw and len(test_raw) >= frame_bytes:
                logger.info("Decode strategy active: %s", strat["name"])
                break
            else:
                stderr_out = ""
                try:
                    decoder.kill()
                    decoder.wait(timeout=5)
                    stderr_out = decoder.stderr.read().decode(errors="replace")[:200]
                except Exception:
                    pass
                logger.info("Decode strategy failed: %s — %s", strat["name"], stderr_out or "no output")
                decoder = None
                test_raw = None
        except Exception as e:
            logger.info("Decode strategy error: %s — %s", strat["name"], e)
            if decoder:
                try:
                    decoder.kill()
                    decoder.wait(timeout=5)
                except Exception:
                    pass
            decoder = None
            test_raw = None

    if decoder is None or test_raw is None:
        logger.error("All decode strategies failed")
        return False

    # --- Async 3-stage pipeline ---
    SENTINEL = None
    decode_q: queue.Queue = queue.Queue(maxsize=4)
    encode_q: queue.Queue = queue.Queue(maxsize=4)
    decode_error = [None]
    encode_error = [None]

    encoder = subprocess.Popen(
        encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=pipe_bufsize, env=_ffmpeg_env(),
    )

    # Start alpha encoder if needed
    if alpha_encode_cmd and alpha_encode_q is not None:
        alpha_encoder = subprocess.Popen(
            alpha_encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=alpha_w * alpha_h * 3 * 8, env=_ffmpeg_env(),
        )

    def decode_thread():
        try:
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
        try:
            while True:
                data = encode_q.get()
                if data is SENTINEL:
                    break
                encoder.stdin.write(data)
        except Exception as e:
            encode_error[0] = e

    def alpha_encode_thread():
        try:
            while True:
                data = alpha_encode_q.get()
                if data is SENTINEL:
                    break
                alpha_encoder.stdin.write(data)
        except Exception as e:
            alpha_encode_error[0] = e

    rec = [None] * 4

    try:
        t_decode = threading.Thread(target=decode_thread, daemon=True)
        t_encode = threading.Thread(target=encode_thread, daemon=True)
        t_decode.start()
        t_encode.start()

        t_alpha_encode = None
        if alpha_encoder and alpha_encode_q is not None:
            t_alpha_encode = threading.Thread(target=alpha_encode_thread, daemon=True)
            t_alpha_encode.start()

        frame_idx = 0
        t_start = time.time()
        while True:
            raw = decode_q.get()
            if raw is SENTINEL:
                break

            # Fast CPU→GPU transfer via pinned memory
            frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, 3)
            pin_buffer.copy_(torch.from_numpy(frame_np.copy()))

            if transfer_stream:
                with torch.cuda.stream(transfer_stream):
                    gpu_input.copy_(
                        pin_buffer.permute(2, 0, 1).unsqueeze(0).to(
                            dtype=dtype, device=_device, non_blocking=True
                        )
                    )
                    gpu_input.div_(255.0)
                transfer_stream.synchronize()
            else:
                gpu_input.copy_(
                    pin_buffer.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype, device=_device)
                )
                gpu_input.div_(255.0)

            # channels_last for tensor core acceleration
            gpu_input = gpu_input.contiguous(memory_format=torch.channels_last)

            # GPU inference (ORT → PyTorch fallback)
            if _ort_engine is not None:
                try:
                    fgr, pha = _ort_engine.infer(gpu_input)
                except Exception as e:
                    logger.warning("ORT inference failed, falling back to PyTorch: %s", e)
                    with torch.no_grad():
                        fgr, pha, *rec = _model(gpu_input, *rec, actual_downsample_ratio)
            else:
                with torch.no_grad():
                    fgr, pha, *rec = _model(gpu_input, *rec, actual_downsample_ratio)

            # --- Post-processing pipeline ---
            # 1. Green spill removal
            if use_despill:
                fgr = despill_green(
                    fgr, pha, despill_strength,
                    dilation_kernel=despill_dilation_kernel,
                    dilation_iters=despill_dilation_iters,
                )

            # 2. Alpha refinement
            if use_refine_alpha:
                pha = refine_alpha(
                    pha, gpu_input, sharpness=alpha_sharpness,
                    threshold_low=refine_threshold_low,
                    threshold_high=refine_threshold_high,
                    laplacian_strength=refine_laplacian_strength,
                    morph_kernel_size=refine_morph_kernel,
                )

            # 3. Optional upscaling
            if upscaler_instance is not None:
                fgr, pha = upscaler_instance.upscale_frame(fgr, pha)

            # 4. Compositing based on output mode
            if output_mode == "alpha_xalpha":
                # Premultiplied alpha (black background where transparent)
                rgb_out = (fgr * pha).clamp_(0, 1)
                frame_out = rgb_out[0].permute(1, 2, 0).mul(255).to(torch.uint8)
                out_pin.copy_(frame_out.cpu())
                encode_q.put(out_pin.numpy().tobytes())

                # Alpha channel → 480p grayscale as RGB
                if alpha_encode_q is not None:
                    alpha_small = F.interpolate(
                        pha, size=(alpha_h, alpha_w), mode="bilinear", align_corners=False,
                    )
                    alpha_rgb = alpha_small.expand(-1, 3, -1, -1)
                    alpha_frame = alpha_rgb[0].permute(1, 2, 0).mul(255).to(torch.uint8)
                    alpha_out_pin.copy_(alpha_frame.cpu())
                    alpha_encode_q.put(alpha_out_pin.numpy().tobytes())
            else:
                # green_screen mode (default): out = green + pha * (fgr - green)
                torch.addcmul(green_tensor.expand_as(fgr), pha, fgr - green_tensor, out=out_gpu)
                out_gpu.clamp_(0.0, 1.0)
                frame_out = out_gpu[0].permute(1, 2, 0).mul(255).to(torch.uint8)
                out_pin.copy_(frame_out.cpu())
                encode_q.put(out_pin.numpy().tobytes())

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                fps_rate = frame_idx / max(elapsed, 0.001)
                pct = min(frame_idx / max(total_frames, 1) * 100, 99)
                logger.info(
                    "  Frame %d/%d (%.1f%%) — %.1f fps",
                    frame_idx, total_frames, pct, fps_rate,
                )

        # Signal encode thread(s) to finish
        encode_q.put(SENTINEL)
        if alpha_encode_q is not None:
            alpha_encode_q.put(SENTINEL)

        t_encode.join(timeout=30)
        if t_alpha_encode:
            t_alpha_encode.join(timeout=30)
        t_decode.join(timeout=10)

        encoder.stdin.close()
        encoder.wait()
        decoder.wait()

        if alpha_encoder:
            alpha_encoder.stdin.close()
            alpha_encoder.wait()

        if decode_error[0]:
            logger.error("Decode thread error: %s", decode_error[0])
        if encode_error[0]:
            logger.error("Encode thread error: %s", encode_error[0])
        if alpha_encode_error[0]:
            logger.error("Alpha encode thread error: %s", alpha_encode_error[0])

        if encoder.returncode != 0:
            err = encoder.stderr.read().decode() if encoder.stderr else ""
            logger.error("Encode pipe failed: %s", err[-500:])
            return False

        if alpha_encoder and alpha_encoder.returncode != 0:
            err = alpha_encoder.stderr.read().decode() if alpha_encoder.stderr else ""
            logger.error("Alpha encode pipe failed: %s", err[-500:])
            return False

        elapsed = time.time() - t_start
        fps_rate = frame_idx / max(elapsed, 0.001)
        logger.info(
            "Streaming matting complete: %d frames in %.1fs (%.1f fps)",
            frame_idx, elapsed, fps_rate,
        )
        return True

    finally:
        if decoder and decoder.poll() is None:
            decoder.kill()
        if encoder and encoder.poll() is None:
            encoder.kill()
        if alpha_encoder and alpha_encoder.poll() is None:
            alpha_encoder.kill()
        if _device and _device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# VR SBS processing (from processor.py:1640)
# ---------------------------------------------------------------------------
def process_vr_sbs(
    input_path: Path,
    output_path: Path,
    probe: dict,
    green_color: list,
    downsample_ratio: float = 0.4,
    bitrate: str = "50M",
    use_refine_alpha: bool = False,
    use_despill: bool = False,
    despill_strength: float = 0.8,
    alpha_sharpness: str = "fine",
    output_mode: str = "green_screen",
    alpha_output_path: Optional[Path] = None,
    upscaler_instance=None,
    fast_decode: bool = True,
    target_resolution: Optional[tuple] = None,
    refine_threshold_low: float = 0.01,
    refine_threshold_high: float = 0.99,
    refine_laplacian_strength: float = 0.3,
    refine_morph_kernel: int = 5,
    despill_dilation_kernel: int = 7,
    despill_dilation_iters: int = 2,
) -> bool:
    """Process VR SBS video: crop L/R eyes, matte separately, hstack merge.

    If target_resolution=(w, h) is set, the merge step scales the hstacked
    output back up to that resolution (e.g. original VR size). Otherwise
    output is at whatever resolution the eyes were processed at.
    """
    global _model

    half_w = probe["width"] // 2
    h = probe["height"]

    work_dir = output_path.parent / f"_vr_work_{output_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    pipeline_kwargs = dict(
        use_refine_alpha=use_refine_alpha,
        use_despill=use_despill,
        despill_strength=despill_strength,
        alpha_sharpness=alpha_sharpness,
        output_mode=output_mode,
        upscaler_instance=upscaler_instance,
        fast_decode=fast_decode,
        refine_threshold_low=refine_threshold_low,
        refine_threshold_high=refine_threshold_high,
        refine_laplacian_strength=refine_laplacian_strength,
        refine_morph_kernel=refine_morph_kernel,
        despill_dilation_kernel=despill_dilation_kernel,
        despill_dilation_iters=despill_dilation_iters,
    )

    try:
        left_tmp = work_dir / "left_matted.mp4"
        right_tmp = work_dir / "right_matted.mp4"

        # Alpha work files for alpha_xalpha mode
        left_alpha_tmp = work_dir / "left_alpha.mp4" if output_mode == "alpha_xalpha" else None
        right_alpha_tmp = work_dir / "right_alpha.mp4" if output_mode == "alpha_xalpha" else None

        # Matte left eye
        logger.info("Processing left eye (%dx%d)", half_w, h)
        success = process_video_streaming(
            input_path, left_tmp, probe,
            green_color=green_color,
            downsample_ratio=downsample_ratio,
            bitrate=bitrate,
            crop_region=(0, 0, half_w, h),
            alpha_output_path=left_alpha_tmp,
            **pipeline_kwargs,
        )
        if not success:
            logger.error("Left eye matting failed")
            return False

        # Reset recurrent state for right eye (keep model loaded)
        _model_reset_rec()

        # Matte right eye
        logger.info("Processing right eye (%dx%d)", half_w, h)
        success = process_video_streaming(
            input_path, right_tmp, probe,
            green_color=green_color,
            downsample_ratio=downsample_ratio,
            bitrate=bitrate,
            crop_region=(half_w, 0, half_w, h),
            alpha_output_path=right_alpha_tmp,
            **pipeline_kwargs,
        )
        if not success:
            logger.error("Right eye matting failed")
            return False

        # Merge L/R into SBS (RGB), optionally scaling to target_resolution
        logger.info("Merging L/R into SBS output")
        success = merge_sbs_videos(
            left_tmp, right_tmp, output_path,
            audio_source=input_path,
            has_audio=probe["has_audio"],
            bitrate=bitrate,
            target_resolution=target_resolution,
        )
        if not success:
            return False

        # Merge L/R alpha into SBS (alpha_xalpha mode)
        if output_mode == "alpha_xalpha" and alpha_output_path and left_alpha_tmp and right_alpha_tmp:
            logger.info("Merging L/R alpha into SBS output")
            success = merge_sbs_videos(
                left_alpha_tmp, right_alpha_tmp, alpha_output_path,
                audio_source=input_path,
                has_audio=False,
                bitrate="2M",
            )
            if not success:
                logger.error("Alpha SBS merge failed")
                return False

        return True

    finally:
        # Clean up temp files
        import shutil
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def _model_reset_rec():
    """Reset model recurrent state between L/R eye processing.

    Resets ORT engine rec state if available, otherwise forces full model
    reload to clear RVM's internal recurrent buffers.
    """
    global _model, _ort_engine, _use_fp16
    if _ort_engine is not None:
        dtype_str = "float16" if _use_fp16 else "float32"
        _ort_engine.reset_recurrent_state(dtype_str)
    else:
        _model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def merge_sbs_videos(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    audio_source: Path,
    has_audio: bool = True,
    bitrate: str = "50M",
    target_resolution: Optional[tuple] = None,
) -> bool:
    """Merge two matted eye videos side-by-side using FFmpeg hstack.

    If target_resolution=(w, h), the hstacked output is scaled to that size.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_args = [
        "-i", str(left_path),
        "-i", str(right_path),
    ]
    if has_audio:
        input_args.extend(["-i", str(audio_source)])

    if target_resolution:
        tw, th = target_resolution
        # Ensure even dimensions
        tw = tw + (tw % 2)
        th = th + (th % 2)
        filter_str = f"[0:v][1:v]hstack=inputs=2[sbs];[sbs]scale={tw}:{th}:flags=lanczos[v]"
    else:
        filter_str = "[0:v][1:v]hstack=inputs=2[v]"

    filter_args = [
        "-filter_complex", filter_str,
        "-map", "[v]",
    ]
    if has_audio:
        filter_args.extend(["-map", "2:a", "-c:a", "copy"])

    cmd = build_encode_cmd(
        input_args, output_path, bitrate,
        extra_input_args=filter_args,
    )

    logger.info("Merging L/R matted videos to SBS: %s", output_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, env=_ffmpeg_env())
    if result.returncode != 0:
        logger.error("SBS merge failed: %s", result.stderr[-500:])
        return False
    return True


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(job: dict) -> dict:
    """RunPod serverless handler entry point."""
    job_input = job["input"]
    job_id = job.get("id", "unknown")

    source_key = job_input["source_key"]
    output_key = job_input["output_key"]
    bucket = job_input.get("s3_bucket", os.environ.get("S3_BUCKET", "ppp-processor-media"))
    vr_type = job_input.get("vr_type", "auto")
    model_type = job_input.get("model_type", "resnet50")
    downsample_ratio = float(job_input.get("downsample_ratio", 0.4))
    green_color = job_input.get("green_color", [0, 177, 64])
    vr_encode_bitrate = job_input.get("vr_encode_bitrate", "50M")
    encode_bitrate = job_input.get("encode_bitrate", "15M")
    use_refine_alpha = bool(job_input.get("refine_alpha", False))

    # New pipeline features
    use_despill = bool(job_input.get("despill", False))
    despill_strength = float(job_input.get("despill_strength", 0.8))
    alpha_sharpness = job_input.get("alpha_sharpness", "fine")
    output_mode = job_input.get("output_mode", "green_screen")
    alpha_output_key = job_input.get("alpha_output_key", "")
    use_upscale = bool(job_input.get("upscale", False))
    upscale_model = job_input.get("upscale_model", "RealESRGAN_x4plus")
    upscale_scale = int(job_input.get("upscale_scale", 4))
    upscale_tile_size = int(job_input.get("upscale_tile_size", 512))

    # Quality tuning params (matching config defaults from settings.yaml)
    refine_threshold_low = float(job_input.get("refine_threshold_low", 0.01))
    refine_threshold_high = float(job_input.get("refine_threshold_high", 0.99))
    refine_laplacian_strength = float(job_input.get("refine_laplacian_strength", 0.3))
    refine_morph_kernel = int(job_input.get("refine_morph_kernel", 5))
    despill_dilation_kernel = int(job_input.get("despill_dilation_kernel", 7))
    despill_dilation_iters = int(job_input.get("despill_dilation_iters", 2))

    logger.info("=== Job %s starting ===", job_id)
    logger.info("Source: s3://%s/%s", bucket, source_key)
    logger.info("VR type: %s, model: %s, downsample: %.2f", vr_type, model_type, downsample_ratio)
    logger.info(
        "Pipeline: despill=%s, refine=%s (sharpness=%s), output_mode=%s, upscale=%s",
        use_despill, use_refine_alpha, alpha_sharpness, output_mode, use_upscale,
    )

    t_job_start = time.time()

    with tempfile.TemporaryDirectory(prefix="runpod_matte_") as tmpdir:
        work_dir = Path(tmpdir)
        input_path = work_dir / Path(source_key).name
        output_path = work_dir / f"{input_path.stem}_matted.mp4"

        # Alpha output path for alpha_xalpha mode
        alpha_local_path = None
        if output_mode == "alpha_xalpha":
            alpha_local_path = work_dir / f"{input_path.stem}_matted_XALPHA.mp4"
            if not alpha_output_key:
                # Auto-generate alpha key from output key
                stem = Path(output_key).stem
                parent = str(Path(output_key).parent).replace("\\", "/")
                alpha_output_key = f"{parent}/{stem}_XALPHA.mp4"

        # 1. Download source from S3
        try:
            s3_download(bucket, source_key, input_path)
        except Exception as e:
            return {"error": f"S3 download failed: {e}"}

        # 2. Probe video
        try:
            probe = probe_video(input_path)
        except Exception as e:
            return {"error": f"Video probe failed: {e}"}

        logger.info(
            "Video: %dx%d @ %.2f fps, %d frames, %.1fs, codec=%s, audio=%s",
            probe["width"], probe["height"], probe["fps"],
            probe["total_frames"], probe["duration"],
            probe.get("codec_name", "?"), probe["has_audio"],
        )

        # 3. Auto-detect VR type if needed
        if vr_type == "auto":
            vr_type = detect_vr_type(input_path, probe["width"], probe["height"])
            logger.info("Auto-detected VR type: %s", vr_type)

        # 4. Load model (cold start or model type change)
        try:
            load_model(model_type)
        except Exception as e:
            return {"error": f"Model load failed: {e}"}

        # 5. Initialize upscaler if requested
        upscaler_inst = None
        if use_upscale:
            try:
                from upscaler import RealESRGANUpscaler
                upscaler_inst = RealESRGANUpscaler(
                    model_name=upscale_model,
                    scale=upscale_scale,
                    fp16=_use_fp16,
                    device=str(_device),
                    tile_size=upscale_tile_size,
                )
                upscaler_inst.load()
                logger.info("Upscaler loaded: %s (scale=%d)", upscale_model, upscale_scale)
            except Exception as e:
                logger.warning("Upscaler init failed, continuing without: %s", e)
                upscaler_inst = None

        # Pipeline kwargs shared between SBS/2D paths
        fast_decode = bool(job_input.get("fast_decode", True))
        pipeline_kwargs = dict(
            use_refine_alpha=use_refine_alpha,
            use_despill=use_despill,
            despill_strength=despill_strength,
            alpha_sharpness=alpha_sharpness,
            output_mode=output_mode,
            upscaler_instance=upscaler_inst,
            fast_decode=fast_decode,
            refine_threshold_low=refine_threshold_low,
            refine_threshold_high=refine_threshold_high,
            refine_laplacian_strength=refine_laplacian_strength,
            refine_morph_kernel=refine_morph_kernel,
            despill_dilation_kernel=despill_dilation_kernel,
            despill_dilation_iters=despill_dilation_iters,
        )

        # 6. Compute smart upscale target
        target_res = compute_target_resolution(
            probe["width"], probe["height"], vr_type,
        )
        if target_res:
            logger.info(
                "Auto-upscale: %dx%d → %dx%d (lanczos)",
                probe["width"], probe["height"], target_res[0], target_res[1],
            )
        else:
            logger.info("No auto-upscale needed for %dx%d", probe["width"], probe["height"])

        # 7. Process
        try:
            if vr_type == "sbs":
                # For VR SBS: use smart target, or at least restore to original if fast_decode shrunk it
                sbs_target = target_res or ((probe["width"], probe["height"]) if fast_decode else None)
                success = process_vr_sbs(
                    input_path, output_path, probe,
                    green_color=green_color,
                    downsample_ratio=downsample_ratio,
                    bitrate=vr_encode_bitrate,
                    alpha_output_path=alpha_local_path,
                    target_resolution=sbs_target,
                    **pipeline_kwargs,
                )
            else:
                # 2D or TB — matte full frame
                success = process_video_streaming(
                    input_path, output_path, probe,
                    green_color=green_color,
                    downsample_ratio=downsample_ratio,
                    bitrate=encode_bitrate,
                    alpha_output_path=alpha_local_path,
                    **pipeline_kwargs,
                )
        except Exception as e:
            logger.exception("Processing failed")
            return {"error": f"Processing failed: {e}"}

        if not success:
            return {"error": "Matting pipeline returned failure"}

        # 8. Post-matte rescale for 2D/TB (VR SBS handled in merge step)
        if target_res and vr_type != "sbs" and output_path.exists():
            tw, th = target_res
            if not rescale_video(output_path, output_path, tw, th, encode_bitrate):
                logger.warning("Rescale failed, uploading at model resolution")

        # 7. Upload result(s) to S3
        try:
            s3_upload(output_path, bucket, output_key)
        except Exception as e:
            return {"error": f"S3 upload failed: {e}"}

        # Upload alpha file if generated
        if output_mode == "alpha_xalpha" and alpha_local_path and alpha_local_path.exists():
            try:
                s3_upload(alpha_local_path, bucket, alpha_output_key)
                logger.info("Alpha uploaded: s3://%s/%s", bucket, alpha_output_key)
            except Exception as e:
                return {"error": f"Alpha S3 upload failed: {e}"}

        elapsed = time.time() - t_job_start
        output_size_mb = output_path.stat().st_size / (1024 * 1024)

        result = {
            "status": "success",
            "output_key": output_key,
            "output_mode": output_mode,
            "vr_type": vr_type,
            "source_resolution": f"{probe['width']}x{probe['height']}",
            "output_resolution": f"{target_res[0]}x{target_res[1]}" if target_res else f"{probe['width']}x{probe['height']}",
            "total_frames": probe["total_frames"],
            "duration_seconds": probe["duration"],
            "output_size_mb": round(output_size_mb, 1),
            "processing_time_seconds": round(elapsed, 1),
        }
        if output_mode == "alpha_xalpha":
            result["alpha_output_key"] = alpha_output_key
        logger.info("=== Job %s complete in %.1fs ===", job_id, elapsed)
        return result


# ---------------------------------------------------------------------------
# RunPod entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting RunPod matte handler")
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Pre-load model at startup (cold start optimization)
    # Non-fatal: if this fails, model loads on first job instead
    model_type = os.environ.get("MODEL_TYPE", "resnet50")
    try:
        load_model(model_type)
    except Exception as e:
        logger.error("Pre-load failed (will retry on first job): %s", e)
        # Ensure handler can still start
        _model = None

    logger.info("Registering handler with RunPod serverless")
    runpod.serverless.start({"handler": handler})
