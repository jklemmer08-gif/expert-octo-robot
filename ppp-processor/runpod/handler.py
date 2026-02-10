"""RunPod Serverless Handler for VR Video Matting.

Self-contained matting handler adapted from src/processor.py.
No Settings/Celery/TRT dependency chain — uses FFmpeg streaming pipes,
boto3 for S3, and module-level model loading for warm reuse.

Job input schema:
    {
        "source_key": "ppp-matte/input/scene_12345.mp4",
        "output_key": "ppp-matte/output/scene_12345_matted.mp4",
        "s3_bucket": "ppp-processor-media",
        "vr_type": "sbs",            # "sbs", "tb", or "2d"
        "model_type": "resnet50",     # "resnet50" or "mobilenetv3"
        "downsample_ratio": 0.4,
        "green_color": [0, 177, 64],
        "vr_encode_bitrate": "50M",
        "encode_bitrate": "15M"
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

    # torch.compile — reduce-overhead mode on Linux for fused kernels
    if _device.type == "cuda" and platform.system() != "Windows":
        try:
            _model = torch.compile(_model, mode="reduce-overhead")
            logger.info("torch.compile() enabled (reduce-overhead)")
        except Exception:
            try:
                _model = torch.compile(_model, backend="eager")
                logger.info("torch.compile() enabled (eager)")
            except Exception as e:
                logger.info("torch.compile() unavailable: %s", e)

    logger.info("Model loaded successfully")


# ---------------------------------------------------------------------------
# Video probing (from processor.py:672)
# ---------------------------------------------------------------------------
def probe_video(video_path: Path) -> dict:
    """Probe video for dimensions, FPS, frame count, and audio presence."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
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


# ---------------------------------------------------------------------------
# NVENC probe (from processor.py:534)
# ---------------------------------------------------------------------------
_nvenc_available: Optional[bool] = None


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
) -> bool:
    """Streaming matting: FFmpeg decode pipe → GPU inference → FFmpeg encode pipe.

    3-stage async pipeline with pre-allocated pinned memory.
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

    frame_bytes = proc_w * proc_h * 3  # RGB24
    pipe_bufsize = frame_bytes * 8

    # Pre-allocate reusable tensors
    dtype = torch.float16 if _use_fp16 else torch.float32
    pin_buffer = torch.empty((proc_h, proc_w, 3), dtype=torch.uint8).pin_memory()
    gpu_input = torch.empty((1, 3, proc_h, proc_w), dtype=dtype, device=_device)

    green_tensor = torch.tensor(
        [[[green_color[0] / 255.0]], [[green_color[1] / 255.0]], [[green_color[2] / 255.0]]],
        dtype=dtype, device=_device,
    )

    out_gpu = torch.empty((1, 3, proc_h, proc_w), dtype=dtype, device=_device)
    out_pin = torch.empty((proc_h, proc_w, 3), dtype=torch.uint8).pin_memory()

    transfer_stream = torch.cuda.Stream() if _device.type == "cuda" else None

    # --- FFmpeg decode pipe (try NVDEC, fallback CPU) ---
    decode_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if _device.type == "cuda":
        decode_cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    decode_cmd.extend(["-i", str(input_path)])
    if vf_decode:
        if _device.type == "cuda":
            decode_cmd.extend(["-vf", f"hwdownload,format=nv12,{vf_decode}"])
        else:
            decode_cmd.extend(["-vf", vf_decode])
    elif _device.type == "cuda":
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
    encode_cmd = build_encode_cmd(
        pipe_input_args, output_path, bitrate,
        extra_input_args=audio_args,
    )

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

    # Try NVDEC, fallback to CPU decode
    decoder = None
    nvdec_failed = False
    try:
        decoder = subprocess.Popen(
            decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=pipe_bufsize,
        )
        test_raw = _read_with_timeout(decoder.stdout, frame_bytes, timeout=10)
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
        test_raw = _read_with_timeout(decoder.stdout, frame_bytes, timeout=10)
        if not test_raw or len(test_raw) < frame_bytes:
            logger.error("CPU decode also failed")
            decoder.kill()
            return False

    # --- Async 3-stage pipeline ---
    SENTINEL = None
    decode_q: queue.Queue = queue.Queue(maxsize=4)
    encode_q: queue.Queue = queue.Queue(maxsize=4)
    decode_error = [None]
    encode_error = [None]

    encoder = subprocess.Popen(
        encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=pipe_bufsize,
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

    rec = [None] * 4

    try:
        t_decode = threading.Thread(target=decode_thread, daemon=True)
        t_encode = threading.Thread(target=encode_thread, daemon=True)
        t_decode.start()
        t_encode.start()

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

            # GPU inference (PyTorch only — no TRT on RunPod)
            with torch.no_grad():
                fgr, pha, *rec = _model(gpu_input, *rec, downsample_ratio)

            # GPU compositing: out = green + pha * (fgr - green)
            torch.addcmul(green_tensor.expand_as(fgr), pha, fgr - green_tensor, out=out_gpu)
            out_gpu.clamp_(0.0, 1.0)

            # Fast GPU→CPU transfer
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
) -> bool:
    """Process VR SBS video: crop L/R eyes, matte separately, hstack merge."""
    global _model

    half_w = probe["width"] // 2
    h = probe["height"]

    work_dir = output_path.parent / f"_vr_work_{output_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        left_tmp = work_dir / "left_matted.mp4"
        right_tmp = work_dir / "right_matted.mp4"

        # Matte left eye
        logger.info("Processing left eye (%dx%d)", half_w, h)
        success = process_video_streaming(
            input_path, left_tmp, probe,
            green_color=green_color,
            downsample_ratio=downsample_ratio,
            bitrate=bitrate,
            crop_region=(0, 0, half_w, h),
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
        )
        if not success:
            logger.error("Right eye matting failed")
            return False

        # Merge L/R into SBS
        logger.info("Merging L/R into SBS output")
        success = merge_sbs_videos(
            left_tmp, right_tmp, output_path,
            audio_source=input_path,
            has_audio=probe["has_audio"],
            bitrate=bitrate,
        )
        return success

    finally:
        # Clean up temp files
        import shutil
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def _model_reset_rec():
    """Reset model recurrent state between L/R eye processing.

    Setting _model to None forces a full reload on the next call to
    load_model(), which clears RVM's internal recurrent buffers.
    """
    global _model
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

    cmd = build_encode_cmd(
        input_args, output_path, bitrate,
        extra_input_args=filter_args,
    )

    logger.info("Merging L/R matted videos to SBS: %s", output_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
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

    logger.info("=== Job %s starting ===", job_id)
    logger.info("Source: s3://%s/%s", bucket, source_key)
    logger.info("VR type: %s, model: %s, downsample: %.2f", vr_type, model_type, downsample_ratio)

    t_job_start = time.time()

    with tempfile.TemporaryDirectory(prefix="runpod_matte_") as tmpdir:
        work_dir = Path(tmpdir)
        input_path = work_dir / Path(source_key).name
        output_path = work_dir / f"{input_path.stem}_matted.mp4"

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
            "Video: %dx%d @ %.2f fps, %d frames, %.1fs, audio=%s",
            probe["width"], probe["height"], probe["fps"],
            probe["total_frames"], probe["duration"], probe["has_audio"],
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

        # 5. Process
        try:
            if vr_type == "sbs":
                success = process_vr_sbs(
                    input_path, output_path, probe,
                    green_color=green_color,
                    downsample_ratio=downsample_ratio,
                    bitrate=vr_encode_bitrate,
                )
            else:
                # 2D or TB (TB treated as 2D for now — matte full frame)
                success = process_video_streaming(
                    input_path, output_path, probe,
                    green_color=green_color,
                    downsample_ratio=downsample_ratio,
                    bitrate=encode_bitrate,
                )
        except Exception as e:
            logger.exception("Processing failed")
            return {"error": f"Processing failed: {e}"}

        if not success:
            return {"error": "Matting pipeline returned failure"}

        # 6. Upload result to S3
        try:
            s3_upload(output_path, bucket, output_key)
        except Exception as e:
            return {"error": f"S3 upload failed: {e}"}

        elapsed = time.time() - t_job_start
        output_size_mb = output_path.stat().st_size / (1024 * 1024)

        result = {
            "status": "success",
            "output_key": output_key,
            "vr_type": vr_type,
            "resolution": f"{probe['width']}x{probe['height']}",
            "total_frames": probe["total_frames"],
            "duration_seconds": probe["duration"],
            "output_size_mb": round(output_size_mb, 1),
            "processing_time_seconds": round(elapsed, 1),
        }
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
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_mem / 1e9)

    # Pre-load model at startup (cold start optimization)
    model_type = os.environ.get("MODEL_TYPE", "resnet50")
    try:
        load_model(model_type)
    except Exception as e:
        logger.error("Failed to pre-load model: %s", e)

    runpod.serverless.start({"handler": handler})
