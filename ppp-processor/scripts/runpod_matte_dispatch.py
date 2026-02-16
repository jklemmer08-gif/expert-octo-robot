#!/usr/bin/env python3
"""RunPod VR Matte Dispatch — upload to S3, submit to RunPod, download result.

Usage:
  python scripts/runpod_matte_dispatch.py --scene 12345 12346 12347
  python scripts/runpod_matte_dispatch.py --file "M:\\path\\to\\vr_video.mp4"
  python scripts/runpod_matte_dispatch.py --tag PPP-VR-Matte
  python scripts/runpod_matte_dispatch.py --status
  python scripts/runpod_matte_dispatch.py --status --job-id abc123

Per-job flow:
  1. Look up scene in Stash (get file path)
  2. Upload source to S3 (ppp-matte/input/)
  3. Submit async job to RunPod endpoint
  4. Poll for completion with progress display
  5. Download result from S3 to N:\\ppp-output\\matte\\
  6. Tag scene in Stash (Green Screen Available, PPP-Processed, Passthrough_simple)
  7. Clean up S3 input file

Requires: RUNPOD_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY env vars.
Config: PPP_CONFIG env var pointing to settings.windows.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
import requests

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.config import Settings, get_settings
from src.integrations.stash import StashClient


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def get_config():
    """Load settings and extract RunPod/S3 config."""
    settings = get_settings()
    api_key = os.environ.get("RUNPOD_API_KEY") or settings.runpod.api_key
    if not api_key:
        print("Error: RUNPOD_API_KEY not set (env var or settings.yaml)")
        sys.exit(1)

    endpoint_id = settings.runpod.serverless_endpoint_id
    if not endpoint_id:
        print("Error: runpod.serverless_endpoint_id not set in config")
        sys.exit(1)

    return {
        "api_key": api_key,
        "endpoint_id": endpoint_id,
        "s3_bucket": settings.runpod.s3_bucket,
        "s3_region": settings.runpod.s3_region,
        "matte": settings.matte,
        "stash_url": settings.paths.stash_url,
        "stash_api_key": settings.paths.stash_api_key,
        "output_dir": Path(settings.paths.output_dir) / "matte",
    }


# ---------------------------------------------------------------------------
# RunPod API
# ---------------------------------------------------------------------------
class RunPodClient:
    """Minimal RunPod serverless API client."""

    BASE_URL = "https://api.runpod.ai/v2"

    def __init__(self, api_key: str, endpoint_id: str):
        self.endpoint_id = endpoint_id
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def submit(self, job_input: dict) -> str:
        """Submit async job. Returns job ID."""
        url = f"{self.BASE_URL}/{self.endpoint_id}/run"
        resp = self.session.post(url, json={"input": job_input})
        resp.raise_for_status()
        data = resp.json()
        return data["id"]

    def status(self, job_id: str) -> dict:
        """Get job status."""
        url = f"{self.BASE_URL}/{self.endpoint_id}/status/{job_id}"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    def cancel(self, job_id: str):
        """Cancel a running job."""
        url = f"{self.BASE_URL}/{self.endpoint_id}/cancel/{job_id}"
        resp = self.session.post(url)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def get_s3_client(region: str = "us-east-1"):
    return boto3.client("s3", region_name=region)


def upload_to_s3(local_path: Path, bucket: str, key: str, region: str):
    """Upload file to S3 with progress."""
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  Uploading {local_path.name} ({size_mb:.1f} MB) to s3://{bucket}/{key}")
    s3 = get_s3_client(region)
    s3.upload_file(str(local_path), bucket, key)
    print(f"  Upload complete")


def download_from_s3(bucket: str, key: str, local_path: Path, region: str):
    """Download file from S3."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading s3://{bucket}/{key} → {local_path}")
    s3 = get_s3_client(region)
    s3.download_file(bucket, key, str(local_path))
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  Downloaded {size_mb:.1f} MB")


def delete_s3_key(bucket: str, key: str, region: str):
    """Delete a key from S3."""
    s3 = get_s3_client(region)
    s3.delete_object(Bucket=bucket, Key=key)


# ---------------------------------------------------------------------------
# Path mapping (Linux ↔ Windows)
# ---------------------------------------------------------------------------
PATH_MAPS = [
    ("/home/jtk1234/media-drive1", "M:\\"),
    ("/home/jtk1234/media-drive2", "N:\\"),
]


def to_local_path(docker_path: str) -> Path:
    """Convert Stash (Linux/Docker) path to local Windows path."""
    for linux_prefix, windows_prefix in PATH_MAPS:
        if docker_path.startswith(linux_prefix):
            return Path(docker_path.replace(linux_prefix, windows_prefix, 1))
    return Path(docker_path)


# ---------------------------------------------------------------------------
# Dispatch logic
# ---------------------------------------------------------------------------
def dispatch_file(
    file_path: Path,
    config: dict,
    runpod: RunPodClient,
    scene_id: str | None = None,
    output_mode: str = "green_screen",
    despill: bool = False,
    despill_strength: float = 0.8,
    refine_alpha: bool = False,
    alpha_sharpness: str = "fine",
    upscale: bool = False,
    upscale_model: str = "RealESRGAN_x4plus",
) -> str | None:
    """Dispatch a single file for RunPod matting. Returns job ID or None."""
    if not file_path.exists():
        print(f"  ERROR: File not found: {file_path}")
        return None

    bucket = config["s3_bucket"]
    region = config["s3_region"]
    mc = config["matte"]

    source_key = f"ppp-matte/input/{file_path.name}"
    output_key = f"ppp-matte/output/{file_path.stem}_matted.mp4"

    # Upload source to S3
    upload_to_s3(file_path, bucket, source_key, region)

    # Build job input
    job_input = {
        "source_key": source_key,
        "output_key": output_key,
        "s3_bucket": bucket,
        "vr_type": "auto",
        "model_type": mc.model_type,
        "downsample_ratio": mc.downsample_ratio,
        "green_color": mc.green_color,
        "vr_encode_bitrate": mc.vr_encode_bitrate,
        "encode_bitrate": mc.encode_bitrate,
        "output_mode": output_mode,
        "despill": despill,
        "despill_strength": despill_strength,
        "refine_alpha": refine_alpha,
        "alpha_sharpness": alpha_sharpness,
        "upscale": upscale,
        "upscale_model": upscale_model,
        # Quality tuning params from config
        "refine_threshold_low": mc.refine_threshold_low,
        "refine_threshold_high": mc.refine_threshold_high,
        "refine_laplacian_strength": mc.refine_laplacian_strength,
        "refine_morph_kernel": mc.refine_morph_kernel,
        "despill_dilation_kernel": mc.despill_dilation_kernel,
        "despill_dilation_iters": mc.despill_dilation_iters,
    }

    # Auto-generate alpha output key for alpha_xalpha mode
    if output_mode == "alpha_xalpha":
        job_input["alpha_output_key"] = f"ppp-matte/output/{file_path.stem}_matted_XALPHA.mp4"

    # Submit to RunPod
    job_id = runpod.submit(job_input)
    print(f"  Submitted RunPod job: {job_id}")
    print(f"  Mode: {output_mode}, despill={despill}, refine={refine_alpha}, upscale={upscale}")

    _notify_dashboard(job_id, job_input)

    return job_id


def _notify_dashboard(job_id: str, job_input: dict) -> None:
    """Append job info to JSONL dropbox for the dashboard to ingest."""
    try:
        jsonl_path = _project_root / "data" / "runpod_jobs.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "job_id": job_id,
            "job_input": job_input,
            "submitted_at": datetime.now().isoformat(),
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # Best-effort — never break dispatch


def poll_job(
    job_id: str,
    config: dict,
    runpod_client: RunPodClient,
    scene_id: str | None = None,
    stash: StashClient | None = None,
) -> bool:
    """Poll a RunPod job until completion. Returns True on success."""
    bucket = config["s3_bucket"]
    region = config["s3_region"]
    output_dir = config["output_dir"]

    poll_interval = 10  # seconds
    max_wait = 7200  # 2 hours
    elapsed = 0

    while elapsed < max_wait:
        status = runpod_client.status(job_id)
        state = status.get("status", "UNKNOWN")

        if state == "COMPLETED":
            output = status.get("output", {})
            if isinstance(output, dict) and output.get("status") == "success":
                output_key = output["output_key"]
                output_mode_result = output.get("output_mode", "green_screen")
                vr_type = output.get("vr_type", "?")
                proc_time = output.get("processing_time_seconds", 0)
                output_mb = output.get("output_size_mb", 0)

                print(f"  Job completed! VR={vr_type}, mode={output_mode_result}, {proc_time:.0f}s, {output_mb:.1f} MB")

                # Download result from S3
                result_filename = Path(output_key).name
                local_output = output_dir / result_filename
                download_from_s3(bucket, output_key, local_output, region)

                # Download alpha file if alpha_xalpha mode
                alpha_output_key = output.get("alpha_output_key")
                if alpha_output_key:
                    alpha_filename = Path(alpha_output_key).name
                    # Place alpha alongside the main video (HereSphere auto-detects)
                    local_alpha = local_output.parent / alpha_filename
                    try:
                        download_from_s3(bucket, alpha_output_key, local_alpha, region)
                        print(f"  Alpha: {local_alpha}")
                    except Exception as e:
                        print(f"  WARNING: Alpha download failed: {e}")

                # Tag scene in Stash
                if scene_id and stash:
                    try:
                        if output_mode_result == "alpha_xalpha":
                            stash.add_tag_to_scene(scene_id, "Alpha Available")
                        else:
                            stash.add_tag_to_scene(scene_id, "Green Screen Available")
                        stash.add_tag_to_scene(scene_id, "PPP-Processed")
                        stash.add_tag_to_scene(scene_id, "Passthrough_simple")
                        print(f"  Tagged scene {scene_id} in Stash")
                    except Exception as e:
                        print(f"  WARNING: Stash tagging failed: {e}")

                # Clean up S3 input
                source_key = f"ppp-matte/input/{Path(output_key).stem.replace('_matted', '')}.mp4"
                try:
                    delete_s3_key(bucket, source_key, region)
                except Exception:
                    pass

                return True
            else:
                error = output.get("error", "Unknown error") if isinstance(output, dict) else str(output)
                print(f"  Job failed: {error}")
                return False

        elif state == "FAILED":
            error = status.get("error", "Unknown error")
            print(f"  Job FAILED: {error}")
            return False

        elif state in ("IN_QUEUE", "IN_PROGRESS"):
            mins = elapsed / 60
            print(f"  [{mins:.0f}m] Status: {state}", end="\r")
        else:
            print(f"  [{elapsed}s] Unknown status: {state}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    print(f"  Timed out after {max_wait}s")
    return False


def dispatch_scenes(scene_ids: list[str], config: dict, pipeline_kwargs: dict | None = None):
    """Dispatch multiple scenes by Stash scene ID."""
    stash = StashClient(config["stash_url"], config["stash_api_key"])
    runpod_client = RunPodClient(config["api_key"], config["endpoint_id"])
    pkw = pipeline_kwargs or {}

    jobs = []

    for scene_id in scene_ids:
        print(f"\n--- Scene {scene_id} ---")
        scene = stash.find_scene_by_id(scene_id)
        if not scene:
            print(f"  Scene not found in Stash")
            continue

        files = scene.get("files", [])
        if not files:
            print(f"  No files for scene")
            continue

        docker_path = files[0].get("path", "")
        local_path = to_local_path(docker_path)
        print(f"  File: {local_path}")

        job_id = dispatch_file(local_path, config, runpod_client, scene_id, **pkw)
        if job_id:
            jobs.append((job_id, scene_id))

    # Poll all jobs
    if jobs:
        print(f"\n=== Polling {len(jobs)} job(s) ===")
        for job_id, scene_id in jobs:
            print(f"\n--- Job {job_id} (scene {scene_id}) ---")
            poll_job(job_id, config, runpod_client, scene_id, stash)


def dispatch_files(file_paths: list[str], config: dict, pipeline_kwargs: dict | None = None):
    """Dispatch files directly (no Stash lookup)."""
    runpod_client = RunPodClient(config["api_key"], config["endpoint_id"])
    pkw = pipeline_kwargs or {}

    jobs = []
    for fp in file_paths:
        local_path = Path(fp)
        print(f"\n--- {local_path.name} ---")
        job_id = dispatch_file(local_path, config, runpod_client, **pkw)
        if job_id:
            jobs.append((job_id, None))

    if jobs:
        print(f"\n=== Polling {len(jobs)} job(s) ===")
        for job_id, _ in jobs:
            print(f"\n--- Job {job_id} ---")
            poll_job(job_id, config, runpod_client)


def dispatch_by_tag(tag_name: str, config: dict, pipeline_kwargs: dict | None = None):
    """Dispatch all scenes with a given Stash tag."""
    stash = StashClient(config["stash_url"], config["stash_api_key"])

    scenes = stash.find_scenes_by_tag(tag_name)
    if not scenes:
        print(f"No scenes found with tag '{tag_name}'")
        return

    scene_ids = [s["id"] for s in scenes]
    print(f"Found {len(scene_ids)} scene(s) with tag '{tag_name}'")
    dispatch_scenes(scene_ids, config, pipeline_kwargs)


def show_status(config: dict, job_id: str | None = None):
    """Show status of a specific job or recent jobs."""
    runpod_client = RunPodClient(config["api_key"], config["endpoint_id"])

    if job_id:
        status = runpod_client.status(job_id)
        print(json.dumps(status, indent=2))
    else:
        print("Use --status --job-id <id> to check a specific job")
        print(f"Endpoint: {config['endpoint_id']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Dispatch VR matte jobs to RunPod serverless",
    )
    parser.add_argument(
        "--scene", nargs="+",
        help="Stash scene ID(s) to process",
    )
    parser.add_argument(
        "--file", nargs="+",
        help="Direct file path(s) to process",
    )
    parser.add_argument(
        "--tag",
        help="Process all Stash scenes with this tag",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show job status",
    )
    parser.add_argument(
        "--job-id",
        help="Specific job ID for --status",
    )

    # Pipeline feature flags
    parser.add_argument(
        "--output-mode", default="green_screen",
        choices=["green_screen", "alpha_xalpha"],
        help="Output mode (default: green_screen)",
    )
    parser.add_argument(
        "--despill", action="store_true",
        help="Enable green spill removal",
    )
    parser.add_argument(
        "--despill-strength", type=float, default=0.8,
        help="Despill strength 0-1 (default: 0.8)",
    )
    parser.add_argument(
        "--refine-alpha", action="store_true",
        help="Enable alpha refinement",
    )
    parser.add_argument(
        "--alpha-sharpness", default="fine",
        choices=["fine", "soft"],
        help="Alpha refinement mode (default: fine)",
    )
    parser.add_argument(
        "--upscale", action="store_true",
        help="Enable Real-ESRGAN 4x upscaling",
    )
    parser.add_argument(
        "--upscale-model", default="RealESRGAN_x4plus",
        choices=["RealESRGAN_x4plus", "realesr-animevideov3"],
        help="Upscaler model (default: RealESRGAN_x4plus)",
    )

    args = parser.parse_args()

    if not any([args.scene, args.file, args.tag, args.status]):
        parser.print_help()
        sys.exit(1)

    config = get_config()

    # Build pipeline kwargs from CLI flags
    pipeline_kwargs = {
        "output_mode": args.output_mode,
        "despill": args.despill,
        "despill_strength": args.despill_strength,
        "refine_alpha": args.refine_alpha,
        "alpha_sharpness": args.alpha_sharpness,
        "upscale": args.upscale,
        "upscale_model": args.upscale_model,
    }

    if args.status:
        show_status(config, args.job_id)
    elif args.scene:
        dispatch_scenes(args.scene, config, pipeline_kwargs)
    elif args.file:
        dispatch_files(args.file, config, pipeline_kwargs)
    elif args.tag:
        dispatch_by_tag(args.tag, config, pipeline_kwargs)


if __name__ == "__main__":
    main()
