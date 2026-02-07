"""Cloud GPU Celery worker for PPP Processor.

Refactored from runpod_worker.py:RunPodClient and RunPodWorker.
Handles budget check, pod lifecycle, upload/download, and processing.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from celery import Task

from src.celery_app import app
from src.config import get_settings
from src.database import JobDatabase

logger = logging.getLogger("ppp.worker.cloud")


# ---------------------------------------------------------------------------
# RunPod API Client (from runpod_worker.py:45-243)
# ---------------------------------------------------------------------------
class RunPodClient:
    """RunPod GraphQL API client for pod management."""

    GRAPHQL_URL = "https://api.runpod.io/graphql"

    def __init__(self, api_key: str):
        import requests
        self._requests = requests
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _gql(self, query: str, variables: dict = None) -> dict:
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = self._requests.post(self.GRAPHQL_URL, headers=self.headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"RunPod API error: {resp.text}")
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL error: {data['errors']}")
        return data.get("data", {})

    def get_available_gpus(self) -> List[Dict]:
        data = self._gql("""
            query { gpuTypes {
                id displayName memoryInGb
                lowestPrice { minimumBidPrice uninterruptablePrice }
            }}
        """)
        return data.get("gpuTypes", [])

    def find_best_gpu(self, min_vram: int = 24, max_price: float = 5.0) -> Optional[Dict]:
        gpus = self.get_available_gpus()
        candidates = []
        for gpu in gpus:
            if gpu.get("memoryInGb", 0) < min_vram:
                continue
            price_info = gpu.get("lowestPrice", {})
            price = price_info.get("uninterruptablePrice") or price_info.get("minimumBidPrice")
            if price and price <= max_price:
                candidates.append({
                    "id": gpu["id"], "name": gpu["displayName"],
                    "vram": gpu["memoryInGb"], "price": price,
                })
        candidates.sort(key=lambda x: x["price"])
        # Prefer 4090
        for c in candidates:
            if "4090" in c["name"]:
                return c
        return candidates[0] if candidates else None

    def create_pod(self, gpu_type_id: str, name: str, container_image: str) -> Dict:
        data = self._gql("""
            mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
                podFindAndDeployOnDemand(input: $input) {
                    id name
                    runtime { ports { ip isIpPublic privatePort publicPort } }
                }
            }
        """, {"input": {
            "cloudType": "SECURE", "gpuTypeId": gpu_type_id, "name": name,
            "imageName": container_image, "ports": "22/tcp",
            "volumeInGb": 50, "containerDiskInGb": 20, "startSsh": True,
        }})
        return data["podFindAndDeployOnDemand"]

    def get_pod_status(self, pod_id: str) -> Dict:
        data = self._gql("""
            query Pod($podId: String!) {
                pod(input: {podId: $podId}) {
                    id runtime {
                        uptimeInSeconds
                        ports { ip isIpPublic privatePort publicPort }
                    }
                }
            }
        """, {"podId": pod_id})
        return data.get("pod", {})

    def terminate_pod(self, pod_id: str) -> bool:
        self._gql("""
            mutation($podId: String!) { podTerminate(input: {podId: $podId}) }
        """, {"podId": pod_id})
        return True

    def wait_for_pod_ready(self, pod_id: str, timeout: int = 300) -> Dict:
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_pod_status(pod_id)
            for port in status.get("runtime", {}).get("ports", []):
                if port.get("privatePort") == 22 and port.get("ip"):
                    return {"ssh_ip": port["ip"], "ssh_port": port.get("publicPort", 22)}
            time.sleep(10)
        raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")


# ---------------------------------------------------------------------------
# Cloud Task Base
# ---------------------------------------------------------------------------
class CloudGPUTask(Task):
    _db = None
    _settings = None

    @property
    def settings(self):
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def db(self):
        if self._db is None:
            db_path = Path(self.settings.paths.temp_dir).parent / "jobs.db"
            self._db = JobDatabase(db_path)
        return self._db


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------
@app.task(base=CloudGPUTask, bind=True, name="src.workers.cloud_worker.process_cloud_job")
def process_cloud_job(self: CloudGPUTask, job_id: str):
    """Cloud processing: budget check -> create pod -> upload -> process -> download -> verify -> terminate."""
    from src.models.schemas import JobStatus

    job = self.db.get_job(job_id)
    if not job:
        return {"status": "error", "message": "Job not found"}

    settings = self.settings
    api_key = settings.runpod.api_key or os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        self.db.update_job_status(job_id, "failed", error="RUNPOD_API_KEY not set")
        return {"status": "failed", "error": "No API key"}

    # Budget check
    total_spent = self.db.get_total_cost()
    if total_spent >= settings.runpod.budget_total:
        self.db.update_job_status(job_id, "failed", error="RunPod budget exhausted")
        return {"status": "failed", "error": "Budget exhausted"}

    client = RunPodClient(api_key)
    pod_id = None
    start_time = time.time()

    worker_id = f"cloud-{job_id[:8]}"

    try:
        self.db.update_job_status(
            job_id, JobStatus.PROCESSING.value,
            worker_id=worker_id, current_stage="finding_gpu",
        )

        gpu = client.find_best_gpu(max_price=settings.runpod.max_cost_per_job)
        if not gpu:
            self.db.update_job_status(job_id, "failed", error="No suitable GPU available")
            return {"status": "failed", "error": "No GPU"}

        logger.info("Using %s @ $%.2f/hr", gpu["name"], gpu["price"])

        # Create pod
        self.db.update_job_status(job_id, "processing", current_stage="creating_pod")
        pod = client.create_pod(
            gpu["id"],
            f"ppp-{Path(job['source_path']).stem[:10]}",
            "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
        )
        pod_id = pod["id"]

        # Wait for ready
        self.db.update_job_status(job_id, "processing", current_stage="waiting_for_pod")
        ssh_info = client.wait_for_pod_ready(pod_id)

        # Setup pod
        self.db.update_job_status(job_id, "processing", current_stage="setting_up")
        _setup_pod(ssh_info)

        # Upload
        self.db.update_job_status(job_id, "processing", current_stage="uploading")
        source_path = Path(job["source_path"])
        remote_input = f"/workspace/input{source_path.suffix}"
        _scp_upload(ssh_info, source_path, remote_input)

        # Process
        self.db.update_job_status(job_id, "processing", current_stage="processing_remote")
        model = job.get("model") or settings.upscale.default_model
        scale = job.get("scale") or settings.upscale.scale_factor
        remote_output = "/workspace/output.mp4"
        _ssh_run(ssh_info, (
            f"cd /workspace && ./realesrgan-ncnn-vulkan "
            f"-i {remote_input} -o {remote_output} "
            f"-n {model} -s {scale} -f mp4"
        ))

        # Download
        self.db.update_job_status(job_id, "processing", current_stage="downloading")
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _scp_download(ssh_info, remote_output, output_path)

        # Verify
        if not output_path.exists() or output_path.stat().st_size < 1024:
            raise RuntimeError("Output file missing or too small")

        # Log cost
        duration_hours = (time.time() - start_time) / 3600
        cost = gpu["price"] * duration_hours
        self.db.log_cost(job_id, worker_id, gpu["name"], cost, duration_hours)

        self.db.update_job_status(
            job_id, JobStatus.COMPLETED.value,
            processing_time=time.time() - start_time,
            actual_cost=cost,
        )

        logger.info("Cloud job %s completed: %.2fh, $%.2f", job_id, duration_hours, cost)
        return {"status": "completed", "cost": cost}

    except Exception as e:
        error_msg = str(e)[:500]
        logger.error("Cloud job %s failed: %s", job_id, error_msg)
        self.db.update_job_status(
            job_id, JobStatus.FAILED.value,
            error=error_msg, processing_time=time.time() - start_time,
        )
        return {"status": "failed", "error": error_msg}

    finally:
        if pod_id:
            try:
                client.terminate_pod(pod_id)
                logger.info("Pod %s terminated", pod_id)
            except Exception as e:
                logger.error("Failed to terminate pod %s: %s", pod_id, e)


# ---------------------------------------------------------------------------
# SSH/SCP helpers (from runpod_worker.py:327-376)
# ---------------------------------------------------------------------------
def _ssh_run(ssh_info: Dict, command: str) -> str:
    cmd = [
        "ssh", "-p", str(ssh_info["ssh_port"]),
        "-o", "StrictHostKeyChecking=no",
        f"root@{ssh_info['ssh_ip']}",
        command,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr


def _scp_upload(ssh_info: Dict, local_path: Path, remote_path: str):
    cmd = [
        "scp", "-P", str(ssh_info["ssh_port"]),
        "-o", "StrictHostKeyChecking=no",
        str(local_path),
        f"root@{ssh_info['ssh_ip']}:{remote_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Upload failed: {result.stderr}")


def _scp_download(ssh_info: Dict, remote_path: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp", "-P", str(ssh_info["ssh_port"]),
        "-o", "StrictHostKeyChecking=no",
        f"root@{ssh_info['ssh_ip']}:{remote_path}",
        str(local_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")


def _setup_pod(ssh_info: Dict):
    """Install Real-ESRGAN on pod (from runpod_worker.py:378-396)."""
    setup_commands = [
        "apt-get update && apt-get install -y ffmpeg wget unzip",
        "pip install torch torchvision --quiet",
        "wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        "unzip -q realesrgan-ncnn-vulkan*.zip -d /workspace/",
        "chmod +x /workspace/realesrgan-ncnn-vulkan",
    ]
    for command in setup_commands:
        _ssh_run(ssh_info, command)
