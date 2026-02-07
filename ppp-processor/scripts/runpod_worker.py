#!/usr/bin/env python3
"""
PPP RunPod Worker - Cloud processing integration
Handles file upload/download and job execution on RunPod GPUs
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import hashlib

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class RunPodConfig:
    api_key: str
    preferred_gpu: str = "NVIDIA RTX 4090"
    fallback_gpu: str = "NVIDIA RTX 3090"
    max_cost_per_job: float = 5.00
    budget_total: float = 75.00
    container_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"

@dataclass
class CloudJob:
    local_path: Path
    remote_path: str
    output_path: Path
    model: str = "realesr-animevideov3"
    scale: int = 2
    target_resolution: str = "6K"  # or "8K"

class RunPodClient:
    """RunPod API client for managing pods and jobs"""
    
    API_BASE = "https://api.runpod.io/v2"
    GRAPHQL_URL = "https://api.runpod.io/graphql"
    
    def __init__(self, config: RunPodConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not installed. Run: pip install requests")
    
    def get_available_gpus(self) -> List[Dict]:
        """Query available GPU types and pricing"""
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
        """
        
        response = requests.post(
            self.GRAPHQL_URL,
            headers=self.headers,
            json={"query": query}
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to query GPUs: {response.text}")
        
        data = response.json()
        return data.get("data", {}).get("gpuTypes", [])
    
    def find_best_gpu(self, min_vram: int = 24) -> Optional[Dict]:
        """Find best available GPU for upscaling"""
        gpus = self.get_available_gpus()
        
        # Filter by VRAM and availability
        candidates = []
        for gpu in gpus:
            if gpu.get("memoryInGb", 0) >= min_vram:
                price_info = gpu.get("lowestPrice", {})
                price = price_info.get("uninterruptablePrice") or price_info.get("minimumBidPrice")
                if price and price <= self.config.max_cost_per_job:
                    candidates.append({
                        "id": gpu["id"],
                        "name": gpu["displayName"],
                        "vram": gpu["memoryInGb"],
                        "price": price
                    })
        
        # Sort by price (cheapest first)
        candidates.sort(key=lambda x: x["price"])
        
        # Prefer 4090 if available
        for gpu in candidates:
            if "4090" in gpu["name"]:
                return gpu
        
        return candidates[0] if candidates else None
    
    def create_pod(self, gpu_type_id: str, name: str = "ppp-worker") -> Dict:
        """Create a new pod for processing"""
        query = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }
                }
            }
        }
        """
        
        variables = {
            "input": {
                "cloudType": "SECURE",
                "gpuTypeId": gpu_type_id,
                "name": name,
                "imageName": self.config.container_image,
                "dockerArgs": "",
                "ports": "22/tcp",
                "volumeInGb": 50,
                "containerDiskInGb": 20,
                "startSsh": True,
                "env": [
                    {"key": "JUPYTER_PASSWORD", "value": "ppp123"}
                ]
            }
        }
        
        response = requests.post(
            self.GRAPHQL_URL,
            headers=self.headers,
            json={"query": query, "variables": variables}
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to create pod: {response.text}")
        
        data = response.json()
        
        if "errors" in data:
            raise RuntimeError(f"GraphQL error: {data['errors']}")
        
        return data["data"]["podFindAndDeployOnDemand"]
    
    def get_pod_status(self, pod_id: str) -> Dict:
        """Get pod status"""
        query = """
        query Pod($podId: String!) {
            pod(input: {podId: $podId}) {
                id
                name
                runtime {
                    uptimeInSeconds
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }
                }
                lastStatusChange
            }
        }
        """
        
        response = requests.post(
            self.GRAPHQL_URL,
            headers=self.headers,
            json={"query": query, "variables": {"podId": pod_id}}
        )
        
        return response.json().get("data", {}).get("pod", {})
    
    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod"""
        query = """
        mutation TerminatePod($podId: String!) {
            podTerminate(input: {podId: $podId})
        }
        """
        
        response = requests.post(
            self.GRAPHQL_URL,
            headers=self.headers,
            json={"query": query, "variables": {"podId": pod_id}}
        )
        
        return response.status_code == 200
    
    def wait_for_pod_ready(self, pod_id: str, timeout: int = 300) -> Dict:
        """Wait for pod to be ready"""
        start = time.time()
        
        while time.time() - start < timeout:
            status = self.get_pod_status(pod_id)
            
            runtime = status.get("runtime", {})
            ports = runtime.get("ports", [])
            
            # Check if SSH port is available
            for port in ports:
                if port.get("privatePort") == 22 and port.get("ip"):
                    return {
                        "ssh_ip": port["ip"],
                        "ssh_port": port.get("publicPort", 22)
                    }
            
            print(f"  Waiting for pod... ({int(time.time() - start)}s)")
            time.sleep(10)
        
        raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")


class RunPodWorker:
    """High-level worker for processing jobs on RunPod"""
    
    def __init__(self, config: Optional[RunPodConfig] = None):
        api_key = os.environ.get("RUNPOD_API_KEY", "")
        
        if config:
            self.config = config
        else:
            self.config = RunPodConfig(api_key=api_key)
        
        if not self.config.api_key:
            print("WARNING: RUNPOD_API_KEY not set. Set it with:")
            print("  export RUNPOD_API_KEY='your-api-key'")
        
        self.client = None
        self.pod_id = None
        self.ssh_info = None
        
        self.base_dir = Path(__file__).parent.parent
        self.cost_log = self.base_dir / "logs" / "runpod_costs.json"
        self.cost_log.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_client(self):
        if not self.client and self.config.api_key:
            self.client = RunPodClient(self.config)
    
    def estimate_cost(self, job: CloudJob) -> Dict:
        """Estimate processing cost for a job"""
        self._init_client()
        
        gpu = self.client.find_best_gpu()
        if not gpu:
            return {"error": "No suitable GPU available"}
        
        # Estimate processing time based on target resolution
        # These are rough estimates - actual time varies
        time_estimates = {
            "6K": 1.5,  # hours for 30-min video
            "8K": 3.0,
        }
        
        est_hours = time_estimates.get(job.target_resolution, 2.0)
        est_cost = gpu["price"] * est_hours
        
        return {
            "gpu": gpu["name"],
            "price_per_hour": gpu["price"],
            "estimated_hours": est_hours,
            "estimated_cost": round(est_cost, 2),
            "budget_remaining": self.config.budget_total - self._get_spent()
        }
    
    def _get_spent(self) -> float:
        """Get total spent from cost log"""
        if not self.cost_log.exists():
            return 0.0
        
        with open(self.cost_log) as f:
            data = json.load(f)
        
        return sum(job.get("cost", 0) for job in data.get("jobs", []))
    
    def _log_cost(self, job_name: str, cost: float, duration_hours: float):
        """Log job cost"""
        if self.cost_log.exists():
            with open(self.cost_log) as f:
                data = json.load(f)
        else:
            data = {"jobs": [], "total_budget": self.config.budget_total}
        
        data["jobs"].append({
            "name": job_name,
            "cost": cost,
            "duration_hours": duration_hours,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        with open(self.cost_log, "w") as f:
            json.dump(data, f, indent=2)
    
    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload file to pod via SCP"""
        if not self.ssh_info:
            raise RuntimeError("No pod connected")
        
        cmd = [
            "scp", "-P", str(self.ssh_info["ssh_port"]),
            "-o", "StrictHostKeyChecking=no",
            str(local_path),
            f"root@{self.ssh_info['ssh_ip']}:{remote_path}"
        ]
        
        print(f"Uploading {local_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return result.returncode == 0
    
    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Download file from pod via SCP"""
        if not self.ssh_info:
            raise RuntimeError("No pod connected")
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "scp", "-P", str(self.ssh_info["ssh_port"]),
            "-o", "StrictHostKeyChecking=no",
            f"root@{self.ssh_info['ssh_ip']}:{remote_path}",
            str(local_path)
        ]
        
        print(f"Downloading to {local_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return result.returncode == 0
    
    def run_remote_command(self, command: str) -> str:
        """Execute command on pod via SSH"""
        if not self.ssh_info:
            raise RuntimeError("No pod connected")
        
        cmd = [
            "ssh", "-p", str(self.ssh_info["ssh_port"]),
            "-o", "StrictHostKeyChecking=no",
            f"root@{self.ssh_info['ssh_ip']}",
            command
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout + result.stderr
    
    def setup_pod(self) -> bool:
        """Install Real-ESRGAN on pod"""
        print("Setting up Real-ESRGAN on pod...")
        
        setup_commands = [
            "apt-get update && apt-get install -y ffmpeg wget unzip",
            "pip install torch torchvision --quiet",
            "wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
            "unzip -q realesrgan-ncnn-vulkan*.zip -d /workspace/",
            "chmod +x /workspace/realesrgan-ncnn-vulkan"
        ]
        
        for cmd in setup_commands:
            print(f"  Running: {cmd[:50]}...")
            output = self.run_remote_command(cmd)
            if "error" in output.lower():
                print(f"  Warning: {output[:200]}")
        
        return True
    
    def process_job(self, job: CloudJob) -> bool:
        """Process a single job on RunPod"""
        self._init_client()
        
        # Check budget
        spent = self._get_spent()
        if spent >= self.config.budget_total:
            print(f"Budget exhausted: ${spent:.2f} / ${self.config.budget_total:.2f}")
            return False
        
        # Find GPU
        print("Finding available GPU...")
        gpu = self.client.find_best_gpu()
        if not gpu:
            print("No suitable GPU available")
            return False
        
        print(f"Using {gpu['name']} @ ${gpu['price']:.2f}/hr")
        
        start_time = time.time()
        
        try:
            # Create pod
            print("Creating pod...")
            pod = self.client.create_pod(gpu["id"], f"ppp-{job.local_path.stem[:10]}")
            self.pod_id = pod["id"]
            print(f"Pod created: {self.pod_id}")
            
            # Wait for ready
            print("Waiting for pod to start...")
            self.ssh_info = self.client.wait_for_pod_ready(self.pod_id)
            print(f"Pod ready: {self.ssh_info['ssh_ip']}:{self.ssh_info['ssh_port']}")
            
            # Setup
            self.setup_pod()
            
            # Upload
            remote_input = f"/workspace/input{job.local_path.suffix}"
            if not self.upload_file(job.local_path, remote_input):
                raise RuntimeError("Upload failed")
            
            # Process
            remote_output = "/workspace/output.mp4"
            process_cmd = f"""
            cd /workspace && ./realesrgan-ncnn-vulkan \
                -i {remote_input} -o {remote_output} \
                -n {job.model} -s {job.scale} -f mp4
            """
            
            print("Processing video...")
            output = self.run_remote_command(process_cmd)
            print(output[-500:] if len(output) > 500 else output)
            
            # Download
            if not self.download_file(remote_output, job.output_path):
                raise RuntimeError("Download failed")
            
            # Calculate cost
            duration_hours = (time.time() - start_time) / 3600
            cost = gpu["price"] * duration_hours
            self._log_cost(job.local_path.name, cost, duration_hours)
            
            print(f"\n✓ Completed in {duration_hours:.2f}h, cost: ${cost:.2f}")
            return True
            
        finally:
            # Always terminate pod
            if self.pod_id:
                print("Terminating pod...")
                self.client.terminate_pod(self.pod_id)
                self.pod_id = None
    
    def show_budget(self):
        """Show budget status"""
        spent = self._get_spent()
        remaining = self.config.budget_total - spent
        
        print("\n" + "="*40)
        print("RUNPOD BUDGET STATUS")
        print("="*40)
        print(f"  Total budget: ${self.config.budget_total:.2f}")
        print(f"  Spent:        ${spent:.2f}")
        print(f"  Remaining:    ${remaining:.2f}")
        
        if self.cost_log.exists():
            with open(self.cost_log) as f:
                data = json.load(f)
            
            jobs = data.get("jobs", [])
            if jobs:
                print(f"\n  Recent jobs:")
                for job in jobs[-5:]:
                    print(f"    - {job['name'][:30]}: ${job['cost']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="PPP RunPod Cloud Worker")
    parser.add_argument("input", nargs="?", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("-m", "--model", default="realesr-animevideov3",
                        help="Upscaling model")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor")
    parser.add_argument("--target", default="6K", choices=["6K", "8K"],
                        help="Target resolution")
    parser.add_argument("--estimate", action="store_true",
                        help="Estimate cost without processing")
    parser.add_argument("--budget", action="store_true",
                        help="Show budget status")
    parser.add_argument("--gpus", action="store_true",
                        help="List available GPUs")
    
    args = parser.parse_args()
    
    worker = RunPodWorker()
    
    if args.budget:
        worker.show_budget()
        return
    
    if args.gpus:
        worker._init_client()
        gpus = worker.client.get_available_gpus()
        print("\nAvailable GPUs:")
        for gpu in gpus:
            price = gpu.get("lowestPrice", {})
            p = price.get("uninterruptablePrice") or price.get("minimumBidPrice") or 0
            print(f"  {gpu['displayName']:30s} {gpu['memoryInGb']:3d}GB  ${p:.2f}/hr")
        return
    
    if not args.input:
        parser.print_help()
        print("\nExamples:")
        print("  python runpod_worker.py video.mp4 -o output.mp4 --target 8K")
        print("  python runpod_worker.py video.mp4 --estimate")
        print("  python runpod_worker.py --budget")
        return
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else input_path.with_stem(
        f"{input_path.stem}_cloud_{args.target}"
    )
    
    job = CloudJob(
        local_path=input_path,
        remote_path="/workspace/input.mp4",
        output_path=output_path,
        model=args.model,
        scale=args.scale,
        target_resolution=args.target
    )
    
    if args.estimate:
        estimate = worker.estimate_cost(job)
        print("\nCost Estimate:")
        for k, v in estimate.items():
            print(f"  {k}: {v}")
        return
    
    success = worker.process_job(job)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
