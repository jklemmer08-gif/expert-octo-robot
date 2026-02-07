"""CLI interface for PPP Processor v2.0.

Provides commands: scan, status, queue, process, batch, qa, budget
that call FastAPI endpoints or work directly with the database.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import httpx


DEFAULT_API = "http://localhost:8000"


def _api(base: str, method: str, path: str, **kwargs) -> dict:
    """Make an API request and return JSON."""
    url = f"{base}{path}"
    resp = getattr(httpx, method.lower())(url, timeout=60, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_scan(args):
    """Trigger a library scan."""
    data = _api(args.api, "post", "/library/scan")
    print(f"Scan complete: {data.get('new_jobs', 0)} new jobs created")


def cmd_status(args):
    """Show system status."""
    data = _api(args.api, "get", "/status")
    queue = data.get("queue", {})
    cost = data.get("cost", {})

    print("\nQueue Status")
    print("=" * 40)
    for key in ["pending", "processing", "sampling", "sample_ready",
                "approved", "encoding", "completed", "failed", "skipped"]:
        val = queue.get(key, 0)
        if val > 0:
            print(f"  {key:15s} {val}")
    print(f"  {'total':15s} {queue.get('total', 0)}")

    if queue.get("avg_time_sec"):
        avg_min = queue["avg_time_sec"] / 60
        print(f"\n  Avg processing time: {avg_min:.1f} min")
        pending = queue.get("pending", 0)
        if pending > 0:
            eta_h = (pending * queue["avg_time_sec"]) / 3600
            print(f"  Est. remaining: {eta_h:.1f} hours")

    total_cost = cost.get("total_cost", 0)
    if total_cost > 0:
        print(f"\n  Total cost: ${total_cost:.2f}")


def cmd_queue(args):
    """List jobs in the queue."""
    params = {"limit": args.limit}
    if args.filter:
        params["status"] = args.filter
    data = _api(args.api, "get", "/jobs", params=params)

    if not data:
        print("No jobs found.")
        return

    print(f"\n{'ID':8s}  {'Status':12s}  {'Model':25s}  {'Title'}")
    print("-" * 80)
    for job in data:
        jid = job.get("id", "")[:8]
        status = job.get("status", "")
        model = job.get("model", "") or ""
        title = (job.get("title") or "")[:40]
        print(f"{jid}  {status:12s}  {model:25s}  {title}")

    print(f"\n{len(data)} jobs shown")


def cmd_process(args):
    """Submit a single video for processing."""
    payload = {"source_path": args.input}
    if args.model:
        payload["model"] = args.model
    if args.scale:
        payload["scale"] = args.scale
    if args.vr:
        payload["force_vr"] = True

    data = _api(args.api, "post", "/jobs", json=payload)
    print(f"Job created: {data.get('id')}")
    print(f"  Status: {data.get('status')}")
    print(f"  Model: {data.get('model')}")
    print(f"  Output: {data.get('output_path')}")


def cmd_batch(args):
    """Import jobs from CSV files (offline, no API needed)."""
    from src.config import get_settings
    from src.database import JobDatabase
    from src.watcher import LibraryScanner

    settings = get_settings()
    db_path = Path(settings.paths.temp_dir).parent / "jobs.db"
    db = JobDatabase(db_path)
    scanner = LibraryScanner(settings, db)

    if args.csv:
        count = scanner.import_from_csv(Path(args.csv), args.tier, args.limit)
        print(f"Imported {count} jobs from {args.csv}")
    else:
        count = scanner.import_all_tiers()
        print(f"Imported {count} jobs from all tiers")

    db.close()


def cmd_qa(args):
    """Show pending QA reviews."""
    data = _api(args.api, "get", "/qa/pending")
    if not data:
        print("No pending QA reviews.")
        return

    print(f"\n{'Sample ID':8s}  {'Job':8s}  {'SSIM':6s}  {'PSNR':6s}  {'Title'}")
    print("-" * 70)
    for s in data:
        sid = (s.get("id") or "")[:8]
        jid = (s.get("job_id") or "")[:8]
        ssim = f"{s['ssim']:.3f}" if s.get("ssim") is not None else "N/A"
        psnr = f"{s['psnr']:.1f}" if s.get("psnr") is not None else "N/A"
        title = (s.get("title") or "")[:35]
        print(f"{sid}  {jid}  {ssim:6s}  {psnr:6s}  {title}")

    print(f"\n{len(data)} pending reviews")


def cmd_budget(args):
    """Show RunPod budget status."""
    data = _api(args.api, "get", "/status")
    cost = data.get("cost", {})

    print("\nRunPod Budget")
    print("=" * 40)
    total = cost.get("total_cost", 0)
    print(f"  Spent:     ${total:.2f}")
    print(f"  Remaining: ${75.0 - total:.2f}")

    by_gpu = cost.get("by_gpu", {})
    if by_gpu:
        print("\n  By GPU:")
        for gpu, info in by_gpu.items():
            print(f"    {gpu}: ${info['cost']:.2f} ({info['jobs']} jobs, {info['hours']:.1f}h)")


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="ppp",
        description="PPP Processor v2.0 CLI",
    )
    parser.add_argument("--api", default=DEFAULT_API,
                        help="API base URL (default: %(default)s)")

    sub = parser.add_subparsers(dest="command")

    # scan
    sub.add_parser("scan", help="Trigger library scan")

    # status
    sub.add_parser("status", help="Show system status")

    # queue
    q = sub.add_parser("queue", help="List jobs")
    q.add_argument("--filter", help="Filter by status")
    q.add_argument("--limit", type=int, default=50)

    # process
    p = sub.add_parser("process", help="Submit single video")
    p.add_argument("input", help="Input video path")
    p.add_argument("-m", "--model", help="Model name")
    p.add_argument("-s", "--scale", type=int, help="Scale factor")
    p.add_argument("--vr", action="store_true", help="Force VR mode")

    # batch
    b = sub.add_parser("batch", help="Import batch jobs from CSV")
    b.add_argument("--csv", help="CSV file path")
    b.add_argument("--tier", default="tier3", choices=["tier1", "tier2", "tier3"])
    b.add_argument("--limit", type=int)

    # qa
    sub.add_parser("qa", help="Show pending QA reviews")

    # budget
    sub.add_parser("budget", help="Show RunPod budget")

    args = parser.parse_args()

    commands = {
        "scan": cmd_scan,
        "status": cmd_status,
        "queue": cmd_queue,
        "process": cmd_process,
        "batch": cmd_batch,
        "qa": cmd_qa,
        "budget": cmd_budget,
    }

    if args.command in commands:
        try:
            commands[args.command](args)
        except httpx.ConnectError:
            print(f"Error: Cannot connect to API at {args.api}")
            print("Start the server: python -m src.main")
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"API error: {e.response.status_code} {e.response.text}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
