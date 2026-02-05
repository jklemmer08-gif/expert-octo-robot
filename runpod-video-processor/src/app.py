"""Flask web UI for video processing — file browser, job control, progress via SSE."""

import logging
import os
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_CRF,
    DEFAULT_MODEL,
    DEFAULT_SCALE,
    INPUT_DIR,
    OUTPUT_DIR,
    WEB_PORT,
)
from src.gpu import detect_gpu, get_gpu_profile, get_vram_usage
from src.pipeline.detector import VRLayout, detect_layout
from src.pipeline.upscaler import process_video
from src.pipeline.validator import validate_input
from src.storage.volume import (
    cleanup_orphaned_temp,
    ensure_dirs,
    get_disk_space,
    list_input_files,
    list_output_files,
)
from src.utils.ffmpeg import get_video_metadata
from src.utils.logging import setup_logging
from src.utils.progress import progress_manager

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

# Global state for current job
_current_job_lock = threading.Lock()
_current_job_id = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/files/input")
def api_input_files():
    """List input files with video metadata."""
    files = list_input_files()
    for f in files:
        try:
            meta = get_video_metadata(f["path"])
            f["width"] = meta["width"]
            f["height"] = meta["height"]
            f["duration"] = meta["duration"]
            f["fps"] = meta["fps"]
            f["codec"] = meta["codec"]
            f["num_frames"] = meta["num_frames"]
            layout = detect_layout(f["path"], meta["width"], meta["height"])
            f["layout"] = layout.value
        except Exception as e:
            f["error"] = str(e)
            f["layout"] = "unknown"
    return jsonify(files)


@app.route("/api/files/output")
def api_output_files():
    """List output files."""
    files = list_output_files()
    for f in files:
        try:
            meta = get_video_metadata(f["path"])
            f["width"] = meta["width"]
            f["height"] = meta["height"]
            f["duration"] = meta["duration"]
        except Exception:
            pass
    return jsonify(files)


@app.route("/api/system")
def api_system():
    """System info — GPU, VRAM, disk space, active profile."""
    gpu_info = detect_gpu()
    profile = get_gpu_profile()
    vram = get_vram_usage()
    disk = get_disk_space()
    return jsonify({
        "gpu": gpu_info,
        "profile": {
            "name": profile.name,
            "tile_size": profile.tile_size,
            "batch_size": profile.batch_size,
            "segment_size": profile.segment_size,
        },
        "vram": vram,
        "disk": disk,
        "models": {k: v["description"] for k, v in AVAILABLE_MODELS.items()},
    })


@app.route("/api/validate", methods=["POST"])
def api_validate():
    """Validate an input file before processing."""
    data = request.get_json()
    path = data.get("path", "")
    result = validate_input(path)
    return jsonify({
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "metadata": result.metadata,
    })


@app.route("/api/process", methods=["POST"])
def api_process():
    """Start a processing job (runs in background thread)."""
    global _current_job_id

    with _current_job_lock:
        if _current_job_id is not None:
            return jsonify({"error": "A job is already running"}), 409

    data = request.get_json()
    input_path = data.get("input_path", "")
    model_name = data.get("model", DEFAULT_MODEL)
    scale = int(data.get("scale", DEFAULT_SCALE))
    crf = int(data.get("crf", DEFAULT_CRF))
    layout_override = data.get("layout", "auto")

    # Validate
    val = validate_input(input_path)
    if not val.valid:
        return jsonify({"error": "Validation failed", "errors": val.errors}), 400

    meta = val.metadata

    # Detect layout
    if layout_override == "auto":
        layout = detect_layout(input_path, meta["width"], meta["height"])
    else:
        layout = VRLayout(layout_override)

    # Generate output path
    input_name = Path(input_path).stem
    output_name = f"{input_name}_upscaled_{scale}x.mkv"
    output_path = str(OUTPUT_DIR / output_name)

    # Get profile
    profile = get_gpu_profile()

    job_id = str(uuid.uuid4())[:8]
    progress_manager.create_job(job_id, total_frames=meta.get("num_frames", 0))

    def progress_callback(update):
        progress_manager.update(job_id, update)

    def run_job():
        global _current_job_id
        with _current_job_lock:
            _current_job_id = job_id

        try:
            result = process_video(
                input_path=input_path,
                output_path=output_path,
                model_name=model_name,
                scale=scale,
                tile_size=profile.tile_size,
                crf=crf,
                layout=layout,
                segment_size=profile.segment_size,
                progress_callback=progress_callback,
            )

            if result["status"] == "success":
                progress_manager.update(job_id, {"stage": "completed", "frame": meta.get("num_frames", 0)})
            else:
                progress_manager.update(job_id, {"stage": "failed", "error": result.get("error", "Unknown error")})

        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e, exc_info=True)
            progress_manager.update(job_id, {"stage": "failed", "error": str(e)})
        finally:
            with _current_job_lock:
                _current_job_id = None

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()

    return jsonify({
        "job_id": job_id,
        "output_path": output_path,
        "layout": layout.value,
        "model": model_name,
        "scale": scale,
        "crf": crf,
    })


@app.route("/api/progress")
def api_progress():
    """SSE endpoint for real-time progress updates."""
    return Response(
        progress_manager.subscribe(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/progress/current")
def api_progress_current():
    """Get current job progress as JSON (polling fallback)."""
    current = progress_manager.get_current_job()
    if current:
        return jsonify(current)
    return jsonify({"stage": "idle"})


def create_app():
    """Application factory."""
    setup_logging()
    ensure_dirs()
    cleanup_orphaned_temp()

    gpu_info = detect_gpu()
    profile = get_gpu_profile()
    logger.info("App started — GPU: %s, Profile: %s", gpu_info.get("name", "none"), profile.name)

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=WEB_PORT, debug=False, threaded=True)
