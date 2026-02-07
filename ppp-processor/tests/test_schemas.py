"""Tests for Pydantic model validation."""

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    ContentType,
    JobCreate,
    JobResponse,
    JobStatus,
    ProcessingPlan,
    QASample,
    VideoInfo,
    VRMetadata,
    WorkerStatus,
)


def test_video_info_creation():
    info = VideoInfo(
        width=3840, height=1920, fps=30.0, duration=120.0,
        codec="hevc", bitrate=50000000, is_vr=True, vr_type="sbs",
    )
    assert info.width == 3840
    assert info.is_vr is True
    assert info.content_type == ContentType.FLAT_2D  # default


def test_video_info_content_type():
    info = VideoInfo(
        width=3840, height=1920, fps=30.0, duration=120.0,
        codec="hevc", bitrate=50000000, content_type=ContentType.VR_SBS,
    )
    assert info.content_type == ContentType.VR_SBS


def test_job_create_minimal():
    job = JobCreate(source_path="/path/to/video.mp4")
    assert job.source_path == "/path/to/video.mp4"
    assert job.tier == "tier3"
    assert job.priority == 0


def test_job_create_full():
    job = JobCreate(
        source_path="/path/video.mp4", model="realesrgan-x4plus",
        scale=4, force_vr=True, tier="tier1", priority=100,
    )
    assert job.model == "realesrgan-x4plus"
    assert job.force_vr is True


def test_job_status_enum():
    assert JobStatus.PENDING == "pending"
    assert JobStatus.SAMPLING == "sampling"
    assert JobStatus.APPROVED == "approved"
    assert JobStatus.ENCODING == "encoding"


def test_job_response():
    r = JobResponse(id="abc", source_path="/path/video.mp4")
    assert r.status == JobStatus.PENDING
    assert r.progress is None


def test_vr_metadata_suffix():
    meta = VRMetadata(is_vr=True, fov_horizontal=180, stereo_mode="sbs")
    assert meta.to_filename_suffix() == "_180_sbs"


def test_vr_metadata_suffix_360():
    meta = VRMetadata(is_vr=True, fov_horizontal=360, stereo_mode="tb")
    assert meta.to_filename_suffix() == "_360_tb"


def test_vr_metadata_suffix_not_vr():
    meta = VRMetadata(is_vr=False)
    assert meta.to_filename_suffix() == ""


def test_qa_sample():
    s = QASample(job_id="j1", ssim=0.92, psnr=32.5, sharpness=150.0)
    assert s.ssim == 0.92
    assert s.auto_approved is None


def test_processing_plan():
    plan = ProcessingPlan(
        model="realesr-animevideov3", scale=2,
        worker_type="local", bitrate="100M",
    )
    assert plan.skip_ai is False
    assert plan.tile_size == 512


def test_worker_status():
    ws = WorkerStatus(id="w1", worker_type="local_gpu")
    assert ws.status == "idle"
    assert ws.jobs_completed == 0


def test_content_type_enum():
    assert ContentType.VR_SBS == "vr_sbs"
    assert ContentType.FLAT_2D == "flat_2d"
