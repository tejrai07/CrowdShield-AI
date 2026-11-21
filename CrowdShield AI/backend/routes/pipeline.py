"""Pipeline management, ML worker control & transparency routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.ml_worker import worker
from backend.schemas import DiversionRecommendation, PipelineStatusResponse
from backend.state import state

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

router = APIRouter(prefix="/pipeline", tags=["Pipeline & Transparency"])


class StartVideoRequest(BaseModel):
    video_path: str = Field(default="sample_data/test_video.mp4", description="Path to video file")
    venue_yaml: str | None = Field(default=None, description="Optional path to venue YAML")
    max_frames: int | None = Field(default=None, description="Maximum frames to process (None for unlimited)")
    loop: bool = Field(default=True, description="Loop video continuously")
    fps: float = Field(default=5.0, description="Target processing/sampling FPS")


class StartReplayRequest(BaseModel):
    jsonl_path: str = Field(default="sample_data/output/demo_pipeline.jsonl", description="Path to JSONL stream file")
    loop: bool = Field(default=True, description="Loop stream continuously")
    fps: float = Field(default=2.0, description="Target frame playback FPS")


@router.get("/status", response_model=PipelineStatusResponse)
def get_pipeline_status() -> PipelineStatusResponse:
    """Current pipeline health, model data notices (synthetic data limitation), and warmup status."""
    return state.get_pipeline_status()


@router.get("/diversions", response_model=list[DiversionRecommendation])
def get_diversion_recommendations() -> list[DiversionRecommendation]:
    """Automated diversion candidates and capacity recommendations."""
    return state.get_diversion_recommendations()


@router.post("/frame")
def ingest_pipeline_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Ingest a validated L1 frame dict directly from the ML pipeline worker or external runner."""
    state.update_frame(frame)
    return {
        "status": "ingested",
        "frame_id": frame.get("frame_id"),
        "timestamp_s": frame.get("timestamp_s"),
    }


@router.get("/worker")
def get_worker_status() -> dict[str, Any]:
    """Get active background ML worker status (running, FPS, frames processed, error)."""
    return worker.get_status()


@router.post("/start")
def start_pipeline_worker(req: StartVideoRequest) -> dict[str, Any]:
    """Start background video inference using InferencePipeline."""
    if worker.get_status()["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="A pipeline or replay session is already active. Call /api/pipeline/stop before starting a new session.",
        )
    vid_path = Path(req.video_path)
    if not vid_path.is_absolute():
        vid_path = _REPO_ROOT / vid_path

    if not vid_path.is_file():
        raise HTTPException(status_code=400, detail=f"Video file not found at: {vid_path}")

    venue_path = None
    if req.venue_yaml:
        p = Path(req.venue_yaml)
        venue_path = p if p.is_absolute() else _REPO_ROOT / p

    ok = worker.start_video(
        video_path=vid_path,
        venue_yaml=venue_path,
        max_frames=req.max_frames,
        loop=req.loop,
        fps=req.fps,
    )
    if not ok:
        raise HTTPException(status_code=500, detail=worker.last_error or "Failed to start video pipeline")
    return {"message": "Pipeline worker started", **worker.get_status()}


@router.post("/replay")
def start_replay_worker(req: StartReplayRequest) -> dict[str, Any]:
    """Start streaming pre-computed or demo JSONL frames into the dashboard state."""
    if worker.get_status()["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="A pipeline or replay session is already active. Call /api/pipeline/stop before starting a new session.",
        )
    jsonl_path = Path(req.jsonl_path)
    if not jsonl_path.is_absolute():
        jsonl_path = _REPO_ROOT / jsonl_path

    if not jsonl_path.is_file():
        raise HTTPException(status_code=400, detail=f"JSONL file not found at: {jsonl_path}")

    ok = worker.start_replay(jsonl_path=jsonl_path, loop=req.loop, fps=req.fps)
    if not ok:
        raise HTTPException(status_code=500, detail=worker.last_error or "Failed to start replay")
    return {"message": "Replay worker started", **worker.get_status()}


@router.post("/stop")
def stop_pipeline_worker() -> dict[str, Any]:
    """Stop active background ML inference or replay worker."""
    worker.stop()
    return {"message": "Worker stopped", **worker.get_status()}
