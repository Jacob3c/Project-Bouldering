"""HTTP API: upload a climbing clip, get metrics and a feedback video back.

Start it from the repo root (it reads and writes data/ relative to there):
    uvicorn boulder.api.app:app --reload
Then open http://127.0.0.1:8000/docs to try every endpoint in the browser.

Endpoints:
    GET  /health                    is the server up
    POST /clips                     upload a video + its facts; starts processing, returns a job
    GET  /jobs/{job_id}             job status: queued, running (with current step), done, failed
    GET  /clips                     every ingested clip
    GET  /clips/{clip_id}           one clip: facts, quality, metrics, feedback events
    GET  /clips/{clip_id}/video     the feedback video (mp4)

An upload goes through the same gate as the command line: the facts become a canonical
file name, `boulder.ingest` checks and records it, and the climb window and wall style are
added to metadata/attempts.csv. Processing then runs in the background, one clip at a time.

Limits (v1, local use): jobs live in memory and are lost on restart; no accounts or auth;
one pipeline runs at a time because pose estimation uses all the CPU.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from boulder.ingest import run as ingest_run
from boulder.ingest.attempts import LOG_COLUMNS, LOG_PATH
from boulder.ingest.clip_name import parse_clip_name
from boulder.pipeline import clip as pipeline

UPLOAD_DIR = Path("data/uploads")
METRIC_ROOT = Path("data/processed/metrics")
QUALITY_ROOT = Path("data/processed/quality")
FEEDBACK_ROOT = Path("data/processed/feedback")
OUTPUT_DIR = Path("data/output")
MODEL_LABEL = pipeline.MODEL_LABEL

app = FastAPI(title="Project Boulder API", version="0.1.0",
              description="Upload a bouldering clip; get pose-based metrics and a feedback video.")

JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()       # guards JOBS
_PIPELINE_LOCK = threading.Lock()   # one pipeline at a time


# ---------------------------------------------------------------- helpers

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe dicts (NaN becomes null, numpy types become plain ones)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _read_one(root: Path, clip_id: str) -> pd.DataFrame | None:
    matches = sorted(root.glob(f"camera=*/model={MODEL_LABEL}/{clip_id}.parquet"))
    return pd.read_parquet(matches[0]) if matches else None


def _update_job(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        JOBS[job_id].update(fields, updated_at=_now())


def build_file_name(date: str, gym: str, colour: str, grade: str, attempt: int,
                    outcome: str, camera: str, suffix: str) -> str:
    """The clip's facts as a file name in the ingest pattern."""
    return f"{date}_{gym}_{colour}-{grade}_att{attempt:02d}_{outcome}_{camera}{suffix.lower()}"


def add_to_attempts_log(file_name: str, climb_start_s: float | None, climb_end_s: float | None,
                        wall_style: str | None, wall_sections: str | None, fall_move: str | None,
                        notes: str | None, path: Path = LOG_PATH) -> bool:
    """Append a row to the attempts log unless the clip already has one. True if added."""
    clip_id = parse_clip_name(file_name).canonical_id
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if parse_clip_name(row.get("file_name", "")).canonical_id == clip_id:
                    return False
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [file_name, climb_start_s, climb_end_s, wall_style, wall_sections, fall_move, notes]
    with path.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if new_file:
            writer.writerow(LOG_COLUMNS)
        writer.writerow(["" if v is None else v for v in values])
    return True


def run_job(job_id: str, raw_path: Path, start_s: float | None, end_s: float | None) -> None:
    """Background task: run the pipeline and record progress on the job."""
    with _PIPELINE_LOCK:
        _update_job(job_id, status="running")
        try:
            result = pipeline.process_clip(raw_path, start_s, end_s,
                                           on_step=lambda name: _update_job(job_id, step=name))
        except Exception as err:  # report any failure on the job instead of losing it
            _update_job(job_id, status="failed", error=f"{type(err).__name__}: {err}")
            return
        _update_job(job_id, status="done", step=None,
                    red_share=round(result["red_share"], 3), events=len(result["events"]))


# ---------------------------------------------------------------- endpoints

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/clips", status_code=202)
def upload_clip(
    background: BackgroundTasks,
    video: UploadFile = File(..., description="the climbing video (.mov, .mp4)"),
    date: str = Form(..., description="YYYY-MM-DD", examples=["2026-11-23"]),
    gym: str = Form(..., examples=["MainRec"]),
    colour: str = Form(..., examples=["Orange"]),
    grade: str = Form(..., description="V-grade, e.g. V3 or V5-6", examples=["V5-6"]),
    outcome: Literal["send", "fall"] = Form(...),
    camera: Literal["handheld", "tripod"] = Form(...),
    attempt: int = Form(1, ge=1),
    climb_start_s: float | None = Form(None, ge=0),
    climb_end_s: float | None = Form(None, ge=0),
    wall_style: str | None = Form(None, examples=["vertical"]),
    wall_sections: str | None = Form(None, examples=["vertical>overhang"]),
    fall_move: str | None = Form(None, description="move number of the fall; blank on a send"),
    notes: str | None = Form(None),
) -> dict:
    """Upload a clip. Returns 202 with a job id; poll GET /jobs/{job_id} for progress."""
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ingest_run.VIDEO_EXTS:
        raise HTTPException(422, f"video must be one of {sorted(ingest_run.VIDEO_EXTS)}")
    if climb_start_s is not None and climb_end_s is not None and climb_start_s >= climb_end_s:
        raise HTTPException(422, "climb_start_s must be before climb_end_s")

    file_name = build_file_name(date, gym, colour, grade, attempt, outcome, camera, suffix)
    info = parse_clip_name(file_name)
    if not info.name_ok or info.issues or info.canonical_id is None:
        raise HTTPException(422, f"bad clip facts: {info.issues or file_name}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / file_name
    with upload_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    result = ingest_run.ingest([upload_path])[0]
    if result.status == ingest_run.CONFLICT:
        raise HTTPException(409, f"{result.clip_id}: {result.detail}")
    if result.status == ingest_run.REJECTED:
        raise HTTPException(422, f"rejected: {result.detail}")
    if result.status == ingest_run.DUPLICATE:
        # same video already ingested under another id: process that one
        clip_id = result.detail.removeprefix("same video as ").strip()
    else:
        clip_id = result.clip_id

    manifest = ingest_run.load_manifest()
    raw_path = Path(manifest.loc[manifest["clip_id"] == clip_id, "raw_path"].iloc[0])
    logged = add_to_attempts_log(file_name, climb_start_s, climb_end_s, wall_style,
                                 wall_sections, fall_move, notes)

    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        JOBS[job_id] = {"job_id": job_id, "clip_id": clip_id, "status": "queued", "step": None,
                        "error": None, "created_at": _now(), "updated_at": _now()}
    background.add_task(run_job, job_id, raw_path, climb_start_s, climb_end_s)
    return {"job_id": job_id, "clip_id": clip_id, "ingest": result.status,
            "added_to_attempts_log": logged}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job (jobs are lost when the server restarts)")
        return dict(job)


@app.get("/clips")
def list_clips() -> list[dict]:
    manifest = ingest_run.load_manifest()
    if manifest.empty:
        return []
    cols = ["clip_id", "date", "gym", "colour", "grade", "attempt", "outcome", "camera",
            "duration_s"]
    out = manifest.reindex(columns=cols)
    out["has_feedback"] = [(OUTPUT_DIR / f"{c}_feedback.mp4").exists() for c in out["clip_id"]]
    return _records(out.sort_values("date", ascending=False))


@app.get("/clips/{clip_id}")
def get_clip(clip_id: str) -> dict:
    manifest = ingest_run.load_manifest()
    row = manifest[manifest["clip_id"] == clip_id] if not manifest.empty else manifest
    if row.empty:
        raise HTTPException(404, f"no clip {clip_id}")
    attempt = _read_one(METRIC_ROOT / "attempts", clip_id)
    quality = _read_one(QUALITY_ROOT / "clips", clip_id)
    events = _read_one(FEEDBACK_ROOT, clip_id)
    return {
        "clip": _records(row.drop(columns=["sha256"], errors="ignore"))[0],
        "quality": _records(quality)[0] if quality is not None else None,
        "metrics": _records(attempt.drop(columns=["metric_config"], errors="ignore"))[0]
        if attempt is not None else None,
        "events": _records(events.drop(columns=["video_id"])) if events is not None else [],
        "video_url": f"/clips/{clip_id}/video"
        if (OUTPUT_DIR / f"{clip_id}_feedback.mp4").exists() else None,
    }


@app.get("/clips/{clip_id}/video")
def get_video(clip_id: str) -> FileResponse:
    path = OUTPUT_DIR / f"{clip_id}_feedback.mp4"
    if not path.exists():
        raise HTTPException(404, "no feedback video yet")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
