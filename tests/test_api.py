"""API tests: real ingest and attempts log in a temp folder, pipeline replaced by a fake."""

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
try:  # FastAPI's TestClient needs httpx2 (newer Starlette) or httpx (older)
    import httpx2  # noqa: F401
except ModuleNotFoundError:
    pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from boulder.api import app as api  # noqa: E402

FORM = {"date": "2026-11-23", "gym": "MainRec", "colour": "Orange", "grade": "V5-6",
        "outcome": "send", "camera": "handheld", "climb_start_s": "5", "climb_end_s": "49",
        "wall_style": "dihedral"}
CLIP_ID = "2026-11-23_MainRec_Orange-V5-6_att01_send_handheld"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # every data/ and metadata/ path lands in the temp folder
    calls = []

    def fake_process(raw_path, start, end, on_step=None, **kw):
        for step in ("pose", "quality", "metrics", "feedback"):
            on_step(step)
        calls.append((raw_path, start, end))
        return {"red_share": 0.25, "events": pd.DataFrame({"flag": ["left_arm_bent"]})}

    monkeypatch.setattr(api.pipeline, "process_clip", fake_process)
    api.JOBS.clear()
    c = TestClient(api.app)
    c.calls = calls
    return c


def _video_bytes(tmp_path, seed=0):
    path = tmp_path / f"v{seed}.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 24))
    for i in range(5):
        writer.write(np.full((24, 32, 3), (seed * 40 + i) % 255, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _upload(client, data, content, name="IMG_0001.avi"):
    return client.post("/clips", data=data, files={"video": (name, content, "video/x-msvideo")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_ingests_logs_and_runs_job(client, tmp_path):
    r = _upload(client, FORM, _video_bytes(tmp_path))
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["clip_id"] == CLIP_ID and body["ingest"] == "ingested"
    assert body["added_to_attempts_log"] is True

    assert (tmp_path / "data/raw" / f"{CLIP_ID}.avi").exists()
    log = pd.read_csv(tmp_path / "metadata/attempts.csv", dtype=str)
    assert log.loc[0, "climb_start_s"] == "5.0" and log.loc[0, "wall_style"] == "dihedral"

    job = client.get(f"/jobs/{body['job_id']}").json()  # background task already ran
    assert job["status"] == "done" and job["red_share"] == 0.25
    assert client.calls[0][1:] == (5.0, 49.0)

    clips = client.get("/clips").json()
    assert [c["clip_id"] for c in clips] == [CLIP_ID]
    detail = client.get(f"/clips/{CLIP_ID}").json()
    assert detail["clip"]["grade"] == "V5-6" and detail["metrics"] is None
    assert client.get(f"/clips/{CLIP_ID}/video").status_code == 404


def test_same_upload_twice_is_not_logged_twice(client, tmp_path):
    content = _video_bytes(tmp_path)
    _upload(client, FORM, content)
    r = _upload(client, FORM, content)
    assert r.json()["ingest"] == "already"
    assert r.json()["added_to_attempts_log"] is False
    assert len(pd.read_csv(tmp_path / "metadata/attempts.csv")) == 1


def test_bad_facts_rejected(client, tmp_path):
    r = _upload(client, {**FORM, "gym": "NoSuchGym"}, _video_bytes(tmp_path))
    assert r.status_code == 422 and "unknown gym" in r.text
    r = _upload(client, {**FORM, "climb_start_s": "50"}, _video_bytes(tmp_path))
    assert r.status_code == 422
    r = _upload(client, FORM, b"not a video", name="notes.txt")
    assert r.status_code == 422


def test_failed_pipeline_reported_on_job(client, tmp_path, monkeypatch):
    def broken(*a, **kw):
        raise FileNotFoundError("no keypoints")

    monkeypatch.setattr(api.pipeline, "process_clip", broken)
    job_id = _upload(client, FORM, _video_bytes(tmp_path)).json()["job_id"]
    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "failed" and "no keypoints" in job["error"]


def test_unknown_job_and_clip(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/clips/nope").status_code == 404
