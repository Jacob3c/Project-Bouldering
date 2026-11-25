"""process_clip calls every stage in order with the right inputs (stages faked)."""

from pathlib import Path

import pandas as pd

from boulder.pipeline import clip as pipeline

RAW = Path("data/raw/2026-11-23_MainRec_Orange-V5-6_att01_send_handheld.mov")
CLIP_ID = RAW.stem


def _fake_stages(monkeypatch, tmp_path, keypoints_exist):
    import boulder.pose.run as pose_run
    import boulder.quality.run as quality_run
    import boulder.render.feedback as feedback
    import boulder.transforms.run as transforms_run

    seen = {}
    kp = tmp_path / "kp.parquet"
    if keypoints_exist:
        kp.write_bytes(b"")
    monkeypatch.setattr(pose_run, "output_path", lambda raw, label: kp)
    monkeypatch.setattr(pose_run, "run", lambda raw, model, **kw: seen.setdefault("pose", kw))
    monkeypatch.setattr(quality_run, "check_file", lambda p: {"status": "warn"})

    def process_file(kp_path, root, cfg, windows=None, **kw):
        seen["windows"] = windows
        return {"foot_moves": 3}

    monkeypatch.setattr(transforms_run, "process_file", process_file)
    monkeypatch.setattr(feedback, "make_feedback", lambda raw, model: {
        "video": Path("out.mp4"), "events": pd.DataFrame(), "red_share": 0.1})
    return seen


def test_runs_every_step_in_order(monkeypatch, tmp_path):
    seen = _fake_stages(monkeypatch, tmp_path, keypoints_exist=False)
    steps = []
    result = pipeline.process_clip(RAW, 5.0, 49.0, on_step=steps.append)
    assert steps == pipeline.STEPS
    assert seen["pose"] == {"skeleton": "body_with_feet"}
    assert seen["windows"] == {CLIP_ID: (5.0, 49.0)}
    assert result["clip_id"] == CLIP_ID and result["quality"]["status"] == "warn"
    assert result["red_share"] == 0.1


def test_existing_keypoints_skip_pose(monkeypatch, tmp_path):
    seen = _fake_stages(monkeypatch, tmp_path, keypoints_exist=True)
    pipeline.process_clip(RAW)
    assert "pose" not in seen
