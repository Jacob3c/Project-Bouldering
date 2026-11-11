# Decision log

Short entries: what was decided, why, and what else was considered.

## 2026-11-10 - Project structure
- **Decision:** `src/` layout with one module per pipeline stage (ingest, pose, transforms, quality, render, pipeline).
- **Why:** each stage can be tested and rerun on its own; orchestration only wires them together.

## 2026-11-10 - Two pose models behind one interface
- **Decision:** wrap MediaPipe Pose Landmarker and RTMPose (via rtmlib) behind a shared `PoseBackend` interface with one output schema; store `model_name` and `model_version` on every row.
- **Why:** lets the models be compared on the same clips and the pipeline be rerun when the model changes, without touching downstream code.
- **Considered:** YOLO-pose (ankle-only feet, AGPL licence).
- **Open:** pick the default model after comparing both on test clips.

## 2026-11-10 - First model comparison (1 handheld clip, 668 frames)
- Both models found the climber in every frame; most joints agree within 5-10 px on a 1080x1920 frame.
- RTMPose confident on elbows/wrists far more often (e.g. left elbow 96% vs 60%), with positions still close, so MediaPipe is often right but unsure.
- MediaPipe reports eyes/nose visible in 100% of frames while the climber faces the wall: its confidence looks unreliable for filtering.
- RTMPose about 5x slower (about 4 min vs 44.5 s); acceptable for batch processing.
- **Leaning:** RTMPose with feet (26 joints). **Next:** check overlays by eye and repeat on 3-5 clips before deciding.
