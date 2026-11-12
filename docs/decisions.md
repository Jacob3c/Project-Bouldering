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
- ~~MediaPipe reports eyes/nose visible in 100% of frames while the climber faces the wall: its confidence looks unreliable for filtering.~~ Corrected 2026-11-11: RTMPose-feet is equally confident on the face, so 17-joint RTMPose was the outlier; the face is likely visible when the climber looks at holds.
- RTMPose about 5x slower (about 4 min vs 44.5 s); acceptable for batch processing.
- **Leaning:** RTMPose with feet (26 joints). **Next:** check overlays by eye and repeat on 3-5 clips before deciding.

## 2026-11-11 - Default pose model: RTMPose with feet (26 joints)
- **Decision:** RTMPose `body_with_feet` (Halpe-26, via rtmlib, `balanced` mode) is the default for `boulder.pose.run`; MediaPipe stays available for comparison.
- **Evidence:** 5 handheld clips, 2,586 frames, share of frames detected with confidence >= 0.5:

| Joint | MediaPipe | RTMPose-feet |
| --- | --- | --- |
| right_elbow | 0.58 | 0.92 |
| right_wrist | 0.55 | 0.80 |
| left_elbow | 0.72 | 0.95 |
| left_wrist | 0.73 | 0.86 |
| right_knee | 0.82 | 0.96 |
| left_heel / right_heel | 0.93 / 0.89 | 0.91 / 0.93 |

- Limb positions agree within 5.5-10.5 px (median), so RTMPose-feet is confident far more often without placing joints differently. Arms drive the main metric (straight vs bent arms), which decides it.
- Heels are on par with MediaPipe, so nothing is lost on footwork; 26 joints also add toes.
- MediaPipe lost the climber in about 24 frames on one clip where RTMPose-feet did not.
- **Cost:** about 4-6x slower (108-204 s vs 23-41 s per clip at stride 2 on an M-series MacBook Air CPU); acceptable for batch processing.
- **Caveats:** handheld footage only; confidence scales differ between models. **Revisit** with tripod clips and after checking overlays by eye.

## 2026-11-11 - First metric definitions
- **Decision:** scale all distances by the clip's median torso length (shoulder midpoint to hip midpoint); compute straight-arm share, foot moves/readjustments, pauses, hip-path efficiency and jerk from smoothed keypoints.
- **Why:** angles are distance-independent and torso length cancels camera distance, so handheld and tripod clips at different distances can be compared.
- **Known limits:** angles are 2D projections (an arm pointing at the camera looks bent); straight-arm share counts all frames, not only frames on a hold; thresholds are first guesses.
- **Next:** calibrate thresholds by comparing sends and falls on the same problems with tripod footage; store each run's `MetricConfig` with the results so reruns are traceable.

## 2026-11-11 - Ingest stage and manifest
- **Decision:** `boulder.ingest.run` is the single entry point for clips. It normalizes the name (date, gym, colour, grade), copies the clip to `data/raw/<canonical_id><ext>` and appends a row to `data/manifest.parquet`. Pose and later stages read only from `data/raw/`.
- **Why:** the same clip was processed twice (`2026-1-19_...` and `2026-01-19_...`). Cleaning names once at the boundary means downstream stages need no changes and can trust their input.
- **Rules:** bad or unknown names are rejected (not copied); a SHA-256 match with the manifest marks a duplicate; the same ID with different bytes is an `id_conflict`; reruns are no-ops. Originals are copied, never moved. The manifest is written to a temp file and renamed, so a crash cannot corrupt it.
- **Considered:** fixing IDs inside each stage (more code to keep in step); moving files instead of copying (risks losing originals while the pipeline is young).
- **Next:** join the attempts log (`metadata/`) for wall angle, wall style and fall move; rerun pose on `data/raw/` and drop the duplicate keypoint file.
