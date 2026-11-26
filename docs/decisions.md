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

## 2026-11-12 - Attempts log: only hand-entered facts, no wall angle
- **Decision:** `metadata/attempts.csv` holds only what can't be read from the file name or video: `file_name`, `wall_style`, `wall_sections`, `fall_move`, `notes`. `boulder.ingest.attempts` normalizes it and left-joins it onto the manifest as `data/clips.parquet`.
- **Why slim:** the old template repeated date, gym, grade, attempt, outcome and camera, which already live in the file name. Two copies of a fact eventually disagree; each fact now has one source.
- **Why no wall angle:** gym boulders change angle by section, so one number per clip would be false precision. `wall_style` (dominant) plus `wall_sections` (order climbed, e.g. `vertical>overhang>slab`) records what can be observed reliably.
- **Considered:** a per-move table (most precise, too slow to fill for now).
- **Next:** estimate wall angle per section from the multi-angle 3D reconstruction in v2.

## 2026-11-12 - Quality checks and their thresholds
- **Evidence** (`boulder.quality.profile`, 5 handheld clips, rtmpose-feet): detection gaps negligible (longest 1 processed frame); MainRec Orange V3-4 torso only 62 px vs 165-241 px for the others; ABP has 2.4% of points out of frame; normal frame-to-frame steps reach 0.6-0.9 torso lengths at the 99th percentile (handheld camera motion adds to every joint); on SouthSide frames 698 and 726 every joint jumped 3.6-6.9 TL at once.
- **Cause of the whole-body jumps:** checked in the video: a person walked in front of the camera, and `pick_main_person` (largest skeleton) switched to them.
- **Decision:** `boulder.quality` marks frames invalid when the hip midpoint is > 1.5 TL from the climber's last good position, staying invalid until the skeleton returns (new position accepted after 2 s: the camera moved for good). Single-joint spikes are measured relative to the hips against a 5-frame rolling median (> 1.0 TL) so camera shake cancels out. Clips get pass / warn / fail with reasons.
- **Considered:** comparing each frame only with the previous one (misses the frames while the passer-by is in view); thresholds from p99 (0.6-0.9 TL would drop normal handheld movement).
- **Next:** track the climber in the pose backend (pick the person nearest the previous frame instead of the largest), then confirm with `quality.run` that SouthSide has no body jumps; make metrics skip invalid frames and dropped points.

## 2026-11-18 - Metrics use quality flags and the climb window
- **Decision:** `boulder.transforms.run` sets confidence to 0 for points with `point_ok = False` (quality checks) and keeps only frames inside the attempts-log climb window. Both are on by default and fall back gracefully (no quality file = raw keypoints; blank times = whole clip).
- **Why mask, not delete:** metrics already treat low-confidence joints as missing and only interpolate short gaps, so a masked passer-by stretch stays a gap instead of being bridged.
- **Also:** `duration_s` is now last minus first timestamp, correct for windows that don't start at 0.
- **Traceability:** each attempt row records `quality_applied`, `invalid_frames_masked`, `window` and the times used.

## 2026-11-18 - Track the climber instead of picking the biggest person
- **Problem:** `pick_main_person` kept the largest skeleton per frame, so a passer-by near the camera took over (SouthSide frames 698-726; found by the quality checks).
- **Decision:** the RTMPose backend follows the climber: biggest person in the first frame, then the person nearest the climber's last position (within 1 body size) and of similar size (< 1.8x). If nobody qualifies the frame gets no skeleton, which the quality and metrics steps already handle as missing. After 30 frames without a match it starts over from the biggest person. `model_version` ends in `/tracked`.
- **Considered:** only flagging bad frames (kept as a safety net in `boulder.quality`).
- **Check:** rerun pose, then `boulder.quality.run`: SouthSide's body-jump frames should drop.

## 2026-11-24 - Feedback video: rule-based red flags
- **Decision:** `boulder.render.feedback` colours limbs red from three rules: an arm bent below `straight_arm_deg` for >= 1.5 s, a foot readjustment (the same moves the metrics count), and a pause (tripod only). Flags come from the saved per-frame metrics and the `MetricConfig` stored with the attempt, so the video and the numbers always agree.
- **Why these rules:** they reuse metrics that already exist and are camera-safe (elbow angle, foot gap), so the feedback works on today's handheld clips. Bent arms are only flagged when held, because pulling through a move needs a bent arm for a moment.
- **Also:** flagged events are saved as a table (`data/processed/feedback/`), so the future app can list them without decoding the video.
- **Refactor:** `foot_move_spans` and `pause_spans` in `transforms.metrics` return where each move or pause happens; the counts are unchanged.
- **Known limits:** thresholds are first guesses; 2D elbow angles mislabel arms pointing at the camera; no flag yet for a fall or a slow move.
- **Next:** check the videos by eye, tune `bent_arm_min_s`; then an API (FastAPI) so a phone app can upload a clip and get the feedback video back. The goal is now a mobile app, not a website.

## 2026-11-24 - Bent-arm flag: clearly bent, held 2 s
- **Evidence:** first feedback video (MainRec Orange V5-6, handheld) was red during 65% of the climb; the timeline was almost solid red, mostly "arm bent".
- **Cause:** the flag reused the metrics' straight-arm line (150 degrees), so a slightly bent arm counted as bent, and 1.5 s is shorter than many normal moves.
- **Decision:** `FeedbackConfig.bent_arm_deg = 120` (clearly bent) and `bent_arm_min_s = 2.0`. The metrics' 150-degree straight-arm share is unchanged. The command now prints the red share of the climb, to compare settings.
- **Next:** rerender all 6 clips; aim for red on a minority of the climb, on moments that look wrong by eye.

## 2026-11-25 - API for the phone app (FastAPI)
- **Decision:** the goal is a phone app, not a website. The app stays thin and the Python pipeline becomes a server: `boulder.api` (FastAPI) takes an upload plus the clip's facts, runs every stage in the background and serves the metrics, events and feedback video. `boulder.pipeline.clip.process_clip` chains the stages for one clip; the command-line tools still run one stage over many clips and write the same files.
- **Why FastAPI:** Python like the pipeline, request checks from type hints, and free interactive docs at `/docs`, which also serve as the contract for the app.
- **Uploads go through ingest:** the facts become a canonical file name and `boulder.ingest.run.ingest` checks it, so names, duplicates and the manifest follow one set of rules. The climb window and wall style are appended to `metadata/attempts.csv`, so a later command-line rerun uses the same window.
- **Jobs:** processing takes minutes (pose), so `POST /clips` returns 202 with a job id at once and the app polls `GET /jobs/{id}`. Jobs live in memory and one pipeline runs at a time.
- **Refactors:** `quality.run.check_file` and `render.feedback.make_feedback` do one clip, so the pipeline can call them; their commands behave as before.
- **Considered:** a task queue (Celery/Redis) and a database for jobs: right for many users, too much for one local user now.
- **Next:** run the API from the phone on the same Wi-Fi; then pick the app stack (iPhone or Android); later persist jobs and add accounts before anyone else uses it.

## 2026-11-25 - Review log for the feedback flags
- **Decision:** my verdicts on each flag (correct / wrong / unsure, plus missed moments and coaching notes) go in `metadata/flag_reviews.csv`. `boulder.review.export` fills in new flags without touching reviewed rows; `boulder.review.score` compares verdicts with the current flags per flag type.
- **Why:** thresholds were tuned by eye from one video. Labelled flags turn tuning into a measurement: a change should lower `wrong` without raising `correct_lost`. The same labels can later train a model (roadmap v3).
- **Matching:** clip + flag type + overlapping time (0.2 s slack), so verdicts survive small shifts after retuning. Unreviewed rows for flags that disappear are dropped; reviewed rows are never dropped.
- **Why CSV now:** works today in VS Code next to the video. The iPhone app will write the same fields through the API later.
- **Next:** review all 6 clips, score, then tune `bent_arm_deg` / `bent_arm_min_s` and the readjustment distance against the labels.

## 2026-11-25 - First review results; verdict = detection, mistake is separate
- **Evidence** (121 flags, 6 handheld clips, reviewed by me): bent arm 17 correct / 5 wrong (77% precision); foot readjust 25 / 74 (25%). Flag length does not separate right from wrong foot flags (median 0.5 s both), nor does both feet being flagged at once.
- **Meaning:** `verdict` records whether the detection was right. Of the 17 correct arm flags, 12 were noted lock-off or no load, so most detected bent arms were not mistakes. A separate `mistake` column (yes / no / blank) now holds that judgement, filled from clear notes (13 rows set to `no`).
- **Decision:** fix foot-readjust detection first, with `boulder.review.calibrate`: it measures the motion behind each reviewed foot flag (which measures separate correct from wrong) and grid-searches `foot_move_speed`, `foot_still_s`, `readjust_max_dist` and the new `move_min_dist` (moves shorter than this are pose jitter; default 0 = old behaviour) against the labels.
- **Next:** pick a foot setting from the calibration (keep most of the 25 correct, drop most of the 74 wrong), check a few new unreviewed flags, re-export and score. Then a load signal for bent arms (hand still on a hold, number of feet on) to tell lock-offs and relaxed arms from real over-gripping.

