# Project-Bouldering
Video-to-analytics pipeline for bouldering: pose keypoints, climbing metrics, and skeleton-overlay feedback.
Project Boulder turns raw climbing footage into reliable, queryable data about how a climber moves. Videos of bouldering attempts are ingested with metadata (grade, wall style, sent or fell), run through pose estimation, and stored as keypoint time series. Transformations compute movement metrics such as arm bend, foot readjustments, hesitation time and hip-path smoothness, and data-quality checks flag low-confidence frames. The output is a feedback video with a skeleton overlay that highlights inefficient moves.

Roadmap: v1 single camera + rule-based metrics → v2 multi-angle 3D reconstruction → v3 learned models and physics simulation.

## Project structure

```
project-bouldering/
├── src/boulder/          # the Python package, one module per pipeline stage
│   ├── ingest/           # raw videos + attempt metadata in
│   ├── pose/             # pose estimation -> keypoint time series
│   ├── transforms/       # movement metrics
│   ├── quality/          # data-quality checks
│   ├── render/           # skeleton-overlay feedback videos
│   ├── pipeline/         # all stages for one clip
│   ├── review/           # your verdicts on feedback flags
│   └── api/              # HTTP API for the app
├── data/                 # local only, never committed
│   ├── raw/              # tripod footage
│   ├── test/             # handheld / test clips
│   ├── processed/        # keypoints and metrics (Parquet)
│   └── output/           # rendered feedback videos
├── metadata/             # attempt log template
├── notebooks/            # exploration
├── tests/
└── docs/decisions.md     # decision log
```

## Setup

Requires Python 3.11 or 3.12 (MediaPipe can lag behind the newest Python release).

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

## Ingest

Every clip enters the pipeline through ingest, which checks the file name, copies the clip to
`data/raw/` under its canonical name and records it in `data/manifest.parquet`:

```bash
python -m boulder.ingest.run data/test --dry-run   # report what would happen
python -m boulder.ingest.run data/test             # copy + record new clips
```

- Names are normalized: `2026-1-19_abp_orange-v3_att1_send_handheld.MOV` becomes
  `2026-01-19_ABP_Orange-V3_att01_send_handheld.mov`. Official gym, colour and wall-style names
  live in `configs/aliases.yaml`.
- Names with an unknown gym or colour, a bad grade or a wrong pattern are **rejected** and not copied.
- A clip whose bytes (SHA-256) are already in the manifest is a **duplicate** and is skipped, even
  under another name. Rerunning ingest changes nothing.
- The manifest holds one row per clip: `clip_id`, parsed name fields, `source_name`, `raw_path`,
  `sha256`, `size_bytes`, `fps`, `n_frames`, `width`, `height`, `duration_s`, `ingested_at`.

Then run pose on the ingested clips: `python -m boulder.pose.run data/raw/*.mov --skip-existing`.

### Attempts log

Facts a file name can't hold go in `metadata/attempts.csv`, one row per clip (blank cells are fine):

| Column | Meaning |
| --- | --- |
| `file_name` | the clip's file name (any spelling the name parser accepts) |
| `climb_start_s` | when the climb starts (leaving the ground): seconds (`4.2`) or m:ss as QuickTime shows it (`0:04.2`) |
| `climb_end_s` | when the climb ends (top-out or fall), same formats |
| `wall_style` | main style: `slab`, `vertical`, `overhang`, `roof`, `arete`, `dihedral` |
| `wall_sections` | styles in climbing order, e.g. `vertical>overhang>slab` |
| `fall_move` | move number of the fall (1 = first move; `start` or `0` = fell on the start holds); blank on a send |
| `notes` | free text |

```bash
python -m boulder.ingest.attempts            # manifest + log -> data/clips.parquet
python -m boulder.ingest.attempts --strict   # fail if the log has any issue
```

Styles are normalized through `configs/aliases.yaml`. Every ingested clip appears in
`data/clips.parquet` (a left join), and problems are printed: unknown styles, log rows that match
no ingested clip, clips listed twice, or a `fall_move` on a send.

## Pose extraction

Default model is RTMPose with feet (26 joints; see `docs/decisions.md`). MediaPipe produces the same output table, so the two can be compared on the same clips:

```bash
python -m boulder.pose.run data/test/clip.mp4 --model both   # MediaPipe + RTMPose
python -m boulder.pose.compare                               # per-joint agreement
python -m boulder.render.overlay data/test/clip.mp4          # both skeletons drawn on the video
```

RTMPose can also output 26 joints including toes and heels: add `--model rtmpose --skeleton body_with_feet`
(saved as `model=rtmpose-feet`), then compare with `--a mediapipe --b rtmpose-feet`.

Keypoints land in `data/processed/keypoints/camera=<handheld|tripod>/model=<name>/<video_id>.parquet`, one row per frame per joint. The camera comes from the file name's last word (`..._tripod.MOV`). Add `--skip-existing` to process only new clips, and `--camera tripod` to `compare` to filter footage.

## Quality checks

```bash
python -m boulder.quality.profile   # read-only report: gaps, confidence, jumps per joint
python -m boulder.quality.run       # flag bad frames/points, pass/warn/fail per clip
```

| Level | Check | Default | Effect |
| --- | --- | --- | --- |
| frame | hip midpoint far from the climber's last good position (passer-by, camera jerk) | > 1.5 torso lengths | frame `valid = False` until the skeleton returns (or 2 s pass) |
| point | confidence too low | < 0.5 | `point_ok = False` |
| point | joint far from its 5-frame rolling median, relative to the hips | > 1.0 torso length | `point_ok = False` |
| clip | climber small in frame / points out of frame / detection gap / handheld | torso < 100 px / > 1% / > 0.5 s / camera tag | warning |
| clip | too many invalid frames | > 20% | fail |

Writes `data/processed/quality/points/...` (keypoints + `point_ok` + `valid`) and
`data/processed/quality/clips/...` (status, reasons, config used). Thresholds live in `QualityConfig`.

## Climbing metrics

```bash
python -m boulder.transforms.run                    # all clips with rtmpose-feet keypoints
python -m boulder.transforms.run --camera tripod --skip-existing
```

Metrics use your earlier steps when their outputs exist: points the quality checks rejected
(`python -m boulder.quality.run`) are ignored, and only the `climb_start_s`..`climb_end_s` window
from the attempts log (`python -m boulder.ingest.attempts`) is measured. Blank times mean the whole
clip. `--no-quality` and `--no-window` turn either off; each attempt row records what was applied
(`quality_applied`, `invalid_frames_masked`, `window`, `climb_start_s`, `climb_end_s`).

Writes a per-frame table and a one-row-per-attempt summary to `data/processed/metrics/{frames,attempts}/camera=<c>/model=<m>/`. Attempt metadata (date, gym, grade, attempt, outcome, camera) is parsed from the clip's file name.

| Metric | Meaning |
| --- | --- |
| `arms_straight_share` | Share of frames with elbow angle >= 150 degrees |
| `foot_moves`, `foot_readjustments` | Foot movements; a move under 0.3 torso lengths counts as a readjustment |
| `pauses`, `pause_total_s` | Hip still (< 0.15 torso lengths/s) for at least 1 s |
| `hip_path_efficiency` | Straight-line hip displacement / hip path length (1 = perfectly direct) |
| `hip_mean_jerk` | Mean jerk of the hip path; lower = smoother |

Distances are in torso lengths (shoulder midpoint to hip midpoint), so clips filmed from different distances are comparable. Thresholds are in `MetricConfig` and are stored with every attempt row.

## Feedback video

```bash
python -m boulder.render.feedback data/raw/<clip>.mov
```

Draws the climber's skeleton on the clip: green during the climb, grey before and after it.
A limb turns red when a rule flags it, and the flag is named in the top-left corner. A timeline
at the bottom marks every flagged moment, with a white cursor for the current frame.

| Flag | Rule (thresholds in `FeedbackConfig` and `MetricConfig`) |
| --- | --- |
| bent arm (red arm) | elbow < 120 degrees (clearly bent) for at least 2 s in a row |
| foot readjust (red leg) | a foot move (gap between the feet changing faster than 1.0 torso lengths/s) shorter than 0.3 torso lengths; shown for at least 0.5 s |
| pause (orange body) | hips still for at least 1 s; tripod clips only |

Run pose, quality and metrics first. Writes `data/output/<video_id>_feedback.mp4` and one row per
flagged event to `data/processed/feedback/camera=<c>/model=<m>/<video_id>.parquet`, then prints the
events with their clip times and the share of the climb that was red.

## API

The API is what the phone app will talk to: upload a clip with its facts, then fetch the
metrics and the feedback video. Start it from the repo root:

```bash
pip install -e ".[dev]"                 # once: adds fastapi, uvicorn, python-multipart, httpx2
uvicorn boulder.api.app:app --reload    # --reload restarts the server when code changes
```

Open http://127.0.0.1:8000/docs for interactive docs: every endpoint can be tried in the browser.

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/health` | `{"status": "ok"}` when the server is up |
| POST | `/clips` | upload a video + date, gym, colour, grade, outcome, camera (+ optional climb times, wall style, notes); returns a `job_id` (202) |
| GET | `/jobs/{job_id}` | `queued`, `running` (with the current `step`), `done` or `failed` (with the `error`) |
| GET | `/clips` | every ingested clip, newest first, with `has_feedback` |
| GET | `/clips/{clip_id}` | facts, quality report, metrics and feedback events for one clip |
| GET | `/clips/{clip_id}/video` | the feedback video (mp4) |

An upload passes the same checks as `boulder.ingest.run` (bad gym, colour or grade -> 422; a
different video with the same ID -> 409), and its climb window and wall style are appended to
`metadata/attempts.csv`. Processing runs in the background, one clip at a time:
pose -> quality -> metrics -> feedback. The same chain runs from the command line for one clip:

```bash
python -m boulder.pipeline.clip data/raw/<clip>.mov --start 4 --end 42
```

Limits for now: local only (no accounts or login), jobs are kept in memory and lost on restart.

## Reviewing the flags

Your verdicts on the red flags are the ground truth for tuning the rules. They live in
`metadata/flag_reviews.csv`, one row per flag:

```bash
python -m boulder.review.export   # add the latest flags (blank verdicts); keeps what you wrote
python -m boulder.review.play     # fast review: loops each flag, one key per verdict
python -m boulder.review.score    # your verdicts vs the current flags, per flag type
python -m boulder.review.calibrate  # which foot-rule settings fit your verdicts best
```

In the player, click the video window, then: `c` correct (detection right), `w` wrong (it didn't happen), `u` unsure, `s` skip,
`n` note (typed in the video window, Enter saves, Esc cancels), `m` missed mistake at this moment, space pause, `,` `.` step a
frame, `r` replay, `b` back, `q` save and quit. Every key press is saved to the CSV at once.

| Column | Meaning |
| --- | --- |
| `video_id` | the clip |
| `flag` | `left_arm_bent`, `right_arm_bent`, `left_foot_readjust`, `right_foot_readjust`, `pause`, or `other` |
| `start`, `end` | clip time, m:ss.s as QuickTime shows it (or seconds) |
| `verdict` | was the detection right: `correct` (it happened), `wrong` (it didn't), `unsure`, or `missed` |
| `mistake` | was it a mistake: `yes`, `no` (needed or harmless: lock-off, foot swap, no load), blank = not judged |
| `note` | why, or what you would coach |
| `source` | `auto` = exported; `you` = a row you added |

Add a mistake the system missed as a new row with `verdict` = `missed`, `source` = `you`, a flag
type (or `other`) and its time. Rows are matched by clip, flag type and overlapping time, so
verdicts survive re-rendering after a threshold change. `score` then shows `precision`
(share of shown flags you called real mistakes), `wrong` (still shown), `correct_lost` (no longer
shown) and `missed_now_caught`.

