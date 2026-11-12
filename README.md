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
│   └── pipeline/         # orchestration
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

## Climbing metrics

```bash
python -m boulder.transforms.run                    # all clips with rtmpose-feet keypoints
python -m boulder.transforms.run --camera tripod --skip-existing
```

Writes a per-frame table and a one-row-per-attempt summary to `data/processed/metrics/{frames,attempts}/camera=<c>/model=<m>/`. Attempt metadata (date, gym, grade, attempt, outcome, camera) is parsed from the clip's file name.

| Metric | Meaning |
| --- | --- |
| `arms_straight_share` | Share of frames with elbow angle >= 150 degrees |
| `foot_moves`, `foot_readjustments` | Foot movements; a move under 0.3 torso lengths counts as a readjustment |
| `pauses`, `pause_total_s` | Hip still (< 0.15 torso lengths/s) for at least 1 s |
| `hip_path_efficiency` | Straight-line hip displacement / hip path length (1 = perfectly direct) |
| `hip_mean_jerk` | Mean jerk of the hip path; lower = smoother |

Distances are in torso lengths (shoulder midpoint to hip midpoint), so clips filmed from different distances are comparable. Thresholds are in `MetricConfig` and are stored with every attempt row.
