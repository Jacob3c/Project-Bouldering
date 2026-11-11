# Project-Bouldering
Video-to-analytics pipeline for bouldering: pose keypoints, climbing metrics, and skeleton-overlay feedback.
Project Boulder turns raw climbing footage into reliable, queryable data about how a climber moves. Videos of bouldering attempts are ingested with metadata (grade, wall angle, sent or fell), run through pose estimation, and stored as keypoint time series. Transformations compute movement metrics such as arm bend, foot readjustments, hesitation time and hip-path smoothness, and data-quality checks flag low-confidence frames. The output is a feedback video with a skeleton overlay that highlights inefficient moves.

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

## Pose extraction

Two interchangeable models produce the same output table, so they can be compared on the same clips:

```bash
python -m boulder.pose.run data/test/clip.mp4 --model both   # MediaPipe + RTMPose
python -m boulder.pose.compare                               # per-joint agreement
python -m boulder.render.overlay data/test/clip.mp4          # both skeletons drawn on the video
```

RTMPose can also output 26 joints including toes and heels: add `--model rtmpose --skeleton body_with_feet`
(saved as `model=rtmpose-feet`), then compare with `--a mediapipe --b rtmpose-feet`.

Keypoints land in `data/processed/keypoints/camera=<handheld|tripod>/model=<name>/<video_id>.parquet`, one row per frame per joint. The camera comes from the file name's last word (`..._tripod.MOV`). Add `--skip-existing` to process only new clips, and `--camera tripod` to `compare` to filter footage.
