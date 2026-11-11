"""Compute climbing metrics for every clip that has keypoints.

Usage:
    python -m boulder.transforms.run                          # rtmpose-feet, all cameras
    python -m boulder.transforms.run --camera tripod
    python -m boulder.transforms.run --model mediapipe --skip-existing

Reads:  data/processed/keypoints/camera=<c>/model=<m>/<video_id>.parquet
Writes: data/processed/metrics/frames/camera=<c>/model=<m>/<video_id>.parquet   (per frame)
        data/processed/metrics/attempts/camera=<c>/model=<m>/<video_id>.parquet (one row)

Then prints one line per attempt. Query everything later with DuckDB, e.g.
    SELECT * FROM read_parquet('data/processed/metrics/attempts/*/*/*.parquet',
                               hive_partitioning = true)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from boulder.ingest.clip_name import parse_clip_name

from .metrics import MetricConfig, attempt_metrics, frame_metrics

KEYPOINT_ROOT = Path("data/processed/keypoints")
METRIC_ROOT = Path("data/processed/metrics")

SUMMARY_COLS = [
    "video_id", "outcome", "grade", "duration_s", "arms_straight_share",
    "foot_moves", "foot_readjustments", "pauses", "pause_total_s",
    "hip_path_efficiency",
]


def process_file(kp_path: Path, metric_root: Path, cfg: MetricConfig,
                 skip_existing: bool = False) -> dict | None:
    camera_dir, model_dir = kp_path.parent.parent.name, kp_path.parent.name
    rel = Path(camera_dir) / model_dir / kp_path.name
    frames_out = metric_root / "frames" / rel
    attempt_out = metric_root / "attempts" / rel
    if skip_existing and attempt_out.exists():
        return pd.read_parquet(attempt_out).iloc[0].to_dict()

    keypoints = pd.read_parquet(kp_path)
    frames = frame_metrics(keypoints, cfg)
    row = {**parse_clip_name(kp_path).to_dict(), **attempt_metrics(frames, cfg)}
    row["model_version"] = keypoints["model_version"].iloc[0] if len(keypoints) else None
    row["metric_config"] = json.dumps(cfg.to_dict(), sort_keys=True)

    for path in (frames_out, attempt_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    frames.insert(0, "video_id", kp_path.stem)
    frames.to_parquet(frames_out, index=False)
    pd.DataFrame([row]).to_parquet(attempt_out, index=False)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rtmpose-feet", help="keypoint model folder name")
    parser.add_argument("--camera", choices=["handheld", "tripod", "unknown", "all"],
                        default="all")
    parser.add_argument("--keypoints", type=Path, default=KEYPOINT_ROOT)
    parser.add_argument("--out", type=Path, default=METRIC_ROOT)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    camera = "*" if args.camera == "all" else args.camera
    files = sorted(args.keypoints.glob(f"camera={camera}/model={args.model}/*.parquet"))
    if not files:
        parser.error(f"no keypoints found for model={args.model}, camera={args.camera} "
                     f"under {args.keypoints}")

    cfg = MetricConfig()
    rows = [r for f in files if (r := process_file(f, args.out, cfg, args.skip_existing))]
    summary = pd.DataFrame(rows)[SUMMARY_COLS]
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))
    print(f"\n{len(rows)} attempts -> {args.out}/attempts/")


if __name__ == "__main__":
    main()
