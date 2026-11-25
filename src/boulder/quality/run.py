"""Run the quality checks on every clip that has keypoints.

Usage:
    python -m boulder.quality.run                       # rtmpose-feet, all cameras
    python -m boulder.quality.run --model mediapipe --camera tripod

Reads:  data/processed/keypoints/camera=<c>/model=<m>/<clip>.parquet
Writes: data/processed/quality/points/camera=<c>/model=<m>/<clip>.parquet
            keypoints + point_ok (bool) + valid (bool, the frame's flag)
        data/processed/quality/clips/camera=<c>/model=<m>/<clip>.parquet
            one row: status (pass/warn/fail), reasons, counts, config used
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from boulder.quality.checks import QualityConfig, check_clip

KEYPOINT_ROOT = Path("data/processed/keypoints")
QUALITY_ROOT = Path("data/processed/quality")


def check_file(kp_path: Path, out_root: Path = QUALITY_ROOT,
               cfg: QualityConfig | None = None) -> dict:
    """Run the checks on one keypoint file, write its points and clip report, return the report."""
    cfg = cfg or QualityConfig()
    camera = kp_path.parent.parent.name.removeprefix("camera=")
    rel = Path(kp_path.parent.parent.name) / kp_path.parent.name / kp_path.name
    points, _, report = check_clip(pd.read_parquet(kp_path), camera, cfg)
    report = {"clip_id": kp_path.stem, "camera": camera, **report,
              "quality_config": json.dumps(cfg.to_dict(), sort_keys=True)}
    for kind, df in (("points", points), ("clips", pd.DataFrame([report]))):
        path = out_root / kind / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rtmpose-feet")
    parser.add_argument("--camera", default="*", help="handheld | tripod | * (all)")
    parser.add_argument("--keypoints", type=Path, default=KEYPOINT_ROOT)
    parser.add_argument("--out", type=Path, default=QUALITY_ROOT)
    args = parser.parse_args()

    cfg = QualityConfig()
    files = sorted(args.keypoints.glob(f"camera={args.camera}/model={args.model}/*.parquet"))
    if not files:
        raise SystemExit(f"no keypoint files for model={args.model} camera={args.camera}")

    rows = [check_file(f, args.out, cfg) for f in files]

    pd.set_option("display.width", 200)
    table = pd.DataFrame(rows).set_index("clip_id")
    print(table[["status", "invalid_frames", "dropped_points_share", "torso_px"]].to_string())
    print()
    for clip_id, r in table.iterrows():
        print(f"{r['status'].upper():<5} {clip_id}: {r['reasons'] or 'ok'}")


if __name__ == "__main__":
    main()
