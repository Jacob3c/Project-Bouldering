"""Run every stage for one ingested clip: pose -> quality -> metrics -> feedback video.

The command-line tools run one stage over many clips; this runs all stages for a single
clip, which is what the API needs after an upload.

Usage:
    python -m boulder.pipeline.clip data/raw/<clip>.mov
    python -m boulder.pipeline.clip data/raw/<clip>.mov --start 4 --end 42

Each stage writes the same files as its command-line tool, so results from the API and
from the commands are interchangeable.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from boulder.ingest.clip_name import parse_clip_name

STEPS = ["pose", "quality", "metrics", "feedback"]
MODEL_LABEL = "rtmpose-feet"


def process_clip(raw_path: Path, climb_start_s: float | None = None,
                 climb_end_s: float | None = None,
                 on_step: Callable[[str], None] | None = None,
                 skip_existing_pose: bool = True) -> dict:
    """Run all stages; call on_step(name) as each one starts. Returns paths and results.

    Pose is the slow step, so by default it is skipped when keypoints already exist.
    """
    # Imported here: loading the pose model libraries is slow and the API test doesn't need them.
    from boulder.pose import run as pose_run
    from boulder.quality.run import check_file
    from boulder.render.feedback import make_feedback
    from boulder.transforms.metrics import MetricConfig
    from boulder.transforms.run import process_file

    raw_path = Path(raw_path)
    step = on_step or (lambda name: None)
    clip_id = parse_clip_name(raw_path).canonical_id or raw_path.stem

    step("pose")
    kp_path = pose_run.output_path(raw_path, MODEL_LABEL)
    if not (skip_existing_pose and kp_path.exists()):
        pose_run.run(raw_path, "rtmpose", skeleton="body_with_feet")

    step("quality")
    quality = check_file(kp_path)

    step("metrics")
    attempt = process_file(kp_path, Path("data/processed/metrics"), MetricConfig(),
                           windows={clip_id: (climb_start_s, climb_end_s)})

    step("feedback")
    feedback = make_feedback(raw_path, MODEL_LABEL)
    return {"clip_id": clip_id, "keypoints": kp_path, "quality": quality, "attempt": attempt,
            **feedback}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="an ingested clip in data/raw/")
    parser.add_argument("--start", type=float, default=None, help="climb start, seconds")
    parser.add_argument("--end", type=float, default=None, help="climb end, seconds")
    parser.add_argument("--redo-pose", action="store_true", help="rerun pose even if done")
    args = parser.parse_args()

    result = process_clip(args.video, args.start, args.end,
                          on_step=lambda name: print(f"-> {name}"),
                          skip_existing_pose=not args.redo_pose)
    print(f"quality: {result['quality']['status']}")
    print(f"red during {result['red_share']:.0%} of the climb, {len(result['events'])} events")
    print(f"wrote {result['video']}")
    if isinstance(result["events"], pd.DataFrame) and len(result["events"]):
        print(result["events"].drop(columns="video_id").round(1).to_string(index=False))


if __name__ == "__main__":
    main()
