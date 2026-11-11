"""RTMPose backend (via rtmlib, which runs RTMPose on onnxruntime).

- Two steps per frame: a person detector finds people, then RTMPose finds joints
- Finds EVERY person; we keep the biggest one (usually the climber nearest the camera)
- "body" = 17 COCO joints; "body_with_feet" = 26 joints incl. big toe, small toe, heel
- Model files download automatically on first use
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .base import COCO17, Keypoint, PoseBackend

# Halpe-26 = COCO-17 + head, neck, hip centre + 6 foot points.
HALPE26 = COCO17 + [
    "head",
    "neck",
    "hip_center",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
]


def pick_main_person(keypoints: np.ndarray, scores: np.ndarray) -> int | None:
    """Index of the person whose skeleton covers the largest area, or None."""
    if keypoints is None or len(keypoints) == 0:
        return None
    areas = []
    for kps, sc in zip(keypoints, scores):
        good = kps[sc > 0.3]
        if len(good) < 2:
            areas.append(0.0)
            continue
        span = good.max(axis=0) - good.min(axis=0)
        areas.append(float(span[0] * span[1]))
    best = int(np.argmax(areas))
    return best if areas[best] > 0 else None


class RTMPoseBackend(PoseBackend):
    name = "rtmpose"

    def __init__(self, skeleton: str = "body", mode: str = "balanced", device: str = "cpu"):
        import rtmlib

        if skeleton == "body":
            solution, self._names = rtmlib.Body, COCO17
        elif skeleton == "body_with_feet":
            solution = getattr(rtmlib, "BodyWithFeet", None) or getattr(rtmlib, "Body_with_feet")
            self._names = HALPE26
        else:
            raise ValueError("skeleton must be 'body' or 'body_with_feet'")
        self.skeleton, self.mode = skeleton, mode
        # mode: "lightweight" (fast) | "balanced" | "performance" (most accurate)
        self._model = solution(mode=mode, backend="onnxruntime", device=device)

    @property
    def version(self) -> str:
        try:
            lib = version("rtmlib")
        except PackageNotFoundError:
            lib = "unknown"
        return f"rtmlib-{lib}/{self.skeleton}-{self.mode}"

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Keypoint]:
        keypoints, scores = self._model(frame_bgr)  # rtmlib takes OpenCV BGR directly
        idx = pick_main_person(keypoints, scores)
        if idx is None:
            return []
        return [
            Keypoint(name=name, x_px=float(x), y_px=float(y), confidence=float(s))
            for name, (x, y), s in zip(self._names, keypoints[idx], scores[idx])
        ]
