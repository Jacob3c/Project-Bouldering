"""RTMPose backend (via rtmlib, which runs RTMPose on onnxruntime).

- Two steps per frame: a person detector finds people, then RTMPose finds joints
- Finds EVERY person; we follow the climber from frame to frame (the biggest person
  in the first frame, then whoever is nearest the climber's last position and of a
  similar size), so a passer-by walking in front of the camera is not picked up
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


TRACK_MAX_JUMP = 1.0         # climber moves at most this many body sizes between processed frames
TRACK_MAX_SIZE_RATIO = 1.8   # a person this much bigger/smaller is someone else (e.g. a passer-by)
TRACK_RESET_FRAMES = 30      # after this many frames with no match, start again from the biggest


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


def person_center_size(kps: np.ndarray, sc: np.ndarray,
                       min_score: float = 0.3) -> tuple[np.ndarray, float] | None:
    """(centre, size) of one skeleton: median of confident joints, diagonal of their box."""
    good = kps[sc > min_score]
    if len(good) < 2:
        return None
    span = good.max(axis=0) - good.min(axis=0)
    return np.median(good, axis=0), float(np.hypot(span[0], span[1]))


def pick_tracked_person(keypoints: np.ndarray, scores: np.ndarray,
                        prev_center: np.ndarray | None, prev_size: float | None,
                        max_jump: float = TRACK_MAX_JUMP,
                        max_size_ratio: float = TRACK_MAX_SIZE_RATIO) -> int | None:
    """Index of the person who is most likely still the climber, or None.

    Without a previous position (first frame), the biggest skeleton is chosen.
    Otherwise: the nearest person whose centre moved at most max_jump body sizes
    and whose size changed by less than max_size_ratio. Nobody qualifies -> None
    (the climber is hidden; better no skeleton than someone else's).
    """
    if prev_center is None or prev_size is None:
        return pick_main_person(keypoints, scores)
    if keypoints is None or len(keypoints) == 0:
        return None
    best, best_dist = None, np.inf
    for i, (kps, sc) in enumerate(zip(keypoints, scores)):
        found = person_center_size(kps, sc)
        if found is None:
            continue
        center, size = found
        ratio = max(size, prev_size) / max(min(size, prev_size), 1e-9)
        dist = float(np.linalg.norm(center - prev_center))
        if ratio < max_size_ratio and dist <= max_jump * prev_size and dist < best_dist:
            best, best_dist = i, dist
    return best


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
        self.reset()

    def reset(self) -> None:
        """Forget the climber's position (start of a new video)."""
        self._prev_center, self._prev_size = None, None
        self._misses, self._last_ts = 0, None

    @property
    def version(self) -> str:
        try:
            lib = version("rtmlib")
        except PackageNotFoundError:
            lib = "unknown"
        return f"rtmlib-{lib}/{self.skeleton}-{self.mode}/tracked"

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Keypoint]:
        if self._last_ts is not None and timestamp_ms < self._last_ts:
            self.reset()                            # time went backwards: a new video
        self._last_ts = timestamp_ms
        keypoints, scores = self._model(frame_bgr)  # rtmlib takes OpenCV BGR directly
        idx = pick_tracked_person(keypoints, scores, self._prev_center, self._prev_size)
        if idx is None:
            self._misses += 1
            if self._misses >= TRACK_RESET_FRAMES:
                self.reset()                        # lost for long: pick the biggest again
            return []
        found = person_center_size(keypoints[idx], scores[idx])
        if found is not None:
            self._prev_center, self._prev_size = found
        self._misses = 0
        return [
            Keypoint(name=name, x_px=float(x), y_px=float(y), confidence=float(s))
            for name, (x, y), s in zip(self._names, keypoints[idx], scores[idx])
        ]
