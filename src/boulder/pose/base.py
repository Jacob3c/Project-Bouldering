"""Shared pieces every pose backend uses.

Every backend (MediaPipe, RTMPose, ...) turns one video frame into a list of
keypoints. This module defines the common output format so the rest of the
pipeline never needs to know which model produced the data.

Output schema (one row per frame per keypoint):

    video_id       str    file name without extension
    model_name     str    "mediapipe" | "rtmpose"
    model_version  str    library version + model variant, e.g. "mediapipe-1.2.2/heavy"
    frame_idx      int    frame number in the source video (0-based)
    timestamp_ms   int    position of the frame in the video
    keypoint       str    joint name, e.g. "left_wrist" (shared names across models)
    x_px, y_px     float  position in pixels (origin top-left)
    x_norm, y_norm float  position as a fraction of image width / height (0-1)
    confidence     float  model's own score; NOT comparable across models
    img_w, img_h   int    frame size in pixels
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

SCHEMA_COLUMNS = [
    "video_id",
    "model_name",
    "model_version",
    "frame_idx",
    "timestamp_ms",
    "keypoint",
    "x_px",
    "y_px",
    "x_norm",
    "y_norm",
    "confidence",
    "img_w",
    "img_h",
]

# The 17 COCO joints. Both models output these, under these exact names,
# which is what lets you compare them joint by joint.
COCO17 = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass
class Keypoint:
    name: str
    x_px: float
    y_px: float
    confidence: float


class PoseBackend(ABC):
    """Interface every model wrapper implements."""

    #: short name stored in the `model_name` column
    name: str = "base"

    @property
    @abstractmethod
    def version(self) -> str:
        """Library version + model variant, stored in `model_version`."""

    @abstractmethod
    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Keypoint]:
        """Return keypoints for the climber in one frame ([] if nobody found).

        `frame_bgr` is an OpenCV image (BGR channel order).
        `timestamp_ms` must increase from call to call within one video.
        """

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release model resources. Override if the backend holds any."""

    def __enter__(self) -> "PoseBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
