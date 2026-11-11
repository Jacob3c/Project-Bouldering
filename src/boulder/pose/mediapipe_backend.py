"""MediaPipe Pose Landmarker backend.

- 33 landmarks, including heels and toes (useful for footwork)
- Tracks ONE person per frame
- Runs on the CPU; needs a .task model file (downloaded automatically into models/)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .base import Keypoint, PoseBackend

# MediaPipe's 33 landmarks, in the order the model returns them.
MEDIAPIPE33 = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

MODEL_URLS = {
    variant: (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
    )
    for variant in ("lite", "full", "heavy")
}

DEFAULT_MODEL_DIR = Path("models")


def ensure_model(variant: str = "heavy", model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Download the .task file the first time; reuse it afterwards."""
    if variant not in MODEL_URLS:
        raise ValueError(f"variant must be one of {list(MODEL_URLS)}, got {variant!r}")
    path = Path(model_dir) / f"pose_landmarker_{variant}.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe {variant} model -> {path}")
        urllib.request.urlretrieve(MODEL_URLS[variant], path)
    return path


class MediaPipeBackend(PoseBackend):
    name = "mediapipe"

    def __init__(self, variant: str = "heavy", model_dir: Path = DEFAULT_MODEL_DIR,
                 min_confidence: float = 0.5):
        import mediapipe as mp  # imported here so the rest of the package loads without it

        self._mp = mp
        self.variant = variant
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(ensure_model(variant, model_dir))
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    @property
    def version(self) -> str:
        return f"mediapipe-{self._mp.__version__}/{self.variant}"

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Keypoint]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        if not result.pose_landmarks:
            return []
        landmarks = result.pose_landmarks[0]  # first (only) person
        return [
            Keypoint(
                name=name,
                x_px=lm.x * w,  # MediaPipe returns 0-1; convert to pixels
                y_px=lm.y * h,
                confidence=float(lm.visibility),
            )
            for name, lm in zip(MEDIAPIPE33, landmarks)
        ]

    def close(self) -> None:
        self._landmarker.close()
