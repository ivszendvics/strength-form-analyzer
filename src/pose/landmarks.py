"""Landmark data structures and the body-landmark naming scheme.

The rest of the application (biomechanics, exercises, visualization) is
written against these types rather than against any pose-estimation
backend's native objects. This keeps ``MediaPipePoseEstimator`` (or any
future backend, e.g. OpenPose) swappable without touching downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Landmark(str, Enum):
    """Body landmarks used by the analysis pipeline.

    Names follow MediaPipe Pose's 33-point topology restricted to the
    subset this project actually uses. Using an explicit enum (rather than
    raw MediaPipe indices) keeps exercise/biomechanics code backend-agnostic.
    """

    NOSE = "nose"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_HEEL = "left_heel"
    RIGHT_HEEL = "right_heel"
    LEFT_FOOT_INDEX = "left_foot_index"
    RIGHT_FOOT_INDEX = "right_foot_index"


@dataclass(frozen=True)
class LandmarkPoint:
    """A single tracked landmark for one frame.

    Coordinates follow MediaPipe's convention: ``x``/``y`` are normalized
    to [0, 1] relative to image width/height (so analysis is reasonably
    robust to different video resolutions), and ``z`` is a rough
    depth relative to the hips, in the same normalized scale as ``x``.
    ``visibility`` is MediaPipe's estimated probability that the landmark
    is visible (not occluded) in the frame, used as a confidence proxy.
    """

    x: float
    y: float
    z: float
    visibility: float

    def as_xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class PoseFrame:
    """Pose estimation result for a single video frame.

    ``landmarks`` maps a :class:`Landmark` to its detected
    :class:`LandmarkPoint`. A landmark absent from the dict means the
    backend did not produce a value for it at all; a landmark present but
    with low ``visibility`` means it was estimated but with low confidence.
    Downstream code must treat these two cases explicitly rather than
    assuming a landmark is always present.
    """

    frame_index: int
    timestamp_seconds: float
    landmarks: dict[Landmark, LandmarkPoint]
    person_detected: bool = True

    def get(self, landmark: Landmark) -> LandmarkPoint | None:
        return self.landmarks.get(landmark)

    def visibility(self, landmark: Landmark) -> float:
        point = self.landmarks.get(landmark)
        return point.visibility if point is not None else 0.0

    def min_visibility(self, landmarks: list[Landmark]) -> float:
        """Lowest visibility among the given landmarks, 0.0 if any are missing."""
        values = [self.visibility(lm) for lm in landmarks]
        return min(values) if values else 0.0
