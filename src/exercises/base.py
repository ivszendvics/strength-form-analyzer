"""Exercise abstraction: how a new exercise plugs into the pipeline.

Adding a new exercise means implementing this interface (angle definitions,
which angle drives rep detection, which form rules apply) plus adding a
matching ``configs/<name>.yaml`` for its thresholds/weights. Nothing else
in the pipeline (pose estimation, smoothing, rep-detector engine,
scoring engine, visualization, CLI) needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.analysis.rules import RuleContext, RuleResult
from src.biomechanics.metrics import RepMetrics
from src.pose.landmarks import Landmark, PoseFrame
from src.reps.detector import Rep, RepDetectorConfig


def select_more_visible_side(
    frame: PoseFrame,
    left_landmarks: list[Landmark],
    right_landmarks: list[Landmark],
) -> str | None:
    """Pick 'left' or 'right' based on which side's landmarks are more visible.

    Returns None if neither side has any visibility at all. This project
    assumes a single-side (or near-side) camera view (see README "Camera
    Assumptions"); picking the more-visible side per frame is a pragmatic
    way to degrade gracefully rather than assuming a fixed camera side.
    """
    left_vis = frame.min_visibility(left_landmarks)
    right_vis = frame.min_visibility(right_landmarks)
    if left_vis <= 0.0 and right_vis <= 0.0:
        return None
    return "left" if left_vis >= right_vis else "right"


class Exercise(ABC):
    """Interface each supported exercise implements."""

    name: str

    @abstractmethod
    def required_landmarks(self) -> list[Landmark]:
        """Landmarks this exercise needs tracked at all."""
        raise NotImplementedError

    @abstractmethod
    def compute_angles(self, frame: PoseFrame, min_visibility: float) -> dict[str, float | None]:
        """Compute every named angle this exercise tracks for one frame.

        Keys are angle names like "knee_angle", "hip_angle", "trunk_angle",
        plus side-specific variants ("left_knee_angle", "right_knee_angle")
        used for asymmetry metrics when both sides are visible enough.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def primary_angle_name(self) -> str:
        """Angle name (a key from ``compute_angles``) that drives rep detection."""
        raise NotImplementedError

    @abstractmethod
    def build_rep_detector_config(self, raw_config: dict) -> RepDetectorConfig:
        """Build rep-detection thresholds from the exercise's parsed YAML config."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_form(self, context: RuleContext, raw_config: dict) -> list[RuleResult]:
        """Run this exercise's form rules for one rep, given its config."""
        raise NotImplementedError

    def rom_key_for_consistency(self) -> str:
        """Angle name whose range of motion is used for the stability/consistency rule."""
        return self.primary_angle_name

    def compute_extra_metrics(
        self,
        metrics: RepMetrics,
        angle_series: dict[str, np.ndarray],
        pose_frames: list[PoseFrame],
        rep: Rep,
    ) -> dict[str, float]:
        """Exercise-specific scalar extras (e.g. asymmetry_percent, start
        angle) to merge into ``RepMetrics.extra`` before form rules run.
        Default: none -- override where an exercise needs rules like
        symmetry, knee tracking, or position consistency.
        """
        return {}
