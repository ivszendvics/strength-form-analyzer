"""Squat exercise: angle definitions, rep-detection driver, and form rules.

Camera assumption: side or front-side view, full body visible (see README
"Camera Assumptions"). The knee/hip/trunk angles below are computed from
whichever side (left/right) has better landmark visibility in each frame,
which degrades gracefully for a side-on camera without assuming a fixed
camera side.
"""

from __future__ import annotations

import numpy as np

from src.analysis.rules import (
    RuleContext,
    RuleResult,
    evaluate_depth,
    evaluate_knee_tracking,
    evaluate_rep_consistency,
    evaluate_symmetry,
    evaluate_tempo,
    evaluate_trunk_lean,
)
from src.biomechanics.angles import joint_angle, trunk_angle_from_vertical
from src.biomechanics.metrics import RepMetrics, asymmetry_percent
from src.exercises.base import Exercise, select_more_visible_side
from src.pose.landmarks import Landmark, PoseFrame
from src.reps.detector import Rep, RepDetectorConfig

_LEFT = [Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE, Landmark.LEFT_SHOULDER]
_RIGHT = [Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE, Landmark.RIGHT_SHOULDER]


class SquatExercise(Exercise):
    name = "squat"

    def required_landmarks(self) -> list[Landmark]:
        return [
            Landmark.LEFT_HIP,
            Landmark.RIGHT_HIP,
            Landmark.LEFT_KNEE,
            Landmark.RIGHT_KNEE,
            Landmark.LEFT_ANKLE,
            Landmark.RIGHT_ANKLE,
            Landmark.LEFT_SHOULDER,
            Landmark.RIGHT_SHOULDER,
        ]

    def compute_angles(self, frame: PoseFrame, min_visibility: float) -> dict[str, float | None]:
        angles: dict[str, float | None] = {}

        left_knee = joint_angle(frame.get(Landmark.LEFT_HIP), frame.get(Landmark.LEFT_KNEE), frame.get(Landmark.LEFT_ANKLE), min_visibility)
        right_knee = joint_angle(frame.get(Landmark.RIGHT_HIP), frame.get(Landmark.RIGHT_KNEE), frame.get(Landmark.RIGHT_ANKLE), min_visibility)
        angles["left_knee_angle"] = left_knee
        angles["right_knee_angle"] = right_knee

        left_hip = joint_angle(frame.get(Landmark.LEFT_SHOULDER), frame.get(Landmark.LEFT_HIP), frame.get(Landmark.LEFT_KNEE), min_visibility)
        right_hip = joint_angle(frame.get(Landmark.RIGHT_SHOULDER), frame.get(Landmark.RIGHT_HIP), frame.get(Landmark.RIGHT_KNEE), min_visibility)
        angles["left_hip_angle"] = left_hip
        angles["right_hip_angle"] = right_hip

        side = select_more_visible_side(frame, _LEFT, _RIGHT)
        if side == "left":
            angles["knee_angle"] = left_knee
            angles["hip_angle"] = left_hip
            angles["trunk_angle"] = trunk_angle_from_vertical(frame.get(Landmark.LEFT_SHOULDER), frame.get(Landmark.LEFT_HIP), min_visibility)
        elif side == "right":
            angles["knee_angle"] = right_knee
            angles["hip_angle"] = right_hip
            angles["trunk_angle"] = trunk_angle_from_vertical(frame.get(Landmark.RIGHT_SHOULDER), frame.get(Landmark.RIGHT_HIP), min_visibility)
        else:
            angles["knee_angle"] = None
            angles["hip_angle"] = None
            angles["trunk_angle"] = None

        return angles

    @property
    def primary_angle_name(self) -> str:
        return "knee_angle"

    def build_rep_detector_config(self, raw_config: dict) -> RepDetectorConfig:
        rd = raw_config.get("rep_detection", {})
        return RepDetectorConfig(
            standing_angle=rd.get("standing_angle", 165.0),
            descending_threshold=rd.get("descending_threshold", 155.0),
            bottom_angle=rd.get("bottom_angle", 110.0),
            ascending_threshold=rd.get("ascending_threshold", 120.0),
            standing_return_threshold=rd.get("standing_return_threshold", 155.0),
            min_rep_duration_seconds=rd.get("min_rep_duration_seconds", 0.5),
            min_frames_in_state=rd.get("min_frames_in_state", 2),
        )

    def evaluate_form(self, context: RuleContext, raw_config: dict) -> list[RuleResult]:
        form_cfg = raw_config.get("form", {})
        depth_cfg = form_cfg.get("depth", {})
        trunk_cfg = form_cfg.get("trunk", {})
        tempo_cfg = form_cfg.get("tempo", {})
        symmetry_cfg = form_cfg.get("symmetry", {})
        knee_tracking_cfg = form_cfg.get("knee_tracking", {})

        return [
            evaluate_depth(
                context,
                angle_key="knee_angle",
                target_angle=depth_cfg.get("target_angle", 100.0),
                tolerance=depth_cfg.get("tolerance", 10.0),
            ),
            evaluate_trunk_lean(
                context,
                max_trunk_angle_threshold=trunk_cfg.get("max_angle", 45.0),
            ),
            evaluate_tempo(
                context,
                max_deviation_ratio=tempo_cfg.get("max_deviation_ratio", 0.4),
            ),
            evaluate_rep_consistency(
                context,
                rom_key="knee_angle",
                min_rom_ratio_vs_earlier=form_cfg.get("consistency", {}).get("min_rom_ratio_vs_earlier", 0.8),
            ),
            evaluate_symmetry(
                context,
                max_asymmetry_percent=symmetry_cfg.get("max_asymmetry_percent", 15.0),
            ),
            evaluate_knee_tracking(
                context,
                knee_drift_ratio=context.metrics.extra.get("knee_drift_ratio"),
                max_drift_ratio=knee_tracking_cfg.get("max_drift_ratio", 0.25),
                camera_view=knee_tracking_cfg.get("camera_view", "side"),
            ),
        ]

    def compute_extra_metrics(
        self,
        metrics: RepMetrics,
        angle_series: dict[str, np.ndarray],
        pose_frames: list[PoseFrame],
        rep: Rep,
    ) -> dict[str, float]:
        extra: dict[str, float] = {}

        left = angle_series.get("left_knee_angle")
        right = angle_series.get("right_knee_angle")
        if left is not None and right is not None:
            asym = asymmetry_percent(left, right, rep.start_frame, rep.end_frame)
            if asym is not None:
                extra["asymmetry_percent"] = asym

        drift = self._knee_ankle_x_drift_ratio(pose_frames, rep)
        if drift is not None:
            extra["knee_drift_ratio"] = drift

        return extra

    # Minimum *average* apparent hip width across the whole rep (as a
    # fraction of image width) required before this metric is reported at
    # all. This has to be an average-over-the-rep gate, not a per-frame
    # one: from a side-on camera (this project's recommended/default squat
    # framing) the two hips project to nearly the same x pixel, so
    # hip_width hovers near zero and landmark jitter alone can push a
    # single frame's value just over almost any per-frame threshold,
    # amplifying that one noisy frame into a wildly inflated ratio (100+,
    # then even after a per-frame gate, 5-6x -- both observed on real
    # side-view footage during testing). Gating on the rep-average instead
    # means a genuinely side-on video reliably skips this metric entirely
    # rather than reporting a misleading number.
    _MIN_AVG_HIP_WIDTH_FOR_DRIFT_RATIO = 0.08

    @classmethod
    def _knee_ankle_x_drift_ratio(cls, pose_frames: list[PoseFrame], rep: Rep) -> float | None:
        """How much the (selected-side) knee wanders horizontally relative to
        the ankle during the rep, normalized by hip width. Intended as a
        front-camera knee-tracking proxy; only meaningful when the camera
        view is roughly frontal (see ``evaluate_knee_tracking``) -- see
        ``_MIN_AVG_HIP_WIDTH_FOR_DRIFT_RATIO`` for why a side-on camera
        skips this metric entirely rather than reporting a noisy value.
        """
        rep_frames = pose_frames[rep.start_frame : rep.end_frame + 1]

        hip_widths: list[float] = []
        for frame in rep_frames:
            left_hip = frame.get(Landmark.LEFT_HIP)
            right_hip = frame.get(Landmark.RIGHT_HIP)
            if left_hip is not None and right_hip is not None:
                hip_widths.append(abs(left_hip.x - right_hip.x))
        if not hip_widths or (sum(hip_widths) / len(hip_widths)) < cls._MIN_AVG_HIP_WIDTH_FOR_DRIFT_RATIO:
            return None

        offsets: list[float] = []
        for frame in rep_frames:
            side = select_more_visible_side(frame, _LEFT, _RIGHT)
            if side is None:
                continue
            knee = frame.get(Landmark.LEFT_KNEE if side == "left" else Landmark.RIGHT_KNEE)
            ankle = frame.get(Landmark.LEFT_ANKLE if side == "left" else Landmark.RIGHT_ANKLE)
            left_hip = frame.get(Landmark.LEFT_HIP)
            right_hip = frame.get(Landmark.RIGHT_HIP)
            if knee is None or ankle is None or left_hip is None or right_hip is None:
                continue
            hip_width = abs(left_hip.x - right_hip.x)
            if hip_width < cls._MIN_AVG_HIP_WIDTH_FOR_DRIFT_RATIO:
                continue
            offsets.append((knee.x - ankle.x) / hip_width)

        if len(offsets) < 2:
            return None
        return max(offsets) - min(offsets)
