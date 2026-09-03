"""Deadlift exercise: angle definitions, rep-detection driver, and form rules.

Camera assumption: side view, full body and the bar path visible (see
README "Camera Assumptions"). Rep detection is driven by hip angle: a
deadlift is fundamentally a hip hinge, and hip angle is a more direct proxy
for "how much the lifter has bent over" than knee angle, which changes less
across the movement. A rep's "bottom" here is the most flexed (bent-over)
position and its "top" is lockout (full hip/knee extension) -- opposite in
spirit to a squat's rep boundaries but handled by the same generic state
machine, since it only needs consistent standing/bottom angle thresholds.
"""

from __future__ import annotations

import numpy as np

from src.analysis.rules import (
    RuleContext,
    RuleResult,
    evaluate_extension,
    evaluate_rep_consistency,
    evaluate_symmetry,
    evaluate_tempo,
    evaluate_trunk_lean,
    evaluate_value_consistency,
)
from src.biomechanics.angles import joint_angle, trunk_angle_from_vertical
from src.biomechanics.metrics import RepMetrics, asymmetry_percent
from src.exercises.base import Exercise, select_more_visible_side
from src.pose.landmarks import Landmark, PoseFrame
from src.reps.detector import Rep, RepDetectorConfig

_LEFT = [Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE]
_RIGHT = [Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE]


class DeadliftExercise(Exercise):
    name = "deadlift"

    def required_landmarks(self) -> list[Landmark]:
        return [
            Landmark.LEFT_SHOULDER,
            Landmark.RIGHT_SHOULDER,
            Landmark.LEFT_HIP,
            Landmark.RIGHT_HIP,
            Landmark.LEFT_KNEE,
            Landmark.RIGHT_KNEE,
            Landmark.LEFT_ANKLE,
            Landmark.RIGHT_ANKLE,
        ]

    def compute_angles(self, frame: PoseFrame, min_visibility: float) -> dict[str, float | None]:
        angles: dict[str, float | None] = {}

        left_hip = joint_angle(frame.get(Landmark.LEFT_SHOULDER), frame.get(Landmark.LEFT_HIP), frame.get(Landmark.LEFT_KNEE), min_visibility)
        right_hip = joint_angle(frame.get(Landmark.RIGHT_SHOULDER), frame.get(Landmark.RIGHT_HIP), frame.get(Landmark.RIGHT_KNEE), min_visibility)
        angles["left_hip_angle"] = left_hip
        angles["right_hip_angle"] = right_hip

        left_knee = joint_angle(frame.get(Landmark.LEFT_HIP), frame.get(Landmark.LEFT_KNEE), frame.get(Landmark.LEFT_ANKLE), min_visibility)
        right_knee = joint_angle(frame.get(Landmark.RIGHT_HIP), frame.get(Landmark.RIGHT_KNEE), frame.get(Landmark.RIGHT_ANKLE), min_visibility)
        angles["left_knee_angle"] = left_knee
        angles["right_knee_angle"] = right_knee

        side = select_more_visible_side(frame, _LEFT, _RIGHT)
        if side == "left":
            angles["hip_angle"] = left_hip
            angles["knee_angle"] = left_knee
            angles["trunk_angle"] = trunk_angle_from_vertical(frame.get(Landmark.LEFT_SHOULDER), frame.get(Landmark.LEFT_HIP), min_visibility)
        elif side == "right":
            angles["hip_angle"] = right_hip
            angles["knee_angle"] = right_knee
            angles["trunk_angle"] = trunk_angle_from_vertical(frame.get(Landmark.RIGHT_SHOULDER), frame.get(Landmark.RIGHT_HIP), min_visibility)
        else:
            angles["hip_angle"] = None
            angles["knee_angle"] = None
            angles["trunk_angle"] = None

        return angles

    @property
    def primary_angle_name(self) -> str:
        return "hip_angle"

    def build_rep_detector_config(self, raw_config: dict) -> RepDetectorConfig:
        rd = raw_config.get("rep_detection", {})
        return RepDetectorConfig(
            standing_angle=rd.get("standing_angle", 170.0),
            descending_threshold=rd.get("descending_threshold", 155.0),
            bottom_angle=rd.get("bottom_angle", 90.0),
            ascending_threshold=rd.get("ascending_threshold", 110.0),
            standing_return_threshold=rd.get("standing_return_threshold", 160.0),
            min_rep_duration_seconds=rd.get("min_rep_duration_seconds", 0.5),
            min_frames_in_state=rd.get("min_frames_in_state", 2),
        )

    def evaluate_form(self, context: RuleContext, raw_config: dict) -> list[RuleResult]:
        form_cfg = raw_config.get("form", {})
        lockout_cfg = form_cfg.get("lockout", {})
        trunk_cfg = form_cfg.get("trunk", {})
        tempo_cfg = form_cfg.get("tempo", {})
        symmetry_cfg = form_cfg.get("symmetry", {})
        starting_position_cfg = form_cfg.get("starting_position", {})

        return [
            evaluate_extension(
                context,
                angle_key="hip_angle",
                target_angle=lockout_cfg.get("target_angle", 170.0),
                tolerance=lockout_cfg.get("tolerance", 10.0),
            ),
            evaluate_trunk_lean(
                context,
                max_trunk_angle_threshold=trunk_cfg.get("max_angle", 55.0),
            ),
            evaluate_tempo(
                context,
                max_deviation_ratio=tempo_cfg.get("max_deviation_ratio", 0.4),
            ),
            evaluate_rep_consistency(
                context,
                rom_key="hip_angle",
                min_rom_ratio_vs_earlier=form_cfg.get("consistency", {}).get("min_rom_ratio_vs_earlier", 0.8),
            ),
            evaluate_symmetry(
                context,
                max_asymmetry_percent=symmetry_cfg.get("max_asymmetry_percent", 15.0),
            ),
            evaluate_value_consistency(
                context,
                extra_key="start_hip_angle",
                max_deviation_degrees=starting_position_cfg.get("max_deviation_degrees", 12.0),
                component_name="starting_position",
                issue_type="INCONSISTENT_STARTING_POSITION",
                description="starting hip angle",
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

        left = angle_series.get("left_hip_angle")
        right = angle_series.get("right_hip_angle")
        if left is not None and right is not None:
            asym = asymmetry_percent(left, right, rep.start_frame, rep.end_frame)
            if asym is not None:
                extra["asymmetry_percent"] = asym

        hip_series = angle_series.get("hip_angle")
        if hip_series is not None and rep.start_frame < len(hip_series):
            start_value = hip_series[rep.start_frame]
            if not np.isnan(start_value):
                extra["start_hip_angle"] = float(start_value)

        return extra
