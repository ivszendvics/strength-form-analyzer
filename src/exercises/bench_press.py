"""Bench press exercise: angle definitions, rep-detection driver, and form rules.

Camera assumption: side view of the bench, full pressing arm and torso
visible (see README "Camera Assumptions"). Unlike the standing exercises,
the lifter is horizontal, so a "trunk lean" reading doesn't apply. The
closest analogous form cue visible from a side camera is how far the elbow
travels from the torso during the press -- commonly coached as an "elbow
tuck" angle -- measured here as the angle at the shoulder between the
hip-shoulder line and the shoulder-elbow line.

Rep detection reuses the same state machine as squat/lunge (not deadlift's
reversed pattern): a bench press starts and ends at lockout (elbows
extended, large elbow angle) with a single dip to the chest in between.
"""

from __future__ import annotations

import numpy as np

from src.analysis.rules import (
    RuleContext,
    RuleResult,
    evaluate_depth,
    evaluate_elbow_flare,
    evaluate_rep_consistency,
    evaluate_symmetry,
    evaluate_tempo,
)
from src.biomechanics.angles import joint_angle
from src.biomechanics.metrics import RepMetrics, asymmetry_percent
from src.exercises.base import Exercise, select_more_visible_side
from src.pose.landmarks import Landmark, PoseFrame
from src.reps.detector import Rep, RepDetectorConfig

_LEFT = [Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST, Landmark.LEFT_HIP]
_RIGHT = [Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST, Landmark.RIGHT_HIP]


class BenchPressExercise(Exercise):
    name = "bench_press"

    def required_landmarks(self) -> list[Landmark]:
        return [
            Landmark.LEFT_SHOULDER,
            Landmark.RIGHT_SHOULDER,
            Landmark.LEFT_ELBOW,
            Landmark.RIGHT_ELBOW,
            Landmark.LEFT_WRIST,
            Landmark.RIGHT_WRIST,
            Landmark.LEFT_HIP,
            Landmark.RIGHT_HIP,
        ]

    def compute_angles(self, frame: PoseFrame, min_visibility: float) -> dict[str, float | None]:
        angles: dict[str, float | None] = {}

        left_elbow = joint_angle(frame.get(Landmark.LEFT_SHOULDER), frame.get(Landmark.LEFT_ELBOW), frame.get(Landmark.LEFT_WRIST), min_visibility)
        right_elbow = joint_angle(frame.get(Landmark.RIGHT_SHOULDER), frame.get(Landmark.RIGHT_ELBOW), frame.get(Landmark.RIGHT_WRIST), min_visibility)
        angles["left_elbow_angle"] = left_elbow
        angles["right_elbow_angle"] = right_elbow

        left_shoulder_angle = joint_angle(frame.get(Landmark.LEFT_HIP), frame.get(Landmark.LEFT_SHOULDER), frame.get(Landmark.LEFT_ELBOW), min_visibility)
        right_shoulder_angle = joint_angle(frame.get(Landmark.RIGHT_HIP), frame.get(Landmark.RIGHT_SHOULDER), frame.get(Landmark.RIGHT_ELBOW), min_visibility)
        angles["left_shoulder_angle"] = left_shoulder_angle
        angles["right_shoulder_angle"] = right_shoulder_angle

        side = select_more_visible_side(frame, _LEFT, _RIGHT)
        if side == "left":
            angles["elbow_angle"] = left_elbow
            angles["shoulder_angle"] = left_shoulder_angle
        elif side == "right":
            angles["elbow_angle"] = right_elbow
            angles["shoulder_angle"] = right_shoulder_angle
        else:
            angles["elbow_angle"] = None
            angles["shoulder_angle"] = None

        return angles

    @property
    def primary_angle_name(self) -> str:
        return "elbow_angle"

    def build_rep_detector_config(self, raw_config: dict) -> RepDetectorConfig:
        rd = raw_config.get("rep_detection", {})
        return RepDetectorConfig(
            standing_angle=rd.get("standing_angle", 165.0),
            descending_threshold=rd.get("descending_threshold", 155.0),
            bottom_angle=rd.get("bottom_angle", 95.0),
            ascending_threshold=rd.get("ascending_threshold", 115.0),
            standing_return_threshold=rd.get("standing_return_threshold", 155.0),
            min_rep_duration_seconds=rd.get("min_rep_duration_seconds", 0.5),
            min_frames_in_state=rd.get("min_frames_in_state", 2),
        )

    def evaluate_form(self, context: RuleContext, raw_config: dict) -> list[RuleResult]:
        form_cfg = raw_config.get("form", {})
        depth_cfg = form_cfg.get("depth", {})
        elbow_position_cfg = form_cfg.get("elbow_position", {})
        tempo_cfg = form_cfg.get("tempo", {})
        symmetry_cfg = form_cfg.get("symmetry", {})

        return [
            evaluate_depth(
                context,
                angle_key="elbow_angle",
                target_angle=depth_cfg.get("target_angle", 85.0),
                tolerance=depth_cfg.get("tolerance", 10.0),
            ),
            evaluate_elbow_flare(
                context,
                max_shoulder_angle_threshold=elbow_position_cfg.get("max_angle", 75.0),
            ),
            evaluate_tempo(
                context,
                max_deviation_ratio=tempo_cfg.get("max_deviation_ratio", 0.4),
            ),
            evaluate_rep_consistency(
                context,
                rom_key="elbow_angle",
                min_rom_ratio_vs_earlier=form_cfg.get("consistency", {}).get("min_rom_ratio_vs_earlier", 0.8),
            ),
            evaluate_symmetry(
                context,
                max_asymmetry_percent=symmetry_cfg.get("max_asymmetry_percent", 15.0),
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

        left = angle_series.get("left_elbow_angle")
        right = angle_series.get("right_elbow_angle")
        if left is not None and right is not None:
            asym = asymmetry_percent(left, right, rep.start_frame, rep.end_frame)
            if asym is not None:
                extra["asymmetry_percent"] = asym

        return extra
