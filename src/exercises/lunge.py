"""Lunge exercise: angle definitions, rep-detection driver, and form rules.

Camera assumption: side view of a forward/reverse split-stance lunge, both
legs visible (see README "Camera Assumptions").

Determining which leg is "front" vs "rear" from 2D landmarks alone is
inherently ambiguous without knowing which way the person is facing. This
implementation uses a documented heuristic: on each frame, the leg whose
ankle sits farther horizontally from the midpoint between the hips is
treated as the front (stepped-out) leg, since a split stance by definition
puts one foot well forward/back of the torso and the other closer to
underneath it. This can misclassify legs in a narrow-stance or in-place
lunge; treat front/rear labeling as a best-effort heuristic, not ground
truth.
"""

from __future__ import annotations

from src.analysis.rules import (
    RuleContext,
    RuleResult,
    evaluate_depth,
    evaluate_rep_consistency,
    evaluate_tempo,
    evaluate_trunk_lean,
)
from src.biomechanics.angles import joint_angle, trunk_angle_from_vertical
from src.exercises.base import Exercise, select_more_visible_side
from src.pose.landmarks import Landmark, PoseFrame
from src.reps.detector import RepDetectorConfig

_LEFT = [Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE, Landmark.LEFT_SHOULDER]
_RIGHT = [Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE, Landmark.RIGHT_SHOULDER]


def _identify_front_side(frame: PoseFrame, min_visibility: float) -> str | None:
    left_hip = frame.get(Landmark.LEFT_HIP)
    right_hip = frame.get(Landmark.RIGHT_HIP)
    left_ankle = frame.get(Landmark.LEFT_ANKLE)
    right_ankle = frame.get(Landmark.RIGHT_ANKLE)
    if None in (left_hip, right_hip, left_ankle, right_ankle):
        return None
    if min(left_hip.visibility, right_hip.visibility, left_ankle.visibility, right_ankle.visibility) < min_visibility:
        return None

    hip_mid_x = (left_hip.x + right_hip.x) / 2.0
    left_offset = abs(left_ankle.x - hip_mid_x)
    right_offset = abs(right_ankle.x - hip_mid_x)
    return "left" if left_offset >= right_offset else "right"


class LungeExercise(Exercise):
    name = "lunge"

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

        front_side = _identify_front_side(frame, min_visibility)
        if front_side == "left":
            angles["front_knee_angle"] = left_knee
            angles["rear_knee_angle"] = right_knee
            hip_landmark, knee_landmark, shoulder_landmark = Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_SHOULDER
        elif front_side == "right":
            angles["front_knee_angle"] = right_knee
            angles["rear_knee_angle"] = left_knee
            hip_landmark, knee_landmark, shoulder_landmark = Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_SHOULDER
        else:
            angles["front_knee_angle"] = None
            angles["rear_knee_angle"] = None
            hip_landmark = knee_landmark = shoulder_landmark = None

        if hip_landmark is not None:
            angles["hip_angle"] = joint_angle(frame.get(shoulder_landmark), frame.get(hip_landmark), frame.get(knee_landmark), min_visibility)
        else:
            angles["hip_angle"] = None

        side = select_more_visible_side(frame, _LEFT, _RIGHT)
        if side == "left":
            angles["trunk_angle"] = trunk_angle_from_vertical(frame.get(Landmark.LEFT_SHOULDER), frame.get(Landmark.LEFT_HIP), min_visibility)
        elif side == "right":
            angles["trunk_angle"] = trunk_angle_from_vertical(frame.get(Landmark.RIGHT_SHOULDER), frame.get(Landmark.RIGHT_HIP), min_visibility)
        else:
            angles["trunk_angle"] = None

        return angles

    @property
    def primary_angle_name(self) -> str:
        return "front_knee_angle"

    def build_rep_detector_config(self, raw_config: dict) -> RepDetectorConfig:
        rd = raw_config.get("rep_detection", {})
        return RepDetectorConfig(
            standing_angle=rd.get("standing_angle", 165.0),
            descending_threshold=rd.get("descending_threshold", 155.0),
            bottom_angle=rd.get("bottom_angle", 100.0),
            ascending_threshold=rd.get("ascending_threshold", 115.0),
            standing_return_threshold=rd.get("standing_return_threshold", 155.0),
            min_rep_duration_seconds=rd.get("min_rep_duration_seconds", 0.5),
            min_frames_in_state=rd.get("min_frames_in_state", 2),
        )

    def evaluate_form(self, context: RuleContext, raw_config: dict) -> list[RuleResult]:
        form_cfg = raw_config.get("form", {})
        depth_cfg = form_cfg.get("depth", {})
        rear_depth_cfg = form_cfg.get("rear_depth", {})
        trunk_cfg = form_cfg.get("trunk", {})
        tempo_cfg = form_cfg.get("tempo", {})

        rear_result = evaluate_depth(
            context,
            angle_key="rear_knee_angle",
            target_angle=rear_depth_cfg.get("target_angle", 100.0),
            tolerance=rear_depth_cfg.get("tolerance", 12.0),
        )
        rear_result.component = "rear_depth"
        for issue in rear_result.issues:
            issue.type = "REAR_KNEE_INSUFFICIENT_DROP"

        return [
            evaluate_depth(
                context,
                angle_key="front_knee_angle",
                target_angle=depth_cfg.get("target_angle", 95.0),
                tolerance=depth_cfg.get("tolerance", 10.0),
            ),
            rear_result,
            evaluate_trunk_lean(
                context,
                max_trunk_angle_threshold=trunk_cfg.get("max_angle", 35.0),
            ),
            evaluate_tempo(
                context,
                max_deviation_ratio=tempo_cfg.get("max_deviation_ratio", 0.4),
            ),
            evaluate_rep_consistency(
                context,
                rom_key="front_knee_angle",
                min_rom_ratio_vs_earlier=form_cfg.get("consistency", {}).get("min_rom_ratio_vs_earlier", 0.8),
            ),
        ]
