"""Per-frame annotated video overlay: skeleton, angles, rep/phase/score HUD."""

from __future__ import annotations

import cv2
import numpy as np

from src.pose.landmarks import Landmark, PoseFrame

# Skeleton bones drawn as (landmark_a, landmark_b) pairs, restricted to the
# subset of landmarks this project tracks.
_SKELETON_BONES: list[tuple[Landmark, Landmark]] = [
    (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
    (Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
    (Landmark.LEFT_ANKLE, Landmark.LEFT_HEEL),
    (Landmark.LEFT_ANKLE, Landmark.LEFT_FOOT_INDEX),
    (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    (Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_HEEL),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_FOOT_INDEX),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
    (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
    (Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
]

_SKELETON_COLOR = (0, 220, 0)
_LOW_CONFIDENCE_COLOR = (0, 165, 255)
_TEXT_COLOR = (255, 255, 255)
_WARNING_COLOR = (0, 140, 255)
_GOOD_COLOR = (0, 200, 0)
_NEEDS_IMPROVEMENT_COLOR = (0, 140, 255)
_UNCERTAIN_COLOR = (180, 180, 180)


def draw_skeleton(frame: np.ndarray, pose_frame: PoseFrame, min_visibility: float) -> None:
    """Draw skeleton bones and joint dots in-place on a BGR frame."""
    if not pose_frame.person_detected:
        return
    h, w = frame.shape[:2]

    for a, b in _SKELETON_BONES:
        pa, pb = pose_frame.get(a), pose_frame.get(b)
        if pa is None or pb is None:
            continue
        color = _SKELETON_COLOR if min(pa.visibility, pb.visibility) >= min_visibility else _LOW_CONFIDENCE_COLOR
        pt_a = (int(pa.x * w), int(pa.y * h))
        pt_b = (int(pb.x * w), int(pb.y * h))
        cv2.line(frame, pt_a, pt_b, color, 2, cv2.LINE_AA)

    for point in pose_frame.landmarks.values():
        color = _SKELETON_COLOR if point.visibility >= min_visibility else _LOW_CONFIDENCE_COLOR
        pt = (int(point.x * w), int(point.y * h))
        cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)


def _put_text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.6) -> int:
    """Draw a line of text with a translucent background box; returns the y
    coordinate the *next* line should start at."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return y + th + 14


def draw_hud(
    frame: np.ndarray,
    rep_number: int | None,
    phase: str,
    angle_values: dict[str, float | None],
    classification: str | None,
    overall_score: float | None,
    warnings: list[str],
    origin: tuple[int, int] = (15, 30),
) -> None:
    """Draw the text HUD: rep number, phase, live angles, form score, warnings."""
    x, y = origin

    y = _put_text(frame, f"REP: {rep_number if rep_number is not None else '-'}", (x, y), _TEXT_COLOR)
    y = _put_text(frame, f"PHASE: {phase}", (x, y), _TEXT_COLOR)
    y += 6

    for name, value in angle_values.items():
        label = name.replace("_", " ").title()
        text = f"{label}: {value:.0f} deg" if value is not None else f"{label}: --"
        y = _put_text(frame, text, (x, y), _TEXT_COLOR, scale=0.55)

    y += 6
    if classification is not None:
        color = {
            "GOOD": _GOOD_COLOR,
            "NEEDS_IMPROVEMENT": _NEEDS_IMPROVEMENT_COLOR,
            "UNCERTAIN": _UNCERTAIN_COLOR,
        }.get(classification, _TEXT_COLOR)
        score_text = f"FORM: {classification.replace('_', ' ')}"
        if overall_score is not None:
            score_text += f" ({overall_score:.0f}/100)"
        y = _put_text(frame, score_text, (x, y), color)

    for warning in warnings:
        y = _put_text(frame, f"! {warning}", (x, y), _WARNING_COLOR, scale=0.5)
