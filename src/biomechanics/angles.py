"""Geometric joint-angle calculations.

All angles are computed in 2D, using the ``(x, y)`` plane of the pose
estimator's normalized image coordinates. This is a deliberate
simplification: MediaPipe's ``z`` is a rough, less-reliable depth estimate,
and single-camera 2D angles are themselves only a *projection* of the true
3D joint angle (see README "Limitations"). No assumption is made about which
direction the person is facing -- the angle at a joint is computed the same
way regardless of whether the person is filmed from their left or right
side; camera-side-specific interpretation (e.g. "forward" trunk lean) is
left to the exercise-analysis layer, which knows the intended camera setup.
"""

from __future__ import annotations

import math

from src.pose.landmarks import LandmarkPoint

# Below this visibility/confidence, a landmark is treated as unusable for
# angle computation rather than trusted at face value.
DEFAULT_MIN_VISIBILITY = 0.5

# Angles computed from noisy landmarks can occasionally come out as
# geometrically impossible for a human joint (e.g. a knee bending past what
# any joint can do); clamp reported angles to this range and treat values
# outside it as unreliable.
MIN_PLAUSIBLE_ANGLE_DEGREES = 0.0
MAX_PLAUSIBLE_ANGLE_DEGREES = 180.0


def angle_between_points(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float | None:
    """Compute the angle ABC (at vertex B) in degrees, in [0, 180].

    Args:
        a: First point (e.g. hip).
        b: Vertex point (e.g. knee).
        c: Third point (e.g. ankle).

    Returns:
        The angle in degrees, or ``None`` if the points are coincident
        (zero-length vector), which makes the angle undefined.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    ba_norm = math.hypot(*ba)
    bc_norm = math.hypot(*bc)
    if ba_norm == 0.0 or bc_norm == 0.0:
        return None

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    cos_angle = dot / (ba_norm * bc_norm)
    # Guard against floating point drift pushing slightly outside [-1, 1].
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def joint_angle(
    a: LandmarkPoint | None,
    b: LandmarkPoint | None,
    c: LandmarkPoint | None,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
) -> float | None:
    """Compute the angle ABC from three landmarks, honoring confidence.

    Returns ``None`` (rather than a possibly-misleading number) if any
    landmark is missing or below ``min_visibility``, or if the resulting
    angle falls outside the plausible range for a human joint.
    """
    if a is None or b is None or c is None:
        return None
    if min(a.visibility, b.visibility, c.visibility) < min_visibility:
        return None

    angle = angle_between_points(a.as_xy(), b.as_xy(), c.as_xy())
    if angle is None:
        return None
    if not (MIN_PLAUSIBLE_ANGLE_DEGREES <= angle <= MAX_PLAUSIBLE_ANGLE_DEGREES):
        return None
    return angle


def trunk_angle_from_vertical(
    shoulder: LandmarkPoint | None,
    hip: LandmarkPoint | None,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
) -> float | None:
    """Forward trunk inclination, in degrees from vertical (0 = upright).

    Computed as the angle between the shoulder-to-hip vector and the
    downward vertical axis. Image ``y`` increases downward, so "vertical"
    is the ``(0, 1)`` direction from hip toward shoulder inverted, i.e. we
    measure the hip->shoulder vector against straight-up ``(0, -1)``.
    """
    if shoulder is None or hip is None:
        return None
    if min(shoulder.visibility, hip.visibility) < min_visibility:
        return None

    dx = shoulder.x - hip.x
    dy = shoulder.y - hip.y
    segment_length = math.hypot(dx, dy)
    if segment_length == 0.0:
        return None

    # Angle between hip->shoulder vector and straight-up (0, -1).
    cos_angle = (-dy) / segment_length
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))
