"""Tests for joint-angle geometry using known configurations."""

import math

from src.biomechanics.angles import (
    angle_between_points,
    joint_angle,
    trunk_angle_from_vertical,
)
from src.pose.landmarks import LandmarkPoint


def test_right_angle() -> None:
    # A at (0,1), B at (0,0) (vertex), C at (1,0) -> 90 degrees.
    angle = angle_between_points((0, 1), (0, 0), (1, 0))
    assert angle is not None
    assert math.isclose(angle, 90.0, abs_tol=1e-6)


def test_straight_line_is_180() -> None:
    # A, B, C collinear with B between A and C -> straight leg, 180 degrees.
    angle = angle_between_points((0, 0), (1, 0), (2, 0))
    assert angle is not None
    assert math.isclose(angle, 180.0, abs_tol=1e-6)


def test_folded_back_is_0() -> None:
    # A and C on the same side of B -> fully folded joint, 0 degrees.
    angle = angle_between_points((0, 0), (1, 0), (0, 0))
    assert angle is not None
    assert math.isclose(angle, 0.0, abs_tol=1e-6)


def test_45_degrees() -> None:
    # A directly above B, C at 45 degrees from vertical.
    angle = angle_between_points((0, 1), (0, 0), (1, 1))
    assert angle is not None
    assert math.isclose(angle, 45.0, abs_tol=1e-6)


def test_coincident_points_return_none() -> None:
    assert angle_between_points((0, 0), (0, 0), (1, 1)) is None


def test_joint_angle_matches_knee_example() -> None:
    hip = LandmarkPoint(x=0.0, y=0.0, z=0.0, visibility=0.9)
    knee = LandmarkPoint(x=0.0, y=1.0, z=0.0, visibility=0.9)
    ankle = LandmarkPoint(x=1.0, y=1.0, z=0.0, visibility=0.9)
    angle = joint_angle(hip, knee, ankle)
    assert angle is not None
    assert math.isclose(angle, 90.0, abs_tol=1e-6)


def test_joint_angle_none_when_missing_landmark() -> None:
    knee = LandmarkPoint(x=0.0, y=1.0, z=0.0, visibility=0.9)
    ankle = LandmarkPoint(x=1.0, y=1.0, z=0.0, visibility=0.9)
    assert joint_angle(None, knee, ankle) is None


def test_joint_angle_none_when_low_confidence() -> None:
    hip = LandmarkPoint(x=0.0, y=0.0, z=0.0, visibility=0.1)
    knee = LandmarkPoint(x=0.0, y=1.0, z=0.0, visibility=0.9)
    ankle = LandmarkPoint(x=1.0, y=1.0, z=0.0, visibility=0.9)
    assert joint_angle(hip, knee, ankle, min_visibility=0.5) is None


def test_trunk_angle_upright_is_zero() -> None:
    hip = LandmarkPoint(x=0.5, y=1.0, z=0.0, visibility=0.9)
    shoulder = LandmarkPoint(x=0.5, y=0.0, z=0.0, visibility=0.9)
    angle = trunk_angle_from_vertical(shoulder, hip)
    assert angle is not None
    assert math.isclose(angle, 0.0, abs_tol=1e-6)


def test_trunk_angle_45_degrees_lean() -> None:
    hip = LandmarkPoint(x=0.0, y=1.0, z=0.0, visibility=0.9)
    shoulder = LandmarkPoint(x=1.0, y=0.0, z=0.0, visibility=0.9)
    angle = trunk_angle_from_vertical(shoulder, hip)
    assert angle is not None
    assert math.isclose(angle, 45.0, abs_tol=1e-6)


def test_trunk_angle_none_when_missing() -> None:
    hip = LandmarkPoint(x=0.0, y=1.0, z=0.0, visibility=0.9)
    assert trunk_angle_from_vertical(None, hip) is None
