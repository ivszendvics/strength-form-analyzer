"""End-to-end (no video) tests for bench press analysis using synthetic data."""

import numpy as np

from src.analysis.form_analyzer import analyze_reps
from src.analysis.scoring import RepClassification
from src.exercises.bench_press import BenchPressExercise
from src.pose.landmarks import Landmark, LandmarkPoint, PoseFrame
from src.reps.detector import detect_reps
from tests.synthetic import generate_squat_angle_sequence

BENCH_PRESS_CONFIG = {
    "rep_detection": {
        "standing_angle": 165,
        "descending_threshold": 155,
        "bottom_angle": 95,
        "ascending_threshold": 115,
        "standing_return_threshold": 155,
        "min_rep_duration_seconds": 0.3,
        "min_frames_in_state": 2,
    },
    "form": {
        "depth": {"target_angle": 85, "tolerance": 10},
        "elbow_position": {"max_angle": 75},
        "tempo": {"max_deviation_ratio": 0.4},
        "consistency": {"min_rom_ratio_vs_earlier": 0.8},
        "symmetry": {"max_asymmetry_percent": 15},
    },
    "scoring": {
        "depth": 0.35,
        "elbow_position": 0.25,
        "tempo": 0.15,
        "stability": 0.15,
        "symmetry": 0.10,
        "good_threshold": 80,
    },
}

_SHOULDER_XY = (0.5, 0.4)
_HIP_XY = (0.5, 0.7)
_UPPER_ARM_LEN = 0.25
_FOREARM_LEN = 0.25


def _arm_points(
    shoulder_xy: tuple[float, float],
    upper_arm_angle_deg: float,
    elbow_angle_deg: float,
    upper_arm_len: float = _UPPER_ARM_LEN,
    forearm_len: float = _FOREARM_LEN,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Place elbow/wrist so ``angle(shoulder, elbow, wrist) ==
    elbow_angle_deg`` exactly, independent of ``upper_arm_angle_deg`` (the
    fixed direction of the upper arm from the shoulder -- how tucked/flared
    it is). Same decoupling trick as tests/test_lunge.py's ``_leg_points``:
    the forearm folds *relative to the upper arm's own direction*, so the
    upper arm's absolute lean never contaminates the measured elbow angle.
    """
    shoulder_x, shoulder_y = shoulder_xy
    upper_arm_rad = np.radians(upper_arm_angle_deg)
    forearm_bend = np.radians(180.0 - elbow_angle_deg)
    elbow_x = shoulder_x + upper_arm_len * np.sin(upper_arm_rad)
    elbow_y = shoulder_y + upper_arm_len * np.cos(upper_arm_rad)
    forearm_angle = upper_arm_rad + forearm_bend
    wrist_x = elbow_x + forearm_len * np.sin(forearm_angle)
    wrist_y = elbow_y + forearm_len * np.cos(forearm_angle)
    return (elbow_x, elbow_y), (wrist_x, wrist_y)


def _synthetic_pose_frames(
    elbow_angles: list[float],
    timestamps: list[float],
    upper_arm_angle_deg: float = 30.0,
) -> list[PoseFrame]:
    frames = []
    for i, (angle_deg, ts) in enumerate(zip(elbow_angles, timestamps)):
        landmarks = {}
        for side, x_offset in (("left", -0.02), ("right", 0.02)):
            shoulder_xy = (_SHOULDER_XY[0] + x_offset, _SHOULDER_XY[1])
            hip_xy = (_HIP_XY[0] + x_offset, _HIP_XY[1])
            elbow, wrist = _arm_points(shoulder_xy, upper_arm_angle_deg, angle_deg)
            prefix = "LEFT" if side == "left" else "RIGHT"
            landmarks[Landmark[f"{prefix}_SHOULDER"]] = LandmarkPoint(x=shoulder_xy[0], y=shoulder_xy[1], z=0.0, visibility=0.95)
            landmarks[Landmark[f"{prefix}_HIP"]] = LandmarkPoint(x=hip_xy[0], y=hip_xy[1], z=0.0, visibility=0.95)
            landmarks[Landmark[f"{prefix}_ELBOW"]] = LandmarkPoint(x=elbow[0], y=elbow[1], z=0.0, visibility=0.95)
            landmarks[Landmark[f"{prefix}_WRIST"]] = LandmarkPoint(x=wrist[0], y=wrist[1], z=0.0, visibility=0.95)

        frames.append(PoseFrame(frame_index=i, timestamp_seconds=ts, landmarks=landmarks, person_detected=True))
    return frames


def _build_angle_series(exercise: BenchPressExercise, pose_frames: list[PoseFrame]) -> dict[str, np.ndarray]:
    raw: dict[str, list[float | None]] = {}
    for frame in pose_frames:
        computed = exercise.compute_angles(frame, min_visibility=0.5)
        for key, value in computed.items():
            raw.setdefault(key, []).append(value)
    return {k: np.array([np.nan if v is None else v for v in vals]) for k, vals in raw.items()}


def test_elbow_angle_tracks_target_independent_of_upper_arm_lean() -> None:
    elbow_angles, timestamps = generate_squat_angle_sequence(num_reps=1, standing_angle=165, bottom_angle=85, noise=0.0)
    for lean in (10.0, 30.0, 60.0):
        pose_frames = _synthetic_pose_frames(elbow_angles, timestamps, upper_arm_angle_deg=lean)
        exercise = BenchPressExercise()
        for i, frame in enumerate(pose_frames):
            angles = exercise.compute_angles(frame, min_visibility=0.5)
            assert np.isclose(angles["elbow_angle"], elbow_angles[i], atol=1e-3)


def test_good_bench_press_scores_high_and_is_classified_good() -> None:
    elbow_angles, timestamps = generate_squat_angle_sequence(num_reps=3, standing_angle=165, bottom_angle=85, noise=0.0)
    pose_frames = _synthetic_pose_frames(elbow_angles, timestamps, upper_arm_angle_deg=30.0)
    exercise = BenchPressExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    reps = detect_reps(
        angle_series["elbow_angle"].tolist(), timestamps, exercise.build_rep_detector_config(BENCH_PRESS_CONFIG), fps=30.0
    )
    assert len(reps) == 3

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, BENCH_PRESS_CONFIG, min_visibility=0.5)
    assert len(results) == 3
    for result in results:
        assert result.form_score.classification == RepClassification.GOOD
        assert result.form_score.overall_score >= 80


def test_shallow_press_flags_insufficient_depth() -> None:
    # Bottom of 140 never reaches the 85+-10 depth target.
    elbow_angles, timestamps = generate_squat_angle_sequence(num_reps=2, standing_angle=165, bottom_angle=140, noise=0.0)
    pose_frames = _synthetic_pose_frames(elbow_angles, timestamps, upper_arm_angle_deg=30.0)
    exercise = BenchPressExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    cfg = {
        **BENCH_PRESS_CONFIG,
        "rep_detection": {**BENCH_PRESS_CONFIG["rep_detection"], "bottom_angle": 145, "ascending_threshold": 150},
    }
    reps = detect_reps(
        angle_series["elbow_angle"].tolist(), timestamps, exercise.build_rep_detector_config(cfg), fps=30.0
    )
    assert len(reps) == 2

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, BENCH_PRESS_CONFIG, min_visibility=0.5)
    for result in results:
        issue_types = {i.type for i in result.form_score.issues}
        assert "INSUFFICIENT_DEPTH" in issue_types
        assert result.form_score.classification == RepClassification.NEEDS_IMPROVEMENT


def test_excessive_elbow_flare_is_flagged() -> None:
    elbow_angles, timestamps = generate_squat_angle_sequence(num_reps=1, standing_angle=165, bottom_angle=85, noise=0.0)
    # A wide upper-arm lean (elbow far from the torso line) should flare
    # the hip-shoulder-elbow angle past the configured threshold.
    pose_frames = _synthetic_pose_frames(elbow_angles, timestamps, upper_arm_angle_deg=85.0)
    exercise = BenchPressExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    reps = detect_reps(
        angle_series["elbow_angle"].tolist(), timestamps, exercise.build_rep_detector_config(BENCH_PRESS_CONFIG), fps=30.0
    )
    assert len(reps) == 1

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, BENCH_PRESS_CONFIG, min_visibility=0.5)
    issue_types = {i.type for i in results[0].form_score.issues}
    assert "EXCESSIVE_ELBOW_FLARE" in issue_types


def test_tucked_elbow_is_not_flagged() -> None:
    elbow_angles, timestamps = generate_squat_angle_sequence(num_reps=1, standing_angle=165, bottom_angle=85, noise=0.0)
    pose_frames = _synthetic_pose_frames(elbow_angles, timestamps, upper_arm_angle_deg=15.0)
    exercise = BenchPressExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    reps = detect_reps(
        angle_series["elbow_angle"].tolist(), timestamps, exercise.build_rep_detector_config(BENCH_PRESS_CONFIG), fps=30.0
    )
    assert len(reps) == 1

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, BENCH_PRESS_CONFIG, min_visibility=0.5)
    issue_types = {i.type for i in results[0].form_score.issues}
    assert "EXCESSIVE_ELBOW_FLARE" not in issue_types
