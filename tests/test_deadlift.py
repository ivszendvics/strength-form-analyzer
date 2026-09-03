"""End-to-end (no video) tests for deadlift analysis using synthetic data."""

import numpy as np

from src.analysis.form_analyzer import analyze_reps
from src.analysis.scoring import RepClassification
from src.exercises.deadlift import DeadliftExercise
from src.pose.landmarks import Landmark, LandmarkPoint, PoseFrame
from src.reps.detector import detect_reps
from tests.synthetic import generate_squat_angle_sequence

DEADLIFT_CONFIG = {
    "rep_detection": {
        "standing_angle": 170,
        "descending_threshold": 155,
        "bottom_angle": 90,
        "ascending_threshold": 110,
        "standing_return_threshold": 160,
        "min_rep_duration_seconds": 0.3,
        "min_frames_in_state": 2,
    },
    "form": {
        "lockout": {"target_angle": 170, "tolerance": 10},
        "trunk": {"max_angle": 55},
        "tempo": {"max_deviation_ratio": 0.4},
        "consistency": {"min_rom_ratio_vs_earlier": 0.8},
        "symmetry": {"max_asymmetry_percent": 15},
        "starting_position": {"max_deviation_degrees": 12},
    },
    "scoring": {
        "lockout": 0.25,
        "trunk": 0.25,
        "tempo": 0.15,
        "stability": 0.15,
        "symmetry": 0.10,
        "starting_position": 0.10,
        "good_threshold": 80,
    },
}


def _synthetic_pose_frames(hip_angles: list[float], timestamps: list[float]) -> list[PoseFrame]:
    """Build minimal PoseFrames whose shoulder-hip-knee geometry reproduces
    the given hip angle exactly, splitting the flexion between trunk lean
    and knee bend (as a real hip-hinge does) rather than putting it all on
    the trunk -- a rigid-trunk/fixed-knee model would force trunk lean to
    equal ``180 - hip_angle`` exactly, which is unrealistically steep for a
    deep hip angle and would trip the trunk-lean rule even for good form.
    """
    trunk_len, thigh_len = 0.4, 0.35
    frames = []
    for i, (angle_deg, ts) in enumerate(zip(hip_angles, timestamps)):
        total_flex = 180.0 - angle_deg
        trunk_lean = np.radians(0.55 * total_flex)
        knee_lean = np.radians(0.45 * total_flex)

        hip_y = 0.5
        shoulder_dir = (np.sin(trunk_lean), -np.cos(trunk_lean))
        knee_dir = (np.sin(knee_lean), np.cos(knee_lean))

        landmarks = {}
        for side, x_base in (("left", 0.45), ("right", 0.55)):
            hip = LandmarkPoint(x=x_base, y=hip_y, z=0.0, visibility=0.95)
            shoulder = LandmarkPoint(
                x=x_base + trunk_len * shoulder_dir[0], y=hip_y + trunk_len * shoulder_dir[1], z=0.0, visibility=0.95
            )
            knee = LandmarkPoint(
                x=x_base + thigh_len * knee_dir[0], y=hip_y + thigh_len * knee_dir[1], z=0.0, visibility=0.95
            )
            ankle = LandmarkPoint(x=knee.x, y=knee.y + 0.35, z=0.0, visibility=0.95)
            prefix = "LEFT" if side == "left" else "RIGHT"
            landmarks[Landmark[f"{prefix}_HIP"]] = hip
            landmarks[Landmark[f"{prefix}_KNEE"]] = knee
            landmarks[Landmark[f"{prefix}_ANKLE"]] = ankle
            landmarks[Landmark[f"{prefix}_SHOULDER"]] = shoulder

        frames.append(PoseFrame(frame_index=i, timestamp_seconds=ts, landmarks=landmarks, person_detected=True))
    return frames


def _build_angle_series(exercise: DeadliftExercise, pose_frames: list[PoseFrame]) -> dict[str, np.ndarray]:
    raw: dict[str, list[float | None]] = {}
    for frame in pose_frames:
        computed = exercise.compute_angles(frame, min_visibility=0.5)
        for key, value in computed.items():
            raw.setdefault(key, []).append(value)
    return {k: np.array([np.nan if v is None else v for v in vals]) for k, vals in raw.items()}


def test_good_deadlift_reaches_lockout_and_scores_high() -> None:
    # generate_squat_angle_sequence models a descent-then-return cycle;
    # reused here for hip angle (170 standing/locked-out -> 85 bent over).
    angles, timestamps = generate_squat_angle_sequence(num_reps=3, standing_angle=170, bottom_angle=85, noise=0.0)
    pose_frames = _synthetic_pose_frames(angles, timestamps)
    exercise = DeadliftExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    reps = detect_reps(angle_series["hip_angle"].tolist(), timestamps, exercise.build_rep_detector_config(DEADLIFT_CONFIG), fps=30.0)
    assert len(reps) == 3

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, DEADLIFT_CONFIG, min_visibility=0.5)
    for result in results:
        assert result.form_score.classification == RepClassification.GOOD


def test_incomplete_lockout_is_flagged() -> None:
    # Standing angle capped at 140 -> never reaches the 170-10=160 lockout threshold.
    angles, timestamps = generate_squat_angle_sequence(num_reps=2, standing_angle=140, bottom_angle=80, noise=0.0)
    pose_frames = _synthetic_pose_frames(angles, timestamps)
    exercise = DeadliftExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    cfg = {
        **DEADLIFT_CONFIG,
        "rep_detection": {**DEADLIFT_CONFIG["rep_detection"], "descending_threshold": 135, "standing_return_threshold": 130},
    }
    reps = detect_reps(angle_series["hip_angle"].tolist(), timestamps, exercise.build_rep_detector_config(cfg), fps=30.0)
    assert len(reps) == 2

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, DEADLIFT_CONFIG, min_visibility=0.5)
    for result in results:
        issue_types = {i.type for i in result.form_score.issues}
        assert "INCOMPLETE_LOCKOUT" in issue_types


def test_excessive_trunk_lean_is_flagged() -> None:
    angles, timestamps = generate_squat_angle_sequence(num_reps=1, standing_angle=170, bottom_angle=40, noise=0.0)
    pose_frames = _synthetic_pose_frames(angles, timestamps)
    exercise = DeadliftExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    reps = detect_reps(angle_series["hip_angle"].tolist(), timestamps, exercise.build_rep_detector_config(DEADLIFT_CONFIG), fps=30.0)
    assert len(reps) == 1

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, DEADLIFT_CONFIG, min_visibility=0.5)
    issue_types = {i.type for i in results[0].form_score.issues}
    assert "EXCESSIVE_TRUNK_LEAN" in issue_types
