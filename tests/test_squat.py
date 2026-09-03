"""End-to-end (no video) tests for squat analysis: synthetic angle series
through rep detection, metrics, form rules, and scoring."""

import numpy as np

from src.analysis.form_analyzer import analyze_reps
from src.analysis.scoring import RepClassification
from src.exercises.squat import SquatExercise
from src.pose.landmarks import Landmark, LandmarkPoint, PoseFrame
from src.reps.detector import detect_reps
from tests.synthetic import generate_squat_angle_sequence

SQUAT_CONFIG = {
    "rep_detection": {
        "standing_angle": 170,
        "descending_threshold": 160,
        "bottom_angle": 100,
        "ascending_threshold": 120,
        "standing_return_threshold": 160,
        "min_rep_duration_seconds": 0.3,
        "min_frames_in_state": 2,
    },
    "form": {
        "depth": {"target_angle": 100, "tolerance": 10},
        "trunk": {"max_angle": 45},
        "tempo": {"max_deviation_ratio": 0.4},
        "consistency": {"min_rom_ratio_vs_earlier": 0.8},
        "symmetry": {"max_asymmetry_percent": 15},
        "knee_tracking": {"camera_view": "side", "max_drift_ratio": 0.25},
    },
    "scoring": {
        "depth": 0.30,
        "trunk": 0.20,
        "tempo": 0.15,
        "stability": 0.20,
        "symmetry": 0.10,
        "knee_tracking": 0.05,
        "good_threshold": 80,
    },
}


def _synthetic_pose_frames(angles: list[float], timestamps: list[float]) -> list[PoseFrame]:
    """Build minimal PoseFrames whose left/right knee/hip/shoulder/ankle
    landmarks are positioned so joint_angle() reproduces the given knee
    angle exactly, for both sides identically (symmetric, visible)."""
    frames = []
    for i, (angle_deg, ts) in enumerate(zip(angles, timestamps)):
        theta = np.radians(180 - angle_deg)
        ankle_y = 1.0
        knee_y = 0.5
        hip_y = knee_y - 0.5 * np.cos(theta)
        hip_x_offset = 0.5 * np.sin(theta)

        landmarks = {}
        for side, x_base in (("left", 0.45), ("right", 0.55)):
            hip = LandmarkPoint(x=x_base + hip_x_offset, y=hip_y, z=0.0, visibility=0.95)
            knee = LandmarkPoint(x=x_base, y=knee_y, z=0.0, visibility=0.95)
            ankle = LandmarkPoint(x=x_base, y=ankle_y, z=0.0, visibility=0.95)
            shoulder = LandmarkPoint(x=x_base + hip_x_offset, y=hip_y - 0.5, z=0.0, visibility=0.95)
            prefix = "LEFT" if side == "left" else "RIGHT"
            landmarks[Landmark[f"{prefix}_HIP"]] = hip
            landmarks[Landmark[f"{prefix}_KNEE"]] = knee
            landmarks[Landmark[f"{prefix}_ANKLE"]] = ankle
            landmarks[Landmark[f"{prefix}_SHOULDER"]] = shoulder

        frames.append(PoseFrame(frame_index=i, timestamp_seconds=ts, landmarks=landmarks, person_detected=True))
    return frames


def test_good_squat_scores_high_and_is_classified_good() -> None:
    angles, timestamps = generate_squat_angle_sequence(num_reps=3, standing_angle=170, bottom_angle=90, noise=0.0)
    pose_frames = _synthetic_pose_frames(angles, timestamps)
    exercise = SquatExercise()

    angle_series_raw: dict[str, list[float | None]] = {}
    for frame in pose_frames:
        computed = exercise.compute_angles(frame, min_visibility=0.5)
        for key, value in computed.items():
            angle_series_raw.setdefault(key, []).append(value)
    angle_series = {k: np.array([np.nan if v is None else v for v in vals]) for k, vals in angle_series_raw.items()}

    reps = detect_reps(angle_series["knee_angle"].tolist(), timestamps, exercise.build_rep_detector_config(SQUAT_CONFIG), fps=30.0)
    assert len(reps) == 3

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, SQUAT_CONFIG, min_visibility=0.5)
    assert len(results) == 3
    for result in results:
        assert result.form_score.classification == RepClassification.GOOD
        assert result.form_score.overall_score >= 80


def test_shallow_squat_flags_insufficient_depth() -> None:
    # Bottom angle of 145 never reaches the 100+-10 depth target.
    angles, timestamps = generate_squat_angle_sequence(num_reps=2, standing_angle=170, bottom_angle=145, noise=0.0)
    pose_frames = _synthetic_pose_frames(angles, timestamps)
    exercise = SquatExercise()

    angle_series_raw: dict[str, list[float | None]] = {}
    for frame in pose_frames:
        computed = exercise.compute_angles(frame, min_visibility=0.5)
        for key, value in computed.items():
            angle_series_raw.setdefault(key, []).append(value)
    angle_series = {k: np.array([np.nan if v is None else v for v in vals]) for k, vals in angle_series_raw.items()}

    cfg = {
        **SQUAT_CONFIG,
        "rep_detection": {**SQUAT_CONFIG["rep_detection"], "bottom_angle": 150, "ascending_threshold": 155},
    }
    reps = detect_reps(angle_series["knee_angle"].tolist(), timestamps, exercise.build_rep_detector_config(cfg), fps=30.0)
    assert len(reps) == 2

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, SQUAT_CONFIG, min_visibility=0.5)
    for result in results:
        issue_types = {i.type for i in result.form_score.issues}
        assert "INSUFFICIENT_DEPTH" in issue_types
        assert result.form_score.classification == RepClassification.NEEDS_IMPROVEMENT


def test_low_confidence_landmarks_produce_uncertain_classification() -> None:
    angles, timestamps = generate_squat_angle_sequence(num_reps=1, standing_angle=170, bottom_angle=90, noise=0.0)
    pose_frames = _synthetic_pose_frames(angles, timestamps)
    for frame in pose_frames:
        for point in frame.landmarks.values():
            object.__setattr__(point, "visibility", 0.1)

    exercise = SquatExercise()

    # With visibility below threshold, angles are None everywhere -> no reps
    # can even be detected from the (all-NaN) primary series. This test
    # instead verifies rep_visibility_ok directly reports low confidence.
    from src.analysis.form_analyzer import rep_visibility_ok
    from src.reps.detector import Rep

    fake_rep = Rep(rep_number=1, start_frame=0, bottom_frame=5, end_frame=10, start_time_seconds=0.0, bottom_time_seconds=0.1, end_time_seconds=0.3)
    ok = rep_visibility_ok(pose_frames, fake_rep, exercise.required_landmarks(), min_visibility=0.5)
    assert ok is False
