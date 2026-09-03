"""End-to-end (no video) tests for lunge analysis using synthetic data."""

import numpy as np

from src.analysis.form_analyzer import analyze_reps
from src.analysis.scoring import RepClassification
from src.exercises.lunge import LungeExercise
from src.pose.landmarks import Landmark, LandmarkPoint, PoseFrame
from src.reps.detector import detect_reps
from tests.synthetic import generate_squat_angle_sequence

LUNGE_CONFIG = {
    "rep_detection": {
        "standing_angle": 165,
        "descending_threshold": 155,
        "bottom_angle": 100,
        "ascending_threshold": 115,
        "standing_return_threshold": 155,
        "min_rep_duration_seconds": 0.3,
        "min_frames_in_state": 2,
    },
    "form": {
        "depth": {"target_angle": 95, "tolerance": 10},
        "rear_depth": {"target_angle": 100, "tolerance": 12},
        "trunk": {"max_angle": 35},
        "tempo": {"max_deviation_ratio": 0.4},
        "consistency": {"min_rom_ratio_vs_earlier": 0.8},
    },
    "scoring": {
        "depth": 0.35,
        "rear_depth": 0.20,
        "trunk": 0.20,
        "tempo": 0.10,
        "stability": 0.15,
        "good_threshold": 80,
    },
}

# The two hip landmarks are kept close together (representing one pelvis,
# as in reality) -- what makes a leg "front" vs "rear" in the synthetic
# frames is each leg's fixed thigh lean, not hip separation.
_FRONT_HIP_X = 0.48
_REAR_HIP_X = 0.52
_FRONT_THIGH_LEAN_DEG = 65.0  # front leg stepped well forward
_REAR_THIGH_LEAN_DEG = 2.0  # rear leg stays nearly under the hip
_THIGH_LEN = 0.35
_SHANK_LEN = 0.35


def _leg_points(
    hip_xy: tuple[float, float],
    thigh_lean_deg: float,
    knee_angle_deg: float,
    fold_sign: float,
    thigh_len: float = _THIGH_LEN,
    shank_len: float = _SHANK_LEN,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Place knee/ankle so ``angle(hip, knee, ankle) == knee_angle_deg``
    exactly, independent of ``thigh_lean_deg``.

    The thigh direction (hip->knee) is set by the fixed stance lean; the
    shank folds *relative to the thigh's own direction* by the amount
    needed to hit the target knee angle, so the absolute stance lean never
    contaminates the angle that gets measured. ``fold_sign`` picks which
    way the shank folds (+1 forward, -1 backward) -- the front leg's shin
    drives forward/down through a lunge, the rear leg's drops back/down.
    """
    hip_x, hip_y = hip_xy
    thigh_lean = np.radians(thigh_lean_deg)
    shank_bend = np.radians(180.0 - knee_angle_deg)
    knee_x = hip_x + thigh_len * np.sin(thigh_lean)
    knee_y = hip_y + thigh_len * np.cos(thigh_lean)
    shank_angle = thigh_lean + fold_sign * shank_bend
    ankle_x = knee_x + shank_len * np.sin(shank_angle)
    ankle_y = knee_y + shank_len * np.cos(shank_angle)
    return (knee_x, knee_y), (ankle_x, ankle_y)


def _synthetic_pose_frames(
    front_angles: list[float],
    rear_angles: list[float],
    timestamps: list[float],
    trunk_lean_deg: float = 0.0,
) -> list[PoseFrame]:
    hip_y = 0.3
    trunk_lean = np.radians(trunk_lean_deg)
    shoulder_x_offset = 0.5 * np.sin(trunk_lean)
    shoulder_y = hip_y - 0.5 * np.cos(trunk_lean)

    frames = []
    for i, (fa, ra, ts) in enumerate(zip(front_angles, rear_angles, timestamps)):
        front_knee, front_ankle = _leg_points((_FRONT_HIP_X, hip_y), _FRONT_THIGH_LEAN_DEG, fa, fold_sign=1.0)
        rear_knee, rear_ankle = _leg_points((_REAR_HIP_X, hip_y), _REAR_THIGH_LEAN_DEG, ra, fold_sign=-1.0)

        landmarks = {
            Landmark.LEFT_HIP: LandmarkPoint(x=_FRONT_HIP_X, y=hip_y, z=0.0, visibility=0.95),
            Landmark.LEFT_KNEE: LandmarkPoint(x=front_knee[0], y=front_knee[1], z=0.0, visibility=0.95),
            Landmark.LEFT_ANKLE: LandmarkPoint(x=front_ankle[0], y=front_ankle[1], z=0.0, visibility=0.95),
            Landmark.LEFT_SHOULDER: LandmarkPoint(
                x=_FRONT_HIP_X + shoulder_x_offset, y=shoulder_y, z=0.0, visibility=0.95
            ),
            Landmark.RIGHT_HIP: LandmarkPoint(x=_REAR_HIP_X, y=hip_y, z=0.0, visibility=0.95),
            Landmark.RIGHT_KNEE: LandmarkPoint(x=rear_knee[0], y=rear_knee[1], z=0.0, visibility=0.95),
            Landmark.RIGHT_ANKLE: LandmarkPoint(x=rear_ankle[0], y=rear_ankle[1], z=0.0, visibility=0.95),
            Landmark.RIGHT_SHOULDER: LandmarkPoint(
                x=_REAR_HIP_X + shoulder_x_offset, y=shoulder_y, z=0.0, visibility=0.95
            ),
        }
        frames.append(PoseFrame(frame_index=i, timestamp_seconds=ts, landmarks=landmarks, person_detected=True))
    return frames


def _build_angle_series(exercise: LungeExercise, pose_frames: list[PoseFrame]) -> dict[str, np.ndarray]:
    raw: dict[str, list[float | None]] = {}
    for frame in pose_frames:
        computed = exercise.compute_angles(frame, min_visibility=0.5)
        for key, value in computed.items():
            raw.setdefault(key, []).append(value)
    return {k: np.array([np.nan if v is None else v for v in vals]) for k, vals in raw.items()}


def test_front_leg_identified_correctly_throughout_rep() -> None:
    """Regression guard for the front/rear identification heuristic: with
    this stance geometry, the (left) leg with the larger thigh lean should
    be identified as "front" on every single frame, at every depth."""
    front_angles, timestamps = generate_squat_angle_sequence(num_reps=1, standing_angle=165, bottom_angle=90, noise=0.0)
    rear_angles, _ = generate_squat_angle_sequence(num_reps=1, standing_angle=165, bottom_angle=95, noise=0.0)
    pose_frames = _synthetic_pose_frames(front_angles, rear_angles, timestamps)
    exercise = LungeExercise()

    for i, frame in enumerate(pose_frames):
        angles = exercise.compute_angles(frame, min_visibility=0.5)
        assert angles["front_knee_angle"] is not None
        assert angles["rear_knee_angle"] is not None
        assert np.isclose(angles["front_knee_angle"], front_angles[i], atol=1e-3)
        assert np.isclose(angles["rear_knee_angle"], rear_angles[i], atol=1e-3)


def test_good_lunge_scores_high_and_is_classified_good() -> None:
    front_angles, timestamps = generate_squat_angle_sequence(num_reps=3, standing_angle=165, bottom_angle=90, noise=0.0)
    rear_angles, _ = generate_squat_angle_sequence(num_reps=3, standing_angle=165, bottom_angle=92, noise=0.0)
    pose_frames = _synthetic_pose_frames(front_angles, rear_angles, timestamps)
    exercise = LungeExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    reps = detect_reps(
        angle_series["front_knee_angle"].tolist(), timestamps, exercise.build_rep_detector_config(LUNGE_CONFIG), fps=30.0
    )
    assert len(reps) == 3

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, LUNGE_CONFIG, min_visibility=0.5)
    assert len(results) == 3
    for result in results:
        assert result.form_score.classification == RepClassification.GOOD
        assert result.form_score.overall_score >= 80


def test_shallow_front_knee_flags_insufficient_depth() -> None:
    # Front bottom of 150 never reaches the 95+-10 depth target.
    front_angles, timestamps = generate_squat_angle_sequence(num_reps=2, standing_angle=165, bottom_angle=150, noise=0.0)
    rear_angles, _ = generate_squat_angle_sequence(num_reps=2, standing_angle=165, bottom_angle=95, noise=0.0)
    pose_frames = _synthetic_pose_frames(front_angles, rear_angles, timestamps)
    exercise = LungeExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    cfg = {
        **LUNGE_CONFIG,
        "rep_detection": {**LUNGE_CONFIG["rep_detection"], "bottom_angle": 155, "ascending_threshold": 158},
    }
    reps = detect_reps(
        angle_series["front_knee_angle"].tolist(), timestamps, exercise.build_rep_detector_config(cfg), fps=30.0
    )
    assert len(reps) == 2

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, LUNGE_CONFIG, min_visibility=0.5)
    for result in results:
        issue_types = {i.type for i in result.form_score.issues}
        assert "INSUFFICIENT_DEPTH" in issue_types
        assert result.form_score.classification == RepClassification.NEEDS_IMPROVEMENT


def test_shallow_rear_knee_flags_rear_insufficient_drop() -> None:
    # Front reaches good depth; rear bottom of 150 never reaches the
    # 100+-12 rear-depth target.
    front_angles, timestamps = generate_squat_angle_sequence(num_reps=1, standing_angle=165, bottom_angle=90, noise=0.0)
    rear_angles, _ = generate_squat_angle_sequence(num_reps=1, standing_angle=165, bottom_angle=150, noise=0.0)
    pose_frames = _synthetic_pose_frames(front_angles, rear_angles, timestamps)
    exercise = LungeExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    reps = detect_reps(
        angle_series["front_knee_angle"].tolist(), timestamps, exercise.build_rep_detector_config(LUNGE_CONFIG), fps=30.0
    )
    assert len(reps) == 1

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, LUNGE_CONFIG, min_visibility=0.5)
    issue_types = {i.type for i in results[0].form_score.issues}
    assert "REAR_KNEE_INSUFFICIENT_DROP" in issue_types


def test_excessive_trunk_lean_is_flagged() -> None:
    front_angles, timestamps = generate_squat_angle_sequence(num_reps=1, standing_angle=165, bottom_angle=90, noise=0.0)
    rear_angles, _ = generate_squat_angle_sequence(num_reps=1, standing_angle=165, bottom_angle=92, noise=0.0)
    pose_frames = _synthetic_pose_frames(front_angles, rear_angles, timestamps, trunk_lean_deg=55.0)
    exercise = LungeExercise()
    angle_series = _build_angle_series(exercise, pose_frames)

    reps = detect_reps(
        angle_series["front_knee_angle"].tolist(), timestamps, exercise.build_rep_detector_config(LUNGE_CONFIG), fps=30.0
    )
    assert len(reps) == 1

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, LUNGE_CONFIG, min_visibility=0.5)
    issue_types = {i.type for i in results[0].form_score.issues}
    assert "EXCESSIVE_TRUNK_LEAN" in issue_types
