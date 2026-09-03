"""Tests for the rep-detection state machine using synthetic angle data."""

from src.reps.detector import RepDetectorConfig, detect_reps
from tests.synthetic import generate_flat_sequence, generate_squat_angle_sequence


def _default_squat_config() -> RepDetectorConfig:
    return RepDetectorConfig(
        standing_angle=170.0,
        descending_threshold=160.0,
        bottom_angle=100.0,
        ascending_threshold=120.0,
        standing_return_threshold=160.0,
        min_rep_duration_seconds=0.3,
        min_frames_in_state=2,
    )


def test_detects_correct_number_of_clean_reps() -> None:
    angles, timestamps = generate_squat_angle_sequence(num_reps=5, noise=0.0)
    reps = detect_reps(angles, timestamps, _default_squat_config(), fps=30.0)
    assert len(reps) == 5


def test_rep_numbers_are_sequential() -> None:
    angles, timestamps = generate_squat_angle_sequence(num_reps=3, noise=0.0)
    reps = detect_reps(angles, timestamps, _default_squat_config(), fps=30.0)
    assert [r.rep_number for r in reps] == [1, 2, 3]


def test_reps_have_sensible_frame_ordering() -> None:
    angles, timestamps = generate_squat_angle_sequence(num_reps=3, noise=0.0)
    reps = detect_reps(angles, timestamps, _default_squat_config(), fps=30.0)
    for rep in reps:
        assert rep.start_frame <= rep.bottom_frame <= rep.end_frame
        assert rep.duration_seconds > 0


def test_detects_reps_with_moderate_noise() -> None:
    angles, timestamps = generate_squat_angle_sequence(num_reps=5, noise=2.0, seed=42)
    from src.biomechanics.smoothing import savitzky_golay

    smoothed = savitzky_golay(angles, window_length=9, polyorder=2).tolist()
    reps = detect_reps(smoothed, timestamps, _default_squat_config(), fps=30.0)
    assert len(reps) == 5


def test_flat_sequence_yields_no_reps() -> None:
    angles, timestamps = generate_flat_sequence(170.0, num_frames=100)
    reps = detect_reps(angles, timestamps, _default_squat_config(), fps=30.0)
    assert len(reps) == 0


def test_small_noise_around_standing_does_not_create_fake_reps() -> None:
    """Noise that never reaches real depth should not register as reps --
    this is the core reason rep detection uses a state machine with
    hysteresis rather than naive local-extrema detection."""
    import numpy as np

    rng = np.random.default_rng(0)
    angles = (170.0 + rng.normal(0, 3.0, size=150)).tolist()
    timestamps = [i / 30.0 for i in range(150)]
    reps = detect_reps(angles, timestamps, _default_squat_config(), fps=30.0)
    assert len(reps) == 0


def test_partial_dip_that_never_reaches_depth_is_not_a_rep() -> None:
    cfg = _default_squat_config()
    # Descend below descending_threshold but never past bottom_angle, then
    # return to standing -- should not count as a completed rep.
    angles = [170.0] * 10 + list(range(170, 130, -2)) + list(range(130, 170, 2)) + [170.0] * 10
    timestamps = [i / 30.0 for i in range(len(angles))]
    reps = detect_reps(angles, timestamps, cfg, fps=30.0)
    assert len(reps) == 0


def test_too_short_rep_duration_is_discarded() -> None:
    cfg = RepDetectorConfig(
        standing_angle=170.0,
        descending_threshold=160.0,
        bottom_angle=100.0,
        ascending_threshold=120.0,
        standing_return_threshold=160.0,
        min_rep_duration_seconds=5.0,  # unreasonably long minimum
        min_frames_in_state=2,
    )
    angles, timestamps = generate_squat_angle_sequence(num_reps=2, rep_duration_seconds=1.0, noise=0.0)
    reps = detect_reps(angles, timestamps, cfg, fps=30.0)
    assert len(reps) == 0


def test_bottom_frame_is_near_minimum_angle() -> None:
    angles, timestamps = generate_squat_angle_sequence(num_reps=1, noise=0.0, bottom_angle=90.0)
    reps = detect_reps(angles, timestamps, _default_squat_config(), fps=30.0)
    assert len(reps) == 1
    rep = reps[0]
    min_angle_in_rep = min(angles[rep.start_frame : rep.end_frame + 1])
    assert angles[rep.bottom_frame] <= min_angle_in_rep + 1e-6
