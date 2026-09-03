"""Tests for signal smoothing: verify noisy synthetic data becomes smoother."""

import numpy as np

from src.biomechanics.smoothing import (
    exponential_smoothing,
    interpolate_missing,
    moving_average,
    savitzky_golay,
    smooth_series,
)


def _noisy_sine(n: int = 200, noise_std: float = 5.0, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    clean = 90 + 40 * np.sin(t)
    noisy = clean + rng.normal(0, noise_std, size=n)
    return noisy.tolist()


def _roughness(values: np.ndarray) -> float:
    """Mean absolute frame-to-frame difference; lower = smoother."""
    return float(np.mean(np.abs(np.diff(values))))


def test_moving_average_reduces_roughness() -> None:
    noisy = _noisy_sine()
    smoothed = moving_average(noisy, window_size=7)
    assert _roughness(smoothed) < _roughness(np.array(noisy))


def test_exponential_smoothing_reduces_roughness() -> None:
    noisy = _noisy_sine()
    smoothed = exponential_smoothing(noisy, alpha=0.3)
    assert _roughness(smoothed) < _roughness(np.array(noisy))


def test_savitzky_golay_reduces_roughness() -> None:
    noisy = _noisy_sine()
    smoothed = savitzky_golay(noisy, window_length=11, polyorder=2)
    assert _roughness(smoothed) < _roughness(np.array(noisy))


def test_savitzky_golay_preserves_peak_better_than_moving_average() -> None:
    """Savitzky-Golay should track a true peak's value more closely than a
    moving average, which flattens extrema -- the reason it's the default."""
    n = 101
    clean = np.concatenate([np.linspace(90, 30, 50), np.linspace(30, 90, 51)])
    rng = np.random.default_rng(1)
    noisy = (clean + rng.normal(0, 2.0, size=n)).tolist()

    true_min_idx = int(np.argmin(clean))
    true_min = clean[true_min_idx]

    savgol = savitzky_golay(noisy, window_length=11, polyorder=2)
    ma = moving_average(noisy, window_size=11)

    savgol_error = abs(savgol[true_min_idx] - true_min)
    ma_error = abs(ma[true_min_idx] - true_min)
    assert savgol_error <= ma_error


def test_smooth_series_dispatches_by_method() -> None:
    noisy = _noisy_sine(n=50)
    from src.biomechanics.smoothing import SmoothingMethod

    result = smooth_series(noisy, method=SmoothingMethod.MOVING_AVERAGE, window_size=5)
    assert _roughness(result) < _roughness(np.array(noisy))


def test_interpolate_missing_fills_interior_gaps() -> None:
    values = [1.0, None, None, 4.0, 5.0]
    result = interpolate_missing(values)
    assert np.allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_interpolate_missing_leaves_edges_nan() -> None:
    values = [None, 2.0, 3.0, None]
    result = interpolate_missing(values)
    assert np.isnan(result[0])
    assert np.isnan(result[-1])
    assert np.allclose(result[1:3], [2.0, 3.0])


def test_smoothing_handles_all_missing() -> None:
    values: list[float | None] = [None, None, None]
    result = savitzky_golay(values)
    assert np.all(np.isnan(result))


def test_smoothing_handles_short_series() -> None:
    values = [10.0, 20.0, 15.0]
    result = savitzky_golay(values, window_length=9, polyorder=2)
    assert len(result) == 3
    assert not np.any(np.isnan(result))


def test_savitzky_golay_handles_leading_and_trailing_nan() -> None:
    """Regression test: real footage often has a landmark unrecoverable at
    the very start/end of a clip (e.g. the person isn't fully in frame on
    frame 0 yet), which interpolate_missing leaves as edge NaN. Previously
    this crashed savgol_filter outright (it rejects any NaN, not just an
    all-NaN array) instead of just leaving those edge positions as NaN."""
    values: list[float | None] = [None, None] + [90.0 + i for i in range(20)] + [None]
    result = savitzky_golay(values, window_length=9, polyorder=2)
    assert np.isnan(result[0]) and np.isnan(result[1])
    assert np.isnan(result[-1])
    assert not np.any(np.isnan(result[2:-1]))


def test_moving_average_edge_nan_does_not_bleed_into_valid_frames() -> None:
    """Regression test: padding a NaN edge with mode='edge' before
    convolving used to propagate that NaN into every valid frame within
    window_size//2 of the edge, silently discarding far more data than
    just the truly-unrecoverable positions."""
    values: list[float | None] = [None] + [100.0] * 20
    result = moving_average(values, window_size=5)
    assert np.isnan(result[0])
    assert not np.any(np.isnan(result[1:]))
