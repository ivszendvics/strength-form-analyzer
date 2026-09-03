"""Signal smoothing for noisy frame-by-frame joint-angle time series.

Three interchangeable smoothing strategies are implemented:

- **Moving average**: simplest, but lags behind fast movement and flattens
  the true peak/valley value, which matters a lot here since rep detection
  and per-rep metrics (e.g. minimum knee angle) depend on accurately
  locating the bottom of a squat.
- **Exponential smoothing**: reacts faster than a moving average but still
  systematically lags a moving signal and biases extrema.
- **Savitzky-Golay filtering**: fits a local polynomial in a sliding window
  instead of averaging, which smooths high-frequency jitter while
  preserving the shape (and location/value) of peaks and valleys much
  better than either alternative.

**Default: Savitzky-Golay.** Rep detection in this project depends on
locating turning points (the bottom of a squat, the lockout of a deadlift)
accurately in both time and angle value, and Savitzky-Golay is the
standard choice in signal processing when peak shape must survive
smoothing (it is likewise the standard choice for noisy sensor/kinematic
time series more generally). Moving average and exponential smoothing are
still provided for comparison/experimentation and are exercised by tests.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.signal import savgol_filter


class SmoothingMethod(str, Enum):
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    SAVGOL = "savgol"


DEFAULT_SMOOTHING_METHOD = SmoothingMethod.SAVGOL


def _to_array(values: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def interpolate_missing(values: list[float | None]) -> np.ndarray:
    """Linearly interpolate interior missing (``None``/NaN) samples.

    Leading/trailing missing samples cannot be interpolated (no data on
    one side) and are left as NaN; callers should treat those positions as
    still-unknown rather than silently trusting an extrapolated value.
    """
    arr = _to_array(values)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return arr

    result = arr.copy()
    indices = np.arange(len(arr))
    first_valid, last_valid = indices[valid][0], indices[valid][-1]
    interior = slice(first_valid, last_valid + 1)
    result[interior] = np.interp(indices[interior], indices[valid], arr[valid])
    return result


def _fill_edges_with_nearest(arr: np.ndarray) -> np.ndarray:
    """Forward/back-fill leading/trailing NaN with the nearest valid value.

    ``interpolate_missing`` deliberately leaves leading/trailing NaN when a
    landmark is unrecoverable at the very start/end of a clip (real footage
    routinely has this -- e.g. the person isn't fully in frame on the first
    processed frame). Some filters (``scipy.signal.savgol_filter``) reject
    any NaN outright, and others (a padded ``np.convolve``) would silently
    propagate a single edge NaN into every output value whose window
    touches it. This produces a fully finite array to run such a filter on;
    callers re-mask the true-NaN positions afterward from the pre-fill
    array so "unrecoverable" positions still come out as NaN in the result.
    """
    if arr.size == 0 or np.all(np.isnan(arr)):
        return arr
    filled = arr.copy()
    valid = ~np.isnan(filled)
    first = int(np.argmax(valid))
    last = len(filled) - 1 - int(np.argmax(valid[::-1]))
    filled[:first] = filled[first]
    filled[last + 1 :] = filled[last]
    return filled


def moving_average(values: list[float | None], window_size: int = 5) -> np.ndarray:
    """Simple centered moving average. Missing samples are interpolated first."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    arr = interpolate_missing(values)
    if window_size == 1 or len(arr) == 0 or np.all(np.isnan(arr)):
        return arr

    finite = _fill_edges_with_nearest(arr)
    kernel = np.ones(window_size) / window_size
    # 'same' mode keeps output length equal to input length.
    padded = np.pad(finite, (window_size // 2, window_size // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")[: len(arr)]
    # Re-apply NaN where the original value was unrecoverable (edges we
    # could not interpolate).
    smoothed[np.isnan(arr)] = np.nan
    return smoothed


def exponential_smoothing(values: list[float | None], alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average. Missing samples are interpolated first."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    arr = interpolate_missing(values)
    result = np.full_like(arr, np.nan)
    if len(arr) == 0:
        return result

    valid = ~np.isnan(arr)
    if not valid.any():
        return result

    first_idx = int(np.argmax(valid))
    result[first_idx] = arr[first_idx]
    prev = arr[first_idx]
    for i in range(first_idx + 1, len(arr)):
        if np.isnan(arr[i]):
            result[i] = np.nan
            continue
        prev = alpha * arr[i] + (1 - alpha) * prev
        result[i] = prev
    return result


def savitzky_golay(values: list[float | None], window_length: int = 9, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay filter. Missing samples are interpolated first.

    ``window_length`` must be odd and greater than ``polyorder``. If the
    series is shorter than ``window_length``, the window is shrunk to fit
    (kept odd) so short clips still produce a result.
    """
    arr = interpolate_missing(values)
    n = len(arr)
    if n == 0 or np.all(np.isnan(arr)):
        return arr

    effective_window = min(window_length, n if n % 2 == 1 else n - 1)
    if effective_window <= polyorder:
        # Too little data to fit the requested polynomial; fall back to
        # returning the interpolated series unsmoothed rather than raising.
        return arr
    if effective_window < 1:
        return arr

    finite = _fill_edges_with_nearest(arr)
    smoothed = savgol_filter(finite, window_length=effective_window, polyorder=polyorder, mode="interp")
    smoothed[np.isnan(arr)] = np.nan
    return smoothed


def smooth_series(
    values: list[float | None],
    method: SmoothingMethod = DEFAULT_SMOOTHING_METHOD,
    **kwargs: float,
) -> np.ndarray:
    """Smooth a time series using the given method (default: Savitzky-Golay)."""
    if method == SmoothingMethod.MOVING_AVERAGE:
        return moving_average(values, **kwargs)
    if method == SmoothingMethod.EXPONENTIAL:
        return exponential_smoothing(values, **kwargs)
    if method == SmoothingMethod.SAVGOL:
        return savitzky_golay(values, **kwargs)
    raise ValueError(f"Unknown smoothing method: {method}")
