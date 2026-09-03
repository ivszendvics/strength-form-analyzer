"""Per-repetition metric extraction from smoothed angle time series.

Generic metrics (duration, ROM, tempo split, min/max angles) are computed
here from plain angle arrays and a detected :class:`~src.reps.detector.Rep`,
so any exercise can reuse them. Exercise-specific metrics (e.g. an
exercise's asymmetry measure) are layered on top by each exercise class in
:mod:`src.exercises`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.reps.detector import Rep


@dataclass
class RepMetrics:
    """Generic per-rep metrics, applicable to any tracked angle."""

    rep_number: int
    start_frame: int
    bottom_frame: int
    end_frame: int
    duration_seconds: float
    descent_duration_seconds: float
    ascent_duration_seconds: float
    tempo_ratio: float | None  # descent_duration / ascent_duration
    range_of_motion_degrees: dict[str, float]
    min_angle_degrees: dict[str, float]
    max_angle_degrees: dict[str, float]
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rep_number": self.rep_number,
            "start_frame": self.start_frame,
            "bottom_frame": self.bottom_frame,
            "end_frame": self.end_frame,
            "duration_seconds": round(self.duration_seconds, 3),
            "descent_duration_seconds": round(self.descent_duration_seconds, 3),
            "ascent_duration_seconds": round(self.ascent_duration_seconds, 3),
            "tempo_ratio": round(self.tempo_ratio, 3) if self.tempo_ratio is not None else None,
            "range_of_motion_degrees": {k: round(v, 2) for k, v in self.range_of_motion_degrees.items()},
            "min_angle_degrees": {k: round(v, 2) for k, v in self.min_angle_degrees.items()},
            "max_angle_degrees": {k: round(v, 2) for k, v in self.max_angle_degrees.items()},
            **{k: round(v, 3) for k, v in self.extra.items()},
        }


def _safe_slice_stat(values: np.ndarray, start: int, end: int, stat: str) -> float | None:
    segment = values[start : end + 1]
    finite = segment[~np.isnan(segment)]
    if finite.size == 0:
        return None
    return float(np.min(finite)) if stat == "min" else float(np.max(finite))


def compute_rep_metrics(
    rep: Rep,
    timestamps: list[float],
    angle_series: dict[str, np.ndarray],
) -> RepMetrics:
    """Compute generic metrics for one rep from named smoothed angle arrays.

    Args:
        rep: The detected rep (frame boundaries).
        timestamps: Per-frame timestamps in seconds, same length as the
            angle series.
        angle_series: Mapping of angle name (e.g. "knee_angle") to a
            smoothed per-frame array (NaN where unavailable).
    """
    descent_duration = rep.bottom_time_seconds - rep.start_time_seconds
    ascent_duration = rep.end_time_seconds - rep.bottom_time_seconds
    tempo_ratio = (descent_duration / ascent_duration) if ascent_duration > 0 else None

    min_angles: dict[str, float] = {}
    max_angles: dict[str, float] = {}
    rom: dict[str, float] = {}
    for name, series in angle_series.items():
        min_val = _safe_slice_stat(series, rep.start_frame, rep.end_frame, "min")
        max_val = _safe_slice_stat(series, rep.start_frame, rep.end_frame, "max")
        if min_val is not None:
            min_angles[name] = min_val
        if max_val is not None:
            max_angles[name] = max_val
        if min_val is not None and max_val is not None:
            rom[name] = max_val - min_val

    return RepMetrics(
        rep_number=rep.rep_number,
        start_frame=rep.start_frame,
        bottom_frame=rep.bottom_frame,
        end_frame=rep.end_frame,
        duration_seconds=rep.duration_seconds,
        descent_duration_seconds=descent_duration,
        ascent_duration_seconds=ascent_duration,
        tempo_ratio=tempo_ratio,
        range_of_motion_degrees=rom,
        min_angle_degrees=min_angles,
        max_angle_degrees=max_angles,
    )


def asymmetry_percent(left_series: np.ndarray, right_series: np.ndarray, start: int, end: int) -> float | None:
    """Left/right asymmetry as a percentage difference in range of motion.

    ``100 * |left_ROM - right_ROM| / max(left_ROM, right_ROM)``. Returns
    ``None`` if either side has no usable data in the slice.
    """
    left_min = _safe_slice_stat(left_series, start, end, "min")
    left_max = _safe_slice_stat(left_series, start, end, "max")
    right_min = _safe_slice_stat(right_series, start, end, "min")
    right_max = _safe_slice_stat(right_series, start, end, "max")
    if None in (left_min, left_max, right_min, right_max):
        return None

    left_rom = left_max - left_min
    right_rom = right_max - right_min
    denom = max(left_rom, right_rom)
    if denom == 0:
        return 0.0
    return 100.0 * abs(left_rom - right_rom) / denom
