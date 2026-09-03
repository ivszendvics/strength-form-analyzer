"""Utilities for generating synthetic joint-angle sequences for testing.

These let rep detection, smoothing, and form-analysis logic be tested
without any video or pose-estimation model, per the project's testing
strategy (tests should not require a GPU or an actual video).
"""

from __future__ import annotations

import numpy as np


def generate_squat_angle_sequence(
    num_reps: int = 5,
    fps: float = 30.0,
    standing_angle: float = 170.0,
    bottom_angle: float = 90.0,
    rep_duration_seconds: float = 2.0,
    rest_duration_seconds: float = 0.5,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Generate a synthetic knee-angle time series for ``num_reps`` squats.

    Each rep is a smooth descent-then-ascent (half a cosine cycle) between
    ``standing_angle`` and ``bottom_angle``, separated by rest periods at
    standing. Returns ``(angles, timestamps)``.
    """
    rng = np.random.default_rng(seed)
    angles: list[float] = []
    timestamps: list[float] = []
    t = 0.0
    dt = 1.0 / fps

    def hold(duration: float) -> None:
        nonlocal t
        steps = max(int(duration * fps), 1)
        for _ in range(steps):
            angles.append(standing_angle)
            timestamps.append(t)
            t += dt

    def rep() -> None:
        nonlocal t
        steps = max(int(rep_duration_seconds * fps), 2)
        for i in range(steps):
            phase = i / (steps - 1)  # 0 -> 1 across the rep
            # cosine: 1 at phase 0/1 (standing), -1 at phase 0.5 (bottom)
            cos_val = np.cos(phase * 2 * np.pi)
            frac_to_bottom = (1 - cos_val) / 2  # 0 at ends, 1 at middle
            angle = standing_angle - frac_to_bottom * (standing_angle - bottom_angle)
            angles.append(float(angle))
            timestamps.append(t)
            t += dt

    hold(rest_duration_seconds)
    for _ in range(num_reps):
        rep()
        hold(rest_duration_seconds)

    if noise > 0:
        noisy = np.array(angles) + rng.normal(0, noise, size=len(angles))
        angles = noisy.tolist()

    return angles, timestamps


def generate_flat_sequence(value: float, num_frames: int, fps: float = 30.0) -> tuple[list[float], list[float]]:
    """A constant-angle sequence (e.g. standing still) -- should yield 0 reps."""
    angles = [value] * num_frames
    timestamps = [i / fps for i in range(num_frames)]
    return angles, timestamps
