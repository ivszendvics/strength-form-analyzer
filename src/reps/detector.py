"""Automatic repetition detection from a smoothed primary-angle time series.

Rep detection is a state machine driven by a single "primary" joint angle
(e.g. knee angle for a squat) crossing configurable thresholds, rather than
naive local-minimum/maximum detection. Naive extrema detection produces
false reps from noise (every tiny wiggle becomes a "rep"); the state
machine instead requires the signal to move meaningfully away from a
resting/standing position, reach a genuine bottom/turnaround, and return
close to the resting position again before a rep counts as complete.

State machine::

    IDLE --(angle drops below descending_threshold)--> DESCENDING
    DESCENDING --(angle rises past ascending_threshold, having reached
                   bottom_angle)--> ASCENDING
    ASCENDING --(angle returns near standing_angle)--> COMPLETED --(next
                  frame)--> IDLE

"Bottom" is not a separate dwell state: which frame was the true bottom can
only be known in hindsight (once the signal turns around and is confirmed
to be rising again), so it is tracked throughout DESCENDING/ASCENDING as a
running minimum (``_bottom_value``/``_bottom_frame``) rather than as a
state the machine is "in". COMPLETED is held for exactly one frame (the
frame the rep closes on) so callers can observe it, then the next
``update()`` call advances to IDLE.

Hysteresis is built in: the thresholds for entering a descent and for
confirming a return to standing are deliberately not the same value, and a
minimum bottom depth / minimum rep duration are enforced, so small noisy
fluctuations around the standing position never trigger a spurious rep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class RepState(str, Enum):
    IDLE = "IDLE"
    DESCENDING = "DESCENDING"
    ASCENDING = "ASCENDING"
    COMPLETED = "COMPLETED"


@dataclass
class Rep:
    """A single detected repetition, in frame indices and seconds."""

    rep_number: int
    start_frame: int
    bottom_frame: int
    end_frame: int
    start_time_seconds: float
    bottom_time_seconds: float
    end_time_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_time_seconds - self.start_time_seconds

    def to_dict(self) -> dict:
        return {
            "rep_number": self.rep_number,
            "start_frame": self.start_frame,
            "bottom_frame": self.bottom_frame,
            "end_frame": self.end_frame,
            "duration_seconds": round(self.duration_seconds, 3),
        }


@dataclass
class RepDetectorConfig:
    """Thresholds controlling rep detection. Exercise-specific.

    Angles are expressed in degrees on the *primary* joint angle for the
    exercise (e.g. knee angle for squat/lunge, hip angle for deadlift),
    where larger = more extended/standing and smaller = more flexed/bottom.

    Attributes:
        standing_angle: Angle considered "fully standing" (top position).
        descending_threshold: Angle the signal must drop below to be
            considered starting a descent (hysteresis gap below
            ``standing_angle`` so noise near standing doesn't start a rep).
        bottom_angle: Angle the signal must drop at or below to confirm a
            genuine "bottom" was reached (minimum depth requirement).
        ascending_threshold: Angle the signal must rise back above,
            after the bottom, before ascent is considered progressing back
            toward standing.
        standing_return_threshold: Angle the signal must rise back to
            (close to ``standing_angle``) to confirm the rep completed.
        min_rep_duration_seconds: Reps shorter than this are discarded as
            noise, not genuine lift tempo.
        min_frames_in_state: Minimum consecutive frames a threshold
            condition must hold before a state transition is accepted,
            providing additional debounce against single-frame jitter.
    """

    standing_angle: float = 165.0
    descending_threshold: float = 155.0
    bottom_angle: float = 110.0
    ascending_threshold: float = 120.0
    standing_return_threshold: float = 155.0
    min_rep_duration_seconds: float = 0.5
    min_frames_in_state: int = 2


@dataclass
class _PendingState:
    frame_index: int
    value: float
    consecutive_frames: int = 0


class RepDetector:
    """Stateful rep detector; feed it one (frame_index, timestamp, angle) at a time."""

    def __init__(self, config: RepDetectorConfig, fps: float) -> None:
        self.config = config
        self.fps = fps
        self.state = RepState.IDLE
        self.reps: list[Rep] = []
        self._rep_counter = 0

        self._descend_pending: _PendingState | None = None
        self._ascend_pending: _PendingState | None = None
        self._standing_pending: _PendingState | None = None

        self._start_frame: int | None = None
        self._start_time: float | None = None
        self._bottom_frame: int | None = None
        self._bottom_time: float | None = None
        self._bottom_value: float = math.inf

    def update(self, frame_index: int, timestamp_seconds: float, angle: float | None) -> Rep | None:
        """Feed one frame's (already smoothed) primary angle. Returns a
        completed :class:`Rep` on the frame that completes it, else None."""
        if self.state == RepState.COMPLETED:
            self._reset_to_idle()

        if angle is None:
            return None

        cfg = self.config

        if self.state == RepState.IDLE:
            if self._debounce("_descend_pending", frame_index, angle < cfg.descending_threshold):
                self.state = RepState.DESCENDING
                self._start_frame = frame_index
                self._start_time = timestamp_seconds
                self._bottom_value = angle
                self._bottom_frame = frame_index
                self._bottom_time = timestamp_seconds
            return None

        if self.state == RepState.DESCENDING:
            if angle < self._bottom_value:
                self._bottom_value = angle
                self._bottom_frame = frame_index
                self._bottom_time = timestamp_seconds
            if self._debounce("_ascend_pending", frame_index, angle > cfg.ascending_threshold):
                if self._bottom_value <= cfg.bottom_angle:
                    self.state = RepState.ASCENDING
                else:
                    # Rose back up without ever reaching sufficient depth:
                    # not a genuine rep attempt (e.g. a partial dip).
                    self._reset_to_idle()
            return None

        if self.state == RepState.ASCENDING:
            if angle < self._bottom_value:
                # Signal dipped back down during "ascent" -- still tracking
                # the same bottom, stay in ASCENDING but keep bottom updated
                # only if it's a deeper true minimum than previously seen.
                self._bottom_value = angle
                self._bottom_frame = frame_index
                self._bottom_time = timestamp_seconds
            if self._debounce("_standing_pending", frame_index, angle >= cfg.standing_return_threshold):
                return self._complete_rep(frame_index, timestamp_seconds)
            return None

        return None

    def _clear_trackers(self) -> None:
        self._descend_pending = None
        self._ascend_pending = None
        self._standing_pending = None
        self._start_frame = None
        self._start_time = None
        self._bottom_frame = None
        self._bottom_time = None
        self._bottom_value = math.inf

    def _reset_to_idle(self) -> None:
        self.state = RepState.IDLE
        self._clear_trackers()

    def _complete_rep(self, frame_index: int, timestamp_seconds: float) -> Rep | None:
        assert self._start_frame is not None
        assert self._start_time is not None
        assert self._bottom_frame is not None
        assert self._bottom_time is not None

        duration = timestamp_seconds - self._start_time
        rep: Rep | None = None
        if duration >= self.config.min_rep_duration_seconds:
            self._rep_counter += 1
            rep = Rep(
                rep_number=self._rep_counter,
                start_frame=self._start_frame,
                bottom_frame=self._bottom_frame,
                end_frame=frame_index,
                start_time_seconds=self._start_time,
                bottom_time_seconds=self._bottom_time,
                end_time_seconds=timestamp_seconds,
            )
            self.reps.append(rep)
        self._clear_trackers()
        # Held for exactly this one frame; the next update() call advances
        # to IDLE (see the top of update()).
        self.state = RepState.COMPLETED
        return rep

    # -- debounce helper ---------------------------------------------------
    # Each tracked condition must hold for `min_frames_in_state` consecutive
    # updates before it is accepted, so a single noisy frame can't flip state.

    def _debounce(self, attr: str, frame_index: int, condition_met: bool) -> bool:
        pending: _PendingState | None = getattr(self, attr)
        if not condition_met:
            setattr(self, attr, None)
            return False

        if pending is None:
            setattr(self, attr, _PendingState(frame_index=frame_index, value=0.0, consecutive_frames=1))
            pending = getattr(self, attr)
        else:
            pending.consecutive_frames += 1

        if pending.consecutive_frames >= self.config.min_frames_in_state:
            setattr(self, attr, None)
            return True
        return False


def detect_reps(
    angles: list[float | None],
    timestamps: list[float],
    config: RepDetectorConfig,
    fps: float,
) -> list[Rep]:
    """Convenience wrapper: run the full state machine over a full sequence."""
    detector = RepDetector(config, fps)
    for frame_index, (angle, ts) in enumerate(zip(angles, timestamps)):
        detector.update(frame_index, ts, angle)
    return detector.reps
