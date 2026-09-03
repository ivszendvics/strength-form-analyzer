"""Pose estimation backend abstraction.

Downstream code (rep detection, form analysis, visualization) depends only
on :class:`PoseEstimator` and the :mod:`src.pose.landmarks` types, never on
a specific backend. This makes it possible to add a second backend (e.g.
OpenPose) later by implementing this interface, without changing anything
else in the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.pose.landmarks import PoseFrame


class PoseEstimator(ABC):
    """Abstract interface for a frame-by-frame body pose estimator."""

    @abstractmethod
    def estimate(self, frame: np.ndarray, frame_index: int, timestamp_seconds: float) -> PoseFrame:
        """Run pose estimation on a single BGR image frame.

        Args:
            frame: An HxWx3 BGR image (as produced by OpenCV's VideoCapture).
            frame_index: Index of this frame within the video.
            timestamp_seconds: Timestamp of this frame within the video.

        Returns:
            A :class:`PoseFrame` describing detected landmarks. If no person
            was detected, implementations should return a ``PoseFrame`` with
            ``person_detected=False`` and an empty landmarks dict rather than
            raising, so the pipeline can keep going and mark that frame
            uncertain.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any backend resources. Default is a no-op."""
        return

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
