"""MediaPipe Pose backend implementing the :class:`PoseEstimator` interface.

MediaPipe's Python API moved from the legacy ``mp.solutions.pose`` module to
the newer Tasks API (``mediapipe.tasks.python.vision.PoseLandmarker``).
``mp.solutions`` is no longer available in current MediaPipe releases, so
this implementation uses the Tasks API directly.

The Tasks API requires a ``.task`` model bundle file rather than bundling
weights inside the pip package. This module downloads one of Google's
official pre-trained bundles on first use and caches it locally (never
committed to the repository -- see ``model_cache_dir``).
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np

from src.pose.base import PoseEstimator
from src.pose.landmarks import Landmark, LandmarkPoint, PoseFrame

logger = logging.getLogger(__name__)

# Official MediaPipe model bundle URLs (see
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).
# "lite" is the smallest/fastest model and is the default so the pipeline
# runs at reasonable speed on a laptop CPU without a GPU.
_MODEL_URLS: dict[str, str] = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

ModelComplexity = Literal["lite", "full", "heavy"]

DEFAULT_MODEL_CACHE_DIR = Path.home() / ".cache" / "strength_form_analyzer" / "models"

# Maps our backend-agnostic Landmark enum to MediaPipe's PoseLandmark names.
# Only the subset of MediaPipe's 33 landmarks this project needs is included.
_LANDMARK_TO_MEDIAPIPE_NAME: dict[Landmark, str] = {
    Landmark.NOSE: "NOSE",
    Landmark.LEFT_SHOULDER: "LEFT_SHOULDER",
    Landmark.RIGHT_SHOULDER: "RIGHT_SHOULDER",
    Landmark.LEFT_ELBOW: "LEFT_ELBOW",
    Landmark.RIGHT_ELBOW: "RIGHT_ELBOW",
    Landmark.LEFT_WRIST: "LEFT_WRIST",
    Landmark.RIGHT_WRIST: "RIGHT_WRIST",
    Landmark.LEFT_HIP: "LEFT_HIP",
    Landmark.RIGHT_HIP: "RIGHT_HIP",
    Landmark.LEFT_KNEE: "LEFT_KNEE",
    Landmark.RIGHT_KNEE: "RIGHT_KNEE",
    Landmark.LEFT_ANKLE: "LEFT_ANKLE",
    Landmark.RIGHT_ANKLE: "RIGHT_ANKLE",
    Landmark.LEFT_HEEL: "LEFT_HEEL",
    Landmark.RIGHT_HEEL: "RIGHT_HEEL",
    Landmark.LEFT_FOOT_INDEX: "LEFT_FOOT_INDEX",
    Landmark.RIGHT_FOOT_INDEX: "RIGHT_FOOT_INDEX",
}


def _ensure_model_downloaded(model_complexity: ModelComplexity, cache_dir: Path) -> Path:
    """Return a local path to the requested model bundle, downloading it if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / f"pose_landmarker_{model_complexity}.task"
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    url = _MODEL_URLS[model_complexity]
    logger.info("Downloading MediaPipe pose model (%s) from %s", model_complexity, url)
    try:
        tmp_path = model_path.with_suffix(".task.part")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.rename(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the MediaPipe pose model from {url}. "
            "Check your internet connection, or manually download the .task "
            f"file and place it at {model_path}."
        ) from exc
    return model_path


class MediaPipePoseEstimator(PoseEstimator):
    """Pose estimator backed by MediaPipe's PoseLandmarker (Tasks API)."""

    def __init__(
        self,
        model_complexity: ModelComplexity = "lite",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
        model_cache_dir: Path | None = None,
        model_path: Path | None = None,
    ) -> None:
        """Create a MediaPipe-backed pose estimator.

        Args:
            model_complexity: Which pretrained bundle to use ("lite" is
                fastest and the recommended default for CPU-only laptops).
            min_detection_confidence: Minimum confidence for the initial
                person detection. Also used downstream as the general
                landmark confidence threshold (see ``--confidence-threshold``).
            min_tracking_confidence: Minimum confidence to keep tracking a
                detected pose across frames.
            num_poses: Maximum number of people to detect. Kept at 1 by
                default: this project assumes a single lifter in frame (see
                README "Camera Assumptions"). Raising it lets the caller
                detect and warn about multiple people instead of silently
                picking one.
            model_cache_dir: Where to cache downloaded ``.task`` model
                bundles. Defaults to ``~/.cache/strength_form_analyzer/models``.
                This directory is machine-local and never part of the repo.
            model_path: Explicit path to a ``.task`` model bundle, bypassing
                automatic download entirely.
        """
        # Imported lazily so that importing this module doesn't require
        # mediapipe unless a MediaPipePoseEstimator is actually constructed.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        # The Tasks API result is a flat, index-ordered list of landmarks
        # rather than a named structure. The index order matches the legacy
        # `mp.solutions.pose.PoseLandmark` enum (BlazePose's fixed 33-point
        # topology), which is still shipped for this purpose.
        self._landmark_index: dict[str, int] = {
            name: member.value for name, member in mp.solutions.pose.PoseLandmark.__members__.items()
        }

        cache_dir = model_cache_dir or DEFAULT_MODEL_CACHE_DIR
        resolved_model_path = model_path or _ensure_model_downloaded(model_complexity, cache_dir)

        # CPU delegate is forced explicitly: the app is documented to run on a
        # normal laptop without a GPU, and MediaPipe's GPU delegate can crash
        # outright on machines without a working GPU/Metal backend available.
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(resolved_model_path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self.num_poses = num_poses
        self._last_timestamp_ms = -1

    def estimate(self, frame: np.ndarray, frame_index: int, timestamp_seconds: float) -> PoseFrame:
        import mediapipe as mp

        # MediaPipe expects RGB; OpenCV frames are BGR.
        rgb_frame = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))

        # VIDEO mode requires strictly increasing timestamps in milliseconds.
        timestamp_ms = max(int(timestamp_seconds * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return PoseFrame(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                landmarks={},
                person_detected=False,
            )

        if len(result.pose_landmarks) > 1:
            logger.warning(
                "Multiple people (%d) detected in frame %d; using the first detected pose. "
                "Results for multi-person video are not reliable -- see README limitations.",
                len(result.pose_landmarks),
                frame_index,
            )

        raw_landmarks = result.pose_landmarks[0]
        landmarks: dict[Landmark, LandmarkPoint] = {}
        for landmark, mp_name in _LANDMARK_TO_MEDIAPIPE_NAME.items():
            index = self._landmark_index[mp_name]
            raw = raw_landmarks[index]
            visibility = raw.visibility if raw.visibility is not None else 1.0
            landmarks[landmark] = LandmarkPoint(
                x=raw.x, y=raw.y, z=raw.z if raw.z is not None else 0.0, visibility=visibility
            )

        return PoseFrame(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            landmarks=landmarks,
            person_detected=True,
        )

    def close(self) -> None:
        self._landmarker.close()
