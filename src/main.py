"""Command-line entry point: video -> pose -> angles -> reps -> form analysis -> outputs.

Usage:
    python -m src.main --video data/input/squat.mp4 --exercise squat --output outputs/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.analysis.form_analyzer import analyze_reps, summarize_workout
from src.biomechanics.smoothing import SmoothingMethod, smooth_series
from src.exercises import EXERCISE_REGISTRY, get_exercise
from src.io.results import build_results_dict, save_csv, save_json
from src.pose.landmarks import PoseFrame
from src.pose.mediapipe_estimator import MediaPipePoseEstimator
from src.reps.detector import RepDetector
from src.visualization.overlay import draw_hud, draw_skeleton
from src.visualization.plots import generate_all_plots

logger = logging.getLogger("strength_form_analyzer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"

# Which computed angle keys are the "headline" ones plotted / shown on the
# overlay HUD for each exercise, in display order.
_DISPLAY_ANGLES: dict[str, list[str]] = {
    "squat": ["knee_angle", "hip_angle", "trunk_angle"],
    "deadlift": ["hip_angle", "knee_angle", "trunk_angle"],
    "lunge": ["front_knee_angle", "rear_knee_angle", "trunk_angle"],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="strength-form-analyzer",
        description="Analyze strength-training exercise form from video using pose estimation.",
    )
    parser.add_argument("--video", required=True, type=Path, help="Path to the input video file.")
    parser.add_argument("--exercise", required=True, choices=sorted(EXERCISE_REGISTRY), help="Exercise type to analyze.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write outputs into.")
    parser.add_argument("--config", type=Path, default=None, help="Override path to the exercise's YAML config (default: configs/<exercise>.yaml).")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum landmark visibility/confidence to trust a landmark (0-1). Default: 0.5.")
    parser.add_argument(
        "--smoothing",
        choices=[m.value for m in SmoothingMethod],
        default=SmoothingMethod.SAVGOL.value,
        help="Angle-smoothing method. Default: savgol (see src/biomechanics/smoothing.py for why).",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame (>=1). Larger values trade temporal resolution for speed. Default: 1.")
    parser.add_argument("--model-complexity", choices=["lite", "full", "heavy"], default="lite", help="MediaPipe pose model size. 'lite' is fastest and the default for CPU-only use.")
    parser.add_argument("--save-video", dest="save_video", action="store_true", default=True, help="Save an annotated output video (default: on).")
    parser.add_argument("--no-save-video", dest="save_video", action="store_false", help="Skip saving the annotated output video.")
    parser.add_argument("--save-plots", dest="save_plots", action="store_true", default=True, help="Save summary plots (default: on).")
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false", help="Skip saving summary plots.")
    parser.add_argument("--save-json", dest="save_json", action="store_true", default=True, help="Save results as JSON (default: on).")
    parser.add_argument("--no-save-json", dest="save_json", action="store_false", help="Skip saving JSON results.")
    parser.add_argument("--save-csv", dest="save_csv", action="store_true", default=True, help="Save per-rep results as CSV (default: on).")
    parser.add_argument("--no-save-csv", dest="save_csv", action="store_false", help="Skip saving CSV results.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def load_exercise_config(exercise_name: str, config_path: Path | None) -> dict:
    path = config_path or (DEFAULT_CONFIGS_DIR / f"{exercise_name}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Exercise config not found: {path}")
    try:
        with path.open() as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping of settings.")
    return config


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file (unsupported or corrupted?): {video_path}")

    # Phone-shot vertical video is commonly stored as landscape frames plus a
    # rotation tag (e.g. 90 degrees) rather than pre-rotated pixels. Without
    # this, frames come out sideways and pose estimation degrades badly.
    # OpenCV/FFMPEG honors the tag and reports already-rotated dimensions
    # once this is enabled; it's a no-op for video with no rotation tag.
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        logger.warning("Video reports invalid FPS; assuming 30.0.")
        fps = 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return capture, fps, frame_count, width, height


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.frame_step < 1:
        logger.error("--frame-step must be >= 1.")
        return 1

    try:
        exercise = get_exercise(args.exercise)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    try:
        config = load_exercise_config(args.exercise, args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    try:
        capture, fps, frame_count, width, height = open_video(args.video)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("Video: %s (%dx%d, %.1f fps, %d frames)", args.video, width, height, fps, frame_count)
    logger.info("Exercise: %s | frame_step=%d | smoothing=%s | model=%s", args.exercise, args.frame_step, args.smoothing, args.model_complexity)

    pose_frames: list[PoseFrame] = []
    raw_angles: dict[str, list[float | None]] = {}
    timestamps: list[float] = []
    warnings: list[str] = []

    with MediaPipePoseEstimator(model_complexity=args.model_complexity, min_detection_confidence=args.confidence_threshold, min_tracking_confidence=args.confidence_threshold) as estimator:
        frame_index = 0
        processed_index = 0
        any_person_detected = False
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.frame_step != 0:
                frame_index += 1
                continue

            timestamp = frame_index / fps
            pose_frame = estimator.estimate(frame, processed_index, timestamp)
            angles = exercise.compute_angles(pose_frame, args.confidence_threshold)

            pose_frames.append(pose_frame)
            timestamps.append(timestamp)
            for key, value in angles.items():
                raw_angles.setdefault(key, []).append(value)
            any_person_detected = any_person_detected or pose_frame.person_detected

            frame_index += 1
            processed_index += 1

    capture.release()

    if processed_index == 0:
        logger.error("No frames were read from the video; it may be empty or corrupted.")
        return 1
    if not any_person_detected:
        warnings.append("No person was detected in any processed frame. Check camera framing/lighting; results below are empty.")
        logger.warning(warnings[-1])

    logger.info("Processed %d frames.", processed_index)

    smoothing_method = SmoothingMethod(args.smoothing)
    angle_series: dict[str, np.ndarray] = {
        key: smooth_series(values, method=smoothing_method) for key, values in raw_angles.items()
    }

    rep_detector_config = exercise.build_rep_detector_config(config)
    primary_series = angle_series.get(exercise.primary_angle_name)
    detector = RepDetector(rep_detector_config, fps=fps / args.frame_step)
    if primary_series is not None:
        for i, (ts, angle) in enumerate(zip(timestamps, primary_series)):
            angle_value = None if np.isnan(angle) else float(angle)
            detector.update(i, ts, angle_value)
    reps = detector.reps
    logger.info("Detected %d rep(s).", len(reps))

    results = analyze_reps(exercise, reps, timestamps, angle_series, pose_frames, config, args.confidence_threshold)
    summary = summarize_workout(results)

    if args.save_json:
        results_dict = build_results_dict(str(args.video), args.exercise, fps, processed_index, results, summary, warnings)
        save_json(results_dict, args.output / "results.json")
        logger.info("Saved JSON results to %s", args.output / "results.json")

    if args.save_csv:
        save_csv(results, args.output / "results.csv")
        logger.info("Saved CSV results to %s", args.output / "results.csv")

    if args.save_plots:
        display_keys = _DISPLAY_ANGLES.get(args.exercise, [exercise.primary_angle_name])
        plot_paths = generate_all_plots(timestamps, angle_series, display_keys, reps, results, args.output / "plots")
        logger.info("Saved %d plot(s) to %s", len(plot_paths), args.output / "plots")

    if args.save_video:
        _write_annotated_video(args, exercise, pose_frames, timestamps, angle_series, reps, results, width, height, fps)
        logger.info("Saved annotated video to %s", args.output / "annotated.mp4")

    _print_summary(args.exercise, summary, warnings)
    return 0


def _frame_annotations(
    exercise_name: str,
    num_frames: int,
    reps,
    results,
) -> list[dict]:
    """Per-frame (rep_number, phase, classification, score, warnings) for the overlay HUD."""
    annotations = [{"rep_number": None, "phase": "IDLE", "classification": None, "score": None, "warnings": []} for _ in range(num_frames)]
    results_by_rep = {r.rep.rep_number: r for r in results}

    for rep in reps:
        result = results_by_rep.get(rep.rep_number)
        classification = result.form_score.classification.value if result else None
        score = result.form_score.overall_score if result else None
        warning_messages = [i.message.replace("Potential issue: ", "") for i in result.form_score.issues[:2]] if result else []
        for frame_idx in range(rep.start_frame, min(rep.end_frame + 1, num_frames)):
            phase = "DESCENDING" if frame_idx <= rep.bottom_frame else "ASCENDING"
            annotations[frame_idx] = {
                "rep_number": rep.rep_number,
                "phase": phase,
                "classification": classification,
                "score": score,
                "warnings": warning_messages,
            }
    return annotations


def _write_annotated_video(args, exercise, pose_frames, timestamps, angle_series, reps, results, width, height, fps) -> None:
    # Re-reads the video from disk rather than keeping every raw frame from
    # the analysis pass in memory -- decoding twice is cheap relative to
    # holding a full video's frames in RAM (see README "Performance").
    display_keys = _DISPLAY_ANGLES.get(args.exercise, [exercise.primary_angle_name])
    annotations = _frame_annotations(args.exercise, len(pose_frames), reps, results)

    capture, _, _, _, _ = open_video(args.video)
    output_path = args.output / "annotated.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.frame_step, (width, height))

    frame_index = 0
    processed_index = 0
    try:
        while processed_index < len(pose_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.frame_step != 0:
                frame_index += 1
                continue

            draw_skeleton(frame, pose_frames[processed_index], args.confidence_threshold)
            angle_values = {key: (None if np.isnan(angle_series[key][processed_index]) else float(angle_series[key][processed_index])) for key in display_keys if key in angle_series}
            ann = annotations[processed_index]
            draw_hud(frame, ann["rep_number"], ann["phase"], angle_values, ann["classification"], ann["score"], ann["warnings"])
            writer.write(frame)

            frame_index += 1
            processed_index += 1
    finally:
        capture.release()
        writer.release()


def _print_summary(exercise_name: str, summary: dict, warnings: list[str]) -> None:
    print("\n=== Strength Form Analyzer: Summary ===")
    print(f"Exercise: {exercise_name}")
    print(f"Total reps detected: {summary['total_reps']}")
    print(f"  GOOD: {summary['good_reps']}  NEEDS_IMPROVEMENT: {summary['needs_improvement_reps']}  UNCERTAIN: {summary['uncertain_reps']}")
    if summary["average_form_score"] is not None:
        print(f"Average form score: {summary['average_form_score']}/100")
    if summary["most_common_issues"]:
        print("Most common issues:")
        for issue in summary["most_common_issues"]:
            print(f"  - {issue['type']}: {issue['count']}x")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(
        "\nNote: results are rule-based form-analysis heuristics from 2D pose estimation, "
        "not medical or professional coaching advice.\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        logging.getLogger("strength_form_analyzer").error("Unexpected error: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
