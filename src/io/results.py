"""Serialize pipeline results to machine-readable JSON/CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.analysis.form_analyzer import RepAnalysisResult


def build_results_dict(
    video_path: str,
    exercise_name: str,
    fps: float,
    frame_count: int,
    results: list[RepAnalysisResult],
    summary: dict,
    warnings: list[str],
) -> dict:
    return {
        "video": video_path,
        "exercise": exercise_name,
        "fps": fps,
        "frame_count": frame_count,
        "summary": summary,
        "warnings": warnings,
        "reps": [
            {
                **r.rep.to_dict(),
                "metrics": r.metrics.to_dict(),
                "form": r.form_score.to_dict(),
            }
            for r in results
        ],
        "disclaimer": (
            "This output reflects rule-based form-analysis heuristics derived from 2D "
            "pose estimation. It is not medical or professional coaching advice."
        ),
    }


def save_json(results_dict: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results_dict, f, indent=2)


def save_csv(results: list[RepAnalysisResult], output_path: Path) -> None:
    """One row per rep, with a stable, flat set of columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rep_number",
        "start_frame",
        "bottom_frame",
        "end_frame",
        "duration_seconds",
        "descent_duration_seconds",
        "ascent_duration_seconds",
        "tempo_ratio",
        "classification",
        "overall_score",
        "issue_types",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            metrics_dict = r.metrics.to_dict()
            writer.writerow(
                {
                    "rep_number": r.rep.rep_number,
                    "start_frame": r.rep.start_frame,
                    "bottom_frame": r.rep.bottom_frame,
                    "end_frame": r.rep.end_frame,
                    "duration_seconds": metrics_dict["duration_seconds"],
                    "descent_duration_seconds": metrics_dict["descent_duration_seconds"],
                    "ascent_duration_seconds": metrics_dict["ascent_duration_seconds"],
                    "tempo_ratio": metrics_dict["tempo_ratio"],
                    "classification": r.form_score.classification.value,
                    "overall_score": round(r.form_score.overall_score, 1),
                    "issue_types": ";".join(i.type for i in r.form_score.issues),
                }
            )
