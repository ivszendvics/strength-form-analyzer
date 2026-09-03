"""Static summary plots: angle-vs-time traces and per-rep metric charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display server needed to save PNGs

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.form_analyzer import RepAnalysisResult
from src.reps.detector import Rep

_REP_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def plot_angle_over_time(
    timestamps: list[float],
    angle_series: dict[str, np.ndarray],
    angle_key: str,
    reps: list[Rep],
    output_path: Path,
    title: str | None = None,
) -> Path:
    """Angle-vs-time trace with detected rep windows shaded and bottoms marked."""
    series = angle_series.get(angle_key)
    fig, ax = plt.subplots(figsize=(10, 4))
    if series is not None:
        ax.plot(timestamps, series, color="#4C72B0", linewidth=1.5)

    for i, rep in enumerate(reps):
        color = _REP_COLORS[i % len(_REP_COLORS)]
        ax.axvspan(rep.start_time_seconds, rep.end_time_seconds, color=color, alpha=0.12)
        ax.axvline(rep.bottom_time_seconds, color=color, linestyle="--", linewidth=1, alpha=0.8)
        ax.annotate(f"#{rep.rep_number}", (rep.start_time_seconds, ax.get_ylim()[1]), fontsize=8, color=color)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title(title or f"{angle_key.replace('_', ' ').title()} over time (shaded = detected rep)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_form_score_vs_rep(results: list[RepAnalysisResult], output_path: Path) -> Path:
    reps = [r.form_score.rep_number for r in results]
    scores = [r.form_score.overall_score for r in results]
    colors = [
        "#55A868" if r.form_score.classification.value == "GOOD"
        else "#C44E52" if r.form_score.classification.value == "NEEDS_IMPROVEMENT"
        else "#999999"
        for r in results
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(reps, scores, color=colors)
    ax.axhline(80, color="black", linestyle="--", linewidth=1, alpha=0.5, label="good threshold (default)")
    ax.set_xlabel("Rep number")
    ax.set_ylabel("Form score (0-100)")
    ax.set_ylim(0, 105)
    ax.set_title("Form score by rep")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_rep_duration_vs_rep(results: list[RepAnalysisResult], output_path: Path) -> Path:
    reps = [r.rep.rep_number for r in results]
    durations = [r.rep.duration_seconds for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(reps, durations, marker="o", color="#4C72B0")
    ax.set_xlabel("Rep number")
    ax.set_ylabel("Duration (s)")
    ax.set_title("Rep duration (tempo) by rep")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_rom_vs_rep(results: list[RepAnalysisResult], angle_key: str, output_path: Path) -> Path:
    reps = [r.rep.rep_number for r in results]
    roms = [r.metrics.range_of_motion_degrees.get(angle_key) for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(reps, roms, marker="o", color="#8172B2")
    ax.set_xlabel("Rep number")
    ax.set_ylabel("Range of motion (degrees)")
    ax.set_title(f"Range of motion ({angle_key.replace('_', ' ')}) by rep")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def generate_all_plots(
    timestamps: list[float],
    angle_series: dict[str, np.ndarray],
    angle_keys_to_plot: list[str],
    reps: list[Rep],
    results: list[RepAnalysisResult],
    output_dir: Path,
) -> list[Path]:
    """Generate the standard plot set and return the list of saved file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for key in angle_keys_to_plot:
        if key in angle_series:
            paths.append(plot_angle_over_time(timestamps, angle_series, key, reps, output_dir / f"{key}_over_time.png"))

    if results:
        paths.append(plot_form_score_vs_rep(results, output_dir / "form_score_by_rep.png"))
        paths.append(plot_rep_duration_vs_rep(results, output_dir / "rep_duration_by_rep.png"))
        primary_key = angle_keys_to_plot[0] if angle_keys_to_plot else None
        if primary_key:
            paths.append(plot_rom_vs_rep(results, primary_key, output_dir / "range_of_motion_by_rep.png"))

    return paths
