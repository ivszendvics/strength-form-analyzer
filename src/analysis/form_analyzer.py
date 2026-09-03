"""Top-level orchestration: turn detected reps into scored, explainable results.

For each detected rep this ties together: generic metric extraction
(:mod:`src.biomechanics.metrics`), exercise-specific extra metrics and form
rules (:mod:`src.exercises`), and the weighted-scoring/classification engine
(:mod:`src.analysis.scoring`). The result is a small, traceable breakdown
per rep: which components contributed what score, and which specific
issues were flagged and why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analysis.rules import RuleContext
from src.analysis.scoring import FormScore, score_rep
from src.biomechanics.metrics import RepMetrics, compute_rep_metrics
from src.exercises.base import Exercise
from src.pose.landmarks import Landmark, PoseFrame
from src.reps.detector import Rep

DEFAULT_MIN_VISIBLE_FRACTION = 0.7


@dataclass
class RepAnalysisResult:
    rep: Rep
    metrics: RepMetrics
    form_score: FormScore


def rep_visibility_ok(
    pose_frames: list[PoseFrame],
    rep: Rep,
    landmarks: list[Landmark],
    min_visibility: float,
    min_visible_fraction: float = DEFAULT_MIN_VISIBLE_FRACTION,
) -> bool:
    """Whether pose confidence during this rep was high enough to trust.

    Requires at least ``min_visible_fraction`` of the rep's frames to have
    a person detected with all required landmarks above ``min_visibility``.
    This is what allows the pipeline to mark a rep UNCERTAIN rather than
    confidently misclassify it when tracking was poor (occlusion, motion
    blur, the person leaving frame, etc.) -- see README "Camera Assumptions".
    """
    frames = pose_frames[rep.start_frame : rep.end_frame + 1]
    if not frames:
        return False
    good_frames = sum(1 for f in frames if f.person_detected and f.min_visibility(landmarks) >= min_visibility)
    return (good_frames / len(frames)) >= min_visible_fraction


def analyze_reps(
    exercise: Exercise,
    reps: list[Rep],
    timestamps: list[float],
    angle_series: dict[str, np.ndarray],
    pose_frames: list[PoseFrame],
    raw_config: dict,
    min_visibility: float,
) -> list[RepAnalysisResult]:
    """Compute metrics, form rules, and a final score for every detected rep, in order.

    Reps are processed in order specifically so each rep's consistency/tempo
    rules can compare against the *previous* reps in the set (see
    ``src/analysis/rules.py``'s tempo/consistency rules).
    """
    scoring_cfg = raw_config.get("scoring", {})
    good_threshold = float(scoring_cfg.get("good_threshold", 80.0))
    weights = {k: float(v) for k, v in scoring_cfg.items() if k != "good_threshold"}

    required_landmarks = exercise.required_landmarks()
    results: list[RepAnalysisResult] = []
    previous_metrics: list[RepMetrics] = []

    for rep in reps:
        metrics = compute_rep_metrics(rep, timestamps, angle_series)
        metrics.extra.update(exercise.compute_extra_metrics(metrics, angle_series, pose_frames, rep))

        visibility_ok = rep_visibility_ok(pose_frames, rep, required_landmarks, min_visibility)

        context = RuleContext(
            rep_number=rep.rep_number,
            metrics=metrics,
            previous_metrics=previous_metrics,
            visibility_ok=visibility_ok,
        )
        rule_results = exercise.evaluate_form(context, raw_config)
        form_score = score_rep(rep.rep_number, rule_results, weights, visibility_ok, good_threshold)

        results.append(RepAnalysisResult(rep=rep, metrics=metrics, form_score=form_score))
        previous_metrics.append(metrics)

    return results


def summarize_workout(results: list[RepAnalysisResult]) -> dict:
    """A final workout-level summary: counts, average score, common issues."""
    if not results:
        return {
            "total_reps": 0,
            "good_reps": 0,
            "needs_improvement_reps": 0,
            "uncertain_reps": 0,
            "average_form_score": None,
            "most_common_issues": [],
        }

    from src.analysis.scoring import RepClassification

    good = sum(1 for r in results if r.form_score.classification == RepClassification.GOOD)
    needs_improvement = sum(1 for r in results if r.form_score.classification == RepClassification.NEEDS_IMPROVEMENT)
    uncertain = sum(1 for r in results if r.form_score.classification == RepClassification.UNCERTAIN)

    scored = [r.form_score.overall_score for r in results if r.form_score.classification != RepClassification.UNCERTAIN]
    avg_score = float(np.mean(scored)) if scored else None

    issue_counts: dict[str, int] = {}
    for r in results:
        for issue in r.form_score.issues:
            issue_counts[issue.type] = issue_counts.get(issue.type, 0) + 1
    most_common = sorted(issue_counts.items(), key=lambda kv: -kv[1])[:5]

    return {
        "total_reps": len(results),
        "good_reps": good,
        "needs_improvement_reps": needs_improvement,
        "uncertain_reps": uncertain,
        "average_form_score": round(avg_score, 1) if avg_score is not None else None,
        "most_common_issues": [{"type": t, "count": c} for t, c in most_common],
    }
