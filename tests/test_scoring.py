"""Tests for the rule-based scoring/classification system."""

from src.analysis.rules import (
    FormIssue,
    RuleContext,
    RuleResult,
    evaluate_depth,
    evaluate_trunk_lean,
)
from src.analysis.scoring import RepClassification, score_rep
from src.biomechanics.metrics import RepMetrics


def _metrics(rep_number: int = 1, min_knee: float = 90.0, max_trunk: float = 20.0, duration: float = 2.0) -> RepMetrics:
    return RepMetrics(
        rep_number=rep_number,
        start_frame=0,
        bottom_frame=15,
        end_frame=30,
        duration_seconds=duration,
        descent_duration_seconds=duration / 2,
        ascent_duration_seconds=duration / 2,
        tempo_ratio=1.0,
        range_of_motion_degrees={"knee_angle": 170 - min_knee},
        min_angle_degrees={"knee_angle": min_knee, "trunk_angle": 0.0},
        max_angle_degrees={"trunk_angle": max_trunk},
    )


def test_all_good_components_yield_good_classification() -> None:
    weights = {"depth": 0.5, "trunk": 0.5}
    results = [
        RuleResult(component="depth", score=100.0),
        RuleResult(component="trunk", score=95.0),
    ]
    result = score_rep(1, results, weights, visibility_ok=True)
    assert result.classification == RepClassification.GOOD
    assert result.overall_score > 90


def test_low_scores_yield_needs_improvement() -> None:
    weights = {"depth": 0.5, "trunk": 0.5}
    results = [
        RuleResult(component="depth", score=40.0, issues=[FormIssue("INSUFFICIENT_DEPTH", "medium", 0.8, "msg")]),
        RuleResult(component="trunk", score=50.0, issues=[FormIssue("EXCESSIVE_TRUNK_LEAN", "medium", 0.7, "msg")]),
    ]
    result = score_rep(1, results, weights, visibility_ok=True)
    assert result.classification == RepClassification.NEEDS_IMPROVEMENT
    assert len(result.issues) == 2


def test_low_visibility_forces_uncertain_regardless_of_score() -> None:
    weights = {"depth": 1.0}
    results = [RuleResult(component="depth", score=100.0)]
    result = score_rep(1, results, weights, visibility_ok=False)
    assert result.classification == RepClassification.UNCERTAIN


def test_multiple_issues_lower_score_more_than_one() -> None:
    weights = {"depth": 0.34, "trunk": 0.33, "tempo": 0.33}
    one_issue = [
        RuleResult(component="depth", score=60.0, issues=[FormIssue("A", "low", 0.5, "m")]),
        RuleResult(component="trunk", score=100.0),
        RuleResult(component="tempo", score=100.0),
    ]
    two_issues = [
        RuleResult(component="depth", score=60.0, issues=[FormIssue("A", "low", 0.5, "m")]),
        RuleResult(component="trunk", score=60.0, issues=[FormIssue("B", "low", 0.5, "m")]),
        RuleResult(component="tempo", score=100.0),
    ]
    score_one = score_rep(1, one_issue, weights, visibility_ok=True)
    score_two = score_rep(1, two_issues, weights, visibility_ok=True)
    assert score_two.overall_score < score_one.overall_score
    assert len(score_two.issues) == 2


def test_weights_are_normalized() -> None:
    # Weights not summing to 1 should still combine correctly (normalized internally).
    weights = {"depth": 2.0, "trunk": 2.0}
    results = [
        RuleResult(component="depth", score=100.0),
        RuleResult(component="trunk", score=0.0),
    ]
    result = score_rep(1, results, weights, visibility_ok=True)
    assert 45.0 <= result.overall_score <= 55.0


def test_issues_sorted_most_severe_first() -> None:
    weights = {"depth": 0.5, "trunk": 0.5}
    results = [
        RuleResult(component="depth", score=50.0, issues=[FormIssue("A", "low", 0.5, "m")]),
        RuleResult(component="trunk", score=30.0, issues=[FormIssue("B", "high", 0.9, "m")]),
    ]
    result = score_rep(1, results, weights, visibility_ok=True)
    assert result.issues[0].severity == "high"


def test_evaluate_depth_full_score_when_target_reached() -> None:
    context = RuleContext(rep_number=1, metrics=_metrics(min_knee=85.0), previous_metrics=[], visibility_ok=True)
    result = evaluate_depth(context, angle_key="knee_angle", target_angle=90.0, tolerance=5.0)
    assert result.score == 100.0
    assert result.issues == []


def test_evaluate_depth_penalizes_shallow_squat() -> None:
    context = RuleContext(rep_number=1, metrics=_metrics(min_knee=140.0), previous_metrics=[], visibility_ok=True)
    result = evaluate_depth(context, angle_key="knee_angle", target_angle=90.0, tolerance=5.0)
    assert result.score < 100.0
    assert result.issues[0].type == "INSUFFICIENT_DEPTH"


def test_evaluate_trunk_lean_flags_excessive_lean() -> None:
    context = RuleContext(rep_number=1, metrics=_metrics(max_trunk=60.0), previous_metrics=[], visibility_ok=True)
    result = evaluate_trunk_lean(context, max_trunk_angle_threshold=45.0)
    assert result.score < 100.0
    assert result.issues[0].type == "EXCESSIVE_TRUNK_LEAN"


def test_evaluate_trunk_lean_no_issue_within_threshold() -> None:
    context = RuleContext(rep_number=1, metrics=_metrics(max_trunk=20.0), previous_metrics=[], visibility_ok=True)
    result = evaluate_trunk_lean(context, max_trunk_angle_threshold=45.0)
    assert result.score == 100.0
    assert result.issues == []
