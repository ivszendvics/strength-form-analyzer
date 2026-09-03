"""Rule-based form-analysis heuristics.

Every rule here produces a 0-100 component score plus zero or more
:class:`FormIssue` objects. These are heuristics derived from geometric
projections of a single 2D camera view -- **not** a medical or professional
coaching assessment. Issue messages are phrased as "potential issue"
observations with an associated confidence, never as a definitive diagnosis
of incorrect technique (see README "Limitations").

Each rule is a small, independently testable function operating on a
:class:`RuleContext`, so exercises can mix and match which rules apply and
with what thresholds (loaded from ``configs/*.yaml``) without duplicating
the underlying scoring math.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.biomechanics.metrics import RepMetrics


@dataclass
class FormIssue:
    type: str
    severity: str  # "low" | "medium" | "high"
    confidence: float  # 0.0-1.0
    message: str

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
            "message": self.message,
        }


@dataclass
class RuleResult:
    component: str
    score: float  # 0-100, higher = better
    issues: list[FormIssue] = field(default_factory=list)


@dataclass
class RuleContext:
    """Everything a rule needs to evaluate one rep."""

    rep_number: int
    metrics: RepMetrics
    previous_metrics: list[RepMetrics]
    visibility_ok: bool


def _clamp_score(score: float) -> float:
    return max(0.0, min(100.0, score))


def evaluate_depth(
    context: RuleContext,
    angle_key: str,
    target_angle: float,
    tolerance: float,
    points_per_degree_short: float = 3.0,
) -> RuleResult:
    """Score how close the rep's minimum angle got to a configured target depth.

    ``target_angle`` is a configurable heuristic, not a universal "correct"
    depth -- appropriate depth varies by anatomy, mobility, and exercise
    intent (see ``configs/*.yaml`` and the README for this assumption).
    Full score if the minimum angle reached ``target_angle - tolerance`` or
    below (i.e. at or past the target); score decreases proportionally to
    how many degrees short of the (tolerance-adjusted) target the rep fell.
    """
    min_angle = context.metrics.min_angle_degrees.get(angle_key)
    if min_angle is None:
        return RuleResult(component="depth", score=50.0, issues=[])  # insufficient data, neutral

    threshold = target_angle + tolerance
    if min_angle <= threshold:
        return RuleResult(component="depth", score=100.0, issues=[])

    degrees_short = min_angle - threshold
    score = _clamp_score(100.0 - degrees_short * points_per_degree_short)
    severity = "high" if degrees_short > 15 else "medium" if degrees_short > 7 else "low"
    confidence = min(0.95, 0.5 + degrees_short / 40.0)
    issue = FormIssue(
        type="INSUFFICIENT_DEPTH",
        severity=severity,
        confidence=confidence,
        message=(
            f"Potential issue: minimum {angle_key.replace('_', ' ')} of {min_angle:.0f} degrees "
            f"is {degrees_short:.0f} degrees short of the configured depth target."
        ),
    )
    return RuleResult(component="depth", score=score, issues=[issue])


def evaluate_trunk_lean(
    context: RuleContext,
    max_trunk_angle_threshold: float,
    points_per_degree_over: float = 2.5,
) -> RuleResult:
    """Score forward trunk inclination against a configured maximum.

    Excessive or rapidly changing trunk lean is flagged as a *potential*
    issue, not a proven fault -- filming angle and individual anthropometry
    both affect the measured trunk angle.
    """
    max_trunk = context.metrics.max_angle_degrees.get("trunk_angle")
    if max_trunk is None:
        return RuleResult(component="trunk", score=50.0, issues=[])

    if max_trunk <= max_trunk_angle_threshold:
        return RuleResult(component="trunk", score=100.0, issues=[])

    degrees_over = max_trunk - max_trunk_angle_threshold
    score = _clamp_score(100.0 - degrees_over * points_per_degree_over)
    severity = "high" if degrees_over > 20 else "medium" if degrees_over > 10 else "low"
    confidence = min(0.9, 0.5 + degrees_over / 50.0)
    issue = FormIssue(
        type="EXCESSIVE_TRUNK_LEAN",
        severity=severity,
        confidence=confidence,
        message=(
            f"Potential issue: peak trunk inclination of {max_trunk:.0f} degrees from vertical "
            f"exceeds the configured comfort threshold of {max_trunk_angle_threshold:.0f} degrees."
        ),
    )
    return RuleResult(component="trunk", score=score, issues=[issue])


def evaluate_tempo(
    context: RuleContext,
    max_deviation_ratio: float = 0.4,
) -> RuleResult:
    """Score rep-duration consistency against this set's prior reps.

    The first rep in a set has nothing to compare against and scores
    neutrally. Later reps are compared to the running average duration of
    all previous reps; a duration that differs by more than
    ``max_deviation_ratio`` (as a fraction of the average) is flagged.
    """
    if not context.previous_metrics:
        return RuleResult(component="tempo", score=100.0, issues=[])

    avg_duration = float(np.mean([m.duration_seconds for m in context.previous_metrics]))
    if avg_duration <= 0:
        return RuleResult(component="tempo", score=100.0, issues=[])

    deviation_ratio = abs(context.metrics.duration_seconds - avg_duration) / avg_duration
    if deviation_ratio <= max_deviation_ratio:
        return RuleResult(component="tempo", score=100.0, issues=[])

    excess = deviation_ratio - max_deviation_ratio
    score = _clamp_score(100.0 - excess * 200.0)
    severity = "medium" if excess > 0.3 else "low"
    issue = FormIssue(
        type="TEMPO_INCONSISTENT",
        severity=severity,
        confidence=0.6,
        message=(
            f"Potential issue: rep duration ({context.metrics.duration_seconds:.2f}s) deviates "
            f"from this set's average ({avg_duration:.2f}s) by {deviation_ratio * 100:.0f}%."
        ),
    )
    return RuleResult(component="tempo", score=score, issues=[issue])


def evaluate_rep_consistency(
    context: RuleContext,
    rom_key: str,
    min_rom_ratio_vs_earlier: float = 0.8,
) -> RuleResult:
    """Score whether range of motion has dropped relative to earlier reps.

    Flags fatigue-pattern degradation: depth/ROM shrinking significantly
    across a set. Neutral (no penalty) for the first rep.
    """
    if not context.previous_metrics:
        return RuleResult(component="stability", score=100.0, issues=[])

    current_rom = context.metrics.range_of_motion_degrees.get(rom_key)
    earlier_roms = [m.range_of_motion_degrees.get(rom_key) for m in context.previous_metrics]
    earlier_roms = [r for r in earlier_roms if r is not None]
    if current_rom is None or not earlier_roms:
        return RuleResult(component="stability", score=100.0, issues=[])

    earlier_avg = float(np.mean(earlier_roms))
    if earlier_avg <= 0:
        return RuleResult(component="stability", score=100.0, issues=[])

    ratio = current_rom / earlier_avg
    if ratio >= min_rom_ratio_vs_earlier:
        return RuleResult(component="stability", score=100.0, issues=[])

    shortfall = min_rom_ratio_vs_earlier - ratio
    score = _clamp_score(100.0 - shortfall * 250.0)
    issue = FormIssue(
        type="REDUCED_RANGE_OF_MOTION",
        severity="medium" if shortfall > 0.15 else "low",
        confidence=0.65,
        message=(
            f"Potential issue: range of motion for this rep ({current_rom:.0f} degrees) is "
            f"{(1 - ratio) * 100:.0f}% smaller than the average of earlier reps in this set "
            f"({earlier_avg:.0f} degrees), which may indicate fatigue or inconsistent depth."
        ),
    )
    return RuleResult(component="stability", score=score, issues=[issue])


def evaluate_symmetry(
    context: RuleContext,
    max_asymmetry_percent: float = 15.0,
) -> RuleResult:
    """Score left/right asymmetry, when both sides were visible enough to measure."""
    asymmetry = context.metrics.extra.get("asymmetry_percent")
    if asymmetry is None:
        return RuleResult(component="symmetry", score=100.0, issues=[])

    if asymmetry <= max_asymmetry_percent:
        return RuleResult(component="symmetry", score=100.0, issues=[])

    excess = asymmetry - max_asymmetry_percent
    score = _clamp_score(100.0 - excess * 2.0)
    issue = FormIssue(
        type="LEFT_RIGHT_ASYMMETRY",
        severity="high" if excess > 20 else "medium" if excess > 10 else "low",
        confidence=0.55,  # asymmetry from a single 2D camera view is a rough proxy
        message=(
            f"Potential issue: left/right range-of-motion asymmetry of {asymmetry:.0f}% "
            f"exceeds the configured threshold of {max_asymmetry_percent:.0f}%."
        ),
    )
    return RuleResult(component="symmetry", score=score, issues=[issue])


def evaluate_extension(
    context: RuleContext,
    angle_key: str,
    target_angle: float,
    tolerance: float,
    points_per_degree_short: float = 3.0,
) -> RuleResult:
    """Score whether the rep's maximum angle reached a configured lockout/extension target.

    The mirror image of :func:`evaluate_depth`: instead of checking how
    deep the minimum angle went, this checks how close the *maximum* angle
    got to full extension (e.g. hip/knee lockout at the top of a deadlift).
    """
    max_angle = context.metrics.max_angle_degrees.get(angle_key)
    if max_angle is None:
        return RuleResult(component="lockout", score=50.0, issues=[])

    threshold = target_angle - tolerance
    if max_angle >= threshold:
        return RuleResult(component="lockout", score=100.0, issues=[])

    degrees_short = threshold - max_angle
    score = _clamp_score(100.0 - degrees_short * points_per_degree_short)
    severity = "high" if degrees_short > 15 else "medium" if degrees_short > 7 else "low"
    confidence = min(0.9, 0.5 + degrees_short / 40.0)
    issue = FormIssue(
        type="INCOMPLETE_LOCKOUT",
        severity=severity,
        confidence=confidence,
        message=(
            f"Potential issue: maximum {angle_key.replace('_', ' ')} of {max_angle:.0f} degrees "
            f"is {degrees_short:.0f} degrees short of the configured full-extension target."
        ),
    )
    return RuleResult(component="lockout", score=score, issues=[issue])


def evaluate_value_consistency(
    context: RuleContext,
    extra_key: str,
    max_deviation_degrees: float,
    component_name: str,
    issue_type: str,
    description: str,
    points_per_degree_over: float = 3.0,
) -> RuleResult:
    """Score how much a scalar extra metric (e.g. starting hip angle) drifts
    from the average of earlier reps in the set. Generic building block used
    for e.g. "inconsistent starting position" / "inconsistent lockout"
    checks. Neutral for the first rep (nothing to compare against).
    """
    if not context.previous_metrics:
        return RuleResult(component=component_name, score=100.0, issues=[])

    current = context.metrics.extra.get(extra_key)
    earlier = [m.extra.get(extra_key) for m in context.previous_metrics]
    earlier = [v for v in earlier if v is not None]
    if current is None or not earlier:
        return RuleResult(component=component_name, score=100.0, issues=[])

    avg = float(np.mean(earlier))
    deviation = abs(current - avg)
    if deviation <= max_deviation_degrees:
        return RuleResult(component=component_name, score=100.0, issues=[])

    excess = deviation - max_deviation_degrees
    score = _clamp_score(100.0 - excess * points_per_degree_over)
    issue = FormIssue(
        type=issue_type,
        severity="medium" if excess > 10 else "low",
        confidence=0.55,
        message=(
            f"Potential issue: {description} ({current:.0f} degrees) differs from this set's "
            f"earlier average ({avg:.0f} degrees) by {deviation:.0f} degrees."
        ),
    )
    return RuleResult(component=component_name, score=score, issues=[issue])


def evaluate_knee_tracking(
    context: RuleContext,
    knee_drift_ratio: float | None,
    max_drift_ratio: float = 0.25,
    camera_view: str = "side",
) -> RuleResult:
    """Score inward/outward knee travel relative to the foot, if measurable.

    This is only meaningful from a front-facing camera; from a side view
    the horizontal knee-vs-ankle drift used here isn't a reliable valgus
    indicator, so the rule is skipped (neutral score, no issue) unless
    ``camera_view == "front"``. This assumption is configurable per
    exercise config.
    """
    if camera_view != "front" or knee_drift_ratio is None:
        return RuleResult(component="knee_tracking", score=100.0, issues=[])

    if knee_drift_ratio <= max_drift_ratio:
        return RuleResult(component="knee_tracking", score=100.0, issues=[])

    excess = knee_drift_ratio - max_drift_ratio
    score = _clamp_score(100.0 - excess * 200.0)
    issue = FormIssue(
        type="POTENTIAL_KNEE_TRACKING_ISSUE",
        severity="medium" if excess > 0.15 else "low",
        confidence=0.45,  # low: 2D horizontal drift is a coarse proxy for knee valgus/varus
        message=(
            "Potential issue: the knee travels substantially relative to the foot during this "
            "rep, which can (but does not necessarily) indicate inward/outward knee collapse."
        ),
    )
    return RuleResult(component="knee_tracking", score=score, issues=[issue])
