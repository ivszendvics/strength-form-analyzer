"""Combine per-rule component scores into an overall rep classification.

Deliberately not a single hardcoded threshold on one signal (e.g.
``if angle < X: bad_rep = True``). Instead, several independent component
scores (depth, trunk, tempo, stability, symmetry, ...) are combined as a
configurable weighted average, and *that* combined score drives the final
classification. This mirrors the "traceable" scoring breakdown described in
the README/design notes: every rep's classification can be explained as a
sum of named, weighted components down to the specific issues that lowered
each one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.analysis.rules import FormIssue, RuleResult


class RepClassification(str, Enum):
    GOOD = "GOOD"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ComponentScore:
    name: str
    score: float
    weight: float

    def to_dict(self) -> dict:
        return {"name": self.name, "score": round(self.score, 1), "weight": self.weight}


@dataclass
class FormScore:
    rep_number: int
    overall_score: float
    classification: RepClassification
    components: list[ComponentScore]
    issues: list[FormIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rep": self.rep_number,
            "classification": self.classification.value,
            "overall_score": round(self.overall_score, 1),
            "components": [c.to_dict() for c in self.components],
            "issues": [i.to_dict() for i in self.issues],
        }


DEFAULT_GOOD_THRESHOLD = 80.0


def score_rep(
    rep_number: int,
    rule_results: list[RuleResult],
    weights: dict[str, float],
    visibility_ok: bool,
    good_threshold: float = DEFAULT_GOOD_THRESHOLD,
) -> FormScore:
    """Combine weighted rule component scores into one :class:`FormScore`.

    Args:
        rule_results: One :class:`RuleResult` per evaluated rule/component.
        weights: Maps component name -> weight (need not sum to 1; they are
            normalized here). A component present in ``rule_results`` but
            missing from ``weights`` is ignored (weight 0).
        visibility_ok: If False, pose confidence was too low to trust this
            rep's measurements; the rep is classified UNCERTAIN regardless
            of the computed score, per the project's stance that the system
            should say "I don't know" rather than confidently guess.
        good_threshold: Minimum overall score (0-100) for GOOD.
    """
    components: list[ComponentScore] = []
    issues: list[FormIssue] = []
    for result in rule_results:
        weight = weights.get(result.component, 0.0)
        components.append(ComponentScore(name=result.component, score=result.score, weight=weight))
        issues.extend(result.issues)

    total_weight = sum(c.weight for c in components)
    if total_weight > 0:
        overall = sum(c.score * c.weight for c in components) / total_weight
    else:
        overall = float("nan")

    if not visibility_ok or total_weight == 0:
        classification = RepClassification.UNCERTAIN
    elif overall >= good_threshold:
        classification = RepClassification.GOOD
    else:
        classification = RepClassification.NEEDS_IMPROVEMENT

    # Sort issues most-severe first for readable output.
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: (severity_rank.get(i.severity, 3), -i.confidence))

    return FormScore(
        rep_number=rep_number,
        overall_score=overall if total_weight > 0 else 0.0,
        classification=classification,
        components=components,
        issues=issues,
    )
