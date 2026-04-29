"""Marketplace-integrity scorecard for a single seller.

Given a dict of current metric values, produce an overall 0–100 score
plus a per-signal breakdown, a suppression-risk label, the top three
issues by severity, and plain-English recommendations a counselor can
hand directly to a seller.

The scoring is a deliberately transparent linear interpolation: at
``benchmark_good`` a signal scores 100, at ``benchmark_poor`` it
scores 0, in between it scales linearly. The same single formula
handles higher-is-better and lower-is-better signals because we use
the numerical direction of the two benchmarks rather than a separate
flag.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from msmt.integrity.signals import (
    SIGNALS,
    SIGNALS_BY_NAME,
    Signal,
    recommendation_for,
)


def _score_signal(signal: Signal, value: float) -> float:
    """Map a metric ``value`` to the 0–100 scale defined by the signal."""
    span = signal.benchmark_good - signal.benchmark_poor
    if span == 0:
        return 100.0 if value == signal.benchmark_good else 0.0
    raw = (value - signal.benchmark_poor) / span * 100.0
    return float(max(0.0, min(100.0, raw)))


def _rating_for(score: float) -> str:
    """Bucket a 0–100 signal score into ``good``/``fair``/``poor``."""
    if score >= 80.0:
        return "good"
    if score >= 50.0:
        return "fair"
    return "poor"


def _suppression_risk(overall_score: float) -> str:
    """Bucket the overall score into a suppression-risk label."""
    if overall_score >= 75.0:
        return "low"
    if overall_score >= 50.0:
        return "medium"
    return "high"


def compute_scorecard(seller_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Compute the integrity scorecard for one seller.

    Parameters
    ----------
    seller_metrics : dict
        Mapping ``signal_name -> value``. Any signal listed in
        :data:`msmt.integrity.signals.SIGNALS` not present in the
        input is treated as scoring 0 (worst case) for that signal,
        which is the safe default — a missing metric usually means
        the seller doesn't measure it at all, which is itself an
        integrity gap. Unknown keys are ignored.

    Returns
    -------
    dict
        Keys:

        * ``overall_score`` — float in ``[0, 100]``.
        * ``signal_scores`` — dict of ``signal_name -> {value,
          score_0_to_100, rating, weight}``.
        * ``suppression_risk`` — ``"low" | "medium" | "high"`` based
          on overall score (>=75 / 50–75 / <50).
        * ``top_issues`` — list of up to three signals with the
          lowest scores, each ``{name, score, rating,
          plain_english}``.
        * ``recommendations`` — list of plain-English action strings,
          one per top issue, in the same order.
    """
    signal_scores: Dict[str, Dict[str, Any]] = {}
    weighted_sum = 0.0

    for sig in SIGNALS:
        if sig.name in seller_metrics:
            value = float(seller_metrics[sig.name])
        else:
            value = sig.benchmark_poor  # missing metric scores 0
        score = _score_signal(sig, value)
        weighted_sum += score * sig.weight
        signal_scores[sig.name] = {
            "value": float(value),
            "score_0_to_100": float(score),
            "rating": _rating_for(score),
            "weight": float(sig.weight),
            "description": sig.description,
        }

    overall_score = float(weighted_sum)
    risk = _suppression_risk(overall_score)

    ranked = sorted(
        signal_scores.items(), key=lambda kv: kv[1]["score_0_to_100"]
    )
    top_issues: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    for name, info in ranked[:3]:
        if info["rating"] == "good":
            break  # don't flag well-performing signals as issues
        top_issues.append(
            {
                "name": name,
                "score": info["score_0_to_100"],
                "rating": info["rating"],
                "value": info["value"],
                "plain_english": (
                    f"{name.replace('_', ' ').title()} is "
                    f"{info['rating']} (score {info['score_0_to_100']:.0f}/100)."
                ),
            }
        )
        recommendations.append(recommendation_for(name))

    return {
        "overall_score": overall_score,
        "signal_scores": signal_scores,
        "suppression_risk": risk,
        "top_issues": top_issues,
        "recommendations": recommendations,
    }


def scorecard_for_synthetic_seller(seed: Optional[int] = 42) -> Dict[str, Any]:
    """Generate a plausible synthetic seller and run the scorecard on it.

    Each metric is drawn from a realistic range (a uniform between the
    poor benchmark and a value modestly past the good benchmark), with
    a small fraction of metrics deliberately pushed below the poor
    benchmark to give the scorecard something to flag. Useful for
    demos and tests.

    Parameters
    ----------
    seed : int, optional
        Seed for the random draw. Defaults to ``42`` so demos are
        reproducible.

    Returns
    -------
    dict
        The scorecard dict, exactly as :func:`compute_scorecard`
        returns. The ``signal_scores`` entries also reflect the
        synthetic input metric.
    """
    rng = np.random.default_rng(seed)
    metrics: Dict[str, float] = {}
    for sig in SIGNALS:
        # Most signals land in a believable mid-good range; ~25% of the
        # time we draw a deliberately weak value so the scorecard has
        # things to recommend.
        weak = rng.random() < 0.25
        if sig.lower_is_better:
            if weak:
                value = float(rng.uniform(sig.benchmark_poor, sig.benchmark_poor * 1.4))
            else:
                value = float(rng.uniform(sig.benchmark_good * 0.7, sig.benchmark_good))
        else:
            if weak:
                value = float(rng.uniform(sig.benchmark_poor * 0.85, sig.benchmark_poor))
            else:
                low = sig.benchmark_good
                high = sig.benchmark_good + 0.5 * (sig.benchmark_good - sig.benchmark_poor)
                value = float(rng.uniform(low, high))
        # Clip integer-shaped signals.
        if sig.name == "image_count":
            value = float(int(round(value)))
        metrics[sig.name] = value

    return compute_scorecard(metrics)
