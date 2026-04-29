"""Smoke tests for the msmt.integrity module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msmt.integrity import (
    SIGNALS,
    compute_scorecard,
    concentration_audit,
    hhi_per_category,
    scorecard_for_synthetic_seller,
)


def test_signal_weights_sum_to_one() -> None:
    """Every signal weight must combine to exactly 1.0."""
    assert pytest.approx(sum(s.weight for s in SIGNALS), abs=1e-9) == 1.0


def test_overall_score_in_range() -> None:
    """compute_scorecard's overall_score must lie in [0, 100]."""
    sc = scorecard_for_synthetic_seller(seed=7)
    assert 0.0 <= sc["overall_score"] <= 100.0


def test_all_good_seller_scores_high() -> None:
    """A seller meeting every 'good' benchmark should score >= 80."""
    good = {s.name: s.benchmark_good for s in SIGNALS}
    sc = compute_scorecard(good)
    assert sc["overall_score"] >= 80.0
    assert sc["suppression_risk"] == "low"


def test_all_poor_seller_scores_low() -> None:
    """A seller hitting every 'poor' benchmark should score <= 20."""
    poor = {s.name: s.benchmark_poor for s in SIGNALS}
    sc = compute_scorecard(poor)
    assert sc["overall_score"] <= 20.0
    assert sc["suppression_risk"] == "high"


def test_top_issues_capped_at_three() -> None:
    """top_issues should never exceed three entries."""
    poor = {s.name: s.benchmark_poor for s in SIGNALS}
    sc = compute_scorecard(poor)
    assert len(sc["top_issues"]) <= 3
    assert len(sc["recommendations"]) == len(sc["top_issues"])


def test_hhi_single_sku_is_10000() -> None:
    """One SKU holding 100% share of a category produces HHI = 10000."""
    df = pd.DataFrame(
        [
            {"category": "X", "sku_id": "x1", "units_sold": 250},
        ]
    )
    out = hhi_per_category(df)
    assert len(out) == 1
    assert out.iloc[0]["hhi"] == pytest.approx(10000.0)
    assert out.iloc[0]["concentration_level"] == "high"
    assert out.iloc[0]["top_seller_share"] == pytest.approx(1.0)


def test_concentration_level_high_above_2500() -> None:
    """concentration_level must be 'high' for HHI > 2500 (DOJ threshold)."""
    # Two SKUs with 80/20 split: HHI = 6400 + 400 = 6800
    df = pd.DataFrame(
        [
            {"category": "Y", "sku_id": "y1", "units_sold": 80},
            {"category": "Y", "sku_id": "y2", "units_sold": 20},
        ]
    )
    out = hhi_per_category(df)
    assert out.iloc[0]["hhi"] == pytest.approx(6800.0)
    assert out.iloc[0]["concentration_level"] == "high"


def test_concentration_audit_narrative_present() -> None:
    """concentration_audit always returns a non-empty narrative string."""
    df = pd.DataFrame(
        [
            {"category": "Z", "sku_id": f"z{i}", "units_sold": 10}
            for i in range(5)
        ]
    )
    audit = concentration_audit(df)
    assert "summary_df" in audit and len(audit["summary_df"]) == 1
    assert isinstance(audit["audit_narrative"], str)
    assert len(audit["audit_narrative"]) > 0
