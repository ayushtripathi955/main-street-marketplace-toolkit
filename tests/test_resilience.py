"""Smoke tests for the msmt.resilience module.

These are not exhaustive; they catch the load-bearing invariants the
rest of the toolkit (and the walkthrough notebook) depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msmt.data import generate_seller_data
from msmt.resilience import (
    classify_pattern,
    reorder_point_for_sku,
    safety_stock_normal,
    stockout_heatmap_data,
    stockout_risk_score,
    suppression_adjusted_stockout_cost,
)


@pytest.fixture(scope="module")
def seller_df() -> pd.DataFrame:
    """A fixed synthetic catalog used by multiple tests."""
    return generate_seller_data(n_skus=50, n_days=365, seed=42)


def test_classifier_recovers_majority_of_synthetic_patterns(seller_df: pd.DataFrame) -> None:
    """At least 70% of synthetic SKUs should be classified correctly.

    This is a smoke check, not a benchmark: the synthetic generator
    and the classifier rules are tuned against the same archetypes,
    so the floor here is intentionally well below the achievable
    accuracy.
    """
    correct = 0
    total = 0
    for _sku_id, sub in seller_df.groupby("sku_id"):
        truth = sub["pattern"].iloc[0]
        pred, _conf = classify_pattern(sub)
        if pred == truth:
            correct += 1
        total += 1
    accuracy = correct / total
    assert accuracy >= 0.70, (
        f"classifier recovered only {accuracy:.0%} of synthetic patterns "
        f"({correct}/{total}); expected >=70%"
    )


def test_safety_stock_normal_monotonic_in_service_level() -> None:
    """Higher service level → strictly higher safety stock, all else equal."""
    args = dict(demand_mean=10.0, demand_std=3.0, lead_time_mean=14.0,
                lead_time_std=1.0)
    levels = [0.90, 0.95, 0.97, 0.98, 0.99]
    values = [safety_stock_normal(service_level=lv, **args) for lv in levels]
    for lo, hi in zip(values, values[1:]):
        assert hi > lo, (
            f"non-monotonic SS in service level: {values}"
        )


def test_reorder_point_at_least_safety_stock(seller_df: pd.DataFrame) -> None:
    """ROP = mean_demand * lead_time + SS, so ROP >= SS for any non-negative demand."""
    for _sku_id, sub in seller_df.groupby("sku_id"):
        info = reorder_point_for_sku(sub, service_level=0.95)
        assert info["rop"] >= info["safety_stock"] - 1e-9, (
            f"rop {info['rop']} < safety_stock {info['safety_stock']} "
            f"for {_sku_id}"
        )


def test_stockout_risk_critical_when_stock_zero(seller_df: pd.DataFrame) -> None:
    """A SKU with stock_on_hand == 0 today should always be 'critical'."""
    sku_id = seller_df["sku_id"].iloc[0]
    sub = seller_df[seller_df["sku_id"] == sku_id].copy()
    sub.loc[sub.index[-1], "stock_on_hand"] = 0
    info = reorder_point_for_sku(sub, service_level=0.95)
    risk = stockout_risk_score(sub, rop=info["rop"])
    assert risk["risk_level"] == "critical"
    assert risk["risk_score"] == pytest.approx(1.0)


def test_stockout_heatmap_one_row_per_sku(seller_df: pd.DataFrame) -> None:
    """Heatmap should contain exactly one row per SKU and be desc-sorted."""
    heat = stockout_heatmap_data(seller_df, service_level=0.95)
    assert len(heat) == seller_df["sku_id"].nunique()
    assert heat["sku_id"].is_unique
    diffs = heat["risk_score"].diff().dropna()
    assert (diffs <= 1e-12).all(), "heatmap is not sorted by risk_score desc"
    expected_cols = {
        "sku_id", "pattern", "rop", "safety_stock", "current_stock",
        "risk_score", "risk_level", "days_until_stockout", "action",
    }
    assert expected_cols.issubset(set(heat.columns))


def test_suppression_cost_exceeds_direct_cost() -> None:
    """With multiplier > 1, total cost includes a non-zero suppression tail."""
    s = suppression_adjusted_stockout_cost(daily_profit=100.0, stockout_days=7)
    assert s["suppression_cost"] > 0
    assert s["total_cost"] == pytest.approx(s["direct_cost"] + s["suppression_cost"])
    flat = suppression_adjusted_stockout_cost(
        daily_profit=100.0, stockout_days=7, suppression_multiplier=1.0
    )
    assert flat["suppression_cost"] == 0.0
    assert flat["total_cost"] == flat["direct_cost"]
