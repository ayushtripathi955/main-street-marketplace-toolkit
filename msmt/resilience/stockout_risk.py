"""Stockout-risk scoring and catalog-level heatmap.

Once the reorder point is known, the next operational question is:
*how exposed is this SKU right now?* The functions here turn the
current on-hand position into a single risk score, a four-level risk
label, and a plain-English action recommendation a non-specialist can
act on.

The catalog-level :func:`stockout_heatmap_data` runs the entire
classify → safety-stock → ROP → risk pipeline across every SKU in a
seller's data and returns one ranked row per SKU. That ranked frame
is what the walkthrough notebook plots.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from msmt.resilience.reorder_point import reorder_point_for_sku


def _risk_level_from_score(score: float) -> str:
    """Bucket a 0-1 risk score into four named levels.

    Thresholds are conservative defaults. A counselor with a different
    risk appetite can adjust them; the rest of the pipeline only cares
    about the label, not how it was derived.
    """
    if score >= 0.80:
        return "critical"
    if score >= 0.55:
        return "high"
    if score >= 0.30:
        return "medium"
    return "low"


def _action_for(level: str, days_until_stockout: float) -> str:
    """Map a risk level + runway into a one-line recommendation."""
    if level == "critical":
        return (
            "Place an emergency reorder today and consider expediting "
            "shipping."
        )
    if level == "high":
        return (
            f"Reorder this week — at the current sell-through rate "
            f"the SKU runs out in roughly {days_until_stockout:.0f} days."
        )
    if level == "medium":
        return "Schedule a reorder in the next two weeks and recheck."
    return "No action needed; recheck on the normal cadence."


def stockout_risk_score(
    sku_df: pd.DataFrame,
    rop: float,
) -> Dict[str, Any]:
    """Score the current stockout risk for one SKU.

    Parameters
    ----------
    sku_df : pandas.DataFrame
        Daily history for one SKU. Must include ``units_sold`` and
        ``stock_on_hand``. The most recent row is treated as "today".
    rop : float
        The SKU's reorder point, as produced by
        :func:`msmt.resilience.reorder_point.reorder_point_for_sku`.

    Returns
    -------
    dict
        Keys:
        ``risk_score`` (float in ``[0, 1]``),
        ``risk_level`` (``"critical" | "high" | "medium" | "low"``),
        ``days_until_stockout`` (float, current sell-through rate),
        ``action`` (str, plain-English recommendation).

    Notes
    -----
    The score is a piecewise-linear function of the ratio
    ``current_stock / rop``:

    * ratio ≤ 0.20 → ``risk_score = 1.0`` (critical)
    * ratio ≥ 1.00 → ``risk_score = 0.0`` (no risk vs ROP)
    * in between, the score scales linearly between those endpoints.

    A SKU at exactly its ROP scores 0 because the ROP is *itself* the
    point at which a reorder should already be placed. SKUs below the
    ROP score progressively higher.
    """
    if "units_sold" not in sku_df.columns or "stock_on_hand" not in sku_df.columns:
        raise ValueError(
            "sku_df must include 'units_sold' and 'stock_on_hand' columns"
        )
    if rop < 0:
        raise ValueError("rop must be non-negative")

    df = sku_df.sort_values("date") if "date" in sku_df.columns else sku_df
    current_stock = float(df["stock_on_hand"].iloc[-1])

    # Use the trailing 28 days to estimate the current sell-through
    # rate; falls back to the full history if there's less than that.
    recent = df.tail(28)
    recent_mean = float(recent["units_sold"].astype(float).mean())
    if recent_mean <= 0:
        recent_mean = float(df["units_sold"].astype(float).mean())

    if recent_mean > 0:
        days_until_stockout = current_stock / recent_mean
    else:
        days_until_stockout = float("inf")

    if rop <= 0:
        score = 1.0 if current_stock <= 0 else 0.0
    else:
        ratio = current_stock / rop
        if ratio <= 0.20:
            score = 1.0
        elif ratio >= 1.00:
            score = 0.0
        else:
            score = float(1.0 - (ratio - 0.20) / 0.80)

    if current_stock <= 0:
        score = 1.0

    level = _risk_level_from_score(score)
    return {
        "risk_score": float(score),
        "risk_level": level,
        "days_until_stockout": float(days_until_stockout),
        "action": _action_for(level, days_until_stockout),
    }


def stockout_heatmap_data(
    seller_df: pd.DataFrame,
    service_level: float = 0.95,
) -> pd.DataFrame:
    """Run the full resilience pipeline across every SKU in a catalog.

    For each ``sku_id`` in ``seller_df``, this function:

    1. Classifies the demand pattern.
    2. Picks the appropriate safety-stock method.
    3. Computes safety stock and the reorder point.
    4. Scores current stockout risk.

    Parameters
    ----------
    seller_df : pandas.DataFrame
        Multi-SKU daily history with columns ``sku_id``, ``date``,
        ``units_sold``, ``stock_on_hand``, and ``lead_time_days``.
    service_level : float, default 0.95
        Cycle service level applied uniformly across SKUs.

    Returns
    -------
    pandas.DataFrame
        One row per SKU with columns: ``sku_id``, ``pattern``,
        ``method_used``, ``rop``, ``safety_stock``, ``current_stock``,
        ``risk_score``, ``risk_level``, ``days_until_stockout``,
        ``action``. Sorted by ``risk_score`` descending so the most
        exposed SKUs surface first.
    """
    required = {"sku_id", "date", "units_sold", "stock_on_hand", "lead_time_days"}
    missing = required - set(seller_df.columns)
    if missing:
        raise ValueError(
            f"seller_df missing required columns: {sorted(missing)}"
        )

    rows = []
    for sku_id, sku_df in seller_df.groupby("sku_id", sort=False):
        rop_info = reorder_point_for_sku(sku_df, service_level=service_level)
        risk = stockout_risk_score(sku_df, rop=rop_info["rop"])
        current_stock = float(
            sku_df.sort_values("date")["stock_on_hand"].iloc[-1]
        )
        rows.append(
            {
                "sku_id": sku_id,
                "pattern": rop_info["pattern"],
                "method_used": rop_info["method_used"],
                "rop": rop_info["rop"],
                "safety_stock": rop_info["safety_stock"],
                "current_stock": current_stock,
                "risk_score": risk["risk_score"],
                "risk_level": risk["risk_level"],
                "days_until_stockout": risk["days_until_stockout"],
                "action": risk["action"],
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)
