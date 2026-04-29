"""Catalog concentration analysis.

For a small marketplace seller, *catalog concentration* is the
single-point-of-failure version of supply-chain risk: if 90% of a
category's volume sits in one or two SKUs, any disruption to those
SKUs takes the whole category offline. The :func:`hhi_per_category`
function measures that concentration with the Herfindahl-Hirschman
Index — the same statistic the U.S. Department of Justice uses to
evaluate market concentration in merger review — and the
:func:`concentration_audit` helper rolls the result into a short
narrative suitable for a state commerce program report.

The HHI thresholds used for the ``concentration_level`` label
(``<1500`` low, ``1500–2500`` moderate, ``>2500`` high) are the DOJ
Horizontal Merger Guidelines thresholds. They are public and are not
proprietary to any platform.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _hhi_for_shares(shares: np.ndarray) -> float:
    """Compute HHI from an array of fractional shares.

    Each share is squared after being expressed as a percent (0–100),
    so the resulting HHI lies in ``[0, 10000]``: a single 100% share
    yields ``100^2 = 10000``.
    """
    if shares.size == 0:
        return 0.0
    pct = shares * 100.0
    return float(np.sum(pct * pct))


def _level_for(hhi: float) -> str:
    """Map an HHI value to ``low``/``moderate``/``high`` per DOJ thresholds."""
    if hhi < 1500.0:
        return "low"
    if hhi <= 2500.0:
        return "moderate"
    return "high"


def hhi_per_category(seller_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Herfindahl-Hirschman Index per category for a seller's catalog.

    Treats each SKU within a category as a "competitor" and measures
    how concentrated the category's volume is across its SKUs. A
    seller whose category-level volume rests on a single SKU will
    have an HHI near 10,000; one whose volume is spread evenly across
    many SKUs will have a low HHI.

    Parameters
    ----------
    seller_df : pandas.DataFrame
        Must include ``category``, ``sku_id``, and ``units_sold``
        columns. The function aggregates ``units_sold`` across
        whatever date grain the input has, so passing in either daily
        rows or already-aggregated SKU totals works the same way.

    Returns
    -------
    pandas.DataFrame
        One row per category, with columns ``category``, ``hhi``
        (float in ``[0, 10000]``), ``concentration_level``
        (``low``/``moderate``/``high`` per the DOJ thresholds),
        ``top_seller_share`` (fraction of category volume held by the
        single largest SKU), and ``seller_count`` (number of distinct
        SKUs in the category).
    """
    required = {"category", "sku_id", "units_sold"}
    missing = required - set(seller_df.columns)
    if missing:
        raise ValueError(
            f"seller_df missing required columns: {sorted(missing)}"
        )

    sku_totals = (
        seller_df.groupby(["category", "sku_id"])["units_sold"].sum().reset_index()
    )

    rows: List[Dict[str, Any]] = []
    for category, sub in sku_totals.groupby("category"):
        total = float(sub["units_sold"].sum())
        if total <= 0:
            shares = np.zeros(len(sub), dtype=float)
            top_share = 0.0
        else:
            shares = (sub["units_sold"].astype(float) / total).to_numpy()
            top_share = float(shares.max())
        hhi = _hhi_for_shares(shares)
        rows.append(
            {
                "category": category,
                "hhi": hhi,
                "concentration_level": _level_for(hhi),
                "top_seller_share": top_share,
                "seller_count": int(len(sub)),
            }
        )
    return pd.DataFrame(rows).sort_values("hhi", ascending=False).reset_index(drop=True)


def concentration_audit(seller_df: pd.DataFrame) -> Dict[str, Any]:
    """Produce a concentration-risk audit suitable for a counselor report.

    Parameters
    ----------
    seller_df : pandas.DataFrame
        Same input as :func:`hhi_per_category`.

    Returns
    -------
    dict
        Keys:

        * ``summary_df`` — the full per-category HHI table.
        * ``high_concentration_categories`` — list of category names
          with HHI above 2,500.
        * ``avg_hhi`` — mean HHI across categories.
        * ``audit_narrative`` — 2–3 plain-English sentences a state
          commerce program staffer can drop into a report.
    """
    summary = hhi_per_category(seller_df)
    high = summary[summary["concentration_level"] == "high"]["category"].tolist()
    avg_hhi = float(summary["hhi"].mean()) if len(summary) else 0.0
    n_total = int(len(summary))
    n_high = len(high)

    if n_total == 0:
        narrative = (
            "No category-level volume was found in the input, so "
            "concentration analysis was skipped."
        )
    elif n_high == 0:
        narrative = (
            f"Catalog concentration is healthy across all "
            f"{n_total} categories — average HHI of "
            f"{avg_hhi:,.0f} sits below the DOJ-defined moderate-"
            f"concentration threshold (1,500). No single SKU dominates "
            "any category; an outage in one SKU would not take a "
            "category offline."
        )
    else:
        narrative = (
            f"Of {n_total} categories, {n_high} show high "
            f"concentration (HHI > 2,500 — the DOJ threshold for "
            f"\"highly concentrated\" markets in merger review): "
            f"{', '.join(high)}. The average HHI across the catalog "
            f"is {avg_hhi:,.0f}. The seller is exposed to single-SKU "
            "outage risk in those categories and should consider "
            "diversifying volume across additional SKUs before scaling."
        )

    return {
        "summary_df": summary,
        "high_concentration_categories": high,
        "avg_hhi": avg_hhi,
        "audit_narrative": narrative,
    }
