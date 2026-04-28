"""Reorder point (ROP) calculations.

The reorder point is the on-hand stock level at which a seller should
place a replenishment order. It is the simplest, most defensible piece
of inventory math a small seller can put in front of a counselor or a
supplier::

    ROP = average_demand_per_day * lead_time_days + safety_stock

The first term covers the units the seller expects to sell while
waiting for the next shipment to arrive; the second term is the buffer
for the variability in demand and lead time. Without safety stock, a
seller stocks out half the time on average — exactly the scenario this
module exists to prevent.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from msmt.resilience.classifier import classify_pattern
from msmt.resilience.safety_stock import (
    safety_stock_intermittent,
    safety_stock_kde,
    safety_stock_normal,
    select_safety_stock_method,
)


def reorder_point(
    demand_mean_per_day: float,
    lead_time_mean_days: float,
    safety_stock: float,
) -> float:
    """Compute the reorder-point trigger level.

    Parameters
    ----------
    demand_mean_per_day : float
        Average daily units sold over the planning horizon. Must be
        non-negative.
    lead_time_mean_days : float
        Average days from placing a reorder to receiving stock. Must be
        non-negative.
    safety_stock : float
        Buffer above expected lead-time demand, typically from one of
        the functions in :mod:`msmt.resilience.safety_stock`. Must be
        non-negative.

    Returns
    -------
    float
        Reorder point in units. Whole units are not enforced — a
        counselor will typically round up before passing the number to
        a seller.
    """
    if demand_mean_per_day < 0 or lead_time_mean_days < 0 or safety_stock < 0:
        raise ValueError("all inputs must be non-negative")
    return float(demand_mean_per_day * lead_time_mean_days + safety_stock)


def _build_lead_time_demand_samples(
    units: np.ndarray, lead_time_days: int
) -> np.ndarray:
    """Roll a window of length ``lead_time_days`` across ``units`` and sum.

    Each entry of the returned array is "total units sold over a
    contiguous lead-time window" — the input distribution the KDE
    method expects.
    """
    if len(units) < lead_time_days:
        return units.sum(keepdims=True).astype(float)
    s = pd.Series(units).rolling(window=lead_time_days, min_periods=lead_time_days).sum()
    return s.dropna().to_numpy(dtype=float)


def reorder_point_for_sku(
    sku_df: pd.DataFrame,
    service_level: float = 0.95,
    pattern: str | None = None,
) -> Dict[str, Any]:
    """Run the full reorder-point pipeline for a single SKU.

    Steps:

    1. Classify the demand pattern (or take it from the caller).
    2. Pick the recommended safety-stock method for that pattern.
    3. Compute safety stock and the reorder point.

    Parameters
    ----------
    sku_df : pandas.DataFrame
        Daily history for one SKU. Must include ``date``,
        ``units_sold``, and ``lead_time_days``. ``lead_time_days`` is
        treated as a per-SKU constant (the synthetic generator and
        most marketplace exports satisfy this).
    service_level : float, default 0.95
        Target cycle service level.
    pattern : str, optional
        Override the auto-classified pattern. Must be one of the five
        canonical pattern names. If omitted, the pattern is inferred
        via :func:`classify_pattern`.

    Returns
    -------
    dict
        Keys:
        ``rop`` (float), ``safety_stock`` (float),
        ``method_used`` (str), ``pattern`` (str),
        ``demand_mean`` (float), ``lead_time_mean`` (float).
    """
    required = {"date", "units_sold", "lead_time_days"}
    missing = required - set(sku_df.columns)
    if missing:
        raise ValueError(f"sku_df missing required columns: {sorted(missing)}")

    df = sku_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    units = df["units_sold"].astype(float).to_numpy()
    lead_time_mean = float(df["lead_time_days"].iloc[0])

    if pattern is None:
        pattern, _confidence = classify_pattern(df)

    method = select_safety_stock_method(pattern)
    demand_mean = float(units.mean()) if units.size else 0.0
    demand_std = float(units.std(ddof=0)) if units.size else 0.0

    if method == "normal":
        ss = safety_stock_normal(
            demand_mean=demand_mean,
            demand_std=demand_std,
            lead_time_mean=lead_time_mean,
            lead_time_std=0.0,
            service_level=service_level,
        )
    elif method == "kde":
        samples = _build_lead_time_demand_samples(units, int(lead_time_mean))
        ss = safety_stock_kde(samples, service_level=service_level)
    elif method == "intermittent":
        ss = safety_stock_intermittent(
            demand_series=units,
            lead_time_mean=lead_time_mean,
            service_level=service_level,
        )
    else:  # pragma: no cover - guarded upstream
        raise ValueError(f"Unknown safety-stock method: {method}")

    rop = reorder_point(demand_mean, lead_time_mean, ss)

    return {
        "rop": rop,
        "safety_stock": float(ss),
        "method_used": method,
        "pattern": pattern,
        "demand_mean": demand_mean,
        "lead_time_mean": lead_time_mean,
    }
