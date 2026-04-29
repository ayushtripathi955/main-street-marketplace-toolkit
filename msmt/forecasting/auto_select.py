"""Automatic forecasting-method selection and end-to-end forecast pipeline.

The toolkit doesn't ask the user "which method do you want?" — it
asks "what kind of SKU is this?" and lets the answer drive the method
choice. :func:`auto_select_method` encodes that mapping; the rules
match Article 3 of the practitioner series exactly.

:func:`run_forecast` and :func:`batch_forecast` then chain the
classifier, the method selector, and the chosen forecaster into one
call so a counselor doesn't have to know the internals.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from msmt.forecasting.baselines import (
    holt_winters_forecast,
    holts_forecast,
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    ses_forecast,
)
from msmt.forecasting.croston import croston_forecast
from msmt.forecasting.prophet_wrapper import is_prophet_available, prophet_forecast
from msmt.resilience.classifier import classify_pattern


def auto_select_method(pattern: str, series_length: int) -> str:
    """Recommend a forecasting method given a demand pattern.

    Selection rules (matching Article 3 of the practitioner series):

    * ``new_sku`` *or* fewer than 56 days of history → ``"moving_average"``
    * ``intermittent`` → ``"croston"``
    * ``holiday_spike`` with ``series_length >= 365`` → ``"prophet"``
    * ``holiday_spike`` with shorter history → ``"holt_winters"``
    * ``weekly_seasonal`` → ``"holt_winters"``
    * ``smooth`` → ``"ses"``
    * any other input → ``"moving_average"`` (safe fallback)

    Parameters
    ----------
    pattern : str
        One of the canonical demand-pattern names. Unknown values fall
        back to the safe default rather than raising.
    series_length : int
        Number of days of history available for the SKU.

    Returns
    -------
    str
        Method name suitable as a key into the dispatch table inside
        :func:`run_forecast`.
    """
    if pattern == "new_sku" or series_length < 56:
        return "moving_average"
    if pattern == "intermittent":
        return "croston"
    if pattern == "holiday_spike":
        return "prophet" if series_length >= 365 else "holt_winters"
    if pattern == "weekly_seasonal":
        return "holt_winters"
    if pattern == "smooth":
        return "ses"
    return "moving_average"


def _dispatch_forecast(
    method: str,
    series: np.ndarray,
    dates: pd.DatetimeIndex,
    horizon: int,
):
    """Run ``method`` on ``series``; always returns ``(f, lo, hi)``."""
    if method == "naive":
        return naive_forecast(series, horizon, return_pi=True)
    if method == "seasonal_naive":
        return seasonal_naive_forecast(series, horizon, return_pi=True)
    if method == "moving_average":
        return moving_average_forecast(series, horizon, return_pi=True)
    if method == "ses":
        return ses_forecast(series, horizon, return_pi=True)
    if method == "holts":
        return holts_forecast(series, horizon, return_pi=True)
    if method == "holt_winters":
        return holt_winters_forecast(series, horizon, return_pi=True)
    if method == "croston":
        return croston_forecast(series, horizon, return_pi=True)
    if method == "prophet":
        return prophet_forecast(series, horizon, dates=dates)
    raise ValueError(f"Unknown forecasting method: {method}")


def run_forecast(
    sku_df: pd.DataFrame,
    horizon: int = 28,
    return_pi: bool = True,
) -> Dict[str, Any]:
    """Classify a SKU and run the recommended forecasting method.

    Parameters
    ----------
    sku_df : pandas.DataFrame
        Single-SKU history. Must include ``date``, ``sku_id``, and
        ``units_sold`` columns.
    horizon : int, default 28
        Number of future days to forecast.
    return_pi : bool, default True
        Kept for API symmetry with the underlying baselines. Currently
        the function always computes the prediction interval — the
        flag is reserved for future versions that may want to skip
        the residual calculation.

    Returns
    -------
    dict
        Keys: ``sku_id``, ``pattern``, ``method_used``, ``forecast``
        (numpy array of length ``horizon``), ``lower_95``, ``upper_95``,
        ``horizon_dates`` (pandas DatetimeIndex of length ``horizon``).
    """
    required = {"date", "sku_id", "units_sold"}
    missing = required - set(sku_df.columns)
    if missing:
        raise ValueError(f"sku_df missing required columns: {sorted(missing)}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    df = sku_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    series = df["units_sold"].astype(float).to_numpy()
    dates = pd.DatetimeIndex(df["date"])
    sku_id = str(df["sku_id"].iloc[0])

    pattern, _confidence = classify_pattern(df)
    method = auto_select_method(pattern, series_length=series.size)

    forecast, lower, upper = _dispatch_forecast(method, series, dates, horizon)
    horizon_dates = pd.date_range(
        dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    return {
        "sku_id": sku_id,
        "pattern": pattern,
        "method_used": method,
        "forecast": np.asarray(forecast, dtype=float),
        "lower_95": np.asarray(lower, dtype=float),
        "upper_95": np.asarray(upper, dtype=float),
        "horizon_dates": horizon_dates,
    }


def batch_forecast(
    seller_df: pd.DataFrame,
    horizon: int = 28,
) -> List[Dict[str, Any]]:
    """Run :func:`run_forecast` for every SKU in a multi-SKU catalog.

    Parameters
    ----------
    seller_df : pandas.DataFrame
        Multi-SKU history. Must include ``sku_id`` plus the columns
        :func:`run_forecast` requires.
    horizon : int, default 28
        Number of future days to forecast for each SKU.

    Returns
    -------
    list of dict
        One forecast dict per SKU, in the order SKUs first appear in
        ``seller_df``.
    """
    if "sku_id" not in seller_df.columns:
        raise ValueError("seller_df must include a 'sku_id' column")

    out: List[Dict[str, Any]] = []
    for _sku_id, sku_df in seller_df.groupby("sku_id", sort=False):
        out.append(run_forecast(sku_df, horizon=horizon))
    return out
