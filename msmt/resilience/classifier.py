"""Demand-pattern classifier.

Given a single SKU's daily sales history, this module decides which of
five demand archetypes the SKU best fits. The five archetypes are the
ones the rest of the toolkit is organized around: *smooth*,
*weekly_seasonal*, *holiday_spike*, *intermittent*, and *new_sku*.

The classifier is intentionally simple. It implements the same
plain-language decision tree a counselor might walk through on paper:

1. Is this listing too new to trust the history? → ``new_sku``.
2. Are there a lot of zero-sale days? → ``intermittent``.
3. Does demand swing in a weekly rhythm? → ``weekly_seasonal``.
4. Are there a few enormous spikes against an otherwise quiet
   baseline? → ``holiday_spike``.
5. Otherwise → ``smooth``.

Each rule has a single threshold so a counselor can read the code and
explain it to a seller without invoking statistics. The thresholds are
defaults; they are not derived from any proprietary dataset. A
practitioner with their own catalog can tune them.

The classifier needs only ``date`` and ``units_sold``, so it works on
real seller data exported from a marketplace portal — there is no need
for a ground-truth ``pattern`` column.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np
import pandas as pd

#: Minimum days of *active* history (first sale → last date) before the
#: classifier treats a SKU as having a real demand pattern at all.
NEW_SKU_HISTORY_DAYS: int = 56  # 8 weeks

#: Above this fraction of zero-sale days the SKU is treated as lumpy.
INTERMITTENT_ZERO_FRACTION: float = 0.40

#: Coefficient of variation across the seven day-of-week means
#: (computed on non-zero days) above which weekly seasonality is
#: considered material.
WEEKLY_DOW_CV_THRESHOLD: float = 0.15

#: Ratio of (max 14-day rolling mean) / (median 14-day rolling mean)
#: above which the demand series is treated as having holiday-style
#: spikes rather than steady demand.
HOLIDAY_SPIKE_RATIO: float = 2.5

_PATTERN_NAMES: Tuple[str, ...] = (
    "new_sku",
    "intermittent",
    "weekly_seasonal",
    "holiday_spike",
    "smooth",
)


def _active_history_days(dates: pd.Series, units: pd.Series) -> int:
    """Days from the first non-zero sale to the last date in the series.

    Used as the working definition of "history" so the classifier
    doesn't get fooled by leading zeros that just mean the listing
    didn't exist yet.
    """
    nonzero = units > 0
    if not nonzero.any():
        return 0
    first_active = dates[nonzero].min()
    last_date = dates.max()
    return int((last_date - first_active).days) + 1


def _zero_fraction(units: pd.Series) -> float:
    """Fraction of days with zero units sold over the full series."""
    if len(units) == 0:
        return 0.0
    return float((units == 0).mean())


def _dow_cv(dates: pd.Series, units: pd.Series) -> float:
    """Coefficient of variation across the seven day-of-week means.

    Zero-sale days are dropped first so the result reflects the *shape*
    of demand on selling days, not the dilution from quiet days.
    """
    nonzero = units > 0
    if nonzero.sum() < 14:
        return 0.0
    df = pd.DataFrame({"dow": dates[nonzero].dt.dayofweek, "u": units[nonzero]})
    means = df.groupby("dow")["u"].mean()
    if means.mean() == 0:
        return 0.0
    return float(means.std(ddof=0) / means.mean())


def _spike_ratio(units: pd.Series) -> float:
    """Max-over-median of the 14-day rolling mean.

    A holiday-spike SKU has a long quiet baseline and a few short
    bursts; the rolling mean separates "background" weeks from "event"
    weeks while smoothing out pure noise.
    """
    if len(units) < 28:
        return 0.0
    rolling = units.rolling(window=14, min_periods=14).mean().dropna()
    if rolling.empty:
        return 0.0
    median = float(rolling.median())
    if median <= 0:
        return float("inf") if rolling.max() > 0 else 0.0
    return float(rolling.max()) / median


def _confidence_from_threshold(value: float, threshold: float, *, above: bool) -> float:
    """Map a metric vs threshold into a 0-1 confidence.

    ``above=True`` means the rule fires when ``value > threshold`` and
    confidence grows the further above the threshold the value sits;
    ``above=False`` is the mirror image (rule fires when value falls
    below the threshold).
    """
    if threshold <= 0:
        return 0.5
    if above:
        if value <= threshold:
            return 0.0
        return float(min(1.0, (value - threshold) / threshold))
    if value >= threshold:
        return 0.0
    return float(min(1.0, (threshold - value) / threshold))


def classify_pattern(
    sku_df: pd.DataFrame,
    *,
    return_metrics: bool = False,
) -> Tuple[str, float] | Tuple[str, float, Mapping[str, float]]:
    """Classify a single SKU's demand into one of five archetypes.

    Parameters
    ----------
    sku_df : pandas.DataFrame
        Daily sales history for a *single* SKU. Must contain a ``date``
        column (datetime-like) and a ``units_sold`` column (numeric).
        Any other columns are ignored.
    return_metrics : bool, default False
        If ``True``, also return the underlying metrics that drove the
        decision (history, zero fraction, day-of-week CV, spike ratio).
        Useful for debugging or for explaining the call to a seller.

    Returns
    -------
    pattern : str
        One of ``"smooth"``, ``"weekly_seasonal"``, ``"holiday_spike"``,
        ``"intermittent"``, ``"new_sku"``.
    confidence : float
        Number in ``[0, 1]``. Higher means the chosen rule fired more
        cleanly (the metric was further from the decision threshold).
        Confidences are *not* probabilities and do not sum across
        patterns; treat them as a "how clear-cut was this call?" signal.
    metrics : dict, optional
        Only returned when ``return_metrics=True``.

    Notes
    -----
    The rules apply in priority order: *new_sku → intermittent →
    weekly_seasonal → holiday_spike → smooth*. This is the same order a
    practitioner would walk through them: dismiss the data-quality case
    first, then the cases that need specialized handling, then default
    to the easy case.
    """
    if "date" not in sku_df.columns or "units_sold" not in sku_df.columns:
        raise ValueError("sku_df must have 'date' and 'units_sold' columns")

    df = sku_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"]
    units = df["units_sold"].astype(float)

    history = _active_history_days(dates, units)
    zero_frac = _zero_fraction(units)
    dow_cv = _dow_cv(dates, units)
    spike_ratio = _spike_ratio(units)

    metrics = {
        "active_history_days": float(history),
        "zero_fraction": zero_frac,
        "dow_cv": dow_cv,
        "spike_ratio": spike_ratio,
    }

    if history < NEW_SKU_HISTORY_DAYS:
        confidence = _confidence_from_threshold(
            float(history), float(NEW_SKU_HISTORY_DAYS), above=False
        )
        pattern = "new_sku"
    elif zero_frac > INTERMITTENT_ZERO_FRACTION:
        confidence = _confidence_from_threshold(
            zero_frac, INTERMITTENT_ZERO_FRACTION, above=True
        )
        pattern = "intermittent"
    elif dow_cv > WEEKLY_DOW_CV_THRESHOLD:
        confidence = _confidence_from_threshold(
            dow_cv, WEEKLY_DOW_CV_THRESHOLD, above=True
        )
        pattern = "weekly_seasonal"
    elif spike_ratio > HOLIDAY_SPIKE_RATIO:
        confidence = _confidence_from_threshold(
            spike_ratio, HOLIDAY_SPIKE_RATIO, above=True
        )
        pattern = "holiday_spike"
    else:
        # Smooth is the residual. Confidence is high when none of the
        # other rules came close to firing; lower when one of them was
        # close to its threshold.
        residual = max(
            zero_frac / INTERMITTENT_ZERO_FRACTION,
            dow_cv / WEEKLY_DOW_CV_THRESHOLD,
            spike_ratio / HOLIDAY_SPIKE_RATIO if spike_ratio > 0 else 0.0,
        )
        confidence = float(max(0.0, 1.0 - residual))
        pattern = "smooth"

    if return_metrics:
        return pattern, confidence, metrics
    return pattern, confidence
