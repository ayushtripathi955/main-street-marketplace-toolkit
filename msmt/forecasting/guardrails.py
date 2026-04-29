"""Forecasting guardrails — five sanity checks that wrap a raw forecast.

A forecast that's correct on average is still dangerous in production
if it's *consistently* wrong, *suddenly* wrong, or being used to
trigger an order quantity that's wildly out of line with recent
history. This module bundles five plain-language checks that run
*after* a forecast is produced and tell a counselor whether to trust
it before passing it to a reorder workflow.

The five guardrails (matching Article 4 of the practitioner series):

1. **Drift detection** — has the forecast been biased the same way
   for several weeks running?
2. **Confidence floor** — is the prediction interval too wide to be
   useful for planning?
3. **Regime change** — have recent actuals fallen outside the PI band
   often enough that the world has shifted under the model?
4. **Reorder cap** — would the order quantity the forecast implies
   be far above the seller's recent run-rate?
5. **Graceful degradation** — when the primary forecast can't be
   trusted (or fails outright), what fallback should be used?
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from msmt.forecasting.baselines import _Z95, _as_array

ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]


# ---------------------------------------------------------------------------
# 1. Drift detection
# ---------------------------------------------------------------------------


def drift_detection(
    actuals: ArrayLike,
    forecasts: ArrayLike,
    threshold_weeks: int = 3,
    tolerance_pct: float = 0.20,
) -> Dict[str, Any]:
    """Detect whether forecast bias has been one-sided for several weeks.

    Sums ``forecast - actual`` into weekly buckets (7-day chunks,
    counting backward from the end). The guardrail fires when:

    * the trailing weekly biases share a sign for at least
      ``threshold_weeks`` consecutive weeks, *and*
    * the cumulative bias over those weeks exceeds ``tolerance_pct``
      of one week's expected sales (``mean(weekly forecast totals)``).

    Parameters
    ----------
    actuals : array-like
        Recent observed daily ``units_sold``.
    forecasts : array-like
        The forecast for the same days as ``actuals``.
    threshold_weeks : int, default 3
        Minimum consecutive same-sign weeks required to fire.
    tolerance_pct : float, default 0.20
        Fraction of one week's expected sales the cumulative bias must
        exceed before firing.

    Returns
    -------
    dict
        Keys ``fired`` (bool), ``direction`` (``"over"``, ``"under"``,
        or ``None``), ``cumulative_gap_pct`` (float),
        ``weeks_trending`` (int), ``recommendation`` (str).
    """
    a = _as_array(actuals)
    f = _as_array(forecasts)
    if a.size != f.size:
        raise ValueError("actuals and forecasts must have equal length")
    if a.size < 7:
        return {
            "fired": False,
            "direction": None,
            "cumulative_gap_pct": 0.0,
            "weeks_trending": 0,
            "recommendation": (
                "Not enough recent data to evaluate drift; recheck after "
                "another week."
            ),
        }

    n_weeks = a.size // 7
    a_tail = a[-n_weeks * 7:]
    f_tail = f[-n_weeks * 7:]
    weekly_actual = a_tail.reshape(n_weeks, 7).sum(axis=1)
    weekly_forecast = f_tail.reshape(n_weeks, 7).sum(axis=1)
    weekly_bias = weekly_forecast - weekly_actual

    weeks_trending = 0
    direction_sign = 0
    for bias in weekly_bias[::-1]:
        if bias == 0:
            break
        sign = 1 if bias > 0 else -1
        if direction_sign == 0:
            direction_sign = sign
            weeks_trending = 1
        elif sign == direction_sign:
            weeks_trending += 1
        else:
            break

    direction = None
    if direction_sign > 0:
        direction = "over"
    elif direction_sign < 0:
        direction = "under"

    cumulative = float(weekly_bias[-weeks_trending:].sum()) if weeks_trending > 0 else 0.0
    expected_per_week = float(weekly_forecast.mean()) if weekly_forecast.size else 0.0
    gap_pct = (
        abs(cumulative) / expected_per_week if expected_per_week > 0 else 0.0
    )

    fired = (
        weeks_trending >= threshold_weeks
        and gap_pct > tolerance_pct
    )
    if fired and direction == "over":
        rec = (
            f"Forecast has run high for {weeks_trending} weeks; reduce "
            "planned orders by the cumulative gap before reordering."
        )
    elif fired and direction == "under":
        rec = (
            f"Forecast has run low for {weeks_trending} weeks; the SKU "
            "may be selling faster than the model expects — review "
            "before the next reorder."
        )
    else:
        rec = "Forecast bias is within tolerance."

    return {
        "fired": bool(fired),
        "direction": direction,
        "cumulative_gap_pct": float(gap_pct),
        "weeks_trending": int(weeks_trending),
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# 2. Confidence floor
# ---------------------------------------------------------------------------


def confidence_floor(
    forecast: ArrayLike,
    lower_95: ArrayLike,
    upper_95: ArrayLike,
    threshold_ratio: float = 0.50,
) -> Dict[str, Any]:
    """Detect when the prediction interval is too wide to plan against.

    Computes the average PI width as a fraction of the average
    forecast level. Fires when that ratio exceeds ``threshold_ratio``
    (default 50% — the band is at least half the forecast level
    itself, in which case the forecast is rarely actionable).

    Parameters
    ----------
    forecast : array-like
        Point forecast.
    lower_95, upper_95 : array-like
        Lower and upper 95% prediction-interval bounds for the same
        horizon.
    threshold_ratio : float, default 0.50
        PI-width-to-forecast-level ratio above which the guardrail
        fires.

    Returns
    -------
    dict
        Keys ``fired`` (bool), ``pi_ratio`` (float),
        ``recommendation`` (str).
    """
    f = _as_array(forecast)
    lo = _as_array(lower_95)
    hi = _as_array(upper_95)
    if not (f.size == lo.size == hi.size):
        raise ValueError(
            "forecast, lower_95, and upper_95 must have equal length"
        )

    width = float(np.mean(hi - lo))
    level = float(np.mean(f))
    pi_ratio = width / level if level > 0 else float("inf")

    fired = pi_ratio > threshold_ratio
    if fired:
        rec = (
            "Prediction interval is wider than the forecast itself; treat "
            "the point forecast as a rough estimate and pad safety stock "
            "rather than ordering to the forecast."
        )
    else:
        rec = "Prediction interval is tight enough to plan against."

    return {
        "fired": bool(fired),
        "pi_ratio": float(pi_ratio),
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# 3. Regime change
# ---------------------------------------------------------------------------


def regime_change_detection(
    actuals: ArrayLike,
    lower_95: ArrayLike,
    upper_95: ArrayLike,
    window_days: int = 7,
    threshold_days: int = 5,
) -> Dict[str, Any]:
    """Detect whether recent actuals have left the PI band repeatedly.

    Counts how many of the last ``window_days`` actuals fall outside
    their corresponding ``[lower_95, upper_95]`` band. Fires when that
    count is at least ``threshold_days`` — i.e. a majority of recent
    days are out-of-band, suggesting the underlying demand process
    has shifted.

    Parameters
    ----------
    actuals : array-like
        Daily actuals aligned with ``lower_95`` and ``upper_95``.
    lower_95, upper_95 : array-like
        PI bounds for the same days as ``actuals``.
    window_days : int, default 7
        Length of the recent window to evaluate.
    threshold_days : int, default 5
        Number of out-of-band days that triggers the guardrail.

    Returns
    -------
    dict
        Keys ``fired`` (bool), ``outside_count`` (int),
        ``window_days`` (int), ``recommendation`` (str).
    """
    a = _as_array(actuals)
    lo = _as_array(lower_95)
    hi = _as_array(upper_95)
    if not (a.size == lo.size == hi.size):
        raise ValueError("actuals, lower_95, upper_95 must have equal length")

    eff_window = min(window_days, a.size)
    if eff_window < 1:
        return {
            "fired": False,
            "outside_count": 0,
            "window_days": 0,
            "recommendation": "Not enough data to evaluate regime change.",
        }

    a_tail = a[-eff_window:]
    lo_tail = lo[-eff_window:]
    hi_tail = hi[-eff_window:]
    outside = int(np.sum((a_tail < lo_tail) | (a_tail > hi_tail)))
    fired = outside >= threshold_days

    if fired:
        rec = (
            f"{outside} of the last {eff_window} days fell outside the "
            "forecast band — refit the model on recent data before the "
            "next reorder."
        )
    else:
        rec = "Recent actuals are tracking the forecast band."

    return {
        "fired": bool(fired),
        "outside_count": outside,
        "window_days": int(eff_window),
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# 4. Reorder cap
# ---------------------------------------------------------------------------


def reorder_cap(
    proposed_order_qty: float,
    trailing_30d_avg: float,
    cap_multiplier: float = 3.0,
) -> Dict[str, Any]:
    """Block a proposed order that's far above recent run-rate.

    Defends against the common failure mode where a noisy forecast or
    an off-by-one in the reorder logic produces an order quantity that
    is many times what the SKU has been doing. Fires when
    ``proposed_order_qty > trailing_30d_avg * cap_multiplier``.

    Parameters
    ----------
    proposed_order_qty : float
        The reorder quantity the upstream pipeline is proposing.
    trailing_30d_avg : float
        Average daily ``units_sold`` over the past 30 days (or the
        full available history if less).
    cap_multiplier : float, default 3.0
        Multiplier on the trailing average above which the cap fires.

    Returns
    -------
    dict
        Keys ``fired`` (bool), ``proposed_qty`` (float),
        ``cap_qty`` (float), ``recommendation`` (str).
    """
    if cap_multiplier < 1.0:
        raise ValueError("cap_multiplier must be >= 1.0")

    cap_qty = float(trailing_30d_avg) * float(cap_multiplier)
    fired = proposed_order_qty > cap_qty
    if fired:
        rec = (
            f"Proposed order ({proposed_order_qty:.0f}) is more than "
            f"{cap_multiplier:.1f}x the recent daily run-rate "
            f"({trailing_30d_avg:.1f}). Review manually before placing."
        )
    else:
        rec = "Proposed order is in line with recent run-rate."

    return {
        "fired": bool(fired),
        "proposed_qty": float(proposed_order_qty),
        "cap_qty": float(cap_qty),
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# 5. Graceful degradation
# ---------------------------------------------------------------------------


_DEGRADATION_LEVELS = {
    1: "primary",
    2: "trailing_30d_mean",
    3: "trailing_7d_mean",
    4: "yesterday",
}


def graceful_degradation(
    pattern: str,
    series: ArrayLike,
    horizon: int,
    primary_forecast_failed: bool = False,
) -> Dict[str, Any]:
    """Return the appropriate fallback forecast when primary fails.

    Hierarchy: primary → 30-day mean → 7-day mean → yesterday. If the
    primary forecast hasn't failed, this returns level 1 with no
    fallback forecast (the caller should keep using their primary). If
    it has failed, the function picks the deepest level the available
    history supports and returns a constant forecast at that level.

    Parameters
    ----------
    pattern : str
        Demand pattern label, used only to colour the recommendation
        text.
    series : array-like
        Historical daily ``units_sold``.
    horizon : int
        Number of days to populate in the fallback forecast.
    primary_forecast_failed : bool, default False
        Set to ``True`` when the primary forecast can't be trusted —
        e.g. the model raised, or upstream guardrails fired enough to
        warrant abandoning it.

    Returns
    -------
    dict
        Keys ``fallback_level`` (1–4), ``method_name`` (str),
        ``forecast`` (numpy array of length ``horizon``, empty at
        level 1), ``recommendation`` (str).
    """
    arr = _as_array(series)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    if not primary_forecast_failed:
        return {
            "fallback_level": 1,
            "method_name": _DEGRADATION_LEVELS[1],
            "forecast": np.zeros(0, dtype=float),
            "recommendation": "Primary forecast is OK; no fallback needed.",
        }

    if arr.size >= 30:
        level = 2
        value = float(arr[-30:].mean())
    elif arr.size >= 7:
        level = 3
        value = float(arr[-7:].mean())
    elif arr.size >= 1:
        level = 4
        value = float(arr[-1])
    else:
        level = 4
        value = 0.0

    forecast = np.full(horizon, value, dtype=float)
    method_name = _DEGRADATION_LEVELS[level]
    rec = (
        f"Primary forecast unavailable for a {pattern} SKU — falling back "
        f"to {method_name.replace('_', ' ')}. Treat the result as a "
        "rough plug, not a planning forecast."
    )
    return {
        "fallback_level": level,
        "method_name": method_name,
        "forecast": forecast,
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------


def _backtest_band(
    series: np.ndarray, window_days: int, lookback: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a rough trailing forecast band over the most recent days.

    Used by :func:`run_guardrails` to give drift and regime-change
    detection inputs they can actually compare against. For each day
    in the most recent ``window_days``, the "forecast" is the mean of
    the prior ``lookback`` days and the band is ±1.96σ around that
    mean. This is intentionally crude: it's a sanity check, not a
    re-fit of the production model.
    """
    n = series.size
    eff_window = min(window_days, max(0, n - lookback))
    if eff_window < 1:
        return (np.zeros(0), np.zeros(0), np.zeros(0))
    actuals = np.zeros(eff_window, dtype=float)
    fc = np.zeros(eff_window, dtype=float)
    lo = np.zeros(eff_window, dtype=float)
    hi = np.zeros(eff_window, dtype=float)
    for i, idx in enumerate(range(n - eff_window, n)):
        prior = series[max(0, idx - lookback) : idx]
        mean = float(prior.mean()) if prior.size else 0.0
        std = float(prior.std(ddof=0)) if prior.size else 0.0
        actuals[i] = series[idx]
        fc[i] = mean
        lo[i] = mean - _Z95 * std
        hi[i] = mean + _Z95 * std
    return actuals, fc, lo, hi


def run_guardrails(
    sku_df: pd.DataFrame,
    forecast_dict: Dict[str, Any],
    proposed_order_qty: Optional[float] = None,
) -> Dict[str, Any]:
    """Run all five guardrails on a SKU and roll up an overall recommendation.

    The drift and regime-change checks need historical forecasts and
    PI bands; production callers don't always have those, so this
    wrapper builds a small backtest band internally — a 28-day
    trailing window with a ±1.96σ envelope around the rolling mean —
    so the guardrails always have something defensible to compare
    against. The other three guardrails (confidence floor, reorder
    cap, graceful degradation) work directly off the forecast dict
    and the SKU history.

    Parameters
    ----------
    sku_df : pandas.DataFrame
        Single-SKU history including at minimum ``date`` and
        ``units_sold``.
    forecast_dict : dict
        The output of :func:`msmt.forecasting.auto_select.run_forecast`.
    proposed_order_qty : float, optional
        The reorder quantity the upstream pipeline is suggesting. If
        omitted, the reorder cap returns a not-applicable result.

    Returns
    -------
    dict
        Keys ``sku_id`` (str), ``any_fired`` (bool), ``guardrails`` (a
        dict with one entry per guardrail), and
        ``overall_recommendation`` (plain-English summary).
    """
    df = sku_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    series = df["units_sold"].astype(float).to_numpy()
    sku_id = forecast_dict.get("sku_id") or (
        str(df["sku_id"].iloc[0]) if "sku_id" in df.columns else "unknown"
    )

    # Drift + regime: build a 28-day trailing backtest band.
    actuals_bt, fc_bt, lo_bt, hi_bt = _backtest_band(
        series, window_days=28, lookback=28
    )
    drift = drift_detection(actuals_bt, fc_bt) if actuals_bt.size else {
        "fired": False,
        "direction": None,
        "cumulative_gap_pct": 0.0,
        "weeks_trending": 0,
        "recommendation": "Not enough history to evaluate drift.",
    }
    regime = regime_change_detection(actuals_bt, lo_bt, hi_bt) if actuals_bt.size else {
        "fired": False,
        "outside_count": 0,
        "window_days": 0,
        "recommendation": "Not enough history to evaluate regime change.",
    }

    # Confidence floor on the future forecast / PI.
    confidence = confidence_floor(
        forecast_dict["forecast"],
        forecast_dict["lower_95"],
        forecast_dict["upper_95"],
    )

    # Reorder cap.
    trailing_30 = float(series[-30:].mean()) if series.size >= 1 else 0.0
    if proposed_order_qty is not None:
        cap = reorder_cap(proposed_order_qty, trailing_30)
    else:
        cap = {
            "fired": False,
            "proposed_qty": 0.0,
            "cap_qty": 3.0 * trailing_30,
            "recommendation": "No proposed order provided; cap not evaluated.",
        }

    # Graceful degradation — assume primary did not fail by default.
    degradation = graceful_degradation(
        pattern=str(forecast_dict.get("pattern", "smooth")),
        series=series,
        horizon=int(len(forecast_dict["forecast"])),
        primary_forecast_failed=False,
    )

    guardrails = {
        "drift": drift,
        "confidence": confidence,
        "regime": regime,
        "cap": cap,
        "degradation": degradation,
    }
    n_fired = sum(1 for g in (drift, confidence, regime, cap) if g.get("fired"))
    any_fired = n_fired > 0

    if n_fired == 0:
        overall = "Forecast looks reliable — proceed with planned order."
    elif n_fired == 1:
        only = next(k for k, v in guardrails.items() if v.get("fired"))
        overall = (
            f"1 guardrail fired ({only}). Review the recommendation below "
            "before reordering."
        )
    else:
        overall = (
            f"{n_fired} guardrails fired. Recommend pausing auto-reorder "
            "and reviewing manually."
        )

    return {
        "sku_id": sku_id,
        "any_fired": bool(any_fired),
        "guardrails": guardrails,
        "overall_recommendation": overall,
    }
