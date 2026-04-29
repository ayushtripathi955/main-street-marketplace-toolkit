"""Demand forecasting + guardrails — pillar 3 of the toolkit.

This subpackage exposes the forecasting tools the rest of the toolkit
and the practitioner article series rely on:

* a small library of baselines (:mod:`msmt.forecasting.baselines`),
* Croston's method for intermittent demand
  (:mod:`msmt.forecasting.croston`),
* a Prophet wrapper with a pure-numpy fallback when Prophet isn't
  installed (:mod:`msmt.forecasting.prophet_wrapper`),
* an automatic method selector and end-to-end forecast pipeline
  (:mod:`msmt.forecasting.auto_select`), and
* five guardrails that wrap a raw forecast with sanity checks before
  it drives a reorder (:mod:`msmt.forecasting.guardrails`).

Most callers will only need :func:`run_forecast`, :func:`batch_forecast`,
and :func:`run_guardrails`; the lower-level functions are exposed for
power users and for the walkthrough notebook.
"""

from msmt.forecasting.auto_select import (
    auto_select_method,
    batch_forecast,
    run_forecast,
)
from msmt.forecasting.baselines import (
    holt_winters_forecast,
    holts_forecast,
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    ses_forecast,
)
from msmt.forecasting.croston import croston_forecast
from msmt.forecasting.guardrails import (
    confidence_floor,
    drift_detection,
    graceful_degradation,
    regime_change_detection,
    reorder_cap,
    run_guardrails,
)
from msmt.forecasting.prophet_wrapper import (
    is_prophet_available,
    prophet_forecast,
)

__all__ = [
    # baselines
    "naive_forecast",
    "seasonal_naive_forecast",
    "moving_average_forecast",
    "ses_forecast",
    "holts_forecast",
    "holt_winters_forecast",
    # specialised
    "croston_forecast",
    "prophet_forecast",
    "is_prophet_available",
    # auto-select
    "auto_select_method",
    "run_forecast",
    "batch_forecast",
    # guardrails
    "drift_detection",
    "confidence_floor",
    "regime_change_detection",
    "reorder_cap",
    "graceful_degradation",
    "run_guardrails",
]
