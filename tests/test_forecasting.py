"""Smoke tests for the msmt.forecasting module.

These cover the load-bearing invariants of the forecasting pipeline
and guardrails. They are intentionally cheap so they can run on every
push.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msmt.data import generate_seller_data
from msmt.forecasting import (
    auto_select_method,
    confidence_floor,
    croston_forecast,
    drift_detection,
    graceful_degradation,
    holt_winters_forecast,
    naive_forecast,
    regime_change_detection,
    reorder_cap,
    run_forecast,
    run_guardrails,
    ses_forecast,
)


@pytest.fixture(scope="module")
def seller_df() -> pd.DataFrame:
    return generate_seller_data(n_skus=50, n_days=365, seed=42)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def test_naive_forecast_returns_correct_length() -> None:
    series = np.array([5.0, 7.0, 9.0, 6.0, 8.0])
    out = naive_forecast(series, horizon=14)
    assert isinstance(out, np.ndarray)
    assert out.shape == (14,)
    assert np.all(out == 8.0)


def test_ses_forecast_within_input_range() -> None:
    """SES forecast (a smoothed level) must lie within [min, max] of inputs."""
    series = np.array([5.0, 7.0, 9.0, 6.0, 8.0, 10.0, 7.0, 9.0])
    forecast = ses_forecast(series, horizon=7)
    assert forecast.min() >= series.min() - 1e-9
    assert forecast.max() <= series.max() + 1e-9


def test_holt_winters_handles_weekly_seasonal_data() -> None:
    """HW must run end-to-end on a realistic weekly-seasonal series."""
    rng = np.random.default_rng(0)
    weeks = 30
    base = np.array([8, 9, 10, 11, 12, 18, 16] * weeks, dtype=float)
    series = base + rng.normal(0, 1, size=base.size)
    f, lo, hi = holt_winters_forecast(series, horizon=14, season_length=7, return_pi=True)
    assert f.shape == (14,)
    assert np.all(np.isfinite(f))
    assert np.all(hi >= lo)


def test_croston_handles_zero_prefix() -> None:
    """All-zero leading days followed by sparse non-zero demand."""
    series = np.zeros(60, dtype=float)
    series[40] = 4
    series[47] = 6
    series[55] = 3
    f, lo, hi = croston_forecast(series, horizon=14, return_pi=True)
    assert f.shape == (14,)
    assert (f > 0).all(), "Croston should produce a positive constant rate"
    assert np.all(lo >= 0), "lower bound should be clipped to 0"


# ---------------------------------------------------------------------------
# Auto-select + run_forecast
# ---------------------------------------------------------------------------


def test_auto_select_method_rules() -> None:
    assert auto_select_method("intermittent", 365) == "croston"
    assert auto_select_method("smooth", 365) == "ses"
    assert auto_select_method("new_sku", 365) == "moving_average"
    assert auto_select_method("smooth", 30) == "moving_average"
    assert auto_select_method("holiday_spike", 400) == "prophet"
    assert auto_select_method("holiday_spike", 200) == "holt_winters"
    assert auto_select_method("weekly_seasonal", 365) == "holt_winters"
    assert auto_select_method("garbage", 365) == "moving_average"


def test_run_forecast_returns_required_keys(seller_df: pd.DataFrame) -> None:
    sku_id = seller_df["sku_id"].iloc[0]
    sub = seller_df[seller_df["sku_id"] == sku_id]
    result = run_forecast(sub, horizon=14)
    expected = {
        "sku_id", "pattern", "method_used", "forecast",
        "lower_95", "upper_95", "horizon_dates",
    }
    assert set(result.keys()) == expected
    assert result["forecast"].shape == (14,)
    assert result["lower_95"].shape == (14,)
    assert result["upper_95"].shape == (14,)
    assert len(result["horizon_dates"]) == 14


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_all_five_guardrails_return_fired_key() -> None:
    """Every guardrail must return a dict with a 'fired' key."""
    actuals = np.array([10.0] * 28)
    forecasts = np.array([10.5] * 28)
    lower = np.array([8.0] * 28)
    upper = np.array([12.0] * 28)

    drift = drift_detection(actuals, forecasts)
    conf = confidence_floor(np.full(7, 10.0), np.full(7, 9.0), np.full(7, 11.0))
    regime = regime_change_detection(actuals[-7:], lower[-7:], upper[-7:])
    cap = reorder_cap(50.0, trailing_30d_avg=20.0)
    deg = graceful_degradation("smooth", actuals, horizon=7)

    for g in (drift, conf, regime, cap):
        assert isinstance(g, dict) and "fired" in g
    # graceful_degradation returns a fallback_level instead of a fired flag.
    assert isinstance(deg, dict) and "fallback_level" in deg


def test_confidence_floor_fires_on_wide_pi() -> None:
    """A PI band wider than the forecast level must fire the floor."""
    forecast = np.full(7, 10.0)
    lower = np.full(7, 1.0)
    upper = np.full(7, 19.0)  # width 18, level 10 -> ratio 1.8
    out = confidence_floor(forecast, lower, upper, threshold_ratio=0.50)
    assert out["fired"] is True
    assert out["pi_ratio"] > 0.50


def test_drift_detection_fires_on_consistent_over_forecast() -> None:
    """4 weeks of forecast > actual must trip the drift guardrail."""
    actuals = np.full(28, 10.0)
    forecasts = np.full(28, 14.0)  # 40% over every day, every week
    out = drift_detection(actuals, forecasts, threshold_weeks=3, tolerance_pct=0.20)
    assert out["fired"] is True
    assert out["direction"] == "over"
    assert out["weeks_trending"] >= 3
    assert out["cumulative_gap_pct"] > 0.20


def test_run_guardrails_smoke(seller_df: pd.DataFrame) -> None:
    """End-to-end run_guardrails returns the canonical report shape."""
    sku_id = seller_df["sku_id"].iloc[0]
    sub = seller_df[seller_df["sku_id"] == sku_id]
    forecast_dict = run_forecast(sub, horizon=14)
    report = run_guardrails(sub, forecast_dict, proposed_order_qty=50.0)
    assert set(report.keys()) == {
        "sku_id", "any_fired", "guardrails", "overall_recommendation"
    }
    assert set(report["guardrails"].keys()) == {
        "drift", "confidence", "regime", "cap", "degradation"
    }
    assert isinstance(report["overall_recommendation"], str)
