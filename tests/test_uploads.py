"""Tests for the app CSV upload templates and validators.

These tests run against the pure-python parser layer in
``app.data_uploads`` — no Streamlit runtime is needed. They cover the
seven validation cases plus the demo-path-still-works invariant and
the "sales reused for forecasting" requirement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# The app/ folder is not a package; add it to sys.path so we can import
# data_uploads as a top-level module the same way Streamlit does.
_APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import data_uploads as du  # type: ignore[import-not-found]

from msmt.data import generate_seller_data
from msmt.forecasting import run_forecast
from msmt.integrity import compute_scorecard
from msmt.resilience import stockout_heatmap_data


# ---------------------------------------------------------------------------
# Template round-trip (the seven cases — case 1: valid integrity)
# ---------------------------------------------------------------------------


def test_integrity_template_parses_clean() -> None:
    df, errors = du.parse_integrity_csv(du.make_integrity_template_csv())
    assert errors == []
    assert df is not None and df.shape == (1, 10)
    assert set(df.columns) == set(du.INTEGRITY_COLUMNS)


def test_inventory_template_parses_clean() -> None:
    """Case 2 + 3: valid inventory file (also serves as the sales upload)."""
    df, errors = du.parse_inventory_csv(du.make_inventory_template_csv())
    assert errors == []
    assert df is not None
    assert set(df.columns) == set(du.INVENTORY_COLUMNS)
    assert df["sku_id"].nunique() >= 2
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


# ---------------------------------------------------------------------------
# Failure modes (cases 4–7)
# ---------------------------------------------------------------------------


def test_missing_required_column_friendly_message() -> None:
    """Case 4: a file missing a required column."""
    raw = b"on_time_shipment_rate,valid_tracking_rate\n0.97,0.95\n"
    df, errors = du.parse_integrity_csv(raw)
    assert df is None
    assert errors and "missing required column" in errors[0].lower()


def test_non_numeric_value_in_numeric_column() -> None:
    """Case 5: a non-numeric in a numeric column."""
    bad = du.make_integrity_template_csv().replace(b"0.94", b"notnum")
    df, errors = du.parse_integrity_csv(bad)
    assert df is None
    assert errors and "non-numeric" in errors[0].lower()
    assert "row(s) [2]" in errors[0]


def test_out_of_range_rate() -> None:
    """Case 6: a rate column above 1.0."""
    bad = du.make_integrity_template_csv().replace(b"0.94", b"1.5")
    df, errors = du.parse_integrity_csv(bad)
    assert df is None
    assert errors and "above the maximum" in errors[0].lower()


def test_empty_file() -> None:
    """Case 7: an empty file."""
    df, errors = du.parse_integrity_csv(b"")
    assert df is None
    assert errors and "empty" in errors[0].lower()


def test_invalid_date_friendly_message() -> None:
    """An invalid date string in the inventory file produces a friendly message."""
    bad = du.make_inventory_template_csv().replace(b"2026-01-01", b"not-a-date")
    df, errors = du.parse_inventory_csv(bad)
    assert df is None
    assert errors and "date" in errors[0].lower()


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_demo_path_unchanged() -> None:
    """The synthetic-data path must keep returning a valid catalog."""
    df = generate_seller_data(n_skus=10, n_days=120, seed=42)
    assert set(["sku_id", "date", "units_sold", "stock_on_hand", "lead_time_days"]
               ).issubset(df.columns)
    # And the heatmap pipeline keeps working on it.
    heat = stockout_heatmap_data(df, service_level=0.95)
    assert len(heat) == df["sku_id"].nunique()


def test_inventory_upload_feeds_both_resilience_and_forecasting() -> None:
    """The single inventory upload must satisfy both downstream modules
    without a second sales-history upload."""
    df, errors = du.parse_inventory_csv(du.make_inventory_template_csv())
    assert errors == []
    # Resilience contract is the strict superset; forecasting contract is
    # {date, sku_id, units_sold}, which is a subset.
    res_required = {"sku_id", "date", "units_sold", "stock_on_hand", "lead_time_days"}
    fc_required = {"date", "sku_id", "units_sold"}
    assert res_required.issubset(df.columns)
    assert fc_required.issubset(df.columns)

    # Forecast runs end-to-end on the uploaded SKU.
    one_sku = df[df["sku_id"] == df["sku_id"].iloc[0]]
    result = run_forecast(one_sku, horizon=7)
    assert result["forecast"].shape == (7,)


def test_integrity_dataframe_to_metrics_round_trip() -> None:
    """Parsed integrity DataFrame feeds compute_scorecard without surgery."""
    df, errors = du.parse_integrity_csv(du.make_integrity_template_csv())
    assert errors == []
    metrics = du.integrity_dataframe_to_metrics(df)
    scorecard = compute_scorecard(metrics)
    assert 0.0 <= scorecard["overall_score"] <= 100.0
    assert scorecard["suppression_risk"] in {"low", "medium", "high"}
