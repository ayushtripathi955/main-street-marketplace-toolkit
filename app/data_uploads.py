"""CSV upload templates and validators for the Streamlit app.

Two upload slots, each parsed and validated independently:

* **Listing-quality file** (Marketplace Integrity input). One row per
  seller (or a single row) with the ten signal columns the integrity
  scorecard reads.
* **Inventory + sales history file** (Supply Resilience and Demand
  Forecasting input). Long-format daily rows with ``sku_id``, ``date``,
  ``units_sold``, ``stock_on_hand``, ``lead_time_days``. The same
  upload feeds both modules; the forecasting page never asks the
  seller for a second file.

Every parser returns ``(dataframe, [])`` on success or
``(None, ["plain English message", ...])`` on failure. No raw
exception ever bubbles up to the user; callers render the messages
with ``st.error``.

Nothing in this module reads or writes the filesystem. The CSV bytes
the seller uploaded are parsed in-memory via :mod:`io.StringIO` and
discarded when the Streamlit session ends.
"""

from __future__ import annotations

import io
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Listing-quality (integrity) schema
# ---------------------------------------------------------------------------

#: Ordered list of the ten integrity signals the scorecard reads. Order
#: matches the template CSV column order for readability.
INTEGRITY_COLUMNS: Tuple[str, ...] = (
    "on_time_shipment_rate",
    "valid_tracking_rate",
    "pre_fulfillment_cancel_rate",
    "late_dispatch_rate",
    "return_rate",
    "order_defect_rate",
    "customer_feedback_score",
    "listing_quality_score",
    "image_count",
    "keyword_coverage",
)

#: Per-signal acceptable value ranges. Used by the validator to surface
#: an out-of-range message in plain English. ``None`` on either bound
#: means "no constraint on that side".
_INTEGRITY_RANGES = {
    "on_time_shipment_rate":       (0.0, 1.0),
    "valid_tracking_rate":         (0.0, 1.0),
    "pre_fulfillment_cancel_rate": (0.0, 1.0),
    "late_dispatch_rate":          (0.0, 1.0),
    "return_rate":                 (0.0, 1.0),
    "order_defect_rate":           (0.0, 1.0),
    "customer_feedback_score":     (1.0, 5.0),
    "listing_quality_score":       (0.0, 100.0),
    "image_count":                 (0.0, None),
    "keyword_coverage":            (0.0, 1.0),
}

#: Example values used in the downloadable template, chosen so the
#: scorecard returns a realistic mixed-quality result out of the box.
_INTEGRITY_EXAMPLE = {
    "on_time_shipment_rate":       0.94,
    "valid_tracking_rate":         0.96,
    "pre_fulfillment_cancel_rate": 0.03,
    "late_dispatch_rate":          0.06,
    "return_rate":                 0.18,
    "order_defect_rate":           0.012,
    "customer_feedback_score":     4.5,
    "listing_quality_score":       72,
    "image_count":                 5,
    "keyword_coverage":            0.65,
}


# ---------------------------------------------------------------------------
# Inventory + sales (resilience + forecasting) schema
# ---------------------------------------------------------------------------

INVENTORY_COLUMNS: Tuple[str, ...] = (
    "sku_id",
    "date",
    "units_sold",
    "stock_on_hand",
    "lead_time_days",
)

_INVENTORY_NUMERIC = {"units_sold", "stock_on_hand", "lead_time_days"}


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------


def make_integrity_template_csv() -> bytes:
    """Return a ready-to-fill integrity template as CSV bytes."""
    df = pd.DataFrame([_INTEGRITY_EXAMPLE], columns=list(INTEGRITY_COLUMNS))
    return df.to_csv(index=False).encode("utf-8")


def make_inventory_template_csv() -> bytes:
    """Return a ready-to-fill inventory + sales template as CSV bytes.

    The template covers two SKUs over seven days so a seller can see
    the long-format shape without having to read documentation.
    """
    dates = pd.date_range("2026-01-01", periods=7, freq="D")
    rows = []
    rng = np.random.default_rng(0)
    for sku, lead_time, base_stock, mean_units in [
        ("SKU-DEMO-0001", 14, 120, 9),
        ("SKU-DEMO-0002", 21, 80, 4),
    ]:
        stock = base_stock
        for d in dates:
            sold = int(max(0, rng.poisson(mean_units)))
            sold = min(sold, stock)
            rows.append(
                {
                    "sku_id": sku,
                    "date": d.strftime("%Y-%m-%d"),
                    "units_sold": sold,
                    "stock_on_hand": stock,
                    "lead_time_days": lead_time,
                }
            )
            stock = max(0, stock - sold)
    df = pd.DataFrame(rows, columns=list(INVENTORY_COLUMNS))
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case + trim header whitespace, in place-equivalent."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _read_csv_bytes(raw: bytes, label: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Parse CSV bytes; return ``(df, [])`` or ``(None, errors)``."""
    if raw is None or len(raw) == 0:
        return None, [
            f"{label}: the file looks empty. Did the export finish? "
            "Try re-exporting it from your seller portal."
        ]
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except pd.errors.EmptyDataError:
        return None, [
            f"{label}: the file is empty or has no header row. "
            "Use the Download template button to get a starting file."
        ]
    except pd.errors.ParserError as exc:
        return None, [
            f"{label}: this doesn't look like a valid CSV "
            f"({exc.__class__.__name__}). Open it in a spreadsheet, "
            "check for stray commas, and re-save."
        ]
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return None, [
            f"{label}: we couldn't read this file "
            f"({exc.__class__.__name__}). Try re-exporting as UTF-8 CSV."
        ]
    if df.empty:
        return None, [
            f"{label}: the file has headers but no data rows. "
            "Add at least one row of data."
        ]
    return _normalize_columns(df), []


def _missing_columns(df: pd.DataFrame, required: Sequence[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def parse_integrity_csv(raw: bytes) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Parse and validate a listing-quality (integrity) CSV.

    Returns a 1-row DataFrame whose columns are the ten signal names.
    If the upload has multiple rows, the first one is used and an
    informational note is appended (not an error).
    """
    label = "Listing-quality file"
    df, errors = _read_csv_bytes(raw, label)
    if df is None:
        return None, errors

    missing = _missing_columns(df, INTEGRITY_COLUMNS)
    if missing:
        return None, [
            f"{label}: missing required column(s): "
            f"{', '.join(missing)}. The Download template button shows "
            "the exact headers."
        ]

    # Keep only the columns we use; coerce numerics.
    df = df[list(INTEGRITY_COLUMNS)].copy()
    bad: List[str] = []
    for col in INTEGRITY_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad_rows = df.index[coerced.isna() & df[col].notna()].tolist()
        if bad_rows:
            bad.append(
                f"{label}: column '{col}' has non-numeric value(s) on "
                f"row(s) {[r + 2 for r in bad_rows]} "
                "(row numbers count the header as row 1)."
            )
        df[col] = coerced

    if bad:
        return None, bad

    # Use the first row only; signal-table is a single seller snapshot.
    first = df.iloc[[0]].copy()

    # Range check.
    out_of_range: List[str] = []
    for col, (lo, hi) in _INTEGRITY_RANGES.items():
        val = float(first[col].iloc[0]) if not pd.isna(first[col].iloc[0]) else None
        if val is None:
            out_of_range.append(
                f"{label}: column '{col}' is empty. Fill it in or use "
                "the template's example value."
            )
            continue
        if lo is not None and val < lo:
            out_of_range.append(
                f"{label}: '{col}' is {val}, below the minimum {lo}. "
                "Rate columns are fractions between 0 and 1 (so 0.94 "
                "means 94%)."
            )
        if hi is not None and val > hi:
            out_of_range.append(
                f"{label}: '{col}' is {val}, above the maximum {hi}. "
                "Check that you used a fraction, not a percent."
            )

    if out_of_range:
        return None, out_of_range

    return first.reset_index(drop=True), []


def parse_inventory_csv(raw: bytes) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Parse and validate the long-format inventory + sales CSV.

    The returned DataFrame is sorted by ``sku_id`` then ``date`` and
    has the exact column set the resilience and forecasting modules
    expect.
    """
    label = "Inventory + sales file"
    df, errors = _read_csv_bytes(raw, label)
    if df is None:
        return None, errors

    missing = _missing_columns(df, INVENTORY_COLUMNS)
    if missing:
        return None, [
            f"{label}: missing required column(s): "
            f"{', '.join(missing)}. The Download template button shows "
            "the exact headers."
        ]

    df = df[list(INVENTORY_COLUMNS)].copy()

    # Parse dates.
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    bad_date_rows = df.index[parsed_dates.isna() & df["date"].notna()].tolist()
    if bad_date_rows:
        return None, [
            f"{label}: column 'date' has value(s) that don't parse as "
            f"dates on row(s) {[r + 2 for r in bad_date_rows[:5]]}"
            f"{' (showing first 5)' if len(bad_date_rows) > 5 else ''}. "
            "Use YYYY-MM-DD format, e.g. 2026-01-15."
        ]
    df["date"] = parsed_dates

    bad: List[str] = []
    for col in _INVENTORY_NUMERIC:
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad_rows = df.index[coerced.isna() & df[col].notna()].tolist()
        if bad_rows:
            bad.append(
                f"{label}: column '{col}' has non-numeric value(s) on "
                f"row(s) {[r + 2 for r in bad_rows[:5]]}"
                f"{' (showing first 5)' if len(bad_rows) > 5 else ''}."
            )
        df[col] = coerced

    if bad:
        return None, bad

    # Range check: non-negative counts.
    neg: List[str] = []
    for col in _INVENTORY_NUMERIC:
        bad_rows = df.index[df[col] < 0].tolist()
        if bad_rows:
            neg.append(
                f"{label}: column '{col}' has negative value(s) on "
                f"row(s) {[r + 2 for r in bad_rows[:5]]}. "
                "Counts can't be negative."
            )

    if neg:
        return None, neg

    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    if (df["sku_id"] == "").any():
        empty_rows = df.index[df["sku_id"] == ""].tolist()
        return None, [
            f"{label}: column 'sku_id' is empty on row(s) "
            f"{[r + 2 for r in empty_rows[:5]]}. Every row needs a SKU."
        ]

    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df, []


def integrity_dataframe_to_metrics(df: pd.DataFrame) -> dict:
    """Convert a parsed integrity DataFrame back into the dict the
    scorecard expects.
    """
    row = df.iloc[0]
    return {col: float(row[col]) for col in INTEGRITY_COLUMNS}
