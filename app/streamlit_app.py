"""Main Street Marketplace Toolkit — interactive Streamlit app.

Single-file Streamlit app exposing all three pillars of the toolkit
(marketplace integrity, supply resilience, demand forecasting) to a
non-technical audience: SBDC counselors, state commerce program
staff, niche marketplace operators, and SMB sellers.

Run from the repo root with::

    streamlit run app/streamlit_app.py

All data shown by the app is **synthetic**, generated in-memory from
``msmt.data.generate_seller_data``. The app does not read or write any
file paths and does not call any external APIs.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from msmt.data import PATTERNS, generate_seller_data
from msmt.forecasting import (
    auto_select_method,
    holt_winters_forecast,
    holts_forecast,
    moving_average_forecast,
    naive_forecast,
    prophet_forecast,
    croston_forecast,
    run_forecast,
    run_guardrails,
    seasonal_naive_forecast,
    ses_forecast,
)
from msmt.integrity import (
    SIGNALS,
    compute_scorecard,
    concentration_audit,
    scorecard_for_synthetic_seller,
)
from msmt.resilience import (
    stockout_heatmap_data,
    suppression_adjusted_stockout_cost,
)


PAGE_HOME = "Home"
PAGE_INTEGRITY = "Marketplace Integrity"
PAGE_RESILIENCE = "Supply Resilience"
PAGE_FORECAST = "Demand Forecasting"

LEVEL_COLORS = {
    "critical": "#c0392b",
    "high": "#e67e22",
    "medium": "#f1c40f",
    "low": "#27ae60",
    "good": "#27ae60",
    "fair": "#f1c40f",
    "poor": "#c0392b",
    "moderate": "#f1c40f",
}


# ---------------------------------------------------------------------------
# Cached data generators
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_seller_data(n_skus: int, n_days: int, seed: int) -> pd.DataFrame:
    return generate_seller_data(n_skus=n_skus, n_days=n_days, seed=seed)


@st.cache_data(show_spinner=False)
def _cached_heatmap(seller_df: pd.DataFrame, service_level: float) -> pd.DataFrame:
    return stockout_heatmap_data(seller_df, service_level=service_level)


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------


def render_home() -> None:
    st.title("Main Street Marketplace Toolkit")
    st.subheader(
        "Open, free marketplace intelligence for U.S. small businesses."
    )

    st.write(
        "An open-source Python toolkit for U.S. SBDC counselors, state "
        "commerce program staff, niche marketplace operators, and small "
        "marketplace sellers. The toolkit translates the kind of "
        "analytics enterprise marketplaces use internally into "
        "transparent, auditable modules anyone can run on their own "
        "laptop. MIT licensed. Free forever."
    )

    cols = st.columns(3)
    with cols[0]:
        st.markdown("### Marketplace Integrity")
        st.write(
            "Score a seller on ten fulfillment, post-purchase, and "
            "content signals. Surface suppression risk and the top "
            "issues to fix before the next reorder."
        )
    with cols[1]:
        st.markdown("### Supply Resilience")
        st.write(
            "Classify each SKU's demand pattern, compute safety stock "
            "and reorder points, and rank SKUs by current stockout "
            "risk — including the platform suppression tail."
        )
    with cols[2]:
        st.markdown("### Forecasting & Guardrails")
        st.write(
            "Auto-select a forecasting method per SKU and wrap the "
            "result in five plain-language guardrails so a counselor "
            "knows when to trust it."
        )

    st.info(
        "**Get started:** pick a module from the sidebar on the left. "
        "Every page is interactive and runs on synthetic data, so you "
        "can explore without uploading anything."
    )

    st.caption(
        "Built by Ayush Tripathi. MIT Licensed. Free forever."
    )


def render_integrity() -> None:
    st.title("Marketplace Integrity")
    st.write(
        "Score a small seller against ten signals across fulfillment, "
        "post-purchase quality, and listing content. The scorecard's "
        "weights and benchmarks are practitioner estimates from "
        "publicly available platform guidance — not platform-disclosed "
        "algorithmic weights."
    )

    mode = st.radio(
        "Input mode",
        ["Use a demo seller", "Enter my own metrics"],
        horizontal=True,
    )

    if mode == "Use a demo seller":
        seed = st.slider("Demo seller seed", 0, 100, 42, 1)
        scorecard = scorecard_for_synthetic_seller(seed=seed)
        metrics = {
            name: info["value"]
            for name, info in scorecard["signal_scores"].items()
        }
    else:
        st.caption(
            "Enter your own metrics. Defaults are at each signal's "
            "'good' benchmark."
        )
        metrics = {}
        cols = st.columns(2)
        for i, sig in enumerate(SIGNALS):
            col = cols[i % 2]
            with col:
                if sig.name == "image_count":
                    metrics[sig.name] = float(
                        st.number_input(
                            sig.name.replace("_", " "),
                            min_value=0,
                            max_value=15,
                            value=int(sig.benchmark_good),
                            step=1,
                            help=sig.description,
                        )
                    )
                elif sig.name == "listing_quality_score":
                    metrics[sig.name] = float(
                        st.slider(
                            sig.name.replace("_", " "),
                            min_value=0,
                            max_value=100,
                            value=int(sig.benchmark_good),
                            help=sig.description,
                        )
                    )
                elif sig.name == "customer_feedback_score":
                    metrics[sig.name] = float(
                        st.slider(
                            sig.name.replace("_", " "),
                            min_value=1.0,
                            max_value=5.0,
                            value=float(sig.benchmark_good),
                            step=0.1,
                            help=sig.description,
                        )
                    )
                else:
                    metrics[sig.name] = float(
                        st.slider(
                            sig.name.replace("_", " "),
                            min_value=0.0,
                            max_value=1.0,
                            value=float(sig.benchmark_good),
                            step=0.01,
                            help=sig.description,
                        )
                    )
        scorecard = compute_scorecard(metrics)

    # Headline
    col_score, col_risk = st.columns(2)
    with col_score:
        st.metric("Overall score", f"{scorecard['overall_score']:.1f} / 100")
    with col_risk:
        st.metric(
            "Suppression risk", scorecard["suppression_risk"].upper()
        )

    if scorecard["overall_score"] < 50:
        st.error(
            "**High suppression risk.** The seller's profile sits below "
            "the threshold most marketplaces use for Buy-Box and "
            "ranking eligibility. Address the top issues below before "
            "the next reorder cycle."
        )
    elif scorecard["overall_score"] < 75:
        st.warning(
            "**Medium suppression risk.** Listing-eligibility metrics "
            "are functional but vulnerable. Knock out one or two of "
            "the top issues to move into the safe zone."
        )

    # Signal breakdown
    st.subheader("Signal breakdown")
    st.write(
        "Each row is one signal. Score is on a 0–100 scale interpolated "
        "between the 'poor' and 'good' benchmarks for that signal."
    )
    rows = []
    for name, info in scorecard["signal_scores"].items():
        rows.append(
            {
                "signal": name,
                "value": round(info["value"], 3),
                "score": round(info["score_0_to_100"], 1),
                "rating": info["rating"],
                "weight": info["weight"],
            }
        )
    score_df = pd.DataFrame(rows).sort_values("score")
    st.dataframe(score_df, width="stretch", hide_index=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(
        score_df["signal"],
        score_df["score"],
        color=[LEVEL_COLORS[r] for r in score_df["rating"]],
    )
    ax.axvline(50, color="#7f8c8d", linestyle="--", linewidth=0.8)
    ax.axvline(80, color="#7f8c8d", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (/100)")
    ax.set_title("Signal scores", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Top issues
    if scorecard["top_issues"]:
        st.subheader("Top issues to fix")
        for issue, rec in zip(
            scorecard["top_issues"], scorecard["recommendations"]
        ):
            st.warning(f"**{issue['plain_english']}**\n\n{rec}")
    else:
        st.success("No signals are flagged as 'poor' or 'fair'.")

    # Concentration analysis
    st.divider()
    st.subheader("Catalog concentration analysis")
    st.write(
        "How exposed is the seller to a single-SKU outage? The "
        "Herfindahl-Hirschman Index (HHI) measures how concentrated "
        "category volume is across SKUs. Thresholds shown are the U.S. "
        "Department of Justice merger-review thresholds."
    )
    n_skus_conc = st.slider(
        "SKUs in synthetic catalog", 10, 100, 50, 5,
        key="conc_n_skus",
    )
    seed_conc = st.number_input(
        "Catalog seed", value=42, step=1, key="conc_seed"
    )
    with st.spinner("Generating synthetic catalog…"):
        catalog = _cached_seller_data(int(n_skus_conc), 365, int(seed_conc))
    audit = concentration_audit(catalog)

    st.dataframe(audit["summary_df"].round(2), width="stretch", hide_index=True)
    st.info(audit["audit_narrative"])

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    summary = audit["summary_df"]
    ax2.bar(
        summary["category"],
        summary["hhi"],
        color=[LEVEL_COLORS[l] for l in summary["concentration_level"]],
    )
    ax2.axhline(1500, color="#7f8c8d", linestyle="--", linewidth=0.9, label="DOJ moderate (1,500)")
    ax2.axhline(2500, color="#34495e", linestyle="--", linewidth=0.9, label="DOJ high (2,500)")
    ax2.set_ylabel("HHI")
    ax2.set_title("HHI by category", loc="left")
    ax2.tick_params(axis="x", rotation=30)
    for label in ax2.get_xticklabels():
        label.set_horizontalalignment("right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)


def render_resilience() -> None:
    st.title("Supply Resilience")
    st.write(
        "Classify each SKU's demand pattern, compute the appropriate "
        "safety stock and reorder point, and rank the catalog by current "
        "stockout exposure. The pipeline runs end-to-end on synthetic "
        "data; the same code works on a real seller-portal export."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_skus = st.slider("Number of SKUs", 10, 100, 50, 5)
    with col2:
        seed = st.number_input("Seed", value=42, step=1, key="res_seed")
    with col3:
        service_level = st.selectbox(
            "Service level",
            [0.90, 0.95, 0.97, 0.98, 0.99],
            index=1,
        )

    with st.spinner("Generating synthetic catalog and running pipeline…"):
        catalog = _cached_seller_data(int(n_skus), 365, int(seed))
        heatmap = _cached_heatmap(catalog, float(service_level))

    counts = heatmap["risk_level"].value_counts().reindex(
        ["critical", "high", "medium", "low"]
    ).fillna(0).astype(int)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Critical", int(counts.get("critical", 0)))
    metric_cols[1].metric("High", int(counts.get("high", 0)))
    metric_cols[2].metric("Medium", int(counts.get("medium", 0)))
    metric_cols[3].metric("Low", int(counts.get("low", 0)))

    st.subheader("Stockout risk heatmap")
    st.write(
        "One row per SKU, sorted by stockout-risk score (highest first). "
        "The 'action' column is the recommendation to walk through with "
        "the seller."
    )
    show_cols = [
        "sku_id", "pattern", "method_used", "rop", "safety_stock",
        "current_stock", "risk_score", "risk_level",
        "days_until_stockout", "action",
    ]
    st.dataframe(
        heatmap[show_cols].round(2),
        width="stretch",
        hide_index=True,
    )

    # Risk-level distribution chart
    fig, ax = plt.subplots(figsize=(9, 3.6))
    order = ["critical", "high", "medium", "low"]
    bar_counts = counts.reindex(order).fillna(0).astype(int)
    ax.bar(
        bar_counts.index,
        bar_counts.values,
        color=[LEVEL_COLORS[l] for l in bar_counts.index],
    )
    for i, v in enumerate(bar_counts.values):
        ax.text(i, v + 0.3, str(int(v)), ha="center", fontsize=10)
    ax.set_ylabel("Number of SKUs")
    ax.set_title("SKUs by risk level", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Suppression cost calculator
    st.divider()
    st.subheader("Stockout cost calculator (with platform suppression)")
    st.write(
        "When a listing goes out of stock, the lost margin during the "
        "outage is only part of the cost — marketplace ranking algorithms "
        "tend to demote listings that go unavailable, and most listings "
        "take some weeks to climb back. This calculator estimates that "
        "tail."
    )
    st.caption(
        "The default 3.0× suppression multiplier and 21-day recovery "
        "window are practitioner estimates from industry observation, "
        "not figures disclosed by any marketplace platform."
    )

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        daily_profit = st.number_input(
            "Daily profit when in stock ($)", value=120.0, step=10.0
        )
    with cc2:
        stockout_days = st.slider("Stockout days", 1, 30, 7, 1)
    with cc3:
        mult = st.slider("Suppression multiplier", 1.0, 5.0, 3.0, 0.1)
    with cc4:
        recovery = st.slider("Recovery days", 0, 60, 21, 1)

    cost = suppression_adjusted_stockout_cost(
        daily_profit=float(daily_profit),
        stockout_days=int(stockout_days),
        suppression_multiplier=float(mult),
        recovery_days=int(recovery),
    )
    out_cols = st.columns(3)
    out_cols[0].metric("Direct cost", f"${cost['direct_cost']:,.0f}")
    out_cols[1].metric("Suppression tail", f"${cost['suppression_cost']:,.0f}")
    out_cols[2].metric("Total cost", f"${cost['total_cost']:,.0f}")


def _force_method(sub: pd.DataFrame, pattern: str, horizon: int) -> Dict[str, Any]:
    series = (
        sub.sort_values("date")["units_sold"].astype(float).to_numpy()
    )
    dates = pd.DatetimeIndex(sub.sort_values("date")["date"])
    method = auto_select_method(pattern, series_length=len(series))
    if method == "naive":
        f, lo, hi = naive_forecast(series, horizon, return_pi=True)
    elif method == "seasonal_naive":
        f, lo, hi = seasonal_naive_forecast(series, horizon, return_pi=True)
    elif method == "moving_average":
        f, lo, hi = moving_average_forecast(series, horizon, return_pi=True)
    elif method == "ses":
        f, lo, hi = ses_forecast(series, horizon, return_pi=True)
    elif method == "holts":
        f, lo, hi = holts_forecast(series, horizon, return_pi=True)
    elif method == "holt_winters":
        f, lo, hi = holt_winters_forecast(series, horizon, return_pi=True)
    elif method == "croston":
        f, lo, hi = croston_forecast(series, horizon, return_pi=True)
    else:  # prophet
        f, lo, hi = prophet_forecast(series, horizon, dates=dates)
    horizon_dates = pd.date_range(
        dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    return {
        "sku_id": str(sub["sku_id"].iloc[0]),
        "pattern": pattern,
        "method_used": method,
        "forecast": np.asarray(f, dtype=float),
        "lower_95": np.asarray(lo, dtype=float),
        "upper_95": np.asarray(hi, dtype=float),
        "horizon_dates": horizon_dates,
        "series": series,
        "dates": dates,
    }


def render_forecasting() -> None:
    st.title("Demand Forecasting & Guardrails")
    st.write(
        "Pick a demand pattern, generate a representative SKU, and run "
        "the auto-selected forecasting method. The toolkit picks the "
        "method based on the demand archetype — you don't have to know "
        "what an exponential smoother is to use it."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        pattern = st.selectbox("Demand pattern", list(PATTERNS), index=0)
    with col2:
        seed = st.number_input("SKU seed", value=42, step=1, key="fc_seed")
    with col3:
        horizon = st.slider("Forecast horizon (days)", 7, 60, 28, 1)

    with st.spinner("Generating SKU and running forecast…"):
        try:
            catalog = _cached_seller_data(50, 365, int(seed))
            sku_ids = catalog[catalog["pattern"] == pattern]["sku_id"].unique()
            if len(sku_ids) == 0:
                st.warning(
                    f"No synthetic SKUs of pattern '{pattern}' at seed "
                    f"{seed}. Try a different seed."
                )
                return
            sub = catalog[catalog["sku_id"] == sku_ids[0]]
            result = _force_method(sub, pattern, int(horizon))
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.error(f"Forecast failed: {exc}")
            return

    info_cols = st.columns(3)
    info_cols[0].metric("Representative SKU", result["sku_id"])
    info_cols[1].metric("Method used", result["method_used"])
    info_cols[2].metric("Horizon", f"{horizon} days")

    # Chart
    st.subheader("Actuals + forecast + 95% prediction interval")
    history_window = 90
    h_dates = result["dates"][-history_window:]
    h_series = result["series"][-history_window:]
    f_dates = result["horizon_dates"]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(h_dates, h_series, color="#34495e", linewidth=1.0, label="actuals (last 90d)")
    ax.plot(f_dates, result["forecast"], color="#2980b9", linewidth=2.0,
            label=f"forecast ({result['method_used']})")
    ax.fill_between(f_dates, result["lower_95"], result["upper_95"],
                    color="#3498db", alpha=0.18, label="95% PI")
    ax.axvline(h_dates[-1], color="#bdc3c7", linestyle="--", linewidth=0.8)
    ax.set_ylabel("units / day")
    ax.set_xlabel("date")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Guardrails
    st.subheader("Guardrails")
    st.write(
        "Five sanity checks that wrap the raw forecast before it drives "
        "a reorder. A 'fired' guardrail is the cue to slow down and "
        "review the recommendation."
    )

    forecast_dict_for_guards = {
        "sku_id": result["sku_id"],
        "pattern": result["pattern"],
        "method_used": result["method_used"],
        "forecast": result["forecast"],
        "lower_95": result["lower_95"],
        "upper_95": result["upper_95"],
        "horizon_dates": result["horizon_dates"],
    }
    proposed = float(result["series"][-30:].mean()) * 30  # ~30 days of stock
    try:
        report = run_guardrails(sub, forecast_dict_for_guards, proposed_order_qty=proposed)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Guardrails failed: {exc}")
        return

    overall = report["overall_recommendation"]
    if report["any_fired"]:
        st.warning(f"**{overall}**")
    else:
        st.success(f"**{overall}**")

    rows = []
    for name, g in report["guardrails"].items():
        if name == "degradation":
            status = f"level {g.get('fallback_level')} ({g.get('method_name')})"
        else:
            status = "🚨 FIRED" if g.get("fired") else "✅ ok"
        rows.append(
            {
                "guardrail": name,
                "status": status,
                "finding": g.get("recommendation", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Main Street Marketplace Toolkit",
        page_icon=":bar_chart:",
        layout="wide",
    )

    with st.sidebar:
        st.markdown("## Main Street Marketplace Toolkit")
        st.caption("Open, free marketplace intelligence.")
        page = st.radio(
            "Module",
            [PAGE_HOME, PAGE_INTEGRITY, PAGE_RESILIENCE, PAGE_FORECAST],
            index=0,
        )
        st.divider()
        st.caption(
            "Built by Ayush Tripathi. MIT Licensed. Free forever.\n\n"
            "All data shown in this app is synthetic — generated "
            "in-memory, never read from disk."
        )

    if page == PAGE_HOME:
        render_home()
    elif page == PAGE_INTEGRITY:
        render_integrity()
    elif page == PAGE_RESILIENCE:
        render_resilience()
    elif page == PAGE_FORECAST:
        render_forecasting()


if __name__ == "__main__":
    main()
