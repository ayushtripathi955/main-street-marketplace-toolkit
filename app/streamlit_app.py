"""Main Street Marketplace Toolkit — interactive Streamlit app.

Single-file Streamlit app exposing all three pillars of the toolkit
(marketplace integrity, supply resilience, demand forecasting) to a
non-technical audience: SBDC counselors, state commerce program
staff, niche marketplace operators, and SMB sellers.

Run from the repo root with::

    streamlit run app/streamlit_app.py

All data shown by the app is **synthetic**, generated in-memory from
``msmt.data.generate_seller_data``. The app does not read or write any
file paths (other than the brand assets it ships with) and does not
call any external APIs.
"""

# Note: Streamlit Community Cloud's built-in analytics dashboard tracks
# unique app sessions and geographic distribution. View at:
# share.streamlit.io -> main-street-marketplace -> Analytics tab

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# data_uploads.py lives next to this file. Add app/ to sys.path so the
# import works whether the app is launched from the repo root (the
# documented entry point) or from elsewhere.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import data_uploads as du  # noqa: E402

from msmt.data import PATTERNS, generate_seller_data
from msmt.forecasting import (
    auto_select_method,
    croston_forecast,
    holt_winters_forecast,
    holts_forecast,
    moving_average_forecast,
    naive_forecast,
    prophet_forecast,
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


# ---------------------------------------------------------------------------
# Brand tokens (mirrors brand/tokens.json — kept inline so the app has zero
# file-read dependencies beyond the logo).
# ---------------------------------------------------------------------------

COLOR_PRIMARY = "#1B4F8A"
COLOR_PRIMARY_LIGHT = "#2D6DB5"
COLOR_PRIMARY_DARK = "#0F3060"
COLOR_ACCENT = "#E85D26"
COLOR_SUCCESS = "#2E7D32"
COLOR_WARNING = "#F57C00"
COLOR_DANGER = "#C62828"
COLOR_NEUTRAL_100 = "#FFFFFF"
COLOR_NEUTRAL_200 = "#F5F6F8"
COLOR_NEUTRAL_300 = "#E8EAED"
COLOR_NEUTRAL_600 = "#5F6368"
COLOR_NEUTRAL_900 = "#1A1A2E"

CHART_COLORS = ["#1B4F8A", "#E85D26", "#2E7D32", "#7B1FA2", "#F57C00"]

LEVEL_COLORS = {
    "critical": COLOR_DANGER,
    "high": "#E67E22",
    "medium": "#F1C40F",
    "low": COLOR_SUCCESS,
    "good": COLOR_SUCCESS,
    "fair": "#F1C40F",
    "poor": COLOR_DANGER,
    "moderate": "#F1C40F",
}

GITHUB_URL = "https://github.com/ayushtripathi955/main-street-marketplace-toolkit"
WEBSITE_URL = "https://mainstreetmarketplace.org"

# Live Medium URLs for the five-part practitioner series. ARTICLES_URL
# points at article 1 as the series entry; the per-article links are
# used by the Home page list and the website article cards.
MEDIUM_ARTICLES = [
    ("A Practitioner's Guide to Quality-Aware Marketplace Ranking",
     "https://medium.com/@ayush.tripathi955/article-1-a-practitioners-guide-to-quality-aware-marketplace-ranking-cc10032bba84"),
    ("Stop Running Out of Stock",
     "https://medium.com/@ayush.tripathi955/article-2-stop-running-out-of-stock-e48d79baca83"),
    ("Short-Horizon Demand Forecasting for SMBs",
     "https://medium.com/@ayush.tripathi955/article-3-short-horizon-demand-forecasting-for-smbs-05fd069d8660"),
    ("Forecasts Fail Quietly",
     "https://medium.com/@ayush.tripathi955/article-4-forecasts-fail-quietly-359cef729bdc"),
    ("Marketplace Concentration Risk",
     "https://medium.com/@ayush.tripathi955/article-5-marketplace-concentration-risk-1d7af1908550"),
]
ARTICLES_URL = MEDIUM_ARTICLES[0][1]

PAGE_HOME = "Home"
PAGE_INTEGRITY = "Marketplace Integrity"
PAGE_RESILIENCE = "Supply Resilience"
PAGE_FORECAST = "Demand Forecasting"

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

LOGO_PATH = Path(__file__).resolve().parent.parent / "brand" / "logo.svg"


# ---------------------------------------------------------------------------
# Session-state helpers — wire one Home-page upload to every module page.
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Make sure the session-state keys we read exist, with safe defaults."""
    if "data_mode" not in st.session_state:
        st.session_state["data_mode"] = "demo"
    if "seller_data" not in st.session_state:
        st.session_state["seller_data"] = {
            "integrity": None,   # 1-row DataFrame (validated)
            "inventory": None,   # long-format DataFrame (validated)
        }
    if "upload_errors" not in st.session_state:
        st.session_state["upload_errors"] = {
            "integrity": [],
            "inventory": [],
        }


def get_active_data(module: str):
    """Resolve the active data source for ``module``.

    Returns a tuple ``(value, source)`` where ``source`` is one of:

    * ``"uploaded"`` — the module has a validated upload to run on.
    * ``"demo"``     — the global mode is ``"demo"`` or the user opted
      into uploads but didn't provide this module's file. In either
      case the module's caller falls back to its existing synthetic
      pathway.

    ``value`` is the uploaded DataFrame when ``source == "uploaded"``,
    otherwise ``None``.
    """
    if st.session_state.get("data_mode") != "uploaded":
        return None, "demo"
    store = st.session_state.get("seller_data", {}) or {}
    if module == "integrity":
        df = store.get("integrity")
    elif module in ("resilience", "forecasting"):
        df = store.get("inventory")
    else:
        df = None
    if df is None or (hasattr(df, "empty") and df.empty):
        return None, "demo"
    return df, "uploaded"


def _data_source_badge(source: str, module_label: str) -> None:
    """Render the small "where this data came from" badge at page top."""
    if source == "uploaded":
        st.markdown(
            f'<div class="src-badge src-badge-uploaded">'
            "Showing results from your uploaded data."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="src-badge src-badge-demo">'
            f"Showing sample data — upload your own on the Home page to "
            f"run this on your {module_label}."
            "</div>",
            unsafe_allow_html=True,
        )


# Six baseline forecasting methods exposed on the auto_select page strip.
# Prophet is intentionally not in the strip — it's a holiday-spike specialist
# that only fires for SKUs with full-year history.
_METHODS_STRIP = [
    ("naive",          "Naive",           "Last value, repeated."),
    ("moving_average", "Moving Average",  "Trailing-window mean. Safe default for short history."),
    ("ses",            "SES",             "Smoothed level. Right when demand is steady."),
    ("holts",          "Holt's",          "Level + trend. Catches persistent drift."),
    ("holt_winters",   "Holt-Winters",    "Level + trend + 7-day cycle."),
    ("croston",        "Croston",         "Models size and incidence separately for lumpy demand."),
]


def _icon_svg(name: str, size: int = 24) -> str:
    """Inline SVG for a pillar icon. Single-color stroke; size in px (square).

    Three civic-tech-style icons in the brand palette:
      integrity   = shield + check (primary blue)
      resilience  = stacked boxes (success green)
      forecasting = bar chart with horizontal guard line (accent orange)
    """
    icons = {
        "integrity": (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            f'width="{size}" height="{size}" fill="none" '
            f'stroke="{COLOR_PRIMARY}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'<path d="M12 2 L20 5 V11 C20 16 16 20 12 22 C8 20 4 16 4 11 V5 Z"/>'
            f'<path d="M9 12 L11 14 L15 10"/>'
            f'</svg>'
        ),
        "resilience": (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            f'width="{size}" height="{size}" fill="none" '
            f'stroke="{COLOR_SUCCESS}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'<rect x="3" y="14" width="18" height="7" rx="1"/>'
            f'<rect x="6" y="8" width="12" height="6" rx="1"/>'
            f'<rect x="9" y="2" width="6" height="6" rx="1"/>'
            f'</svg>'
        ),
        "forecasting": (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            f'width="{size}" height="{size}" fill="none" '
            f'stroke="{COLOR_ACCENT}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'<line x1="2" y1="20" x2="22" y2="20"/>'
            f'<line x1="6" y1="20" x2="6" y2="15"/>'
            f'<line x1="11" y1="20" x2="11" y2="11"/>'
            f'<line x1="16" y1="20" x2="16" y2="13"/>'
            f'<line x1="21" y1="20" x2="21" y2="9"/>'
            f'<line x1="2" y1="6" x2="22" y2="6" stroke-dasharray="2 2"/>'
            f'</svg>'
        ),
    }
    return icons.get(name, "")


# ---------------------------------------------------------------------------
# Global CSS injection
# ---------------------------------------------------------------------------


def inject_css() -> None:
    """Inject the global stylesheet that gives the app its branded look."""
    css = f"""
    <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        }}
        .block-container {{
            max-width: 980px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {COLOR_NEUTRAL_200};
            border-right: 1px solid {COLOR_NEUTRAL_300};
        }}
        section[data-testid="stSidebar"] .sidebar-logo {{
            color: {COLOR_PRIMARY};
            margin-bottom: 0.5rem;
        }}
        section[data-testid="stSidebar"] .sidebar-title {{
            color: {COLOR_PRIMARY};
            font-weight: 700;
            font-size: 1.05rem;
            line-height: 1.3;
        }}
        section[data-testid="stSidebar"] .sidebar-subtitle {{
            color: {COLOR_NEUTRAL_600};
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }}
        section[data-testid="stSidebar"] .sidebar-footer {{
            color: {COLOR_NEUTRAL_600};
            font-size: 0.8rem;
            line-height: 1.5;
        }}
        section[data-testid="stSidebar"] .sidebar-footer a {{
            color: {COLOR_PRIMARY};
            text-decoration: none;
        }}
        [data-testid="stMetric"] {{
            background-color: {COLOR_NEUTRAL_100};
            border: 1px solid {COLOR_NEUTRAL_300};
            border-radius: 8px;
            padding: 1rem 1.25rem;
            box-shadow: 0 1px 4px rgba(27, 79, 138, 0.05);
            transition: box-shadow 0.15s ease;
        }}
        [data-testid="stMetric"]:hover {{
            box-shadow: 0 2px 12px rgba(27, 79, 138, 0.10);
        }}
        [data-testid="stAlert"] {{
            border-radius: 8px;
            border: 1px solid transparent;
        }}
        h1, h2, h3, h4 {{
            color: {COLOR_NEUTRAL_900};
            letter-spacing: -0.01em;
        }}
        .pillar-card {{
            background: {COLOR_NEUTRAL_100};
            border: 1px solid {COLOR_NEUTRAL_300};
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 12px rgba(27, 79, 138, 0.06);
            height: 100%;
        }}
        .pillar-card h3 {{
            color: {COLOR_PRIMARY};
            margin-top: 0;
            margin-bottom: 0.75rem;
            font-size: 1.15rem;
        }}
        .pillar-card .pillar-icon {{
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }}
        .risk-card {{
            border-radius: 10px;
            padding: 1.1rem 1rem;
            color: white;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        }}
        .risk-card .risk-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.92;
            margin-bottom: 0.25rem;
        }}
        .risk-card .risk-count {{
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }}
        .footer-band {{
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid {COLOR_NEUTRAL_300};
            color: {COLOR_NEUTRAL_600};
            font-size: 0.85rem;
            line-height: 1.6;
        }}
        .footer-band a {{
            color: {COLOR_PRIMARY};
            text-decoration: none;
        }}
        .footer-band a:hover {{ text-decoration: underline; }}
        .hero-eyebrow {{
            display: inline-block;
            background: {COLOR_NEUTRAL_200};
            color: {COLOR_PRIMARY};
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            margin-bottom: 1rem;
        }}
        .data-disclosure {{
            background: {COLOR_NEUTRAL_200};
            border-left: 4px solid {COLOR_PRIMARY};
            padding: 1rem 1.25rem;
            border-radius: 6px;
            color: {COLOR_NEUTRAL_900};
            font-size: 0.9rem;
            line-height: 1.6;
        }}
        .top-issue {{
            background: {COLOR_NEUTRAL_100};
            border: 1px solid {COLOR_NEUTRAL_300};
            border-left: 4px solid {COLOR_WARNING};
            border-radius: 6px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }}
        .top-issue .issue-headline {{
            font-weight: 600;
            color: {COLOR_NEUTRAL_900};
            margin-bottom: 0.35rem;
        }}
        .top-issue .issue-rec {{
            color: {COLOR_NEUTRAL_600};
            font-size: 0.95rem;
            line-height: 1.55;
        }}
        /* Section cards: style Streamlit's bordered containers per spec. */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {COLOR_NEUTRAL_100} !important;
            border: 1px solid {COLOR_NEUTRAL_300} !important;
            border-radius: 10px !important;
            padding: 24px !important;
            margin-bottom: 20px !important;
            box-shadow: 0 1px 4px rgba(27, 79, 138, 0.06) !important;
        }}
        /* Module-page intro card. */
        .intro-card {{
            background: #EEF4FB;
            border-left: 4px solid {COLOR_PRIMARY};
            padding: 16px;
            border-radius: 6px;
            color: {COLOR_NEUTRAL_900};
            font-size: 0.95rem;
            line-height: 1.55;
            margin-bottom: 1.5rem;
        }}
        /* Signal-breakdown HTML table with color-tinted rating cells. */
        .signal-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.93rem;
            margin: 0.25rem 0 1rem;
            color: {COLOR_NEUTRAL_900};
        }}
        .signal-table th,
        .signal-table td {{
            padding: 0.6rem 0.85rem;
            text-align: left;
            border-bottom: 1px solid {COLOR_NEUTRAL_300};
        }}
        .signal-table thead th {{
            background: {COLOR_NEUTRAL_200};
            color: {COLOR_NEUTRAL_600};
            font-weight: 600;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .signal-table tbody tr:last-child td {{ border-bottom: none; }}
        .signal-table td.rating-cell {{
            font-weight: 600;
            text-transform: capitalize;
        }}
        /* Pillar accent strip — colored top border on each Home pillar card. */
        .pillar-card.pillar-integrity   {{ border-top: 4px solid {COLOR_PRIMARY}; }}
        .pillar-card.pillar-resilience  {{ border-top: 4px solid {COLOR_SUCCESS}; }}
        .pillar-card.pillar-forecasting {{ border-top: 4px solid {COLOR_ACCENT}; }}
        /* Sidebar active-page indicator (uses :has() — degrades gracefully). */
        section[data-testid="stSidebar"] [role="radiogroup"] label {{
            position: relative;
            padding-left: 14px !important;
            transition: background-color 0.15s ease;
        }}
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
            background: rgba(27, 79, 138, 0.06);
            border-radius: 4px;
        }}
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 4px;
            height: 24px;
            background-color: {COLOR_PRIMARY};
            border-radius: 2px;
        }}
        /* Sidebar logo: cap rendered width and tighten vertical spacing. */
        section[data-testid="stSidebar"] .sidebar-logo {{
            margin: 0 0 0.35rem 0;
        }}
        section[data-testid="stSidebar"] .sidebar-logo svg {{
            max-width: 140px;
            height: auto;
            display: block;
        }}
        section[data-testid="stSidebar"] .sidebar-title {{
            margin-top: 0.1rem;
        }}
        section[data-testid="stSidebar"] .sidebar-subtitle {{
            margin-bottom: 0.65rem;
        }}
        section[data-testid="stSidebar"] .sidebar-about {{
            margin-top: 0.85rem;
            padding-top: 0.85rem;
            border-top: 1px solid {COLOR_NEUTRAL_300};
            color: {COLOR_NEUTRAL_600};
            font-size: 0.78rem;
            line-height: 1.55;
        }}
        /* Home page: full-width "Why this exists" hero card. */
        .why-card {{
            background: #EEF4FB;
            border-left: 4px solid {COLOR_PRIMARY};
            padding: 20px;
            border-radius: 8px;
            color: {COLOR_NEUTRAL_900};
            font-size: 1.0rem;
            line-height: 1.65;
            margin: 0.5rem 0 1.5rem;
            font-style: italic;
        }}
        .why-card strong {{ font-style: normal; color: {COLOR_PRIMARY}; }}
        /* Pillar SVG icons replace the emoji block. */
        .pillar-card .pillar-icon svg {{ display: block; }}
        /* Integrity page: top-issue / score-gap quick card next to suppression metric. */
        .quick-card {{
            background: {COLOR_NEUTRAL_100};
            border: 1px solid {COLOR_NEUTRAL_300};
            border-left: 4px solid {COLOR_ACCENT};
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-top: 0.5rem;
            box-shadow: 0 1px 4px rgba(27, 79, 138, 0.05);
        }}
        .quick-card .quick-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLOR_NEUTRAL_600};
            font-weight: 600;
        }}
        .quick-card .quick-value {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {COLOR_NEUTRAL_900};
            margin: 0.15rem 0;
        }}
        .quick-card .quick-sub {{
            font-size: 0.82rem;
            color: {COLOR_NEUTRAL_600};
        }}
        /* Forecasting page: horizontal "What we considered" methods strip. */
        .methods-strip {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
            margin: 0.5rem 0 1.5rem;
        }}
        .methods-strip .strip-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLOR_NEUTRAL_600};
            font-weight: 600;
            margin-right: 0.35rem;
        }}
        .method-chip {{
            background: {COLOR_NEUTRAL_200};
            color: {COLOR_NEUTRAL_600};
            border: 1px solid {COLOR_NEUTRAL_300};
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            cursor: help;
            transition: background 0.15s ease, color 0.15s ease;
        }}
        .method-chip:hover {{
            background: {COLOR_NEUTRAL_300};
            color: {COLOR_NEUTRAL_900};
        }}
        .method-chip-active {{
            background: rgba(27, 79, 138, 0.10);
            color: {COLOR_PRIMARY};
            border-color: {COLOR_PRIMARY};
            font-weight: 600;
        }}
        .method-chip-prophet {{
            background: rgba(232, 93, 38, 0.10);
            color: {COLOR_ACCENT};
            border-color: {COLOR_ACCENT};
            font-weight: 600;
        }}
        /* Per-page data-source badge (sample vs. uploaded). */
        .src-badge {{
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 1rem;
            letter-spacing: 0.02em;
        }}
        .src-badge-uploaded {{
            background: rgba(46, 125, 50, 0.10);
            color: {COLOR_SUCCESS};
            border: 1px solid rgba(46, 125, 50, 0.40);
        }}
        .src-badge-demo {{
            background: {COLOR_NEUTRAL_200};
            color: {COLOR_NEUTRAL_600};
            border: 1px solid {COLOR_NEUTRAL_300};
        }}
        .upload-status {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.5rem 0 0.85rem;
        }}
        .upload-status .pill {{
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            border: 1px solid {COLOR_NEUTRAL_300};
            background: {COLOR_NEUTRAL_100};
            color: {COLOR_NEUTRAL_600};
        }}
        .upload-status .pill-ok {{
            background: rgba(46, 125, 50, 0.10);
            color: {COLOR_SUCCESS};
            border-color: rgba(46, 125, 50, 0.40);
        }}
        .upload-status .pill-error {{
            background: rgba(198, 40, 40, 0.10);
            color: {COLOR_DANGER};
            border-color: rgba(198, 40, 40, 0.40);
        }}
        .upload-privacy {{
            font-size: 0.82rem;
            color: {COLOR_NEUTRAL_600};
            margin-top: 0.5rem;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def load_logo_svg(color: str = COLOR_PRIMARY) -> str:
    """Return the logo SVG with ``currentColor`` resolved to ``color``."""
    try:
        svg = LOGO_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""
    # Drop the XML prologue (Streamlit's markdown chokes on it occasionally).
    svg = svg.replace('<?xml version="1.0" encoding="UTF-8"?>', "").strip()
    return f'<div style="color:{color};">{svg}</div>'


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
# Plotly chart helpers
# ---------------------------------------------------------------------------


def _plotly_layout_defaults(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color=COLOR_NEUTRAL_900,
            size=13,
        ),
        title=dict(text=title, x=0, font=dict(size=16, color=COLOR_NEUTRAL_900)) if title else None,
        margin=dict(l=10, r=10, t=40 if title else 10, b=40),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, sans-serif"),
    )
    return fig


def gauge_chart(score: float) -> go.Figure:
    """Score gauge for the integrity overall score."""
    if score >= 75:
        bar_color = COLOR_SUCCESS
    elif score >= 50:
        bar_color = COLOR_WARNING
    else:
        bar_color = COLOR_DANGER
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 32, "color": COLOR_NEUTRAL_900}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLOR_NEUTRAL_600},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": COLOR_NEUTRAL_200,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(198, 40, 40, 0.10)"},
                    {"range": [50, 75], "color": "rgba(245, 124, 0, 0.10)"},
                    {"range": [75, 100], "color": "rgba(46, 125, 50, 0.10)"},
                ],
                "threshold": {
                    "line": {"color": COLOR_NEUTRAL_900, "width": 2},
                    "thickness": 0.75,
                    "value": 75,
                },
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLOR_NEUTRAL_900),
    )
    return fig


def signal_bar_chart(score_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=score_df["signal"],
            x=score_df["score"],
            orientation="h",
            marker=dict(color=[LEVEL_COLORS[r] for r in score_df["rating"]]),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}/100<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=50, line=dict(color=COLOR_NEUTRAL_600, dash="dash", width=1),
                  annotation_text="fair", annotation_position="top")
    fig.add_vline(x=80, line=dict(color=COLOR_NEUTRAL_600, dash="dot", width=1),
                  annotation_text="good", annotation_position="top")
    fig.update_xaxes(range=[0, 100], title="Score (0–100)",
                     gridcolor=COLOR_NEUTRAL_300, zerolinecolor=COLOR_NEUTRAL_300)
    fig.update_yaxes(title=None, gridcolor=COLOR_NEUTRAL_300)
    return _plotly_layout_defaults(fig, "Signal scores by rating")


def hhi_bar_chart(summary: pd.DataFrame) -> go.Figure:
    colors = [LEVEL_COLORS[l] for l in summary["concentration_level"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["category"],
            y=summary["hhi"],
            marker=dict(color=colors),
            customdata=np.stack([summary["concentration_level"]], axis=-1),
            hovertemplate=(
                "<b>%{x}</b><br>HHI: %{y:,.0f}<br>"
                "Level: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=1500, line=dict(color=COLOR_NEUTRAL_600, dash="dash", width=1),
                  annotation_text="DOJ moderate (1,500)", annotation_position="top right")
    fig.add_hline(y=2500, line=dict(color=COLOR_NEUTRAL_900, dash="dash", width=1),
                  annotation_text="DOJ high (2,500)", annotation_position="top right")
    fig.update_xaxes(title=None, tickangle=-30, gridcolor=COLOR_NEUTRAL_300)
    fig.update_yaxes(title="HHI", gridcolor=COLOR_NEUTRAL_300, zerolinecolor=COLOR_NEUTRAL_300)
    return _plotly_layout_defaults(fig, "Catalog concentration by category (HHI)")


def risk_distribution_chart(counts: pd.Series) -> go.Figure:
    order = ["critical", "high", "medium", "low"]
    counts = counts.reindex(order).fillna(0).astype(int)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker=dict(color=[LEVEL_COLORS[lvl] for lvl in counts.index]),
            text=counts.values,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y} SKUs<extra></extra>",
        )
    )
    fig.update_xaxes(title=None, gridcolor=COLOR_NEUTRAL_300)
    fig.update_yaxes(title="Number of SKUs", gridcolor=COLOR_NEUTRAL_300)
    return _plotly_layout_defaults(fig, "SKUs by stockout-risk level")


def forecast_chart(history_dates, history_series, fc_dates, forecast, lower, upper, method: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fc_dates, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_dates, y=lower, line=dict(width=0), fill="tonexty",
            fillcolor="rgba(45, 109, 181, 0.15)", name="95% PI", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history_dates, y=history_series,
            line=dict(color=COLOR_PRIMARY_DARK, width=2),
            name="Actuals (last 90d)",
            hovertemplate="%{x|%b %d}<br>%{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_dates, y=forecast,
            line=dict(color=COLOR_ACCENT, width=2.5, dash="dash"),
            name=f"Forecast ({method})",
            hovertemplate="%{x|%b %d}<br>%{y:.1f}<extra></extra>",
        )
    )
    if len(history_dates):
        fig.add_vline(x=history_dates[-1], line=dict(color=COLOR_NEUTRAL_600, dash="dot", width=1))
    fig.update_xaxes(title="date", gridcolor=COLOR_NEUTRAL_300, zerolinecolor=COLOR_NEUTRAL_300)
    fig.update_yaxes(title="units / day", gridcolor=COLOR_NEUTRAL_300, zerolinecolor=COLOR_NEUTRAL_300)
    return _plotly_layout_defaults(fig, "Actuals + forecast + 95% prediction interval")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def render_home() -> None:
    st.markdown(load_logo_svg(COLOR_PRIMARY), unsafe_allow_html=True)
    st.markdown('<div class="hero-eyebrow">Open Source · MIT Licensed · Free Forever</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<h1 style='margin-bottom:0.25rem;'>Main Street Marketplace Toolkit</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:1.15rem;color:{COLOR_NEUTRAL_600};margin-top:0;'>"
        "Open, free marketplace intelligence for U.S. small businesses.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # "Why this exists" hero card — full width, above the three pillars.
    st.markdown(
        '<div class="why-card">'
        "This toolkit translates marketplace analytics methods that "
        "Fortune-100 retail platforms run internally into open, "
        "non-proprietary frameworks. The same questions an enterprise "
        "data team answers for a billion-dollar catalog — "
        "<strong>Where am I exposed? What should I reorder? Can I "
        "trust this forecast?</strong> — answered for the SMB seller "
        "using a few hundred SKUs and a laptop."
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3, gap="medium")
    pillars = [
        ("integrity", "Marketplace Integrity", "pillar-integrity",
         "Score listing health and concentration risk across ten quality "
         "signals. Surface suppression risk before it costs you ranking."),
        ("resilience", "Supply Resilience", "pillar-resilience",
         "Diagnose stockout risk and compute safety stock and reorder "
         "points for every SKU — with the platform suppression tail "
         "factored in."),
        ("forecasting", "Forecasting & Guardrails", "pillar-forecasting",
         "Auto-select the right forecasting method per SKU and wrap the "
         "result in five protective guardrails that say when not to trust "
         "the forecast."),
    ]
    for col, (icon_name, title, accent_cls, body) in zip(cols, pillars):
        with col:
            st.markdown(
                f'<div class="pillar-card {accent_cls}">'
                f'<div class="pillar-icon">{_icon_svg(icon_name, 24)}</div>'
                f"<h3>{title}</h3><p>{body}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### Built for")
    st.markdown(
        "- **SBDC counselors** running small-business client clinics\n"
        "- **State and regional commerce program staff** monitoring "
        "marketplace exposure\n"
        "- **Niche marketplace operators** (regional, vertical, or co-op)\n"
        "- **SMB sellers** on Amazon, Walmart, Etsy, Shopify, eBay\n"
        "- **Policy researchers** studying platform dynamics for U.S. "
        "small businesses"
    )

    st.markdown("---")
    st.markdown(
        '<div class="data-disclosure">'
        "<strong>All data shown is synthetic</strong> — generated in-memory, "
        "never read from disk. Every calculation works identically on real "
        "seller-portal exports.</div>",
        unsafe_allow_html=True,
    )

    render_upload_section()

    st.markdown("")
    cta_cols = st.columns(2)
    with cta_cols[0]:
        st.link_button("View source on GitHub →", GITHUB_URL, width="stretch")
    with cta_cols[1]:
        st.link_button("Read the article series →", ARTICLES_URL, width="stretch")


def render_upload_section() -> None:
    """Home-page section: optionally upload one or two CSVs and have
    every module pick them up.
    """
    with st.expander("Use your own data (optional)", expanded=False):
        st.write(
            "Upload one or two CSV files here and every tool on the "
            "left will switch to your data automatically. Skip this and "
            "keep clicking around — every page works on the built-in "
            "sample data too."
        )
        st.markdown(
            '<div class="upload-privacy">'
            "Your file is processed in your browser session and is not "
            "stored, saved to disk, or sent anywhere."
            "</div>",
            unsafe_allow_html=True,
        )

        tab_int, tab_inv = st.tabs(
            ["Listing-quality data (Integrity)",
             "Inventory + sales history (Resilience & Forecasting)"]
        )

        with tab_int:
            st.caption(
                "One row with your seller-performance metrics. Drives "
                "the Marketplace Integrity scorecard."
            )
            st.download_button(
                "Download template",
                data=du.make_integrity_template_csv(),
                file_name="msmt_integrity_template.csv",
                mime="text/csv",
                key="dl_integrity",
            )
            up_int = st.file_uploader(
                "Upload your listing-quality CSV",
                type=["csv"],
                key="up_integrity",
                accept_multiple_files=False,
            )
            if up_int is not None:
                df, errors = du.parse_integrity_csv(up_int.getvalue())
                st.session_state["seller_data"]["integrity"] = df
                st.session_state["upload_errors"]["integrity"] = errors
                if errors:
                    for msg in errors:
                        st.error(msg)
                else:
                    st.success("Listing-quality file looks good.")

        with tab_inv:
            st.caption(
                "Long format: one row per (SKU, day). The same upload "
                "feeds both the Supply Resilience heatmap and the Demand "
                "Forecasting page — you only upload it once."
            )
            st.download_button(
                "Download template",
                data=du.make_inventory_template_csv(),
                file_name="msmt_inventory_template.csv",
                mime="text/csv",
                key="dl_inventory",
            )
            up_inv = st.file_uploader(
                "Upload your inventory + sales CSV",
                type=["csv"],
                key="up_inventory",
                accept_multiple_files=False,
            )
            if up_inv is not None:
                df, errors = du.parse_inventory_csv(up_inv.getvalue())
                st.session_state["seller_data"]["inventory"] = df
                st.session_state["upload_errors"]["inventory"] = errors
                if errors:
                    for msg in errors:
                        st.error(msg)
                else:
                    st.success(
                        f"Inventory + sales file looks good "
                        f"({df['sku_id'].nunique()} SKUs, {len(df)} rows)."
                    )

        # Status pills.
        store = st.session_state.get("seller_data", {})
        errs = st.session_state.get("upload_errors", {})
        pills = []
        for key, label in [("integrity", "Listing quality"),
                           ("inventory", "Inventory + sales")]:
            if errs.get(key):
                pills.append(f'<span class="pill pill-error">✗ {label}</span>')
            elif store.get(key) is not None:
                pills.append(f'<span class="pill pill-ok">✓ {label}</span>')
            else:
                pills.append(f'<span class="pill">○ {label}</span>')
        st.markdown(
            f'<div class="upload-status">{"".join(pills)}</div>',
            unsafe_allow_html=True,
        )

        # Mode toggle.
        any_valid = any(
            store.get(k) is not None and not errs.get(k)
            for k in ("integrity", "inventory")
        )
        current = st.session_state.get("data_mode", "demo")
        cols = st.columns([1, 1])
        with cols[0]:
            if current == "demo":
                if st.button(
                    "Use my uploaded data",
                    type="primary",
                    disabled=not any_valid,
                    key="btn_use_uploaded",
                    help=None if any_valid
                    else "Upload at least one valid CSV above to switch.",
                ):
                    st.session_state["data_mode"] = "uploaded"
                    st.rerun()
            else:
                if st.button(
                    "Switch back to sample data",
                    key="btn_use_demo",
                ):
                    st.session_state["data_mode"] = "demo"
                    st.rerun()
        with cols[1]:
            mode_label = (
                "Currently using: your uploaded data"
                if current == "uploaded"
                else "Currently using: built-in sample data"
            )
            st.caption(mode_label)


_RATING_BG = {
    "good": "#E8F5E9",
    "fair": "#FFF8E1",
    "poor": "#FFEBEE",
}


def _render_signal_table_html(score_df: pd.DataFrame) -> None:
    """Render the signal-breakdown table as HTML with a tinted rating cell."""
    rows = []
    for _, row in score_df.iterrows():
        bg = _RATING_BG.get(str(row["rating"]), "transparent")
        rows.append(
            "<tr>"
            f"<td>{row['signal']}</td>"
            f"<td>{row['value']:.3f}</td>"
            f"<td>{row['score']:.1f}</td>"
            f"<td class='rating-cell' style='background-color:{bg};'>"
            f"{row['rating']}</td>"
            f"<td>{row['weight']:.2f}</td>"
            "</tr>"
        )
    table_html = (
        "<table class='signal-table'>"
        "<thead><tr>"
        "<th>Signal</th><th>Value</th><th>Score</th>"
        "<th>Rating</th><th>Weight</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_integrity() -> None:
    st.title("Marketplace Integrity")
    st.write(
        "Score a small seller against ten signals across fulfillment, "
        "post-purchase quality, and listing content. The scorecard's "
        "weights and benchmarks are practitioner estimates from publicly "
        "available platform guidance — not platform-disclosed algorithmic "
        "weights."
    )

    st.markdown(
        '<div class="intro-card">This page scores a marketplace seller '
        "against 10 quality signals across fulfillment, post-purchase, "
        "and content. It shows you exactly where ranking is at risk — "
        "and what to fix first.</div>",
        unsafe_allow_html=True,
    )

    uploaded_df, source = get_active_data("integrity")
    _data_source_badge(source, module_label="listing-quality data")

    if source == "uploaded" and uploaded_df is not None:
        # Uploaded path: pull the 10-signal row straight out.
        metrics = du.integrity_dataframe_to_metrics(uploaded_df)
        scorecard = compute_scorecard(metrics)
        st.caption("Score computed from your uploaded CSV.")
    else:
        if (st.session_state.get("data_mode") == "uploaded"
                and st.session_state["seller_data"].get("integrity") is None):
            st.info(
                "You opted into uploaded data but haven't given me a "
                "listing-quality CSV yet — running this page on sample "
                "data. Add the file on the Home page to switch."
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
            st.caption("Defaults are at each signal's 'good' benchmark.")
            metrics: Dict[str, float] = {}
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

    # Headline: gauge + metric tiles
    head_cols = st.columns([2, 1])
    with head_cols[0]:
        st.plotly_chart(
            gauge_chart(scorecard["overall_score"]),
            config=PLOTLY_CONFIG,
            width="stretch",
        )
    with head_cols[1]:
        st.metric("Overall score", f"{scorecard['overall_score']:.1f} / 100")
        st.metric("Suppression risk", scorecard["suppression_risk"].upper())
        # Quick-glance summary of the single weakest signal.
        if scorecard["top_issues"]:
            top = scorecard["top_issues"][0]
            score_gap = max(0.0, 80.0 - float(top["score"]))
            top_name = str(top["name"]).replace("_", " ").title()
            st.markdown(
                '<div class="quick-card">'
                '<div class="quick-label">Top issue</div>'
                f'<div class="quick-value">{top_name}</div>'
                f'<div class="quick-sub">Score gap to good: '
                f'{score_gap:.1f} pts</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="quick-card" style="border-left-color:'
                f'{COLOR_SUCCESS};">'
                '<div class="quick-label">Top issue</div>'
                '<div class="quick-value">None — all signals at "good"</div>'
                '<div class="quick-sub">Recheck on your normal cadence.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    if scorecard["overall_score"] < 50:
        st.error(
            "**High suppression risk.** The seller's profile sits below "
            "the threshold most marketplaces use for Buy-Box and ranking "
            "eligibility. Address the top issues below before the next "
            "reorder cycle."
        )
    elif scorecard["overall_score"] < 75:
        st.warning(
            "**Medium suppression risk.** Listing-eligibility metrics are "
            "functional but vulnerable. Knock out one or two of the top "
            "issues to move into the safe zone."
        )

    # Signal breakdown table + chart (wrapped in a section card)
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

    with st.container(border=True):
        st.subheader("Signal breakdown")
        _render_signal_table_html(score_df)
        st.plotly_chart(
            signal_bar_chart(score_df),
            config=PLOTLY_CONFIG,
            width="stretch",
        )

    # Top issues (wrapped)
    with st.container(border=True):
        st.subheader("Top issues to fix")
        if scorecard["top_issues"]:
            for issue, rec in zip(scorecard["top_issues"], scorecard["recommendations"]):
                st.markdown(
                    f'<div class="top-issue">'
                    f'<div class="issue-headline">{issue["plain_english"]}</div>'
                    f'<div class="issue-rec">→ {rec}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No signals are flagged as 'fair' or 'poor'.")

    # Concentration analysis (wrapped)
    with st.container(border=True):
        st.subheader("Catalog concentration analysis")
        st.write(
            "How exposed is this seller to a single-SKU outage? The "
            "Herfindahl-Hirschman Index (HHI) measures how concentrated "
            "category volume is across SKUs. Thresholds shown are the U.S. "
            "Department of Justice merger-review thresholds."
        )

        # Concentration needs a `category` column. If the user uploaded
        # an inventory file with one, we use it; otherwise we fall back
        # to the synthetic catalog for *this section only*.
        inv_df, inv_src = get_active_data("resilience")
        if inv_src == "uploaded" and inv_df is not None and "category" in inv_df.columns:
            catalog = inv_df
            st.caption("Concentration computed from your uploaded inventory file.")
        else:
            if inv_src == "uploaded" and inv_df is not None:
                st.info(
                    "Your uploaded file doesn't include a 'category' "
                    "column, so this section is running on the built-in "
                    "synthetic catalog. Add a 'category' column to your "
                    "CSV to switch."
                )
            cc_cols = st.columns(2)
            with cc_cols[0]:
                n_skus_conc = st.slider("SKUs in synthetic catalog", 10, 100, 50, 5,
                                        key="conc_n_skus")
            with cc_cols[1]:
                seed_conc = st.number_input("Catalog seed", value=42, step=1, key="conc_seed")

            with st.spinner("Generating synthetic catalog…"):
                catalog = _cached_seller_data(int(n_skus_conc), 365, int(seed_conc))
        audit = concentration_audit(catalog)

        st.dataframe(audit["summary_df"].round(2), width="stretch", hide_index=True)
        st.info(audit["audit_narrative"])
        st.plotly_chart(hhi_bar_chart(audit["summary_df"]), config=PLOTLY_CONFIG, width="stretch")

        with st.expander("What does this mean for me?"):
            st.markdown(
                "- **Overall score below 75** is the cue to schedule a "
                "performance review with the seller. The top-issues list is "
                "the agenda.\n"
                "- **HHI above 2,500** in any category means the seller's "
                "volume in that category rests on too few SKUs. Ask whether "
                "the dominant SKU has a backup listing, an alternate "
                "supplier, or a second-source manufacturer.\n"
                "- The signals here cover what marketplaces *publish* about "
                "what they reward. They don't include any platform-private "
                "signals; if your marketplace exposes additional metrics in "
                "its seller portal, fold them into your own conversation."
            )


def render_resilience() -> None:
    st.title("Supply Resilience")
    st.write(
        "Classify each SKU's demand pattern, compute the appropriate "
        "safety stock and reorder point, and rank the catalog by current "
        "stockout exposure. The pipeline runs end-to-end on synthetic "
        "data; the same code works on a real seller-portal export."
    )

    st.markdown(
        '<div class="intro-card">This page classifies each SKU\'s demand '
        "pattern, computes the appropriate safety stock and reorder "
        "point, and ranks your catalog by current stockout exposure.</div>",
        unsafe_allow_html=True,
    )

    uploaded_df, source = get_active_data("resilience")
    _data_source_badge(source, module_label="inventory + sales history")

    if source == "uploaded" and uploaded_df is not None:
        cfg_cols = st.columns(2)
        with cfg_cols[0]:
            st.metric("SKUs in your upload", int(uploaded_df["sku_id"].nunique()))
        with cfg_cols[1]:
            service_level = st.selectbox(
                "Service level",
                [0.90, 0.95, 0.97, 0.98, 0.99],
                index=1,
                key="res_sl_up",
            )
        catalog = uploaded_df
    else:
        if (st.session_state.get("data_mode") == "uploaded"
                and st.session_state["seller_data"].get("inventory") is None):
            st.info(
                "You opted into uploaded data but haven't given me an "
                "inventory + sales CSV yet — running this page on sample "
                "data. Add the file on the Home page to switch."
            )
        cfg_cols = st.columns(3)
        with cfg_cols[0]:
            n_skus = st.slider("Number of SKUs", 10, 100, 50, 5)
        with cfg_cols[1]:
            seed = st.number_input("Seed", value=42, step=1, key="res_seed")
        with cfg_cols[2]:
            service_level = st.selectbox(
                "Service level", [0.90, 0.95, 0.97, 0.98, 0.99], index=1
            )
        with st.spinner("Generating catalog…"):
            catalog = _cached_seller_data(int(n_skus), 365, int(seed))

    with st.spinner("Running resilience pipeline…"):
        try:
            heatmap = stockout_heatmap_data(catalog, service_level=float(service_level))
        except Exception as exc:  # defensive — keep the page alive on bad upload
            st.error(
                f"Couldn't run the resilience pipeline on this data: {exc}."
            )
            return

    counts = heatmap["risk_level"].value_counts().reindex(
        ["critical", "high", "medium", "low"]
    ).fillna(0).astype(int)

    # Color-coded risk metric cards
    risk_cols = st.columns(4, gap="small")
    cards = [
        ("CRITICAL", int(counts.get("critical", 0)), LEVEL_COLORS["critical"]),
        ("HIGH",     int(counts.get("high", 0)),     LEVEL_COLORS["high"]),
        ("MEDIUM",   int(counts.get("medium", 0)),   LEVEL_COLORS["medium"]),
        ("LOW",      int(counts.get("low", 0)),      LEVEL_COLORS["low"]),
    ]
    for col, (label, value, color) in zip(risk_cols, cards):
        with col:
            st.markdown(
                f'<div class="risk-card" style="background:{color};">'
                f'<div class="risk-label">{label}</div>'
                f'<div class="risk-count">{value}</div></div>',
                unsafe_allow_html=True,
            )

    # Stockout risk heatmap (wrapped)
    with st.container(border=True):
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
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "risk_score", min_value=0.0, max_value=1.0, format="%.2f",
                ),
            },
        )
        st.plotly_chart(
            risk_distribution_chart(counts),
            config=PLOTLY_CONFIG,
            width="stretch",
        )

    # Suppression cost calculator (wrapped)
    with st.container(border=True):
        st.subheader("Stockout cost calculator (with platform suppression)")
        st.write(
            "When a listing goes out of stock, the lost margin during the "
            "outage is only part of the cost — marketplace ranking algorithms "
            "tend to demote listings that go unavailable, and most listings "
            "take some weeks to climb back."
        )
        st.caption(
            "The default 3.0× suppression multiplier and 21-day recovery "
            "window are practitioner estimates from industry observation, "
            "not figures disclosed by any marketplace platform."
        )

        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            daily_profit = st.number_input("Daily profit when in stock ($)", value=120.0, step=10.0)
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

        # Total cost as the headline; direct + suppression as supporting metrics
        st.markdown(
            f'<div style="background:{COLOR_DANGER};color:white;border-radius:10px;'
            f'padding:1.5rem;text-align:center;margin-bottom:1rem;">'
            f'<div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;'
            f'opacity:0.92;">Total stockout cost</div>'
            f'<div style="font-size:2.5rem;font-weight:700;line-height:1;">'
            f'${cost["total_cost"]:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
        sub_cols = st.columns(2)
        with sub_cols[0]:
            st.metric("Direct cost (lost sales)", f"${cost['direct_cost']:,.0f}")
        with sub_cols[1]:
            st.metric("Suppression tail", f"${cost['suppression_cost']:,.0f}",
                      delta=f"{(cost['suppression_cost']/max(cost['direct_cost'],1)*100):.0f}% of direct"
                      if cost["direct_cost"] > 0 else None)

        with st.expander("Why does suppression matter?"):
            st.markdown(
                "Marketplace ranking algorithms reward *consistent* availability. "
                "When a listing goes out of stock, the algorithm sees:\n\n"
                "1. **Outage window** — the seller earns $0/day. This is the "
                "*direct cost* shown above.\n"
                "2. **Restock + recovery window** — the listing is back in stock, "
                "but its ranking is depressed. The seller still earns less than "
                "their normal day-rate for several weeks. This is the "
                "*suppression tail*.\n\n"
                "For most SMB sellers, the suppression tail is **larger than the "
                "direct cost** — and is the reason a one-week stockout can "
                "wipe out a quarter of profit."
            )


def _force_method(sub: pd.DataFrame, pattern: str, horizon: int) -> Dict[str, Any]:
    """Run the auto-selected method for ``pattern`` on a single SKU."""
    series = sub.sort_values("date")["units_sold"].astype(float).to_numpy()
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


_METHOD_RATIONALE = {
    "naive": "Last value, repeated. The simplest baseline.",
    "seasonal_naive": "Same day next week. Strong on weekly cycles.",
    "moving_average": "Trailing-window mean. Safest when there's "
                       "little history or the SKU is brand-new.",
    "ses": "Smoothed level only. Right when demand is steady and "
            "neither trending nor seasonal.",
    "holts": "Smoothed level plus trend. Use when there's a "
              "persistent upward or downward drift.",
    "holt_winters": "Level + trend + seasonal cycle. Right when "
                     "demand swings predictably each week.",
    "croston": "Models size and incidence separately. Right for "
                "intermittent demand with many zero days.",
    "prophet": "Bayesian additive model with holiday effects. Right "
                "for SKUs with full-year history and clear holiday "
                "spikes.",
}


def render_forecasting() -> None:
    st.title("Demand Forecasting & Guardrails")
    st.write(
        "Pick a demand pattern, generate a representative SKU, and run "
        "the auto-selected forecasting method. The toolkit picks the "
        "method based on the demand archetype — you don't have to know "
        "what an exponential smoother is to use it."
    )

    st.markdown(
        '<div class="intro-card">This page picks the right forecasting '
        "method for your SKU automatically, then wraps the forecast in "
        "five guardrails that tell you when not to trust the prediction."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_df, source = get_active_data("forecasting")
    _data_source_badge(source, module_label="inventory + sales history")

    if source == "uploaded" and uploaded_df is not None:
        all_skus = sorted(uploaded_df["sku_id"].unique().tolist())
        sel_cols = st.columns(2)
        with sel_cols[0]:
            sku_choice = st.selectbox("Pick a SKU", all_skus, key="fc_sku_up")
        with sel_cols[1]:
            horizon = st.slider("Forecast horizon (days)", 7, 60, 28, 1,
                                key="fc_horizon_up")
        sub = uploaded_df[uploaded_df["sku_id"] == sku_choice]
        with st.spinner("Running forecast on your data…"):
            try:
                result = run_forecast(sub, horizon=int(horizon))
                # Wrap the run_forecast dict in the extras the page renderer
                # below expects (series + dates + the pattern-derived rationale).
                result["series"] = (
                    sub.sort_values("date")["units_sold"].astype(float).to_numpy()
                )
                result["dates"] = pd.DatetimeIndex(sub.sort_values("date")["date"])
            except Exception as exc:
                st.error(f"Forecast failed on this SKU: {exc}")
                return
    else:
        if (st.session_state.get("data_mode") == "uploaded"
                and st.session_state["seller_data"].get("inventory") is None):
            st.info(
                "You opted into uploaded data but haven't given me an "
                "inventory + sales CSV yet — running this page on sample "
                "data. Add the file on the Home page to switch."
            )
        cfg_cols = st.columns(3)
        with cfg_cols[0]:
            pattern = st.selectbox("Demand pattern", list(PATTERNS), index=0)
        with cfg_cols[1]:
            seed = st.number_input("SKU seed", value=42, step=1, key="fc_seed")
        with cfg_cols[2]:
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

    # Method selection (wrapped in a section card)
    method = result["method_used"]
    rationale = _METHOD_RATIONALE.get(method, "")
    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:1.5rem;align-items:center;">'
            f'<div><div style="font-size:0.75rem;color:{COLOR_NEUTRAL_600};'
            f'text-transform:uppercase;letter-spacing:0.05em;">SKU</div>'
            f'<div style="font-weight:700;color:{COLOR_NEUTRAL_900};">{result["sku_id"]}</div></div>'
            f'<div><div style="font-size:0.75rem;color:{COLOR_NEUTRAL_600};'
            f'text-transform:uppercase;letter-spacing:0.05em;">Pattern</div>'
            f'<div style="font-weight:700;color:{COLOR_NEUTRAL_900};">{result["pattern"]}</div></div>'
            f'<div><div style="font-size:0.75rem;color:{COLOR_NEUTRAL_600};'
            f'text-transform:uppercase;letter-spacing:0.05em;">Method</div>'
            f'<div style="font-weight:700;color:{COLOR_PRIMARY};">{method}</div></div>'
            f'<div><div style="font-size:0.75rem;color:{COLOR_NEUTRAL_600};'
            f'text-transform:uppercase;letter-spacing:0.05em;">Horizon</div>'
            f'<div style="font-weight:700;color:{COLOR_NEUTRAL_900};">{horizon} days</div></div>'
            f'</div>'
            f'<div style="margin-top:0.85rem;color:{COLOR_NEUTRAL_600};font-size:0.92rem;">'
            f'<strong>Why this method?</strong> {rationale}</div>',
            unsafe_allow_html=True,
        )

    # "What we considered" — horizontal strip listing the six baseline
    # methods with a checkmark on the auto-selected one. Prophet is
    # surfaced separately as a holiday-spike specialist.
    chips_html = ['<div class="strip-label">What we considered</div>']
    for key, label, tip in _METHODS_STRIP:
        cls = "method-chip"
        if method == key:
            cls += " method-chip-active"
            label_text = f"✓ {label}"
        else:
            label_text = label
        chips_html.append(
            f'<div class="{cls}" title="{tip}">{label_text}</div>'
        )
    if method == "prophet":
        chips_html.append(
            '<div class="method-chip method-chip-prophet" '
            'title="Bayesian additive model with holiday effects. '
            'Selected for full-year holiday-spike SKUs.">'
            '✓ Prophet</div>'
        )
    else:
        chips_html.append(
            '<div class="method-chip" '
            'title="Bayesian additive model with holiday effects. '
            'Selected for full-year holiday-spike SKUs.">'
            '+ Prophet</div>'
        )
    st.markdown(
        f'<div class="methods-strip">{"".join(chips_html)}</div>',
        unsafe_allow_html=True,
    )

    # Forecast chart (wrapped)
    history_window = 90
    h_dates = result["dates"][-history_window:]
    h_series = result["series"][-history_window:]
    with st.container(border=True):
        st.plotly_chart(
            forecast_chart(
                h_dates, h_series, result["horizon_dates"],
                result["forecast"], result["lower_95"], result["upper_95"],
                method,
            ),
            config=PLOTLY_CONFIG,
            width="stretch",
        )

    # Guardrails (wrapped)
    forecast_dict = {
        "sku_id": result["sku_id"],
        "pattern": result["pattern"],
        "method_used": result["method_used"],
        "forecast": result["forecast"],
        "lower_95": result["lower_95"],
        "upper_95": result["upper_95"],
        "horizon_dates": result["horizon_dates"],
    }
    proposed = float(result["series"][-30:].mean()) * 30
    try:
        report = run_guardrails(sub, forecast_dict, proposed_order_qty=proposed)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Guardrails failed: {exc}")
        return

    with st.container(border=True):
        st.subheader("Guardrails")
        st.write(
            "Five sanity checks that wrap the raw forecast before it drives "
            "a reorder. A 'fired' guardrail is the cue to slow down and "
            "review the recommendation."
        )

        overall = report["overall_recommendation"]
        n_fired = sum(
            1 for k, v in report["guardrails"].items()
            if k != "degradation" and v.get("fired")
        )
        if n_fired == 0:
            st.success(f"**{overall}**")
        elif n_fired == 1:
            st.warning(f"**{overall}**")
        else:
            st.error(f"**{overall}**")

        rows = []
        for name, g in report["guardrails"].items():
            if name == "degradation":
                status = f"L{g.get('fallback_level')} ({g.get('method_name')})"
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

        with st.expander("What do I do when a guardrail fires?"):
            st.markdown(
                "| Guardrail | If fired | Recommended action |\n"
                "|---|---|---|\n"
                "| **drift** | Forecast has been biased the same way for 3+ weeks | "
                "Refit the model; check whether price, packaging, or competition changed |\n"
                "| **confidence** | PI band is wider than the forecast level | "
                "Pad safety stock instead of ordering to the point forecast |\n"
                "| **regime** | Recent actuals are outside the forecast band | "
                "Hold off on auto-reorders until you have 1–2 weeks of new-normal data |\n"
                "| **cap** | Proposed order is more than 3× recent run-rate | "
                "Manual review of the reorder math — expected this big a jump? |\n"
                "| **degradation** | Primary forecast unavailable | "
                "Fall back to the trailing 30-day mean (level 2); revisit next week |"
            )


# ---------------------------------------------------------------------------
# Sidebar + footer
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f'<div class="sidebar-logo">{load_logo_svg(COLOR_PRIMARY)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-title">Main Street Marketplace Toolkit</div>'
            '<div class="sidebar-subtitle">Open marketplace intelligence for U.S. small businesses.</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Module",
            [PAGE_HOME, PAGE_INTEGRITY, PAGE_RESILIENCE, PAGE_FORECAST],
            index=0,
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(
            f'<div class="sidebar-footer">'
            f"All data is synthetic. MIT Licensed.<br/><br/>"
            f'<a href="{GITHUB_URL}" target="_blank">GitHub</a> · '
            f'<a href="{ARTICLES_URL}" target="_blank">Articles</a> · '
            f'<a href="{WEBSITE_URL}" target="_blank">mainstreetmarketplace.org</a>'
            f"<br/><br/>Built by Ayush Tripathi, San Francisco."
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-about">'
            "v0.5.0 · Built April 2026<br/>"
            "A non-commercial public-goods project."
            '</div>',
            unsafe_allow_html=True,
        )
    return page


def render_footer() -> None:
    st.markdown(
        f'<div class="footer-band">'
        f"<strong>Main Street Marketplace Toolkit</strong>  ·  MIT Licensed  ·  Free forever<br/>"
        f"Built by Ayush Tripathi · Data analytics and marketplace strategy practitioner, San Francisco<br/>"
        f'<a href="{GITHUB_URL}" target="_blank">GitHub</a>  ·  '
        f'<a href="{ARTICLES_URL}" target="_blank">Articles</a>  ·  '
        f'<a href="{WEBSITE_URL}" target="_blank">mainstreetmarketplace.org</a>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Main Street Marketplace Toolkit",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()
    inject_css()
    page = render_sidebar()

    if page == PAGE_HOME:
        render_home()
    elif page == PAGE_INTEGRITY:
        render_integrity()
    elif page == PAGE_RESILIENCE:
        render_resilience()
    elif page == PAGE_FORECAST:
        render_forecasting()

    render_footer()


if __name__ == "__main__":
    main()
