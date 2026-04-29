"""Marketplace integrity signals.

This module defines the ten signals the toolkit uses to score a small
seller's marketplace integrity — the operational and content-quality
metrics that, in aggregate, predict whether a marketplace's ranking
and Buy-Box logic will reward or penalize the seller.

The signals are organized in three groups, each contributing a fixed
share of the overall scorecard:

* **Fulfillment** (40%) — did the order ship on time, with valid
  tracking, and without late dispatch or pre-fulfillment cancels?
* **Post-purchase** (35%) — once the order arrived, did the buyer
  keep it, and did the experience meet expectations?
* **Content** (25%) — does the listing itself look like a serious
  product?

.. important::
   The weights and thresholds defined here are **practitioner
   estimates based on publicly available platform guidance** (seller
   help centers, public Buy-Box documentation, and the Walmart Listing
   Quality Dashboard published scoring guidance). They are *not*
   platform-disclosed algorithmic weights and they are not derived
   from any internal marketplace data. A seller with their own
   historical correlation between these signals and their actual
   suppression / Buy-Box performance should override these defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class Signal:
    """A single marketplace-integrity signal.

    Attributes
    ----------
    name : str
        Machine-friendly identifier used as a dict key in
        :func:`msmt.integrity.scorecard.compute_scorecard`.
    description : str
        One-line, plain-English description suitable for showing to
        an SBDC counselor or a seller.
    weight : float
        Share of the overall scorecard, in ``[0, 1]``. Weights across
        all signals sum to ``1.0``.
    benchmark_good : float
        The metric value at which the signal scores ``100`` on the
        toolkit's 0–100 scale.
    benchmark_poor : float
        The metric value at which the signal scores ``0``.

    Notes
    -----
    The direction (higher-is-better vs lower-is-better) is encoded by
    the numerical relationship between ``benchmark_good`` and
    ``benchmark_poor``. If ``benchmark_good > benchmark_poor`` the
    signal is higher-is-better (e.g. on-time shipment rate); if
    ``benchmark_good < benchmark_poor`` it is lower-is-better (e.g.
    return rate). The scoring function in
    :mod:`msmt.integrity.scorecard` uses a single linear-interpolation
    formula that handles both cases.
    """

    name: str
    description: str
    weight: float
    benchmark_good: float
    benchmark_poor: float

    @property
    def lower_is_better(self) -> bool:
        """True when a smaller metric value is better."""
        return self.benchmark_good < self.benchmark_poor


SIGNALS: Tuple[Signal, ...] = (
    # --- Fulfillment (40%) ---------------------------------------------
    Signal(
        name="on_time_shipment_rate",
        description=(
            "Fraction of orders that ship by their promised dispatch "
            "deadline. Direct input to Buy-Box eligibility on most "
            "marketplaces."
        ),
        weight=0.15,
        benchmark_good=0.97,
        benchmark_poor=0.90,
    ),
    Signal(
        name="valid_tracking_rate",
        description=(
            "Fraction of shipped orders carrying a valid, scannable "
            "tracking number. Missing or invalid tracking is a fast path "
            "to seller-performance flags."
        ),
        weight=0.10,
        benchmark_good=0.95,
        benchmark_poor=0.90,
    ),
    Signal(
        name="pre_fulfillment_cancel_rate",
        description=(
            "Fraction of orders the seller cancels before shipment "
            "(usually because of stockouts). Lower is better."
        ),
        weight=0.10,
        benchmark_good=0.02,
        benchmark_poor=0.05,
    ),
    Signal(
        name="late_dispatch_rate",
        description=(
            "Fraction of orders dispatched after the promised handling "
            "window. Lower is better."
        ),
        weight=0.05,
        benchmark_good=0.02,
        benchmark_poor=0.05,
    ),
    # --- Post-purchase (35%) -------------------------------------------
    Signal(
        name="return_rate",
        description=(
            "Fraction of shipped orders the buyer returns. High return "
            "rates are read as listing-vs-reality mismatch by ranking "
            "algorithms. Lower is better."
        ),
        weight=0.15,
        benchmark_good=0.05,
        benchmark_poor=0.15,
    ),
    Signal(
        name="order_defect_rate",
        description=(
            "Fraction of orders with a buyer-reported defect (negative "
            "feedback, A-to-Z claim, chargeback). The headline "
            "marketplace seller-health metric. Lower is better."
        ),
        weight=0.10,
        benchmark_good=0.01,
        benchmark_poor=0.03,
    ),
    Signal(
        name="customer_feedback_score",
        description=(
            "Average buyer rating across product and seller reviews, "
            "0–5 scale."
        ),
        weight=0.10,
        benchmark_good=4.7,
        benchmark_poor=4.0,
    ),
    # --- Content (25%) -------------------------------------------------
    Signal(
        name="listing_quality_score",
        description=(
            "Composite listing-completeness score on the 0–100 scale "
            "used by the Walmart Listing Quality Dashboard and similar "
            "marketplace tools."
        ),
        weight=0.15,
        benchmark_good=80.0,
        benchmark_poor=60.0,
    ),
    Signal(
        name="image_count",
        description=(
            "Number of distinct product images on the listing. More "
            "images correlate with higher conversion."
        ),
        weight=0.05,
        benchmark_good=6.0,
        benchmark_poor=3.0,
    ),
    Signal(
        name="keyword_coverage",
        description=(
            "Fraction of category-relevant keywords present somewhere "
            "in the title, bullets, or description."
        ),
        weight=0.05,
        benchmark_good=0.80,
        benchmark_poor=0.50,
    ),
)


SIGNALS_BY_NAME: Mapping[str, Signal] = {s.name: s for s in SIGNALS}


_SIGNAL_RECOMMENDATIONS: Mapping[str, str] = {
    "on_time_shipment_rate": (
        "Your on-time shipment rate is below platform thresholds. Late "
        "shipments are a direct input to Buy Box eligibility. Prioritize "
        "carrier reliability or switch fulfillment methods."
    ),
    "valid_tracking_rate": (
        "Too many shipped orders are missing valid tracking. Set up "
        "automated tracking upload from your carrier and run a weekly "
        "exception report on shipments without a scannable number."
    ),
    "pre_fulfillment_cancel_rate": (
        "You are cancelling too many orders before shipment, usually a "
        "sign of stockouts the catalog didn't reflect. Plug "
        "msmt.resilience reorder-point output into your reorder workflow "
        "to cut these."
    ),
    "late_dispatch_rate": (
        "Orders are dispatching late. Review handling-time settings on "
        "the listing and your warehouse cut-off times — most late "
        "dispatches are a same-day-cutoff calendar mistake, not a "
        "throughput problem."
    ),
    "return_rate": (
        "Your return rate is elevated. High returns signal listing-"
        "reality mismatch to the algorithm. Audit your top-returned "
        "SKUs for image accuracy and description completeness."
    ),
    "order_defect_rate": (
        "Order defect rate is above the typical marketplace threshold. "
        "This is the single fastest way to lose Buy Box. Triage the "
        "open defects and address each one before the next reorder "
        "cycle."
    ),
    "customer_feedback_score": (
        "Average buyer rating is dragging the listing's visibility. "
        "Ask satisfied buyers for reviews via the marketplace's "
        "permitted review-request flow and respond promptly to "
        "negative feedback."
    ),
    "listing_quality_score": (
        "Listing-quality score is below the 'good' threshold. Run "
        "the marketplace's listing-quality report and complete the "
        "missing attributes — this typically lifts the score within "
        "24 hours."
    ),
    "image_count": (
        "Too few product images on the listing. Add at least a hero "
        "shot, two angles, a scale reference, and a packaging or "
        "in-use shot — six is the practical floor."
    ),
    "keyword_coverage": (
        "Listing is missing category-relevant keywords. Pull the "
        "category's top search terms and add the missing ones to the "
        "title, bullets, and back-end search fields."
    ),
}


def recommendation_for(signal_name: str) -> str:
    """Return the canned plain-English recommendation for a signal."""
    return _SIGNAL_RECOMMENDATIONS.get(
        signal_name,
        "Review this signal against the platform's seller help center "
        "and address the gap.",
    )


def total_weight() -> float:
    """Return the sum of weights across all signals (sanity check helper)."""
    return float(sum(s.weight for s in SIGNALS))


# Sanity check at import time so a future edit can't silently break the
# weights-sum-to-one invariant the scorecard depends on.
assert abs(total_weight() - 1.0) < 1e-9, (
    f"Signal weights must sum to 1.0; got {total_weight()}"
)
