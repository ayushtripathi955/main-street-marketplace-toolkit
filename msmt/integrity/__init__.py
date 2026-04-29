"""Marketplace integrity — pillar 1 of the toolkit.

This subpackage answers two operational questions a counselor or a
seller most often asks first:

1. **Is this seller's listing/operational profile in good shape?**
   :func:`compute_scorecard` runs ten signals across fulfillment,
   post-purchase quality, and content completeness and produces an
   overall score, a suppression-risk label, and a short list of
   prioritized recommendations.

2. **How concentrated is this seller's catalog?**
   :func:`hhi_per_category` and :func:`concentration_audit` measure
   per-category Herfindahl-Hirschman Index — the same statistic the
   U.S. Department of Justice uses for merger review — to surface
   single-SKU points of failure.

The signals, weights, and benchmarks live in
:mod:`msmt.integrity.signals` and are documented as practitioner
estimates from publicly available platform guidance, not as
platform-disclosed algorithmic weights.
"""

from msmt.integrity.concentration import (
    concentration_audit,
    hhi_per_category,
)
from msmt.integrity.scorecard import (
    compute_scorecard,
    scorecard_for_synthetic_seller,
)
from msmt.integrity.signals import (
    SIGNALS,
    SIGNALS_BY_NAME,
    Signal,
    recommendation_for,
)

__all__ = [
    "Signal",
    "SIGNALS",
    "SIGNALS_BY_NAME",
    "recommendation_for",
    "compute_scorecard",
    "scorecard_for_synthetic_seller",
    "hhi_per_category",
    "concentration_audit",
]
