"""Supply resilience module — pillar 2 of the toolkit.

Helps a small marketplace seller (or the counselor advising one) answer
the practical question that drives most stockouts: *given the demand I
can see and the lead time I'm working with, when do I reorder, and how
much buffer do I need?*

The module is built around five plain-language steps:

1. **Classify** what kind of demand pattern a SKU has
   (:func:`msmt.resilience.classifier.classify_pattern`).
2. **Pick a safety-stock method** appropriate to that pattern
   (:func:`msmt.resilience.safety_stock.select_safety_stock_method`).
3. **Compute safety stock** under that method
   (:mod:`msmt.resilience.safety_stock`).
4. **Compute the reorder point**
   (:mod:`msmt.resilience.reorder_point`).
5. **Score stockout risk** for the current on-hand position and
   summarize across the catalog
   (:mod:`msmt.resilience.stockout_risk`).

A separate helper, :func:`msmt.resilience.platform_suppression
.suppression_adjusted_stockout_cost`, lets a seller reason about the
*ranking-tail* cost of a stockout — not just the lost sales during the
out-of-stock window, but the lift the listing tends to lose for some
weeks afterward as marketplace ranking algorithms re-learn it.

All thresholds in this module are deliberately conservative defaults a
non-specialist can tune. None of the numbers here are platform-disclosed
figures.
"""

from msmt.resilience.classifier import classify_pattern
from msmt.resilience.platform_suppression import (
    suppression_adjusted_stockout_cost,
)
from msmt.resilience.reorder_point import (
    reorder_point,
    reorder_point_for_sku,
)
from msmt.resilience.safety_stock import (
    SERVICE_LEVEL_Z,
    safety_stock_intermittent,
    safety_stock_kde,
    safety_stock_normal,
    select_safety_stock_method,
)
from msmt.resilience.stockout_risk import (
    stockout_heatmap_data,
    stockout_risk_score,
)

__all__ = [
    "classify_pattern",
    "SERVICE_LEVEL_Z",
    "safety_stock_normal",
    "safety_stock_kde",
    "safety_stock_intermittent",
    "select_safety_stock_method",
    "reorder_point",
    "reorder_point_for_sku",
    "stockout_risk_score",
    "stockout_heatmap_data",
    "suppression_adjusted_stockout_cost",
]
