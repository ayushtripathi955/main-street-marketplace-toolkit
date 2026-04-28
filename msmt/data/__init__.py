"""Data utilities for the Main Street Marketplace Toolkit.

This subpackage exposes synthetic data generators used throughout the toolkit
for examples, notebooks, and tests. All data produced here is **synthetic**:
it does not originate from, and is not derived from, any proprietary or
real-world seller dataset.
"""

from msmt.data.synthetic import (
    PATTERNS,
    generate_holiday_spike,
    generate_intermittent,
    generate_new_sku,
    generate_seller_data,
    generate_smooth,
    generate_weekly_seasonal,
)

__all__ = [
    "PATTERNS",
    "generate_seller_data",
    "generate_smooth",
    "generate_weekly_seasonal",
    "generate_holiday_spike",
    "generate_intermittent",
    "generate_new_sku",
]
