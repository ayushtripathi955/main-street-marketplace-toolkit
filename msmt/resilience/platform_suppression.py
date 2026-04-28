"""Suppression-adjusted stockout cost.

A stockout costs a marketplace seller more than the lost sales during
the out-of-stock window. Marketplace ranking and recommendation
algorithms tend to demote listings that go unavailable, and listings
generally take some weeks of post-restock sales to climb back to where
they were. Most small sellers underestimate this "suppression tail"
because the direct lost-sale figure is the only one that shows up in
their seller portal.

This module exposes a single function,
:func:`suppression_adjusted_stockout_cost`, that lets a seller put a
defensible (if rough) dollar figure on that tail so it can be weighed
against, for example, the cost of expedited shipping or carrying more
safety stock.

.. important::
   The default ``suppression_multiplier`` of 3.0× and ``recovery_days``
   of 21 are **practitioner estimates derived from industry observation
   of how marketplace listings tend to behave after stockouts**. They
   are not figures disclosed, confirmed, or published by any
   marketplace platform. A seller with their own historical
   stock-out / recovery data should override these defaults.
"""

from __future__ import annotations

from typing import Dict


def suppression_adjusted_stockout_cost(
    daily_profit: float,
    stockout_days: float,
    suppression_multiplier: float = 3.0,
    recovery_days: float = 21.0,
) -> Dict[str, float]:
    """Estimate the full cost of a stockout, including the ranking tail.

    The estimate has two parts:

    * **Direct cost** — profit lost on the days the listing was
      unavailable. Equal to ``daily_profit * stockout_days``.
    * **Suppression cost** — additional profit that fails to
      materialize during the recovery window because the listing's
      visibility is reduced after coming back in stock. Computed as::

          suppression_cost = daily_profit
                             * (suppression_multiplier - 1)
                             * recovery_days
                             * (stockout_days / 7)

      The ``stockout_days / 7`` factor scales the tail with the length
      of the outage in weeks, so a one-day outage incurs roughly a
      seventh of the tail of a seven-day outage.

    Parameters
    ----------
    daily_profit : float
        Per-day profit the listing earns when fully in stock and ranked
        normally. Use a recent average; a counselor will typically use
        gross margin per unit times average daily units sold. Must be
        non-negative.
    stockout_days : float
        Length of the out-of-stock window in days. Must be non-negative.
    suppression_multiplier : float, default 3.0
        Practitioner estimate of how much *more* profit a listing earns
        on a normal day than it earns during the post-stockout recovery
        window, expressed as a multiplier. ``1.0`` means the seller
        believes there is no suppression effect; ``3.0`` means a normal
        day earns roughly 3× a recovery-window day. **Not** a
        platform-disclosed figure.
    recovery_days : float, default 21.0
        Days from coming back in stock until the listing is back to
        normal performance. **Not** a platform-disclosed figure.

    Returns
    -------
    dict
        Keys:
        ``direct_cost`` (float),
        ``suppression_cost`` (float),
        ``total_cost`` (float),
        ``stockout_days`` (float),
        ``recovery_days`` (float),
        ``suppression_multiplier`` (float).
    """
    if daily_profit < 0:
        raise ValueError("daily_profit must be non-negative")
    if stockout_days < 0:
        raise ValueError("stockout_days must be non-negative")
    if suppression_multiplier < 1.0:
        raise ValueError(
            "suppression_multiplier must be >= 1.0; "
            "use 1.0 to model 'no suppression effect'"
        )
    if recovery_days < 0:
        raise ValueError("recovery_days must be non-negative")

    direct_cost = float(daily_profit * stockout_days)
    suppression_cost = float(
        daily_profit
        * (suppression_multiplier - 1.0)
        * recovery_days
        * (stockout_days / 7.0)
    )
    total = direct_cost + suppression_cost

    return {
        "direct_cost": direct_cost,
        "suppression_cost": suppression_cost,
        "total_cost": total,
        "stockout_days": float(stockout_days),
        "recovery_days": float(recovery_days),
        "suppression_multiplier": float(suppression_multiplier),
    }
