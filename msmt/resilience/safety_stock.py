"""Safety-stock calculations for small-business inventory planning.

Safety stock is the buffer a seller carries on top of the units they
expect to sell during the lead-time window. The "right" amount depends
on how variable demand is, how variable the lead time is, and how
willing the seller is to risk a stockout. This module exposes three
ways of estimating that buffer, each suited to a different shape of
demand:

* :func:`safety_stock_normal` — the textbook formula. Best when daily
  demand is roughly bell-curve-shaped and the lead time wobbles a
  little. This is the default for *smooth* and *weekly_seasonal* SKUs.
* :func:`safety_stock_kde` — a non-parametric percentile of bootstrapped
  "demand during lead time" samples. Best when historical demand is
  skewed, fat-tailed, or has known holiday spikes that the normal
  formula will under-buffer.
* :func:`safety_stock_intermittent` — a simplified Croston-inspired
  estimate for SKUs with many zero-sale days. **Not** the full
  Croston/SBA method.

:func:`select_safety_stock_method` returns the recommended method name
for a given demand pattern so the rest of the pipeline can stay
generic.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

#: Z-table mapping cycle-service-level → standard-normal Z-score.
#:
#: These are the practitioner shorthand values used in most inventory
#: textbooks. They are not interpolated; pick the closest service level
#: from this set or pass your own Z directly.
SERVICE_LEVEL_Z: Mapping[float, float] = {
    0.90: 1.28,
    0.95: 1.65,
    0.97: 1.88,
    0.98: 2.05,
    0.99: 2.33,
}


def _z_for(service_level: float) -> float:
    """Resolve a service level to its standard-normal Z-score.

    Accepts an exact match from :data:`SERVICE_LEVEL_Z`; otherwise picks
    the closest tabulated value and warns implicitly via clamping.
    """
    if service_level in SERVICE_LEVEL_Z:
        return SERVICE_LEVEL_Z[service_level]
    if not 0.5 < service_level < 1.0:
        raise ValueError(
            f"service_level must be in (0.5, 1.0); got {service_level}"
        )
    closest = min(SERVICE_LEVEL_Z.keys(), key=lambda k: abs(k - service_level))
    return SERVICE_LEVEL_Z[closest]


def safety_stock_normal(
    demand_mean: float,
    demand_std: float,
    lead_time_mean: float,
    lead_time_std: float = 0.0,
    service_level: float = 0.95,
) -> float:
    """Textbook safety stock for normally-distributed demand.

    Implements the classic combined-variance formula::

        SS = Z * sqrt( LT_mean * sigma_D^2 + D_mean^2 * sigma_LT^2 )

    Use this when daily demand is roughly bell-curve-shaped and the
    lead time only wobbles a little. For lumpy or spiky SKUs, the
    formula systematically under-buffers because real demand has fat
    tails the normal distribution doesn't see.

    Parameters
    ----------
    demand_mean : float
        Average daily units sold over the planning horizon.
    demand_std : float
        Standard deviation of daily units sold.
    lead_time_mean : float
        Average lead time in days from reorder to receipt.
    lead_time_std : float, default 0.0
        Standard deviation of lead time in days. Set to 0 if the lead
        time is effectively fixed.
    service_level : float, default 0.95
        Cycle service level — the probability of *not* stocking out
        during a single replenishment cycle. Must be one of the values
        in :data:`SERVICE_LEVEL_Z`.

    Returns
    -------
    float
        Safety stock in units. Always non-negative.
    """
    if demand_mean < 0 or demand_std < 0 or lead_time_mean < 0 or lead_time_std < 0:
        raise ValueError("means and stds must be non-negative")

    z = _z_for(service_level)
    variance = lead_time_mean * (demand_std ** 2) + (demand_mean ** 2) * (
        lead_time_std ** 2
    )
    return float(z * np.sqrt(max(variance, 0.0)))


def safety_stock_kde(
    demand_during_leadtime_samples: Iterable[float],
    service_level: float = 0.95,
) -> float:
    """Non-parametric safety stock from bootstrapped demand-during-lead-time.

    Takes a sequence of historical (or simulated) "total units sold over
    a single lead-time window" observations and returns the
    ``service_level`` percentile minus the median. Equivalent in spirit
    to a kernel-density estimate without the kernel: the sample itself
    is the distribution.

    Use this when demand is skewed, has fat tails, or has a few large
    holiday spikes that the normal formula will under-buffer.

    Parameters
    ----------
    demand_during_leadtime_samples : iterable of float
        Historical observations of total units sold during a single
        lead-time window. A common way to build these is to roll a
        window of length ``lead_time_mean`` across daily sales and sum.
    service_level : float, default 0.95
        Cycle service level. Used as an upper percentile of the sample.

    Returns
    -------
    float
        Safety stock in units (the gap between the upper percentile and
        the median demand-during-lead-time). Always non-negative.
    """
    samples = np.asarray(list(demand_during_leadtime_samples), dtype=float)
    if samples.size == 0:
        raise ValueError("demand_during_leadtime_samples is empty")
    if not 0.5 < service_level < 1.0:
        raise ValueError(
            f"service_level must be in (0.5, 1.0); got {service_level}"
        )

    upper = float(np.percentile(samples, service_level * 100.0))
    median = float(np.percentile(samples, 50.0))
    return float(max(0.0, upper - median))


def safety_stock_intermittent(
    demand_series: Iterable[float],
    lead_time_mean: float,
    service_level: float = 0.95,
) -> float:
    """Simplified, Croston-inspired safety stock for intermittent demand.

    For SKUs that sell on only a fraction of days, the normal formula's
    "average demand × Z × sqrt(LT)" reasoning breaks down: most of the
    variance lives in *whether* a sale happens, not how big it is. This
    function separates those two pieces and combines them::

        non_zero = demand on days with sales
        SS = mean(non_zero) * (1 + CV(non_zero) * Z) * lead_time_mean

    .. note::
       This is a **simplified, practitioner-friendly adaptation** of
       Croston's idea, not the full Croston or Syntetos-Boylan
       Approximation (SBA) method. It keeps the spirit of "model size
       and incidence separately" while staying readable for a non-
       specialist. For high-stakes intermittent SKUs, use a proper
       Croston/SBA implementation.

    Parameters
    ----------
    demand_series : iterable of float
        Daily units-sold history for the SKU. Zero days are kept; the
        function filters them internally.
    lead_time_mean : float
        Average lead time in days.
    service_level : float, default 0.95
        Cycle service level. Mapped to a Z-score via
        :data:`SERVICE_LEVEL_Z`.

    Returns
    -------
    float
        Safety stock in units. Always non-negative. Returns ``0.0`` if
        the SKU has no non-zero days.
    """
    arr = np.asarray(list(demand_series), dtype=float)
    if lead_time_mean < 0:
        raise ValueError("lead_time_mean must be non-negative")

    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return 0.0

    mean_nz = float(nonzero.mean())
    std_nz = float(nonzero.std(ddof=0))
    cv_nz = std_nz / mean_nz if mean_nz > 0 else 0.0
    z = _z_for(service_level)

    return float(mean_nz * (1.0 + cv_nz * z) * lead_time_mean)


def select_safety_stock_method(pattern: str) -> str:
    """Recommend a safety-stock method for a given demand pattern.

    Returns one of ``"normal"``, ``"kde"``, or ``"intermittent"``. The
    recommendations follow the rule-of-thumb pairing that practitioners
    use:

    * smooth, weekly_seasonal → normal (cheap and accurate enough)
    * holiday_spike           → kde (fat-tailed, normal under-buffers)
    * intermittent            → intermittent (Croston-inspired)
    * new_sku                 → normal, with the caveat that any
      estimate is unreliable until the SKU has more history

    Parameters
    ----------
    pattern : str
        One of the five demand patterns produced by
        :func:`msmt.resilience.classifier.classify_pattern`.

    Returns
    -------
    str
        The method name. Useful as a key into a dispatch table.
    """
    mapping = {
        "smooth": "normal",
        "weekly_seasonal": "normal",
        "holiday_spike": "kde",
        "intermittent": "intermittent",
        "new_sku": "normal",
    }
    if pattern not in mapping:
        raise ValueError(
            f"Unknown pattern '{pattern}'. Expected one of {sorted(mapping)}"
        )
    return mapping[pattern]
