"""Synthetic seller data generator.

This module produces **synthetic** daily SKU-level seller data resembling what
a U.S. small business might export from a marketplace seller portal (e.g.,
Amazon Seller Central, Walmart Seller Center). No proprietary data is used,
referenced, or reverse-engineered. The generators here exist so that the rest
of the toolkit (integrity, resilience, forecasting modules) and its examples
can run end-to-end without anyone having to share real seller data.

Five demand patterns are supported, chosen to reflect the archetypes that
small marketplace sellers actually face:

    1. ``smooth``           — low-variance, near-stationary demand.
    2. ``weekly_seasonal``  — steady demand with a day-of-week cycle.
    3. ``holiday_spike``    — predictable lifts at known holiday windows.
    4. ``intermittent``     — lumpy demand with many zero-sales days.
    5. ``new_sku``          — short history, ramp-in from launch date.

The output schema is intentionally minimal but sufficient for forecasting and
inventory examples::

    date, sku_id, units_sold, listing_price, stock_on_hand,
    lead_time_days, category, pattern

The ``pattern`` column lets downstream notebooks filter by demand archetype;
it is not assumed by integrity/resilience modules.

All generators accept either an integer seed or a ``numpy.random.Generator``
so calls compose cleanly. Given the same seed, the same DataFrame is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

PATTERNS: tuple[str, ...] = (
    "smooth",
    "weekly_seasonal",
    "holiday_spike",
    "intermittent",
    "new_sku",
)

PATTERN_CODES: dict[str, str] = {
    "smooth": "SM",
    "weekly_seasonal": "WS",
    "holiday_spike": "HS",
    "intermittent": "IM",
    "new_sku": "NS",
}

_CATEGORIES: tuple[str, ...] = (
    "Home & Kitchen",
    "Beauty & Personal Care",
    "Outdoors",
    "Pet Supplies",
    "Office Products",
    "Toys & Games",
    "Health & Household",
    "Arts & Crafts",
)

_RngLike = Union[int, np.random.Generator, None]


def _as_rng(seed: _RngLike) -> np.random.Generator:
    """Coerce a seed or Generator into a ``numpy.random.Generator``."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class _SkuSpec:
    """Per-SKU static attributes that don't vary day-to-day."""

    sku_id: str
    category: str
    base_price: float
    lead_time_days: int
    initial_stock: int


def _draw_sku_spec(rng: np.random.Generator, sku_id: str) -> _SkuSpec:
    """Draw plausible static attributes for one SKU."""
    category = rng.choice(_CATEGORIES)
    base_price = float(np.round(rng.uniform(8.0, 60.0), 2))
    lead_time_days = int(rng.integers(7, 45))
    initial_stock = int(rng.integers(60, 400))
    return _SkuSpec(
        sku_id=sku_id,
        category=str(category),
        base_price=base_price,
        lead_time_days=lead_time_days,
        initial_stock=initial_stock,
    )


def _price_series(
    rng: np.random.Generator, base_price: float, n_days: int
) -> np.ndarray:
    """Mostly-stable price with occasional small promotional dips."""
    prices = np.full(n_days, base_price, dtype=float)
    n_promos = int(rng.integers(0, max(2, n_days // 60)))
    for _ in range(n_promos):
        start = int(rng.integers(0, n_days))
        length = int(rng.integers(2, 8))
        discount = float(rng.uniform(0.05, 0.20))
        prices[start : start + length] = np.round(
            base_price * (1.0 - discount), 2
        )
    return prices


def _apply_inventory_dynamics(
    units_sold: np.ndarray,
    initial_stock: int,
    lead_time_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Walk stock forward with periodic restocks tied to lead time.

    Stock is decremented by sales each day. When stock falls below a simple
    reorder threshold, a replenishment of roughly ``lead_time_days`` of demand
    arrives after the lead time. Sales are capped at on-hand stock to keep
    the series internally consistent.
    """
    n_days = units_sold.shape[0]
    stock = np.zeros(n_days, dtype=int)
    on_hand = int(initial_stock)
    avg_daily = max(1.0, float(units_sold.mean() if units_sold.size else 1.0))
    reorder_threshold = int(avg_daily * lead_time_days * 0.6)
    reorder_qty = int(avg_daily * lead_time_days * 1.5)
    pending: list[tuple[int, int]] = []  # (arrival_day, qty)

    for day in range(n_days):
        for arrival_day, qty in list(pending):
            if arrival_day == day:
                on_hand += qty
                pending.remove((arrival_day, qty))

        sold = int(min(units_sold[day], on_hand))
        units_sold[day] = sold
        on_hand -= sold

        if on_hand <= reorder_threshold and not pending:
            jitter = int(rng.integers(-2, 3))
            arrival = min(n_days - 1, day + max(1, lead_time_days + jitter))
            pending.append((arrival, reorder_qty))

        stock[day] = on_hand

    return stock


def _assemble_frame(
    dates: pd.DatetimeIndex,
    units_sold: np.ndarray,
    spec: _SkuSpec,
    pattern: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build the canonical DataFrame for a single SKU."""
    units_sold = units_sold.astype(int)
    prices = _price_series(rng, spec.base_price, len(dates))
    stock = _apply_inventory_dynamics(
        units_sold, spec.initial_stock, spec.lead_time_days, rng
    )
    return pd.DataFrame(
        {
            "date": dates,
            "sku_id": spec.sku_id,
            "units_sold": units_sold,
            "listing_price": prices,
            "stock_on_hand": stock,
            "lead_time_days": spec.lead_time_days,
            "category": spec.category,
            "pattern": pattern,
        }
    )


def _date_index(n_days: int, end_date: Optional[str] = None) -> pd.DatetimeIndex:
    """Build a daily DatetimeIndex of length ``n_days`` ending at ``end_date``.

    A fixed default end date keeps holiday-spike outputs reproducible across
    runs that don't pass an explicit anchor.
    """
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp("2025-12-31")
    start = end - pd.Timedelta(days=n_days - 1)
    return pd.date_range(start=start, end=end, freq="D")


# ---------------------------------------------------------------------------
# Per-pattern generators
# ---------------------------------------------------------------------------


def generate_smooth(
    sku_id: str = "SKU-SM-0001",
    n_days: int = 365,
    seed: _RngLike = 42,
    end_date: Optional[str] = None,
    mean_units: float = 12.0,
) -> pd.DataFrame:
    """Generate one SKU with smooth, near-stationary daily demand.

    Parameters
    ----------
    sku_id : str
        Identifier assigned to the generated SKU.
    n_days : int
        Number of consecutive daily observations to produce.
    seed : int, numpy.random.Generator, or None
        Seed or Generator controlling all randomness.
    end_date : str, optional
        Last calendar date (inclusive), as ``YYYY-MM-DD``. Defaults to
        ``2025-12-31`` so outputs are reproducible without an anchor.
    mean_units : float
        Approximate average daily units sold.

    Returns
    -------
    pandas.DataFrame
        One row per day with the canonical seller schema.
    """
    rng = _as_rng(seed)
    dates = _date_index(n_days, end_date)
    spec = _draw_sku_spec(rng, sku_id)
    units = rng.poisson(lam=mean_units, size=n_days).astype(float)
    return _assemble_frame(dates, units, spec, "smooth", rng)


def generate_weekly_seasonal(
    sku_id: str = "SKU-WS-0001",
    n_days: int = 365,
    seed: _RngLike = 42,
    end_date: Optional[str] = None,
    mean_units: float = 14.0,
    weekend_lift: float = 1.6,
) -> pd.DataFrame:
    """Generate one SKU with steady demand plus a day-of-week cycle.

    Saturday/Sunday demand is multiplied by ``weekend_lift``; midweek dips
    slightly to keep the weekly average near ``mean_units``.

    Parameters
    ----------
    sku_id : str
        Identifier assigned to the generated SKU.
    n_days : int
        Number of consecutive daily observations to produce.
    seed : int, numpy.random.Generator, or None
        Seed or Generator controlling all randomness.
    end_date : str, optional
        Last calendar date (inclusive). Defaults to ``2025-12-31``.
    mean_units : float
        Approximate weekly-average daily units sold.
    weekend_lift : float
        Multiplier applied to Saturday and Sunday demand.

    Returns
    -------
    pandas.DataFrame
        One row per day with the canonical seller schema.
    """
    rng = _as_rng(seed)
    dates = _date_index(n_days, end_date)
    spec = _draw_sku_spec(rng, sku_id)

    dow = dates.dayofweek.to_numpy()
    multipliers = np.where(dow >= 5, weekend_lift, 0.92)
    lam = mean_units * multipliers
    units = rng.poisson(lam=lam).astype(float)
    return _assemble_frame(dates, units, spec, "weekly_seasonal", rng)


def generate_holiday_spike(
    sku_id: str = "SKU-HS-0001",
    n_days: int = 365,
    seed: _RngLike = 42,
    end_date: Optional[str] = None,
    mean_units: float = 10.0,
    spike_multiplier: float = 10.0,
) -> pd.DataFrame:
    """Generate one SKU with predictable holiday-window spikes.

    Spikes are placed around Black Friday, Cyber Monday, and mid-December
    within whatever calendar window the date index covers. Each spike is a
    Gaussian bump centered on the target date.

    Parameters
    ----------
    sku_id : str
        Identifier assigned to the generated SKU.
    n_days : int
        Number of consecutive daily observations to produce.
    seed : int, numpy.random.Generator, or None
        Seed or Generator controlling all randomness.
    end_date : str, optional
        Last calendar date (inclusive). Defaults to ``2025-12-31``.
    mean_units : float
        Baseline average daily units sold outside the holiday windows.
    spike_multiplier : float
        Peak demand at the center of a holiday window, relative to baseline.

    Returns
    -------
    pandas.DataFrame
        One row per day with the canonical seller schema.
    """
    rng = _as_rng(seed)
    dates = _date_index(n_days, end_date)
    spec = _draw_sku_spec(rng, sku_id)

    baseline = np.full(n_days, mean_units, dtype=float)

    holiday_anchors: List[pd.Timestamp] = []
    for year in range(dates[0].year, dates[-1].year + 1):
        nov = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
        fridays = nov[nov.dayofweek == 4]
        if len(fridays) >= 4:
            black_friday = fridays[3]
            holiday_anchors.append(black_friday)
            holiday_anchors.append(black_friday + pd.Timedelta(days=3))
        holiday_anchors.append(pd.Timestamp(f"{year}-12-15"))

    sigma_days = 3.5
    day_index = np.arange(n_days)
    for anchor in holiday_anchors:
        if anchor < dates[0] or anchor > dates[-1]:
            continue
        center = (anchor - dates[0]).days
        bump = (spike_multiplier - 1.0) * mean_units * np.exp(
            -0.5 * ((day_index - center) / sigma_days) ** 2
        )
        baseline = baseline + bump

    units = rng.poisson(lam=baseline).astype(float)
    return _assemble_frame(dates, units, spec, "holiday_spike", rng)


def generate_intermittent(
    sku_id: str = "SKU-IM-0001",
    n_days: int = 365,
    seed: _RngLike = 42,
    end_date: Optional[str] = None,
    sale_probability: float = 0.18,
    mean_units_when_sold: float = 4.0,
) -> pd.DataFrame:
    """Generate one SKU with lumpy/intermittent demand (many zero days).

    On each day a Bernoulli draw decides whether any sales occur; on "on"
    days, units are drawn from a Poisson with mean ``mean_units_when_sold``.

    Parameters
    ----------
    sku_id : str
        Identifier assigned to the generated SKU.
    n_days : int
        Number of consecutive daily observations to produce.
    seed : int, numpy.random.Generator, or None
        Seed or Generator controlling all randomness.
    end_date : str, optional
        Last calendar date (inclusive). Defaults to ``2025-12-31``.
    sale_probability : float
        Probability that any units sell on a given day (0 < p < 1).
    mean_units_when_sold : float
        Mean of the positive-day Poisson distribution.

    Returns
    -------
    pandas.DataFrame
        One row per day with the canonical seller schema.
    """
    rng = _as_rng(seed)
    dates = _date_index(n_days, end_date)
    spec = _draw_sku_spec(rng, sku_id)

    sale_days = rng.random(n_days) < sale_probability
    units = np.where(
        sale_days,
        rng.poisson(lam=mean_units_when_sold, size=n_days),
        0,
    ).astype(float)
    return _assemble_frame(dates, units, spec, "intermittent", rng)


def generate_new_sku(
    sku_id: str = "SKU-NS-0001",
    n_days: int = 365,
    seed: _RngLike = 42,
    end_date: Optional[str] = None,
    history_days: int = 45,
    target_mean_units: float = 9.0,
) -> pd.DataFrame:
    """Generate one new SKU with limited history and a ramp from launch.

    Days before the launch date show zero sales and zero stock so downstream
    code can detect them as "no history yet" rather than a true zero-demand
    day. After launch, demand ramps from ~25% of ``target_mean_units`` to the
    target level over the first three weeks.

    Parameters
    ----------
    sku_id : str
        Identifier assigned to the generated SKU.
    n_days : int
        Total length of the date index. Days before launch will be zeros.
    seed : int, numpy.random.Generator, or None
        Seed or Generator controlling all randomness.
    end_date : str, optional
        Last calendar date (inclusive). Defaults to ``2025-12-31``.
    history_days : int
        How many days of post-launch history to populate, ending at
        ``end_date``. Must be <= ``n_days``.
    target_mean_units : float
        Mean daily units the SKU ramps toward.

    Returns
    -------
    pandas.DataFrame
        One row per day with the canonical seller schema.
    """
    if history_days > n_days:
        raise ValueError("history_days cannot exceed n_days")

    rng = _as_rng(seed)
    dates = _date_index(n_days, end_date)
    spec = _draw_sku_spec(rng, sku_id)

    units = np.zeros(n_days, dtype=float)
    launch_idx = n_days - history_days
    ramp_len = min(21, history_days)
    ramp = np.linspace(0.25, 1.0, ramp_len)
    post_launch_lambdas = np.full(history_days, target_mean_units, dtype=float)
    post_launch_lambdas[:ramp_len] = target_mean_units * ramp
    units[launch_idx:] = rng.poisson(lam=post_launch_lambdas)

    df = _assemble_frame(dates, units, spec, "new_sku", rng)
    df.loc[df.index < launch_idx, "stock_on_hand"] = 0
    return df


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------


def _allocate_pattern_counts(n_skus: int) -> dict[str, int]:
    """Split ``n_skus`` across the five patterns with a sensible default mix.

    The mix leans toward smooth and weekly_seasonal (the common cases) while
    still guaranteeing at least one SKU of every pattern when n_skus >= 5.
    """
    if n_skus < 1:
        raise ValueError("n_skus must be >= 1")

    weights = {
        "smooth": 0.35,
        "weekly_seasonal": 0.25,
        "holiday_spike": 0.15,
        "intermittent": 0.15,
        "new_sku": 0.10,
    }
    counts = {p: int(np.floor(n_skus * w)) for p, w in weights.items()}

    if n_skus >= len(PATTERNS):
        for p in PATTERNS:
            if counts[p] == 0:
                counts[p] = 1

    remaining = n_skus - sum(counts.values())
    ordered = sorted(PATTERNS, key=lambda p: -weights[p])
    i = 0
    while remaining > 0:
        counts[ordered[i % len(ordered)]] += 1
        remaining -= 1
        i += 1
    while remaining < 0:
        for p in ordered[::-1]:
            if counts[p] > 0:
                counts[p] -= 1
                remaining += 1
                if remaining == 0:
                    break

    return counts


def generate_seller_data(
    n_skus: int = 50,
    n_days: int = 365,
    seed: _RngLike = 42,
    end_date: Optional[str] = None,
    patterns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Generate a synthetic multi-SKU seller dataset mixing demand patterns.

    The returned DataFrame is the canonical input format used by examples and
    tests across the rest of the toolkit.

    Parameters
    ----------
    n_skus : int
        Total number of distinct SKUs to generate.
    n_days : int
        Length of the daily date index for each SKU.
    seed : int, numpy.random.Generator, or None
        Seed or Generator controlling all randomness. With the same seed,
        ``n_skus``, ``n_days``, ``end_date``, and ``patterns``, this function
        returns an identical DataFrame.
    end_date : str, optional
        Last calendar date (inclusive), as ``YYYY-MM-DD``. Defaults to
        ``2025-12-31``.
    patterns : sequence of str, optional
        Restrict generation to a subset of :data:`PATTERNS`. If ``None``, all
        five patterns are mixed using a default allocation.

    Returns
    -------
    pandas.DataFrame
        One row per (sku_id, date) with columns: ``date``, ``sku_id``,
        ``units_sold``, ``listing_price``, ``stock_on_hand``,
        ``lead_time_days``, ``category``, ``pattern``. Sorted by
        ``(sku_id, date)``.

    Notes
    -----
    All output is **synthetic** and not derived from any proprietary or
    real-world seller dataset.

    Examples
    --------
    >>> from msmt.data import generate_seller_data
    >>> df = generate_seller_data(n_skus=10, n_days=180, seed=0)
    >>> sorted(df["pattern"].unique())  # doctest: +ELLIPSIS
    [...]
    """
    rng = _as_rng(seed)

    if patterns is None:
        counts = _allocate_pattern_counts(n_skus)
    else:
        invalid = set(patterns) - set(PATTERNS)
        if invalid:
            raise ValueError(
                f"Unknown patterns: {sorted(invalid)}. Allowed: {PATTERNS}"
            )
        per = max(1, n_skus // len(patterns))
        counts = {p: per for p in patterns}
        remaining = n_skus - sum(counts.values())
        i = 0
        plist = list(patterns)
        while remaining > 0:
            counts[plist[i % len(plist)]] += 1
            remaining -= 1
            i += 1

    pattern_to_fn = {
        "smooth": generate_smooth,
        "weekly_seasonal": generate_weekly_seasonal,
        "holiday_spike": generate_holiday_spike,
        "intermittent": generate_intermittent,
        "new_sku": generate_new_sku,
    }

    frames: list[pd.DataFrame] = []
    sku_counter = 1
    for pattern in PATTERNS:
        code = PATTERN_CODES[pattern]
        for _ in range(counts.get(pattern, 0)):
            sku_id = f"SKU-{code}-{sku_counter:04d}"
            sku_counter += 1
            child_seed = int(rng.integers(0, 2**31 - 1))
            df = pattern_to_fn[pattern](
                sku_id=sku_id,
                n_days=n_days,
                seed=child_seed,
                end_date=end_date,
            )
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return out
