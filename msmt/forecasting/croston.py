"""Croston's method for intermittent demand.

Many SMB SKUs sell on only a fraction of days. Smoothing daily totals
directly underestimates the eventual order size on selling days, which
is exactly the quantity inventory planning cares about. Croston's
1972 trick is to separate the two things that vary:

* the *size* of demand on days when sales happen, and
* the *interval* between such days.

Each is smoothed independently, and the per-day forecast is their
ratio.

.. note::
   This module implements the **original Croston (1972) method**. For
   production use with very sparse demand, consider the
   Syntetos-Boylan Approximation (SBA), which corrects Croston's known
   bias by multiplying the forecast by ``1 - alpha / 2``.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from msmt.forecasting.baselines import _Z95, _as_array, _pi_from_residuals

ForecastReturn = Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]


def _croston_in_sample(arr: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    """Run Croston in-sample and return (per-day fitted forecasts, final rate).

    The in-sample forecast at each time step ``t`` is the most recent
    smoothed-size / smoothed-interval ratio. At time steps that fall
    before any non-zero observation the ratio is undefined and we
    return ``0`` for those positions, matching the convention used in
    the literature.
    """
    n = arr.size
    fitted = np.zeros(n, dtype=float)

    nonzero_mask = arr > 0
    if not nonzero_mask.any():
        return fitted, 0.0

    first = int(np.argmax(nonzero_mask))
    size = float(arr[first])
    interval = float(first + 1)
    rate = size / interval
    fitted[first:] = rate

    last_nonzero_idx = first
    for t in range(first + 1, n):
        if arr[t] > 0:
            new_interval = float(t - last_nonzero_idx)
            size = alpha * float(arr[t]) + (1 - alpha) * size
            interval = alpha * new_interval + (1 - alpha) * interval
            rate = size / interval if interval > 0 else 0.0
            last_nonzero_idx = t
        fitted[t] = rate

    return fitted, rate


def croston_forecast(
    series: Union[np.ndarray, pd.Series, Sequence[float]],
    horizon: int,
    alpha: float = 0.1,
    return_pi: bool = False,
) -> ForecastReturn:
    """Croston's method for intermittent / lumpy demand.

    Returns a constant per-day forecast for the horizon: the smoothed
    average daily demand obtained by separately smoothing non-zero
    sales sizes and inter-demand intervals, then dividing.

    Parameters
    ----------
    series : array-like
        Historical daily ``units_sold``.
    horizon : int
        Number of days to forecast.
    alpha : float, default 0.1
        Smoothing parameter applied to both the size and interval
        series. Croston's original recommendation is in the
        ``[0.05, 0.20]`` range; ``0.1`` is the practitioner default.
    return_pi : bool, default False
        If True, also return ``(forecast, lower_95, upper_95)``. The
        interval is computed from in-sample one-step-ahead residuals
        and assumes residuals are approximately stationary; for very
        sparse demand it can be wide because the residuals include
        many zero days.

    Returns
    -------
    numpy.ndarray, or tuple of three arrays if ``return_pi`` is True.

    Notes
    -----
    For all-zero history this returns a zero forecast; production
    callers should treat zero-rate Croston output as a signal to look
    at the underlying SKU rather than to commit to a zero reorder.
    """
    arr = _as_array(series)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    fitted, rate = _croston_in_sample(arr, alpha)
    forecast = np.full(horizon, rate, dtype=float)
    if not return_pi:
        return forecast

    residuals = arr - fitted
    lower, upper = _pi_from_residuals(forecast, residuals)
    lower = np.maximum(lower, 0.0)
    return forecast, lower, upper
