"""Prophet wrapper with a pure-numpy fallback.

This module exposes a single function, :func:`prophet_forecast`, that
prefers Meta's `Prophet <https://facebook.github.io/prophet/>`_ when
it's installed and falls back to a transparent numpy additive model
when it isn't. The fallback is deliberately simple — it isn't trying
to compete with Prophet, just to keep the rest of the toolkit working
in environments where Prophet can't be installed (e.g. very new
Python versions before Prophet has wheels available).

The fallback decomposes a daily series into:

* a **linear trend** fit by ordinary least squares against day index,
* a **day-of-week seasonal** component (the seven mean residuals after
  removing the trend), and
* an optional **holiday lift** computed as the mean residual on
  user-supplied holiday dates.

All three components are projected forward by extending the day index
and looking up the appropriate weekday / holiday slot. The 95%
prediction interval is built from in-sample residual standard
deviation, matching the convention used in the rest of the baselines
module.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from msmt.forecasting.baselines import _Z95, _as_array

try:  # opportunistic — Prophet is an optional extra.
    from prophet import Prophet  # type: ignore[import-not-found]
    _HAS_PROPHET = True
except Exception:  # pragma: no cover - exercised only when Prophet missing
    Prophet = None  # type: ignore[assignment]
    _HAS_PROPHET = False


def is_prophet_available() -> bool:
    """Return True when Prophet was importable at module-load time."""
    return _HAS_PROPHET


def _numpy_fallback(
    series: np.ndarray,
    horizon: int,
    dates: pd.DatetimeIndex,
    holidays: Optional[pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy additive trend + weekly + holiday model."""
    n = series.size
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, series, 1)
    trend = slope * t + intercept

    detrended = series - trend
    dow = dates.dayofweek.to_numpy()
    seasonal = np.zeros(n, dtype=float)
    dow_means = np.zeros(7, dtype=float)
    for d in range(7):
        mask = dow == d
        if mask.any():
            dow_means[d] = float(detrended[mask].mean())
    seasonal = dow_means[dow]

    holiday_dates: set[pd.Timestamp] = set()
    holiday_lift = 0.0
    if holidays is not None and len(holidays) > 0:
        if "ds" not in holidays.columns:
            raise ValueError("holidays DataFrame must have a 'ds' column")
        holiday_dates = {pd.Timestamp(d).normalize() for d in holidays["ds"]}
        in_sample_holiday_mask = np.array(
            [pd.Timestamp(d).normalize() in holiday_dates for d in dates]
        )
        if in_sample_holiday_mask.any():
            holiday_lift = float(
                (detrended - seasonal)[in_sample_holiday_mask].mean()
            )

    holiday_indicator = np.zeros(n, dtype=float)
    if holiday_dates:
        holiday_indicator = np.array(
            [
                holiday_lift if pd.Timestamp(d).normalize() in holiday_dates else 0.0
                for d in dates
            ]
        )

    fitted = trend + seasonal + holiday_indicator
    residuals = series - fitted
    sigma = float(residuals.std(ddof=0)) if residuals.size > 0 else 0.0

    future_t = np.arange(n, n + horizon, dtype=float)
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_trend = slope * future_t + intercept
    future_seasonal = dow_means[future_dates.dayofweek.to_numpy()]
    future_holiday = np.zeros(horizon, dtype=float)
    if holiday_dates:
        future_holiday = np.array(
            [
                holiday_lift if pd.Timestamp(d).normalize() in holiday_dates else 0.0
                for d in future_dates
            ]
        )

    forecast = future_trend + future_seasonal + future_holiday
    half = _Z95 * sigma
    return forecast, forecast - half, forecast + half


def prophet_forecast(
    series: Union[np.ndarray, pd.Series, Sequence[float]],
    horizon: int,
    dates: pd.DatetimeIndex,
    holidays: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast with Prophet when available; fall back to numpy otherwise.

    Parameters
    ----------
    series : array-like
        Historical daily ``units_sold``. Must align with ``dates``.
    horizon : int
        Number of future days to forecast.
    dates : pandas.DatetimeIndex
        The dates corresponding to ``series``. Must be daily-frequency
        and the same length as ``series``.
    holidays : pandas.DataFrame, optional
        Prophet-style holidays frame with a ``ds`` column of holiday
        dates. ``holiday`` (label) and ``lower_window`` /
        ``upper_window`` columns are accepted but unused by the
        fallback. If ``None``, the holiday component is omitted.

    Returns
    -------
    forecast : numpy.ndarray
        Point forecast of length ``horizon``.
    lower_95 : numpy.ndarray
        Lower bound of the 95% prediction interval.
    upper_95 : numpy.ndarray
        Upper bound of the 95% prediction interval.

    Notes
    -----
    When Prophet is not installed, this function falls back to a
    transparent additive numpy model: linear OLS trend + day-of-week
    dummy means + optional holiday lift, with a residual-std-based
    95% PI. The fallback is documented and unit-tested; it is not a
    Prophet replacement, just a graceful degradation path.
    """
    arr = _as_array(series)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(dates) != arr.size:
        raise ValueError("dates length must match series length")
    dates = pd.DatetimeIndex(dates)

    if not _HAS_PROPHET:
        return _numpy_fallback(arr, horizon, dates, holidays)

    df = pd.DataFrame({"ds": dates, "y": arr})
    model_kwargs = {
        "weekly_seasonality": True,
        "yearly_seasonality": False,
        "daily_seasonality": False,
        "interval_width": 0.95,
    }
    if holidays is not None and len(holidays) > 0:
        model_kwargs["holidays"] = holidays
    model = Prophet(**model_kwargs)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq="D")
    pred = model.predict(future).tail(horizon)
    forecast = pred["yhat"].to_numpy(dtype=float)
    lower = pred["yhat_lower"].to_numpy(dtype=float)
    upper = pred["yhat_upper"].to_numpy(dtype=float)
    return forecast, lower, upper
