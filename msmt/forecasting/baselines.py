"""Baseline forecasting methods.

Six methods, in order of how much structure they assume about the
demand series:

* :func:`naive_forecast` — last value, repeated.
* :func:`seasonal_naive_forecast` — last full season, repeated.
* :func:`moving_average_forecast` — trailing-window mean, repeated.
* :func:`ses_forecast` — simple exponential smoothing.
* :func:`holts_forecast` — exponential smoothing with trend.
* :func:`holt_winters_forecast` — exponential smoothing with trend +
  seasonality.

Every function accepts a 1-D numpy array or pandas Series and a
``horizon`` integer, and returns a numpy array of length ``horizon``.
Pass ``return_pi=True`` to also receive a 95% prediction interval in
the form ``(forecast, lower_95, upper_95)``. The interval is computed
from the in-sample one-step-ahead residuals as ``mean ± 1.96 * std``,
which is a deliberately simple approach; it assumes residuals are
approximately normal and stationary, which is reasonable for the SMB
catalog scale this toolkit targets.

Parameter optimization (α for SES, α/β for Holt, α/β/γ for
Holt-Winters) uses :func:`scipy.optimize.minimize` when SciPy is
already installed in the environment, and falls back to a 9-point
grid search over ``[0.1, 0.9]`` per parameter when it isn't. SciPy is
*not* a declared runtime dependency of the toolkit; if it happens to
be present (e.g. via :mod:`statsforecast`) we use it.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:  # opportunistic — scipy is not a declared runtime dep.
    from scipy.optimize import minimize as _scipy_minimize
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - exercised only when scipy is absent
    _HAS_SCIPY = False

ForecastReturn = Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
_Z95 = 1.96
_GRID = np.linspace(0.1, 0.9, 9)


def _as_array(series: Union[np.ndarray, pd.Series, Sequence[float]]) -> np.ndarray:
    """Coerce ``series`` into a 1-D float numpy array."""
    if isinstance(series, pd.Series):
        return series.to_numpy(dtype=float)
    return np.asarray(list(series), dtype=float)


def _pi_from_residuals(
    forecast: np.ndarray, residuals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a symmetric 95% PI band around ``forecast`` from residuals."""
    if residuals.size == 0:
        std = 0.0
    else:
        std = float(np.std(residuals, ddof=0))
    half = _Z95 * std
    return forecast - half, forecast + half


def _minimize_bounded(
    loss: Callable[[np.ndarray], float],
    init: Sequence[float],
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Minimize a scalar loss subject to per-parameter bounds.

    Tries SciPy's L-BFGS-B when available; otherwise grid-searches the
    Cartesian product of :data:`_GRID` clipped to each bound. The grid
    is coarse but adequate for SMB-scale series — the loss surfaces
    here are smooth and the optima are usually shallow.
    """
    if _HAS_SCIPY:
        result = _scipy_minimize(
            lambda x: loss(np.asarray(x, dtype=float)),
            x0=np.asarray(init, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        return np.asarray(result.x, dtype=float)

    grids = []
    for lo, hi in bounds:
        g = _GRID[(_GRID >= lo) & (_GRID <= hi)]
        if g.size == 0:
            g = np.array([(lo + hi) / 2.0])
        grids.append(g)
    mesh = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(grids), -1).T
    best, best_loss = mesh[0], float("inf")
    for params in mesh:
        val = loss(params)
        if val < best_loss:
            best, best_loss = params, val
    return np.asarray(best, dtype=float)


def naive_forecast(
    series: Union[np.ndarray, pd.Series],
    horizon: int,
    return_pi: bool = False,
) -> ForecastReturn:
    """Forecast = last observed value, repeated for ``horizon`` days.

    The simplest possible baseline. Useful as a sanity check: any
    method that doesn't beat naive on a given SKU probably isn't
    worth the complexity it adds.

    Parameters
    ----------
    series : array-like
        Historical daily ``units_sold``.
    horizon : int
        Number of days to forecast.
    return_pi : bool, default False
        If True, also return ``(forecast, lower_95, upper_95)``.

    Returns
    -------
    numpy.ndarray, or tuple of three arrays if ``return_pi`` is True.
    """
    arr = _as_array(series)
    if horizon < 1 or arr.size < 1:
        raise ValueError("series must be non-empty and horizon >= 1")
    last = float(arr[-1])
    forecast = np.full(horizon, last, dtype=float)
    if not return_pi:
        return forecast
    residuals = np.diff(arr)
    lower, upper = _pi_from_residuals(forecast, residuals)
    return forecast, lower, upper


def seasonal_naive_forecast(
    series: Union[np.ndarray, pd.Series],
    horizon: int,
    season_length: int = 7,
    return_pi: bool = False,
) -> ForecastReturn:
    """Forecast = the last full season, repeated to cover ``horizon``.

    The default ``season_length=7`` produces a "same day next week"
    forecast — a strong baseline for weekly-seasonal SKUs.

    Parameters
    ----------
    series : array-like
        Historical daily ``units_sold``.
    horizon : int
        Number of days to forecast.
    season_length : int, default 7
        Length of one season in days.
    return_pi : bool, default False
        If True, also return ``(forecast, lower_95, upper_95)``.

    Returns
    -------
    numpy.ndarray, or tuple of three arrays if ``return_pi`` is True.
    """
    arr = _as_array(series)
    if season_length < 1:
        raise ValueError("season_length must be >= 1")
    if arr.size < season_length:
        # Not enough history for a full season — fall back to naive.
        return naive_forecast(arr, horizon, return_pi=return_pi)
    last_season = arr[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    forecast = np.tile(last_season, reps)[:horizon]
    if not return_pi:
        return forecast
    residuals = arr[season_length:] - arr[:-season_length]
    lower, upper = _pi_from_residuals(forecast, residuals)
    return forecast, lower, upper


def moving_average_forecast(
    series: Union[np.ndarray, pd.Series],
    horizon: int,
    window: int = 14,
    return_pi: bool = False,
) -> ForecastReturn:
    """Forecast = trailing-window mean, repeated for ``horizon`` days.

    A safer default than naive when daily demand is noisy. The default
    14-day window is short enough to track recent shifts and long
    enough to average out day-to-day noise.

    Parameters
    ----------
    series : array-like
        Historical daily ``units_sold``.
    horizon : int
        Number of days to forecast.
    window : int, default 14
        Length of the trailing window (in days).
    return_pi : bool, default False
        If True, also return ``(forecast, lower_95, upper_95)``.

    Returns
    -------
    numpy.ndarray, or tuple of three arrays if ``return_pi`` is True.
    """
    arr = _as_array(series)
    if window < 1:
        raise ValueError("window must be >= 1")
    eff_window = min(window, arr.size)
    mean = float(arr[-eff_window:].mean()) if arr.size > 0 else 0.0
    forecast = np.full(horizon, mean, dtype=float)
    if not return_pi:
        return forecast
    rolling = pd.Series(arr).rolling(eff_window, min_periods=eff_window).mean()
    residuals = arr[eff_window:] - rolling.shift(1).dropna().to_numpy()[: arr.size - eff_window]
    lower, upper = _pi_from_residuals(forecast, residuals)
    return forecast, lower, upper


# ---------------------------------------------------------------------------
# Exponential smoothing family
# ---------------------------------------------------------------------------


def _ses_fit(arr: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Run SES and return (in-sample one-step forecasts, level series)."""
    n = arr.size
    level = np.zeros(n, dtype=float)
    fitted = np.zeros(n, dtype=float)
    level[0] = arr[0]
    fitted[0] = arr[0]
    for t in range(1, n):
        fitted[t] = level[t - 1]
        level[t] = alpha * arr[t] + (1 - alpha) * level[t - 1]
    return fitted, level


def _ses_loss(arr: np.ndarray) -> Callable[[np.ndarray], float]:
    def loss(params: np.ndarray) -> float:
        alpha = float(params[0])
        fitted, _ = _ses_fit(arr, alpha)
        return float(np.mean((arr[1:] - fitted[1:]) ** 2))
    return loss


def ses_forecast(
    series: Union[np.ndarray, pd.Series],
    horizon: int,
    alpha: Optional[float] = None,
    return_pi: bool = False,
) -> ForecastReturn:
    """Simple exponential smoothing.

    Recursively smooths the level::

        L_t = alpha * y_t + (1 - alpha) * L_{t-1}

    The forecast is the final level, held flat across the horizon.
    Best for *smooth*-pattern SKUs without trend or seasonality.

    Parameters
    ----------
    series : array-like
        Historical daily ``units_sold``.
    horizon : int
        Number of days to forecast.
    alpha : float, optional
        Smoothing parameter in ``(0, 1)``. If ``None``, optimized to
        minimize one-step-ahead MSE.
    return_pi : bool, default False
        If True, also return ``(forecast, lower_95, upper_95)``.

    Returns
    -------
    numpy.ndarray, or tuple of three arrays if ``return_pi`` is True.
    """
    arr = _as_array(series)
    if arr.size < 2:
        raise ValueError("series must have at least 2 observations for SES")
    if alpha is None:
        alpha = float(_minimize_bounded(_ses_loss(arr), init=[0.3], bounds=[(0.05, 0.95)])[0])
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    fitted, level = _ses_fit(arr, alpha)
    forecast = np.full(horizon, level[-1], dtype=float)
    if not return_pi:
        return forecast
    residuals = arr[1:] - fitted[1:]
    lower, upper = _pi_from_residuals(forecast, residuals)
    return forecast, lower, upper


def _holt_fit(
    arr: np.ndarray, alpha: float, beta: float
) -> Tuple[np.ndarray, float, float]:
    """Run Holt's linear method; return (fitted, final_level, final_trend)."""
    n = arr.size
    level = arr[0]
    trend = arr[1] - arr[0] if n > 1 else 0.0
    fitted = np.zeros(n, dtype=float)
    fitted[0] = arr[0]
    for t in range(1, n):
        fitted[t] = level + trend
        new_level = alpha * arr[t] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend
    return fitted, level, trend


def _holt_loss(arr: np.ndarray) -> Callable[[np.ndarray], float]:
    def loss(params: np.ndarray) -> float:
        alpha, beta = float(params[0]), float(params[1])
        fitted, _, _ = _holt_fit(arr, alpha, beta)
        return float(np.mean((arr[1:] - fitted[1:]) ** 2))
    return loss


def holts_forecast(
    series: Union[np.ndarray, pd.Series],
    horizon: int,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    return_pi: bool = False,
) -> ForecastReturn:
    """Holt's linear (double exponential smoothing) method.

    Smooths a level *and* a trend, projecting the trend forward across
    the horizon. Use when a SKU has a persistent upward or downward
    drift on top of otherwise smooth demand.

    Parameters
    ----------
    series : array-like
        Historical daily ``units_sold``.
    horizon : int
        Number of days to forecast.
    alpha, beta : float, optional
        Smoothing parameters in ``(0, 1)``. If either is ``None``,
        both are jointly optimized to minimize one-step-ahead MSE.
    return_pi : bool, default False
        If True, also return ``(forecast, lower_95, upper_95)``.

    Returns
    -------
    numpy.ndarray, or tuple of three arrays if ``return_pi`` is True.
    """
    arr = _as_array(series)
    if arr.size < 3:
        raise ValueError("series must have at least 3 observations for Holt")
    if alpha is None or beta is None:
        params = _minimize_bounded(
            _holt_loss(arr), init=[0.3, 0.1], bounds=[(0.05, 0.95), (0.05, 0.95)]
        )
        alpha = float(params[0]) if alpha is None else alpha
        beta = float(params[1]) if beta is None else beta

    fitted, level, trend = _holt_fit(arr, alpha, beta)
    horizons = np.arange(1, horizon + 1)
    forecast = level + trend * horizons
    if not return_pi:
        return forecast
    residuals = arr[1:] - fitted[1:]
    lower, upper = _pi_from_residuals(forecast, residuals)
    return forecast, lower, upper


def _hw_fit(
    arr: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    season_length: int,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Run additive Holt-Winters; return (fitted, level, trend, seasonals)."""
    n = arr.size
    if n < 2 * season_length:
        raise ValueError(
            "Holt-Winters needs at least 2 full seasons of history"
        )
    initial = arr[:season_length]
    level = float(initial.mean())
    trend = float(
        (arr[season_length : 2 * season_length].mean() - level) / season_length
    )
    seasonals = arr[:season_length] - level
    fitted = np.zeros(n, dtype=float)
    fitted[:season_length] = level + trend + seasonals
    for t in range(season_length, n):
        s_idx = t % season_length
        fitted[t] = level + trend + seasonals[s_idx]
        new_level = alpha * (arr[t] - seasonals[s_idx]) + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        seasonals[s_idx] = gamma * (arr[t] - new_level) + (1 - gamma) * seasonals[s_idx]
        level, trend = new_level, new_trend
    return fitted, level, trend, seasonals


def _hw_loss(
    arr: np.ndarray, season_length: int
) -> Callable[[np.ndarray], float]:
    def loss(params: np.ndarray) -> float:
        alpha, beta, gamma = (float(params[i]) for i in range(3))
        try:
            fitted, _, _, _ = _hw_fit(arr, alpha, beta, gamma, season_length)
        except ValueError:
            return float("inf")
        return float(np.mean((arr[season_length:] - fitted[season_length:]) ** 2))
    return loss


def holt_winters_forecast(
    series: Union[np.ndarray, pd.Series],
    horizon: int,
    season_length: int = 7,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    return_pi: bool = False,
) -> ForecastReturn:
    """Additive Holt-Winters (triple exponential smoothing).

    Tracks a level, a trend, and a seasonal cycle of length
    ``season_length`` (default 7 days). The forecast extrapolates the
    level + trend and adds the recurring seasonal pattern.

    The implementation is the additive variant — it adds seasonal
    deltas rather than multiplying by seasonal factors. For SKUs whose
    seasonal swings scale with overall volume, a multiplicative
    variant would be more appropriate; the additive version is robust
    and adequate for the SMB use case.

    Parameters
    ----------
    series : array-like
        Historical daily ``units_sold``. Must have at least
        ``2 * season_length`` observations.
    horizon : int
        Number of days to forecast.
    season_length : int, default 7
        Length of the seasonal cycle in days.
    alpha, beta, gamma : float, optional
        Smoothing parameters for level, trend, and seasonality. Any
        that are ``None`` are jointly optimized to minimize one-step-
        ahead MSE.
    return_pi : bool, default False
        If True, also return ``(forecast, lower_95, upper_95)``.

    Returns
    -------
    numpy.ndarray, or tuple of three arrays if ``return_pi`` is True.
    """
    arr = _as_array(series)
    if arr.size < 2 * season_length:
        raise ValueError("series must cover at least 2 full seasons")
    if alpha is None or beta is None or gamma is None:
        params = _minimize_bounded(
            _hw_loss(arr, season_length),
            init=[0.3, 0.1, 0.3],
            bounds=[(0.05, 0.95), (0.05, 0.95), (0.05, 0.95)],
        )
        alpha = float(params[0]) if alpha is None else alpha
        beta = float(params[1]) if beta is None else beta
        gamma = float(params[2]) if gamma is None else gamma

    fitted, level, trend, seasonals = _hw_fit(arr, alpha, beta, gamma, season_length)
    forecast = np.zeros(horizon, dtype=float)
    for h in range(horizon):
        forecast[h] = level + (h + 1) * trend + seasonals[(arr.size + h) % season_length]
    if not return_pi:
        return forecast
    residuals = arr[season_length:] - fitted[season_length:]
    lower, upper = _pi_from_residuals(forecast, residuals)
    return forecast, lower, upper
