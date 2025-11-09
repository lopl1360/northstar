"""Technical indicator feature calculations."""

from __future__ import annotations

import pandas as pd


def sma(prices: pd.Series, window: int) -> float:
    """Return the simple moving average for the final ``window`` observations.

    Args:
        prices: Series of historical prices ordered chronologically.
        window: Number of observations to include in the average.

    Returns:
        The simple moving average over the most recent ``window`` values.

    Raises:
        ValueError: If ``window`` is not positive or the series is shorter than
            ``window`` observations.
    """

    if window <= 0:
        raise ValueError("window must be positive")

    if len(prices) < window:
        raise ValueError("not enough data to compute SMA")

    return float(prices.tail(window).mean())


def atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """Calculate the 14 period Average True Range (ATR)."""

    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low and close series must have the same length")

    if len(high) < 14:
        raise ValueError("at least 14 observations are required to compute ATR14")

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    return float(true_range.tail(14).mean())
