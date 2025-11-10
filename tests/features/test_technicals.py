"""Tests for technical indicator helpers."""

import pandas as pd

from src.features.technicals import atr14, sma


def test_sma_simple_sequence():
    prices = pd.Series([1, 2, 3, 4, 5])
    assert sma(prices, 3) == 4


def test_atr14_monotonic_series():
    values = list(range(15))
    high = pd.Series([v + 1 for v in values])
    low = pd.Series(values)
    close = pd.Series([v + 0.5 for v in values])

    assert atr14(high, low, close) == 1.5
