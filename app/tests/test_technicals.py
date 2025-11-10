import pandas as pd
import pytest
from pathlib import Path

from src.features.technicals import atr14, sma


FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "synthetic_prices.csv"


def test_sma_returns_mean_of_trailing_window():
    prices = pd.Series([1, 2, 3, 4, 5, 6])
    assert sma(prices, 3) == pytest.approx((4 + 5 + 6) / 3)


def test_sma_validates_inputs():
    prices = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        sma(prices, 0)
    with pytest.raises(ValueError):
        sma(prices, 4)


def test_atr14_matches_fixture_expectation():
    frame = pd.read_csv(FIXTURE_PATH, parse_dates=["date"])
    computed = atr14(frame["high"], frame["low"], frame["close"])
    assert computed == pytest.approx(2.0)


def test_atr14_validates_series_lengths():
    frame = pd.read_csv(FIXTURE_PATH, parse_dates=["date"])
    with pytest.raises(ValueError):
        atr14(frame["high"], frame["low"], frame["close"].iloc[:-1])
    with pytest.raises(ValueError):
        atr14(frame["high"].iloc[:10], frame["low"].iloc[:10], frame["close"].iloc[:10])
