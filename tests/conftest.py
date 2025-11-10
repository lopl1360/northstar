"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import pytest


@pytest.fixture
def synthetic_fundamentals() -> Dict[str, Any]:
    """Return a deterministic set of fundamental metrics for tests."""

    return {
        "price": 125.5,
        "metric": {
            "marketCapitalization": 750_000_000,
            "marketcap": 750_000_000,
            "avgVolume": 1_200_000,
            "sector": "Technology",
            "fcfPerShareTTM": 4.25,
            "revenueGrowthTTM": 0.18,
        },
    }


@pytest.fixture
def synthetic_ohlc() -> Dict[str, List[float]]:
    """Return synthetic OHLC data shaped like Finnhub candles."""

    start = datetime(2024, 1, 1)
    timestamps = [int((start + timedelta(days=idx)).timestamp()) for idx in range(5)]
    opens = [100.0 + idx for idx in range(5)]
    highs = [value + 1.5 for value in opens]
    lows = [value - 1.5 for value in opens]
    closes = [value + 0.25 for value in opens]
    volumes = [150_000 + idx * 10_000 for idx in range(5)]

    return {
        "s": "ok",
        "t": timestamps,
        "o": opens,
        "h": highs,
        "l": lows,
        "c": closes,
        "v": volumes,
    }


@pytest.fixture
def synthetic_ohlc_frame(synthetic_ohlc: Dict[str, List[float]]) -> pd.DataFrame:
    """Return the OHLC data as a pandas DataFrame for convenience."""

    return pd.DataFrame(
        {
            "timestamp": synthetic_ohlc["t"],
            "open": synthetic_ohlc["o"],
            "high": synthetic_ohlc["h"],
            "low": synthetic_ohlc["l"],
            "close": synthetic_ohlc["c"],
            "volume": synthetic_ohlc["v"],
        }
    )


class _StubFinnhubClient:
    def __init__(self, fundamentals: Dict[str, Any], candles: Dict[str, List[float]]) -> None:
        self._fundamentals = fundamentals
        self._candles = candles

    def get_symbols(self, exchange: str) -> List[Dict[str, Any]]:
        return [
            {"symbol": f"{exchange}:AAA", "description": "Alpha Corp", "currency": "USD"},
            {"displaySymbol": f"{exchange}:BBB", "description": "Beta Ltd", "currency": "CAD"},
        ]

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        return {"c": self._fundamentals["price"], "symbol": symbol}

    def get_basic_financials(self, symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "metric": self._fundamentals["metric"]}

    def get_stock_candles(self, symbol: str, *, start, end, resolution: str = "D") -> Dict[str, Any]:
        return self._candles

    def get_earnings_calendar(self, symbol: str) -> Dict[str, Any]:
        return {"earningsCalendar": []}

    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        return {"ticker": symbol, "name": f"{symbol} Incorporated"}


class _StubTwelveDataClient:
    def __init__(self, price: float) -> None:
        self._price = price

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        return {"price": self._price, "symbol": symbol}


@pytest.fixture
def stub_finnhub_client_instance(
    synthetic_fundamentals: Dict[str, Any], synthetic_ohlc: Dict[str, List[float]]
) -> _StubFinnhubClient:
    return _StubFinnhubClient(synthetic_fundamentals, synthetic_ohlc)


@pytest.fixture
def stub_twelvedata_client_instance(synthetic_fundamentals: Dict[str, Any]) -> _StubTwelveDataClient:
    return _StubTwelveDataClient(synthetic_fundamentals["price"])


@pytest.fixture(autouse=True)
def stub_data_clients(
    monkeypatch: pytest.MonkeyPatch,
    stub_finnhub_client_instance: _StubFinnhubClient,
    stub_twelvedata_client_instance: _StubTwelveDataClient,
) -> None:
    from src.data import fetch

    monkeypatch.setattr(fetch, "FinnhubClient", lambda *args, **kwargs: stub_finnhub_client_instance)
    monkeypatch.setattr(fetch, "TwelveDataClient", lambda *args, **kwargs: stub_twelvedata_client_instance)


@pytest.fixture
def stub_finnhub_client(stub_finnhub_client_instance: _StubFinnhubClient) -> _StubFinnhubClient:
    return stub_finnhub_client_instance


@pytest.fixture
def stub_twelvedata_client(stub_twelvedata_client_instance: _StubTwelveDataClient) -> _StubTwelveDataClient:
    return stub_twelvedata_client_instance
