"""Tests for the universe construction helpers."""

import pytest

from src.data import universe


def test_enrich_symbol_uses_stubbed_clients(stub_finnhub_client, synthetic_fundamentals):
    enriched = universe._enrich_symbol(stub_finnhub_client, "TEST")

    assert enriched is not None
    assert enriched["price"] == pytest.approx(synthetic_fundamentals["price"])
    assert enriched["marketCap"] == pytest.approx(
        synthetic_fundamentals["metric"]["marketCapitalization"]
    )
    expected_dollar_vol = (
        synthetic_fundamentals["price"] * synthetic_fundamentals["metric"]["avgVolume"]
    )
    assert enriched["avgDollarVol"] == pytest.approx(expected_dollar_vol)


def test_normalise_symbol_handles_missing_fields():
    payload = {"displaySymbol": "ABC", "description": "Alpha Beta", "currency": "CAD"}
    normalised = universe._normalise_symbol("TSX", payload)

    assert normalised == {
        "symbol": "ABC",
        "exchange": "TSX",
        "currency": "CAD",
        "name": "Alpha Beta",
    }


def test_apply_filters_enforces_thresholds():
    passing = {"price": 25.0, "avgDollarVol": 500_000.0, "marketCap": 300_000_000.0}
    failing = {"price": 1.0, "avgDollarVol": 50_000.0, "marketCap": 10_000_000.0}

    assert universe._apply_filters(passing) is True
    assert universe._apply_filters(failing) is False
