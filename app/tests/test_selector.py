import pandas as pd
import pytest
import numpy as np
from pathlib import Path

from src.picks.selector import _compute_targets


FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "synthetic_prices.csv"


def load_fixture_rows(count: int = 3) -> pd.DataFrame:
    frame = pd.read_csv(FIXTURE_PATH, parse_dates=["date"])
    return frame.iloc[:count].copy()


def test_target_price_uses_conservative_anchor():
    frame = load_fixture_rows()
    enriched = _compute_targets(frame)

    # Row 0: multiples reversion lower than DCF component
    first = enriched.iloc[0]
    price = 100
    dcf_component = price * (1 + 0.6 * 0.4)
    multiples_price = price * (12 / 10)
    assert first["target_price"] == pytest.approx(min(dcf_component, multiples_price))

    # Row 1: multiples reversion below DCF component, should pick multiples price
    second = enriched.iloc[1]
    price = 101
    dcf_component = price * (1 + 0.6 * 0.10)
    multiples_price = price * (15 / 18)
    assert second["target_price"] == pytest.approx(min(dcf_component, multiples_price))

    # Row 2: multiples missing, should fall back to DCF component
    third = enriched.iloc[2]
    price = 102
    dcf_component = price * (1 + 0.6 * 0.25)
    assert third["target_price"] == pytest.approx(dcf_component)


def test_stop_loss_selects_best_available_anchor():
    frame = load_fixture_rows()
    enriched = _compute_targets(frame)

    # Row 0: ATR stop should dominate
    first = enriched.iloc[0]
    atr_stop = 100 - 2 * 1.5
    support_stop = 92
    fixed_stop = 100 * 0.85
    assert first["stop_loss"] == pytest.approx(max(atr_stop, support_stop, fixed_stop))

    # Row 1: support level higher than ATR-derived stop
    second = enriched.iloc[1]
    atr_stop = 101 - 2 * 10
    support_stop = 96
    fixed_stop = 101 * 0.85
    assert second["stop_loss"] == pytest.approx(max(atr_stop, support_stop, fixed_stop))

    # Row 2: fallback to fixed percentage when other anchors missing
    third = enriched.iloc[2]
    atr_stop = 102 - 2 * 60
    support_stop = float("nan")
    fixed_stop = 102 * 0.85
    assert third["stop_loss"] == pytest.approx(fixed_stop)

    # Stop return should respect price relationship and avoid inf
    assert np.isfinite(first["stop_return"])
    assert np.isfinite(second["stop_return"])
    assert np.isfinite(third["stop_return"])
