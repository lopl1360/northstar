import pytest

from src.models.valuation import mini_dcf


def manual_mini_dcf(price, rev_growth_ttm, fcfps_ttm, discount_floor=0.08, terminal_g=0.03):
    if price <= 0 or fcfps_ttm <= 0:
        return 0.0, 0.0

    year1_growth = max(-0.05, min(rev_growth_ttm, 0.20))
    years = 5
    step = (terminal_g - year1_growth) / (years - 1)
    growth_path = [year1_growth + step * i for i in range(years)]

    discount_rate = max(discount_floor, 0.10)
    if terminal_g >= discount_rate:
        terminal_g = min(terminal_g, discount_rate - 1e-6)

    fcf = fcfps_ttm
    projections = []
    for growth in growth_path:
        fcf *= 1.0 + growth
        projections.append(fcf)

    pv_fcf = 0.0
    for idx, cash_flow in enumerate(projections, start=1):
        pv_fcf += cash_flow / ((1.0 + discount_rate) ** idx)

    terminal_value = projections[-1] * (1.0 + terminal_g) / (discount_rate - terminal_g)
    pv_terminal = terminal_value / ((1.0 + discount_rate) ** years)

    fair_value = pv_fcf + pv_terminal
    return fair_value, (fair_value - price) / price


def test_mini_dcf_clamps_growth_rates():
    price = 50.0
    rev_growth_ttm = 0.8  # should clamp to 0.20
    fcfps_ttm = 3.0

    expected = manual_mini_dcf(price, 0.20, fcfps_ttm)
    fair_value, upside = mini_dcf(price, rev_growth_ttm, fcfps_ttm)

    assert fair_value == pytest.approx(expected[0], rel=1e-6)
    assert upside == pytest.approx(expected[1], rel=1e-6)


def test_mini_dcf_handles_negative_growth_floor():
    price = 40.0
    rev_growth_ttm = -0.20  # should clamp to -0.05
    fcfps_ttm = 2.5

    expected = manual_mini_dcf(price, -0.05, fcfps_ttm)
    fair_value, upside = mini_dcf(price, rev_growth_ttm, fcfps_ttm)

    assert fair_value == pytest.approx(expected[0], rel=1e-6)
    assert upside == pytest.approx(expected[1], rel=1e-6)


def test_mini_dcf_adjusts_terminal_growth_when_needed():
    price = 45.0
    rev_growth_ttm = 0.05
    fcfps_ttm = 4.0
    discount_floor = 0.12
    terminal_g = 0.15

    expected = manual_mini_dcf(price, rev_growth_ttm, fcfps_ttm, discount_floor, terminal_g)
    fair_value, upside = mini_dcf(price, rev_growth_ttm, fcfps_ttm, discount_floor, terminal_g)

    assert fair_value == pytest.approx(expected[0], rel=1e-6)
    assert upside == pytest.approx(expected[1], rel=1e-6)


def test_mini_dcf_returns_zero_for_invalid_inputs():
    assert mini_dcf(0.0, 0.1, 2.0) == (0.0, 0.0)
    assert mini_dcf(10.0, 0.1, 0.0) == (0.0, 0.0)
    assert mini_dcf(-5.0, 0.1, 2.0) == (0.0, 0.0)
