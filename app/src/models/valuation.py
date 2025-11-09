"""Simplified discounted cash flow valuation helpers."""

from __future__ import annotations

from typing import Tuple


def mini_dcf(
    price: float,
    rev_growth_ttm: float,
    fcfps_ttm: float,
    discount_floor: float = 0.08,
    terminal_g: float = 0.03,
) -> Tuple[float, float]:
    """Estimate a fair value using a very small DCF model.

    Args:
        price: Current share price.
        rev_growth_ttm: Trailing twelve month revenue growth (as decimal).
        fcfps_ttm: Trailing twelve month free cash flow per share.
        discount_floor: Minimum discount rate to use.
        terminal_g: Terminal growth assumption (as decimal).

    Returns:
        A tuple containing the estimated fair value and the upside
        relative to the given ``price``.
    """

    if price <= 0 or fcfps_ttm <= 0:
        return 0.0, 0.0

    # Clamp first year growth and fade linearly to the terminal rate by year 5.
    year1_growth = max(-0.05, min(rev_growth_ttm, 0.20))
    years = 5
    step = (terminal_g - year1_growth) / (years - 1)
    growth_path = [year1_growth + step * i for i in range(years)]

    discount_rate = max(discount_floor, 0.10)
    # Avoid invalid Gordon growth scenarios.
    if terminal_g >= discount_rate:
        terminal_g = min(terminal_g, discount_rate - 1e-6)

    # Project free cash flow per share for each year.
    projections = []
    fcf = fcfps_ttm
    for growth in growth_path:
        fcf *= 1.0 + growth
        projections.append(fcf)

    # Present value of five years of FCF per share.
    pv_fcf = 0.0
    for idx, cash_flow in enumerate(projections, start=1):
        pv_fcf += cash_flow / ((1.0 + discount_rate) ** idx)

    terminal_value = (
        projections[-1] * (1.0 + terminal_g) / (discount_rate - terminal_g)
    )
    pv_terminal = terminal_value / ((1.0 + discount_rate) ** years)

    fair_value = pv_fcf + pv_terminal
    dcf_upside = (fair_value - price) / price

    return fair_value, dcf_upside
