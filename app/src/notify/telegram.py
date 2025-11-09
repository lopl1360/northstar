"""Telegram notification helpers."""

from __future__ import annotations

import logging
import math
import os
from datetime import date
import pandas as pd
from telegram import Bot
from telegram.error import TelegramError

LOGGER = logging.getLogger(__name__)


def _coerce_float(value: object) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _format_pick_lines(rank: int, row: pd.Series) -> tuple[str, str, str]:
    symbol = str(row.get("symbol", "?"))

    price = _coerce_float(row.get("price"))
    target = _coerce_float(row.get("target_price"))
    target_return = _coerce_float(row.get("target_return"))
    stop = _coerce_float(row.get("stop_loss"))
    stop_return = _coerce_float(row.get("stop_return"))
    dcf_upside = _coerce_float(row.get("dcf_upside"))
    ev_ebit = _coerce_float(row.get("ev_ebit"))
    sector_median = _coerce_float(row.get("sector_median_ev_ebit"))
    roe = _coerce_float(row.get("roe"))
    sma200 = _coerce_float(row.get("sma_200", row.get("sma200")))

    if not math.isnan(price) and not math.isnan(sma200):
        sma_flag = "✅ Above" if price >= sma200 else "⚠️ Below"
    else:
        sma_flag = "ℹ️ N/A"

    def fmt(value: float, pattern: str) -> str:
        return pattern.format(value) if not math.isnan(value) else "N/A"

    price_text = fmt(price, "{:.2f}")
    target_text = fmt(target, "{:.2f}")
    target_pct = target_return * 100 if not math.isnan(target_return) else math.nan
    target_pct_text = fmt(target_pct, "{:+.1f}")
    stop_text = fmt(stop, "{:.2f}")
    stop_pct = stop_return * 100 if not math.isnan(stop_return) else math.nan
    stop_pct_text = fmt(stop_pct, "{:.1f}")
    dcf_text = fmt(dcf_upside, "{:+.0%}")
    ev_ebit_text = fmt(ev_ebit, "{:.1f}")
    sector_median_text = fmt(sector_median, "{:.1f}")
    roe_pct = roe * 100 if not math.isnan(roe) else math.nan
    roe_text = fmt(roe_pct, "{:.0f}")

    line_one = f"{rank}) {symbol} | Price {price_text}"
    line_two = (
        f"   Target {target_text}  ({target_pct_text}%)   "
        f"Stop {stop_text} ({stop_pct_text}%)"
    )
    line_three = (
        f"   DCF {dcf_text} | EV/EBIT {ev_ebit_text} vs sector {sector_median_text} | "
        f"ROE {roe_text}% | SMA200 {sma_flag}"
    )

    return line_one, line_two, line_three


def _build_message(run_date: date, picks: pd.DataFrame | None) -> str:
    lines = [f"📈 Daily ideas — {run_date.isoformat()}"]

    if picks is None or picks.empty:
        lines.append("No ideas today.")
        return "\n".join(lines)

    for rank, (_, row) in enumerate(picks.iterrows(), start=1):
        lines.extend(_format_pick_lines(rank, row))

    return "\n".join(lines)


def send_daily_message(run_date: date, picks_df: pd.DataFrame | None) -> None:
    """Send the formatted daily message to Telegram."""

    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    bot = Bot(token=token)
    message = _build_message(run_date, picks_df)

    try:
        bot.send_message(chat_id=chat_id, text=message)
    except TelegramError as exc:  # pragma: no cover - relies on network failures
        LOGGER.error("event=telegram_send status=failed error=%s", exc)


__all__ = ["send_daily_message"]
