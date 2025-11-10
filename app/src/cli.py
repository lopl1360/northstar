"""Command line interface for ad-hoc calculations and integrations."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from .config import MissingEnvironmentVariableError, get_config
from .data.fetch import (
    ClientSideError,
    DataProviderError,
    FinnhubClient,
    TwelveDataClient,
)
from .features.multiples import compute_multiple_scores
from .features.quality import compute_quality_scores
from .features.technicals import atr14, sma
from .models.valuation import mini_dcf
from .notify.telegram import send_text_message
from .picks.selector import compute_targets
from .ranking.scorer import momentum_penalty, score as composite_score

LOGGER = logging.getLogger(__name__)


def _build_client(config) -> FinnhubClient:
    fallback = None
    if config.twelvedata_key:
        fallback = TwelveDataClient(api_key=config.twelvedata_key)
    return FinnhubClient(api_key=config.finnhub_token, fallback_client=fallback)


def _normalise_date(ds: Optional[str]) -> date:
    if not ds:
        return date.today()
    try:
        return datetime.strptime(ds, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date format '{ds}'. Use YYYY-MM-DD.") from exc


def _extract_metric(metrics: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _candles_to_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, Mapping) or payload.get("s") != "ok":
        raise ValueError("No price history available for the requested range")

    timestamps = payload.get("t") or []
    opens = payload.get("o") or []
    highs = payload.get("h") or []
    lows = payload.get("l") or []
    closes = payload.get("c") or []
    volumes = payload.get("v") or []

    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit="s"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    frame = frame.set_index("timestamp")
    frame = frame.sort_index()
    return frame


def _compute_drawdown(close: pd.Series, window: int = 21) -> float:
    recent = close.tail(window)
    if recent.empty:
        return math.nan
    peak = recent.max()
    last = recent.iloc[-1]
    if peak == 0:
        return 0.0
    return float((last - peak) / peak)


def _format_value(value: float | None, pattern: str) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return "N/A"
    return pattern.format(numeric)


def _format_table_line(row: Mapping[str, Any]) -> str:
    columns = [
        ("symbol", "{:<8}"),
        ("price", "{:>8.2f}"),
        ("target_price", "{:>8.2f}"),
        ("stop_loss", "{:>8.2f}"),
        ("dcf_upside_pct", "{:>11.0f}"),
        ("ev_ebit", "{:>8.1f}"),
        ("roe_pct", "{:>6.0f}"),
        ("sma_100", "{:>8.2f}"),
        ("sma_200", "{:>8.2f}"),
        ("composite_score", "{:>10.2f}"),
    ]

    formatted = []
    for key, pattern in columns:
        value = row.get(key)
        text = _format_value(value, pattern)
        formatted.append(text)
    return " ".join(formatted)


def _build_summary(row: Mapping[str, Any]) -> list[str]:
    bullets = []
    price = row.get("price")
    target = row.get("target_price")
    stop = row.get("stop_loss")
    target_ret = row.get("target_return")
    stop_ret = row.get("stop_return")
    dcf = row.get("dcf_upside")
    composite = row.get("composite_score")

    def fmt(value: float | None, pattern: str) -> str:
        return _format_value(value, pattern)

    target_pct = target_ret * 100 if target_ret is not None else None
    stop_pct = stop_ret * 100 if stop_ret is not None else None
    target_pct_text = fmt(target_pct, "{:+.1f}")
    stop_pct_text = fmt(stop_pct, "{:+.1f}")
    roe_pct = row.get("roe") * 100 if row.get("roe") is not None else None

    target_suffix = f" ({target_pct_text}%)" if target_pct_text != "N/A" else ""
    stop_suffix = f" ({stop_pct_text}%)" if stop_pct_text != "N/A" else ""

    bullets.append(
        f"- Price {fmt(price, '${:.2f}')} | Target {fmt(target, '${:.2f}')}{target_suffix}"
    )
    bullets.append(
        f"- Stop {fmt(stop, '${:.2f}')}{stop_suffix} | DCF Upside {fmt(dcf, '{:+.0%}')}"
    )
    roe_text = fmt(roe_pct, "{:.0f}")
    roe_segment = f"ROE {roe_text}%" if roe_text != "N/A" else "ROE N/A"

    bullets.append(
        f"- Composite score {fmt(composite, '{:.2f}')} | EV/EBIT {fmt(row.get('ev_ebit'), '{:.1f}')} | {roe_segment}"
    )
    return bullets


def _send_telegram(summary_lines: list[str]) -> None:
    message = "\n".join(summary_lines)
    message_id = send_text_message(message)
    print(f"Sent Telegram message_id={message_id}")


def _handle_provider_error(exc: DataProviderError, context: str) -> int:
    status = getattr(exc, "status_code", None)
    if status == 429:
        print("Rate limit reached. Please retry in a minute.")
        return 2
    print(f"{context}: {exc}")
    return 1


def _command_calc(args: argparse.Namespace) -> int:
    try:
        config = get_config()
    except MissingEnvironmentVariableError as exc:
        print(f"Missing required environment variable: {exc.name}")
        return 1

    client = _build_client(config)

    symbol = args.symbol.upper()
    run_date = _normalise_date(args.ds)
    end = datetime.combine(run_date, datetime.max.time())
    start = end - timedelta(days=400)

    try:
        financials = client.get_basic_financials(symbol)
        quote = client.get_quote(symbol)
        profile = client.get_company_profile(symbol)
        candles = client.get_stock_candles(symbol, start=start, end=end)
    except ClientSideError as exc:
        return _handle_provider_error(exc, "Data provider error")
    except DataProviderError as exc:
        return _handle_provider_error(exc, "Failed to retrieve market data")

    metrics: Mapping[str, Any] = {}
    if isinstance(financials, Mapping):
        metric_payload = financials.get("metric")
        if isinstance(metric_payload, Mapping):
            metrics = metric_payload

    if not isinstance(quote, Mapping):
        print("Unexpected quote payload received from Finnhub")
        return 1

    price = _extract_metric(quote, ["c", "pc"])
    if price is None:
        print(f"Quote for {symbol} is unavailable.")
        return 1

    volume = _extract_metric(quote, ["v", "volume"])
    sector = "Unknown"
    if isinstance(profile, Mapping):
        sector = str(profile.get("sector") or profile.get("finnhubIndustry") or "Unknown")

    try:
        history = _candles_to_frame(candles)
    except ValueError as exc:
        print(str(exc))
        return 1
    if len(history) < 200:
        print("Not enough historical data to compute technical indicators.")
        return 1

    close = history["close"].astype(float)
    high = history["high"].astype(float)
    low = history["low"].astype(float)

    try:
        sma_100 = float(sma(close, 100))
        sma_200 = float(sma(close, 200))
    except ValueError:
        print("Not enough data to compute moving averages.")
        return 1

    try:
        atr = float(atr14(high, low, close))
    except ValueError:
        print("Not enough data to compute ATR14.")
        return 1

    drawdown = _compute_drawdown(close)

    def metric(*keys: str) -> Optional[float]:
        return _extract_metric(metrics, keys)

    pe_ttm = metric("peTTM", "pe_ttm", "pe")
    ps_ttm = metric("psTTM", "priceToSalesRatioTTM", "ps_ttm")
    ev_ebit = metric("enterpriseValueEbit", "evToEbit", "ev_ebit")
    fcf_yield = metric("freeCashFlowYield", "fcfYield")
    roe = metric("roe", "returnOnEquityTTM", "roeTTM")
    free_cash_flow = metric("freeCashFlowTTM", "freeCashFlow")
    net_debt_ebitda = metric("netDebtEbitda", "netDebtToEBITDA")
    interest_coverage = metric("interestCoverageTTM", "interestCoverage")
    margin_trend = metric("operatingMarginTTM")
    margin_5y = metric("operatingMargin5Y")
    if margin_trend is not None and margin_5y is not None:
        margin_trend = margin_trend - margin_5y

    rev_growth_ttm = metric("revenueGrowthTTM", "revenueGrowth")
    fcfps_ttm = metric("freeCashFlowPerShareTTM", "freeCashFlowPerShare")

    if fcf_yield is None and fcfps_ttm and price:
        fcf_yield = fcfps_ttm / price

    volume_value = volume if volume is not None else math.nan

    record = {
        "symbol": symbol,
        "sector": sector,
        "price": price,
        "vol": volume_value,
        "pe_ttm": pe_ttm,
        "ps_ttm": ps_ttm,
        "ev_ebit": ev_ebit,
        "fcf_yield": fcf_yield,
        "roe": roe,
        "free_cash_flow": free_cash_flow,
        "net_debt_ebitda": net_debt_ebitda,
        "interest_coverage": interest_coverage,
        "margin_trend_5y": margin_trend,
        "rev_growth_ttm": rev_growth_ttm or 0.0,
        "fcfps_ttm": fcfps_ttm or 0.0,
        "sma_100": sma_100,
        "sma_200": sma_200,
        "ATR14": atr,
        "drawdown_1m": drawdown,
    }

    frame = pd.DataFrame([record]).set_index("symbol")
    frame["sma100"] = frame["sma_100"]
    frame["sma200"] = frame["sma_200"]

    quality = compute_quality_scores(
        frame[["roe", "free_cash_flow", "net_debt_ebitda", "interest_coverage", "margin_trend_5y"]]
    )
    multiples = compute_multiple_scores(
        frame[["pe_ttm", "ps_ttm", "ev_ebit", "fcf_yield", "sector"]]
    )
    frame["quality_score"] = quality
    frame["quality"] = quality
    frame["multiple_score"] = multiples
    frame["momentum_score"] = momentum_penalty(
        frame["price"], frame["sma100"], frame["sma200"], frame["drawdown_1m"]
    )

    fv, dcf_upside = mini_dcf(
        price=price,
        rev_growth_ttm=rev_growth_ttm or 0.0,
        fcfps_ttm=fcfps_ttm or 0.0,
    )
    frame["fv_dcf"] = fv
    frame["dcf_upside"] = dcf_upside

    composite = composite_score(frame)
    frame["composite_score"] = composite["composite_score"]

    targets = compute_targets(frame.reset_index())
    targets.index = frame.index
    frame = frame.join(targets)

    row = frame.iloc[0].to_dict()
    row.update(
        {
            "symbol": symbol,
            "target_price": row.get("target_price"),
            "stop_loss": row.get("stop_loss"),
            "target_return": row.get("target_return"),
            "stop_return": row.get("stop_return"),
            "dcf_upside": dcf_upside,
            "dcf_upside_pct": dcf_upside * 100 if dcf_upside is not None else None,
            "composite_score": row.get("composite_score"),
            "sma_100": sma_100,
            "sma_200": sma_200,
            "price": price,
            "ev_ebit": ev_ebit,
            "roe": roe,
            "roe_pct": roe * 100 if roe is not None else None,
        }
    )

    header = (
        "SYMBOL    PRICE    TARGET     STOP  DCF_UPSIDE  EV/EBIT    ROE    SMA100    SMA200  COMPOSITE"
    )
    print(header)
    print(_format_table_line(row))

    summary = [f"{symbol} snapshot"] + _build_summary(row)
    for line in summary:
        print(line)

    if args.verbose:
        payload = {
            "quote": quote,
            "metrics": metrics,
            "profile": profile,
            "row": row,
        }
        print(json.dumps(payload, indent=2, default=lambda o: float(o) if isinstance(o, pd.Series) else str(o)))

    if args.send_telegram:
        _send_telegram(summary)

    return 0


def _command_tg_test(args: argparse.Namespace) -> int:
    try:
        get_config()
    except MissingEnvironmentVariableError as exc:
        print(f"Missing required environment variable: {exc.name}")
        return 1

    text = args.text or "✅ Northstar Telegram test OK"
    try:
        message_id = send_text_message(text)
    except Exception as exc:  # pragma: no cover - network errors
        print(f"Failed to send Telegram message: {exc}")
        return 1

    print(message_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Northstar CLI")
    subparsers = parser.add_subparsers(dest="command")

    calc = subparsers.add_parser("calc", help="Run a one-off calculation for a ticker")
    calc.add_argument("--symbol", required=True, help="Ticker symbol to analyse")
    calc.add_argument("--ds", help="Date context in YYYY-MM-DD format")
    calc.add_argument(
        "--send-telegram",
        action="store_true",
        help="Send a Telegram summary after the calculation",
    )
    calc.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate metrics and payload details",
    )
    calc.set_defaults(func=_command_calc)

    tg_test = subparsers.add_parser("tg-test", help="Send a Telegram test message")
    tg_test.add_argument("--text", help="Custom text to send")
    tg_test.set_defaults(func=_command_tg_test)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
