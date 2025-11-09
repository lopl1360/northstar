"""Synthetic nightly job implementation used for the kata.

The job fabricates a realistic looking data set, computes technical and
fundamental features using the helper modules in the repository and writes the
results to CSV files.  The implementation is intentionally deterministic so
that unit tests can assert on the generated outputs.  Structured logging is
used extensively to expose the progress of the pipeline.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
import random
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

import pandas as pd
from threading import Lock

from ..features.multiples import compute_multiple_scores
from ..features.quality import compute_quality_scores
from ..features.technicals import atr14, sma
from ..models.valuation import mini_dcf
from ..notify.telegram import TelegramNotifier
from ..picks.selector import select
from ..ranking.scorer import momentum_penalty, score as composite_score

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(os.getenv("NIGHTLY_OUTPUT_DIR", "./nightly_output"))
DEFAULT_SYMBOLS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "UNH",
    "XOM",
    "CVX",
    "LLY",
    "MA",
    "V",
)
SECTORS = (
    "Technology",
    "Healthcare",
    "Energy",
    "Financials",
    "Industrials",
    "Consumer Discretionary",
    "Utilities",
)


@dataclass(frozen=True)
class SymbolUniverseEntry:
    """Representation of a security in today's universe."""

    symbol: str
    sector: str


class TokenBucketLimiter:
    """Simple rate limiter allowing ``rate`` operations per second."""

    def __init__(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError("rate must be greater than zero")
        self.rate = rate
        self._lock = Lock()
        self._tokens: deque[float] = deque()
        self._window = 1.0

    @contextmanager
    def limit(self) -> Iterator[None]:
        while True:
            now = time.perf_counter()
            with self._lock:
                while self._tokens and now - self._tokens[0] > self._window:
                    self._tokens.popleft()
                if len(self._tokens) < math.floor(self.rate):
                    self._tokens.append(now)
                    break
            time.sleep(1.0 / self.rate)
        try:
            yield
        finally:
            pass


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def ensure_schema(output_dir: Path) -> None:
    """Prepare the output directory that mimics a database schema."""

    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = time.perf_counter() - start
    LOGGER.info("event=ensure_schema status=ok duration=%.3f path=%s", duration, output_dir)


def build_universe(today: date) -> List[SymbolUniverseEntry]:
    """Return a deterministic but varied security universe for ``today``."""

    rng = random.Random(int(today.strftime("%Y%m%d")))
    entries: List[SymbolUniverseEntry] = []
    for symbol in DEFAULT_SYMBOLS:
        sector = rng.choice(SECTORS)
        entries.append(SymbolUniverseEntry(symbol=symbol, sector=sector))

    LOGGER.info("event=build_universe status=ok count=%d", len(entries))
    return entries


def _generate_price_history(rng: random.Random, *, days: int = 260) -> pd.DataFrame:
    price = rng.uniform(20.0, 250.0)
    close: List[float] = []
    high: List[float] = []
    low: List[float] = []
    volume: List[int] = []

    for _ in range(days):
        drift = rng.uniform(-0.04, 0.05)
        price = max(5.0, price * (1.0 + drift))
        candle_high = price * (1.0 + rng.uniform(0.0, 0.02))
        candle_low = price * (1.0 - rng.uniform(0.0, 0.02))
        vol = int(rng.uniform(500_000, 5_000_000))

        close.append(price)
        high.append(candle_high)
        low.append(candle_low)
        volume.append(vol)

    frame = pd.DataFrame({
        "close": pd.Series(close),
        "high": pd.Series(high),
        "low": pd.Series(low),
        "volume": pd.Series(volume),
    })
    return frame


def _synthetic_fundamentals(rng: random.Random) -> Dict[str, float]:
    return {
        "pe_ttm": round(rng.uniform(8, 35), 2),
        "ps_ttm": round(rng.uniform(1, 12), 2),
        "ev_ebit": round(rng.uniform(6, 30), 2),
        "fcf_yield": round(rng.uniform(0.01, 0.12), 4),
        "roe": round(rng.uniform(0.05, 0.35), 4),
        "free_cash_flow": round(rng.uniform(1.0, 12.0), 2),
        "net_debt_ebitda": round(rng.uniform(0.2, 3.0), 3),
        "interest_coverage": round(rng.uniform(4.0, 25.0), 3),
        "margin_trend_5y": round(rng.uniform(-0.02, 0.08), 4),
        "rev_growth_ttm": round(rng.uniform(-0.05, 0.25), 4),
        "fcfps_ttm": round(rng.uniform(1.0, 10.0), 3),
    }


def _compute_drawdown(close: pd.Series, *, window: int = 21) -> float:
    recent = close.tail(window)
    peak = recent.max()
    last = recent.iloc[-1]
    if peak == 0:
        return 0.0
    return float((last - peak) / peak)


def fetch_symbol_metrics(
    entry: SymbolUniverseEntry,
    *,
    limiter: TokenBucketLimiter,
    seed: int,
    max_retries: int = 3,
) -> Optional[Dict[str, object]]:
    """Generate a synthetic snapshot for ``entry`` with retry logging."""

    rng = random.Random(seed)
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        started = time.perf_counter()
        try:
            with limiter.limit():
                if rng.random() < 0.1 and attempts < max_retries:
                    raise RuntimeError("synthetic transient error")

                history = _generate_price_history(rng)
                close = history["close"]
                price = float(close.iloc[-1])
                fundamentals = _synthetic_fundamentals(rng)
                volume = int(history["volume"].iloc[-1])
                earnings_flag = rng.random() < 0.1

                drawdown = _compute_drawdown(close)
                sma100 = sma(close, 100)
                sma200 = sma(close, 200)
                atr = atr14(history["high"], history["low"], close)

                payload: Dict[str, object] = {
                    "symbol": entry.symbol,
                    "sector": entry.sector,
                    "price": price,
                    "vol": volume,
                    "sma_100": float(sma100),
                    "sma_200": float(sma200),
                    "ATR14": float(atr),
                    "drawdown_1m": float(drawdown),
                    "earnings_within_48h": earnings_flag,
                    "history": history,
                    **fundamentals,
                }

                duration = time.perf_counter() - started
                LOGGER.info(
                    "event=fetch_symbol status=success symbol=%s attempts=%d duration=%.3f",
                    entry.symbol,
                    attempts,
                    duration,
                )
                return payload
        except Exception as exc:  # pragma: no cover - exercised during runtime
            duration = time.perf_counter() - started
            LOGGER.warning(
                "event=fetch_symbol status=retry symbol=%s attempts=%d duration=%.3f error=%s",
                entry.symbol,
                attempts,
                duration,
                exc,
            )
            time.sleep(0.05 * attempts)

    LOGGER.error("event=fetch_symbol status=failed symbol=%s attempts=%d", entry.symbol, max_retries)
    return None


def _prepare_feature_frame(records: List[Dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame = frame.set_index("symbol")
    frame["sma100"] = frame["sma_100"]
    frame["sma200"] = frame["sma_200"]
    return frame


def _compute_sector_medians(df: pd.DataFrame) -> pd.Series:
    medians = df.groupby("sector")["ev_ebit"].median()
    return df["sector"].map(medians).rename("sector_median_ev_ebit")


def _build_daily_rows(df: pd.DataFrame, today: date) -> List[Dict[str, object]]:
    ds = today.isoformat()
    rows: List[Dict[str, object]] = []
    for symbol, row in df.iterrows():
        rows.append(
            {
                "symbol": symbol,
                "ds": ds,
                "price": float(row["price"]),
                "vol": int(row["vol"]),
                "pe_ttm": float(row["pe_ttm"]),
                "ps_ttm": float(row["ps_ttm"]),
                "ev_ebit": float(row["ev_ebit"]),
                "fcf_yield": float(row["fcf_yield"]),
                "roe": float(row["roe"]),
                "net_debt_ebitda": float(row["net_debt_ebitda"]),
                "margin_trend_5y": float(row["margin_trend_5y"]),
                "sma_100": float(row["sma_100"]),
                "sma_200": float(row["sma_200"]),
                "atr_14": float(row["ATR14"]),
                "earnings_within_48h": int(bool(row["earnings_within_48h"])),
            }
        )
    return rows


def _build_valuation_rows(df: pd.DataFrame, today: date) -> List[Dict[str, object]]:
    ds = today.isoformat()
    rows: List[Dict[str, object]] = []
    for symbol, row in df.iterrows():
        rows.append(
            {
                "symbol": symbol,
                "ds": ds,
                "fv_dcf": float(row["fv_dcf"]),
                "dcf_upside": float(row["dcf_upside"]),
                "multiple_score": float(row["multiple_score"]),
                "quality_score": float(row["quality_score"]),
                "momentum_score": float(row["momentum_score"]),
                "composite_score": float(row["composite_score"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def _send_telegram(today: date, picks: pd.DataFrame, notifier: TelegramNotifier) -> None:
    lines = [f"Nightly Picks – {today.isoformat()}"]
    if picks.empty:
        lines.append("No qualifying selections today.")
    else:
        for rank, (_, row) in enumerate(picks.iterrows(), start=1):
            symbol = row.get("symbol", "?")
            sector = row.get("sector", "Unknown")
            target = float(row.get("target_price", float("nan")))
            stop = float(row.get("stop_loss", float("nan")))
            composite = float(row.get("composite_score", float("nan")))
            lines.append(
                f"{rank}. {symbol} ({sector}) target={target:.2f} stop={stop:.2f} score={composite:.2f}"
            )
    notifier.send_message("\n".join(lines))


def main() -> None:
    _setup_logging()

    today = date.today()
    output_dir = DEFAULT_OUTPUT_DIR / today.strftime("%Y%m%d")
    ensure_schema(output_dir)

    universe = build_universe(today)
    if not universe:
        LOGGER.warning("event=nightly status=empty_universe")
        return

    limiter = TokenBucketLimiter(rate=5)

    fetched: List[Dict[str, object]] = []
    start_fetch = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                fetch_symbol_metrics,
                entry,
                limiter=limiter,
                seed=int(f"{today.strftime('%Y%m%d')}" + str(idx)),
            ): entry
            for idx, entry in enumerate(universe)
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                fetched.append(result)

    duration_fetch = time.perf_counter() - start_fetch
    LOGGER.info("event=fetch_batch status=done count=%d duration=%.3f", len(fetched), duration_fetch)

    if not fetched:
        LOGGER.warning("event=nightly status=no_data")
        return

    frame = _prepare_feature_frame(fetched)
    frame["price"] = frame["price"].astype(float)

    quality = compute_quality_scores(
        frame[["roe", "free_cash_flow", "net_debt_ebitda", "interest_coverage", "margin_trend_5y"]]
    )
    multiples = compute_multiple_scores(frame[["pe_ttm", "ps_ttm", "ev_ebit", "fcf_yield", "sector"]])
    frame["quality_score"] = quality
    frame["multiple_score"] = multiples
    frame["quality"] = quality

    frame["momentum_score"] = momentum_penalty(
        frame["price"], frame["sma100"], frame["sma200"], frame["drawdown_1m"]
    )

    fv_values: List[float] = []
    upside_values: List[float] = []
    for _, row in frame.iterrows():
        fv, upside = mini_dcf(
            price=float(row["price"]),
            rev_growth_ttm=float(row["rev_growth_ttm"]),
            fcfps_ttm=float(row["fcfps_ttm"]),
        )
        fv_values.append(fv)
        upside_values.append(upside)
    frame["fv_dcf"] = fv_values
    frame["dcf_upside"] = upside_values

    composite = composite_score(frame)
    frame["composite_score"] = composite["composite_score"]
    frame["sector_median_ev_ebit"] = _compute_sector_medians(frame)

    daily_rows = _build_daily_rows(frame, today)
    valuation_rows = _build_valuation_rows(frame, today)

    daily_path = output_dir / "daily_metrics.csv"
    valuations_path = output_dir / "valuations.csv"
    _write_csv(daily_path, daily_rows)
    _write_csv(valuations_path, valuation_rows)
    LOGGER.info("event=write_csv table=daily_metrics rows=%d path=%s", len(daily_rows), daily_path)
    LOGGER.info("event=write_csv table=valuations rows=%d path=%s", len(valuation_rows), valuations_path)

    picks_input = frame.reset_index()
    picks, notes = select(picks_input)
    picks_path = output_dir / "picks.csv"
    pick_rows: List[Dict[str, object]] = []
    for rank, (_, row) in enumerate(picks.iterrows(), start=1):
        pick_rows.append(
            {
                "ds": today.isoformat(),
                "rank_order": rank,
                "symbol": row.get("symbol"),
                "target_price": float(row.get("target_price", float("nan"))),
                "stop_loss": float(row.get("stop_loss", float("nan"))),
                "notes_json": json.dumps(notes),
                "sent_telegram": 0,
            }
        )

    _write_csv(picks_path, pick_rows)
    LOGGER.info("event=write_csv table=picks rows=%d path=%s", len(pick_rows), picks_path)

    notifier = TelegramNotifier(
        os.getenv("TELEGRAM_TOKEN"),
        os.getenv("TELEGRAM_CHAT_ID"),
        outbox_path=output_dir / "telegram_outbox.log",
    )
    _send_telegram(today, picks, notifier)
    LOGGER.info(
        "event=nightly status=completed universe=%d metrics=%d picks=%d",
        len(universe),
        len(daily_rows),
        len(pick_rows),
    )


if __name__ == "__main__":
    main()
