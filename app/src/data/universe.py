"""Helpers for constructing the daily investment universe.

The implementation pulls tradable symbols from Finnhub, enriches the entries
with quote and fundamental data, filters them according to the universe
criteria, applies a deterministic sector rotation and finally upserts the
result into the ``symbols`` table.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from app.src.config import get_config
from app.src.data.fetch import DataProviderError, FinnhubClient, TwelveDataClient
from app.src.storage import db

LOGGER = logging.getLogger(__name__)

EXCHANGES: Tuple[str, ...] = ("US", "TO", "V")


def _parse_rotation_sectors(raw: str) -> Tuple[set[str], List[set[str]]]:
    """Parse the rotation schedule configuration string.

    The configuration supports a handful of human-friendly formats:

    * JSON list of lists – e.g. ``[["Energy", "Utilities"], ["Materials"]]``.
    * JSON object containing ``core`` and ``rotation`` keys.
    * Delimited strings using ``|`` or ``;`` for groups and commas for sectors.
    * Plain comma separated strings which are treated as single-entry groups.
    """

    core: set[str] = set()
    rotation: List[set[str]] = []

    def _normalise_group(group: Iterable[str]) -> set[str]:
        return {str(item).strip() for item in group if str(item).strip()}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        core = _normalise_group(parsed.get("core", []))
        rotation = [
            grp
            for grp in (
                _normalise_group(group) for group in parsed.get("rotation", []) or []
            )
            if grp
        ]
    elif isinstance(parsed, list):
        rotation = [grp for grp in (_normalise_group(group) for group in parsed) if grp]
    else:
        groups = [segment.strip() for segment in re.split(r"[|;\n]", raw) if segment.strip()]
        for group in groups:
            sector_group = _normalise_group(group.split(","))
            if sector_group:
                rotation.append(sector_group)

        if not rotation:
            # Treat a simple comma separated list as individual one-element groups.
            tokens = _normalise_group(raw.split(","))
            rotation = [{token} for token in tokens]

    rotation = [group for group in rotation if group]
    return core, rotation


def _build_client() -> FinnhubClient:
    """Instantiate the Finnhub client with optional TwelveData fallback."""

    config = get_config()
    fallback = None
    if config.twelvedata_key:
        fallback = TwelveDataClient(api_key=config.twelvedata_key)
    return FinnhubClient(api_key=config.finnhub_token, fallback_client=fallback)


def _normalise_symbol(exchange: str, payload: Mapping[str, object]) -> Optional[Dict[str, object]]:
    """Normalise Finnhub's symbol payload into the structure we need."""

    symbol = str(payload.get("symbol") or payload.get("displaySymbol") or "").strip()
    if not symbol:
        return None

    name = str(payload.get("description") or "").strip() or symbol
    currency = str(payload.get("currency") or "").strip() or "USD"

    return {
        "symbol": symbol,
        "exchange": exchange,
        "currency": currency,
        "name": name,
    }


def _safe_metric_value(metrics: Mapping[str, object], *keys: str) -> Optional[float]:
    """Retrieve the first present numeric metric from ``metrics``."""

    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _enrich_symbol(
    client: FinnhubClient,
    symbol: str,
) -> Optional[Dict[str, object]]:
    """Fetch financial metrics and quotes for ``symbol``.

    Returns ``None`` when the necessary data points are missing.
    """

    try:
        financials = client.get_basic_financials(symbol)
        quote = client.get_quote(symbol)
    except DataProviderError as exc:
        LOGGER.warning("Skipping %s due to data provider error: %s", symbol, exc)
        return None

    metrics: Mapping[str, object] = {}
    if isinstance(financials, Mapping):
        metric_payload = financials.get("metric")
        if isinstance(metric_payload, Mapping):
            metrics = metric_payload

    price = _safe_metric_value(quote if isinstance(quote, Mapping) else {}, "c")
    if price is None:
        return None

    market_cap = _safe_metric_value(metrics, "marketCapitalization", "marketcap")
    if market_cap is None:
        return None

    average_volume = _safe_metric_value(
        metrics,
        "10DayAverageTradingVolume",
        "3MonthAverageTradingVolume",
        "52WeekAverageTradingVolume",
        "avgVolume",
    )
    if average_volume is None:
        return None

    avg_dollar_volume = price * average_volume
    sector = str(metrics.get("sector") or "").strip() or "Unknown"

    return {
        "price": price,
        "marketCap": market_cap,
        "avgDollarVol": avg_dollar_volume,
        "sector": sector,
    }


def _merge_records(
    base: MutableMapping[str, object],
    enrichment: Mapping[str, object],
) -> MutableMapping[str, object]:
    """Merge the enrichment dictionary into the base symbol record."""

    base.update(enrichment)
    return base


def _apply_filters(symbol: Mapping[str, object]) -> bool:
    """Return ``True`` when ``symbol`` passes the universe requirements."""

    if symbol.get("price", 0) is None or symbol.get("avgDollarVol", 0) is None:
        return False
    if symbol.get("marketCap", 0) is None:
        return False

    try:
        price = float(symbol.get("price", 0))
        avg_dollar_vol = float(symbol.get("avgDollarVol", 0))
        market_cap = float(symbol.get("marketCap", 0))
    except (TypeError, ValueError):
        return False

    if price < 2:
        return False
    if avg_dollar_vol < 300_000:
        return False
    if market_cap < 150_000_000:
        return False
    return True


def _select_rotation_groups(
    symbols: Sequence[Mapping[str, object]],
    *,
    weekday: int,
    core_sectors: set[str],
    rotation_groups: Sequence[set[str]],
) -> List[Mapping[str, object]]:
    """Return the subset of ``symbols`` that belong to today's rotation."""

    if not symbols:
        return []

    if core_sectors:
        core = [symbol for symbol in symbols if str(symbol.get("sector")) in core_sectors]
    else:
        core = []

    if rotation_groups:
        todays_group = rotation_groups[weekday % len(rotation_groups)]
    else:
        todays_group = set()

    rotating = [
        symbol
        for symbol in symbols
        if todays_group and str(symbol.get("sector")) in todays_group
    ]

    if not core_sectors and not todays_group:
        # No rotation schedule defined; return the full set.
        return list(symbols)

    seen: set[Tuple[str, str]] = set()
    ordered: List[Mapping[str, object]] = []

    for bucket in (core, rotating):
        for symbol in bucket:
            key = (str(symbol.get("symbol")), str(symbol.get("exchange")))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(symbol)

    return ordered


def _prepare_rows(
    symbols: Iterable[Mapping[str, object]],
    *,
    today: date,
) -> List[Dict[str, object]]:
    """Prepare rows suitable for ``db.upsert_symbols``."""

    today_iso = today.isoformat()
    rows: List[Dict[str, object]] = []
    for symbol in symbols:
        rows.append(
            {
                "symbol": symbol.get("symbol"),
                "exchange": symbol.get("exchange"),
                "currency": symbol.get("currency"),
                "name": symbol.get("name"),
                "sector": symbol.get("sector"),
                "is_active": True,
                "first_seen": today_iso,
                "last_seen": today_iso,
            }
        )
    return rows


def build_universe(today: date) -> List[dict]:
    """Build the daily investment universe for ``today``.

    Returns the list of symbol dictionaries enriched with sector information.
    """

    client = _build_client()
    config = get_config()
    core_sectors, rotation_groups = _parse_rotation_sectors(config.rotation_sectors)

    all_symbols: List[Dict[str, object]] = []

    for exchange in EXCHANGES:
        try:
            payload = client.get_symbols(exchange)
        except DataProviderError as exc:
            LOGGER.error("Failed to fetch symbols for %s: %s", exchange, exc)
            continue

        if not isinstance(payload, Sequence):
            continue

        for symbol_payload in payload:
            if not isinstance(symbol_payload, Mapping):
                continue
            normalised = _normalise_symbol(exchange, symbol_payload)
            if not normalised:
                continue

            enriched = _enrich_symbol(client, normalised["symbol"])
            if not enriched:
                continue

            merged = _merge_records(normalised, enriched)
            if _apply_filters(merged):
                all_symbols.append(merged)

    filtered = _select_rotation_groups(
        all_symbols,
        weekday=today.weekday(),
        core_sectors=core_sectors,
        rotation_groups=rotation_groups,
    )

    if not filtered:
        LOGGER.warning("Universe build produced an empty symbol set for %s", today)
        return []

    db_rows = _prepare_rows(filtered, today=today)
    if db_rows:
        db.upsert_symbols(db_rows)

    return filtered


__all__ = ["build_universe"]
