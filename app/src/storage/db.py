"""Database utilities for the storage layer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import quote_plus

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine as sa_create_engine

from app.src.config import get_config


_ENGINE: Engine | None = None


def create_engine() -> Engine:
    """Create (or return a cached) SQLAlchemy engine using MySQL credentials."""

    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    config = get_config()
    user = quote_plus(config.mysql_user)
    password = quote_plus(config.mysql_password)
    host = config.mysql_host
    port = config.mysql_port
    database = config.mysql_db

    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    _ENGINE = sa_create_engine(url, pool_pre_ping=True, future=True)
    return _ENGINE


def ensure_schema(engine: Engine | None = None) -> None:
    """Create the database schema if it does not already exist."""

    engine = engine or create_engine()
    schema_path = Path(__file__).with_name("schema.sql")
    schema_sql = schema_path.read_text(encoding="utf-8")

    statements = [stmt.strip() for stmt in schema_sql.split(";")]
    with engine.begin() as connection:
        for statement in statements:
            if statement:
                connection.exec_driver_sql(statement)


def upsert_symbols(
    rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    engine: Engine | None = None,
) -> None:
    """Insert or update symbol metadata."""

    _execute_many(
        """
        INSERT INTO symbols (
            symbol,
            exchange,
            currency,
            name,
            sector,
            is_active,
            first_seen,
            last_seen
        ) VALUES (
            :symbol,
            :exchange,
            :currency,
            :name,
            :sector,
            :is_active,
            :first_seen,
            :last_seen
        )
        ON DUPLICATE KEY UPDATE
            exchange = VALUES(exchange),
            currency = VALUES(currency),
            name = VALUES(name),
            sector = VALUES(sector),
            is_active = VALUES(is_active),
            first_seen = VALUES(first_seen),
            last_seen = VALUES(last_seen)
        """,
        rows,
        engine,
    )


def insert_daily_metrics(
    rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    engine: Engine | None = None,
) -> None:
    """Insert or update daily metrics data."""

    _execute_many(
        """
        INSERT INTO daily_metrics (
            symbol,
            ds,
            price,
            vol,
            pe_ttm,
            ps_ttm,
            ev_ebit,
            fcf_yield,
            roe,
            net_debt_ebitda,
            margin_trend_5y,
            sma_100,
            sma_200,
            atr_14,
            earnings_within_48h
        ) VALUES (
            :symbol,
            :ds,
            :price,
            :vol,
            :pe_ttm,
            :ps_ttm,
            :ev_ebit,
            :fcf_yield,
            :roe,
            :net_debt_ebitda,
            :margin_trend_5y,
            :sma_100,
            :sma_200,
            :atr_14,
            :earnings_within_48h
        )
        ON DUPLICATE KEY UPDATE
            price = VALUES(price),
            vol = VALUES(vol),
            pe_ttm = VALUES(pe_ttm),
            ps_ttm = VALUES(ps_ttm),
            ev_ebit = VALUES(ev_ebit),
            fcf_yield = VALUES(fcf_yield),
            roe = VALUES(roe),
            net_debt_ebitda = VALUES(net_debt_ebitda),
            margin_trend_5y = VALUES(margin_trend_5y),
            sma_100 = VALUES(sma_100),
            sma_200 = VALUES(sma_200),
            atr_14 = VALUES(atr_14),
            earnings_within_48h = VALUES(earnings_within_48h)
        """,
        rows,
        engine,
    )


def insert_valuations(
    rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    engine: Engine | None = None,
) -> None:
    """Insert or update valuation data."""

    _execute_many(
        """
        INSERT INTO valuations (
            symbol,
            ds,
            fv_dcf,
            dcf_upside,
            multiple_score,
            quality_score,
            momentum_score,
            composite_score
        ) VALUES (
            :symbol,
            :ds,
            :fv_dcf,
            :dcf_upside,
            :multiple_score,
            :quality_score,
            :momentum_score,
            :composite_score
        )
        ON DUPLICATE KEY UPDATE
            fv_dcf = VALUES(fv_dcf),
            dcf_upside = VALUES(dcf_upside),
            multiple_score = VALUES(multiple_score),
            quality_score = VALUES(quality_score),
            momentum_score = VALUES(momentum_score),
            composite_score = VALUES(composite_score)
        """,
        rows,
        engine,
    )


def insert_picks(
    rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    engine: Engine | None = None,
) -> None:
    """Insert or update pick recommendations."""

    _execute_many(
        """
        INSERT INTO picks (
            ds,
            rank_order,
            symbol,
            target_price,
            stop_loss,
            notes_json,
            sent_telegram
        ) VALUES (
            :ds,
            :rank_order,
            :symbol,
            :target_price,
            :stop_loss,
            :notes_json,
            :sent_telegram
        )
        ON DUPLICATE KEY UPDATE
            symbol = VALUES(symbol),
            target_price = VALUES(target_price),
            stop_loss = VALUES(stop_loss),
            notes_json = VALUES(notes_json),
            sent_telegram = VALUES(sent_telegram)
        """,
        rows,
        engine,
    )


def _execute_many(
    sql: str,
    rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    engine: Engine | None,
) -> None:
    """Execute a parameterized statement for a batch of rows."""

    payload = list(rows)
    if not payload:
        return

    engine = engine or create_engine()
    statement = text(sql)
    with engine.begin() as connection:
        connection.execute(statement, payload)


__all__ = [
    "create_engine",
    "ensure_schema",
    "insert_daily_metrics",
    "insert_picks",
    "insert_valuations",
    "upsert_symbols",
]

