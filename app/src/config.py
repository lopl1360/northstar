"""Application configuration helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


class MissingEnvironmentVariableError(RuntimeError):
    """Raised when a required environment variable is missing."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Environment variable '{name}' is required but was not set.")
        self.name = name


def _get_env(name: str, *, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Fetch an environment variable value.

    Parameters
    ----------
    name:
        Name of the environment variable to retrieve.
    default:
        Default value if the variable is not set and is not required.
    required:
        Whether the environment variable must be present.
    """

    value = os.environ.get(name)
    if value is not None:
        value = value.strip()

    if required and not value:
        raise MissingEnvironmentVariableError(name)

    if value:
        return value

    return default


@dataclass(frozen=True)
class Config:
    """Dataclass representing configuration for the application."""

    finnhub_token: str
    twelvedata_key: Optional[str]
    mysql_host: str
    mysql_port: int
    mysql_db: str
    mysql_user: str
    mysql_password: str
    redis_url: str
    telegram_token: str
    telegram_chat_id: str
    timezone: str
    universe_min_dollar_vol: int
    rotation_sectors: str


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and memoize the application configuration."""

    def require(name: str) -> str:
        value = _get_env(name, required=True)
        assert value is not None  # for type checkers
        return value

    mysql_port_str = require("MYSQL_PORT")
    try:
        mysql_port = int(mysql_port_str)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError("MYSQL_PORT must be an integer") from exc

    universe_min_dollar_vol_str = _get_env(
        "UNIVERSE_MIN_DOLLAR_VOL", default="300000"
    )
    try:
        universe_min_dollar_vol = int(universe_min_dollar_vol_str or 0)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError("UNIVERSE_MIN_DOLLAR_VOL must be an integer") from exc

    return Config(
        finnhub_token=require("FINNHUB_TOKEN"),
        twelvedata_key=_get_env("TWELVEDATA_KEY"),
        mysql_host=require("MYSQL_HOST"),
        mysql_port=mysql_port,
        mysql_db=require("MYSQL_DB"),
        mysql_user=require("MYSQL_USER"),
        mysql_password=require("MYSQL_PASSWORD"),
        redis_url=require("REDIS_URL"),
        telegram_token=require("TELEGRAM_TOKEN"),
        telegram_chat_id=require("TELEGRAM_CHAT_ID"),
        timezone=_get_env("TIMEZONE", default="America/Vancouver") or "America/Vancouver",
        universe_min_dollar_vol=universe_min_dollar_vol,
        rotation_sectors=require("ROTATION_SECTORS"),
    )


__all__ = ["Config", "get_config", "MissingEnvironmentVariableError"]
