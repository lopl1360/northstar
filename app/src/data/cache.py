"""Helpers for interacting with Redis-backed caching primitives."""
from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any, Generator, Optional

import redis


_REDIS_CLIENT: Optional[redis.Redis] = None
_DEFAULT_REDIS_URL = "redis://localhost:6379/0"
_RATE_LIMIT_PREFIX = "rate_limit:"
_RATE_LIMIT_WINDOW_SECONDS = 60


def _get_redis_url() -> str:
    """Return the Redis connection URL, defaulting to localhost."""
    return os.getenv("REDIS_URL", _DEFAULT_REDIS_URL)


def get_client() -> redis.Redis:
    """Return a cached Redis client instance."""
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        _REDIS_CLIENT = redis.Redis.from_url(_get_redis_url(), decode_responses=True)
    return _REDIS_CLIENT


def get_json(key: str) -> Any:
    """Retrieve a JSON-serialized object from Redis.

    Args:
        key: The Redis key to retrieve.

    Returns:
        The decoded JSON value, or ``None`` when the key is absent.
    """
    value = get_client().get(key)
    if value is None:
        return None
    return json.loads(value)


def set_json(key: str, obj: Any, ttl_seconds: Optional[int] = None) -> None:
    """Store a JSON-serializable object in Redis.

    Args:
        key: Redis key under which the object should be stored.
        obj: JSON-serializable value to store.
        ttl_seconds: Optional expiration in seconds.
    """
    payload = json.dumps(obj)
    client = get_client()
    if ttl_seconds is None:
        client.set(key, payload)
    else:
        client.setex(key, ttl_seconds, payload)


def _rate_limit_key(key: str) -> str:
    return f"{_RATE_LIMIT_PREFIX}{key}"


@contextlib.contextmanager
def rate_limiter(key: str, max_calls_per_minute: int) -> Generator[None, None, None]:
    """Context manager enforcing a simple Redis-backed rate limit.

    The limiter tracks the number of entries within the previous 60 seconds and
    sleeps until the window resets when the limit has been reached.
    """
    if max_calls_per_minute <= 0:
        raise ValueError("max_calls_per_minute must be greater than zero")

    client = get_client()
    storage_key = _rate_limit_key(key)

    while True:
        count = client.incr(storage_key)
        if count == 1:
            client.expire(storage_key, _RATE_LIMIT_WINDOW_SECONDS)
        if count <= max_calls_per_minute:
            break
        client.decr(storage_key)
        ttl = client.ttl(storage_key)
        sleep_time = ttl if ttl and ttl > 0 else _RATE_LIMIT_WINDOW_SECONDS
        time.sleep(sleep_time)

    try:
        yield
    finally:
        # No cleanup necessary; the counter naturally expires.
        pass
