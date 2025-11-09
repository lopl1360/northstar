import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from data import cache  # noqa: E402


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str) -> None:
        self.store[key] = value
        self.expirations.pop(key, None)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.set(key, value)
        self.expirations[key] = ttl

    def incr(self, key: str) -> int:
        current = int(self.store.get(key, "0")) + 1
        self.store[key] = str(current)
        return current

    def decr(self, key: str) -> int:
        current = int(self.store.get(key, "0")) - 1
        if current <= 0:
            self.delete(key)
            return 0
        self.store[key] = str(current)
        return current

    def expire(self, key: str, ttl: int) -> None:
        self.expirations[key] = ttl

    def ttl(self, key: str) -> int:
        return self.expirations.get(key, -1)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)
        self.expirations.pop(key, None)


@pytest.fixture(autouse=True)
def reset_client(monkeypatch):
    monkeypatch.setattr(cache, "_REDIS_CLIENT", None)


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(cache, "get_client", lambda: client)
    return client


def test_set_and_get_json_roundtrip(fake_client):
    payload = {"alpha": 1, "beta": [1, 2, 3]}

    cache.set_json("payload", payload)

    assert fake_client.store["payload"] == json.dumps(payload)
    assert cache.get_json("payload") == payload


def test_set_json_with_ttl(fake_client):
    payload = {"expiring": True}

    cache.set_json("payload", payload, ttl_seconds=30)

    assert fake_client.store["payload"] == json.dumps(payload)
    assert fake_client.expirations["payload"] == 30


def test_rate_limiter_waits_when_limit_reached(fake_client, monkeypatch):
    sleep_calls: list[int] = []

    def fake_sleep(seconds: int) -> None:
        sleep_calls.append(seconds)
        fake_client.delete(cache._rate_limit_key("api"))

    monkeypatch.setattr(cache.time, "sleep", fake_sleep)

    with cache.rate_limiter("api", max_calls_per_minute=2):
        pass
    with cache.rate_limiter("api", max_calls_per_minute=2):
        pass
    with cache.rate_limiter("api", max_calls_per_minute=2):
        pass

    assert sleep_calls == [cache._RATE_LIMIT_WINDOW_SECONDS]


def test_rate_limiter_rejects_non_positive_limits(fake_client):
    with pytest.raises(ValueError):
        with cache.rate_limiter("api", 0):
            pass
