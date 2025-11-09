"""HTTP clients for interacting with third-party market data providers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from requests import Response
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .cache import rate_limiter


class DataProviderError(RuntimeError):
    """Base exception for data provider client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ClientSideError(DataProviderError):
    """Raised when the upstream API returns a 4xx response."""


class ServerSideError(DataProviderError):
    """Raised when the upstream API returns a 5xx response."""


def _extract_error_message(response: Response) -> str:
    """Attempt to extract a helpful error message from an HTTP response."""

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if value:
                return str(value)
        return json.dumps(payload)

    text = response.text.strip()
    if text:
        return text
    return f"HTTP {response.status_code}"


def _safe_json(response: Response) -> Any:
    """Return the JSON payload from ``response`` or raise an informative error."""

    try:
        return response.json()
    except ValueError as exc:
        raise DataProviderError("Invalid JSON payload received from upstream service") from exc


@dataclass
class _RequestConfig:
    url: str
    params: Dict[str, Any]
    timeout: float
    rate_limit_key: str


class FinnhubClient:
    """Client for interacting with the Finnhub API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = "https://finnhub.io/api/v1",
        session: Optional[requests.Session] = None,
        fallback_client: Optional["TwelveDataClient"] = None,
        max_calls_per_minute: int = 60,
        request_timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.fallback_client = fallback_client
        self.max_calls_per_minute = max_calls_per_minute
        self.request_timeout = request_timeout

        self._retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
            retry=retry_if_exception_type((requests.RequestException, ServerSideError)),
        )

    # ------------------------------------------------------------------
    # Public API
    def get_symbols(self, exchange: str) -> Any:
        """Return the list of symbols traded on ``exchange``."""

        params = {"exchange": exchange}
        return self._request("/stock/symbol", params, rate_limit_key="symbols")

    def get_quote(self, symbol: str) -> Any:
        """Return the latest price quote for ``symbol``.

        When Finnhub fails, attempts a fallback request using TwelveData if
        available.
        """

        params = {"symbol": symbol}
        try:
            return self._request("/quote", params, rate_limit_key="quote")
        except DataProviderError:
            if self.fallback_client is not None:
                return self.fallback_client.get_quote(symbol)
            raise

    def get_basic_financials(self, symbol: str) -> Any:
        """Retrieve basic financial metrics for ``symbol``."""

        params = {"symbol": symbol, "metric": "all"}
        return self._request("/stock/metric", params, rate_limit_key="financials")

    def get_earnings_calendar(self, symbol: str) -> Any:
        """Fetch earnings calendar entries for ``symbol``."""

        params = {"symbol": symbol}
        return self._request("/calendar/earnings", params, rate_limit_key="earnings")

    # ------------------------------------------------------------------
    # Internal helpers
    def _request(self, path: str, params: Dict[str, Any], *, rate_limit_key: str) -> Any:
        if not self.api_key:
            raise DataProviderError("Finnhub API key is not configured")

        url = f"{self.base_url}{path}"
        merged_params = dict(params)
        merged_params["token"] = self.api_key

        config = _RequestConfig(
            url=url,
            params=merged_params,
            timeout=self.request_timeout,
            rate_limit_key=f"finnhub:{rate_limit_key}",
        )

        try:
            response = self._retrying.call(self._perform_request, config)
        except requests.RequestException as exc:
            raise DataProviderError("Error communicating with Finnhub") from exc
        return _safe_json(response)

    def _perform_request(self, config: _RequestConfig) -> Response:
        limiter = rate_limiter(config.rate_limit_key, self.max_calls_per_minute)
        with limiter:
            response = self.session.get(config.url, params=config.params, timeout=config.timeout)

        status = response.status_code
        if 400 <= status < 500:
            raise ClientSideError(_extract_error_message(response), status)
        if status >= 500:
            raise ServerSideError(_extract_error_message(response), status)
        return response


class TwelveDataClient:
    """Lightweight client for fetching quotes from TwelveData."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = "https://api.twelvedata.com",
        session: Optional[requests.Session] = None,
        max_calls_per_minute: int = 60,
        request_timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.max_calls_per_minute = max_calls_per_minute
        self.request_timeout = request_timeout

        self._retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
            retry=retry_if_exception_type((requests.RequestException, ServerSideError)),
        )

    def get_quote(self, symbol: str) -> Any:
        if not self.api_key:
            raise DataProviderError("TwelveData API key is not configured")

        url = f"{self.base_url}/quote"
        params = {"symbol": symbol, "apikey": self.api_key}
        config = _RequestConfig(
            url=url,
            params=params,
            timeout=self.request_timeout,
            rate_limit_key="twelvedata:quote",
        )

        try:
            response = self._retrying.call(self._perform_request, config)
        except requests.RequestException as exc:
            raise DataProviderError("Error communicating with TwelveData") from exc
        payload = _safe_json(response)
        if isinstance(payload, dict) and payload.get("status") == "error":
            raise ClientSideError(_extract_error_message(response), response.status_code)
        if isinstance(payload, dict) and payload.get("code") and payload.get("message"):
            raise ClientSideError(str(payload.get("message")), response.status_code)
        return payload

    def _perform_request(self, config: _RequestConfig) -> Response:
        limiter = rate_limiter(config.rate_limit_key, self.max_calls_per_minute)
        with limiter:
            response = self.session.get(config.url, params=config.params, timeout=config.timeout)

        status = response.status_code
        if 400 <= status < 500:
            raise ClientSideError(_extract_error_message(response), status)
        if status >= 500:
            raise ServerSideError(_extract_error_message(response), status)
        return response

