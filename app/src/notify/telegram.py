"""Utility helpers for sending Telegram notifications.

The real production environment would use the Telegram HTTP API.  For the
purposes of the kata we implement a small helper that behaves like the real
integration while remaining resilient when the credentials are missing or the
network is unavailable.  The helper will therefore never raise an exception –
instead it records the failure in the logs and writes the attempted message to
an "outbox" text file for later inspection.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelegramSettings:
    """Configuration needed to talk to the Telegram API."""

    token: Optional[str]
    chat_id: Optional[str]
    timeout: float = 5.0
    dry_run: bool = True
    outbox_path: Path | None = None


class TelegramNotifier:
    """Small utility for posting messages to Telegram.

    Parameters
    ----------
    token:
        Bot token provided by BotFather.
    chat_id:
        Identifier for the channel or chat to post to.
    timeout:
        Number of seconds to wait for the HTTP request to complete.
    dry_run:
        When ``True`` (the default) messages are not sent to Telegram.  This
        is useful for development environments where network access is
        intentionally disabled.  The attempted message is still persisted to
        the outbox file so that the user can review the payload.
    outbox_path:
        Optional location where attempted messages should be appended.  When
        omitted the helper simply logs the message.
    """

    _API_BASE_URL = "https://api.telegram.org"

    def __init__(
        self,
        token: Optional[str],
        chat_id: Optional[str],
        *,
        timeout: float = 5.0,
        dry_run: Optional[bool] = None,
        outbox_path: Path | None = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        env_dry_run = os.getenv("TELEGRAM_DRY_RUN")
        if dry_run is None:
            dry_run = env_dry_run is None or env_dry_run not in {"0", "false", "False"}

        self.settings = TelegramSettings(
            token=token,
            chat_id=chat_id,
            timeout=timeout,
            dry_run=dry_run,
            outbox_path=outbox_path,
        )
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # Public API
    def send_message(self, text: str) -> bool:
        """Attempt to send ``text`` to Telegram.

        Returns ``True`` when the message was accepted by Telegram, ``False``
        otherwise.  Any failures are logged and persisted to the outbox.
        """

        if not text:
            LOGGER.info("event=telegram_send status=skipped reason=empty_message")
            return False

        self._persist_message(text)

        if not self.settings.token or not self.settings.chat_id:
            LOGGER.info(
                "event=telegram_send status=skipped reason=missing_credentials length=%d",
                len(text),
            )
            return False

        if self.settings.dry_run:
            LOGGER.info(
                "event=telegram_send status=dry_run length=%d chat_id=%s",
                len(text),
                self.settings.chat_id,
            )
            return False

        url = f"{self._API_BASE_URL}/bot{self.settings.token}/sendMessage"
        payload = {"chat_id": self.settings.chat_id, "text": text}

        try:
            response = self._session.post(url, json=payload, timeout=self.settings.timeout)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("ok"):
                LOGGER.info(
                    "event=telegram_send status=success length=%d chat_id=%s",
                    len(text),
                    self.settings.chat_id,
                )
                return True
            LOGGER.warning(
                "event=telegram_send status=unexpected_response payload=%s",
                json.dumps(data, ensure_ascii=False),
            )
        except Exception as exc:  # pragma: no cover - exercised during runtime
            LOGGER.warning("event=telegram_send status=failed error=%s", exc)

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    def _persist_message(self, message: str) -> None:
        path = self.settings.outbox_path
        if path is None:
            return

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{timestamp} {message}\n")
        except OSError as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("event=telegram_outbox status=failed error=%s", exc)


__all__ = ["TelegramNotifier", "TelegramSettings"]
