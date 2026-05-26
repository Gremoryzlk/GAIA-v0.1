"""GAIA v7.3 HMAC-signed logging module."""

import hashlib
import hmac
import logging
import os
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from shared.types import Event

_hmac_key: bytes = bytes(32)
_logger: Optional[logging.Logger] = None


class _HMACFormatter(logging.Formatter):
    def __init__(self, key: bytes) -> None:
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        self._key = key

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        sig = hmac.new(self._key, base.encode(), digestmod=hashlib.sha256).hexdigest()
        return f"{base} | sig={sig[:16]}"


def get_logger(
    name: str = "gaia",
    hmac_key_hex: Optional[str] = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    global _hmac_key, _logger

    # Приоритет: аргумент → GAIA_HMAC_KEY → GAIA_SECRET_KEY → нулевой ключ (dev only)
    resolved_key_hex = (
        hmac_key_hex
        or os.environ.get("GAIA_HMAC_KEY")
        or _secret_key_as_hex()
        or "00" * 32
    )

    if resolved_key_hex == "00" * 32:
        import warnings
        warnings.warn(
            "HMAC key is all zeros — set GAIA_HMAC_KEY or GAIA_SECRET_KEY in environment.",
            RuntimeWarning,
            stacklevel=2,
        )

    _hmac_key = bytes.fromhex(resolved_key_hex[:64])

    if _logger is not None and _logger.name == name:
        return _logger

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_HMACFormatter(_hmac_key))
        logger.addHandler(handler)
        logger.propagate = False

    _logger = logger
    return logger


def _secret_key_as_hex() -> Optional[str]:
    """Конвертирует GAIA_SECRET_KEY в hex для использования как HMAC-ключ."""
    secret = os.environ.get("GAIA_SECRET_KEY", "")
    if not secret:
        return None
    raw = secret.encode("utf-8")
    # SHA-256 от ключа → 32 байта → 64 hex символа
    return hashlib.sha256(raw).hexdigest()


def sign_event(event: "Event") -> str:
    event_data = f"{event.event_type}:{event.source}:{event.timestamp.isoformat()}"
    return hmac.new(_hmac_key, event_data.encode(), digestmod=hashlib.sha256).hexdigest()


def verify_event(event: "Event") -> bool:
    if event.signature is None:
        return False
    expected = sign_event(event)
    return hmac.compare_digest(event.signature, expected)


def debug(message: str, **kwargs: object) -> None:
    (_logger or logging.getLogger("gaia")).debug(message, **kwargs)


def info(message: str, **kwargs: object) -> None:
    (_logger or logging.getLogger("gaia")).info(message, **kwargs)


def warning(message: str, **kwargs: object) -> None:
    (_logger or logging.getLogger("gaia")).warning(message, **kwargs)


def error(message: str, **kwargs: object) -> None:
    (_logger or logging.getLogger("gaia")).error(message, **kwargs)


def critical(message: str, **kwargs: object) -> None:
    (_logger or logging.getLogger("gaia")).critical(message, **kwargs)