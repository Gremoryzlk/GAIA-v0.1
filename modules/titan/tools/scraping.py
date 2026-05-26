"""Titan scraping tool — HTTP scraping с rate limiting и BS4 парсингом.

Используется когда Titan сервер недоступен (локальный fallback).
Основной режим: задача отправляется на Titan сервер через TitanCluster.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from modules.hephaestus.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_RATE_LIMIT_RPS = 2.0   # запросов в секунду
_DEFAULT_TIMEOUT_SEC = 30
_MAX_CONTENT_BYTES = 5 * 1024 * 1024  # 5MB max per page


class _RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / rps
        self._last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        wait_time = self._interval - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self._last_call = time.monotonic()


class ScrapingTool(BaseTool):
    """Локальный fallback scraping инструмент.

    Выполняет HTTP запросы с rate limiting.
    При наличии BeautifulSoup4 — парсит HTML структуру.
    """

    def __init__(self, rate_limit_rps: float = _DEFAULT_RATE_LIMIT_RPS) -> None:
        super().__init__(
            name="scraping",
            description="HTTP scraping with rate limiting and HTML parsing",
        )
        self._limiter = _RateLimiter(rate_limit_rps)

    def validate_params(self, params: Dict[str, Any]):
        url = params.get("url")
        if not url:
            return False, "url required"
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, "url must be http or https"
        return True, None

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        valid, err = self.validate_params(params)
        if not valid:
            return ToolResult(success=False, output="", error=err)

        url = params["url"]
        timeout = params.get("timeout", _DEFAULT_TIMEOUT_SEC)
        extract_text = params.get("extract_text", True)
        extract_links = params.get("extract_links", False)
        selector = params.get("selector")  # CSS selector для BS4

        self._limiter.wait()

        try:
            content, status_code = self._fetch(url, timeout)

            if not content:
                return ToolResult(
                    success=False, output="", error=f"Empty response from {url}"
                )

            result: Dict[str, Any] = {
                "url": url,
                "status_code": status_code,
                "content_length": len(content),
            }

            # Парсинг через BS4 если доступен
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, "html.parser")

                if selector:
                    elements = soup.select(selector)
                    result["selected_count"] = len(elements)
                    result["selected_text"] = [el.get_text(strip=True) for el in elements[:50]]

                if extract_text:
                    # Удаляем script/style
                    for tag in soup(["script", "style", "noscript"]):
                        tag.decompose()
                    result["text"] = soup.get_text(separator=" ", strip=True)[:50000]
                    result["title"] = (soup.title.string.strip() if soup.title else "")

                if extract_links:
                    links = [
                        {"href": a.get("href", ""), "text": a.get_text(strip=True)[:100]}
                        for a in soup.find_all("a", href=True)[:200]
                    ]
                    result["links"] = links

            except ImportError:
                # BS4 не установлен — возвращаем сырой HTML
                result["html"] = content[:10000]
                logger.debug("ScrapingTool: bs4 not available, returning raw HTML")

            return ToolResult(
                success=True,
                output=json.dumps(result, ensure_ascii=False),
                metadata={"url": url, "status_code": status_code},
            )

        except Exception as e:
            logger.error("ScrapingTool: error fetching %s: %s", url, e)
            return ToolResult(success=False, output="", error=str(e))

    def _fetch(self, url: str, timeout: int) -> tuple[str, int]:
        """Делает HTTP запрос. Пробует httpx → urllib fallback."""
        try:
            import httpx
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "GAIA/7.3 Titan"})
                content = resp.text[:_MAX_CONTENT_BYTES]
                return content, resp.status_code
        except ImportError:
            pass

        # Fallback: urllib
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GAIA/7.3 Titan"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(_MAX_CONTENT_BYTES)
            encoding = resp.headers.get_content_charset("utf-8")
            return raw.decode(encoding, errors="replace"), resp.status
