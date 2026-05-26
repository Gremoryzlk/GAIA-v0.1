"""TitanClient — HTTP/WebSocket клиент для Titan серверов.

Протокол:
- HTTP для быстрых задач (compute, короткий scraping)
- WebSocket для долгих задач (ML training, большой scraping)
Аутентификация: JWT (1 час) + HMAC подпись каждого запроса.
Таймаут: 240 секунд.
"""

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_SEC = 240
_WS_TIMEOUT_SEC = 240
_JWT_EXPIRY_SEC = 3600


def _build_jwt(server_secret: str, agent_id: str, expiry_sec: int = _JWT_EXPIRY_SEC) -> str:
    """Строит минимальный JWT (header.payload.sig) без внешних зависимостей."""
    import base64

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "sub": agent_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + expiry_sec,
        "jti": uuid4().hex,
    }).encode())
    signing_input = f"{header}.{payload}"
    sig = _b64url(
        hmac.new(
            server_secret.encode(),
            signing_input.encode(),
            hashlib.sha256,
        ).digest()
    )
    return f"{signing_input}.{sig}"


def _hmac_sign(secret: str, body: bytes) -> str:
    """HMAC-SHA256 подпись тела запроса."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class TitanClient:
    """Клиент для одного Titan сервера.

    Инстанциируется через TitanCluster — не создавать напрямую.
    """

    def __init__(
        self,
        server_url: str,
        server_secret: str,
        agent_id: str,
        task_affinity: list[str] | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self._secret = server_secret
        self._agent_id = agent_id
        self.task_affinity: list[str] = task_affinity or []
        self._jwt_token: str = ""
        self._jwt_expires_at: float = 0.0

    # ─── JWT management ───────────────────────────────────────────────────────

    def _get_token(self) -> str:
        """Возвращает валидный JWT, обновляет если истёк."""
        if time.time() >= self._jwt_expires_at - 60:
            self._jwt_token = _build_jwt(self._secret, self._agent_id)
            self._jwt_expires_at = time.time() + _JWT_EXPIRY_SEC
            logger.debug("TitanClient: JWT refreshed for %s", self.server_url)
        return self._jwt_token

    def _auth_headers(self, body: bytes) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "X-GAIA-HMAC": _hmac_sign(self._secret, body),
            "X-GAIA-Agent": self._agent_id,
            "Content-Type": "application/json",
        }

    # ─── HTTP transport ───────────────────────────────────────────────────────

    def http_dispatch(
        self,
        task_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Выполняет задачу через HTTP POST /dispatch.

        Используется для быстрых задач (compute, короткий scraping).
        Возвращает результат или raises RuntimeError при ошибке.
        """
        try:
            import urllib.request
            import urllib.error

            body = json.dumps({
                "task_id": uuid4().hex,
                "task_type": task_type,
                "payload": payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }).encode()

            req = urllib.request.Request(
                url=f"{self.server_url}/dispatch",
                data=body,
                headers=self._auth_headers(body),
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
                result = json.loads(resp.read().decode())
                logger.info(
                    "TitanClient HTTP: task_type=%s server=%s status=ok",
                    task_type, self.server_url,
                )
                return result

        except Exception as e:
            logger.error(
                "TitanClient HTTP error: task_type=%s server=%s error=%s",
                task_type, self.server_url, e,
            )
            raise RuntimeError(f"Titan HTTP dispatch failed: {e}") from e

    # ─── WebSocket transport ──────────────────────────────────────────────────

    def ws_dispatch(
        self,
        task_type: str,
        payload: Dict[str, Any],
        on_checkpoint: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Выполняет долгую задачу через WebSocket /ws/dispatch.

        Используется для scraping, ML training.
        on_checkpoint вызывается на каждый промежуточный чекпоинт.
        """
        try:
            import websocket  # websocket-client

            ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
            ws_url = f"{ws_url}/ws/dispatch"

            body_dict = {
                "task_id": uuid4().hex,
                "task_type": task_type,
                "payload": payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            body_bytes = json.dumps(body_dict).encode()
            headers = self._auth_headers(body_bytes)

            result_container: Dict[str, Any] = {}
            error_container: Dict[str, str] = {}

            def _on_message(ws, message: str) -> None:  # noqa: ANN001
                try:
                    msg = json.loads(message)
                    msg_type = msg.get("type")
                    if msg_type == "checkpoint":
                        logger.debug(
                            "TitanClient WS checkpoint: task_type=%s step=%s",
                            task_type, msg.get("step"),
                        )
                        if on_checkpoint:
                            on_checkpoint(msg)
                    elif msg_type == "result":
                        result_container.update(msg.get("data", {}))
                        ws.close()
                    elif msg_type == "error":
                        error_container["msg"] = msg.get("message", "unknown error")
                        ws.close()
                except Exception as e:
                    error_container["msg"] = str(e)
                    ws.close()

            def _on_error(ws, error) -> None:  # noqa: ANN001
                error_container["msg"] = str(error)

            ws = websocket.WebSocketApp(
                ws_url,
                header=[f"{k}: {v}" for k, v in headers.items()],
                on_message=_on_message,
                on_error=_on_error,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)

            # Отправляем задачу после соединения — через reconnect
            # (websocket-client не поддерживает send до on_open, поэтому переоткрываем)
            result_container.clear()
            error_container.clear()

            import threading

            connected = threading.Event()
            done = threading.Event()

            def _on_open(ws) -> None:  # noqa: ANN001
                connected.set()
                ws.send(json.dumps(body_dict))

            ws2 = websocket.WebSocketApp(
                ws_url,
                header=[f"{k}: {v}" for k, v in headers.items()],
                on_open=_on_open,
                on_message=_on_message,
                on_error=_on_error,
                on_close=lambda ws, c, m: done.set(),
            )

            t = threading.Thread(
                target=ws2.run_forever,
                kwargs={"ping_interval": 30, "ping_timeout": 10},
                daemon=True,
            )
            t.start()
            done.wait(timeout=_WS_TIMEOUT_SEC)

            if error_container:
                raise RuntimeError(f"Titan WS error: {error_container['msg']}")
            if not result_container:
                raise RuntimeError("Titan WS timeout — no result received")

            logger.info(
                "TitanClient WS: task_type=%s server=%s status=ok",
                task_type, self.server_url,
            )
            return result_container

        except ImportError:
            logger.warning(
                "websocket-client not installed — falling back to HTTP for %s",
                task_type,
            )
            return self.http_dispatch(task_type, payload)
        except Exception as e:
            logger.error(
                "TitanClient WS error: task_type=%s server=%s error=%s",
                task_type, self.server_url, e,
            )
            raise RuntimeError(f"Titan WS dispatch failed: {e}") from e

    def health_check(self) -> bool:
        """Ping GET /health. Возвращает True если сервер доступен."""
        try:
            import urllib.request
            token = self._get_token()
            req = urllib.request.Request(
                url=f"{self.server_url}/health",
                headers={"Authorization": f"Bearer {token}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
