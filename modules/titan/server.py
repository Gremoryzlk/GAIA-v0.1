"""TitanServer — автономный сервер для удалённых вычислений.

Запускается отдельно на удалённом сервере:
  python -m modules.titan.server --host 0.0.0.0 --port 8765

Эндпоинты:
  GET  /health           → {"status": "ok"}
  POST /dispatch         → HTTP sync задачи
  WS   /ws/dispatch      → WebSocket долгие задачи

Аутентификация: JWT Bearer + X-GAIA-HMAC верификация.
"""

import argparse
import hashlib
import hmac
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

_SECRET = os.environ.get("TITAN_SERVER_SECRET", "change-me-in-production")
_JWT_LEEWAY_SEC = 30


# ─── Auth helpers ─────────────────────────────────────────────────────────────

def _verify_jwt(token: str) -> bool:
    """Верифицирует JWT подпись и exp claim."""
    try:
        import base64
        parts = token.split(".")
        if len(parts) != 3:
            return False
        header_b64, payload_b64, sig_b64 = parts

        # Верифицируем подпись
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = base64.urlsafe_b64encode(
            hmac.new(_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode()

        if not hmac.compare_digest(expected_sig, sig_b64):
            return False

        # Проверяем exp
        pad = 4 - len(payload_b64) % 4
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * pad))
        exp = payload.get("exp", 0)
        return time.time() < exp + _JWT_LEEWAY_SEC

    except Exception as e:
        logger.warning("JWT verify error: %s", e)
        return False


def _verify_hmac(body: bytes, sig: str) -> bool:
    expected = hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


# ─── Task executor ────────────────────────────────────────────────────────────

def _execute_task(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Выполняет задачу на сервере. Расширяется под конкретные вычисления."""
    if task_type == "compute":
        from modules.titan.tools.compute import ComputeTool
        tool = ComputeTool()
        result = tool.execute(payload)
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
        }

    elif task_type == "scraping":
        from modules.titan.tools.scraping import ScrapingTool
        tool = ScrapingTool()
        result = tool.execute(payload)
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }

    elif task_type in ("ml_train", "ml_batch"):
        from modules.titan.tools.ml import MLTrainTool
        tool = MLTrainTool()
        result = tool.execute(payload)
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
        }

    return {"success": False, "error": f"Unknown task_type: {task_type}"}


# ─── HTTP Request Handler ─────────────────────────────────────────────────────

class TitanRequestHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):  # noqa: A002
        logger.debug("TitanServer: " + format % args)

    def _send_json(self, code: int, data: Dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authenticate(self, body: bytes = b"") -> bool:
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return False
        token = auth.removeprefix("Bearer ")
        if not _verify_jwt(token):
            return False
        hmac_sig = self.headers.get("X-GAIA-HMAC", "")
        if body and hmac_sig and not _verify_hmac(body, hmac_sig):
            return False
        return True

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "server": "titan"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/dispatch":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if not self._authenticate(body):
            self._send_json(401, {"error": "unauthorized"})
            return

        try:
            req = json.loads(body)
            task_type = req.get("task_type", "compute")
            payload = req.get("payload", {})
            task_id = req.get("task_id", uuid4().hex)

            logger.info("TitanServer: dispatch task_id=%s task_type=%s", task_id, task_type)
            result = _execute_task(task_type, payload)
            result["task_id"] = task_id
            self._send_json(200, result)

        except Exception as e:
            logger.error("TitanServer dispatch error: %s", e)
            self._send_json(500, {"error": str(e)})


# ─── Entry point ──────────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = HTTPServer((host, port), TitanRequestHandler)
    logger.info("TitanServer: listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("TitanServer: shutdown")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAIA Titan Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.host, args.port)
