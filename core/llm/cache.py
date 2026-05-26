"""LLM Cache with SQLite backend — постоянное соединение."""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMCache:
    """Thread-safe LLM response cache с постоянным SQLite соединением."""

    def __init__(self, db_path: str = "data/llm_cache.db", default_ttl_hours: int = 24):
        self.db_path = Path(db_path)
        self.default_ttl_hours = default_ttl_hours
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Возвращает постоянное соединение для текущего потока."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_database(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at
            ON llm_cache(expires_at)
        """)
        conn.commit()

    def _hash(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        hash_input = prompt
        if model:
            hash_input += f"|model:{model}"
        for key in sorted(kwargs.keys()):
            hash_input += f"|{key}:{kwargs[key]}"
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT value, expires_at FROM llm_cache WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            if time.time() > row["expires_at"]:
                conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                conn.commit()
                return None
            return row["value"]

    def set(
        self,
        key: str,
        value: Any,
        ttl_hours: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
        expires_at = time.time() + (ttl * 3600)
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """INSERT OR REPLACE INTO llm_cache
                   (key, value, created_at, expires_at, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, str(value), time.time(), expires_at,
                 json.dumps(metadata) if metadata else None),
            )
            conn.commit()

    def purge_expired(self) -> int:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "DELETE FROM llm_cache WHERE expires_at < ?", (time.time(),)
            )
            conn.commit()
            return cursor.rowcount

    def clear(self) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM llm_cache")
            conn.commit()

    def size(self) -> int:
        with self._lock:
            conn = self._get_connection()
            return conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
