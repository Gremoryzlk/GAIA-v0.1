"""MemoryBroker for GAIA v7.3 with SQLite WAL persistence."""

import json
import logging
import sqlite3
import threading
from typing import Any, Optional

from core.eventbus.event import EventPriority
from core.eventbus.eventbus import EventBus
from shared.constants import MEMORY_WORKING_TTL_HOURS, MEMORY_LONGTERM_MAX_ENTRIES

logger = logging.getLogger(__name__)

_DEFAULT_DB = "data/system.db"


class MemoryBroker:
    """Thread-safe memory broker with SQLite WAL persistence."""

    def __init__(self, db_path: str = _DEFAULT_DB) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_db()
        self._eventbus = EventBus()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                self._db_path, check_same_thread=False
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")
        return self._local.conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS working_memory (
                    key TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ttl_hours INTEGER DEFAULT 24,
                    PRIMARY KEY (agent_id, key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS longterm_memory (
                    key TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence_score REAL DEFAULT 1.0,
                    ttl INTEGER,
                    provenance_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (agent_id, key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    trace TEXT NOT NULL,
                    reward REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    requester_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    key_pattern TEXT,
                    reason TEXT,
                    granted INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def read(self, agent_id: str, memory_type: str, key: str) -> Optional[Any]:
        with self._lock:
            conn = self._get_connection()
            if memory_type == "working":
                row = conn.execute(
                    "SELECT value FROM working_memory WHERE agent_id=? AND key=?",
                    (agent_id, key),
                ).fetchone()
                return json.loads(row[0]) if row else None
            elif memory_type == "longterm":
                row = conn.execute(
                    "SELECT value FROM longterm_memory WHERE agent_id=? AND key=?",
                    (agent_id, key),
                ).fetchone()
                return json.loads(row[0]) if row else None
            elif memory_type == "decisions":
                row = conn.execute(
                    "SELECT decision FROM decision_memory "
                    "WHERE agent_id=? ORDER BY id DESC LIMIT 1",
                    (agent_id,),
                ).fetchone()
                return json.loads(row[0]) if row else None
            logger.warning("Unknown memory_type: %s", memory_type)
            return None

    def write(
        self,
        agent_id: str,
        memory_type: str,
        key: str,
        value: Any,
        ttl_hours: Optional[int] = None,
        confidence_score: float = 1.0,
        provenance_hash: Optional[str] = None,
    ) -> bool:
        with self._lock:
            conn = self._get_connection()
            serialized = json.dumps(value)
            try:
                if memory_type == "working":
                    conn.execute(
                        """INSERT INTO working_memory (key, agent_id, value, ttl_hours)
                           VALUES (?, ?, ?, ?)
                           ON CONFLICT(agent_id, key) DO UPDATE SET
                               value=excluded.value,
                               ttl_hours=excluded.ttl_hours,
                               created_at=CURRENT_TIMESTAMP""",
                        (key, agent_id, serialized, ttl_hours or MEMORY_WORKING_TTL_HOURS),
                    )
                elif memory_type == "longterm":
                    count = conn.execute(
                        "SELECT COUNT(*) FROM longterm_memory"
                    ).fetchone()[0]
                    if count >= MEMORY_LONGTERM_MAX_ENTRIES:
                        conn.execute(
                            """DELETE FROM longterm_memory WHERE rowid IN (
                                SELECT rowid FROM longterm_memory
                                ORDER BY created_at ASC LIMIT 1000
                            )"""
                        )
                        logger.warning("Longterm memory eviction triggered.")
                    conn.execute(
                        """INSERT INTO longterm_memory
                               (key, agent_id, value, confidence_score, ttl, provenance_hash)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(agent_id, key) DO UPDATE SET
                               value=excluded.value,
                               confidence_score=excluded.confidence_score,
                               provenance_hash=excluded.provenance_hash,
                               created_at=CURRENT_TIMESTAMP""",
                        (key, agent_id, serialized, confidence_score, ttl_hours, provenance_hash),
                    )
                elif memory_type == "decisions":
                    conn.execute(
                        "INSERT INTO decision_memory (agent_id, decision, trace) VALUES (?, ?, ?)",
                        (agent_id, serialized, json.dumps({})),
                    )
                else:
                    logger.warning("Unknown memory_type: %s", memory_type)
                    return False
                conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error("Memory write error: %s", e)
                conn.rollback()
                return False

    def purge_expired_working(self) -> int:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                """DELETE FROM working_memory
                   WHERE datetime(created_at, '+' || ttl_hours || ' hours') < datetime('now')"""
            )
            conn.commit()
            count = cursor.rowcount
            if count:
                logger.info("Purged %d expired working memory entries.", count)
            return count

    def get_permission(self, requester_id: str, owner_id: str, memory_type: str) -> bool:
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                """SELECT granted FROM permissions
                   WHERE requester_id=? AND owner_id=? AND memory_type=?
                   ORDER BY created_at DESC LIMIT 1""",
                (requester_id, owner_id, memory_type),
            ).fetchone()
            return bool(row[0]) if row else False

    def grant_permission(
        self,
        requester_id: str,
        owner_id: str,
        memory_type: str,
        key_pattern: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """INSERT INTO permissions
                   (requester_id, owner_id, memory_type, key_pattern, reason, granted)
                   VALUES (?, ?, ?, ?, ?, 1)""",
                (requester_id, owner_id, memory_type, key_pattern, reason),
            )
            conn.commit()