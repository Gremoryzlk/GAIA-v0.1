"""TitanWorker — обработка результатов и checkpoint.

Маршрутизация результатов:
  < 1MB   → MemoryBroker
  1-100MB → файл в data/
  > 100MB → chunked файл

Checkpoint для долгих задач (каждые N страниц/шагов).
"""

import gzip
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

_MEMORY_BROKER_MAX_BYTES = 1 * 1024 * 1024       # 1MB
_FILE_MAX_BYTES = 100 * 1024 * 1024              # 100MB
_DATA_DIR = Path("data")
_CHECKPOINT_DIR = _DATA_DIR / "titan_checkpoints"
_CHUNK_SIZE = 10 * 1024 * 1024                   # 10MB per chunk


class TitanWorker:
    """Роутер результатов и менеджер чекпоинтов.

    Используется TitanAgent для сохранения результатов задач.
    """

    def __init__(self, memory_broker=None) -> None:
        self._broker = memory_broker
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Result routing ───────────────────────────────────────────────────────

    def store_result(
        self,
        task_id: str,
        task_type: str,
        result_data: Any,
    ) -> Dict[str, Any]:
        """Сохраняет результат по размеру.

        Returns: {"storage": "memory"|"file"|"chunked", "location": str, ...}
        """
        serialized = self._serialize(result_data)
        size = len(serialized)

        logger.info(
            "TitanWorker: storing result task_id=%s task_type=%s size=%d bytes",
            task_id, task_type, size,
        )

        if size <= _MEMORY_BROKER_MAX_BYTES:
            return self._store_memory(task_id, task_type, serialized)
        elif size <= _FILE_MAX_BYTES:
            return self._store_file(task_id, task_type, serialized)
        else:
            return self._store_chunked(task_id, task_type, serialized)

    def _serialize(self, data: Any) -> bytes:
        if isinstance(data, bytes):
            return data
        return json.dumps(data, ensure_ascii=False, default=str).encode()

    def _store_memory(self, task_id: str, task_type: str, data: bytes) -> Dict[str, Any]:
        key = f"titan_result_{task_id}"
        value = data.decode(errors="replace")
        if self._broker:
            try:
                self._broker.write(
                    key=key,
                    value=value,
                    memory_type="working",
                    agent_id="titan",
                    metadata={"task_type": task_type, "size": len(data)},
                )
            except Exception as e:
                logger.warning("TitanWorker: MemoryBroker write failed: %s", e)
        return {
            "storage": "memory",
            "key": key,
            "size_bytes": len(data),
        }

    def _store_file(self, task_id: str, task_type: str, data: bytes) -> Dict[str, Any]:
        path = _DATA_DIR / f"titan_{task_type}_{task_id}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as f:
            f.write(data)
        logger.info("TitanWorker: stored to file %s (%d bytes)", path, len(data))
        return {
            "storage": "file",
            "path": str(path),
            "size_bytes": len(data),
        }

    def _store_chunked(self, task_id: str, task_type: str, data: bytes) -> Dict[str, Any]:
        chunk_dir = _DATA_DIR / f"titan_chunked_{task_id}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunks = []
        for i, offset in enumerate(range(0, len(data), _CHUNK_SIZE)):
            chunk_path = chunk_dir / f"chunk_{i:04d}.gz"
            with gzip.open(chunk_path, "wb") as f:
                f.write(data[offset:offset + _CHUNK_SIZE])
            chunks.append(str(chunk_path))

        manifest = {
            "task_id": task_id,
            "task_type": task_type,
            "total_bytes": len(data),
            "chunk_count": len(chunks),
            "chunks": chunks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = chunk_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info(
            "TitanWorker: stored chunked %d chunks total=%d bytes dir=%s",
            len(chunks), len(data), chunk_dir,
        )
        return {
            "storage": "chunked",
            "manifest_path": str(manifest_path),
            "chunk_count": len(chunks),
            "size_bytes": len(data),
        }

    # ─── Checkpoint ───────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        task_id: str,
        step: int,
        state: Dict[str, Any],
    ) -> str:
        """Сохраняет чекпоинт долгой задачи."""
        path = _CHECKPOINT_DIR / f"{task_id}_step_{step:06d}.json"
        path.write_text(json.dumps({
            "task_id": task_id,
            "step": step,
            "state": state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, default=str))
        logger.debug("TitanWorker: checkpoint saved task_id=%s step=%d", task_id, step)
        return str(path)

    def load_latest_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Загружает последний чекпоинт задачи или None."""
        checkpoints = sorted(_CHECKPOINT_DIR.glob(f"{task_id}_step_*.json"))
        if not checkpoints:
            return None
        data = json.loads(checkpoints[-1].read_text())
        logger.info(
            "TitanWorker: loaded checkpoint task_id=%s step=%d",
            task_id, data.get("step"),
        )
        return data

    def cleanup_checkpoints(self, task_id: str) -> int:
        """Удаляет все чекпоинты задачи после завершения."""
        removed = 0
        for p in _CHECKPOINT_DIR.glob(f"{task_id}_step_*.json"):
            p.unlink()
            removed += 1
        logger.debug("TitanWorker: removed %d checkpoints for %s", removed, task_id)
        return removed
