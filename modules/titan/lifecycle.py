"""DataLifecycleManager — управление жизненным циклом данных Titan.

Правила:
  Сырые данные (HTML)     → удалить после обработки
  Структурированные батчи → gzip → archive/ → удалить через 30 дней
  ModelState (веса)       → data/model_states/ — хранить всегда
  Логи обучения           → 7 дней → удалить
"""

import gzip
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data")
_ARCHIVE_DIR = _DATA_DIR / "archive"
_MODEL_STATES_DIR = _DATA_DIR / "model_states"
_TRAINING_LOGS_DIR = Path("logs") / "training"

_ARCHIVE_DELETE_AFTER_DAYS = 30
_TRAINING_LOG_DELETE_AFTER_DAYS = 7


class DataLifecycleManager:
    """Управляет хранением и очисткой данных в соответствии с политикой GAIA."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        lc = cfg.get("lifecycle", {})
        self._html_delete = lc.get("html_delete_after_processing", True)
        self._archive_after_days = lc.get("archive_delete_after_days", _ARCHIVE_DELETE_AFTER_DAYS)
        self._log_delete_days = lc.get("training_log_delete_after_days", _TRAINING_LOG_DELETE_AFTER_DAYS)

        for d in [_ARCHIVE_DIR, _MODEL_STATES_DIR, _TRAINING_LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    # ─── HTML / raw data ──────────────────────────────────────────────────────

    def delete_raw_html(self, path: str | Path) -> bool:
        """Удаляет сырой HTML файл после обработки."""
        if not self._html_delete:
            return False
        p = Path(path)
        if p.exists():
            p.unlink()
            logger.debug("DataLifecycle: deleted raw HTML %s", p)
            return True
        return False

    # ─── Structured batches ───────────────────────────────────────────────────

    def archive_batch(self, batch_data: Any, batch_id: str) -> str:
        """Архивирует структурированный батч в gzip → archive/."""
        path = _ARCHIVE_DIR / f"batch_{batch_id}.json.gz"
        serialized = json.dumps(batch_data, default=str, ensure_ascii=False).encode()
        with gzip.open(path, "wb") as f:
            f.write(serialized)
        logger.info("DataLifecycle: archived batch %s (%d bytes)", batch_id, len(serialized))
        return str(path)

    def purge_old_archives(self) -> int:
        """Удаляет архивы старше archive_after_days дней."""
        cutoff = time.time() - self._archive_after_days * 86400
        removed = 0
        for p in _ARCHIVE_DIR.glob("*.gz"):
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
                logger.debug("DataLifecycle: purged old archive %s", p)
        if removed:
            logger.info("DataLifecycle: purged %d old archives", removed)
        return removed

    # ─── ModelState ───────────────────────────────────────────────────────────

    def save_model_state(self, state_data: Dict[str, Any], version: str) -> str:
        """Сохраняет ModelState — хранится всегда, не удаляется."""
        path = _MODEL_STATES_DIR / f"model_state_v{version}.json"
        path.write_text(json.dumps({
            **state_data,
            "version": version,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, default=str, indent=2))
        logger.info("DataLifecycle: saved ModelState version=%s", version)
        return str(path)

    def list_model_states(self) -> list[str]:
        """Возвращает список всех сохранённых версий ModelState."""
        return sorted(
            p.stem.removeprefix("model_state_v")
            for p in _MODEL_STATES_DIR.glob("model_state_v*.json")
        )

    def load_model_state(self, version: str) -> Optional[Dict[str, Any]]:
        """Загружает ModelState по версии."""
        path = _MODEL_STATES_DIR / f"model_state_v{version}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # ─── Training logs ────────────────────────────────────────────────────────

    def write_training_log(self, log_entry: Dict[str, Any]) -> str:
        """Записывает запись в лог обучения."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = _TRAINING_LOGS_DIR / f"train_{today}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")
        return str(path)

    def purge_old_training_logs(self) -> int:
        """Удаляет логи обучения старше log_delete_days дней."""
        cutoff = time.time() - self._log_delete_days * 86400
        removed = 0
        for p in _TRAINING_LOGS_DIR.glob("train_*.jsonl"):
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
                logger.debug("DataLifecycle: purged training log %s", p)
        if removed:
            logger.info("DataLifecycle: purged %d old training logs", removed)
        return removed

    # ─── Full maintenance pass ────────────────────────────────────────────────

    def run_maintenance(self) -> Dict[str, int]:
        """Запускает полный цикл обслуживания данных."""
        archives_removed = self.purge_old_archives()
        logs_removed = self.purge_old_training_logs()
        logger.info(
            "DataLifecycle: maintenance done archives_removed=%d logs_removed=%d",
            archives_removed, logs_removed,
        )
        return {
            "archives_removed": archives_removed,
            "logs_removed": logs_removed,
        }
