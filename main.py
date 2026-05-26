"""GAIA v7.3 Main Entry Point."""

import importlib
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from shared.logging import get_logger
from core.eventbus.event import Event
from core.eventbus.eventbus import EventBus
from shared.types import EventPriority
from core.heuristic.heuristic_core import HeuristicCore
from agents.gaia_agent import GaiaAgent
from core.database.schema import init_database
from core.watchdog import AgentWatchdog
from core.safety.meta_controller import MetaController, SafetyGate, ModelState

_MAX_LINE_BYTES = 1 * 1024 * 1024

_event_bus: Optional[EventBus] = None
_gaia_agent: Optional[GaiaAgent] = None
_db_conn: Optional[sqlite3.Connection] = None
_module_agents: List[Any] = []
_watchdog: Any = None
_meta_controller: Any = None


def validate_secret_key() -> str:
    secret_key = os.getenv("GAIA_SECRET_KEY", "")
    if not secret_key:
        print("CRITICAL: GAIA_SECRET_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    if len(secret_key) < 32:
        print(f"CRITICAL: GAIA_SECRET_KEY must be >= 32 chars.", file=sys.stderr)
        sys.exit(1)
    return secret_key


def load_manifest(manifest_path: str = "manifest.json") -> Dict[str, Any]:
    """Загружает manifest.json если существует."""
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r") as f:
        return json.load(f)


def load_modules(manifest: Dict[str, Any], logger: logging.Logger) -> List[Any]:
    """Загружает и инициализирует модули из manifest.json."""
    agents = []
    modules = manifest.get("modules", {})

    for name, cfg in modules.items():
        if not cfg.get("enabled", False):
            logger.info("Module '%s' disabled — skipping.", name)
            continue

        module_path = cfg.get("path", f"modules/{name}")
        config_path = os.path.join(module_path, "config.json")

        try:
            # Читаем config.json модуля
            if not os.path.exists(config_path):
                logger.warning("Module '%s': config.json not found at %s.", name, config_path)
                continue

            with open(config_path) as f:
                module_cfg = json.load(f)

            agent_class_path = module_cfg.get("agent_class")
            if not agent_class_path:
                logger.warning("Module '%s': no agent_class in config.", name)
                continue

            # Динамический импорт
            module_name, class_name = agent_class_path.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)

            # Создаём агента с параметрами из конфига
            kwargs = {}
            if "workspace_dir" in module_cfg:
                kwargs["workspace_dir"] = module_cfg["workspace_dir"]

            agent = cls(**kwargs)
            agents.append(agent)
            logger.info("Module '%s' loaded: %s (tools: %s)",
                        name, agent.agent_id,
                        agent.list_tools() if hasattr(agent, "list_tools") else "n/a")

        except Exception as e:
            logger.error("Failed to load module '%s': %s — continuing without it.", name, e)

    return agents


def create_system_db(db_path: str = "data/system.db") -> sqlite3.Connection:
    conn = init_database(db_path)
    return conn


def signal_handler(signum, frame) -> None:
    global _event_bus, _db_conn
    logger = get_logger()
    logger.info("Signal %d received — shutting down.", signum)

    global _watchdog
    if _watchdog:
        _watchdog.stop()

    if _event_bus:
        _event_bus.publish(Event(
            event_type="system.shutdown",
            priority=EventPriority.CRITICAL,
            source="main",
            payload={"reason": f"signal_{signum}",
                     "timestamp": datetime.now(timezone.utc).isoformat()},
        ))
        time.sleep(0.3)
        _event_bus.shutdown()

    if _db_conn:
        _db_conn.close()

    sys.exit(0)


def _parse_priority(priority_str: str) -> EventPriority:
    try:
        return EventPriority[priority_str.upper()]
    except KeyError:
        return EventPriority.NORMAL


def main() -> None:
    global _event_bus, _gaia_agent, _db_conn, _module_agents

    load_dotenv(override=True)
    validate_secret_key()

    logger = get_logger()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    _db_conn = create_system_db()
    logger.info("Database ready.")

    _event_bus = EventBus()
    heuristic_core = HeuristicCore()
    _gaia_agent = GaiaAgent(heuristic_core=heuristic_core)

    # Загружаем модули из manifest.json
    manifest = load_manifest()
    if manifest:
        _module_agents = load_modules(manifest, logger)
        logger.info("Modules loaded: %d", len(_module_agents))
    else:
        logger.info("No manifest.json found — running without modules.")

    # MetaController для auto-reset агентов
    import numpy as np
    _meta_controller = MetaController(
        eval_fn=lambda s: float(np.sum(s.weights)),
        safety_gate=SafetyGate([]),
        max_rollback_depth=3,
        evolution_budget=100.0,
    )
    # Инициализируем начальное состояние
    _initial_state = ModelState(np.ones(4), {"level": "system"}, 0, 1.0)
    _initial_state.compute_signature()
    _meta_controller.step(_initial_state)

    # Запускаем watchdog
    _watchdog = AgentWatchdog(check_interval=30, meta_controller=_meta_controller)
    _watchdog.register(_gaia_agent)
    for agent in _module_agents:
        _watchdog.register(agent)
    _watchdog.start()
    logger.info("Watchdog started for %d agents.", 1 + len(_module_agents))

    _event_bus.publish(Event(
        event_type="system.ready",
        priority=EventPriority.CRITICAL,
        source="main",
        payload={
            "phase": manifest.get("phase", 0),
            "modules": list(manifest.get("modules", {}).keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    ))
    _event_bus.dispatch_next()

    logger.info("GAIA v7.3 Phase %s started. Waiting for tasks on stdin...",
                manifest.get("phase", 0))

    try:
        for line in sys.stdin:
            if len(line) > _MAX_LINE_BYTES:
                logger.error("Input line too large (%d bytes), skipping.", len(line))
                continue

            line = line.strip()
            if not line:
                continue

            try:
                task_data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON on stdin: %s", e)
                continue

            try:
                priority = _parse_priority(task_data.get("priority", "NORMAL"))
                _event_bus.publish(Event(
                    event_type="task",
                    priority=priority,
                    source="stdin",
                    payload=task_data,
                ))
                result = _gaia_agent.process_task(task_data)
                logger.info("Task result: %s", result)
                # Записываем llm_interpretation отдельно если есть
                if result.get("llm_interpretation"):
                    log_entry = json.dumps(result["llm_interpretation"], ensure_ascii=False, indent=2)
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/gaia.log", "a", encoding="utf-8") as f:
                        f.write(f"{datetime.now(timezone.utc).isoformat()} LLM_INTERPRETATION\n{log_entry}\n---\n")
                _event_bus.dispatch_next()
            except Exception as e:
                logger.error("Error processing task: %s", e)

    except EOFError:
        pass

    logger.info("EOF — shutting down.")
    if _watchdog:
        _watchdog.stop()
    _event_bus.publish(Event(
        event_type="system.shutdown",
        priority=EventPriority.CRITICAL,
        source="main",
        payload={"reason": "EOF", "timestamp": datetime.now(timezone.utc).isoformat()},
    ))
    _event_bus.dispatch_next()
    time.sleep(0.2)
    _event_bus.shutdown()

    if _db_conn:
        _db_conn.close()

    logger.info("GAIA v7.3 exited normally.")


if __name__ == "__main__":
    main()
