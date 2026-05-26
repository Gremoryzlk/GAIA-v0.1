"""LLM Interpreter for GAIA v7.3 — OpenRouter API integration.

EA-13: LLM output is data only. Routing and control flow stay in system code.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.llm.cache import LLMCache
from core.llm.prompt_templates import format_task_prompt
from shared.constants import (
    LLM_CACHE_TTL_HOURS,
    LLM_MAX_TOKENS,
    LLM_MODEL_DEFAULT,
    LLM_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class TaskContext:
    """Structured task context parsed from LLM output.

    All fields have safe defaults — caller must not trust LLM output blindly.
    Routing decisions are made by system code, not by these values directly.
    """

    task_type: str = "unknown"
    priority: int = 0                          # 0-10, LLM suggestion only
    entities: List[str] = field(default_factory=list)
    intent: str = ""
    constraints: List[str] = field(default_factory=list)
    expected_output: str = ""
    raw: str = ""                              # исходный LLM ответ для аудита
    from_cache: bool = False
    parse_error: Optional[str] = None         # если JSON не распарсился

    def is_valid(self) -> bool:
        """Минимальная валидность — есть task_type и intent."""
        return bool(self.task_type and self.task_type != "unknown" and self.intent)


class LLMInterpreter:
    """Вызывает OpenRouter API и возвращает TaskContext.

    Архитектурные ограничения (EA-13):
    - LLM определяет ТОЛЬКО task_type, entities, intent — не маршрут
    - Все решения о маршрутизации принимает HeuristicCore
    - При любой ошибке возвращает TaskContext с parse_error — не падает
    """

    def __init__(
        self,
        model: Optional[str] = None,
        cache: Optional[LLMCache] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model or os.environ.get("LLM_MODEL", LLM_MODEL_DEFAULT)
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._cache = cache or LLMCache(default_ttl_hours=LLM_CACHE_TTL_HOURS)

        if not self._api_key:
            logger.warning(
                "OPENROUTER_API_KEY not set — LLMInterpreter will return empty context."
            )

    def interpret(self, task: Dict[str, Any]) -> TaskContext:
        """Интерпретировать задачу через LLM.

        Args:
            task: Словарь задачи из EventBus.

        Returns:
            TaskContext с распарсенными полями. Никогда не бросает исключение.
        """
        user_input = self._task_to_text(task)
        prompt = format_task_prompt(user_input=user_input, template_type="task_parse")

        # Проверяем кэш
        cache_key = self._cache._hash(prompt, model=self._model)
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("LLM cache hit for task_id=%s.", task.get("task_id"))
            return self._parse_response(cached, from_cache=True)

        # Нет API ключа — возвращаем пустой контекст
        if not self._api_key:
            return TaskContext(parse_error="no_api_key")

        # Вызов API
        raw = self._call_api(prompt)
        if raw is None:
            return TaskContext(parse_error="api_error")

        # Кэшируем сырой ответ
        self._cache.set(
            cache_key,
            raw,
            metadata={"model": self._model, "task_id": task.get("task_id")},
        )

        return self._parse_response(raw, from_cache=False)

    def interpret_with_fallback(
        self, task: Dict[str, Any], error_context: str
    ) -> TaskContext:
        """Fallback-интерпретация при ошибке основного пути."""
        user_input = self._task_to_text(task)
        prompt = format_task_prompt(
            user_input=user_input,
            template_type="fallback",
            error_context=error_context,
        )

        if not self._api_key:
            return TaskContext(parse_error="no_api_key")

        raw = self._call_api(prompt)
        if raw is None:
            return TaskContext(parse_error="api_error")

        # Fallback ответ — plain text, не JSON
        return TaskContext(
            task_type="fallback",
            intent=raw[:500],   # обрезаем на случай длинного ответа
            raw=raw,
        )

    # ── Внутренние методы ─────────────────────────────────────────────────────

    def _task_to_text(self, task: Dict[str, Any]) -> str:
        """Конвертирует dict задачи в текст для промпта."""
        payload = task.get("payload", {})
        task_type = task.get("task_type", "")
        if isinstance(payload, dict):
            parts = [f"{k}: {v}" for k, v in payload.items() if v]
            text = "; ".join(parts)
        else:
            text = str(payload)
        return f"{task_type}: {text}".strip(": ")

    def _call_api(self, prompt: str) -> Optional[str]:
        """HTTP запрос к OpenRouter API. Возвращает текст ответа или None."""
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": 0,        # детерминированный вывод
        }).encode("utf-8")

        req = urllib.request.Request(
            OPENROUTER_API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://gaia.local",
                "X-Title": "GAIA v7.3",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")[:300]
            logger.error("OpenRouter HTTP %d: %s", e.code, body_text)
            return None

        except urllib.error.URLError as e:
            logger.error("OpenRouter connection error: %s", e.reason)
            return None

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("OpenRouter response parse error: %s", e)
            return None

        except Exception as e:
            logger.error("OpenRouter unexpected error: %s", e)
            return None

    def _parse_response(self, raw: str, from_cache: bool = False) -> TaskContext:
        """Парсит JSON ответ LLM в TaskContext.

        Если JSON невалидный — возвращает TaskContext с parse_error,
        не бросает исключение (EA-13: LLM ошибка не должна ронять систему).
        """
        # Убираем markdown-обёртку если модель добавила ```json ... ```
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning("LLM response is not valid JSON: %s", e)
            return TaskContext(raw=raw, from_cache=from_cache, parse_error=str(e))

        # Извлекаем поля с безопасными дефолтами
        # priority ограничиваем 0-10 независимо от того что вернул LLM
        raw_priority = data.get("priority", 0)
        try:
            priority = max(0, min(10, int(raw_priority)))
        except (TypeError, ValueError):
            priority = 0

        entities = data.get("entities", [])
        if not isinstance(entities, list):
            entities = []

        constraints = data.get("constraints", [])
        if not isinstance(constraints, list):
            constraints = [str(constraints)] if constraints else []

        return TaskContext(
            task_type=str(data.get("task_type", "unknown"))[:64],
            priority=priority,
            entities=[str(e)[:128] for e in entities[:20]],   # лимит
            intent=str(data.get("intent", ""))[:512],
            constraints=[str(c)[:256] for c in constraints[:10]],
            expected_output=str(data.get("expected_output", ""))[:256],
            raw=raw,
            from_cache=from_cache,
        )