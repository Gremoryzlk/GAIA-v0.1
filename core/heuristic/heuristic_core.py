"""HeuristicCore — deterministic rule-based decision engine for GAIA v7.3."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from shared.constants import AUTO_EXECUTE_THRESHOLD, ASK_USER_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    decision: str
    confidence: float
    threshold: float
    rule_id: Optional[str] = None
    trace: List[str] = field(default_factory=list)


@dataclass
class Rule:
    rule_id: str
    priority: int                                   # lower = evaluated first
    condition: Callable[[Dict[str, Any]], bool]
    action: str
    confidence: float


class HeuristicCore:
    """Deterministic rule-based decision engine.

    Rules are evaluated in priority order (ascending). First matching
    rule wins. If no rule matches, falls back to ASK_USER with low
    confidence to force escalation.
    """

    def __init__(self) -> None:
        self._rules: List[Rule] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        self.add_rule(Rule(
            rule_id="r001",
            priority=1,
            condition=lambda ctx: ctx.get("task_type") == "noop",
            action="skip",
            confidence=1.0,
        ))
        self.add_rule(Rule(
            rule_id="r002",
            priority=2,
            condition=lambda ctx: isinstance(ctx.get("priority"), int) and ctx.get("priority", 0) >= 8,
            action="auto_execute",
            confidence=AUTO_EXECUTE_THRESHOLD,
        ))
        self.add_rule(Rule(
            rule_id="r003",
            priority=3,
            condition=lambda ctx: isinstance(ctx.get("priority"), int) and ctx.get("priority", 0) >= 4,
            action="notify",
            confidence=0.75,
        ))

    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    def evaluate(self, context: Dict[str, Any]) -> Decision:
        trace: List[str] = []

        # Обогащаем контекст данными от LLM если есть
        if "llm_interpretation" in context:
            llm = context["llm_interpretation"]
            if llm.get("priority") in ("HIGH", "CRITICAL"):
                context["priority"] = 10
            context["task_type"] = llm.get("task_type", context.get("task_type", "unknown"))
            context["urgency"] = llm.get("urgency", context.get("urgency", 0.0))

        for rule in self._rules:
            try:
                matched = rule.condition(context)
            except Exception as e:
                logger.error("Rule %s condition error: %s", rule.rule_id, e)
                matched = False

            trace.append(f"{rule.rule_id}={'hit' if matched else 'miss'}")

            if matched:
                logger.debug(
                    "Rule %s matched for task_id=%s → action=%s confidence=%.2f",
                    rule.rule_id,
                    context.get("task_id"),
                    rule.action,
                    rule.confidence,
                )
                return Decision(
                    decision=rule.action,
                    confidence=rule.confidence,
                    threshold=AUTO_EXECUTE_THRESHOLD,
                    rule_id=rule.rule_id,
                    trace=trace,
                )

        # No rule matched — escalate
        logger.warning(
            "No rule matched for task_id=%s — escalating.",
            context.get("task_id"),
        )
        return Decision(
            decision="ask_user",
            confidence=ASK_USER_THRESHOLD,
            threshold=AUTO_EXECUTE_THRESHOLD,
            rule_id=None,
            trace=trace,
        )