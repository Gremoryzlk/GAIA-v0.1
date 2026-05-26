"""RuleEngine for HeuristicCore - IF->THEN->WEIGHT rules with O(1) lookup."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    rule_id: str
    applies_to: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    weight: float
    confidence: float = 1.0


class RuleEngine:
    def __init__(self, rules_path: Optional[Path] = None) -> None:
        self._rules: Dict[str, List[Rule]] = {}
        if rules_path:
            self.load_rules(rules_path)

    def load_rules(self, rules_path: Path) -> None:
        with open(rules_path, "r") as f:
            data = json.load(f)
        for rule_data in data.get("rules", []):
            rule = Rule(
                rule_id=rule_data["rule_id"],
                applies_to=rule_data["applies_to"],
                condition=rule_data["condition"],
                action=rule_data["action"],
                weight=rule_data.get("weight", 1.0),
                confidence=rule_data.get("confidence", 1.0),
            )
            self._rules.setdefault(rule.applies_to, []).append(rule)
        total = sum(len(v) for v in self._rules.values())
        logger.info("Rules loaded: count=%d", total)

    def get_rules(self, applies_to: str) -> List[Rule]:
        return self._rules.get(applies_to, [])

    def evaluate(self, applies_to: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        for rule in self.get_rules(applies_to):
            if self._match_condition(rule.condition, context):
                results.append({
                    "rule_id": rule.rule_id,
                    "action": rule.action,
                    "weight": rule.weight,
                    "confidence": rule.confidence,
                })
        return results

    def _coerce(self, expected: str, actual: Any) -> Any:
        """Явное приведение строки из JSON к типу actual (только int и float)."""
        if isinstance(actual, bool):
            return expected.lower() in ("true", "1", "yes")
        if isinstance(actual, int):
            try:
                return int(expected)
            except (ValueError, TypeError):
                return expected
        if isinstance(actual, float):
            try:
                return float(expected)
            except (ValueError, TypeError):
                return expected
        return expected

    def _match_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        for key, expected in condition.items():
            if key not in context:
                return False
            actual = context[key]

            # Приводим строковые значения из JSON к типу контекста
            if isinstance(expected, str) and not isinstance(actual, str):
                expected = self._coerce(expected, actual)

            if isinstance(expected, dict):
                if not isinstance(actual, dict):
                    return False
                for k, v in expected.items():
                    if k not in actual or actual[k] != v:
                        return False
            elif actual != expected:
                return False
        return True