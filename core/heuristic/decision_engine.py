"""DecisionEngine for HeuristicCore - multi-criteria evaluation and thresholds."""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from shared.logging import get_logger
from shared.constants import (
    AUTO_EXECUTE_THRESHOLD,
    NOTIFY_THRESHOLD,
    ASK_USER_THRESHOLD,
)

logger = get_logger()


@dataclass
class Decision:
    decision_id: str
    confidence: float
    action: str  # auto_execute | notify | ask_user
    rationale: str
    trace: List[Dict[str, Any]]


class DecisionEngine:
    """Multi-criteria decision engine with threshold-based actions."""

    def evaluate(
        self,
        rule_results: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Decision:
        if not rule_results:
            return Decision(
                decision_id=self._generate_id(context),
                confidence=0.0,
                action="ask_user",
                rationale="No matching rules found",
                trace=[],
            )

        total_weight = sum(r["weight"] for r in rule_results)
        weighted_confidence = (
            sum(r["weight"] * r["confidence"] for r in rule_results) / total_weight
            if total_weight > 0
            else 0.0
        )

        if weighted_confidence >= AUTO_EXECUTE_THRESHOLD:
            action = "auto_execute"
        elif weighted_confidence >= NOTIFY_THRESHOLD:
            action = "notify"
        else:
            action = "ask_user"

        decision = Decision(
            decision_id=self._generate_id(context),
            confidence=weighted_confidence,
            action=action,
            rationale=f"Confidence {weighted_confidence:.2f} triggers {action}",
            trace=rule_results,
        )

        logger.info(
            "Decision made: id=%s confidence=%.2f action=%s",
            decision.decision_id,
            decision.confidence,
            decision.action,
        )

        return decision

    def _generate_id(self, context: Dict[str, Any]) -> str:
        canonical = str(sorted(context.items()))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]