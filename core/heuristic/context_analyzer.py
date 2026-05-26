"""ContextAnalyzer for HeuristicCore - entity extraction and urgency detection."""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """Extract entities, urgency, and context graph from input."""

    def analyze(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        result = {
            "entities": self._extract_entities(text),
            "urgency": self._detect_urgency(text),
            "context_graph": self._build_context_graph(text),
            "metadata": metadata or {},
        }
        logger.debug("Context analyzed: %d entities", len(result["entities"]))
        return result

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        entities = []
        patterns = [
            (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", "PERSON"),
            (r"\b\d{4}-\d{2}-\d{2}\b", "DATE"),
            (r"\b\d+\.\d+\.\d+\.\d+\b", "IP_ADDRESS"),
            (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "EMAIL"),
        ]
        for pattern, entity_type in patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    "text": match.group(),
                    "type": entity_type,
                    "start": str(match.start()),
                    "end": str(match.end()),
                })
        return entities

    def _detect_urgency(self, text: str) -> float:
        urgent_keywords = [
            "urgent", "immediately", "asap", "critical", "emergency",
            "now", "today", "important", "priority",
        ]
        text_lower = text.lower()
        matches = sum(1 for kw in urgent_keywords if kw in text_lower)
        return min(1.0, matches / len(urgent_keywords))

    def _build_context_graph(self, text: str) -> Dict[str, Any]:
        words = text.lower().split()
        return {
            "word_count": len(words),
            "unique_words": len(set(words)),
            "key_terms": sorted(w for w in set(words) if len(w) > 5)[:10]
        }