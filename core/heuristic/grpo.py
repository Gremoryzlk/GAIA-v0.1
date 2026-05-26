"""GRPO (Gradient-based Reward Policy Optimization) for GAIA v7.3.

Deterministic variant selection using relative advantage calculation.
Part of HeuristicCore reasoning pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GRPO:
    """Gradient-based Reward Policy Optimization for variant selection.
    
    Evaluates multiple decision variants using relative advantage calculation
    to select the optimal choice deterministically.
    
    Attributes:
        num_variants: Number of variants to evaluate (8 for HeuristicCore, 5 for GrowthEngine)
        baseline_score: Reference score for advantage calculation
    """
    
    def __init__(self, num_variants: int = 8):
        """Initialize GRPO with specified number of variants.
        
        Args:
            num_variants: Number of variants to evaluate (default 8 for HeuristicCore)
        """
        self.num_variants = num_variants
        self.baseline_score: float = 0.0
        self._last_advantages: Optional[np.ndarray] = None
        self._last_selected: Optional[int] = None
    
    def compute_advantages(
        self,
        rewards: np.ndarray,
        baseline: Optional[float] = None
    ) -> np.ndarray:
        """Compute relative advantages for each variant.
        
        Advantage = reward - baseline (mean reward by default)
        
        Args:
            rewards: Array of reward scores for each variant
            baseline: Optional baseline score (defaults to mean of rewards)
            
        Returns:
            Array of advantage values for each variant
        """
        if len(rewards) != self.num_variants:
            raise ValueError(
                f"Expected {self.num_variants} rewards, got {len(rewards)}"
            )
        
        if not np.all(np.isfinite(rewards)):
            raise ValueError(f"rewards содержит NaN или Inf: {rewards}")
        
        baseline_score = baseline if baseline is not None else float(np.mean(rewards))
        self.baseline_score = baseline_score
        
        advantages = rewards - baseline_score
        self._last_advantages = advantages
        
        return advantages
    
    def select_variant(self, advantages: np.ndarray) -> int:
        """Select the variant with highest advantage (deterministic).
        
        Args:
            advantages: Array of advantage values for each variant
            
        Returns:
            Index of selected variant (0-based)
        """
        if len(advantages) != self.num_variants:
            raise ValueError(
                f"Expected {self.num_variants} advantages, got {len(advantages)}"
            )
        
        selected_idx = int(np.argmax(advantages))
        self._last_selected = selected_idx
        
        return selected_idx
    
    def evaluate_and_select(
        self,
        rewards: np.ndarray,
        baseline: Optional[float] = None
    ) -> Tuple[int, np.ndarray]:
        """Compute advantages and select best variant in one step.
        
        Args:
            rewards: Array of reward scores for each variant
            baseline: Optional baseline score
            
        Returns:
            Tuple of (selected variant index, advantages array)
        """
        advantages = self.compute_advantages(rewards, baseline)
        selected = self.select_variant(advantages)
        return selected, advantages
    
    def get_selection_confidence(self, advantages: np.ndarray) -> float:
        """Calculate confidence in the selection based on advantage margin.
        
        Higher margin between top and second-best indicates higher confidence.
        
        Args:
            advantages: Array of advantage values
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if len(advantages) < 2:
            return 1.0
        
        sorted_adv = np.sort(advantages)[::-1]
        margin = sorted_adv[0] - sorted_adv[1]
        
        # Normalize margin to confidence (sigmoid-like scaling)
        confidence = 1.0 / (1.0 + np.exp(-margin))
        
        return float(confidence)
    
    def update_baseline(self, new_reward: float, alpha: float = 0.1) -> float:
        """Update baseline score using exponential moving average.
        
        Args:
            new_reward: New reward observation
            alpha: EMA smoothing factor (0 < alpha <= 1)
            
        Returns:
            Updated baseline score
        """
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in range (0, 1]")
        
        self.baseline_score = (
            alpha * new_reward + (1 - alpha) * self.baseline_score
        )
        return self.baseline_score
    
    def get_state(self) -> Dict[str, Any]:
        """Get current GRPO state for serialization.
        
        Returns:
            Dictionary containing current state
        """
        return {
            "num_variants": self.num_variants,
            "baseline_score": self.baseline_score,
            "last_advantages": (
                self._last_advantages.tolist() 
                if self._last_advantages is not None else None
            ),
            "last_selected": self._last_selected
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load GRPO state from serialized data.
        
        Args:
            state: Dictionary containing saved state
        """
        self.num_variants = state.get("num_variants", self.num_variants)
        self.baseline_score = state.get("baseline_score", 0.0)
        
        if state.get("last_advantages") is not None:
            self._last_advantages = np.array(state["last_advantages"])
        else:
            self._last_advantages = None
        
        self._last_selected = state.get("last_selected")


def create_heuristic_core_grpo() -> GRPO:
    """Factory function for HeuristicCore GRPO (8 variants).
    
    Returns:
        GRPO instance configured for HeuristicCore
    """
    return GRPO(num_variants=8)


def create_growth_engine_grpo() -> GRPO:
    """Factory function for GrowthEngine GRPO (5 variants).
    
    Returns:
        GRPO instance configured for GrowthEngine
    """
    return GRPO(num_variants=5)