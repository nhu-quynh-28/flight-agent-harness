"""Semantic action-loop detection layer for the agent harness.

Computes canonical action fingerprints based on (tool_name, sorted(arguments))
over a sliding window, with semantic exemptions for legitimate polling actions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from config import DEFAULT_CONFIG


def make_action_fingerprint(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Create a deterministic, canonical string fingerprint from tool name and sorted arguments."""
    normalized_items: List[Tuple[str, str]] = []
    for k, v in sorted(arguments.items(), key=lambda x: x[0]):
        # Normalize strings for comparison
        val_str = str(v).strip()
        normalized_items.append((k, val_str))

    args_repr = ",".join(f"{k}={v}" for k, v in normalized_items)
    return f"{tool_name}({args_repr})"


@dataclass
class LoopDetectionResult:
    """Structured report produced by the loop detector."""
    detected: bool
    reason: str
    fingerprint: str
    recent_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "reason": self.reason,
            "fingerprint": self.fingerprint,
            "recent_actions": self.recent_actions,
        }


class SemanticLoopDetector:
    """Stateful sliding-window detector for repeated agent actions."""

    def __init__(
        self,
        window_size: int = DEFAULT_CONFIG.loop_window,
        repeat_threshold: int = DEFAULT_CONFIG.loop_repeat_k,
        safe_repeat_actions: Optional[Set[str]] = None,
    ) -> None:
        self.window_size = window_size
        self.repeat_threshold = repeat_threshold
        self.safe_repeat_actions = (
            safe_repeat_actions if safe_repeat_actions is not None
            else set(DEFAULT_CONFIG.safe_repeat_actions)
        )
        self.history: deque[str] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        """Clear action history."""
        self.history.clear()

    def record_and_check(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> LoopDetectionResult:
        """Record an attempted action and verify if it exceeds the loop threshold.

        Returns structured LoopDetectionResult indicating whether execution should halt.
        """
        fingerprint = make_action_fingerprint(tool_name, arguments)

        # Semantic exemption: some actions like polling status are legitimate to repeat
        if tool_name in self.safe_repeat_actions:
            self.history.append(fingerprint)
            return LoopDetectionResult(
                detected=False,
                reason=f"Action '{tool_name}' is semantically whitelisted for repeated polling.",
                fingerprint=fingerprint,
                recent_actions=list(self.history),
            )

        # Count previous occurrences of this exact fingerprint in current window
        prior_count = sum(1 for fp in self.history if fp == fingerprint)
        self.history.append(fingerprint)

        current_count = prior_count + 1
        if current_count >= self.repeat_threshold:
            return LoopDetectionResult(
                detected=True,
                reason=(
                    f"Semantic loop detected: Action '{fingerprint}' occurred "
                    f"{current_count} times within the recent window of {self.window_size}."
                ),
                fingerprint=fingerprint,
                recent_actions=list(self.history),
            )

        return LoopDetectionResult(
            detected=False,
            reason="Action within normal non-repeating parameters.",
            fingerprint=fingerprint,
            recent_actions=list(self.history),
        )
