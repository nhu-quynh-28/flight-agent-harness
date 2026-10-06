"""Safe human handoff data structures and package generator.

Preserves full conversational and operational context upon abnormal or safe stopping,
enabling seamless handoff to human operators without silent failure or lost state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HandoffPackage:
    """Structured artifact prepared when the agent must halt or escalate to a human."""
    stop_reason: str
    attempted_actions: List[Dict[str, Any]] = field(default_factory=list)
    current_state: Dict[str, Any] = field(default_factory=dict)
    question_for_human: str = ""
    suggested_alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "attempted_actions": self.attempted_actions,
            "current_state": self.current_state,
            "question_for_human": self.question_for_human,
            "suggested_alternatives": self.suggested_alternatives,
        }


def create_handoff(
    stop_reason: str,
    attempted_actions: List[Dict[str, Any]],
    current_state: Dict[str, Any],
    question_for_human: str,
    suggested_alternatives: Optional[List[str]] = None,
) -> HandoffPackage:
    """Generate a structured handoff package for human escalation."""
    return HandoffPackage(
        stop_reason=stop_reason,
        attempted_actions=attempted_actions,
        current_state=current_state,
        question_for_human=question_for_human,
        suggested_alternatives=suggested_alternatives or [],
    )
