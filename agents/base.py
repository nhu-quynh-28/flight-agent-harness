"""Base data structures and result types shared across all three agent architectures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harness.completion import CompletionResult
from harness.handoff import HandoffPackage


@dataclass
class AgentRunResult:
    """Standardized result returned by all three agent designs for fair evaluation."""
    agent_name: str
    query: str
    success: bool
    final_answer: str
    completion_result: Optional[CompletionResult] = None
    handoff: Optional[HandoffPackage] = None
    steps_executed: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    permission_blocks: int = 0
    loop_detected: bool = False
    stop_reason: Optional[str] = None
    duration_seconds: float = 0.0
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "query": self.query,
            "success": self.success,
            "final_answer": self.final_answer,
            "completion_verified": self.completion_result.completed if self.completion_result else False,
            "completion_reason": self.completion_result.reason if self.completion_result else "N/A",
            "handoff": self.handoff.to_dict() if self.handoff else None,
            "steps_executed": self.steps_executed,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "permission_blocks": self.permission_blocks,
            "loop_detected": self.loop_detected,
            "stop_reason": self.stop_reason,
            "duration_seconds": round(self.duration_seconds, 4),
        }
