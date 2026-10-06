"""Metrics computation and statistical aggregation for agent evaluation.

Computes aggregate performance, safety, and operational metrics across test runs
without fabricating numbers or using misleading precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AgentAggregateMetrics:
    """Aggregate statistics for an agent across an evaluation suite."""
    agent_name: str
    total_cases: int
    successful_cases: int
    task_success_rate: float
    completion_correctness_rate: float
    avg_model_calls: float
    avg_tool_calls: float
    loop_detection_rate: float
    permission_block_rate: float
    handoff_rate: float
    recovery_rate: float
    avg_duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "total_cases": self.total_cases,
            "successful_cases": self.successful_cases,
            "task_success_rate_pct": round(self.task_success_rate * 100, 1),
            "completion_correctness_pct": round(self.completion_correctness_rate * 100, 1),
            "avg_model_calls": round(self.avg_model_calls, 2),
            "avg_tool_calls": round(self.avg_tool_calls, 2),
            "loop_detection_rate_pct": round(self.loop_detection_rate * 100, 1),
            "permission_block_rate_pct": round(self.permission_block_rate * 100, 1),
            "handoff_rate_pct": round(self.handoff_rate * 100, 1),
            "recovery_rate_pct": round(self.recovery_rate * 100, 1),
            "avg_duration_seconds": round(self.avg_duration_seconds, 4),
        }


def calculate_metrics(agent_name: str, records: List[Dict[str, Any]]) -> AgentAggregateMetrics:
    """Compute aggregate evaluation metrics from a list of run records."""
    total = len(records)
    if total == 0:
        return AgentAggregateMetrics(
            agent_name=agent_name,
            total_cases=0,
            successful_cases=0,
            task_success_rate=0.0,
            completion_correctness_rate=0.0,
            avg_model_calls=0.0,
            avg_tool_calls=0.0,
            loop_detection_rate=0.0,
            permission_block_rate=0.0,
            handoff_rate=0.0,
            recovery_rate=0.0,
            avg_duration_seconds=0.0,
        )

    successful = sum(1 for r in records if r.get("success", False))
    completion_verified = sum(1 for r in records if r.get("completion_verified", False))
    total_model_calls = sum(r.get("model_calls", 0) for r in records)
    total_tool_calls = sum(r.get("tool_calls", 0) for r in records)
    loop_detected = sum(1 for r in records if r.get("loop_detected", False))
    permission_blocked = sum(1 for r in records if r.get("permission_blocks", 0) > 0)
    handoff_count = sum(1 for r in records if r.get("handoff") is not None)
    total_duration = sum(r.get("duration_seconds", 0.0) for r in records)

    # Recovery cases: specifically TC11 and TC12 where adaptability is tested
    recovery_cases = [r for r in records if r.get("case_id") in ("TC11_RECOVERY_AFTER_FAILED_OBSERVATION", "TC12_PLAN_INVALIDATION_ALTERNATIVE")]
    recovery_success = sum(1 for r in recovery_cases if r.get("success", False))
    recovery_rate = (recovery_success / len(recovery_cases)) if recovery_cases else 0.0

    return AgentAggregateMetrics(
        agent_name=agent_name,
        total_cases=total,
        successful_cases=successful,
        task_success_rate=successful / total,
        completion_correctness_rate=completion_verified / total,
        avg_model_calls=total_model_calls / total,
        avg_tool_calls=total_tool_calls / total,
        loop_detection_rate=loop_detected / total,
        permission_block_rate=permission_blocked / total,
        handoff_rate=handoff_count / total,
        recovery_rate=recovery_rate,
        avg_duration_seconds=total_duration / total,
    )
