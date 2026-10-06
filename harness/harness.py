"""Main Harness coordinator for safe agent execution.

Unifies constraints-as-data, code-based completion checks, permission control,
semantic loop detection, execution budgets, and safe human handoff into a cohesive,
agent-agnostic safety harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import Config, DEFAULT_CONFIG
from harness.completion import (
    CompletionResult,
    check_booking_completed,
    check_cancellation_completed,
    check_search_completed,
)
from harness.constraints import ConstraintResult, check_constraints
from harness.handoff import HandoffPackage, create_handoff
from harness.loop_detector import LoopDetectionResult, SemanticLoopDetector
from harness.permissions import PermissionResult, UserSession, check_permission
from tools.flight_tools import (
    execute_book_flight,
    execute_cancel_booking,
    execute_get_booking_status,
    execute_get_flight_details,
    execute_search_flights,
)


@dataclass
class PreCheckResult:
    """Outcome of pre-execution evaluation."""
    proceed: bool
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    handoff: Optional[HandoffPackage] = None


class Harness:
    """The central agent harness implementing all safety and verification layers."""

    def __init__(
        self,
        config: Optional[Config] = None,
        session: Optional[UserSession] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.session = session or UserSession()

        # Protection Mechanism 1: Framework hard limit
        self.framework_max_steps = self.config.framework_max_steps
        self.step_counter = 0

        # Protection Mechanism 2: Model & Tool call budgets
        self.max_model_calls = self.config.max_model_calls
        self.max_tool_calls = self.config.max_tool_calls
        self.model_calls_count = 0
        self.tool_calls_count = 0

        # Protection Mechanism 3: Semantic loop detector
        self.loop_detector = SemanticLoopDetector(
            window_size=self.config.loop_window,
            repeat_threshold=self.config.loop_repeat_k,
            safe_repeat_actions=self.config.safe_repeat_actions,
        )

        # Execution tracking and audit state
        self.attempted_actions: List[Dict[str, Any]] = []
        self.permission_blocks: List[Dict[str, Any]] = []
        self.constraint_violations: List[Dict[str, Any]] = []
        self.loop_occurrences: List[Dict[str, Any]] = []
        self.active_handoff: Optional[HandoffPackage] = None
        self.stopped = False
        self.stop_reason: Optional[str] = None
        self.completion_verified = False

        # Raw tool dispatch registry
        self._raw_dispatchers: Dict[str, Callable[..., Dict[str, Any]]] = {
            "search_flights": lambda args: execute_search_flights(
                args.get("origin", ""),
                args.get("destination", ""),
                args.get("date", ""),
            ),
            "get_flight_details": lambda args: execute_get_flight_details(
                args.get("flight_id", "")
            ),
            "book_flight": lambda args: execute_book_flight(
                flight_id=args.get("flight_id", ""),
                passenger_name=args.get("passenger_name", ""),
                passenger_email=args.get("passenger_email", ""),
                confirmed_by_user=args.get("confirmed_by_user", False),
                user_id=self.session.user_id,
            ),
            "get_booking_status": lambda args: execute_get_booking_status(
                args.get("booking_id", "")
            ),
            "cancel_booking": lambda args: execute_cancel_booking(
                args.get("booking_id", ""),
                user_id=self.session.user_id,
            ),
        }

    def reset(self) -> None:
        """Reset the harness tracking state for a new execution run."""
        self.step_counter = 0
        self.model_calls_count = 0
        self.tool_calls_count = 0
        self.attempted_actions.clear()
        self.permission_blocks.clear()
        self.constraint_violations.clear()
        self.loop_occurrences.clear()
        self.loop_detector.reset()
        self.active_handoff = None
        self.stopped = False
        self.stop_reason = None
        self.completion_verified = False

    def record_model_call(self) -> bool:
        """Record an LLM model invocation and check model call budget."""
        if self.stopped:
            return False

        self.model_calls_count += 1
        self.step_counter += 1

        # Check framework hard step limit
        if self.step_counter > self.framework_max_steps:
            self.stopped = True
            self.stop_reason = "FRAMEWORK_STEP_LIMIT_EXCEEDED"
            self.active_handoff = self.create_handoff(
                stop_reason=self.stop_reason,
                question="The execution reached the framework hard recursion limit. Would you like to review the task?",
            )
            return False

        # Check model budget
        if self.model_calls_count > self.max_model_calls:
            self.stopped = True
            self.stop_reason = "MODEL_BUDGET_EXCEEDED"
            self.active_handoff = self.create_handoff(
                stop_reason=self.stop_reason,
                question=f"Agent exceeded maximum model calls budget ({self.max_model_calls}). How would you like to proceed?",
            )
            return False

        return True

    def before_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> PreCheckResult:
        """Evaluate all safety checks prior to executing a tool.

        Order of evaluation:
        1. Framework hard step limit
        2. Tool call budget
        3. Permission checks (blocks unauthorized/unconfirmed actions)
        4. Constraints as data validation
        5. Semantic loop detection
        """
        self.step_counter += 1

        # 1. Framework hard step limit
        if self.step_counter > self.framework_max_steps:
            self.stopped = True
            self.stop_reason = "FRAMEWORK_STEP_LIMIT_EXCEEDED"
            self.active_handoff = self.create_handoff(
                stop_reason=self.stop_reason,
                question=f"Execution stopped: Reached framework hard step limit ({self.framework_max_steps}).",
            )
            return PreCheckResult(
                proceed=False,
                status="FRAMEWORK_LIMIT_EXCEEDED",
                message=f"Hard limit of {self.framework_max_steps} steps reached.",
                handoff=self.active_handoff,
            )

        # 2. Tool call budget
        if self.tool_calls_count >= self.max_tool_calls:
            self.stopped = True
            self.stop_reason = "TOOL_BUDGET_EXCEEDED"
            self.active_handoff = self.create_handoff(
                stop_reason=self.stop_reason,
                question=f"Tool call budget ({self.max_tool_calls}) exhausted. Do you wish to authorize more operations?",
            )
            return PreCheckResult(
                proceed=False,
                status="TOOL_BUDGET_EXCEEDED",
                message=f"Maximum tool calls budget ({self.max_tool_calls}) exceeded.",
                handoff=self.active_handoff,
            )

        # 3. Permission control check
        perm = check_permission(tool_name, arguments, self.session)
        if not perm.allowed:
            block_info = {
                "tool_name": tool_name,
                "arguments": arguments,
                "code": perm.code,
                "message": perm.message,
            }
            self.permission_blocks.append(block_info)
            return PreCheckResult(
                proceed=False,
                status="PERMISSION_REQUIRED",
                message=perm.message,
                details=block_info,
            )

        # 4. Constraints as data validation
        constraint = check_constraints(tool_name, arguments)
        if not constraint.allowed:
            violation_info = {
                "tool_name": tool_name,
                "arguments": arguments,
                "code": constraint.code,
                "message": constraint.message,
                "details": constraint.details,
            }
            self.constraint_violations.append(violation_info)
            return PreCheckResult(
                proceed=False,
                status=constraint.code,
                message=constraint.message,
                details=violation_info,
            )

        # 5. Semantic loop detection
        loop_res = self.loop_detector.record_and_check(tool_name, arguments)
        if loop_res.detected:
            self.stopped = True
            self.stop_reason = "LOOP_DETECTED"
            loop_info = {
                "tool_name": tool_name,
                "arguments": arguments,
                "reason": loop_res.reason,
                "fingerprint": loop_res.fingerprint,
                "recent_actions": loop_res.recent_actions,
            }
            self.loop_occurrences.append(loop_info)
            self.active_handoff = self.create_handoff(
                stop_reason=self.stop_reason,
                question=(
                    f"Agent detected a repeating loop on action '{loop_res.fingerprint}'. "
                    "Would you like to provide alternative instructions or parameters?"
                ),
                suggested_alternatives=[
                    "Change search date or airports",
                    "Choose an alternative flight manually",
                    "Cancel current operation",
                ],
            )
            return PreCheckResult(
                proceed=False,
                status="LOOP_DETECTED",
                message=loop_res.reason,
                details=loop_info,
                handoff=self.active_handoff,
            )

        return PreCheckResult(proceed=True, status="ALLOWED", message="All safety checks passed.")

    def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        observation: Any,
    ) -> None:
        """Record observation and update execution audit log."""
        self.tool_calls_count += 1
        self.attempted_actions.append({
            "step": self.step_counter,
            "tool_name": tool_name,
            "arguments": arguments,
            "observation": observation,
        })

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Safely execute a tool through the full harness lifecycle.

        If blocked or stopped, returns a structured observation without invoking the tool.
        """
        pre_check = self.before_tool_call(tool_name, arguments)

        if not pre_check.proceed:
            structured_block = {
                "status": pre_check.status,
                "message": pre_check.message,
                "details": pre_check.details,
                "suggestions": [
                    "Review requirements before retrying action.",
                    pre_check.message,
                ],
                "next_action": "handle_blocked_action" if not self.stopped else "stop_and_handoff",
            }
            if pre_check.handoff:
                structured_block["handoff"] = pre_check.handoff.to_dict()

            # Record blocked attempt
            self.tool_calls_count += 1
            self.attempted_actions.append({
                "step": self.step_counter,
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": structured_block,
            })
            return structured_block

        # Tool is allowed; invoke underlying implementation
        dispatcher = self._raw_dispatchers.get(tool_name)
        if not dispatcher:
            obs = {
                "status": "ERROR",
                "message": f"Unknown tool '{tool_name}'.",
                "next_action": "stop",
            }
        else:
            obs = dispatcher(arguments)

        self.after_tool_call(tool_name, arguments, obs)
        return obs

    def check_completion(
        self,
        task_type: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Verify completion against real database state."""
        if task_type == "booking":
            res = check_booking_completed(
                booking_id=kwargs.get("booking_id"),
                expected_flight_id=kwargs.get("expected_flight_id"),
                expected_passenger=kwargs.get("expected_passenger"),
            )
        elif task_type == "cancellation":
            res = check_cancellation_completed(
                booking_id=kwargs.get("booking_id")
            )
        elif task_type == "search":
            res = check_search_completed(
                search_results=kwargs.get("search_results")
            )
        else:
            res = CompletionResult(
                completed=False,
                reason=f"Unknown task type '{task_type}' for completion verification.",
            )

        if res.completed:
            self.completion_verified = True
        return res

    def create_handoff(
        self,
        stop_reason: str,
        question: str,
        suggested_alternatives: Optional[List[str]] = None,
    ) -> HandoffPackage:
        """Create a full handoff package with current state snapshot."""
        return create_handoff(
            stop_reason=stop_reason,
            attempted_actions=list(self.attempted_actions),
            current_state={
                "step_counter": self.step_counter,
                "model_calls": self.model_calls_count,
                "tool_calls": self.tool_calls_count,
                "booking_completed": self.completion_verified,
                "permission_blocks": len(self.permission_blocks),
                "constraint_violations": len(self.constraint_violations),
                "loops_detected": len(self.loop_occurrences),
            },
            question_for_human=question,
            suggested_alternatives=suggested_alternatives,
        )

    def should_stop(self) -> Tuple[bool, Optional[str]]:
        """Return whether execution has been terminated by any safety mechanism."""
        return self.stopped, self.stop_reason

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a structured summary of the current harness state."""
        return {
            "steps": self.step_counter,
            "model_calls": self.model_calls_count,
            "tool_calls": self.tool_calls_count,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "completion_verified": self.completion_verified,
            "permission_blocks_count": len(self.permission_blocks),
            "constraint_violations_count": len(self.constraint_violations),
            "loop_occurrences_count": len(self.loop_occurrences),
            "has_active_handoff": self.active_handoff is not None,
        }
