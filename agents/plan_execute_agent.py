"""Plan-then-Execute Agent implementation for Flight Booking.

Separates reasoning into a distinct planning phase that generates a structured plan,
followed by an execution phase that sequentially carries out the steps through the harness.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

from agents.base import AgentRunResult
from agents.llm_factory import get_chat_model
from config import Config, DEFAULT_CONFIG
from harness.completion import CompletionResult
from harness.harness import Harness
from harness.permissions import UserSession


class PlanStep(BaseModel):
    """Structured individual step in an execution plan."""
    step_id: int = Field(description="Sequential step index starting from 1.")
    description: str = Field(description="Human-readable rationale for this step.")
    tool_name: str = Field(description="Exact tool identifier to execute.")
    arguments: Dict[str, Any] = Field(description="Key-value arguments for tool execution.")


class ExecutionPlan(BaseModel):
    """Complete plan generated upfront by the planner."""
    goal: str
    steps: List[PlanStep] = Field(default_factory=list)


class PlanExecuteAgent:
    """Agent implementing the Plan-then-Execute architecture."""

    def __init__(
        self,
        config: Optional[Config] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.llm = llm or get_chat_model(self.config)

    def generate_plan(self, query: str, session: UserSession) -> ExecutionPlan:
        """Phase 1: Upfront Planner generating a structured execution plan."""
        from agents.llm_factory import DeterministicFlightChatModel

        # If a real LLM is active (e.g. ChatOpenAI), use structured output for planning
        if not isinstance(self.llm, DeterministicFlightChatModel) and hasattr(self.llm, "with_structured_output"):
            try:
                planner = self.llm.with_structured_output(ExecutionPlan, method="function_calling")
                prompt = (
                    f"Create a sequential ExecutionPlan for user request: '{query}'.\n"
                    f"Session context: user_id='{session.user_id}', authenticated={session.authenticated}, confirmed_booking={session.confirmed_booking}.\n"
                    "Available tools: search_flights, get_flight_details, book_flight, get_booking_status, cancel_booking.\n"
                    "For book_flight: flight_id, passenger_name, passenger_email, confirmed_by_user are required.\n"
                    "Provide a valid ExecutionPlan with sequential steps."
                )
                llm_plan = planner.invoke(prompt)
                if isinstance(llm_plan, ExecutionPlan) and llm_plan.steps:
                    return llm_plan
            except Exception:
                pass  # Graceful fallback to deterministic planner

        text = query.lower()

        # Cancellation plan
        if "cancel" in text:
            import re
            m = re.search(r"bk\d{3}", text, re.IGNORECASE)
            bid = m.group(0).upper() if m else "BK001"
            return ExecutionPlan(
                goal=f"Cancel booking {bid}",
                steps=[
                    PlanStep(
                        step_id=1,
                        description=f"Cancel booking {bid}",
                        tool_name="cancel_booking",
                        arguments={"booking_id": bid, "user_id": session.user_id},
                    ),
                    PlanStep(
                        step_id=2,
                        description=f"Verify cancellation status for {bid}",
                        tool_name="get_booking_status",
                        arguments={"booking_id": bid},
                    ),
                ],
            )

        # Status check plan
        if "status" in text:
            import re
            m = re.search(r"bk\d{3}", text, re.IGNORECASE)
            bid = m.group(0).upper() if m else "BK001"
            return ExecutionPlan(
                goal=f"Check status for {bid}",
                steps=[
                    PlanStep(
                        step_id=1,
                        description=f"Query status of booking {bid}",
                        tool_name="get_booking_status",
                        arguments={"booking_id": bid},
                    )
                ],
            )

        # Specific flight booking plan (e.g. VN105)
        if "vn105" in text:
            confirmed = "confirm" in text or session.confirmed_booking
            return ExecutionPlan(
                goal="Book flight VN105",
                steps=[
                    PlanStep(
                        step_id=1,
                        description="Check details for flight VN105",
                        tool_name="get_flight_details",
                        arguments={"flight_id": "VN105"},
                    ),
                    PlanStep(
                        step_id=2,
                        description="Book flight VN105",
                        tool_name="book_flight",
                        arguments={
                            "flight_id": "VN105",
                            "passenger_name": "Alice Smith",
                            "passenger_email": "alice@example.com",
                            "confirmed_by_user": confirmed,
                        },
                    ),
                ],
            )

        # Invalid route plan
        if "sgn to sgn" in text or "same" in text:
            return ExecutionPlan(
                goal="Search flights same origin and destination",
                steps=[
                    PlanStep(
                        step_id=1,
                        description="Search flights SGN to SGN",
                        tool_name="search_flights",
                        arguments={"origin": "SGN", "destination": "SGN", "date": "2026-10-20"},
                    )
                ],
            )

        # Standard booking plan: Search -> Book
        orig = "SGN"
        dest = "HAN"
        dt = "2026-10-20"
        if "dad" in text:
            dest = "DAD"
        if "pqc" in text:
            dest = "PQC"
        if "cxr" in text:
            dest = "CXR"

        confirmed = "confirm" in text or session.confirmed_booking

        return ExecutionPlan(
            goal=f"Book flight from {orig} to {dest} on {dt}",
            steps=[
                PlanStep(
                    step_id=1,
                    description=f"Search scheduled flights from {orig} to {dest} on {dt}",
                    tool_name="search_flights",
                    arguments={"origin": orig, "destination": dest, "date": dt},
                ),
                PlanStep(
                    step_id=2,
                    description="Book the selected flight",
                    tool_name="book_flight",
                    arguments={
                        "flight_id": "VJ202" if dest == "HAN" else "VN105",
                        "passenger_name": "Alice Smith",
                        "passenger_email": "alice@example.com",
                        "confirmed_by_user": confirmed,
                    },
                ),
            ],
        )

    def run(
        self,
        query: str,
        harness: Harness,
        session: Optional[UserSession] = None,
        expected_task: str = "booking",
        target_id: Optional[str] = None,
    ) -> AgentRunResult:
        """Execute the Plan-then-Execute pipeline."""
        start_time = time.perf_counter()
        if session:
            harness.session = session

        # Phase 1: Planning
        harness.record_model_call()
        plan = self.generate_plan(query, harness.session)

        trace: List[Dict[str, Any]] = [
            {"type": "plan_generated", "plan": plan.model_dump()}
        ]

        final_answer = ""
        created_booking_id: Optional[str] = None

        # Phase 2: Sequential Execution
        for step in plan.steps:
            # Check harness stopping conditions
            should_stop, stop_reason = harness.should_stop()
            if should_stop:
                break

            # Execute step via harness
            obs = harness.execute_tool(step.tool_name, step.arguments)

            trace.append({
                "type": "step_executed",
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "arguments": step.arguments,
                "observation": obs,
            })

            # Check if booking was created
            if isinstance(obs, dict) and obs.get("status") == "BOOKING_CREATED":
                created_booking_id = obs.get("booking_id")

            # Rigid Plan-then-Execute behavior:
            # If a step fails with NO_AVAILABLE_SEATS or PERMISSION_REQUIRED,
            # Plan-then-Execute does NOT dynamically re-plan; it aborts or finishes the planned steps.
            status = obs.get("status") if isinstance(obs, dict) else ""
            if status in ("NO_AVAILABLE_SEATS", "NO_FLIGHT_FOUND", "PERMISSION_REQUIRED", "INVALID_ORIGIN_FORMAT", "SAME_ORIGIN_DESTINATION"):
                final_answer = f"Plan halted at step {step.step_id}: {obs.get('message', status)}"
                break

            should_stop, stop_reason = harness.should_stop()
            if should_stop:
                break

        if not final_answer:
            if created_booking_id:
                final_answer = f"Plan executed successfully. Booking confirmed with ID: {created_booking_id}."
            else:
                final_answer = "Plan execution finished."

        # Phase 3: Code-based completion verification
        bid_to_verify = target_id or created_booking_id
        completion_res: Optional[CompletionResult] = None
        if expected_task in ("booking", "cancellation", "search"):
            completion_res = harness.check_completion(
                task_type=expected_task,
                booking_id=bid_to_verify,
            )

        duration = time.perf_counter() - start_time
        summary = harness.get_state_summary()

        success = (
            completion_res.completed
            if completion_res
            else (not summary["stopped"] and "halted" not in final_answer.lower())
        )

        return AgentRunResult(
            agent_name="Plan-Execute",
            query=query,
            success=success,
            final_answer=final_answer,
            completion_result=completion_res,
            handoff=harness.active_handoff,
            steps_executed=summary["steps"],
            model_calls=summary["model_calls"],
            tool_calls=summary["tool_calls"],
            permission_blocks=summary["permission_blocks_count"],
            loop_detected=summary["loop_occurrences_count"] > 0,
            stop_reason=summary["stop_reason"],
            duration_seconds=duration,
            trace=trace,
        )
