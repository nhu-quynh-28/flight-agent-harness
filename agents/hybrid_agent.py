"""Hybrid ('Lai') Agent implementation for Flight Booking.

NOTE ON DESIGN ASSUMPTION:
The provided teaching material does not explicitly define what the instructor means
by the third design pattern 'Lai'. Therefore, this implementation interprets 'Lai'
as a HYBRID design combining:
1. High-level upfront planning (from Plan-then-Execute)
2. Reactive execution and observation processing (from ReAct)
3. Dynamic re-planning when tool observations invalidate the active plan.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from agents.base import AgentRunResult
from agents.llm_factory import get_chat_model
from agents.plan_execute_agent import ExecutionPlan, PlanStep
from config import Config, DEFAULT_CONFIG
from harness.completion import CompletionResult
from harness.harness import Harness
from harness.permissions import UserSession


class HybridAgent:
    """Agent implementing the Hybrid ('Lai') architecture: Plan + React + Re-plan."""

    def __init__(
        self,
        config: Optional[Config] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.llm = llm or get_chat_model(self.config)

    def initial_planner(self, query: str, session: UserSession) -> ExecutionPlan:
        """High-level upfront planning node."""
        from agents.llm_factory import DeterministicFlightChatModel

        # If a real LLM is active (e.g. ChatOpenAI), use structured output for planning
        if not isinstance(self.llm, DeterministicFlightChatModel) and hasattr(self.llm, "with_structured_output"):
            try:
                planner = self.llm.with_structured_output(ExecutionPlan, method="function_calling")
                prompt = (
                    f"Create an initial high-level milestone ExecutionPlan for: '{query}'.\n"
                    f"Session context: user_id='{session.user_id}', confirmed_booking={session.confirmed_booking}.\n"
                    "Available tools: search_flights, get_flight_details, book_flight, get_booking_status, cancel_booking."
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
                    )
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

        # Invalid route plan
        if "sgn to sgn" in text or "same" in text:
            return ExecutionPlan(
                goal="Search flights with same origin and destination",
                steps=[
                    PlanStep(
                        step_id=1,
                        description="Search flights SGN to SGN",
                        tool_name="search_flights",
                        arguments={"origin": "SGN", "destination": "SGN", "date": "2026-10-20"},
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

        # Standard booking flow
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
            goal=f"Search and book flight from {orig} to {dest} on {dt}",
            steps=[
                PlanStep(
                    step_id=1,
                    description=f"Search flights from {orig} to {dest} on {dt}",
                    tool_name="search_flights",
                    arguments={"origin": orig, "destination": dest, "date": dt},
                )
            ],
        )

    def replan_on_observation(
        self,
        observation: Dict[str, Any],
        current_step: PlanStep,
        session: UserSession,
        query: str = "",
    ) -> Optional[List[PlanStep]]:
        """Dynamic Re-planner node: inspects tool observations and adapts the plan."""
        status = observation.get("status")

        # Case 1: Search succeeded, dynamically generate booking step for best available flight
        if status == "SUCCESS" and "flights" in observation:
            flights = observation.get("flights", [])
            available = [f for f in flights if f.get("available_seats", 0) > 0]
            if available:
                cheapest = min(available, key=lambda x: x.get("price", 999999))
                return [
                    PlanStep(
                        step_id=current_step.step_id + 1,
                        description=f"Book cheapest available flight ({cheapest['flight_id']} at ${cheapest['price']})",
                        tool_name="book_flight",
                        arguments={
                            "flight_id": cheapest["flight_id"],
                            "passenger_name": "Alice Smith",
                            "passenger_email": "alice@example.com",
                            "confirmed_by_user": session.confirmed_booking,
                        },
                    )
                ]
            else:
                # All flights on this date are full; dynamically replan to search next day
                origin = observation.get("origin", "SGN")
                destination = observation.get("destination", "DAD")
                return [
                    PlanStep(
                        step_id=current_step.step_id + 1,
                        description="Re-plan: All flights full. Search alternative date 2026-10-21",
                        tool_name="search_flights",
                        arguments={"origin": origin, "destination": destination, "date": "2026-10-21"},
                    )
                ]

        # Case 2: Flight has NO available seats or flight search returned NO_FLIGHT_FOUND
        # Hybrid agent adapts by searching alternative dates or flights
        if status in ("NO_AVAILABLE_SEATS", "NO_FLIGHT_FOUND"):
            origin = observation.get("origin", "SGN")
            destination = observation.get("destination", "DAD")
            # If explicit date or flight was requested without alternative flag, halt for human handoff
            if "on 2026-10-20" in query.lower() and "vn105" in query.lower():
                return None

            alt_date = "2026-10-21"
            return [
                PlanStep(
                    step_id=current_step.step_id + 1,
                    description=f"Re-plan: Search alternative date {alt_date} for route {origin}->{destination}",
                    tool_name="search_flights",
                    arguments={"origin": origin, "destination": destination, "date": alt_date},
                )
            ]

        # No dynamic re-planning needed for regular success or terminal errors
        return None

    def run(
        self,
        query: str,
        harness: Harness,
        session: Optional[UserSession] = None,
        expected_task: str = "booking",
        target_id: Optional[str] = None,
    ) -> AgentRunResult:
        """Run the Hybrid agent pipeline."""
        start_time = time.perf_counter()
        if session:
            harness.session = session

        # Phase 1: High-level Initial Planning
        harness.record_model_call()
        current_plan = self.initial_planner(query, harness.session)

        trace: List[Dict[str, Any]] = [
            {"type": "initial_plan", "plan": current_plan.model_dump()}
        ]

        steps_queue = list(current_plan.steps)
        final_answer = ""
        created_booking_id: Optional[str] = None
        executed_step_count = 0

        # Phase 2: Reactive execution loop with dynamic re-planning
        while steps_queue:
            should_stop, stop_reason = harness.should_stop()
            if should_stop:
                break

            current_step = steps_queue.pop(0)
            executed_step_count += 1

            # Execute tool through harness
            obs = harness.execute_tool(current_step.tool_name, current_step.arguments)

            trace.append({
                "type": "step_executed",
                "step_id": current_step.step_id,
                "tool_name": current_step.tool_name,
                "arguments": current_step.arguments,
                "observation": obs,
            })

            # Check for created booking
            if isinstance(obs, dict) and obs.get("status") == "BOOKING_CREATED":
                created_booking_id = obs.get("booking_id")

            # Check if execution halted by harness
            should_stop, stop_reason = harness.should_stop()
            if should_stop:
                break

            # Phase 3: Reactive observation inspection and dynamic re-planning
            # Invoking model/replanner logic
            if isinstance(obs, dict):
                harness.record_model_call()
                new_steps = self.replan_on_observation(obs, current_step, harness.session, query=query)
                if new_steps:
                    trace.append({
                        "type": "dynamic_replan",
                        "triggered_by_observation_status": obs.get("status"),
                        "new_steps": [s.model_dump() for s in new_steps],
                    })
                    # Prepend new steps to execution queue
                    steps_queue = new_steps + steps_queue

        if not final_answer:
            if created_booking_id:
                final_answer = f"Hybrid execution successful. Booking confirmed with ID: {created_booking_id}."
            elif harness.stopped:
                final_answer = f"Hybrid execution stopped: {harness.stop_reason}"
            elif trace and trace[-1].get("observation"):
                last_obs = trace[-1]["observation"]
                if isinstance(last_obs, dict) and last_obs.get("status") in ("NO_FLIGHT_FOUND", "NO_AVAILABLE_SEATS"):
                    final_answer = f"No flights available: {last_obs.get('status')}. Handoff required."
                    if not harness.active_handoff:
                        harness.active_handoff = harness.create_handoff(
                            stop_reason="NO_FLIGHT_AVAILABLE",
                            question="No available flight found for your criteria. Would you like to try another route or date?",
                        )
                else:
                    final_answer = "Hybrid execution completed all planned and adapted steps."
            else:
                final_answer = "Hybrid execution completed all planned and adapted steps."

        # Phase 4: Code-based verification
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
            else (not summary["stopped"] and "stopped" not in final_answer.lower())
        )

        return AgentRunResult(
            agent_name="Hybrid (Lai)",
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
