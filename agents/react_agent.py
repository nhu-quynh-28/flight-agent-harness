"""ReAct (Reasoning + Acting) Agent implementation for Flight Booking.

Executes an iterative thought -> action -> observation -> thought loop,
routing every action through the shared safety harness.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel

from agents.base import AgentRunResult
from agents.llm_factory import get_chat_model
from config import Config, DEFAULT_CONFIG
from harness.completion import CompletionResult
from harness.harness import Harness
from harness.permissions import UserSession
from tools.flight_tools import FLIGHT_TOOLS


REACT_SYSTEM_PROMPT = (
    "You are a helpful flight booking assistant. "
    "You have access to tools for searching flights, viewing flight details, booking flights, "
    "checking booking status, and cancelling bookings.\n"
    "Important rules:\n"
    "1. When booking a flight, passenger name, passenger email, and flight_id are required.\n"
    "2. Booking requires user confirmation. If the user explicitly confirmed, set confirmed_by_user=True.\n"
    "3. If a flight has no available seats or no flights are found, observe suggestions and consider searching alternative dates or flights.\n"
    "4. When finished or when asking the user a question, respond clearly."
)


class ReactAgent:
    """Agent implementing the ReAct pattern with harness-governed execution."""

    def __init__(
        self,
        config: Optional[Config] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.llm = llm or get_chat_model(self.config)
        # Bind tools if model supports tool binding
        try:
            self.model_with_tools = self.llm.bind_tools(FLIGHT_TOOLS)
        except Exception:
            self.model_with_tools = self.llm

    def run(
        self,
        query: str,
        harness: Harness,
        session: Optional[UserSession] = None,
        expected_task: str = "booking",
        target_id: Optional[str] = None,
    ) -> AgentRunResult:
        """Run the ReAct agent loop on the provided query under harness control."""
        start_time = time.perf_counter()
        if session:
            harness.session = session

        from langchain_core.messages import SystemMessage
        messages: List[BaseMessage] = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        trace: List[Dict[str, Any]] = []
        final_answer = ""

        # Iterative ReAct loop bounded by harness framework limit
        while True:
            # Check if harness requested a stop (budget, hard limit, or loop)
            should_stop, stop_reason = harness.should_stop()
            if should_stop:
                break

            # 1. Record model call and verify model call budget
            if not harness.record_model_call():
                break

            # 2. Invoke model for reasoning and next action decision
            try:
                ai_message: AIMessage = self.model_with_tools.invoke(messages)
            except Exception as exc:
                final_answer = f"Model execution error: {exc}"
                break

            messages.append(ai_message)
            trace.append({
                "type": "thought_and_action",
                "content": ai_message.content,
                "tool_calls": getattr(ai_message, "tool_calls", []),
            })

            # Check if model produced a final textual answer without tool calls
            tool_calls = getattr(ai_message, "tool_calls", [])
            if not tool_calls:
                final_answer = str(ai_message.content)
                break

            # 3. Execute tool calls through the safety harness
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                call_id = tc.get("id", f"call_{len(trace)}")

                # Execute action via harness
                obs = harness.execute_tool(tool_name, tool_args)

                # Append tool observation message
                messages.append(
                    ToolMessage(
                        content=json.dumps(obs),
                        tool_call_id=call_id,
                        name=tool_name,
                    )
                )
                trace.append({
                    "type": "observation",
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "observation": obs,
                })

                # Check if harness triggered a stop on this tool execution
                should_stop, stop_reason = harness.should_stop()
                if should_stop:
                    break

            if should_stop:
                break

        # Verification phase: Verify completion through code in database
        completion_res: Optional[CompletionResult] = None
        bid_to_verify = target_id
        if not bid_to_verify:
            # Extract created booking ID from attempted actions if available
            for action in reversed(harness.attempted_actions):
                obs = action.get("observation", {})
                if isinstance(obs, dict) and "booking_id" in obs and obs.get("status") == "BOOKING_CREATED":
                    bid_to_verify = obs["booking_id"]
                    break

        if expected_task in ("booking", "cancellation", "search"):
            completion_res = harness.check_completion(
                task_type=expected_task,
                booking_id=bid_to_verify,
            )

        duration = time.perf_counter() - start_time
        summary = harness.get_state_summary()

        # Determine success
        success = (
            completion_res.completed
            if completion_res
            else (not summary["stopped"] and bool(final_answer))
        )

        return AgentRunResult(
            agent_name="ReAct",
            query=query,
            success=success,
            final_answer=final_answer or ("Stopped: " + (summary["stop_reason"] or "Unknown")),
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
