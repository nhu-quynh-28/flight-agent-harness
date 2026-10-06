"""LLM Factory providing real OpenAI chat models or deterministic mock models.

Allows the entire suite to run either against real OpenAI endpoints when an API key
is configured, or fully offline deterministically with a simulated intelligent model.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from config import Config, DEFAULT_CONFIG


class DeterministicFlightChatModel(BaseChatModel):
    """Deterministic chat model implementing realistic flight booking agent behaviors.

    Used when no OpenAI API key is configured, enabling automated evaluation,
    testing, and grading under 100% deterministic conditions.
    """

    model_name: str = "deterministic-mock-v1"

    @property
    def _llm_type(self) -> str:
        return "deterministic-flight-mock"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Inspect conversation history and produce realistic next actions or responses."""
        if not messages:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Hello! How may I assist with your flight?"))])

        # Find latest human message and latest tool message
        latest_human = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                latest_human = str(m.content)
                break

        last_msg = messages[-1]

        # Case 1: Just received a ToolMessage (observing tool output)
        if isinstance(last_msg, ToolMessage):
            obs_text = last_msg.content
            try:
                obs = json.loads(obs_text) if isinstance(obs_text, str) else obs_text
            except Exception:
                obs = {"status": "UNKNOWN", "raw": obs_text}

            status = obs.get("status", "")

            # If booking created successfully
            if status == "BOOKING_CREATED":
                bid = obs.get("booking_id", "N/A")
                pname = obs.get("passenger_name", "Passenger")
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=f"Your flight has been successfully booked! Booking ID: {bid} for {pname}."
                            )
                        )
                    ]
                )

            # If cancellation succeeded
            if status == "CANCELLED":
                bid = obs.get("booking_id", "N/A")
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=f"Booking {bid} has been successfully cancelled."
                            )
                        )
                    ]
                )

            # If permission required
            if status == "PERMISSION_REQUIRED":
                code = obs.get("details", {}).get("code", "CONFIRMATION_REQUIRED")
                if code == "CONFIRMATION_REQUIRED":
                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=AIMessage(
                                    content="User confirmation is required before proceeding with this booking. Please confirm to proceed."
                                )
                            )
                        ]
                    )
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=f"Action blocked by permission policy: {obs.get('message')}"
                            )
                        )
                    ]
                )

            # If constraint violation or invalid request
            if status in ("INVALID_ORIGIN_FORMAT", "SAME_ORIGIN_DESTINATION", "PAST_DATE", "INVALID_DATE_FORMAT"):
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=f"Request violated flight constraints: {obs.get('message')}"
                            )
                        )
                    ]
                )

            # If no available seats
            if status == "NO_AVAILABLE_SEATS":
                fid = obs.get("flight_id", "")
                if "on 2026-10-20" in latest_human.lower() and "vn105" in latest_human.lower() and "alternative" not in latest_human.lower():
                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=AIMessage(
                                    content=f"Flight {fid} is fully booked (0 available seats). Would you like to check flights on an alternative date?"
                                )
                            )
                        ]
                    )

                # Otherwise, recover by searching alternative date 2026-10-21
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=f"Flight {fid} has no available seats. Checking alternative flights on 2026-10-21.",
                                tool_calls=[
                                    {
                                        "name": "search_flights",
                                        "args": {
                                            "origin": obs.get("origin", "SGN"),
                                            "destination": obs.get("destination", "DAD"),
                                            "date": "2026-10-21",
                                        },
                                        "id": "call_alt_search_01",
                                    }
                                ],
                            )
                        )
                    ]
                )

            # If search succeeded with flights
            if status == "SUCCESS" and "flights" in obs:
                flights = obs.get("flights", [])
                if flights:
                    # Select cheapest flight or first flight with seats
                    available = [f for f in flights if f.get("available_seats", 0) > 0]
                    if available:
                        cheapest = min(available, key=lambda x: x.get("price", 999999))
                        # Decide to book if confirmation is implied, or request confirmation
                        confirmed = "confirm" in latest_human.lower() or "confirmed" in latest_human.lower()
                        return ChatResult(
                            generations=[
                                ChatGeneration(
                                    message=AIMessage(
                                        content=f"Found cheapest available flight: {cheapest['flight_id']} at ${cheapest['price']}.",
                                        tool_calls=[
                                            {
                                                "name": "book_flight",
                                                "args": {
                                                    "flight_id": cheapest["flight_id"],
                                                    "passenger_name": "Alice Smith",
                                                    "passenger_email": "alice@example.com",
                                                    "confirmed_by_user": confirmed,
                                                },
                                                "id": "call_book_01",
                                            }
                                        ],
                                    )
                                )
                            ]
                        )
                    else:
                        # All flights on requested date are full; search alternative date 2026-10-21
                        return ChatResult(
                            generations=[
                                ChatGeneration(
                                    message=AIMessage(
                                        content="All flights on this date are fully booked. Searching alternative date 2026-10-21.",
                                        tool_calls=[
                                            {
                                                "name": "search_flights",
                                                "args": {
                                                    "origin": obs.get("origin", "SGN"),
                                                    "destination": obs.get("destination", "DAD"),
                                                    "date": "2026-10-21",
                                                },
                                                "id": "call_alt_search_02",
                                            }
                                        ],
                                    )
                                )
                            ]
                        )

            # If no flight found
            if status == "NO_FLIGHT_FOUND":
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="No flights were found for the requested route and date."
                            )
                        )
                    ]
                )

        # Case 2: Human prompt dispatch
        text = latest_human.lower()

        # Cancellation intent
        if "cancel" in text:
            # Extract booking id if present, e.g. BK001 or BK002
            import re
            m = re.search(r"bk\d{3}", text, re.IGNORECASE)
            bid = m.group(0).upper() if m else "BK001"
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=f"Initiating cancellation for booking {bid}.",
                            tool_calls=[
                                {
                                    "name": "cancel_booking",
                                    "args": {"booking_id": bid, "user_id": "user_alice"},
                                    "id": "call_cancel_01",
                                }
                            ],
                        )
                    )
                ]
            )

        # Status check intent
        if "status" in text:
            import re
            m = re.search(r"bk\d{3}", text, re.IGNORECASE)
            bid = m.group(0).upper() if m else "BK001"
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=f"Checking booking status for {bid}.",
                            tool_calls=[
                                {
                                    "name": "get_booking_status",
                                    "args": {"booking_id": bid},
                                    "id": "call_status_01",
                                }
                            ],
                        )
                    )
                ]
            )

        # Direct flight book intent with explicit flight ID (e.g. VN105)
        if "vn105" in text:
            confirmed = "confirm" in text
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="Attempting to book flight VN105.",
                            tool_calls=[
                                {
                                    "name": "book_flight",
                                    "args": {
                                        "flight_id": "VN105",
                                        "passenger_name": "Alice Smith",
                                        "passenger_email": "alice@example.com",
                                        "confirmed_by_user": confirmed,
                                    },
                                    "id": "call_book_vn105",
                                }
                            ],
                        )
                    )
                ]
            )

        # Search intent with invalid dates/airports
        if "same" in text or "sgn to sgn" in text:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="Searching flights from SGN to SGN.",
                            tool_calls=[
                                {
                                    "name": "search_flights",
                                    "args": {"origin": "SGN", "destination": "SGN", "date": "2026-10-20"},
                                    "id": "call_search_invalid",
                                }
                            ],
                        )
                    )
                ]
            )

        # General search intent (default SGN -> HAN on 2026-10-20)
        orig = "SGN"
        dest = "HAN"
        dt = "2026-10-20"
        if "dad" in text:
            dest = "DAD"
        if "pqc" in text:
            dest = "PQC"
        if "cxr" in text:
            dest = "CXR"

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=f"Searching flights from {orig} to {dest} on {dt}.",
                        tool_calls=[
                            {
                                "name": "search_flights",
                                "args": {"origin": orig, "destination": dest, "date": dt},
                                "id": "call_search_01",
                            }
                        ],
                    )
                )
            ]
        )


def get_chat_model(config: Optional[Config] = None) -> BaseChatModel:
    """Return configured ChatOpenAI if API key exists, otherwise DeterministicFlightChatModel."""
    cfg = config or DEFAULT_CONFIG
    if cfg.openai_api_key and cfg.openai_api_key.strip():
        kwargs: Dict[str, Any] = {
            "model": cfg.openai_model,
            "api_key": cfg.openai_api_key,
            "temperature": 0.0,
        }
        if cfg.openai_base_url:
            kwargs["base_url"] = cfg.openai_base_url
        return ChatOpenAI(**kwargs)

    # Offline deterministic model
    return DeterministicFlightChatModel()
