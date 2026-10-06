"""Flight booking tools implemented as LangChain structured tools.

Every tool returns structured observations with clear status codes, business details,
actionable suggestions, and recommended next steps to assist agent decision-making.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from data.flights import (
    find_flights,
    get_flight_by_id,
    create_booking,
    get_booking,
    cancel_booking_in_store,
)


# ============================================================================
# Pydantic Input Schemas
# ============================================================================

class SearchFlightsInput(BaseModel):
    """Input for searching scheduled flights."""
    origin: str = Field(description="3-letter IATA airport code of origin, e.g. 'SGN'")
    destination: str = Field(description="3-letter IATA airport code of destination, e.g. 'HAN'")
    date: str = Field(description="Departure date in YYYY-MM-DD format, e.g. '2026-10-20'")


class FlightDetailsInput(BaseModel):
    """Input for retrieving specific flight details."""
    flight_id: str = Field(description="Flight identifier code, e.g. 'VN101'")


class BookFlightInput(BaseModel):
    """Input for booking a flight ticket."""
    flight_id: str = Field(description="Flight identifier code to book, e.g. 'VN101'")
    passenger_name: str = Field(description="Full name of the passenger, e.g. 'Alice Smith'")
    passenger_email: str = Field(description="Email address of the passenger, e.g. 'alice@example.com'")
    confirmed_by_user: bool = Field(
        default=False,
        description="Explicit user confirmation flag. Must be True to proceed with booking.",
    )


class BookingStatusInput(BaseModel):
    """Input for checking booking status."""
    booking_id: str = Field(description="Unique booking confirmation ID, e.g. 'BK001'")


class CancelBookingInput(BaseModel):
    """Input for cancelling an existing booking."""
    booking_id: str = Field(description="Unique booking confirmation ID to cancel, e.g. 'BK001'")
    user_id: str = Field(
        default="guest",
        description="Identifier of the user requesting cancellation.",
    )


# ============================================================================
# Core Business Implementation Functions (returning structured dicts)
# ============================================================================

def execute_search_flights(origin: str, destination: str, date: str) -> Dict[str, Any]:
    """Execute flight search with rich structured observation."""
    origin_clean = origin.strip().upper()
    dest_clean = destination.strip().upper()
    date_clean = date.strip()

    flights = find_flights(origin_clean, dest_clean, date_clean)

    if flights:
        return {
            "status": "SUCCESS",
            "count": len(flights),
            "origin": origin_clean,
            "destination": dest_clean,
            "date": date_clean,
            "flights": flights,
            "suggestions": [
                f"Found {len(flights)} matching flight(s). Compare prices and departure times.",
                "Verify seat availability before initiating booking.",
            ],
            "next_action": "select_and_confirm_flight",
        }

    return {
        "status": "NO_FLIGHT_FOUND",
        "origin": origin_clean,
        "destination": dest_clean,
        "date": date_clean,
        "flights": [],
        "suggestions": [
            f"No direct flights from {origin_clean} to {dest_clean} on {date_clean}.",
            "Try searching alternative dates such as 2026-10-21.",
            "Verify airport codes (e.g., SGN, HAN, DAD, CXR, PQC).",
        ],
        "next_action": "search_alternative",
    }


def execute_get_flight_details(flight_id: str) -> Dict[str, Any]:
    """Execute flight details lookup with rich structured observation."""
    fid = flight_id.strip().upper()
    flight = get_flight_by_id(fid)

    if flight:
        return {
            "status": "SUCCESS",
            "flight_id": fid,
            "flight": flight,
            "available_seats": flight["available_seats"],
            "price": flight["price"],
            "suggestions": (
                ["Seats available. Ask user for confirmation to book."]
                if flight["available_seats"] > 0
                else [
                    f"Flight {fid} has 0 seats available.",
                    "Search for alternative flights on the same or adjacent dates.",
                ]
            ),
            "next_action": (
                "request_booking_confirmation"
                if flight["available_seats"] > 0
                else "search_alternative"
            ),
        }

    return {
        "status": "NO_FLIGHT_FOUND",
        "flight_id": fid,
        "suggestions": [
            f"Flight identifier '{fid}' does not exist in the catalog.",
            "Call search_flights to see valid available flight IDs.",
        ],
        "next_action": "search_flights",
    }


def execute_book_flight(
    flight_id: str,
    passenger_name: str,
    passenger_email: str,
    confirmed_by_user: bool = False,
    user_id: str = "guest",
) -> Dict[str, Any]:
    """Execute booking logic with seat validation and structured output."""
    fid = flight_id.strip().upper()
    flight = get_flight_by_id(fid)

    if not flight:
        return {
            "status": "NO_FLIGHT_FOUND",
            "flight_id": fid,
            "suggestions": [
                f"Cannot book non-existent flight '{fid}'.",
                "Search flights first to pick a valid flight ID.",
            ],
            "next_action": "search_flights",
        }

    if flight["available_seats"] <= 0:
        return {
            "status": "NO_AVAILABLE_SEATS",
            "flight_id": fid,
            "available_seats": 0,
            "origin": flight["origin"],
            "destination": flight["destination"],
            "date": flight["date"],
            "suggestions": [
                f"Flight {fid} is sold out (0 available seats).",
                f"Search alternative flights between {flight['origin']} and {flight['destination']}.",
            ],
            "next_action": "search_alternative",
        }

    # Create the booking
    booking = create_booking(
        flight_id=fid,
        passenger_name=passenger_name,
        passenger_email=passenger_email,
        user_id=user_id,
    )

    return {
        "status": "BOOKING_CREATED",
        "booking_id": booking["booking_id"],
        "flight_id": fid,
        "passenger_name": booking["passenger_name"],
        "passenger_email": booking["passenger_email"],
        "price": booking["price"],
        "booking_status": booking["status"],
        "suggestions": [
            f"Booking {booking['booking_id']} successfully confirmed for {booking['passenger_name']}.",
            "Verify booking completion through harness verification.",
        ],
        "next_action": "verify_completion",
    }


def execute_get_booking_status(booking_id: str) -> Dict[str, Any]:
    """Execute booking status query with structured observation."""
    bid = booking_id.strip().upper()
    booking = get_booking(bid)

    if booking:
        return {
            "status": "SUCCESS",
            "booking_id": bid,
            "booking": booking,
            "booking_status": booking["status"],
            "suggestions": [
                f"Booking {bid} status is {booking['status']}.",
            ],
            "next_action": "none",
        }

    return {
        "status": "BOOKING_NOT_FOUND",
        "booking_id": bid,
        "suggestions": [
            f"No booking found matching ID '{bid}'.",
            "Verify the booking ID was typed correctly.",
        ],
        "next_action": "ask_human",
    }


def execute_cancel_booking(booking_id: str, user_id: str = "guest") -> Dict[str, Any]:
    """Execute booking cancellation with structured observation."""
    bid = booking_id.strip().upper()
    booking = get_booking(bid)

    if not booking:
        return {
            "status": "BOOKING_NOT_FOUND",
            "booking_id": bid,
            "suggestions": [
                f"Cannot cancel non-existent booking '{bid}'.",
            ],
            "next_action": "ask_human",
        }

    try:
        updated = cancel_booking_in_store(bid, user_id)
        return {
            "status": "CANCELLED",
            "booking_id": bid,
            "previous_status": booking["status"],
            "current_status": updated["status"],
            "flight_id": updated["flight_id"],
            "suggestions": [
                f"Booking {bid} has been successfully cancelled and seat restored.",
            ],
            "next_action": "verify_cancellation",
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "booking_id": bid,
            "message": str(exc),
            "suggestions": ["Check booking details and retry."],
            "next_action": "ask_human",
        }


# ============================================================================
# LangChain Tools Decorated with @tool
# ============================================================================

@tool(args_schema=SearchFlightsInput)
def search_flights(origin: str, destination: str, date: str) -> Dict[str, Any]:
    """Search for available flights between two airports on a specific departure date.
    Returns structured flight list, price, seats, and next actions.
    """
    return execute_search_flights(origin, destination, date)


@tool(args_schema=FlightDetailsInput)
def get_flight_details(flight_id: str) -> Dict[str, Any]:
    """Retrieve detailed schedule, pricing, and seat availability for a specific flight ID."""
    return execute_get_flight_details(flight_id)


@tool(args_schema=BookFlightInput)
def book_flight(
    flight_id: str,
    passenger_name: str,
    passenger_email: str,
    confirmed_by_user: bool = False,
) -> Dict[str, Any]:
    """Book a seat on a flight. Requires user confirmation before execution."""
    return execute_book_flight(
        flight_id=flight_id,
        passenger_name=passenger_name,
        passenger_email=passenger_email,
        confirmed_by_user=confirmed_by_user,
    )


@tool(args_schema=BookingStatusInput)
def get_booking_status(booking_id: str) -> Dict[str, Any]:
    """Check the real-time reservation status of a flight booking by its booking ID."""
    return execute_get_booking_status(booking_id)


@tool(args_schema=CancelBookingInput)
def cancel_booking(booking_id: str, user_id: str = "guest") -> Dict[str, Any]:
    """Cancel an active booking by its booking ID. Requires authenticated user."""
    return execute_cancel_booking(booking_id, user_id)


# Shared tool registry available to all agents
FLIGHT_TOOLS = [
    search_flights,
    get_flight_details,
    book_flight,
    get_booking_status,
    cancel_booking,
]
