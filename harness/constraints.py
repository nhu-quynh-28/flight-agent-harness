"""Constraints as Data validation layer for the flight booking harness.

Enforces business and schema constraints declaratively as data structures,
returning structured validation reports instead of relying on generic uncaught exceptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from data.flights import get_flight_by_id
from config import DEFAULT_CONFIG


@dataclass
class ConstraintResult:
    """Structured result of a constraint validation check."""
    allowed: bool
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# Airport code format: 3 uppercase English letters
AIRPORT_REGEX = re.compile(r"^[A-Z]{3}$")
# Flight ID format: 2-3 airline letters followed by 2-4 digits (e.g. VN101, VJ202, QH303)
FLIGHT_ID_REGEX = re.compile(r"^[A-Z]{2,3}\d{2,4}$")
# Email format
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_search_constraints(arguments: Dict[str, Any]) -> ConstraintResult:
    """Validate constraints for search_flights."""
    origin = arguments.get("origin", "").strip().upper()
    destination = arguments.get("destination", "").strip().upper()
    date_str = arguments.get("date", "").strip()

    # Rule 1: Airport format
    if not AIRPORT_REGEX.match(origin):
        return ConstraintResult(
            allowed=False,
            code="INVALID_ORIGIN_FORMAT",
            message=f"Origin airport '{origin}' is invalid. Must be a 3-letter IATA code (e.g. 'SGN').",
            details={"origin": origin},
        )

    if not AIRPORT_REGEX.match(destination):
        return ConstraintResult(
            allowed=False,
            code="INVALID_DESTINATION_FORMAT",
            message=f"Destination airport '{destination}' is invalid. Must be a 3-letter IATA code (e.g. 'HAN').",
            details={"destination": destination},
        )

    # Rule 2: Origin != Destination
    if origin == destination:
        return ConstraintResult(
            allowed=False,
            code="SAME_ORIGIN_DESTINATION",
            message=f"Origin '{origin}' and destination '{destination}' cannot be the same airport.",
            details={"origin": origin, "destination": destination},
        )

    # Rule 3: Valid date format and parseability
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if dt.year < 2024:
            return ConstraintResult(
                allowed=False,
                code="PAST_DATE",
                message=f"Travel date '{date_str}' is in the past.",
                details={"date": date_str},
            )
    except ValueError:
        return ConstraintResult(
            allowed=False,
            code="INVALID_DATE_FORMAT",
            message=f"Date '{date_str}' is invalid. Must be in YYYY-MM-DD format.",
            details={"date": date_str},
        )

    return ConstraintResult(allowed=True, code="ALLOWED", message="Search constraints satisfied.")


def validate_flight_details_constraints(arguments: Dict[str, Any]) -> ConstraintResult:
    """Validate constraints for get_flight_details."""
    flight_id = arguments.get("flight_id", "").strip().upper()

    if not FLIGHT_ID_REGEX.match(flight_id):
        return ConstraintResult(
            allowed=False,
            code="INVALID_FLIGHT_ID_FORMAT",
            message=f"Flight ID '{flight_id}' is invalid. Must match pattern like 'VN101'.",
            details={"flight_id": flight_id},
        )

    return ConstraintResult(allowed=True, code="ALLOWED", message="Flight details constraints satisfied.")


def validate_book_constraints(arguments: Dict[str, Any]) -> ConstraintResult:
    """Validate constraints for book_flight."""
    flight_id = arguments.get("flight_id", "").strip().upper()
    passenger_name = arguments.get("passenger_name", "").strip()
    passenger_email = arguments.get("passenger_email", "").strip()

    # Rule 1: Flight ID format
    if not FLIGHT_ID_REGEX.match(flight_id):
        return ConstraintResult(
            allowed=False,
            code="INVALID_FLIGHT_ID_FORMAT",
            message=f"Flight ID '{flight_id}' is invalid.",
            details={"flight_id": flight_id},
        )

    # Rule 2: Passenger name present
    if not passenger_name or len(passenger_name) < 2:
        return ConstraintResult(
            allowed=False,
            code="MISSING_PASSENGER_NAME",
            message="Passenger name is required and must be at least 2 characters long.",
            details={"passenger_name": passenger_name},
        )

    # Rule 3: Passenger email format
    if not EMAIL_REGEX.match(passenger_email):
        return ConstraintResult(
            allowed=False,
            code="INVALID_PASSENGER_EMAIL",
            message=f"Passenger email '{passenger_email}' is not a valid email address.",
            details={"passenger_email": passenger_email},
        )

    # Rule 4: Flight existence & available seats
    flight = get_flight_by_id(flight_id)
    if not flight:
        return ConstraintResult(
            allowed=False,
            code="FLIGHT_NOT_FOUND",
            message=f"Flight '{flight_id}' does not exist in the flight database.",
            details={"flight_id": flight_id},
        )

    if flight["available_seats"] <= 0:
        return ConstraintResult(
            allowed=False,
            code="NO_AVAILABLE_SEATS",
            message=f"Flight '{flight_id}' has 0 available seats remaining.",
            details={"flight_id": flight_id, "available_seats": 0},
        )

    # Rule 5: Price ceiling
    if flight.get("price", 0.0) > DEFAULT_CONFIG.booking_price_limit:
        return ConstraintResult(
            allowed=False,
            code="PRICE_EXCEEDS_LIMIT",
            message=f"Flight price ${flight['price']} exceeds maximum allowed booking limit of ${DEFAULT_CONFIG.booking_price_limit}.",
            details={"price": flight.get("price"), "limit": DEFAULT_CONFIG.booking_price_limit},
        )

    return ConstraintResult(allowed=True, code="ALLOWED", message="Booking constraints satisfied.")


def validate_booking_status_constraints(arguments: Dict[str, Any]) -> ConstraintResult:
    """Validate constraints for get_booking_status."""
    booking_id = arguments.get("booking_id", "").strip().upper()
    if not booking_id:
        return ConstraintResult(
            allowed=False,
            code="MISSING_BOOKING_ID",
            message="booking_id cannot be empty.",
            details={"booking_id": booking_id},
        )
    return ConstraintResult(allowed=True, code="ALLOWED", message="Status query constraints satisfied.")


def validate_cancel_constraints(arguments: Dict[str, Any]) -> ConstraintResult:
    """Validate constraints for cancel_booking."""
    booking_id = arguments.get("booking_id", "").strip().upper()
    if not booking_id:
        return ConstraintResult(
            allowed=False,
            code="MISSING_BOOKING_ID",
            message="booking_id cannot be empty.",
            details={"booking_id": booking_id},
        )
    return ConstraintResult(allowed=True, code="ALLOWED", message="Cancel constraints satisfied.")


# Constraint rule mapping by tool name
CONSTRAINT_VALIDATORS = {
    "search_flights": validate_search_constraints,
    "get_flight_details": validate_flight_details_constraints,
    "book_flight": validate_book_constraints,
    "get_booking_status": validate_booking_status_constraints,
    "cancel_booking": validate_cancel_constraints,
}


def check_constraints(tool_name: str, arguments: Dict[str, Any]) -> ConstraintResult:
    """Dispatch constraint validation for a given tool and arguments."""
    validator = CONSTRAINT_VALIDATORS.get(tool_name)
    if validator:
        return validator(arguments)
    # Default allow if no specific constraints registered
    return ConstraintResult(allowed=True, code="ALLOWED", message=f"No constraints registered for tool '{tool_name}'.")
