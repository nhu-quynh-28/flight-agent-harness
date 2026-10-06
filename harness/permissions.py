"""Permission control layer for the flight booking harness.

Enforces role and session-based access control policies prior to tool execution,
blocking unauthorized or unconfirmed sensitive actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from data.flights import get_booking


@dataclass
class UserSession:
    """Represents the caller's session context."""
    user_id: str = "guest"
    authenticated: bool = False
    confirmed_booking: bool = False

    def authenticate(self, user_id: str) -> None:
        self.user_id = user_id
        self.authenticated = True

    def set_booking_confirmed(self, confirmed: bool = True) -> None:
        self.confirmed_booking = confirmed


@dataclass
class PermissionResult:
    """Structured report of permission authorization."""
    allowed: bool
    code: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "message": self.message,
        }


def check_permission(
    tool_name: str,
    arguments: Dict[str, Any],
    session: Optional[UserSession] = None,
) -> PermissionResult:
    """Evaluate permission rules for a proposed tool execution.

    Policies:
    1. search_flights: Public (unauthenticated allowed).
    2. get_flight_details: Public (unauthenticated allowed).
    3. get_booking_status: Public (unauthenticated allowed).
    4. book_flight: Requires explicit user confirmation either in arguments or session.
    5. cancel_booking: Requires authentication AND ownership of the target booking.
    """
    user_session = session or UserSession()

    # Rule 1: Public queries
    if tool_name in ("search_flights", "get_flight_details", "get_booking_status"):
        return PermissionResult(allowed=True, code="ALLOWED", message="Action is publicly permitted.")

    # Rule 2: Booking confirmation requirement
    if tool_name == "book_flight":
        confirmed_in_args = bool(arguments.get("confirmed_by_user", False))
        confirmed_in_session = user_session.confirmed_booking

        if not (confirmed_in_args or confirmed_in_session):
            return PermissionResult(
                allowed=False,
                code="CONFIRMATION_REQUIRED",
                message="User confirmation is required before booking a flight.",
            )
        return PermissionResult(allowed=True, code="ALLOWED", message="Booking confirmation verified.")

    # Rule 3: Cancellation authentication & ownership requirement
    if tool_name == "cancel_booking":
        if not user_session.authenticated:
            return PermissionResult(
                allowed=False,
                code="AUTHENTICATION_REQUIRED",
                message="Authentication is required to cancel a booking.",
            )

        booking_id = arguments.get("booking_id", "")
        if booking_id:
            booking = get_booking(booking_id)
            if booking:
                owner = booking.get("user_id")
                # Disallow if booking belongs to a different user
                if owner and owner != user_session.user_id:
                    return PermissionResult(
                        allowed=False,
                        code="UNAUTHORIZED_ACCESS",
                        message=f"User '{user_session.user_id}' is not authorized to cancel booking belonging to '{owner}'.",
                    )

        return PermissionResult(allowed=True, code="ALLOWED", message="Cancellation authorized.")

    # Default fallback
    return PermissionResult(allowed=True, code="ALLOWED", message="Default permission granted.")
