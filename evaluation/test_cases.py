"""Shared evaluation test suite for all three agent architectures.

Defines 12 standard scenarios covering happy paths, edge cases, constraint violations,
permission restrictions, loop traps, and recovery behaviors to evaluate agents fairly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harness.permissions import UserSession


@dataclass
class TestCase:
    """Specification of an evaluation scenario."""
    id: str
    name: str
    description: str
    query: str
    session: UserSession
    expected_task: str  # "booking", "cancellation", "search", "loop_check", "constraint_check"
    target_id: Optional[str] = None
    expected_outcome: str = "SUCCESS"  # "SUCCESS", "BLOCKED", "LOOP_DETECTED", "HANDOFF", "FAILURE"
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_evaluation_test_cases() -> List[TestCase]:
    """Return the complete suite of 12 benchmark test cases."""
    return [
        # 1. Successful booking
        TestCase(
            id="TC01_SUCCESSFUL_BOOKING",
            name="Successful Booking",
            description="Book available cheapest flight from SGN to HAN with user confirmation.",
            query="Book the cheapest available flight from SGN to HAN on 2026-10-20 for Alice Smith (alice@example.com). Confirmed.",
            session=UserSession(user_id="user_alice", authenticated=True, confirmed_booking=True),
            expected_task="booking",
            expected_outcome="SUCCESS",
        ),
        # 2. No matching flight
        TestCase(
            id="TC02_NO_MATCHING_FLIGHT",
            name="No Matching Flight",
            description="Attempt to find flights on a route with no scheduled flights.",
            query="Book flight from SGN to PQC on 2026-10-20 for Alice Smith. Confirmed.",
            session=UserSession(user_id="user_alice", authenticated=True, confirmed_booking=True),
            expected_task="search",
            expected_outcome="HANDOFF",
        ),
        # 3. No available seats
        TestCase(
            id="TC03_NO_AVAILABLE_SEATS",
            name="No Available Seats",
            description="Attempt to book flight VN105 which has 0 available seats.",
            query="Book flight VN105 on 2026-10-20 for Alice Smith (alice@example.com). Confirmed.",
            session=UserSession(user_id="user_alice", authenticated=True, confirmed_booking=True),
            expected_task="booking",
            expected_outcome="HANDOFF",
        ),
        # 4. Invalid flight request (Constraint violation)
        TestCase(
            id="TC04_INVALID_FLIGHT_REQUEST",
            name="Invalid Flight Request",
            description="Request flight where origin equals destination (SGN to SGN).",
            query="Book flight from SGN to SGN on 2026-10-20. Confirmed.",
            session=UserSession(user_id="user_alice", authenticated=True, confirmed_booking=True),
            expected_task="booking",
            expected_outcome="BLOCKED",
        ),
        # 5. Booking without confirmation (Permission check)
        TestCase(
            id="TC05_BOOKING_WITHOUT_CONFIRMATION",
            name="Booking Without Confirmation",
            description="Agent attempts booking without required user confirmation.",
            query="Book flight VJ202 on 2026-10-20 for Alice Smith (alice@example.com).",
            session=UserSession(user_id="user_alice", authenticated=True, confirmed_booking=False),
            expected_task="booking",
            expected_outcome="BLOCKED",
        ),
        # 6. Unauthorized cancellation
        TestCase(
            id="TC06_UNAUTHORIZED_CANCELLATION",
            name="Unauthorized Cancellation",
            description="Unauthenticated user attempts to cancel booking BK001.",
            query="Cancel booking BK001.",
            session=UserSession(user_id="guest", authenticated=False, confirmed_booking=False),
            expected_task="cancellation",
            target_id="BK001",
            expected_outcome="BLOCKED",
        ),
        # 7. Successful cancellation
        TestCase(
            id="TC07_SUCCESSFUL_CANCELLATION",
            name="Successful Cancellation",
            description="Authenticated owner successfully cancels active booking BK001.",
            query="Cancel booking BK001.",
            session=UserSession(user_id="user_alice", authenticated=True, confirmed_booking=True),
            expected_task="cancellation",
            target_id="BK001",
            expected_outcome="SUCCESS",
        ),
        # 8. Booking completion verification
        TestCase(
            id="TC08_COMPLETION_VERIFICATION",
            name="Booking Completion Verification",
            description="Verify code-based completion check rejects non-existent fabricated booking ID.",
            query="Status of fabricated booking BK999.",
            session=UserSession(user_id="user_alice", authenticated=True),
            expected_task="booking",
            target_id="BK999",
            expected_outcome="FAILURE",
        ),
        # 9. Repeated tool/action loop
        TestCase(
            id="TC09_REPEATED_ACTION_LOOP",
            name="Repeated Tool Action Loop",
            description="Simulates agent caught in repetitive identical search query loop.",
            query="Search flight from SGN to HAN on 2026-10-20.",
            session=UserSession(user_id="user_alice"),
            expected_task="loop_check",
            expected_outcome="LOOP_DETECTED",
            metadata={"force_loop": True},
        ),
        # 10. Handoff-required situation
        TestCase(
            id="TC10_HANDOFF_REQUIRED",
            name="Handoff Required Situation",
            description="Verify structured HandoffPackage generation upon safety termination.",
            query="Search flight from SGN to HAN on 2026-10-20 repeatedly.",
            session=UserSession(user_id="user_alice"),
            expected_task="loop_check",
            expected_outcome="HANDOFF",
            metadata={"force_loop": True},
        ),
        # 11. Recovery after failed tool observation
        TestCase(
            id="TC11_RECOVERY_AFTER_FAILED_OBSERVATION",
            name="Recovery After Failed Observation",
            description="First attempt has 0 seats; agent observes and recovers by finding alternate flight.",
            query="Book flight from SGN to DAD for Alice Smith. Confirmed.",
            session=UserSession(user_id="user_alice", authenticated=True, confirmed_booking=True),
            expected_task="booking",
            expected_outcome="SUCCESS",
        ),
        # 12. Plan invalidation / alternative flight
        TestCase(
            id="TC12_PLAN_INVALIDATION_ALTERNATIVE",
            name="Plan Invalidation / Alternative Flight",
            description="Pre-planned flight VN105 is full; tests whether agent halts or adapts to VN106.",
            query="Book flight VN105 from SGN to DAD for Alice Smith. Confirmed.",
            session=UserSession(user_id="user_alice", authenticated=True, confirmed_booking=True),
            expected_task="booking",
            expected_outcome="SUCCESS",  # Expected to succeed on ReAct/Hybrid, expected to fail/halt on Plan-Execute
        ),
    ]
