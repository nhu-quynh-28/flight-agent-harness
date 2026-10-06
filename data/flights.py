"""Deterministic mock flight database and in-memory booking store.

Provides predictable test data for all three agent architectures to ensure
apples-to-apples evaluation without reliance on external airline APIs.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

# Initial mock flight catalog
INITIAL_FLIGHTS: List[Dict[str, Any]] = [
    # Scenario 1 & 2: Multiple flights on same route & date (SGN -> HAN on 2026-10-20)
    {
        "flight_id": "VN101",
        "airline": "Vietnam Airlines",
        "origin": "SGN",
        "destination": "HAN",
        "date": "2026-10-20",
        "departure": "08:00",
        "arrival": "10:15",
        "price": 120.0,
        "available_seats": 5,
    },
    {
        "flight_id": "VJ202",
        "airline": "Vietjet Air",
        "origin": "SGN",
        "destination": "HAN",
        "date": "2026-10-20",
        "departure": "13:30",
        "arrival": "15:45",
        "price": 85.0,
        "available_seats": 3,
    },
    {
        "flight_id": "QH303",
        "airline": "Bamboo Airways",
        "origin": "SGN",
        "destination": "HAN",
        "date": "2026-10-20",
        "departure": "18:00",
        "arrival": "20:10",
        "price": 140.0,
        "available_seats": 10,
    },
    # Scenario 3: Flight with NO available seats (SGN -> DAD on 2026-10-20)
    {
        "flight_id": "VN105",
        "airline": "Vietnam Airlines",
        "origin": "SGN",
        "destination": "DAD",
        "date": "2026-10-20",
        "departure": "09:00",
        "arrival": "10:20",
        "price": 110.0,
        "available_seats": 0,  # Fully booked
    },
    # Scenario 5: Alternative flight next day (SGN -> DAD on 2026-10-21)
    {
        "flight_id": "VN106",
        "airline": "Vietnam Airlines",
        "origin": "SGN",
        "destination": "DAD",
        "date": "2026-10-21",
        "departure": "09:30",
        "arrival": "10:50",
        "price": 95.0,
        "available_seats": 4,
    },
    # Additional route for varied testing: HAN -> CXR
    {
        "flight_id": "VJ404",
        "airline": "Vietjet Air",
        "origin": "HAN",
        "destination": "CXR",
        "date": "2026-10-22",
        "departure": "07:15",
        "arrival": "09:05",
        "price": 75.0,
        "available_seats": 8,
    },
]

# Initial pre-existing mock bookings
INITIAL_BOOKINGS: Dict[str, Dict[str, Any]] = {
    "BK001": {
        "booking_id": "BK001",
        "flight_id": "VN101",
        "user_id": "user_alice",
        "passenger_name": "Alice Smith",
        "passenger_email": "alice@example.com",
        "price": 120.0,
        "status": "CONFIRMED",
        "created_at": "2026-10-01T10:00:00Z",
    },
    "BK002": {
        "booking_id": "BK002",
        "flight_id": "VJ202",
        "user_id": "user_bob",
        "passenger_name": "Bob Jones",
        "passenger_email": "bob@example.com",
        "price": 85.0,
        "status": "CANCELLED",
        "created_at": "2026-10-02T11:00:00Z",
    },
}

# In-memory working stores
_flights_store: List[Dict[str, Any]] = copy.deepcopy(INITIAL_FLIGHTS)
_bookings_store: Dict[str, Dict[str, Any]] = copy.deepcopy(INITIAL_BOOKINGS)
_booking_counter: int = 100


def reset_mock_data() -> None:
    """Reset the mock flight and booking databases to their initial clean state."""
    global _flights_store, _bookings_store, _booking_counter
    _flights_store = copy.deepcopy(INITIAL_FLIGHTS)
    _bookings_store = copy.deepcopy(INITIAL_BOOKINGS)
    _booking_counter = 100


def get_all_flights() -> List[Dict[str, Any]]:
    """Return all flights currently in the catalog."""
    return [copy.deepcopy(f) for f in _flights_store]


def get_flight_by_id(flight_id: str) -> Optional[Dict[str, Any]]:
    """Lookup a single flight by its unique identifier."""
    for flight in _flights_store:
        if flight["flight_id"].upper() == flight_id.strip().upper():
            return copy.deepcopy(flight)
    return None


def find_flights(origin: str, destination: str, date: str) -> List[Dict[str, Any]]:
    """Find flights matching origin, destination, and departure date."""
    orig = origin.strip().upper()
    dest = destination.strip().upper()
    dt = date.strip()

    matches = []
    for flight in _flights_store:
        if (
            flight["origin"] == orig
            and flight["destination"] == dest
            and flight["date"] == dt
        ):
            matches.append(copy.deepcopy(flight))
    return matches


def get_booking(booking_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve booking by ID."""
    b = _bookings_store.get(booking_id.strip().upper())
    return copy.deepcopy(b) if b else None


def create_booking(
    flight_id: str,
    passenger_name: str,
    passenger_email: str,
    user_id: str = "guest",
) -> Dict[str, Any]:
    """Create a new booking and decrement flight available seats."""
    global _booking_counter
    target = None
    for flight in _flights_store:
        if flight["flight_id"].upper() == flight_id.strip().upper():
            target = flight
            break

    if not target:
        raise ValueError(f"Flight ID '{flight_id}' does not exist.")

    if target["available_seats"] <= 0:
        raise ValueError(f"Flight '{flight_id}' has no available seats.")

    # Decrement seat
    target["available_seats"] -= 1
    _booking_counter += 1
    new_id = f"BK{_booking_counter}"

    record = {
        "booking_id": new_id,
        "flight_id": target["flight_id"],
        "user_id": user_id,
        "passenger_name": passenger_name.strip(),
        "passenger_email": passenger_email.strip(),
        "price": target["price"],
        "status": "CONFIRMED",
        "created_at": "2026-10-06T12:00:00Z",
    }
    _bookings_store[new_id] = record
    return copy.deepcopy(record)


def cancel_booking_in_store(booking_id: str, user_id: str) -> Dict[str, Any]:
    """Mark a booking as CANCELLED and increment flight seats back."""
    bid = booking_id.strip().upper()
    if bid not in _bookings_store:
        raise KeyError(f"Booking '{booking_id}' not found.")

    record = _bookings_store[bid]
    if record["status"] == "CANCELLED":
        return copy.deepcopy(record)

    record["status"] = "CANCELLED"

    # Restore seat in flight catalog
    for flight in _flights_store:
        if flight["flight_id"] == record["flight_id"]:
            flight["available_seats"] += 1
            break

    return copy.deepcopy(record)
