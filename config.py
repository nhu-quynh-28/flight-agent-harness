"""Centralized configuration module for Flight Booking Agent and Harness.

Loads environment variables and sets system-wide defaults for model parameters,
execution budgets, loop detection thresholds, and safety limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load variables from .env file if available
load_dotenv()


@dataclass
class Config:
    """Central configuration class."""

    # Model settings
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    openai_base_url: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", None) or None
    )

    # Protection mechanism 1: Framework hard limit / recursion limit
    framework_max_steps: int = 15

    # Protection mechanism 2: Model and tool call budgets
    max_model_calls: int = 8
    max_tool_calls: int = 20

    # Protection mechanism 3: Semantic loop detection thresholds
    loop_window: int = 6
    loop_repeat_k: int = 2

    # Business constraints
    booking_price_limit: float = 1000.0
    allowed_airports: tuple[str, ...] = ("SGN", "HAN", "DAD", "PQC", "CXR")

    # Safe actions that are allowed to be repeated without triggering loop stop
    # e.g., polling/checking booking status during async updates
    safe_repeat_actions: set[str] = field(
        default_factory=lambda: {"get_booking_status"}
    )


# Global default configuration instance
DEFAULT_CONFIG = Config()
