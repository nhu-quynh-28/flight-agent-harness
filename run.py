#!/usr/bin/env python3
"""Main Command-Line Interface (CLI) for Flight Booking Agent and Harness.

Supports running individual agents, testing specific scenarios, executing interactive
queries, and running the full benchmark evaluation suite comparing all three agent designs.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional
from tabulate import tabulate

from agents.hybrid_agent import HybridAgent
from agents.plan_execute_agent import PlanExecuteAgent
from agents.react_agent import ReactAgent
from config import DEFAULT_CONFIG
from data.flights import get_all_flights, reset_mock_data
from evaluation.evaluator import Evaluator
from evaluation.test_cases import get_evaluation_test_cases
from harness.harness import Harness
from harness.permissions import UserSession


def print_banner() -> None:
    """Print project header banner."""
    print("=" * 70)
    print("  FLIGHT BOOKING AGENT - BTVN#3 (Agentic AI Coursework)")
    print("  Architectures: ReAct | Plan-then-Execute | Hybrid ('Lai')")
    print("  Harness: Constraints | Permissions | Loops | Budget | Handoff")
    print("=" * 70)


def run_interactive(agent_type: str = "react", config: Optional[Config] = None, verbose: bool = False) -> None:
    """Run interactive terminal session with the chosen agent."""
    active_cfg = config or DEFAULT_CONFIG
    print_banner()
    mode_str = "OFFLINE MOCK LLM" if not active_cfg.openai_api_key else f"REAL LLM ({active_cfg.openai_model})"
    print(f"Starting interactive session with agent: [{agent_type.upper()}] | Mode: [{mode_str}]")
    print("Type 'exit' or 'quit' to end session. Type 'flights' to list available mock flights.\n")

    agent_map = {
        "react": ReactAgent(active_cfg),
        "plan": PlanExecuteAgent(active_cfg),
        "hybrid": HybridAgent(active_cfg),
    }
    agent = agent_map.get(agent_type.lower(), agent_map["react"])
    harness = Harness(config=active_cfg, session=UserSession(user_id="user_alice", authenticated=True, confirmed_booking=True))

    while True:
        try:
            query = input("\nUser > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting session. Goodbye!")
                break
            if query.lower() == "flights":
                flights = get_all_flights()
                headers = ["Flight ID", "Airline", "From", "To", "Date", "Departure", "Price", "Seats"]
                rows = [[f["flight_id"], f["airline"], f["origin"], f["destination"], f["date"], f["departure"], f"${f['price']}", f["available_seats"]] for f in flights]
                print("\nCurrent Mock Flight Catalog:")
                print(tabulate(rows, headers=headers, tablefmt="github"))
                continue

            # Run agent under harness
            harness.reset()
            result = agent.run(query=query, harness=harness)

            print(f"\n[{result.agent_name} Response]:")
            print(f"{result.final_answer}")

            if result.handoff:
                print("\n[HANDOFF PACKAGE GENERATED]:")
                print(json.dumps(result.handoff.to_dict(), indent=2))

            if verbose:
                print("\n--- Detailed Execution Trace ---")
                print(f"Success: {result.success}")
                print(f"Model calls: {result.model_calls}, Tool calls: {result.tool_calls}")
                print(f"Harness stopped: {result.stop_reason}")
                print(f"Completion check: {result.completion_result.to_dict() if result.completion_result else 'None'}")
                print(f"Duration: {result.duration_seconds:.4f}s")

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break


def run_single_case(agent_type: str, case_id: str, config: Optional[Config] = None, verbose: bool = False) -> None:
    """Run a specific test case with an agent."""
    active_cfg = config or DEFAULT_CONFIG
    evaluator = Evaluator(active_cfg)
    cases = get_evaluation_test_cases()
    matched = [c for c in cases if case_id.lower() in c.id.lower() or case_id.lower() in c.name.lower()]

    if not matched:
        print(f"Error: No test case matching '{case_id}' found.")
        print("Available test cases:")
        for c in cases:
            print(f"  - {c.id}: {c.name}")
        return

    case = matched[0]
    agent_name_map = {
        "react": "ReAct",
        "plan": "Plan-Execute",
        "hybrid": "Hybrid",
    }
    canonical_agent_name = agent_name_map.get(agent_type.lower(), "ReAct")
    agent_obj = evaluator.agents[canonical_agent_name]

    print_banner()
    mode_str = "OFFLINE MOCK LLM" if not active_cfg.openai_api_key else f"REAL LLM ({active_cfg.openai_model})"
    print(f"Running Scenario: {case.id} - {case.name} | Mode: [{mode_str}]")
    print(f"Description: {case.description}")
    print(f"Query: {case.query}")
    print(f"Agent Pattern: {canonical_agent_name}\n")

    record = evaluator.run_case_with_agent(canonical_agent_name, agent_obj, case)

    print("Result Summary:")
    print(f"  - Success: {record.success}")
    print(f"  - Code Completion Verified: {record.completion_verified}")
    print(f"  - Model Calls: {record.model_calls}")
    print(f"  - Tool Calls: {record.tool_calls}")
    print(f"  - Permission Blocks: {record.permission_blocks}")
    print(f"  - Loop Detected: {record.loop_detected}")
    print(f"  - Stop Reason: {record.stop_reason or 'None'}")
    print(f"  - Duration: {record.duration_seconds:.4f}s")
    print(f"  - Final Answer: {record.final_answer}")

    if record.handoff:
        print("\n[Handoff Package]:")
        print(json.dumps(record.handoff, indent=2))


def run_full_evaluation(config: Optional[Config] = None, verbose: bool = False) -> None:
    """Execute evaluation suite across all 3 agent architectures and print results."""
    active_cfg = config or DEFAULT_CONFIG
    print_banner()
    mode_str = "OFFLINE MOCK LLM" if not active_cfg.openai_api_key else f"REAL LLM ({active_cfg.openai_model})"
    print(f"Executing comprehensive evaluation across 12 test cases for all 3 agent designs | Mode: [{mode_str}]")
    print("Ensuring identical mock flights, harness safety checks, and completion verifications.\n")

    evaluator = Evaluator(active_cfg)
    res = evaluator.evaluate_all()

    # Print summary table
    print("=" * 80)
    print("  COMPARATIVE EVALUATION SUMMARY")
    print("=" * 80)
    print(evaluator.print_comparison_table(res))
    print("=" * 80)

    # Print per-test-case matrix
    cases = get_evaluation_test_cases()
    records_by_agent = res["records"]

    matrix_rows = []
    for c in cases:
        row = [c.name[:24]]
        for agent_name in ["ReAct", "Plan-Execute", "Hybrid"]:
            recs = [r for r in records_by_agent[agent_name] if r.case_id == c.id]
            if recs:
                r = recs[0]
                status = "PASS" if r.success else "FAIL"
                row.append(f"{status} (t:{r.tool_calls}, m:{r.model_calls})")
            else:
                row.append("N/A")
        matrix_rows.append(row)

    print("\nPer-Scenario Detailed Results Matrix:")
    print(tabulate(matrix_rows, headers=["Scenario", "ReAct", "Plan-Execute", "Hybrid"], tablefmt="github"))
    print("\nNote: 't' = tool calls, 'm' = model calls.")


def main() -> None:
    """CLI Entrypoint parsing arguments."""
    import copy

    parser = argparse.ArgumentParser(
        description="Flight Booking Agent with Harness Safety Layers (ReAct, Plan-then-Execute, Hybrid)."
    )
    parser.add_argument(
        "--agent",
        choices=["react", "plan", "hybrid"],
        default="react",
        help="Agent design pattern to run (default: react)",
    )
    parser.add_argument(
        "--case",
        type=str,
        help="Specific test case ID or substring (e.g. TC01, TC05, successful_booking)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a direct natural language query with the chosen agent",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run full 12-scenario benchmark evaluation comparing all 3 agent designs",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force deterministic offline mock LLM mode (ignores .env API key)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive CLI dialogue mode",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed traces and step outputs",
    )

    args = parser.parse_args()

    # Determine active configuration
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if args.mock:
        cfg.openai_api_key = ""

    if args.evaluate:
        run_full_evaluation(config=cfg, verbose=args.verbose)
    elif args.case:
        run_single_case(args.agent, args.case, config=cfg, verbose=args.verbose)
    elif args.query:
        print_banner()
        evaluator = Evaluator(cfg)
        agent_name_map = {"react": "ReAct", "plan": "Plan-Execute", "hybrid": "Hybrid"}
        name = agent_name_map[args.agent]
        agent = evaluator.agents[name]
        harness = Harness(cfg, session=UserSession(user_id="user_alice", authenticated=True, confirmed_booking=True))
        res = agent.run(args.query, harness)
        print(f"[{name} Response]:\n{res.final_answer}")
        if res.handoff:
            print(f"\n[Handoff]:\n{json.dumps(res.handoff.to_dict(), indent=2)}")
    elif args.interactive or len(sys.argv) == 1:
        run_interactive(agent_type=args.agent, config=cfg, verbose=args.verbose)
    else:
        # Default run single query with chosen agent
        run_interactive(agent_type=args.agent, config=cfg, verbose=args.verbose)


if __name__ == "__main__":
    main()
