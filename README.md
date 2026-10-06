# Flight Booking Agent with Multi-Layer Safety Harness

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2+-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An educational coursework project implementing a Flight Booking Agent using **LangChain** and **LangGraph**, governed by a multi-layer safety **Harness** and evaluated across three distinct agent design patterns: **ReAct**, **Plan-then-Execute**, and **Hybrid ("Lai")**.

---

## 📌 Key Architectural Highlights

1. **Shared Environment & Fair Evaluation:**  
   All three agent patterns share the **exact same** mock flight data, tools, constraints, completion verifications, permission policies, and test cases. Only the agent decision-making strategy differs.
2. **Four Core Harness Requirements:**
   - **Constraints as Data (`harness/constraints.py`):** Declarative validation rules (airports, dates, price limits, passenger details) returning structured results instead of raising raw exceptions.
   - **Code-Based Completion Verification (`harness/completion.py`):** Verifies task completion against actual in-memory database records, rejecting conversational LLM hallucinations.
   - **Permission Control (`harness/permissions.py`):** Blocks unconfirmed bookings and unauthorized cancellations before tool execution.
   - **Safe Human Handoff (`harness/handoff.py`):** Generates structured context packages (reason, state, attempted actions, clarifying questions) when stopping abnormally.
3. **Three Safe Stopping Protection Mechanisms:**
   - **Framework Hard Step Limit (`framework_max_steps = 15`):** Terminates infinite recursions.
   - **Model & Tool Call Budget (`max_model_calls = 8`, `max_tool_calls = 20`):** Prevents resource exhaustion.
   - **Semantic Loop Detector (`harness/loop_detector.py`):** Computes canonical action fingerprints `(tool_name, sorted(args))` over a sliding window (`LOOP_WINDOW = 6`, `LOOP_REPEAT_K = 2`), with semantic exemptions for legitimate polling actions (e.g., checking booking status).
4. **Three Agent Design Patterns:**
   - **ReAct (`agents/react_agent.py`):** Iterative Thought $\to$ Action $\to$ Observation loop with rapid recovery from failures.
   - **Plan-then-Execute (`agents/plan_execute_agent.py`):** Upfront structured planning followed by sequential execution (economical but rigid).
   - **Hybrid / Lai (`agents/hybrid_agent.py`):** High-level planning + reactive execution + dynamic re-planning when observations invalidate the active plan.

---

## 📂 Project Structure

```
flight-booking-agent/
│
├── README.md                  # Project documentation & execution guide
├── requirements.txt           # Python dependencies
├── .env.example               # Template for environment variables
├── .gitignore
│
├── config.py                  # Central configuration & safety thresholds
│
├── data/
│   ├── __init__.py
│   └── flights.py             # Deterministic mock flight DB & booking store
│
├── tools/
│   ├── __init__.py
│   └── flight_tools.py        # LangChain structured tools with rich observations
│
├── harness/
│   ├── __init__.py
│   ├── constraints.py         # Constraints as data validation
│   ├── completion.py          # Code-based completion criteria
│   ├── permissions.py         # Permission control & session management
│   ├── loop_detector.py       # Semantic action-loop detection
│   ├── handoff.py             # Safe handoff data structures
│   └── harness.py             # Central harness coordinator
│
├── agents/
│   ├── __init__.py
│   ├── base.py                # Standardized AgentRunResult
│   ├── llm_factory.py         # Real ChatOpenAI or Deterministic Mock LLM
│   ├── react_agent.py         # 1. ReAct architecture
│   ├── plan_execute_agent.py  # 2. Plan-then-Execute architecture
│   └── hybrid_agent.py        # 3. Hybrid ('Lai') architecture
│
├── evaluation/
│   ├── __init__.py
│   ├── test_cases.py          # 12 Shared benchmark test scenarios
│   ├── metrics.py             # Statistical metrics computation
│   └── evaluator.py           # Benchmark runner
│
├── run.py                     # Command-line interface (CLI)
│
└── report/
    └── report.md              # Complete technical coursework report
```

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.11+
- Virtual environment recommended

### 2. Installation
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration (Dual-Mode Execution)
The project supports two execution modes:

- **Mode A: Offline Deterministic Mock LLM (Default)**  
  Runs with 0 API cost and 100% deterministic reproducibility for automated grading and offline testing.
- **Mode B: Real LLM via OpenAI API (or OpenRouter, Ollama, Groq, DeepSeek)**  
  Copy `.env.example` to `.env` and fill in your credentials:
  ```bash
  cp .env.example .env
  ```
  Edit `.env`:
  ```env
  OPENAI_API_KEY=sk-proj-...
  OPENAI_MODEL=gpt-4o-mini
  # Optional: For OpenRouter, Ollama, or local proxies:
  # OPENAI_BASE_URL=https://openrouter.ai/api/v1
  ```

---

## 💻 CLI Usage

All capabilities can be run through `run.py`:

### 1. Run Full Benchmark Evaluation
Runs all 12 test scenarios across all 3 agent designs and outputs comparative tables:
```bash
# Evaluates with the active configuration (Real LLM if .env is set, or Mock if not)
python run.py --evaluate

# Force offline mock LLM mode (fast, 0 token cost, deterministic):
python run.py --evaluate --mock
```

### 2. Run an Agent with an Individual Test Case
Test scenarios include: `TC01` (Successful Booking), `TC04` (Invalid Flight), `TC05` (Permission Check), `TC09` (Loop Detection), `TC11` (Recovery), `TC12` (Plan Invalidation):
```bash
# Run ReAct on successful booking
python run.py --agent react --case TC01

# Run Plan-then-Execute on booking without confirmation (triggers permission block)
python run.py --agent plan --case TC05

# Run Hybrid on plan invalidation (adapts and books alternative flight)
python run.py --agent hybrid --case TC12

# Add --mock to force offline mock execution for any test case:
python run.py --agent hybrid --case TC12 --mock
```

### 3. Run Direct Natural Language Query
```bash
# Ask an agent to search and book
python run.py --agent react --query "I want to book the cheapest available flight from SGN to HAN on 2026-10-20 for Alice Smith (alice@example.com). Confirmed."

# Add -v or --verbose for detailed step-by-step traces
python run.py --agent hybrid --case TC01 -v
```

### 4. Interactive Terminal Mode
```bash
python run.py --interactive --agent hybrid
```
*(In interactive mode, type `flights` to view the mock catalog or `exit` to quit).*

---

## 📊 Experimental Results

Real results obtained from executing `python run.py --evaluate`:

| Agent Pattern | Task Success Rate | Code Completion | Avg Model Calls | Avg Tool Calls | Loop Rate | Perm Block Rate | Recovery Rate | Avg Duration |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ReAct** | **100.0%** | 33.3% | 2.67 | 2.25 | 16.7% | 16.7% | **100.0%** | 0.0007s |
| **Plan-then-Execute** | **83.3%** | 16.7% | **0.83** | **1.67** | 16.7% | 16.7% | **0.0%** | **0.0001s** |
| **Hybrid ("Lai")** | **100.0%** | 33.3% | 2.50 | 2.00 | 16.7% | 16.7% | **100.0%** | 0.0011s |

### Core Insights:
- **Plan-then-Execute** is the most cost-efficient in fixed scenarios (averaging 0.83 model calls) but fails completely (0.0% recovery) when an initial flight is full.
- **ReAct** recovers perfectly from dynamic obstacles but requires significantly more model reasoning calls.
- **Hybrid ("Lai")** strikes the best balance by following high-level plan milestones while dynamically re-planning upon encountering failed observations.
- All three patterns reliably respect the safety harness layers: blocking circular routes (TC04), blocking unconfirmed bookings (TC05), catching action loops (TC09), and generating handoffs (TC10).


