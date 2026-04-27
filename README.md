# Interview ChatBot — Multi-Agent System
> Deep Learning / LLM Course Project | Spring 2026

A multi-agent interview preparation and resume review system built with [Google ADK](https://google.github.io/adk-docs/).

---

## Architecture

```
User Input (resume + job description)
        │
        ▼
┌─────────────────────┐
│  Context Optimizer  │  ← Parses resume, scrapes job/company info via web search
│      Agent          │    Populates shared session state with focus areas
└────────┬────────────┘
         │ handoff (focus_areas → session state)
         ▼
┌─────────────────────┐
│  Simulation         │  ← Conducts real-time mock interview
│  Specialist Agent   │    Generates role-specific questions, tracks responses
└────────┬────────────┘
         │ handoff (transcript → session state)
         ▼
┌─────────────────────┐
│  Verifier / Critic  │  ← Validates outputs, checks guardrails
│      Agent          │    Generates final PDF performance report via file I/O
└─────────────────────┘
```

## Project Structure

```
interview_chatbot/
├── agents/
│   ├── __init__.py
│   ├── context_optimizer.py      # Agent 1: Resume parser + job scraper
│   ├── simulation_specialist.py  # Agent 2: Mock interview conductor
│   └── verifier_critic.py        # Agent 3: Validator + report generator
├── tools/
│   ├── __init__.py
│   ├── resume_parser.py          # Tool: Parse PDF resume → structured data
│   ├── web_search.py             # Tool: Scrape job/company info
│   ├── report_generator.py       # Tool: Generate PDF performance report
│   └── vector_memory.py          # Tool: Long-term user profile storage
├── evaluation/
│   ├── test_cases.json           # 30 synthetic candidate-job pairs
│   └── scoring.py                # Evaluation script (success rate, latency, cost)
├── tests/
│   └── test_agents.py            # Unit tests per agent
├── agent.py                      # Root agent + ADK runner entrypoint
├── shared_state.py               # Shared session state schema
├── guardrails.py                 # Explicit guardrail definitions
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_ORG/DL_final_project.git
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Fill in your GOOGLE_API_KEY from Google AI Studio
```

### 5. Run the ADK web UI locally
```bash
adk web
```

---

## Tools Used

| Tool | Purpose | Agent |
|---|---|---|
| `resume_parser` | Extract structured data from PDF resume | Context Optimizer |
| `google_search` (ADK built-in) | Scrape job posting & company values | Context Optimizer |
| `file_io / report_generator` | Write final performance report as PDF | Verifier/Critic |
| `vector_memory` | Store/retrieve long-term user profiles | All agents |

## Guardrails

See `guardrails.py` for full list. Summary:
- Agents may only discuss professional interview and resume topics
- PII fields (phone, address, name) are masked before inter-agent handoffs
- No critique of personal characteristics unrelated to job performance
- Verifier flags and suppresses factually ungrounded or off-role outputs

## Transparency And Differentiation

This project is intentionally designed to be more inspectable and customizable than
closed commercial interview-prep products and more operationally grounded than many
academic prototypes.

What distinguishes it:
- Explicit tool logging: session state records tool invocations, inputs, timestamps, and summarized outputs for auditability.
- Defined guardrail surface: policy checks for PII leakage, topic drift, personal-attribute critique, and off-role feedback are implemented in code rather than hidden behind vendor defaults.
- Configurable policy model: guardrail behavior is represented as inspectable policy settings in session state, which makes experimentation and ablation studies possible.
- Open evaluation path: `evaluation/scoring.py` provides a baseline framework for repeatable testing across handoffs, question coverage, guardrail violations, and report generation.
- In-app transparency: the Streamlit sidebar exposes recent tool activity and guardrail events so users and evaluators can inspect system behavior during a live run.

Compared with platforms such as LinkedIn Interview Prep or Big Interview, the emphasis
here is not only mock-interview usability but also observability, controllable safety
logic, and evaluation-readiness for research or coursework.

## Evaluation

Run the evaluation suite:
```bash
python evaluation/scoring.py
```

Reports: success rate, average latency, average token cost, and error categories.

---
