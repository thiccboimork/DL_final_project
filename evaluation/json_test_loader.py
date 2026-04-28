"""
evaluation/json_test_loader.py
--------------------------------
Parses structured JSON test cases into the SessionState format used by
the Interview ChatBot pipeline.

Supports loading from:
  - A dict (e.g. pasted inline or returned from an API)
  - A .json file containing a single object or a list of objects
  - A raw JSON string

Usage:
    from evaluation.json_test_loader import load_test_case, load_test_cases_from_file

    # Single dict
    session_state, benchmarks = load_test_case(my_dict)

    # From file (single object or list)
    cases = load_test_cases_from_file("evaluation/test_cases.json")
    for session_state, benchmarks in cases:
        ...

    # From a raw JSON string
    cases = load_test_cases_from_string(json_str)
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import project's own shared state schema
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_state import (
    SessionState,
    ResumeData,
    JobContext,
    InterviewTranscript,
    InterviewPhase,
)


# ---------------------------------------------------------------------------
# Benchmarks container (not in shared_state — evaluation-only)
# ---------------------------------------------------------------------------

@dataclass
class TestBenchmarks:
    """
    Evaluation-time expectations extracted from verifier_benchmarks in the JSON.
    Not forwarded to agents; used only by scoring.py to assess agent output.
    """
    test_case_id: str = ""
    target_difficulty: str = ""
    industry: str = ""
    expected_focus_areas: list[str] = field(default_factory=list)
    must_flag: list[str] = field(default_factory=list)
    masking_required: list[str] = field(default_factory=list)
    topic_lock: str = ""
    # Ground-truth interview data for quality checks
    suggested_questions: list[str] = field(default_factory=list)
    sample_responses: list[dict] = field(default_factory=list)
    # PII values kept here so scoring.py can verify masking occurred
    pii_values: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_cite(s: str) -> str:
    """Remove citation markers like ' [cite: 74]' from benchmark strings."""
    return re.sub(r"\s*\[cite:\s*\d+\]", "", s).strip()


def _parse_resume(raw: dict) -> tuple[ResumeData, dict[str, str]]:
    """
    Returns (ResumeData, pii_dict).
    ResumeData has PII stripped (matching the real parse_resume tool behaviour).
    pii_dict holds the raw PII values for masking verification in scoring.py.
    """
    pii = raw.get("pii_fields", {})
    pii_dict = {
        "full_name": pii.get("full_name", ""),
        "phone": pii.get("phone", ""),
        "email": pii.get("email", ""),
        "address": pii.get("address", ""),
    }

    # Build a raw_text representation from the resume fields (no PII)
    experience = raw.get("experience", [])
    exp_text = "\n".join(
        f"{e.get('role','')} at {e.get('company','')} ({e.get('duration','')}): "
        f"{e.get('responsibilities','')}"
        for e in experience
    )
    summary = raw.get("summary", "")
    skills = raw.get("skills", [])

    resume = ResumeData(
        raw_text=f"{summary}\n{exp_text}".strip(),
        skills=skills,
        experience_years=_infer_years(experience),
        education=raw.get("education", []),
    )
    return resume, pii_dict


def _infer_years(experience: list[dict]) -> int:
    """
    Rough heuristic: count distinct years mentioned in duration strings.
    Falls back to len(experience) * 2 if parsing fails.
    """
    years = set()
    for e in experience:
        dur = e.get("duration", "")
        found = re.findall(r"\b(20\d{2}|19\d{2})\b", dur)
        years.update(int(y) for y in found)
    if len(years) >= 2:
        return max(years) - min(years)
    return len(experience) * 2


def _parse_job(raw: dict) -> JobContext:
    return JobContext(
        job_title=raw.get("job_title", ""),
        company_name=raw.get("company_name", ""),
        required_skills=raw.get("required_skills", []),
        company_values=[raw.get("company_values", "")],
        focus_areas=[],   # filled in by Context Optimizer at runtime; pre-populated below
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_test_case(data: dict) -> tuple[SessionState, TestBenchmarks]:
    """
    Parse a single test-case dict.

    Returns:
        (SessionState, TestBenchmarks)

        SessionState  — ready to inject as ADK initial session state.
                        Contains resume + job context with PII stripped.
                        focus_areas pre-populated from expected_focus_areas
                        so tests can skip the live Context Optimizer step.

        TestBenchmarks — evaluation-only metadata; pass to scoring.evaluate_session().
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    meta = data.get("metadata", {})
    sim  = data.get("interview_simulation_data", {})
    bench_raw = data.get("verifier_benchmarks", {})
    guardrails = bench_raw.get("guardrail_checks", {})

    # --- Resume ---
    resume, pii_dict = _parse_resume(data.get("candidate_resume", {}))

    # --- Job ---
    job = _parse_job(data.get("job_listing", {}))
    job.focus_areas = meta.get("expected_focus_areas", [])

    # --- Session state ---
    state = SessionState(
        phase=InterviewPhase.CONTEXT_LOADING,
        resume=resume,
        job_context=job,
        transcript=InterviewTranscript(),
    )

    # --- Benchmarks ---
    benchmarks = TestBenchmarks(
        test_case_id=data.get("test_case_id", ""),
        target_difficulty=meta.get("target_difficulty", ""),
        industry=meta.get("industry", ""),
        expected_focus_areas=meta.get("expected_focus_areas", []),
        must_flag=[_strip_cite(f) for f in bench_raw.get("must_flag", [])],
        masking_required=guardrails.get("masking_required", []),
        topic_lock=_strip_cite(guardrails.get("topic_lock", "")),
        suggested_questions=sim.get("suggested_interviewer_questions", []),
        sample_responses=sim.get("sample_candidate_responses", []),
        pii_values=pii_dict,
    )

    return state, benchmarks


def load_test_cases_from_file(path: str | Path) -> list[tuple[SessionState, TestBenchmarks]]:
    """
    Load one or more test cases from a JSON file.

    The file may contain:
      - A single test-case object  → returns a 1-element list
      - A list of test-case objects → returns a list with one entry per object
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test case file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return [load_test_case(tc) for tc in payload]
    elif isinstance(payload, dict):
        return [load_test_case(payload)]
    else:
        raise ValueError(f"Unexpected JSON root type: {type(payload).__name__}")


def load_test_cases_from_string(json_str: str) -> list[tuple[SessionState, TestBenchmarks]]:
    """
    Convenience wrapper — parse one or more test cases from a raw JSON string.
    Useful for pasting directly in tests or the Streamlit UI.
    """
    payload = json.loads(json_str)
    if isinstance(payload, list):
        return [load_test_case(tc) for tc in payload]
    return [load_test_case(payload)]


def session_state_to_agent_context(state: SessionState, benchmarks: TestBenchmarks) -> str:
    """
    Serialise the loaded session into a plain-text context string that can be
    injected as the opening system message / user turn in the ADK runner.

    This bypasses the live Context Optimizer agent so tests focus on
    Simulation Specialist and Verifier behaviour.
    """
    skills_str = ", ".join(state.resume.skills) if state.resume.skills else "N/A"
    req_str    = ", ".join(state.job_context.required_skills) if state.job_context.required_skills else "N/A"
    focus_str  = "\n".join(f"  - {fa}" for fa in state.job_context.focus_areas) or "  - None specified"
    values_str = " ".join(state.job_context.company_values)

    return f"""[TEST CASE: {benchmarks.test_case_id}]
Difficulty: {benchmarks.target_difficulty} | Industry: {benchmarks.industry}

CANDIDATE SUMMARY:
{state.resume.raw_text}

SKILLS: {skills_str}

JOB: {state.job_context.job_title} @ {state.job_context.company_name}
COMPANY VALUES: {values_str}
REQUIRED SKILLS: {req_str}

FOCUS AREAS TO PROBE:
{focus_str}

Context loading complete. Handing off to Simulation Specialist."""


# ---------------------------------------------------------------------------
# CLI: python -m evaluation.json_test_loader <file.json>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pprint

    if len(sys.argv) < 2:
        print("Usage: python json_test_loader.py <test_case.json>")
        sys.exit(1)

    cases = load_test_cases_from_file(sys.argv[1])
    for state, bench in cases:
        print(f"\n{'='*60}")
        print(f"Test Case : {bench.test_case_id}")
        print(f"Difficulty: {bench.target_difficulty}  |  Industry: {bench.industry}")
        print(f"Job       : {state.job_context.job_title} @ {state.job_context.company_name}")
        print(f"Skills    : {', '.join(state.resume.skills)}")
        print(f"Focus areas:")
        for fa in bench.expected_focus_areas:
            print(f"  - {fa}")
        print(f"Must-flag checks ({len(bench.must_flag)}):")
        for mf in bench.must_flag:
            print(f"  • {mf}")
        print(f"PII fields to mask: {bench.masking_required}")
        print(f"\nAgent context string preview:")
        print(session_state_to_agent_context(state, bench)[:400] + "...")