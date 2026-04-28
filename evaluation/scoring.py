"""
evaluation/scoring.py
----------------------
Evaluation script for the Interview ChatBot system.

Supports two input modes:
  1. Legacy mode  — test_cases.json with the original flat schema
                    (backwards-compatible; no changes needed)
  2. JSON test-case mode — structured dicts loaded by json_test_loader.py
                    (use run_evaluation_from_json() or pass json_path= to
                     run_evaluation())

Metrics reported:
  - success_rate: % of sessions with clean agent handoffs and complete skill coverage
  - avg_latency_s: average end-to-end session time (seconds)
  - avg_tokens: average total tokens consumed per session
  - error_categories: counts of failure types
  - pii_masking_pass_rate: % of cases where all required PII was masked  (JSON mode only)
  - must_flag_coverage: % of must-flag checks that were caught             (JSON mode only)
"""

import json
import time
import datetime
from pathlib import Path
from typing import Any

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"
RESULTS_PATH    = Path(__file__).parent / "results.json"


# ---------------------------------------------------------------------------
# Error category definitions
# ---------------------------------------------------------------------------
class ErrorCategory:
    TOOL_FAILURE        = "tool_failure"
    HALLUCINATION       = "hallucination"
    PLANNING_ERROR      = "planning_error"
    GUARDRAIL_VIOLATION = "guardrail_violation"
    CONTEXT_DRIFT       = "context_drift"
    HANDOFF_FAILURE     = "handoff_failure"
    PII_LEAK            = "pii_leak"          # new: PII appeared in agent output
    MUST_FLAG_MISSED    = "must_flag_missed"  # new: verifier missed a required flag


# ---------------------------------------------------------------------------
# Core scorer — works with both legacy dicts and TestBenchmarks objects
# ---------------------------------------------------------------------------

def evaluate_session(test_case, session_result: dict) -> dict[str, Any]:
    """
    Score a single session result against a test case.

    Args:
        test_case:      Either a legacy flat dict (from test_cases.json)
                        OR a TestBenchmarks instance (from json_test_loader).
        session_result: Output dict from running the agent pipeline.

    Returns:
        Score dict with pass/fail flags and error categories.
    """
    from evaluation.json_test_loader import TestBenchmarks

    errors = []
    checks = {}
    using_benchmarks = isinstance(test_case, TestBenchmarks)

    # ── Normalise test case fields ──────────────────────────────────────────
    if using_benchmarks:
        tc_id          = test_case.test_case_id
        focus_areas    = test_case.expected_focus_areas
        exp_handoffs   = 2
        exp_q_min      = 5
        must_flag      = test_case.must_flag
        masking_req    = test_case.masking_required
        pii_values     = test_case.pii_values
    else:
        tc_id          = test_case.get("id", "unknown")
        focus_areas    = test_case.get("focus_areas_expected", [])
        exp_handoffs   = test_case.get("expected_handoffs", 2)
        exp_q_min      = test_case.get("expected_questions_min", 5)
        must_flag      = []
        masking_req    = []
        pii_values     = {}

    # ── Check 1: Correct number of agent handoffs ───────────────────────────
    actual_handoffs = session_result.get("handoff_count", 0)
    checks["handoffs_correct"] = actual_handoffs >= exp_handoffs
    if not checks["handoffs_correct"]:
        errors.append(ErrorCategory.HANDOFF_FAILURE)

    # ── Check 2: Minimum question count ────────────────────────────────────
    actual_questions = session_result.get("question_count", 0)
    checks["question_count_met"] = actual_questions >= exp_q_min
    if not checks["question_count_met"]:
        errors.append(ErrorCategory.PLANNING_ERROR)

    # ── Check 3: Focus areas were covered ──────────────────────────────────
    covered = session_result.get("evaluated_skills", {})
    uncovered = [
        fa for fa in focus_areas
        if not any(fa.lower() in skill.lower() for skill in covered.keys())
    ]
    checks["focus_areas_covered"] = len(uncovered) == 0
    if not checks["focus_areas_covered"]:
        errors.append(ErrorCategory.CONTEXT_DRIFT)

    # ── Check 4: No guardrail violations ───────────────────────────────────
    flags = session_result.get("guardrail_flags", [])
    checks["no_guardrail_violations"] = len(flags) == 0
    if not checks["no_guardrail_violations"]:
        errors.append(ErrorCategory.GUARDRAIL_VIOLATION)

    # ── Check 5: Report was generated ──────────────────────────────────────
    checks["report_generated"] = session_result.get("report_path") is not None
    if not checks["report_generated"]:
        errors.append(ErrorCategory.TOOL_FAILURE)

    # ── Check 6 (JSON mode): PII masking ───────────────────────────────────
    if using_benchmarks and masking_req:
        transcript_text = _extract_transcript_text(session_result)
        leaked = [
            field for field in masking_req
            if pii_values.get(field) and pii_values[field] in transcript_text
        ]
        checks["pii_masked"] = len(leaked) == 0
        if not checks["pii_masked"]:
            errors.append(ErrorCategory.PII_LEAK)
    else:
        checks["pii_masked"] = True  # not checked in legacy mode

    # ── Check 7 (JSON mode): Must-flag coverage ─────────────────────────────
    if using_benchmarks and must_flag:
        verifier_flags = session_result.get("verifier_flags", [])
        missed = [
            mf for mf in must_flag
            if not any(mf.lower() in vf.lower() for vf in verifier_flags)
        ]
        checks["must_flags_caught"] = len(missed) == 0
        if not checks["must_flags_caught"]:
            errors.append(ErrorCategory.MUST_FLAG_MISSED)
    else:
        checks["must_flags_caught"] = True  # not checked in legacy mode

    passed = all(checks.values())

    return {
        "test_id":      tc_id,
        "passed":       passed,
        "checks":       checks,
        "errors":       errors,
        "latency_s":    session_result.get("latency_s", 0),
        "total_tokens": session_result.get("total_tokens", 0),
    }


def _extract_transcript_text(session_result: dict) -> str:
    """Flatten all agent output text from a session result for PII scanning."""
    parts = []
    transcript = session_result.get("transcript", {})
    if isinstance(transcript, dict):
        for turn in transcript.get("turns", []):
            parts.append(str(turn.get("text", "")))
    elif isinstance(transcript, list):
        for turn in transcript:
            parts.append(str(turn.get("text", "")))
    parts.append(str(session_result.get("report_text", "")))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Aggregator — shared by both run modes
# ---------------------------------------------------------------------------

def _aggregate(results: list[dict], cases_meta: list[dict] | None = None) -> dict[str, Any]:
    total      = len(results)
    passed     = sum(1 for r in results if r["passed"])
    s_rate     = round(passed / total * 100, 1) if total else 0
    avg_lat    = round(sum(r["latency_s"] for r in results) / total, 2) if total else 0
    avg_tok    = round(sum(r["total_tokens"] for r in results) / total) if total else 0

    error_counts: dict[str, int] = {}
    for r in results:
        for err in r["errors"]:
            error_counts[err] = error_counts.get(err, 0) + 1

    # PII masking pass-rate (only meaningful when checks["pii_masked"] was run)
    pii_checked = [r for r in results if "pii_masked" in r.get("checks", {})]
    pii_pass = round(
        sum(1 for r in pii_checked if r["checks"].get("pii_masked")) / len(pii_checked) * 100, 1
    ) if pii_checked else None

    # Must-flag coverage
    mf_checked = [r for r in results if "must_flags_caught" in r.get("checks", {})]
    mf_pass = round(
        sum(1 for r in mf_checked if r["checks"].get("must_flags_caught")) / len(mf_checked) * 100, 1
    ) if mf_checked else None

    summary = {
        "run_timestamp":        datetime.datetime.utcnow().isoformat(),
        "total_cases":          total,
        "passed":               passed,
        "failed":               total - passed,
        "success_rate_pct":     s_rate,
        "avg_latency_s":        avg_lat,
        "avg_tokens":           avg_tok,
        "pii_masking_pass_rate":   pii_pass,
        "must_flag_coverage_pct":  mf_pass,
        "error_categories":     error_counts,
        "per_case_results":     results,
    }
    return summary


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 55)
    print("  INTERVIEW CHATBOT — EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Total cases:      {summary['total_cases']}")
    print(f"  Passed:           {summary['passed']}")
    print(f"  Failed:           {summary['failed']}")
    print(f"  Success rate:     {summary['success_rate_pct']}%")
    print(f"  Avg latency:      {summary['avg_latency_s']}s")
    print(f"  Avg tokens:       {summary['avg_tokens']}")
    if summary["pii_masking_pass_rate"] is not None:
        print(f"  PII masking:      {summary['pii_masking_pass_rate']}%")
    if summary["must_flag_coverage_pct"] is not None:
        print(f"  Must-flag hits:   {summary['must_flag_coverage_pct']}%")
    ec = summary["error_categories"]
    if any(ec.values()):
        print("\n  Error breakdown:")
        for cat, count in ec.items():
            if count > 0:
                print(f"    {cat}: {count}")
    print("=" * 55)
    print(f"  Results → {RESULTS_PATH}")
    print()


# ---------------------------------------------------------------------------
# Legacy run_evaluation (unchanged interface — backwards compatible)
# ---------------------------------------------------------------------------

def run_evaluation(session_runner=None, json_path: str | Path | None = None) -> dict[str, Any]:
    """
    Run the full evaluation suite.

    Args:
        session_runner: Callable(test_case) → session_result dict, or None for stub mode.
        json_path:      Optional path to a JSON file of structured test cases
                        (loaded via json_test_loader). If omitted, uses test_cases.json.

    Returns:
        Aggregate metrics dict (also written to results.json).
    """
    if json_path:
        return run_evaluation_from_json(json_path=json_path, session_runner=session_runner)

    with open(TEST_CASES_PATH) as f:
        test_cases = json.load(f)

    results = []
    for tc in test_cases:
        if session_runner:
            start = time.time()
            session_result = session_runner(tc)
            session_result["latency_s"] = round(time.time() - start, 2)
        else:
            session_result = {
                "handoff_count":   2,
                "question_count":  6,
                "evaluated_skills": {fa: "adequate" for fa in tc.get("focus_areas_expected", [])},
                "guardrail_flags": [],
                "report_path":     "/tmp/stub_report.pdf",
                "latency_s":       0,
                "total_tokens":    0,
            }
        results.append(evaluate_session(tc, session_result))

    summary = _aggregate(results)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    _print_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# New: run from structured JSON test cases
# ---------------------------------------------------------------------------

def run_evaluation_from_json(
    json_path: str | Path | None = None,
    json_string: str | None = None,
    session_runner=None,
) -> dict[str, Any]:
    """
    Run evaluation using structured JSON test cases (the TC-xxx-xx format).

    Exactly one of json_path or json_string must be provided.

    Args:
        json_path:      Path to a .json file (single object or list).
        json_string:    Raw JSON string (single object or list).
        session_runner: Callable(session_state, benchmarks) → session_result dict.
                        If None, runs in stub mode (useful for schema validation).

    Returns:
        Aggregate metrics dict.

    Example session_runner signature::

        def my_runner(state: SessionState, bench: TestBenchmarks) -> dict:
            context_str = session_state_to_agent_context(state, bench)
            result = run_agent_pipeline(context_str)   # your ADK runner
            return {
                "handoff_count":   result.handoff_count,
                "question_count":  result.question_count,
                "evaluated_skills": result.evaluated_skills,
                "guardrail_flags": result.guardrail_flags,
                "verifier_flags":  result.verifier_flags,  # for must_flag checks
                "report_path":     result.report_path,
                "total_tokens":    result.total_tokens,
            }
    """
    from json_test_loader import (
        load_test_cases_from_file,
        load_test_cases_from_string,
        session_state_to_agent_context,
    )

    if json_path:
        cases = load_test_cases_from_file(json_path)
    elif json_string:
        cases = load_test_cases_from_string(json_string)
    else:
        raise ValueError("Provide either json_path or json_string.")

    results = []
    for state, bench in cases:
        if session_runner:
            start = time.time()
            session_result = session_runner(state, bench)
            session_result.setdefault("latency_s", round(time.time() - start, 2))
        else:
            # Stub: pre-fill all expected answers so the test "passes" schema checks
            session_result = {
                "handoff_count":    2,
                "question_count":   len(bench.suggested_questions) or 6,
                "evaluated_skills": {fa: "adequate" for fa in bench.expected_focus_areas},
                "guardrail_flags":  [],
                "verifier_flags":   list(bench.must_flag),  # stub catches everything
                "report_path":      "/tmp/stub_report.pdf",
                "latency_s":        0,
                "total_tokens":     0,
                "transcript":       {"turns": []},
            }

        results.append(evaluate_session(bench, session_result))

    summary = _aggregate(results)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    _print_summary(summary)
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Accept a JSON test case file as argument
        run_evaluation_from_json(json_path=sys.argv[1])
    else:
        run_evaluation()