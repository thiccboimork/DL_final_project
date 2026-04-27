"""
guardrails.py
-------------
Explicit guardrail definitions for the Interview ChatBot system.
The Verifier/Critic agent calls `scan_report_for_violations` before
revealing any PDF path or report content to the user.

Three layers of protection:
  1. PII detection  — phones, emails, addresses must never appear in output
  2. Topic scope    — off-topic content is flagged and suppressed
  3. Personal critique — feedback on protected attributes is blocked
"""

import re
from typing import Optional

from observability import DEFAULT_GUARDRAIL_CONFIG

# ---------------------------------------------------------------------------
# 1. Topic scope — agents must stay within professional interview/resume topics
# ---------------------------------------------------------------------------
DISALLOWED_TOPICS = [
    "politics", "religion", "romantic relationships", "medical diagnosis",
    "financial investment advice", "legal advice",
]

# ---------------------------------------------------------------------------
# 2. PII patterns — must be fully absent from any user-facing output
# ---------------------------------------------------------------------------
PII_PATTERNS = {
    "phone":   re.compile(r"\b(\+?1[-.\ s]?)?\(?\d{3}\)?[-.\ s]?\d{3}[-.\ s]?\d{4}\b"),
    "email":   re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "address": re.compile(
        r"\d{1,5}\s[\w\s]{1,50}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|"
        r"Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl)\b", re.IGNORECASE
    ),
    # Candidate full names often leaked via "Dear John Smith" / "Hi Alex Lee"
    "salutation_name": re.compile(
        r"\b(?:Dear|Hi|Hello|Hey)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
    ),
}

# ---------------------------------------------------------------------------
# 3. Personal-characteristic critique — never comment on these
# ---------------------------------------------------------------------------
PROHIBITED_CRITIQUE_ATTRIBUTES = [
    "age", "gender", "race", "ethnicity", "nationality", "religion",
    "disability", "appearance", "accent", "marital status",
]

# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------
SEVERITY_BLOCK  = "BLOCK"   # Stop output entirely, do not show PDF path
SEVERITY_REDACT = "REDACT"  # Strip the offending text, show cleaned output
SEVERITY_WARN   = "WARN"    # Log the flag but allow output (soft guardrail)

VIOLATION_SEVERITY = {
    "pii":              SEVERITY_BLOCK,
    "off_topic":        SEVERITY_WARN,
    "personal_critique":SEVERITY_BLOCK,
    "off_role":         SEVERITY_WARN,
}


# ---------------------------------------------------------------------------
# Helper: strip_pii
# ---------------------------------------------------------------------------
def strip_pii(text: str) -> str:
    """Replace PII matches with redacted placeholders."""
    for label, pattern in PII_PATTERNS.items():
        tag = "PII" if label == "salutation_name" else label.upper()
        text = pattern.sub(f"[{tag} REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Helper: check_topic_scope
# ---------------------------------------------------------------------------
def check_topic_scope(text: str) -> Optional[str]:
    """
    Returns a flag message if the text drifts off-topic, else None.
    """
    text_lower = text.lower()
    for topic in DISALLOWED_TOPICS:
        if topic in text_lower:
            return f"GUARDRAIL[off_topic]: Off-topic content detected — '{topic}'"
    return None


# ---------------------------------------------------------------------------
# Helper: check_personal_critique
# ---------------------------------------------------------------------------
def check_personal_critique(text: str) -> Optional[str]:
    """
    Returns a flag message if text critiques a prohibited personal attribute.
    """
    text_lower = text.lower()
    for attr in PROHIBITED_CRITIQUE_ATTRIBUTES:
        if attr in text_lower:
            return (
                f"GUARDRAIL[personal_critique]: Prohibited attribute mentioned — '{attr}'. "
                "Feedback must relate to job-relevant skills only."
            )
    return None


# ---------------------------------------------------------------------------
# Helper: check_pii_leak
# ---------------------------------------------------------------------------
def check_pii_leak(text: str) -> Optional[str]:
    """
    Returns a flag message if any PII pattern is found in text, else None.
    """
    for label, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            return f"GUARDRAIL[pii]: PII detected — '{label}' pattern found in output"
    return None


# ---------------------------------------------------------------------------
# Helper: validate_output  (used by agents during the interview itself)
# ---------------------------------------------------------------------------
def validate_output(text: str, job_role: str) -> list[str]:
    """
    Run all guardrail checks on an agent output.
    Returns a list of flag messages (empty list = clean).
    Used during the interview turn-by-turn.
    """
    flags = []

    pii_flag = check_pii_leak(text)
    if pii_flag:
        flags.append(pii_flag)

    topic_flag = check_topic_scope(text)
    if topic_flag:
        flags.append(topic_flag)

    critique_flag = check_personal_critique(text)
    if critique_flag:
        flags.append(critique_flag)

    if job_role and job_role.lower() not in text.lower():
        flags.append(
            f"GUARDRAIL[off_role]: Output may be off-role — no mention of '{job_role}'."
        )

    return flags


def get_guardrail_capabilities() -> dict:
    """
    Structured description of the project's explicit guardrail surface.
    """
    return {
        "configurable_policies": DEFAULT_GUARDRAIL_CONFIG.copy(),
        "blocked_topics": DISALLOWED_TOPICS,
        "pii_checks": list(PII_PATTERNS.keys()),
        "prohibited_critique_attributes": PROHIBITED_CRITIQUE_ATTRIBUTES,
        "severity_map": VIOLATION_SEVERITY.copy(),
        "evaluation_ready": True,
        "transparency_features": [
            "session-level tool logging",
            "session-level guardrail flag logging",
            "inspectable severity mapping",
            "open evaluation script",
        ],
    }


# ---------------------------------------------------------------------------
# PRIMARY ENTRY POINT: scan_report_for_violations
# ---------------------------------------------------------------------------
def scan_report_for_violations(
    report_text: str,
    job_role: str = "",
) -> dict:
    """
    Full guardrail scan of the final report text before it is shown to the user.

    This is the function the Verifier/Critic calls via `run_guardrail_scan`
    (the ADK tool wrapper defined in verifier_critic.py).

    Args:
        report_text: The full text content of the generated report.
        job_role:    The target job role (used for off-role check).

    Returns:
        {
            "verdict":        "PASS" | "BLOCK" | "WARN",
            "flags":          list[str],   # all violations found
            "blocking_flags": list[str],   # only BLOCK-severity violations
            "cleaned_text":   str,         # report text with PII stripped
            "show_pdf":       bool,        # True only if no BLOCK violations
        }
    """
    flags = []

    # --- PII scan (BLOCK severity) ---
    pii_flag = check_pii_leak(report_text)
    if pii_flag:
        flags.append(pii_flag)

    # --- Off-topic scan (WARN severity) ---
    topic_flag = check_topic_scope(report_text)
    if topic_flag:
        flags.append(topic_flag)

    # --- Personal critique scan (BLOCK severity) ---
    critique_flag = check_personal_critique(report_text)
    if critique_flag:
        flags.append(critique_flag)

    # --- Off-role soft check (WARN severity) ---
    if job_role and job_role.lower() not in report_text.lower():
        flags.append(
            f"GUARDRAIL[off_role]: Report does not mention '{job_role}' — "
            "may be off-role."
        )

    # Determine which flags are blocking
    blocking_flags = [
        f for f in flags
        if any(
            f"GUARDRAIL[{sev_key}]" in f
            for sev_key, sev in VIOLATION_SEVERITY.items()
            if sev == SEVERITY_BLOCK
        )
    ]

    # Strip PII from the text regardless (defence in depth)
    cleaned_text = strip_pii(report_text)

    # Determine verdict
    if blocking_flags:
        verdict = "BLOCK"
    elif flags:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "verdict":        verdict,
        "flags":          flags,
        "blocking_flags": blocking_flags,
        "cleaned_text":   cleaned_text,
        "show_pdf":       verdict != "BLOCK",
    }
