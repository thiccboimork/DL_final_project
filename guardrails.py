"""
guardrails.py
-------------
Explicit guardrail definitions for the Interview ChatBot system.
The Verifier/Critic agent enforces these before any output is finalized.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Topic scope — agents must stay within professional interview/resume topics
# ---------------------------------------------------------------------------
DISALLOWED_TOPICS = [
    "politics", "religion", "romantic relationships", "medical diagnosis",
    "financial investment advice", "legal advice",
]

# ---------------------------------------------------------------------------
# 2. PII patterns — stripped before any inter-agent handoff
# ---------------------------------------------------------------------------
PII_PATTERNS = {
    "phone":   re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email":   re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "address": re.compile(
        r"\d{1,5}\s[\w\s]{1,50}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|"
        r"Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl)\b", re.IGNORECASE
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
# Helper functions
# ---------------------------------------------------------------------------

def strip_pii(text: str) -> str:
    """Replace PII matches with redacted placeholders."""
    for label, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[{label.upper()} REDACTED]", text)
    return text


def check_topic_scope(text: str) -> Optional[str]:
    """
    Returns a flag message if the text appears to drift off-topic,
    otherwise returns None.
    """
    text_lower = text.lower()
    for topic in DISALLOWED_TOPICS:
        if topic in text_lower:
            return f"GUARDRAIL: Off-topic content detected — '{topic}'"
    return None


def check_personal_critique(text: str) -> Optional[str]:
    """
    Returns a flag message if the text critiques a prohibited personal attribute.
    """
    text_lower = text.lower()
    for attr in PROHIBITED_CRITIQUE_ATTRIBUTES:
        if attr in text_lower:
            return (
                f"GUARDRAIL: Prohibited personal critique detected — '{attr}'. "
                "Feedback must relate to job-relevant skills only."
            )
    return None


def validate_output(text: str, job_role: str) -> list[str]:
    """
    Run all guardrail checks on an agent output.
    Returns a list of flag messages (empty list = clean).
    """
    flags = []

    topic_flag = check_topic_scope(text)
    if topic_flag:
        flags.append(topic_flag)

    critique_flag = check_personal_critique(text)
    if critique_flag:
        flags.append(critique_flag)

    # Minimal grounding check: if the job role is provided, the output
    # should at least reference it or a closely related term.
    if job_role and job_role.lower() not in text.lower():
        flags.append(
            f"GUARDRAIL: Output may be off-role — no mention of '{job_role}'."
        )

    return flags