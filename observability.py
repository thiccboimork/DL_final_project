"""
observability.py
----------------
Shared helpers for tool logging and guardrail event tracking.
"""

from __future__ import annotations

import datetime
from typing import Any


DEFAULT_GUARDRAIL_CONFIG = {
    "pii_filter": True,
    "topic_scope": True,
    "personal_critique_block": True,
    "off_role_warning": True,
    "log_tool_calls": True,
    "log_guardrail_events": True,
}


def get_state_value(state, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def set_state_value(state, key: str, value) -> None:
    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)


def ensure_state_defaults(state) -> None:
    defaults = {
        "tool_call_log": [],
        "guardrail_flags": [],
        "guardrail_config": DEFAULT_GUARDRAIL_CONFIG.copy(),
    }
    for key, value in defaults.items():
        if get_state_value(state, key, None) is None:
            set_state_value(state, key, value.copy() if isinstance(value, dict) else list(value) if isinstance(value, list) else value)


def log_tool_call(
    state,
    agent: str,
    tool: str,
    args: dict[str, Any],
    result: dict[str, Any] | str | None,
) -> None:
    ensure_state_defaults(state)
    logs = get_state_value(state, "tool_call_log", [])
    if logs is None:
        logs = []

    result_summary = result if isinstance(result, str) else str(result)
    logs.append({
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent": agent,
        "tool": tool,
        "args": args,
        "result_summary": result_summary[:300],
    })
    set_state_value(state, "tool_call_log", logs[-50:])


def log_guardrail_event(
    state,
    stage: str,
    verdict: str,
    flags: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    ensure_state_defaults(state)
    logs = get_state_value(state, "tool_call_log", [])
    if logs is None:
        logs = []

    logs.append({
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent": "guardrail_system",
        "tool": "guardrail_scan",
        "args": {
            "stage": stage,
            "metadata": metadata or {},
        },
        "result_summary": str({
            "verdict": verdict,
            "flags": flags,
        })[:300],
    })
    set_state_value(state, "tool_call_log", logs[-50:])

