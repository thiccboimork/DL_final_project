"""
shared_state.py
---------------
Defines the shared session state schema passed between agents.
All agents read from and write to this structure via ADK's session state.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class InterviewPhase(str, Enum):
    CONTEXT_LOADING = "context_loading"
    INTERVIEW_ACTIVE = "interview_active"
    VERIFICATION = "verification"
    REPORT_READY = "report_ready"


@dataclass
class ResumeData:
    """Structured resume data extracted by the Context Optimizer."""
    raw_text: str = ""
    skills: list[str] = field(default_factory=list)
    experience_years: int = 0
    education: list[str] = field(default_factory=list)
    # PII is stripped before populating this object
    # name, phone, address are NOT stored here


@dataclass
class JobContext:
    """Job and company context scraped by the Context Optimizer."""
    job_title: str = ""
    company_name: str = ""
    required_skills: list[str] = field(default_factory=list)
    company_values: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)  # gaps to probe


@dataclass
class InterviewTranscript:
    """Running transcript of the mock interview."""
    turns: list[dict] = field(default_factory=list)
    # Each turn: {"role": "interviewer"|"candidate", "text": str, "timestamp": str}
    evaluated_skills: dict[str, str] = field(default_factory=dict)
    # skill_name → "strong" | "adequate" | "needs_improvement"


@dataclass
class SessionState:
    """
    Root shared state object. ADK stores this as JSON in the session.
    Agents access it via tool_context.state.
    """
    phase: InterviewPhase = InterviewPhase.CONTEXT_LOADING
    question_count: int = 0
    resume: ResumeData = field(default_factory=ResumeData)
    job_context: JobContext = field(default_factory=JobContext)
    transcript: InterviewTranscript = field(default_factory=InterviewTranscript)
    tool_call_log: list[dict] = field(default_factory=list)
    guardrail_flags: list[str] = field(default_factory=list)
    report_path: Optional[str] = None

    def log_tool_call(self, agent: str, tool: str, args: dict, result: dict):
        """Append a tool call record to the session log."""
        import datetime
        self.tool_call_log.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": agent,
            "tool": tool,
            "args": args,
            "result_summary": str(result)[:200],
        })