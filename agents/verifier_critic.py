"""
agents/verifier_critic.py
--------------------------
Agent 3: Verifier / Critic

Changes from previous version:
  - generate_report wrapper fills in safe defaults so the LLM can never
    crash the call by omitting guardrail_flags or any optional section.
  - Instruction rewritten with explicit, unambiguous tool call templates
    so the model knows exactly what arguments to supply.
  - Stores report_path in session state so the UI can show a download button.
  - Adds strengths / work_on / expand_on / next_steps to the report call.
"""

from google.adk.agents import Agent

from shared_state import InterviewPhase
from tools.report_generator import generate_report as _generate_report_impl
from tools.vector_memory import store_user_profile
from guardrails import scan_report_for_violations
from observability import log_guardrail_event, log_tool_call


def _get(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _set(state, key, value):
    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)


# ---------------------------------------------------------------------------
# Tool: generate_report  (safe wrapper around the implementation)
# ---------------------------------------------------------------------------

def generate_report(
    tool_context,
    candidate_name: str = "Candidate",
    job_role: str = "",
    company: str = "",
    evaluated_skills: dict = None,
    transcript_summary: str = "",
    guardrail_flags: list = None,  # Can be passed by LLM or None
    strengths: list = None,
    work_on: list = None,
    expand_on: list = None,
    next_steps: list = None,
) -> dict:
    """
    Corrected wrapper to prevent UnboundLocalError and handle missing flags.
    """
    state = tool_context.state
    
    # FIX: Ensure guardrail_flags always has a value before being used
    if guardrail_flags is None:
        # Pull from session state if not explicitly passed by the agent
        guardrail_flags = _get(state, "guardrail_flags", []) or []

    result = _generate_report_impl(
        candidate_name=candidate_name or "Candidate",
        job_role=job_role or "",
        company=company or "",
        evaluated_skills=evaluated_skills or {},
        transcript_summary=transcript_summary or "",
        guardrail_flags=guardrail_flags, # Now guaranteed to be defined
        strengths=strengths,
        work_on=work_on,
        expand_on=expand_on,
        next_steps=next_steps,
        tool_context=tool_context,
    )

    if result.get("status") == "success":
        state = tool_context.state
        _set(state, "report_path", result["report_path"])
    
        from shared_state import InterviewPhase
        _set(state, "phase", InterviewPhase.REPORT_READY.value)
        
    return result


# ---------------------------------------------------------------------------
# Tool: run_guardrail_scan
# ---------------------------------------------------------------------------

def run_guardrail_scan(tool_context, report_text: str, job_role: str = "") -> dict:
    """
    Scan the report text for PII leaks, off-topic content, and prohibited
    personal critiques BEFORE revealing the PDF path to the user.

    Args:
        report_text: The text you drafted for the report (summary + bullet sections).
        job_role:    Target job role for the off-role grounding check.

    Returns:
        {verdict, flags, blocking_flags, cleaned_text, show_pdf}
        verdict   = "PASS" | "WARN" | "BLOCK"
        show_pdf  = True unless verdict is "BLOCK"
    """
    result = scan_report_for_violations(report_text, job_role)
    state  = tool_context.state

    if result["flags"]:
        existing = _get(state, "guardrail_flags", []) or []
        existing.extend(result["flags"])
        _set(state, "guardrail_flags", existing)

    log_guardrail_event(
        state, stage="final_report_scan",
        verdict=result["verdict"], flags=result["flags"],
        metadata={"job_role": job_role},
    )
    log_tool_call(
        state, "verifier_critic", "run_guardrail_scan",
        {"job_role": job_role},
        {"verdict": result["verdict"], "flag_count": len(result["flags"])},
    )
    return result


# ---------------------------------------------------------------------------
# Agent instruction
# ---------------------------------------------------------------------------

VERIFIER_CRITIC_INSTRUCTION = """
You are the Verifier/Critic — the grading and quality-assurance agent at the
end of the interview pipeline. Your job is to grade the candidate, write a
structured report, and deliver clear, actionable feedback.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — READ THE SESSION DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have access to:
  • transcript_turns  — the full Q&A from the interview (or pre-loaded synthetic data)
  • job_context       — job_title, company_name, required_skills, focus_areas
  • resume            — candidate's skills and experience

Read all of this before grading anything.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — GRADE AND WRITE YOUR FOUR SECTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Produce these six things in your head before calling any tool:

A) evaluated_skills (dict)
   Grade EACH skill in job_context.required_skills AND each focus_area:
     "strong"            → answered with depth, specifics, concrete results
     "adequate"          → correct but shallow or missing evidence
     "needs_improvement" → evasive, wrong, or not demonstrated

B) transcript_summary (2–3 sentences)
   • Sentence 1: overall impression.
   • Sentence 2: the strongest moment, citing the actual topic.
   • Sentence 3: the most important gap, citing the actual skill.

C) strengths (list of 3–5 strings)
   Each string: "What they did well" + "why it matters for this specific role".
   Be specific — reference the actual question topic or a phrase they used.

D) work_on (list of 2–4 strings)
   Each string: "The gap" + "why it matters" + "one concrete action to fix it".
   Example: "Budget management at $1M+ scale is untested. The candidate
   cited a $250k ceiling with no plan to scale. Action: shadow a senior PM
   managing a large multi-department budget before applying."

E) expand_on (list of 2–3 strings)
   For "adequate" skills only: explain the NEXT level and how to reach it.
   Example: "Conflict resolution is present but surface-level. Deepen by
   practicing the full STAR format with executive-stakeholder examples."

F) next_steps (list of exactly 3 strings)
   Concrete, actionable items — name a course, book, exercise, or behaviour.
   Each must directly address a gap from work_on or expand_on.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — CALL generate_report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Call generate_report with ALL of these arguments:

  generate_report(
    candidate_name    = "Candidate",
    job_role          = <job_context.job_title>,
    company           = <job_context.company_name>,
    evaluated_skills  = <your dict from Step 2A>,
    transcript_summary= <your 2–3 sentences from Step 2B>,
    strengths         = <your list from Step 2C>,
    work_on           = <your list from Step 2D>,
    expand_on         = <your list from Step 2E>,
    next_steps        = <your list from Step 2F>,
  )

NOTE: Do NOT pass guardrail_flags — it is read automatically from session state.
NOTE: Do NOT skip any argument. Every argument listed above is required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — GUARDRAIL SCAN (required before showing PDF path)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Immediately after generate_report returns "status": "success", call:

  run_guardrail_scan(
    report_text = <transcript_summary + " " + " ".join(strengths + work_on + expand_on)>,
    job_role    = <job_context.job_title>,
  )

Result handling:
  • verdict "PASS" or "WARN" → show the PDF path in your response.
  • verdict "BLOCK"          → DO NOT show the PDF path. Deliver feedback
                               as plain text using cleaned_text only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — DELIVER THE RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Call store_user_profile to persist this session.

Format your response using EXACTLY these headers:

**📄 Your Report**
Path: [report_path from generate_report result]
Grade: [overall_grade] ([overall_score]/100) — [verdict one-liner]

**✅ Strengths**
[Your bullet list from Step 2C]

**🎯 Work On**
[Your bullet list from Step 2D]

**💡 Expand On**
[Your bullet list from Step 2E]

**🚀 Recommended Next Steps**
1. [step 1]
2. [step 2]
3. [step 3]

━━━━━━━━━━━━━━━━━━
HARD RULES
━━━━━━━━━━━━━━━━━━
• NEVER critique personal characteristics (age, gender, race, accent, etc.).
• NEVER fabricate skill ratings — every rating must come from the transcript.
• NEVER show the PDF path if guardrail verdict is BLOCK.
• NEVER omit arguments from the generate_report call.
• If generate_report returns an error, deliver all feedback in plain text.
• Every bullet in work_on must include one concrete action.
"""

verifier_critic_agent = Agent(
    name="verifier_critic",
    model="gemini-2.5-flash-lite",
    description=(
        "Grades the interview transcript, writes Strengths / Work On / Expand On "
        "sections, generates a PDF report with overall score and letter grade, "
        "runs a guardrail scan, and delivers structured feedback."
    ),
    instruction=VERIFIER_CRITIC_INSTRUCTION,
    tools=[generate_report, run_guardrail_scan, store_user_profile],
)