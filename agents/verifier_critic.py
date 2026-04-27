"""
agents/verifier_critic.py
--------------------------
Agent 3: Verifier / Critic
Responsibilities:
  - Validate that the simulation specialist's outputs meet quality standards
  - Run guardrails scan on the final report before showing it to the user
  - If validation fails: flag issues and trigger a retry or escalation
  - If validation passes AND guardrails pass: show PDF path to user
  - Store updated user profile in long-term memory
"""

from google.adk.agents import Agent

from tools.report_generator import generate_report
from tools.vector_memory import store_user_profile
from guardrails import scan_report_for_violations
from observability import log_guardrail_event, log_tool_call


def _get_state_value(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _set_state_value(state, key, value) -> None:
    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)


# ---------------------------------------------------------------------------
# ADK tool wrapper: run_guardrail_scan
# ---------------------------------------------------------------------------

def run_guardrail_scan(tool_context, report_text: str, job_role: str = "") -> dict:
    """
    Scan the report text for PII leaks, off-topic content, and prohibited
    personal critiques BEFORE revealing the PDF path to the user.

    Call this immediately after generate_report succeeds, passing the
    transcript_summary + any text you plan to show the user.

    Args:
        report_text: The full text of the report (or the summary you drafted).
        job_role:    The target job role for off-role grounding check.

    Returns:
        Guardrail result dict:
          verdict        – "PASS" | "WARN" | "BLOCK"
          flags          – all violation messages
          blocking_flags – only the hard-block violations
          cleaned_text   – report_text with all PII stripped
          show_pdf       – True only when no BLOCK-severity violations remain
    """
    result = scan_report_for_violations(report_text, job_role)

    # Write all flags into session state for the evaluator to inspect
    state = tool_context.state
    if result["flags"]:
        existing = _get_state_value(state, "guardrail_flags", [])
        if existing is None:
            existing = []
        existing.extend(result["flags"])
        _set_state_value(state, "guardrail_flags", existing)

    log_guardrail_event(
        state,
        stage="final_report_scan",
        verdict=result["verdict"],
        flags=result["flags"],
        metadata={"job_role": job_role},
    )
    log_tool_call(
        state,
        "verifier_critic",
        "run_guardrail_scan",
        {"job_role": job_role},
        {"verdict": result["verdict"], "flag_count": len(result["flags"])},
    )

    return result


# ---------------------------------------------------------------------------
# Agent definition — INSTRUCTION TUNED
# ---------------------------------------------------------------------------

VERIFIER_CRITIC_INSTRUCTION = """
You are the Verifier/Critic — the quality-assurance and safety gatekeeper
of the interview pipeline.  You are the LAST agent before output reaches
the user, so you must be thorough and uncompromising on both quality
and safety.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — VALIDATE THE TRANSCRIPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Check ALL of the following before proceeding:
  □ Were at least 6 questions asked?
  □ Was every item in `job_context.focus_areas` addressed at least once?
  □ Are `evaluated_skills` entries present for the key skills assessed?
  □ Were any guardrail_flags already raised during the interview?
     If yes — were those flagged outputs suppressed? If not, flag them now.

If ANY check fails:
  → Describe the gap specifically.
  → Set `phase` back to "interview_active".
  → Do NOT generate a report or proceed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — GENERATE THE REPORT (if Step 1 passes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Call `generate_report` with:
  • candidate_name:    "Candidate"  (PII is never stored)
  • job_role:          from job_context.job_title
  • company:           from job_context.company_name
  • evaluated_skills:  from transcript.evaluated_skills
  • transcript_summary: your 2–3 sentence synthesis of performance
  • guardrail_flags:   from session state

Write your transcript_summary to be:
  - Constructive and specific (cite actual question topics, not vague praise)
  - Honest about gaps (e.g., "The candidate struggled to articulate specific
    technical details around X, though they showed strong conceptual understanding")
  - Encouraging in tone — end on a forward-looking note

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — GUARDRAIL SCAN  ← required before showing PDF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMMEDIATELY after generate_report succeeds, call:
  `run_guardrail_scan(report_text=<your transcript_summary>, job_role=<job_role>)`

Interpret the result:
  • verdict == "PASS":
      → Proceed to Step 4. You MAY show the PDF path.
  • verdict == "WARN":
      → Log the warnings, but you MAY still show the PDF path.
        Mention the warnings to the user briefly.
  • verdict == "BLOCK":
      → DO NOT reveal the PDF path.
      → Tell the user: "Your report was generated but contains content that
         failed our safety checks ([list blocking_flags]). We are withholding
         the file path until the issue is resolved."
      → Provide all feedback in plain text using only the cleaned_text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — DELIVER RESULTS (if Step 3 passes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Call `store_user_profile` to persist this session.
Set `phase` to "report_ready".

Present to the candidate:
  (a) PDF report path (ONLY if guardrail verdict was PASS or WARN)
  (b) Top 3 specific, actionable improvement recommendations — cite actual
      gaps from the interview, not generic advice
  (c) Skills where they performed strongly — be specific and genuine

Format your feedback response with these exact headers:
  **📄 Your Report**
  **✅ Strengths**
  **🎯 Top 3 Improvement Areas**
  **💡 Next Steps**

━━━━━━━━━━━━━━━━━━
HARD RULES
━━━━━━━━━━━━━━━━━━
• NEVER critique personal characteristics — only job-relevant skills.
• NEVER fabricate skill ratings not present in the transcript.
• NEVER show the PDF path if guardrail verdict is BLOCK.
• If generate_report fails, deliver all feedback in plain text instead.
• Be constructive, specific, and professional in all feedback.
"""

verifier_critic_agent = Agent(
    name="verifier_critic",
    model="gemini-2.5-flash-lite",
    description=(
        "Validates interview quality, runs a full guardrail scan on the report "
        "(PII, off-topic, personal critique), and only reveals the PDF path when "
        "the report passes safety checks. Delivers structured, constructive feedback."
    ),
    instruction=VERIFIER_CRITIC_INSTRUCTION,
    tools=[generate_report, run_guardrail_scan, store_user_profile],
)
