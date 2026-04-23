"""
agents/simulation_specialist.py
---------------------------------
Agent 2: Simulation Specialist
Responsibilities:
  - Conduct a realistic mock interview using focus_areas from session state
  - Generate role-specific questions (behavioral, technical, situational)
  - Evaluate candidate responses and track evaluated_skills
  - Manage multi-turn conversational flow with full transcript memory
  - Automatically trigger handoff to Verifier/Critic after 6 questions

Tools used:
  - record_answer        → appends each Q&A turn to the transcript (multi-turn memory)
  - conclude_interview   → flips phase to VERIFICATION when question_count >= 6
  - get_transcript_summary → lets the agent read earlier answers for follow-up questions
  - google_search        → (optional) look up role-specific interview patterns

State Locking:
  Every tool that writes shared state acquires/releases the `agent_lock`
  field so concurrent agents cannot overwrite each other's data.
"""

from google.adk.agents import Agent
from google.adk.tools import google_search
from shared_state import InterviewPhase
import datetime


# ---------------------------------------------------------------------------
# Shared-state helpers  (handle both dict and dataclass state)
# ---------------------------------------------------------------------------

def _acquire_lock(state, agent_name: str) -> bool:
    """Try to acquire the agent lock. Returns True if acquired, False if busy."""
    if isinstance(state, dict):
        if state.get("agent_lock") not in (None, "", agent_name):
            return False
        state["agent_lock"] = agent_name
    else:
        current = getattr(state, "agent_lock", None)
        if current not in (None, "", agent_name):
            return False
        state.agent_lock = agent_name
    return True


def _release_lock(state):
    if isinstance(state, dict):
        state["agent_lock"] = None
    else:
        if hasattr(state, "agent_lock"):
            state.agent_lock = None


def _get(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _set(state, key, value):
    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)


def _append_turns(state, turn_q: dict, turn_a: dict):
    """Write a Q+A pair into the transcript regardless of state shape."""
    transcript = _get(state, "transcript", None)
    if transcript is None:
        # Flat dict state (ADK InMemorySessionService default)
        turns = _get(state, "transcript_turns", [])
        turns.append(turn_q)
        turns.append(turn_a)
        _set(state, "transcript_turns", turns)
    elif isinstance(transcript, dict):
        transcript.setdefault("turns", [])
        transcript["turns"].append(turn_q)
        transcript["turns"].append(turn_a)
        _set(state, "transcript", transcript)
    else:
        # Dataclass InterviewTranscript
        transcript.turns.append(turn_q)
        transcript.turns.append(turn_a)


def _read_turns(state) -> list:
    transcript = _get(state, "transcript", None)
    if transcript is None:
        return _get(state, "transcript_turns", [])
    if isinstance(transcript, dict):
        return transcript.get("turns", [])
    return transcript.turns


# ---------------------------------------------------------------------------
# Tool 1: record_answer  ← multi-turn memory
# ---------------------------------------------------------------------------

def record_answer(tool_context, question: str, answer: str) -> dict:
    """
    Append one interviewer question + candidate answer to the running
    transcript in session state, then return the updated question count.

    CALL THIS after every candidate response, before asking the next question.
    This is the mechanism that gives the agent memory across turns.

    Args:
        question: The exact question the interviewer just asked.
        answer:   The candidate's verbatim response.

    Returns:
        dict with keys:
          status        – "recorded" | "locked"
          question_count – total questions asked so far (int)
          message        – human-readable confirmation
    """
    state = tool_context.state

    if not _acquire_lock(state, "simulation_specialist"):
        return {
            "status": "locked",
            "message": "Another agent is writing to shared state. Retry in a moment.",
            "question_count": _get(state, "question_count", 0),
        }

    try:
        now = datetime.datetime.utcnow().isoformat()
        _append_turns(
            state,
            {"role": "interviewer", "text": question, "timestamp": now},
            {"role": "candidate",   "text": answer,   "timestamp": now},
        )
        count = _get(state, "question_count", 0) + 1
        _set(state, "question_count", count)

        return {
            "status": "recorded",
            "question_count": count,
            "message": f"Turn {count} saved to transcript.",
        }
    finally:
        _release_lock(state)


# ---------------------------------------------------------------------------
# Tool 2: get_transcript_summary  ← cross-question memory retrieval
# ---------------------------------------------------------------------------

def get_transcript_summary(tool_context) -> dict:
    """
    Return all previous Q&A pairs so you can reference earlier answers
    in follow-up questions.

    Example usage: "You mentioned X in Question 1 — can you expand on that?"

    Returns:
        dict with keys:
          question_count  – total questions asked so far (int)
          turns           – list of {q_num, question, answer}
    """
    state = tool_context.state
    raw_turns = _read_turns(state)

    paired = []
    i = 0
    q_num = 0
    while i < len(raw_turns) - 1:
        if (raw_turns[i]["role"] == "interviewer"
                and raw_turns[i + 1]["role"] == "candidate"):
            q_num += 1
            paired.append({
                "q_num":    q_num,
                "question": raw_turns[i]["text"],
                "answer":   raw_turns[i + 1]["text"],
            })
            i += 2
        else:
            i += 1

    return {
        "question_count": _get(state, "question_count", 0),
        "turns": paired,
    }


# ---------------------------------------------------------------------------
# Tool 3: conclude_interview  ← handoff trigger
# ---------------------------------------------------------------------------

def conclude_interview(tool_context) -> str:
    """
    Flip SessionState.phase to VERIFICATION to hand off to the Verifier/Critic.

    Only callable when question_count >= 6.  Acquires the agent lock before
    writing so no other agent can overwrite state simultaneously.

    Returns a confirmation string.
    """
    state = tool_context.state

    count = _get(state, "question_count", 0)
    if count < 6:
        return (
            f"Cannot conclude yet — only {count}/6 questions recorded. "
            "Keep interviewing and call record_answer after each response."
        )

    if not _acquire_lock(state, "simulation_specialist"):
        return (
            "State is locked by another agent. "
            "Handoff will proceed once the lock is released."
        )

    try:
        _set(state, "phase", InterviewPhase.VERIFICATION)
        return (
            "Interview complete. Phase set to VERIFICATION. "
            "Handing off to Verifier/Critic for evaluation."
        )
    finally:
        _release_lock(state)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

SIMULATION_SPECIALIST_INSTRUCTION = """
You are the Simulation Specialist — a sharp, professional interviewer conducting
a realistic 6-question mock interview.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT TURN LOOP  (follow for every single turn)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Call `get_transcript_summary` to retrieve all prior Q&A pairs.
   • If any earlier answer is relevant to your next question, explicitly
     reference it: "You mentioned X in Question N — can you walk me through…"
   • This is mandatory for Question 3 onward.

2. Ask ONE question. Wait for the candidate's response.

3. Call `record_answer(question=<your exact question>, answer=<candidate response>)`.
   • The return value includes the updated `question_count`.

4. Check question_count:
   • question_count < 6  → loop back to step 1.
   • question_count >= 6 → call `conclude_interview`, then say:
     "Interview complete. Handing off to Verifier/Critic for evaluation."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION STRATEGY  (draw from these sources)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Priority order:
  1. `job_context.focus_areas`  — skill gaps to probe
  2. `job_context.required_skills` — role requirements
  3. `resume.skills`           — depth-check claimed skills

Mix question types across 6 turns:
  • Behavioural  → "Tell me about a time when…"
  • Situational  → "How would you handle…"
  • Technical    → Role-specific knowledge check
  • Follow-up    → References a prior answer (use get_transcript_summary)

━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━
• One question per turn. Never stack multiple questions.
• Acknowledge the answer briefly (≤1 sentence) before the next question.
• NO coaching feedback or scores during the interview — save for the report.
• NEVER ask about age, gender, race, disability, or any personal attribute.
• If the candidate asks to end early: record the last answer, then call
  conclude_interview.
• You may call `google_search` for role-specific question inspiration.
"""

simulation_specialist_agent = Agent(
    name="simulation_specialist",
    model="gemini-2.5-flash-lite",
    description=(
        "Conducts a realistic 6-question mock interview. Remembers every answer "
        "via the shared transcript and references earlier responses in later questions. "
        "Automatically flips phase to VERIFICATION after question 6."
    ),
    instruction=SIMULATION_SPECIALIST_INSTRUCTION
)
