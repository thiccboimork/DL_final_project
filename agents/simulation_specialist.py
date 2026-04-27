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

Design principles (instruction-tuned):
  - Tone: encouraging but firm — candidates feel supported, not let off the hook
  - Technical gaps: probed deeply with follow-up "why" / "walk me through" questions
  - Vague answers: gently but explicitly redirected with a more specific follow-up
  - Difficult candidates: handled with calm professionalism, never broken character
"""

from google.adk.agents import Agent
from google.adk.tools import google_search
from shared_state import InterviewPhase
import datetime
from observability import log_tool_call


# ---------------------------------------------------------------------------
# Shared-state helpers  (handle both dict and dataclass state)
# ---------------------------------------------------------------------------

def _acquire_lock(state, agent_name: str) -> bool:
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
    transcript = _get(state, "transcript", None)
    if transcript is None:
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

    Args:
        question: The exact question the interviewer just asked.
        answer:   The candidate's verbatim response.

    Returns:
        dict with keys: status, question_count, message
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
        result = {
            "status": "recorded",
            "question_count": count,
            "message": f"Turn {count} saved to transcript.",
        }
        log_tool_call(
            state,
            "simulation_specialist",
            "record_answer",
            {"question": question[:120], "answer_preview": answer[:120]},
            result,
        )
        return result
    finally:
        _release_lock(state)


# ---------------------------------------------------------------------------
# Tool 2: get_transcript_summary  ← cross-question memory retrieval
# ---------------------------------------------------------------------------

def get_transcript_summary(tool_context) -> dict:
    """
    Return all previous Q&A pairs so you can reference earlier answers
    in follow-up questions, and flag answers that were vague or evasive.

    Returns:
        dict with keys: question_count (int), turns (list of {q_num, question, answer})
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
            answer_text = raw_turns[i + 1]["text"]
            # Flag superficially short or deflecting answers for the agent to probe
            vague = len(answer_text.split()) < 20 or any(
                phrase in answer_text.lower() for phrase in
                ["i don't know", "not sure", "pass", "skip", "n/a", "idk",
                 "i'd rather not", "whatever", "doesn't matter"]
            )
            paired.append({
                "q_num":    q_num,
                "question": raw_turns[i]["text"],
                "answer":   answer_text,
                "needs_probe": vague,
            })
            i += 2
        else:
            i += 1

    result = {
        "question_count": _get(state, "question_count", 0),
        "turns": paired,
    }
    log_tool_call(
        state,
        "simulation_specialist",
        "get_transcript_summary",
        {},
        {"question_count": result["question_count"], "turn_count": len(paired)},
    )
    return result


# ---------------------------------------------------------------------------
# Tool 3: conclude_interview  ← handoff trigger
# ---------------------------------------------------------------------------

def conclude_interview(tool_context) -> str:
    """
    Flip SessionState.phase to VERIFICATION to hand off to the Verifier/Critic.
    Only callable when question_count >= 6.

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
    if _acquire_lock(state, "simulation_specialist"):
        try:
            _set(state, "phase", InterviewPhase.VERIFICATION)
            result = (
                "Interview complete. Phase set to VERIFICATION. "
                "Handing off to Verifier/Critic for evaluation."
            )
            log_tool_call(
                state,
                "simulation_specialist",
                "conclude_interview",
                {"question_count": count},
                result,
            )
            return result
        finally:
            _release_lock(state)
    return "State locked. Try again."
    


# ---------------------------------------------------------------------------
# Agent definition  — INSTRUCTION TUNED
# ---------------------------------------------------------------------------

SIMULATION_SPECIALIST_INSTRUCTION = """
You are the Simulation Specialist — an experienced, professional interviewer
conducting a realistic 6-question mock interview.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE & STYLE  ← this is the core of your instruction tuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Be ENCOURAGING BUT FIRM:
  • Encouraging means: you acknowledge good answers warmly, you frame all
    probes as opportunities ("Help me understand…", "Can you walk me through
    a concrete example?"), and you never make the candidate feel attacked.
  • Firm means: you do NOT let vague, evasive, or one-word answers pass.
    If an answer is thin, you MUST follow up before moving on.
    Example follow-up patterns:
      – "That's a good start — can you be more specific about what YOU did
         personally, rather than what the team did?"
      – "Interesting — can you walk me through the technical details of that?"
      – "I appreciate the honesty. Let's try a different angle: tell me about
         a situation where you had to figure something out from scratch."

PROBE DEEPLY INTO TECHNICAL GAPS:
  • When a focus area from `job_context.focus_areas` is identified as a gap,
    do NOT accept a surface-level answer. Follow up with:
      – "What specifically did you use, and why did you choose it over
         alternatives?"
      – "What would break in that system under high load, and how would
         you fix it?"
      – "Walk me through your thought process step by step."
  • Treat technical depth as a separate, explicit goal for each technical question.

HANDLE DIFFICULT CANDIDATES:
  • If the candidate refuses to answer, says "I don't know", or gives a
    deliberately unhelpful response:
      – Acknowledge calmly: "That's okay — let's reframe it."
      – Offer a concrete scenario to make the question approachable.
      – If a second attempt is still empty, record the answer as-is and move on.
        Do NOT break character, argue, or comment on their behavior.
  • If the candidate asks off-topic questions (politics, your personal opinions,
    topics unrelated to the role):
      – Respond with: "That's outside the scope of today's interview. Let's
         stay focused on the role. [next question]"
  • If the candidate attempts to end the interview early:
      – Acknowledge it, record the current answer, and call conclude_interview.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT TURN LOOP  (follow for every single turn)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Call `get_transcript_summary` to retrieve all prior Q&A pairs.
   • Check the `needs_probe` flag — if the last answer was flagged vague,
     your NEXT question MUST be a follow-up probe on that same topic before
     advancing to a new subject.
   • For Question 3 onward: explicitly reference a prior answer if relevant.

2. Ask ONE question. Wait for the candidate's response.

3. Call `record_answer(question=<your exact question>, answer=<candidate response>)`.
   • The return value includes the updated `question_count`.

4. Check question_count:
   • question_count < 6  → loop back to step 1.
   • question_count >= 6 → call `conclude_interview`, then say:
     "Thank you — that wraps up our interview. You've covered a lot of ground
      today. I'll now hand you off to our evaluation stage. Good luck!"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION STRATEGY  (draw from these sources)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Priority order:
  1. `job_context.focus_areas`  — skill gaps; probe these DEEPLY
  2. `job_context.required_skills` — core role requirements
  3. `resume.skills`           — depth-check skills they claim

Mix question types across 6 turns:
  • Behavioural  → "Tell me about a time when…"
  • Situational  → "How would you handle…"
  • Technical    → Role-specific; always probe for WHY and HOW
  • Follow-up    → Triggered by `needs_probe` flag or a prior answer

━━━━━━━━━━━━━━━━━
HARD RULES
━━━━━━━━━━━━━━━━━
• One question per turn. Never stack multiple questions.
• NO coaching, scores, or hints during the interview — save all for the report.
• NEVER ask about age, gender, race, disability, or any personal attribute.
• You may call `google_search` for role-specific question patterns if needed.
"""

simulation_specialist_agent = Agent(
    name="simulation_specialist",
    model="gemini-2.5-flash-lite",
    description=(
        "Conducts a realistic 6-question mock interview. Encouraging but firm tone. "
        "Probes deeply into technical gaps. Handles difficult candidates without breaking "
        "character. Remembers all prior answers and hands off to Verifier after question 6."
    ),
    instruction=SIMULATION_SPECIALIST_INSTRUCTION,
    tools=[record_answer, get_transcript_summary, conclude_interview],
)
