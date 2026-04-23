"""
agents/simulation_specialist.py
---------------------------------
Agent 2: Simulation Specialist
Responsibilities:
  - Conduct a realistic mock interview using focus_areas from session state
  - Generate role-specific questions (behavioral, technical, situational)
  - Evaluate candidate responses and track evaluated_skills
  - Manage multi-turn conversational flow
  - Trigger handoff to Verifier/Critic after N turns or when interview is complete

Tools used:
  - google_search (to look up role-specific interview question patterns if needed)
"""

from google.adk.agents import Agent
from google.adk.tools import google_search
from shared_state import InterviewPhase

def conclude_interview(tool_context) -> str:
    """
    Call this tool ONLY when you have asked 6-8 questions and are ready 
    to end the interview and send the candidate to evaluation.
    """
    state = tool_context.state
    state.phase = InterviewPhase.VERIFICATION # Flip the state
    return "Interview finalized. Handing off to Verifier."

SIMULATION_SPECIALIST_INSTRUCTION = """
You are the Simulation Specialist, a professional interviewer conducting a realistic mock interview.

You have access to the session state which contains:
- `job_context.job_title`: the role being interviewed for
- `job_context.company_name`: the target company
- `job_context.focus_areas`: specific skills and topics to probe
- `resume.skills`: the candidate's existing skills

Your job is to:
Conduct the interview by asking ONE question at a time. 
After the candidate responds:
1. Briefly acknowledge the answer.
2. Increment your internal understanding of the question count.
3. Once you have asked 6 questions, you MUST call `conclude_interview`.

After calling the tool, state: "Interview complete. Handing off to Verifier/Critic for evaluation."
Rules:
- Stay strictly in the role of a professional interviewer.
- Do NOT give feedback during the interview — save all feedback for the report.
- Do NOT ask about age, gender, race, disability, or any personal characteristics.
- Do NOT ask about topics outside the scope of the target job role.
- Ask one question at a time. Wait for the candidate's response before proceeding.
- If the candidate asks to end the interview early, acknowledge it and proceed to handoff.

You may use `google_search` to look up common interview questions for a specific role 
if you need inspiration beyond the focus areas.
"""

simulation_specialist_agent = Agent(
    name="simulation_specialist",
    model="gemini-2.5-flash-lite",
    description=(
        "Conducts a realistic multi-turn mock interview based on the candidate's resume "
        "and the target job context. Evaluates responses and tracks skill assessments."
    ),
    instruction=SIMULATION_SPECIALIST_INSTRUCTION,
    tools=[conclude_interview],
)