# tools/interview_tools.py
from shared_state import SessionState

def conclude_interview(state: SessionState) -> str:
    """
    Call this tool only when the question limit has been reached 
    to move the candidate to the feedback phase.
    """
    state.phase = "VERIFICATION"
    return "Interview concluded. The Verifier will now analyze your performance."