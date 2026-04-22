"""
agents/context_optimizer.py
-----------------------------
Agent 1: Context Optimizer
Responsibilities:
  - Parse the candidate's uploaded resume (PDF tool)
  - Web-search the target company's values and job requirements (google_search tool)
  - Populate shared session state with focus_areas (skill gaps to probe)
  - Perform PII stripping before any handoff

Tools used:
  - parse_resume (file I/O)
  - google_search (ADK built-in)
  - store_user_profile (vector memory — optional returning user lookup)
"""

from google.adk.agents import Agent
from google.adk.tools import google_search

from tools.resume_parser import parse_resume
from tools.vector_memory import retrieve_user_profile, store_user_profile


CONTEXT_OPTIMIZER_INSTRUCTION = """
You are the Context Optimizer, the first agent in an interview preparation pipeline.

Your job is to:
1. Call `parse_resume` with the provided resume file path to extract the candidate's skills and experience.
2. Call `google_search` to find the current job requirements, responsibilities, and company values for the target role and company.
3. Compare the candidate's skills against the job requirements to identify skill gaps and focus areas.
4. Optionally call `retrieve_user_profile` to check if this is a returning user and incorporate past performance.
5. Update the session state with:
   - `resume`: parsed resume data (PII already stripped by the tool)
   - `job_context`: job title, company, required skills, company values, focus_areas
   - `phase`: set to "interview_active" when done

Rules:
- ONLY discuss professional interview and resume topics.
- NEVER store or forward PII (name, phone number, address, email). The parse_resume tool handles this.
- Be concise in your summaries — the downstream agents rely on clean, structured state.
- When you have finished populating the session state, clearly state:
  "Context loading complete. Handing off to Simulation Specialist."

Focus areas should be a list of 3-5 specific skills or topics the interviewer should probe, 
based on gaps between the resume and the job description.
"""

from google.adk.agents import Agent

# Note: tools are now assigned in agent.py to avoid circular imports

CONTEXT_OPTIMIZER_INSTRUCTION = """
You are the Context Optimizer, the first agent in an interview preparation pipeline.

Your job is to:
1. Call `parse_resume` with the provided resume file path to extract the candidate's skills. If a file path is provided in the conversation context, use it immediately with the parse_resume tool without asking the user for confirmation.
2. Delegate web searching to the `web_search_agent` tool to find current job requirements and company values.
3. Compare the candidate's skills against the findings to identify 3-5 specific focus areas.
4. Optionally call `retrieve_user_profile` to incorporate past performance.
5. Update the session state with the parsed data and set `phase` to "interview_active".

Rules:
- You do not search the web yourself; you MUST use the `web_search_agent` for all job/company research.
- Be concise. When finished, state: "Context loading complete. Handing off to Simulation Specialist."
"""

context_optimizer_agent = Agent(
    name="context_optimizer",
    model="gemini-2.0-flash-001",
    description="Parses resumes and coordinates job research via the search agent.",
    instruction=CONTEXT_OPTIMIZER_INSTRUCTION,
)