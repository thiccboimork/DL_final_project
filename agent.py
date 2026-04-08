"""
agent.py
---------
Root agent entrypoint for the Interview ChatBot.
Wires together the three specialized agents using ADK's SequentialAgent pattern.

Run locally with:
    adk web
or:
    adk run interview_chatbot

The root agent acts as the orchestrator, routing to the correct sub-agent
based on the current interview phase in session state.
"""

import os
from dotenv import load_dotenv
from google.adk.agents import Agent

from agents.context_optimizer import context_optimizer_agent
from agents.simulation_specialist import simulation_specialist_agent
from agents.verifier_critic import verifier_critic_agent

load_dotenv()

ROOT_INSTRUCTION = """
You are the orchestrator of an AI-powered interview preparation system.

You manage a pipeline of three specialized agents:
1. context_optimizer — runs first, parses the resume and loads job context
2. simulation_specialist — conducts the mock interview
3. verifier_critic — validates outputs and generates the performance report

Routing rules:
- If `phase` is "context_loading" (or not yet set): delegate to `context_optimizer`
- If `phase` is "interview_active": delegate to `simulation_specialist`
- If `phase` is "verification": delegate to `verifier_critic`
- If `phase` is "report_ready": the pipeline is complete. Thank the user.

When a user first messages you, greet them warmly, explain the system, and ask for:
1. Their resume (PDF file path or pasted text)
2. The job title and company they are targeting

Once you have both, set `phase` to "context_loading" and hand off to the context_optimizer.

Always stay encouraging and professional. This system is designed to help candidates succeed.
"""

# Root agent with sub-agents registered
root_agent = Agent(
    name="interview_chatbot",
    model="gemini-2.0-flash",
    description="AI-powered interview preparation and resume review multi-agent system.",
    instruction=ROOT_INSTRUCTION,
    sub_agents=[
        context_optimizer_agent,
        simulation_specialist_agent,
        verifier_critic_agent,
    ],
)