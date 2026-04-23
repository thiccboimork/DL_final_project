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
from google.adk.agents import Agent
from google.adk.tools import google_search, AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Import your existing agents and tools
from google.adk.runners import Runner
from agents.context_optimizer import context_optimizer_agent
from agents.simulation_specialist import simulation_specialist_agent
from tools.resume_parser import parse_resume
from tools.vector_memory import retrieve_user_profile, store_user_profile

# 1. Set environment variables for Vertex AI
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
os.environ["GOOGLE_CLOUD_PROJECT"] =  "project-ebc25092-1828-435a-a1c"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# 2. Define the specialized Search Agent
# This isolates the built-in search tool to avoid the 400 error.
search_agent = Agent(
    name="web_search_agent",
    model="gemini-2.5-flash-lite",
    instruction="""You are a web search assistant. 
    Your only job is to use the google_search tool to find job requirements, 
    company values, and interview patterns for specific roles.""",
    tools=[google_search]
)

# 3. Configure the Context Optimizer to use the Search Agent as a tool
context_optimizer_agent.model = "gemini-2.5-flash-lite"
context_optimizer_agent.tools = [
    parse_resume,
    retrieve_user_profile,
    store_user_profile,
    AgentTool(agent=search_agent)
]

# 4. Standard ADK Runner Setup
session_service = InMemorySessionService()

def get_runner():
    return Runner(
        agent=context_optimizer_agent,
        app_name="InterviewPrepper",
        session_service=session_service
    )