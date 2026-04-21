import streamlit as st
import asyncio
from agent import get_runner
from google.genai import types
from shared_state import InterviewPhase

st.set_page_config(page_title="Interview Prepper", layout="wide")

# 1. Initialize ADK Runner and Session in Streamlit
if "runner" not in st.session_state:
    st.session_state.runner = get_runner()
    st.session_state.session_id = "session_123" # Or use a unique ID
    st.session_state.messages = []
    st.session_state.current_phase = InterviewPhase.CONTEXT_LOADING

    asyncio.run(st.session_state.runner.session_service.create_session(
        user_id="user_1", 
        session_id=st.session_state.session_id,
        app_name="InterviewPrepper"
    ))

# 2. File Upload for Resume
# --- SIDEBAR STATUS INDICATOR ---
with st.sidebar:
    st.header("Project Status")
    
    # Define display names using your Enum members as keys
    phase_display = {
        InterviewPhase.CONTEXT_LOADING: "🔍 Optimizing Context",
        InterviewPhase.INTERVIEW_ACTIVE: "🎙️ Interview in Progress",
        InterviewPhase.VERIFICATION: "✅ Verifying Responses",
        InterviewPhase.REPORT_READY: "📄 Report Ready",
    }
    
    # 1. Get the current phase from state
    # We default to the first enum value if it's the "Loading..." placeholder string
    current_phase = st.session_state.current_phase
    
    # 2. Display the status info
    display_text = phase_display.get(current_phase, "⏳ Initializing...")
    st.info(f"**Current State:**\n{display_text}")
    
    # 3. Simple Progress Bar
    # We create a list of the string values to find the index
    phase_values = [p.value for p in InterviewPhase]
    
    if current_phase in phase_values:
        progress_val = (phase_values.index(current_phase) + 1) / len(phase_values)
        st.progress(progress_val)
    else:
        st.progress(0.0)

# --- MAIN CHAT UI ---
st.title("🤖 Interview Prepper")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We use st.status to show the "thinking" process live
        with st.status("Agent is working...", expanded=False) as status:
            response_placeholder = st.empty()
            full_response = ""
            
            async def run_chat_logic():
                content = types.Content(role="user", parts=[types.Part(text=prompt)])
                
                async for event in st.session_state.runner.run_async(
                    user_id="user_1", 
                    session_id=st.session_state.session_id, 
                    new_message=content
                ):
                    # Update the UI phase by looking at the actual session state
                    session = await st.session_state.runner.session_service.get_session(
                        user_id="user_1", 
                        session_id=st.session_state.session_id,
                        app_name="InterviewPrepper"
                    )
                    if session and hasattr(session.state, 'phase'):
                        st.session_state.current_phase = session.state.phase
                    
                    if event.is_final_response():
                        return event.content.parts[0].text
            
            # Run the logic and update the UI
            full_response = asyncio.run(run_chat_logic())
            status.update(label="Response complete!", state="complete", expanded=False)
            response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun() # Refresh to update sidebar status immediately