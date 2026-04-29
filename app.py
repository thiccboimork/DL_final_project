import streamlit as st
import streamlit.components.v1 as components
from streamlit_pdf_viewer import pdf_viewer
import asyncio
from agent import get_runner
from google.genai import types
from shared_state import InterviewPhase
import os
import speech_recognition as sr
import tempfile
import json
import datetime

from agents.context_optimizer import context_optimizer_agent
from agents.simulation_specialist import simulation_specialist_agent
from guardrails import get_guardrail_capabilities, validate_output
from observability import (
    DEFAULT_GUARDRAIL_CONFIG,
    log_guardrail_event,
    log_tool_call,
)

import base64

def show_final_report():
    """Renders the PDF report in the main chat area."""
    session = asyncio.run(get_live_session())
    if session:
        report_path = _get_state_value(session.state, "report_path")
        
        if report_path and os.path.exists(report_path):
            st.divider()
            st.subheader("📄 Your Professional Interview Report")
            # Render in the main chat area, not the sidebar
            pdf_viewer(report_path, width="100%", height=800)
        else:
            st.info("⌛ Finalizing the visual report layout...")

    report_path = _get_state_value(session.state, "report_path")

st.set_page_config(page_title="Interview Prepper", layout="wide")

if not os.path.exists("reports"):
    os.makedirs("reports")

def _get_state_value(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _set_state_value(state, key, value) -> None:
    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)


def ensure_local_observability_defaults() -> None:
    if "ui_tool_call_log" not in st.session_state:
        st.session_state.ui_tool_call_log = []
    if "ui_guardrail_flags" not in st.session_state:
        st.session_state.ui_guardrail_flags = []
    if "ui_guardrail_events" not in st.session_state:
        st.session_state.ui_guardrail_events = []
    if "ui_guardrail_config" not in st.session_state:
        st.session_state.ui_guardrail_config = DEFAULT_GUARDRAIL_CONFIG.copy()


def append_local_log(agent: str, tool: str, args: dict, result) -> None:
    ensure_local_observability_defaults()
    result_summary = result if isinstance(result, str) else str(result)
    st.session_state.ui_tool_call_log.append({
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent": agent,
        "tool": tool,
        "args": args,
        "result_summary": result_summary[:300],
    })
    st.session_state.ui_tool_call_log = st.session_state.ui_tool_call_log[-50:]


def append_guardrail_event(verdict: str, flags: list[str], metadata: dict) -> None:
    ensure_local_observability_defaults()
    st.session_state.ui_guardrail_events.append({
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verdict": verdict,
        "flags": flags,
        "metadata": metadata,
    })
    st.session_state.ui_guardrail_events = st.session_state.ui_guardrail_events[-30:]


async def ensure_session_defaults() -> None:
    ensure_local_observability_defaults()
    session = await st.session_state.runner.session_service.get_session(
        user_id="user_1",
        session_id=st.session_state.session_id,
        app_name="InterviewPrepper"
    )
    if not session:
        return

    defaults = {
        "phase": InterviewPhase.CONTEXT_LOADING.value,
        "question_count": 0,
        "guardrail_flags": [],
        "tool_call_log": [],
        "guardrail_config": DEFAULT_GUARDRAIL_CONFIG.copy(),
        "report_path": None,
        "agent_lock": None,
    }
    for key, value in defaults.items():
        if _get_state_value(session.state, key, None) is None:
            _set_state_value(session.state, key, value)


async def set_interview_phase(phase: InterviewPhase, question_count: int | None = None) -> None:
    session = await st.session_state.runner.session_service.get_session(
        user_id="user_1",
        session_id=st.session_state.session_id,
        app_name="InterviewPrepper"
    )
    if not session:
        return

    _set_state_value(session.state, "phase", phase.value)
    if question_count is not None:
        _set_state_value(session.state, "question_count", question_count)


async def get_live_session():
    return await st.session_state.runner.session_service.get_session(
        user_id="user_1",
        session_id=st.session_state.session_id,
        app_name="InterviewPrepper"
    )


def append_frontend_log(tool: str, args: dict, result) -> None:
    append_local_log("streamlit_ui", tool, args, result)
    session = asyncio.run(get_live_session())
    if not session:
        return
    log_tool_call(session.state, "streamlit_ui", tool, args, result)


def run_response_guardrail_scan(response_text: str) -> None:
    session = asyncio.run(get_live_session())
    if not session:
        return

    job_role = ""
    if isinstance(session.state, dict):
        job_context = session.state.get("job_context", {})
        if isinstance(job_context, dict):
            job_role = job_context.get("job_title", "")
    else:
        job_context = getattr(session.state, "job_context", None)
        if job_context is not None:
            job_role = getattr(job_context, "job_title", "")

    flags = validate_output(response_text, job_role)
    if flags:
        existing = _get_state_value(session.state, "guardrail_flags", [])
        existing.extend(flags)
        _set_state_value(session.state, "guardrail_flags", existing[-20:])
        st.session_state.ui_guardrail_flags.extend(flags)
        st.session_state.ui_guardrail_flags = st.session_state.ui_guardrail_flags[-20:]
        verdict = "WARN"
    else:
        verdict = "PASS"

    append_local_log(
        "guardrail_system",
        "guardrail_scan",
        {"stage": "assistant_response", "phase": str(st.session_state.current_phase)},
        {"verdict": verdict, "flags": flags},
    )
    append_guardrail_event(
        verdict,
        flags,
        {"stage": "assistant_response", "phase": str(st.session_state.current_phase)},
    )
    log_guardrail_event(
        session.state,
        stage="assistant_response",
        verdict=verdict,
        flags=flags,
        metadata={"phase": str(st.session_state.current_phase)},
    )


def transcribe_uploaded_audio(audio_file) -> str | None:
    recognizer = sr.Recognizer()
    temp_path = None

    try:
        suffix = os.path.splitext(audio_file.name)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(audio_file.getbuffer())
            temp_path = temp_audio.name

        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)

        transcript = recognizer.recognize_google(audio_data)
        append_frontend_log(
            "transcribe_uploaded_audio",
            {"filename": audio_file.name},
            {"status": "success", "transcript_preview": transcript[:120]},
        )
        return transcript
    except sr.UnknownValueError:
        st.warning("I couldn't understand that recording. Please try again or type your answer.")
        append_frontend_log(
            "transcribe_uploaded_audio",
            {"filename": audio_file.name},
            {"status": "error", "message": "unknown_value"},
        )
    except sr.RequestError as exc:
        st.error(f"Speech recognition is unavailable right now: {exc}")
        append_frontend_log(
            "transcribe_uploaded_audio",
            {"filename": audio_file.name},
            {"status": "error", "message": str(exc)},
        )
    except Exception as exc:
        st.error(f"Audio transcription failed: {exc}")
        append_frontend_log(
            "transcribe_uploaded_audio",
            {"filename": audio_file.name},
            {"status": "error", "message": str(exc)},
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return None


def render_tts_controls(message_text: str, index: int) -> None:
    col_play, col_stop = st.columns([1, 1])
    with col_play:
        if st.button("Play response", key=f"play_tts_{index}"):
            st.session_state.tts_text = message_text
            st.session_state.tts_nonce = st.session_state.get("tts_nonce", 0) + 1
            append_frontend_log(
                "play_tts",
                {"message_index": index},
                {"status": "queued", "text_preview": message_text[:120]},
            )
            st.rerun()
    with col_stop:
        if st.button("Stop audio", key=f"stop_tts_{index}"):
            st.session_state.tts_text = ""
            st.session_state.tts_nonce = st.session_state.get("tts_nonce", 0) + 1
            append_frontend_log(
                "stop_tts",
                {"message_index": index},
                {"status": "stopped"},
            )
            st.rerun()


def render_tts_trigger() -> None:
    if "tts_nonce" not in st.session_state:
        return

    safe_text = json.dumps(st.session_state.get("tts_text", ""))
    nonce = st.session_state.get("tts_nonce", 0)
    components.html(
        f"""
        <script>
          const nonce = {nonce};
          const text = {safe_text};
          window.speechSynthesis.cancel();
          if (text) {{
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1;
            utterance.pitch = 1;
            window.speechSynthesis.speak(utterance);
          }}
        </script>
        """,
        height=0,
    )


def render_message_history() -> None:
    for index, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if (
                msg["role"] == "assistant"
                and st.session_state.get("tts_enabled", False)
                and msg["content"] == st.session_state.get("last_tts_message")
            ):
                render_tts_controls(msg["content"], index)


def interview_has_started() -> bool:
    return any(msg["role"] == "assistant" for msg in st.session_state.messages)


def submit_user_prompt(prompt_text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    append_frontend_log(
        "submit_user_prompt",
        {"phase": str(st.session_state.current_phase)},
        {"status": "submitted", "prompt_preview": prompt_text[:120]},
    )
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.status("Agent is working...", expanded=False) as status:
            response_placeholder = st.empty()
            full_response = ""

            async def run_chat_logic():
                await ensure_session_defaults()
                if st.session_state.current_phase == InterviewPhase.CONTEXT_LOADING:
                    st.session_state.runner.agent = context_optimizer_agent
                elif st.session_state.current_phase == InterviewPhase.INTERVIEW_ACTIVE:
                    st.session_state.runner.agent = simulation_specialist_agent
                elif st.session_state.current_phase in (
                    InterviewPhase.VERIFICATION, InterviewPhase.REPORT_READY
                ):
                    from agents.verifier_critic import verifier_critic_agent
                    st.session_state.runner.agent = verifier_critic_agent

                context_prefix = ""
                if "resume_path" in st.session_state:
                    context_prefix = (
                        f"(System Note: The user's resume is located at: "
                        f"{st.session_state.resume_path})\n"
                    )

                full_prompt = context_prefix + prompt_text
                content = types.Content(role="user", parts=[types.Part(text=full_prompt)])

                async for event in st.session_state.runner.run_async(
                    user_id="user_1",
                    session_id=st.session_state.session_id,
                    new_message=content
                ):
                    session = await st.session_state.runner.session_service.get_session(
                        user_id="user_1",
                        session_id=st.session_state.session_id,
                        app_name="InterviewPrepper"
                    )
                    if session:
                        raw_phase = (
                            session.state.get("phase")
                            if isinstance(session.state, dict)
                            else getattr(session.state, "phase", None)
                        )
                        if raw_phase:
                            try:
                                st.session_state.current_phase = InterviewPhase(raw_phase)
                            except ValueError:
                                pass

                    if event.is_final_response():
                        response_text = event.content.parts[0].text

                        handoff_signals = [
                            "Handing off to Verifier",
                            "Phase set to VERIFICATION",
                            "INTERVIEW_COMPLETE_SIGNAL",
                        ]

                        if any(signal in response_text for signal in handoff_signals):
                            st.session_state.current_phase = InterviewPhase.VERIFICATION
                            return response_text

                        if "Handing off to Simulation Specialist" in response_text:
                            st.session_state.current_phase = InterviewPhase.INTERVIEW_ACTIVE
                            session = await st.session_state.runner.session_service.get_session(
                                user_id="user_1",
                                session_id=st.session_state.session_id,
                                app_name="InterviewPrepper"
                            )
                            if session:
                                if isinstance(session.state, dict):
                                    session.state["phase"] = InterviewPhase.INTERVIEW_ACTIVE.value
                                else:
                                    session.state.phase = InterviewPhase.INTERVIEW_ACTIVE

                        if "Handing off to Verifier" in response_text or "Phase set to VERIFICATION" in response_text:
                            st.session_state.current_phase = InterviewPhase.VERIFICATION
                        if "INTERVIEW_COMPLETE_SIGNAL" in response_text:
                            st.session_state.current_phase = InterviewPhase.VERIFICATION
                            st.rerun()

                        return response_text

            full_response = asyncio.run(run_chat_logic())
            status.update(label="Response complete!", state="complete", expanded=False)
            response_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            append_frontend_log(
                "assistant_response",
                {"phase": str(st.session_state.current_phase)},
                {"status": "completed", "response_preview": full_response[:160]},
            )
            run_response_guardrail_scan(full_response)
            if st.session_state.get("tts_enabled", False):
                st.session_state.last_tts_message = full_response
            st.rerun()

# 1. Initialize ADK Runner and Session in Streamlit
if "runner" not in st.session_state:
    st.session_state.runner = get_runner()
    st.session_state.session_id = "session_123" # Or use a unique ID
    st.session_state.messages = []
    st.session_state.current_phase = InterviewPhase.CONTEXT_LOADING
    st.session_state.tts_enabled = False
    st.session_state.last_tts_message = None
    st.session_state.tts_text = ""
    st.session_state.tts_nonce = 0
    ensure_local_observability_defaults()

    asyncio.run(st.session_state.runner.session_service.create_session(
        user_id="user_1", 
        session_id=st.session_state.session_id,
        app_name="InterviewPrepper"
    ))
    asyncio.run(ensure_session_defaults())

# 2. File Upload for Resume
# --- SIDEBAR STATUS INDICATOR ---
with st.sidebar:
    st.divider()
    st.subheader("🛠️ Developer Tools")
    if st.button("Enter Interview Mode"):
        st.session_state.current_phase = InterviewPhase.INTERVIEW_ACTIVE
        asyncio.run(ensure_session_defaults())
        asyncio.run(set_interview_phase(InterviewPhase.INTERVIEW_ACTIVE))
        st.rerun()
    if st.button("Skip to Verification"):
        st.session_state.current_phase = InterviewPhase.VERIFICATION
        asyncio.run(ensure_session_defaults())
        asyncio.run(set_interview_phase(InterviewPhase.VERIFICATION, question_count=6))
        st.rerun()

    # ── Synthetic-data skip panel ──────────────────────────────────────────
    st.divider()
    st.subheader("⚡ Quick-Test: Skip Interview")
    st.caption(
        "Load a test-case JSON, inject the synthetic Q&A as the transcript, "
        "and jump straight to Verification — no live interview needed."
    )

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    tc_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".json")]
    ) if os.path.isdir(data_dir) else []

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_tc = st.selectbox(
            "Test case",
            options=tc_files,
            index=0 if tc_files else None,
            key="quick_test_selector",
        )
    with col_btn:
        st.write("")  # vertical alignment spacer
        run_quick = st.button("▶ Run", key="quick_test_run")

    if run_quick and selected_tc:
        try:
            from evaluation.json_test_loader import (
                load_test_cases_from_file,
                session_state_to_agent_context,
            )
            tc_path = os.path.join(data_dir, selected_tc)
            cases = load_test_cases_from_file(tc_path)
            tc_state, tc_bench = cases[0]

            # ── Inject synthetic Q&A into the session transcript ──────────
            async def _inject_synthetic_session():
                session = await st.session_state.runner.session_service.get_session(
                    user_id="user_1",
                    session_id=st.session_state.session_id,
                    app_name="InterviewPrepper",
                )
                if not session:
                    return

                # Build transcript turns from sample_candidate_responses
                import datetime as _dt
                turns = []
                suggested_qs = tc_bench.suggested_questions
                sample_resps = tc_bench.sample_responses
                for i, resp in enumerate(sample_resps):
                    q = (suggested_qs[i] if i < len(suggested_qs)
                         else f"Question about {resp.get('question_topic', 'the role')}")
                    a = resp.get("response", "(no response)")
                    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    turns.append({"role": "interviewer", "text": q, "timestamp": ts})
                    turns.append({"role": "candidate",   "text": a, "timestamp": ts})

                # Pad with filler Q&A if fewer than 6 pairs
                while len(turns) // 2 < 6:
                    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    turns.append({"role": "interviewer",
                                  "text": "Tell me about a recent professional challenge.",
                                  "timestamp": ts})
                    turns.append({"role": "candidate",
                                  "text": "I led a cross-team project under a tight deadline "
                                          "and coordinated deliverables across four departments.",
                                  "timestamp": ts})

                state = session.state
                if isinstance(state, dict):
                    # Job context
                    state["job_context"] = {
                        "job_title":      tc_state.job_context.job_title,
                        "company_name":   tc_state.job_context.company_name,
                        "required_skills":tc_state.job_context.required_skills,
                        "company_values": tc_state.job_context.company_values,
                        "focus_areas":    tc_state.job_context.focus_areas,
                    }
                    # Resume (no PII)
                    state["resume"] = {
                        "raw_text":        tc_state.resume.raw_text,
                        "skills":          tc_state.resume.skills,
                        "experience_years":tc_state.resume.experience_years,
                        "education":       tc_state.resume.education,
                    }
                    # Transcript
                    state["transcript"] = {"turns": turns, "evaluated_skills": {}}
                    state["transcript_turns"] = turns
                    state["question_count"]   = len(turns) // 2
                    state["phase"]            = "verification"
                    state["guardrail_flags"]  = []
                    state["agent_lock"]       = None
                else:
                    from shared_state import ResumeData, JobContext, InterviewTranscript
                    state.job_context = tc_state.job_context
                    state.resume      = tc_state.resume
                    state.transcript  = InterviewTranscript(
                        turns=turns, evaluated_skills={}
                    )
                    from shared_state import InterviewPhase as _Phase
                    state.phase          = _Phase.VERIFICATION
                    state.question_count = len(turns) // 2
                    state.guardrail_flags = []
                    state.agent_lock      = None

            asyncio.run(_inject_synthetic_session())

            # ── Build the context message the Verifier will receive ────────
            ctx_msg = session_state_to_agent_context(tc_state, tc_bench)
            ctx_msg += (
                "\n\nNote: This is a synthetic-data test run. The transcript has been "
                "pre-populated with sample Q&A. Please grade the candidate, produce "
                "all report sections, and call generate_report now."
            )
            st.session_state.preloaded_context = ctx_msg

            # ── Flip UI phase and inject the initial assistant message ─────
            st.session_state.current_phase = InterviewPhase.VERIFICATION
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"✅ **Synthetic session loaded: {tc_bench.test_case_id}**\n\n"
                    f"**Role:** {tc_state.job_context.job_title} @ "
                    f"{tc_state.job_context.company_name}\n"
                    f"**Difficulty:** {tc_bench.target_difficulty} · "
                    f"**Industry:** {tc_bench.industry}\n\n"
                    f"Transcript pre-populated with {len(tc_bench.sample_responses)} "
                    f"sample Q&A pairs (padded to 6). "
                    "Type **'generate report'** to trigger the Verifier/Critic."
                ),
            })
            st.rerun()

        except Exception as exc:
            st.error(f"Quick-test failed: {exc}")
    st.header("📄 Candidate Documents")
    uploaded_file = st.file_uploader("Upload your PDF resume", type="pdf")

    if uploaded_file is not None:
        # Save the file to a temp folder so the agent can access the path
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Loaded: {uploaded_file.name}")
        # Store the path in session state for the agent to use
        st.session_state.resume_path = os.path.abspath(file_path)

    # ── JSON Test Case Loader ─────────────────────────────────────────────
    st.header("🧪 JSON Test Case")
    st.caption("Load a structured test case to bypass live resume upload and pre-populate the session.")

    json_input_mode = st.radio(
        "Input mode",
        ["Upload .json file", "Paste JSON"],
        horizontal=True,
        key="json_input_mode",
    )

    loaded_test_case = None

    if json_input_mode == "Upload .json file":
        json_file = st.file_uploader(
            "Upload test case (.json)", type="json", key="json_test_uploader"
        )
        if json_file is not None:
            try:
                json_bytes = json_file.read().decode("utf-8")
                from evaluation.json_test_loader import load_test_cases_from_string
                cases = load_test_cases_from_string(json_bytes)
                loaded_test_case = cases[0]  # use first if list
                if len(cases) > 1:
                    st.info(f"File contains {len(cases)} test cases — loaded the first one.")
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")
    else:
        json_text = st.text_area(
            "Paste JSON test case",
            height=150,
            placeholder='{"test_case_id": "TC-001", "metadata": {...}, ...}',
            key="json_test_paste",
        )
        if json_text.strip():
            if st.button("Load Test Case", key="load_json_btn"):
                try:
                    from evaluation.json_test_loader import load_test_cases_from_string
                    cases = load_test_cases_from_string(json_text)
                    loaded_test_case = cases[0]
                except Exception as e:
                    st.error(f"Failed to parse JSON: {e}")

    if loaded_test_case is not None:
        state, benchmarks = loaded_test_case
        st.session_state.loaded_test_benchmarks = benchmarks

        # Pre-populate session with parsed resume + job context
        from evaluation.json_test_loader import session_state_to_agent_context
        context_msg = session_state_to_agent_context(state, benchmarks)
        st.session_state.preloaded_context = context_msg

        # Show a summary so the user knows what was loaded
        with st.expander(f"✅ Loaded: {benchmarks.test_case_id}", expanded=True):
            st.markdown(f"**Job:** {state.job_context.job_title} @ {state.job_context.company_name}")
            st.markdown(f"**Difficulty:** {benchmarks.target_difficulty} | **Industry:** {benchmarks.industry}")
            st.markdown("**Focus areas:**")
            for fa in benchmarks.expected_focus_areas:
                st.markdown(f"- {fa}")
            st.markdown(f"**Must-flag checks:** {len(benchmarks.must_flag)}")
            st.markdown(f"**PII fields to mask:** {', '.join(benchmarks.masking_required)}")
        st.success("Test case loaded — start the interview to run the simulation.")

    st.header("Project Status")
    
    # Define display names using Enum members as keys
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

    session = asyncio.run(get_live_session())
    if session:
        report_path = _get_state_value(session.state, "report_path")
        phase = _get_state_value(session.state, "phase")

        if report_path and os.path.exists(report_path):
            st.subheader("📄 Final Interview Report")
            # This will render the PDF directly on the page
            show_final_report()
            
            # Keep the download button in the sidebar for convenience
            with st.sidebar:
                with open(report_path, "rb") as f:
                    st.download_button("📥 Download PDF", f, file_name=os.path.basename(report_path))
    
    # 3. Simple Progress Bar
    # We create a list of the string values to find the index
    phase_values = [p.value for p in InterviewPhase]
    
    if current_phase in phase_values:
        progress_val = (phase_values.index(current_phase) + 1) / len(phase_values)
        st.progress(progress_val)
    else:
        st.progress(0.0)

    st.header("Transparency")
    guardrail_capabilities = get_guardrail_capabilities()
    st.caption("Explicit policies, inspectable logs, and open evaluation hooks.")
    if st.button("Add test log entry", key="add_test_log_entry"):
        append_frontend_log(
            "manual_test",
            {"source": "sidebar_button"},
            {"status": "ok", "message": "Tool log is live"},
        )
        append_guardrail_event(
            "PASS",
            [],
            {"stage": "manual_test", "phase": str(st.session_state.current_phase)},
        )
        st.rerun()
    with st.expander("Guardrail Policies", expanded=False):
        st.json(st.session_state.get("ui_guardrail_config", guardrail_capabilities["configurable_policies"]))
    with st.expander("Recent Tool Calls", expanded=False):
        recent_logs = st.session_state.get("ui_tool_call_log", [])[-8:]
        if recent_logs:
            for entry in reversed(recent_logs):
                st.caption(
                    f"{entry['timestamp']} | {entry['agent']} | {entry['tool']} | "
                    f"{entry['result_summary']}"
                )
        else:
            st.caption("No tool calls logged yet.")
    with st.expander("Guardrail Status", expanded=False):
        active_flags = st.session_state.get("ui_guardrail_flags", [])
        recent_guardrail_events = st.session_state.get("ui_guardrail_events", [])[-6:]
        pass_count = sum(1 for event in st.session_state.get("ui_guardrail_events", []) if event["verdict"] == "PASS")
        warn_count = sum(1 for event in st.session_state.get("ui_guardrail_events", []) if event["verdict"] == "WARN")
        block_count = sum(1 for event in st.session_state.get("ui_guardrail_events", []) if event["verdict"] == "BLOCK")
        st.caption(f"Scans: {len(st.session_state.get('ui_guardrail_events', []))} | PASS: {pass_count} | WARN: {warn_count} | BLOCK: {block_count}")
        if recent_guardrail_events:
            for event in reversed(recent_guardrail_events):
                st.caption(
                    f"{event['timestamp']} | {event['metadata'].get('stage', 'unknown')} | "
                    f"{event['verdict']} | flags={len(event['flags'])}"
                )
        else:
            st.caption("No guardrail scans recorded yet.")
        if active_flags:
            st.warning("\n".join(active_flags))
        else:
            st.caption("No guardrail violations raised in this session.")

# --- MAIN CHAT UI ---
st.title("🤖 Interview Prepper")

if st.session_state.current_phase == InterviewPhase.INTERVIEW_ACTIVE:
    st.header("Interview Mode")
    st.caption("Start the interview first, then answer each question by speaking or typing.")

    if not interview_has_started():
        st.info("Press the button below to have the interviewer ask the first question.")
        if st.button("Start interview", key="start_interview_button"):
            submit_user_prompt(
                "Begin the mock software engineering interview now. "
                "Introduce the session briefly and ask the first interview question."
            )

    st.session_state.tts_enabled = st.checkbox(
        "Read assistant replies aloud",
        value=st.session_state.get("tts_enabled", False),
        help="When enabled, each assistant reply gets an audio player you can press play on.",
    )

    audio_input_available = hasattr(st, "audio_input")

    if audio_input_available:
        recorded_audio = st.audio_input("Record your answer")
    else:
        recorded_audio = st.file_uploader(
            "Upload a recorded answer",
            type=["wav", "mp3", "m4a"],
            key="fallback_audio_uploader",
            help="Your Streamlit version does not support in-browser recording, so upload an audio file instead.",
        )

    if recorded_audio is not None:
        st.audio(recorded_audio)
        if st.button("Transcribe and send audio", key="send_audio_prompt"):
            transcript = transcribe_uploaded_audio(recorded_audio)
            if transcript:
                st.info(f"Transcript: {transcript}")
                submit_user_prompt(transcript)
else:
    st.session_state.tts_enabled = False

render_message_history()
render_tts_trigger()

# ── Auto-fire preloaded context when phase is VERIFICATION ───────────────
if (
    st.session_state.current_phase == InterviewPhase.VERIFICATION
    and st.session_state.get("preloaded_context")
    and not st.session_state.get("preloaded_context_sent")
):
    ctx = st.session_state.pop("preloaded_context")
    st.session_state["preloaded_context_sent"] = True
    submit_user_prompt(ctx)

if prompt := st.chat_input("Say something..."):
    submit_user_prompt(prompt)

if st.session_state.current_phase == InterviewPhase.REPORT_READY:
    show_final_report()