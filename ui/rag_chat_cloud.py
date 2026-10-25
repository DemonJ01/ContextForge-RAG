import os
import sys
import streamlit as st
import requests
import time
import uuid
import logfire

# Streamlit's console entrypoint adds ui/, so include its sibling rag package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from rag.client_config import QueryResponseError, query_response


# Initialize Logfire
try:
    logfire.configure(token=st.secrets.get("LOGFIRE_TOKEN", os.getenv("LOGFIRE_TOKEN")))
    logfire.instrument_requests()   # propagates trace context to the FastAPI backend
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception:
    LOGFIRE_STATUS = "Standby (No Token)"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"

# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 RAG")
    st.markdown("---")

    try:
        configured_backend = st.secrets.get("BACKEND_URL")
    except Exception:
        configured_backend = None
    base_url = (configured_backend or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000").strip().rstrip("/")

    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")
    
    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        logfire.warning(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- MAIN CHAT ---
st.title("🤖 RAG Assistant")

# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            data = {}
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    with logfire.span("📡 Calling RAG Backend"):
                        url = f"{base_url}/query"
                        payload = {"q": prompt, "thread_id": st.session_state.session_id}
                        response = requests.post(url, json=payload, timeout=60)

                        data = query_response(response)

                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.markdown(f"⚙️ {step}", unsafe_allow_html=False)

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                except requests.Timeout:
                    status.update(label="❌ Search timed out", state="error")
                    st.error("The search timed out. Please try again; cloud models may take longer on their first request.")
                    st.stop()
                except requests.ConnectionError:
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Cannot reach the backend. Check that the RAG server is running and BACKEND_URL is correct.")
                    st.stop()
                except QueryResponseError as e:
                    status.update(label="❌ Search could not be completed", state="error")
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Search could not be completed", state="error")
                    st.error("The search could not be displayed. Please try again or check the server logs.")
                    st.stop()

            # Answer streaming — outside status so it's always visible
            answer_placeholder = st.empty()
            full_answer = data.get("answer") or "No response."

            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)
            answer_placeholder.markdown(full_answer)

            # Sources — outside status so they're visible after it collapses
            sources = data.get("sources", [])
            if sources:
                with st.expander(f"📄 Retrieved Context ({len(sources)} chunks)"):
                    for i, source in enumerate(sources):
                        st.caption(f"Chunk {i + 1}")
                        st.info(source)
            elif data.get("status") == "no_answer":
                st.caption("ℹ️ No supporting evidence was found in the indexed documents.")
            else:
                st.caption("ℹ️ This response has no retrieved source passages.")

            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")
