"""Streamlit chat interface for the ContextForge-RAG HTTP API."""
from uuid import uuid4

import requests
import streamlit as st

from rag.client_config import backend_url


def configured_backend_url(use_secrets=False):
    fallback = backend_url()
    if use_secrets:
        try:
            configured = st.secrets.get("BACKEND_URL")
        except Exception:
            # A local Streamlit launch normally has no secrets.toml.
            configured = None
        if isinstance(configured, str) and configured.strip():
            return configured.strip().rstrip("/")
    return fallback


def query_backend(base_url, payload):
    response = requests.post(f"{base_url}/query", json=payload, timeout=(5, 60))
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str) or not data["answer"].strip():
        raise ValueError("The backend response is missing an answer.")
    return data


def error_message(error):
    if isinstance(error, requests.Timeout):
        return "The backend took too long to respond. Try again."
    if isinstance(error, requests.ConnectionError):
        return "Could not connect to the ContextForge-RAG backend. Check BACKEND_URL and make sure the API is running."
    if isinstance(error, requests.HTTPError):
        response = error.response
        code = response.status_code if response is not None else "error"
        return f"The backend could not complete the request (HTTP {code}). Check the backend and try again."
    return "The backend returned an invalid response. Try again."


def render_message(message):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return
        if message["role"] == "assistant" and message.get("mode") == "local":
            st.text(message["content"], width="stretch")
        else:
            st.markdown(message["content"])
        if message["role"] != "assistant":
            return

        citations = message.get("citations", [])
        sources = message.get("sources", [])
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                for number, citation in enumerate(citations, 1):
                    label = citation.get("filename") or citation.get("title") or "Document"
                    st.caption(f"[{number}] {label}")
                    st.text(citation["excerpt"], width="stretch")
        elif sources:
            with st.expander(f"Sources ({len(sources)})"):
                for number, source in enumerate(sources, 1):
                    st.caption(f"[{number}]")
                    st.text(source, width="stretch")
        else:
            st.caption("No supporting sources were returned.")

        if message.get("mode") == "local":
            st.caption("Local mode · answers contain excerpts from the indexed documents.")
        elif message.get("mode") == "cloud":
            st.caption("Cloud mode")


def answer_message(data):
    citations = data.get("citations")
    sources = data.get("sources")
    return {
        "role": "assistant",
        "content": data["answer"],
        "mode": data.get("mode"),
        "status": data.get("status"),
        "citations": [citation for citation in citations
                      if isinstance(citation, dict) and isinstance(citation.get("excerpt"), str)]
        if isinstance(citations, list) else [],
        "sources": [source for source in sources if isinstance(source, str)]
        if isinstance(sources, list) else [],
    }


def render_app(use_secrets=False):
    st.set_page_config(page_title="ContextForge-RAG", page_icon="📚", layout="wide")
    base_url = configured_backend_url(use_secrets)
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("ContextForge-RAG")
        scope = st.selectbox("Document scope", ["reference", "all"],
                             format_func=lambda value: "Reference documents" if value == "reference" else "All documents",
                             key="document_scope")
        top_k = st.slider("Maximum sources", min_value=1, max_value=8, value=4, key="maximum_sources")
        st.caption(f"Backend: {base_url}")
        if st.button("New conversation", type="primary"):
            st.session_state.messages = []
            st.session_state.session_id = uuid4().hex
            st.rerun()

    st.title("ContextForge-RAG")
    st.caption("Ask questions about your indexed documents and inspect the supporting sources.")
    retry_index = None
    for index, message in enumerate(st.session_state.messages):
        render_message(message)
        if message.get("error") and index == len(st.session_state.messages) - 1:
            if st.button("Retry", key=f"retry_{index}"):
                retry_index = index

    prompt = st.chat_input("Ask about your documentation…", max_chars=2000)
    payload = None
    if retry_index is not None:
        payload = st.session_state.messages[retry_index]["request"]
    elif prompt:
        prompt = prompt.strip()
        if not prompt:
            st.warning("Enter a question before sending.")
            return
        payload = {"q": prompt, "thread_id": st.session_state.session_id, "scope": scope, "top_k": top_k}
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    if payload is not None:
        with st.spinner("Searching documents…"):
            try:
                reply = answer_message(query_backend(base_url, payload))
            except (requests.RequestException, ValueError, TypeError) as error:
                reply = {"role": "assistant", "content": error_message(error), "error": True, "request": payload}
        if retry_index is not None:
            st.session_state.messages[retry_index] = reply
        else:
            st.session_state.messages.append(reply)
        st.rerun()
