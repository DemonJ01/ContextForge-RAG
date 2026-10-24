import logfire
from rag.agents.conversation_state import AgentState
from rag.gateway import portkey_client, extract_cache_status


def _no_evidence_response(state: AgentState):
    answer = "I couldn't find enough source evidence to answer this question. Please try another question or check the cloud document index."
    return {
        "final_answer": answer,
        "status": "no_answer",
        "plan": state.get("plan", []) + ["Insufficient source evidence. Technical answer withheld."],
        "messages": [{"role": "assistant", "content": answer}],
    }


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]
    raw_documents = state.get("documents")
    documents = []
    for document in raw_documents if isinstance(raw_documents, (list, tuple)) else []:
        if not isinstance(document, str):
            continue
        text = document.strip()
        # The retriever prefixes text with CONTENT:, including empty payloads.
        if text.startswith("CONTENT:"):
            text = text[len("CONTENT:"):].strip()
        if text:
            documents.append(document)
    if query != "CONVERSATIONAL" and not documents:
        logfire.info("Skipping technical synthesis: no source evidence was retrieved.")
        return _no_evidence_response(state)

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are ContextForge-RAG, a friendly and helpful assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in documents:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break
        if not full_context.strip():
            # An oversized first chunk may leave no usable context within budget.
            return _no_evidence_response(state)

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
