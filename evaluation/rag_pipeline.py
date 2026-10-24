"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each golden sample.
Captures: actual_response (truncated to 300 chars), actual_contexts (from sources),
and actual_tools_called (detected from thought_process).
"""


import time
import copy
import json
import os
import requests
from uuid import uuid4

from evaluation.telemetry import logfire
from rag.client_config import backend_url

API_URL = backend_url() + "/query"
RESPONSE_TRUNCATE = 300
DELAY_BETWEEN_CALLS = 10   # seconds — stays within Groq RPM on the main key
REQUEST_TIMEOUT = 120      # seconds — guardrails + LangGraph + Groq can take >60s




def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from /query response to a tool name.
    Planner sets:  'Intent: Technical' + 'Search Term: ...' → retrieve_documents
                   'Intent: Conversational/Memory'           → direct_answer
    rag/api.py sets: 'Intent: Guardrails Fired'              → guardrails
    """
    joined = " ".join(thought_process).lower()
    if "guardrails fired" in joined:
        return "guardrails"
    if "intent: technical" in joined or "search term:" in joined or "context retrieved" in joined or "bm25 retrieval" in joined:
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
    return "unknown"


def run_pipeline(golden_dataset: dict, progress_callback=None, *, delay_seconds=DELAY_BETWEEN_CALLS) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with actual_response, actual_contexts, actual_tools_called filled.
    progress_callback(i, total, question, stage, response="") is called per step.
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)
    run_id = uuid4().hex

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": question, "thread_id": f"eval_{run_id}_{i}"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []

                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = sources[:5]
                    sample["actual_tools_called"] = [detect_tool(thought_process)]
                    sample["request_error"] = "" if raw_answer else "The backend returned an empty answer."

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chunks=len(sources),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = []
                    sample["actual_tools_called"] = ["unknown"]
                    sample["request_error"] = f"Cannot reach {API_URL}. Start the backend or check BACKEND_URL."

                except Exception as e:
                    logfire.error(f"❌ Query failed: {e}")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = []
                    sample["actual_tools_called"] = ["unknown"]
                    sample["request_error"] = str(e)

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(delay_seconds)

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
        
        
def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "rag_golden_dataset.json")
    with open(golden_path) as f:
        return json.load(f)
