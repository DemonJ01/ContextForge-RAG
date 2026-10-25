"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each golden sample.
Captures full responses and retrieved contexts, plus tool routing and request
outcomes. Previews can be shortened by the UI without changing judged evidence.
"""


import time
import copy
import json
import os
import requests
import logfire
from uuid import uuid4

from rag.client_config import backend_url

API_URL = backend_url() + "/query"
DELAY_BETWEEN_CALLS = 10   # seconds — request pacing; provider quotas still apply
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


def summarize_pipeline(dataset: dict) -> dict:
    """Report request coverage separately from answer-quality metrics."""
    samples = dataset.get("rag_samples", [])
    completed = sum(sample.get("pipeline_status") == "completed" for sample in samples)
    failed = sum(sample.get("pipeline_status") == "error" for sample in samples)
    total = len(samples)
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "pending": total - completed - failed,
        "coverage": round(completed / total, 3) if total else 0.0,
    }


def run_pipeline(golden_dataset: dict, progress_callback=None, *,
                 retry_failed_only=False, checkpoint_callback=None) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with actual_response, actual_contexts, actual_tools_called filled.
    progress_callback(i, total, question, stage, response="") is called per step.
    Failed requests have pipeline_status="error" and an actionable pipeline_error.
    A checkpoint callback receives an independent dataset snapshot after each
    request; retry_failed_only keeps successfully captured responses intact.
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)
    run_id = uuid4().hex
    pending = []
    for i, sample in enumerate(samples):
        if (retry_failed_only and sample.get("pipeline_status") == "completed"
                and isinstance(sample.get("actual_response"), str)
                and sample["actual_response"].strip()):
            continue
        sample.update(actual_response="", actual_contexts=[], actual_tools_called=["unknown"],
                      pipeline_status="pending", pipeline_error=None)
        pending.append(i)
    dataset["pipeline_summary"] = summarize_pipeline(dataset)

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for position, i in enumerate(pending):
            sample = samples[i]
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
                    if not isinstance(data, dict):
                        raise ValueError("The API response must be an object.")
                    if data.get("status") == "error":
                        raise ValueError("The backend reported an unsuccessful query.")

                    raw_answer = data.get("answer")
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []
                    if not isinstance(raw_answer, str) or not raw_answer.strip():
                        raise ValueError("The API returned no answer text.")
                    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
                        raise ValueError("The API returned invalid source excerpts.")
                    if not isinstance(thought_process, list) or not all(isinstance(s, str) for s in thought_process):
                        raise ValueError("The API returned invalid routing information.")

                    sample["actual_response"] = raw_answer
                    sample["actual_contexts"] = sources
                    sample["actual_tools_called"] = [detect_tool(thought_process)]
                    sample["pipeline_status"] = "completed"
                    sample["pipeline_error"] = None

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chunks=len(sources),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    sample["pipeline_status"] = "error"
                    sample["pipeline_error"] = "Cannot reach the backend. Check BACKEND_URL and that the API is running."

                except Exception as e:
                    logfire.error(f"❌ Query failed: {e}")
                    sample["pipeline_status"] = "error"
                    status_code = getattr(getattr(e, "response", None), "status_code", None)
                    sample["pipeline_error"] = (
                        f"Backend returned HTTP {status_code}. Check server configuration and logs."
                        if status_code is not None else
                        f"Query failed ({type(e).__name__}). Check the API response and server logs."
                    )

            dataset["pipeline_summary"] = summarize_pipeline(dataset)
            if checkpoint_callback:
                checkpoint_callback(copy.deepcopy(dataset))
            if progress_callback:
                stage = "done" if sample["pipeline_status"] == "completed" else "error"
                progress_callback(i, n, question, stage, sample["actual_response"])

            if position < len(pending) - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
        
        
def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "rag_golden_dataset.json")
    with open(golden_path) as f:
        return json.load(f)
