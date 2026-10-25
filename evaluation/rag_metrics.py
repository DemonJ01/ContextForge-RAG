"""
Phase 2 — RAGAS + Tool Correctness metrics.
Uses JUDGE_GROQ when configured, otherwise the main GROQ_API_KEY.
LLM-based metrics run one sample at a time with cooldowns between requests.
Complete answers and retrieved evidence are preserved for meaningful judging;
provider quotas and response lengths determine token cost and runtime.
"""


import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
import logfire
import pandas as pd
from openai import AsyncOpenAI


from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "llama-3.1-8b-instant"
COOLDOWN_STANDARD = 62
COOLDOWN_MINI = 40       # pacing between individual samples; actual provider quotas still apply
GENERAL_BATCH_SIZE = 1  # one sample at a time: abatch_score fires calls concurrently per sample,
                         # so batch>1 stacks multiple samples' async calls inside the same second


def _build_judge():
    api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    llm = llm_factory(JUDGE_MODEL, provider="openai", client=client)
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )
    return llm, embeddings

async def _cooldown(seconds: int, label: str, status_cb=None):
    msg = f"⏳ {seconds}s cooldown after {label} (Groq TPM buffer)..."
    if status_cb:
        status_cb(msg)
    remaining = seconds
    while remaining > 0:
        interval = min(10, remaining)
        await asyncio.sleep(interval)
        remaining -= interval
    if status_cb:
        status_cb("✅ Ready — continuing evaluation.")
        
        
def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns answered samples from successful requests without shortening evidence.
    Failed or pending requests are reported by capture coverage, never silently
    included in the averages as valid answers.
    """
    valid = []
    for s in golden_dataset["rag_samples"]:
        if s.get("pipeline_status") in {"error", "pending"}:
            continue
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        # An empty live retrieval is a result, never permission to substitute
        # answer-key evidence and inflate grounding/recall metrics.
        raw_contexts = s.get("actual_contexts", s.get("relevant_contexts", [])) or []
        valid.append({**s, "actual_contexts": list(raw_contexts)})
    return valid


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    if len(samples) != len(scores):
        raise ValueError(f"{metric_key}: expected {len(samples)} scores, received {len(scores)}.")
    return pd.DataFrame([
        {"question": s["question"][:65], metric_key: round(float(r.value), 3)}
        for s, r in zip(samples, scores)
    ])


def run_metrics_sync(golden_dataset: dict, status_cb=None) -> dict:
    """Run judging in an isolated loop; deliver queued progress on the caller thread.

    Streamlit can reuse or already be running an event loop. The worker owns its
    loop, while all UI callbacks stay on the Streamlit script thread.
    """
    updates = Queue()

    def worker():
        return asyncio.run(run_all_metrics(golden_dataset, status_cb=updates.put))

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-eval") as executor:
        future = executor.submit(worker)
        while not future.done():
            try:
                message = updates.get(timeout=0.1)
            except Empty:
                continue
            if status_cb:
                status_cb(message)
        while not updates.empty():
            message = updates.get_nowait()
            if status_cb:
                status_cb(message)
        return future.result()


async def _batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "") -> list:
    """
    Runs abatch_score in chunks of GENERAL_BATCH_SIZE with cooldowns between chunks.
    Reduces concurrent load; provider limits still depend on complete input sizes.
    """
    all_scores = []
    batches = [inputs[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(inputs), GENERAL_BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            await _cooldown(COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)
        scores = await metric.abatch_score(batch)
        all_scores.extend(scores)
    return all_scores

async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """
    Runs all 6 experiments. Returns dict keyed by metric name → DataFrame.
    status_cb(message: str) is called for live UI updates.
    """
    samples = _prep_samples(golden_dataset)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    if status_cb:
        status_cb(f"Scoring {len(samples)}/{len(golden_dataset['rag_samples'])} captured answers; failed or pending requests are excluded.")
    judge_llm, ragas_embeddings = _build_judge()
    results = {}

    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=len(samples)):

        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 1/6 — Faithfulness ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 1 — Faithfulness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(Faithfulness(llm=judge_llm), inputs, samples, status_cb, "Faithfulness")
            df = _score_df("faithfulness", samples, scores)
            results["faithfulness"] = df
            logfire.info("🧪 Faithfulness done", avg=round(df["faithfulness"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Faithfulness", status_cb)

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 2/6 — Answer Relevancy ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 2 — Answer Relevancy"):
            inputs = [
                {"user_input": s["question"], "response": s["actual_response"]}
                for s in samples
            ]
            scores = await _batched_score(
                AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Relevancy"
            )
            df = _score_df("answer_relevancy", samples, scores)
            results["answer_relevancy"] = df
            logfire.info("🧪 Answer Relevancy done", avg=round(df["answer_relevancy"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Answer Relevancy", status_cb)

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 3/6 — Context Precision ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 3 — Context Precision"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(ContextPrecision(llm=judge_llm), inputs, samples, status_cb, "Context Precision")
            df = _score_df("context_precision", samples, scores)
            results["context_precision"] = df
            logfire.info("🧪 Context Precision done", avg=round(df["context_precision"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Precision", status_cb)

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 4/6 — Context Recall ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 4 — Context Recall"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(ContextRecall(llm=judge_llm), inputs, samples, status_cb, "Context Recall")
            df = _score_df("context_recall", samples, scores)
            results["context_recall"] = df
            logfire.info("🧪 Context Recall done", avg=round(df["context_recall"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Recall", status_cb)

        # ── Exp 5: Answer Correctness ────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 5/6 — Answer Correctness ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 5 — Answer Correctness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "reference": s["reference"],
                }
                for s in samples
            ]
            all_scores = await _batched_score(
                AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Correctness"
            )
            df = _score_df("answer_correctness", samples, all_scores)
            results["answer_correctness"] = df
            logfire.info("🧪 Answer Correctness done", avg=round(df["answer_correctness"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Answer Correctness", status_cb)

        # ── Exp 6: Tool Correctness (no LLM — Jaccard) ───────────────────────
        if status_cb:
            status_cb("⚡ Exp 6/6 — Tool Correctness (zero LLM calls)...")
        with logfire.span("🧪 Exp 6 — Tool Correctness"):
            tool_rows = []
            for s in samples:
                called = set(s.get("actual_tools_called") or [])
                expected = set(s.get("expected_tools") or [])
                union = len(called | expected)
                score = len(called & expected) / union if union > 0 else 0.0
                tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
            df = pd.DataFrame(tool_rows)
            results["tool_correctness"] = df
            logfire.info("🧪 Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb("✅ All 6 experiments complete!")

    return results
