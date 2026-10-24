"""Review the evaluation corpus and test the running RAG API."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pandas as pd
import requests
import streamlit as st

from evaluation.rag_pipeline import run_pipeline, load_golden_dataset
from evaluation.guardrails_evaluation import run_guardrails_eval, compute_guardrails_metrics
from evaluation.telemetry import logfire
from rag.client_config import backend_url

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ContextForge-RAG Evaluation",
    page_icon="🧪",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SCORE_COLORS = {
    "green":  "#d4edda",
    "yellow": "#fff3cd",
    "red":    "#f8d7da",
}


def _badge(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    return "🔴"


def _grade(score: float) -> str:
    if score >= 0.75:
        return "✅ Good"
    elif score >= 0.5:
        return "⚠️ Fair"
    return "❌ Poor"


def _color_score(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 0.75:
        return f"background-color: {SCORE_COLORS['green']}"
    elif val >= 0.5:
        return f"background-color: {SCORE_COLORS['yellow']}"
    return f"background-color: {SCORE_COLORS['red']}"


def _render_metric_table(df: pd.DataFrame, metric_col: str, title: str):
    avg = df[metric_col].mean()
    st.markdown(f"**{title}** — AVG: {_badge(avg)} `{avg:.2f}` {_grade(avg)}")
    styled = df.style.map(_color_score, subset=[metric_col]).format({metric_col: "{:.3f}"})
    st.dataframe(styled, width="stretch", hide_index=True)


def _run_async(coro):
    return asyncio.run(coro)


def _load_metrics_runner():
    """Judge packages and credentials are only needed for the explicit scoring step."""
    if not os.getenv("JUDGE_GROQ", "").strip():
        raise ValueError("Set JUDGE_GROQ to a dedicated Groq evaluation key, then restart this dashboard.")
    try:
        from evaluation.rag_metrics import run_all_metrics
    except ImportError as exc:
        raise RuntimeError(
            "Judge metrics need optional packages. Install them with "
            "`python -m pip install -r requirements-evaluation.txt`, then restart this dashboard. "
            f"Missing dependency: {exc.name or str(exc)}."
        ) from exc
    return run_all_metrics


def _has_responses(dataset):
    return bool(dataset and any(sample.get("actual_response", "").strip() for sample in dataset["rag_samples"]))


def _guardrails_mode_notice():
    if st.session_state.get("pipeline_mode") == "local":
        st.info("The backend is running in local mode. Cloud safety rails are inactive; these results measure local API behavior.")


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "golden" not in st.session_state:
    st.session_state.golden = load_golden_dataset()
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "enriched_dataset" not in st.session_state:
    st.session_state.enriched_dataset = None
if "guardrails_results" not in st.session_state:
    st.session_state.guardrails_results = None
if "metric_results" not in st.session_state:
    st.session_state.metric_results = None
if "pipeline_rows" not in st.session_state:
    st.session_state.pipeline_rows = []

golden = st.session_state.golden

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("ContextForge-RAG Evaluation")
st.caption(
    "Step 1: Review ground truth → Step 2: Run live pipeline → Step 3: Score with RAGAS"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📋 Step 1 — Ground Truth", "🚀 Step 2 — Live Pipeline", "📊 Step 3 — Eval Metrics"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ground Truth
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ground Truth Dataset")
    st.markdown(
        "These are the **golden Q&A pairs** built by parsing your real enterprise documents. "
        "Each entry has a question, a reference answer (ground truth), and the expected tool the RAG agent should call."
    )

    rag_rows = []
    for s in golden["rag_samples"]:
        rag_rows.append({
            "ID": s["id"],
            "Domain": s["domain"].replace("_", " ").title(),
            "Question": s["question"],
            "Reference Answer": s["reference"][:120] + "..." if len(s["reference"]) > 120 else s["reference"],
            "Expected Tool": s["expected_tools"][0] if s["expected_tools"] else "—",
        })
    df_golden = pd.DataFrame(rag_rows)
    st.dataframe(df_golden, width="stretch", hide_index=True)
    st.caption(f"{len(rag_rows)} reference questions")

    st.divider()

    st.subheader("Guardrails Test Cases")
    st.markdown(
        "These inputs test whether the safety rails correctly **block adversarial inputs** "
        "and **let through legitimate questions**."
    )

    g_rows = []
    for g in golden["guardrails_samples"]:
        expected_label = "🛡️ Block" if g["expected_blocked"] else "✅ Pass"
        g_rows.append({
            "ID": g["id"],
            "Input": g["input"],
            "Expected": expected_label,
            "Type": g["type"],
            "Description": g["description"],
        })
    st.dataframe(pd.DataFrame(g_rows), width="stretch", hide_index=True)
    blocked_count = sum(bool(sample["expected_blocked"]) for sample in golden["guardrails_samples"])
    st.caption(f"{len(g_rows)} guardrail cases: {blocked_count} expected to block, {len(g_rows) - blocked_count} expected to pass")

    with st.expander("View raw rag_golden_dataset.json"):
        st.json(golden)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Pipeline
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Live Pipeline — Collect Real Responses")
    st.markdown(
        f"Sends each reference question to your **running FastAPI app** (`{backend_url()}/query`). "
        "Captures the actual response, retrieved contexts, and tool called. "
        "Responses are truncated to 300 chars to save tokens for the RAGAS judging step."
    )
    st.info(
        "Start the backend first: `python run_rag.py`. Set `BACKEND_URL` to use a different API.",
        icon="⚠️",
    )

    col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
    run_pipeline_btn = col_p1.button(
        "▶️ Run Live Pipeline",
        type="primary",
        width="stretch",
        disabled=st.session_state.pipeline_done,
    )
    reset_btn = col_p2.button(
        "🔄 Reset & Re-run",
        width="stretch",
        disabled=not st.session_state.pipeline_done,
    )

    if reset_btn:
        st.session_state.pipeline_done = False
        st.session_state.enriched_dataset = None
        st.session_state.guardrails_results = None
        st.session_state.metric_results = None
        st.session_state.pipeline_rows = []
        st.session_state.pipeline_mode = None
        st.rerun()

    if run_pipeline_btn:
        try:
            health_response = requests.get(backend_url() + "/health", timeout=5)
            health_response.raise_for_status()
            backend_mode = health_response.json().get("mode", "cloud")
            st.session_state.pipeline_mode = backend_mode
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Cannot reach a healthy backend at {backend_url()}. Start `python run_rag.py` and retry. Details: {exc}")
            st.stop()
        delay_seconds = 0 if backend_mode == "local" else 10
        st.session_state.pipeline_rows = []
        progress_bar = st.progress(0, text="Starting pipeline...")
        live_table_slot = st.empty()
        status_slot = st.empty()

        def pipeline_cb(i, total, question, stage, response=""):
            pct = int((i / total) * 100)
            if stage == "calling":
                progress_bar.progress(pct, text=f"[{i+1}/{total}] Calling /query: {question[:60]}...")
            else:
                short_q = question[:55] + "..." if len(question) > 55 else question
                short_r = response[:80] + "..." if len(response) > 80 else response
                st.session_state.pipeline_rows.append({
                    "#": i + 1,
                    "Question": short_q,
                    "Live Response (truncated)": short_r if short_r else "⚠️ No response",
                    "Status": "✅" if short_r else "❌",
                })
                live_table_slot.dataframe(
                    pd.DataFrame(st.session_state.pipeline_rows),
                    width="stretch",
                    hide_index=True,
                )
                progress_bar.progress(
                    int(((i + 1) / total) * 100),
                    text=f"[{i+1}/{total}] ✅ Done",
                )

        with logfire.span("🚀 Streamlit — Run Pipeline Button"):
            enriched = run_pipeline(golden, progress_callback=pipeline_cb, delay_seconds=delay_seconds)
            st.session_state.enriched_dataset = enriched

        successful = sum(bool(sample.get("actual_response")) for sample in enriched["rag_samples"])
        progress_bar.progress(100, text="Pipeline requests finished")
        if successful == len(enriched["rag_samples"]):
            status_slot.success(f"{successful} responses collected and stored in session.")
        else:
            status_slot.warning(f"{successful}/{len(enriched['rag_samples'])} responses collected. Failed requests are excluded from judge metrics.")
            st.dataframe(pd.DataFrame([
                {"Question": sample["question"], "Error": sample.get("request_error", "No response")}
                for sample in enriched["rag_samples"] if not sample.get("actual_response")
            ]), hide_index=True, width="stretch")

        # ── Guardrails tests ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Guardrails Tests")
        _guardrails_mode_notice()
        g_progress = st.progress(0, text="Running guardrails tests...")
        g_status_slot = st.empty()

        def g_cb(i, total, input_text):
            g_progress.progress(
                int((i / total) * 100),
                text=f"[{i+1}/{total}] Testing: {input_text[:60]}...",
            )

        with logfire.span("🛡️ Streamlit — Guardrails Tests"):
            g_results = run_guardrails_eval(
                enriched["guardrails_samples"], progress_callback=g_cb,
                delay_seconds=0 if backend_mode == "local" else 2,
            )
            g_metrics = compute_guardrails_metrics(g_results)
            st.session_state.guardrails_results = g_results
            st.session_state.pipeline_done = successful > 0

        g_progress.progress(100, text="Guardrail requests finished")
        if g_metrics["errors"]:
            st.warning(f"{g_metrics['errors']} guardrail requests failed. Metrics cover {g_metrics['completed']} completed requests.")

        g_rows_live = []
        for r in g_results:
            result_label = {
                "TP": "🛡️ Blocked ✅", "TN": "✅ Passed ✅",
                "FP": "🛡️ Blocked ❌ (False Positive)", "FN": "✅ Passed ❌ (Missed)",
            }.get(r["result"], r["result"])
            g_rows_live.append({
                "ID": r["id"],
                "Input": r["input"][:70],
                "Expected": "🛡️ Block" if r["expected_blocked"] else "✅ Pass",
                "Actual": "Request failed" if r["actual_blocked"] is None else "Blocked" if r["actual_blocked"] else "Passed",
                "Result": result_label,
            })
        st.dataframe(pd.DataFrame(g_rows_live), width="stretch", hide_index=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Correct", f"{g_metrics['correct']}/{g_metrics['completed']}")
        mc2.metric("Precision", f"{g_metrics['precision']:.2f}")
        mc3.metric("Recall", f"{g_metrics['recall']:.2f}")
        mc4.metric("Accuracy", f"{g_metrics['accuracy']:.2f}")
        if st.session_state.pipeline_done:
            st.rerun()

    elif st.session_state.pipeline_done:
        saved_samples = st.session_state.enriched_dataset["rag_samples"]
        successful = sum(bool(sample.get("actual_response")) for sample in saved_samples)
        if successful == len(saved_samples):
            st.success(f"{successful} responses collected and stored in session. Reset to run again.")
        else:
            st.warning(f"{successful}/{len(saved_samples)} responses collected. Failed requests are excluded from judge metrics. Reset to retry.")

        resp_rows = []
        for s in st.session_state.enriched_dataset["rag_samples"]:
            resp_rows.append({
                "#": s["id"],
                "Domain": s["domain"].replace("_", " ").title(),
                "Question": s["question"][:60],
                "Live Response": s["actual_response"][:100] + "..." if len(s.get("actual_response","")) > 100 else s.get("actual_response",""),
                "Tool Called": s["actual_tools_called"][0] if s.get("actual_tools_called") else "—",
                "Contexts Retrieved": len(s.get("actual_contexts", [])),
                "Request Error": s.get("request_error", ""),
            })
        st.dataframe(pd.DataFrame(resp_rows), width="stretch", hide_index=True)

        if st.session_state.guardrails_results:
            st.divider()
            st.subheader("Guardrails Results (from previous run)")
            _guardrails_mode_notice()
            g_rows_prev = []
            for r in st.session_state.guardrails_results:
                result_label = {
                    "TP": "🛡️ Blocked ✅", "TN": "✅ Passed ✅",
                    "FP": "Blocked ❌ FP", "FN": "Passed ❌ FN",
                }.get(r["result"], r["result"])
                g_rows_prev.append({
                    "ID": r["id"],
                    "Input": r["input"][:70],
                    "Result": result_label,
                    "Request Error": r.get("request_error", ""),
                })
            st.dataframe(pd.DataFrame(g_rows_prev), width="stretch", hide_index=True)
            gm = compute_guardrails_metrics(st.session_state.guardrails_results)
            if gm["errors"]:
                st.warning(f"{gm['errors']} guardrail requests failed. Metrics cover {gm['completed']} completed requests.")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Correct", f"{gm['correct']}/{gm['completed']}")
            mc2.metric("Precision", f"{gm['precision']:.2f}")
            mc3.metric("Recall", f"{gm['recall']:.2f}")
            mc4.metric("Accuracy", f"{gm['accuracy']:.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eval Metrics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Eval Metrics — RAGAS + Tool Correctness")

    if not _has_responses(st.session_state.enriched_dataset):
        st.warning("⚠️ Complete Step 2 (Live Pipeline) first to collect responses.")
    else:
        st.markdown(
            "Runs all **6 metric experiments** on the stored responses. "
            "LLM-based metrics call Groq using a dedicated `JUDGE_GROQ` key and may incur provider charges. "
            "They also download a local embedding model on first use. "
            "Samples run one at a time with 40s cooldowns, so a full run can take about 50 minutes."
        )
        st.info(
            "Token key used: `JUDGE_GROQ` (separate from production key). "
            "Install `requirements-evaluation.txt` and set this key before scoring. "
            "Provider rate limits depend on your account; the cooldowns do not guarantee quota availability.",
            icon="ℹ️",
        )

        run_metrics_btn = st.button(
            "▶️ Run Eval Metrics",
            type="primary",
            disabled=not st.session_state.pipeline_done,
        )

        if run_metrics_btn:
            try:
                run_all_metrics = _load_metrics_runner()
            except (ValueError, RuntimeError) as exc:
                st.warning(str(exc))
                st.stop()
            status_slot = st.empty()
            results_slots = {}

            metric_display_names = {
                "faithfulness":      "Exp 1 — Faithfulness",
                "answer_relevancy":  "Exp 2 — Answer Relevancy",
                "context_precision": "Exp 3 — Context Precision",
                "context_recall":    "Exp 4 — Context Recall",
                "answer_correctness":"Exp 5 — Answer Correctness",
                "tool_correctness":  "Exp 6 — Tool Correctness",
            }
            for key, title in metric_display_names.items():
                results_slots[key] = st.empty()

            def status_cb(msg: str):
                status_slot.info(msg)

            try:
                with logfire.span("📊 Streamlit — Run Metrics Button"):
                    metric_results = _run_async(
                        run_all_metrics(st.session_state.enriched_dataset, status_cb=status_cb)
                    )
                    st.session_state.metric_results = metric_results
            except Exception as exc:
                st.error(f"Judge scoring failed: {exc}. Check the optional packages, model download access, and JUDGE_GROQ quota, then retry.")
                st.stop()

            status_slot.success("✅ All 6 experiments complete!")

            for key, title in metric_display_names.items():
                if key in metric_results:
                    with results_slots[key].container():
                        _render_metric_table(metric_results[key], key, title)

        elif st.session_state.metric_results:
            st.success("✅ Metrics already computed. Showing results below.")
            metric_display_names = {
                "faithfulness":      "Exp 1 — Faithfulness",
                "answer_relevancy":  "Exp 2 — Answer Relevancy",
                "context_precision": "Exp 3 — Context Precision",
                "context_recall":    "Exp 4 — Context Recall",
                "answer_correctness":"Exp 5 — Answer Correctness",
                "tool_correctness":  "Exp 6 — Tool Correctness",
            }
            for key, title in metric_display_names.items():
                if key in st.session_state.metric_results:
                    _render_metric_table(st.session_state.metric_results[key], key, title)

        # ── Final Summary ─────────────────────────────────────────────────────
        if st.session_state.metric_results:
            st.divider()
            st.subheader("Final Summary")

            mr = st.session_state.metric_results
            summary = [
                ("Faithfulness",       mr.get("faithfulness",      pd.DataFrame()).get("faithfulness",      pd.Series()).mean()),
                ("Answer Relevancy",   mr.get("answer_relevancy",  pd.DataFrame()).get("answer_relevancy",  pd.Series()).mean()),
                ("Context Precision",  mr.get("context_precision", pd.DataFrame()).get("context_precision", pd.Series()).mean()),
                ("Context Recall",     mr.get("context_recall",    pd.DataFrame()).get("context_recall",    pd.Series()).mean()),
                ("Answer Correctness", mr.get("answer_correctness",pd.DataFrame()).get("answer_correctness",pd.Series()).mean()),
                ("Tool Correctness",   mr.get("tool_correctness",  pd.DataFrame()).get("tool_correctness",  pd.Series()).mean()),
            ]

            cols = st.columns(len(summary))
            for col, (name, score) in zip(cols, summary):
                if pd.notna(score):
                    col.metric(
                        label=name,
                        value=f"{score:.2f}",
                        delta=_grade(score),
                    )

            if st.session_state.guardrails_results:
                gm = compute_guardrails_metrics(st.session_state.guardrails_results)
                st.metric(
                    label="🛡️ Guardrails Accuracy",
                    value=f"{gm['correct']}/{gm['completed']}",
                    delta=f"Precision {gm['precision']:.2f} | Recall {gm['recall']:.2f} | Errors {gm['errors']}",
                )

            summary_df = pd.DataFrame([
                {"Metric": name, "Score": f"{score:.3f}" if pd.notna(score) else "—", "Grade": _grade(score) if pd.notna(score) else "—"}
                for name, score in summary
            ])
            st.dataframe(summary_df, width="stretch", hide_index=True)
