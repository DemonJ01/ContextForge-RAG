# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: logfire must be configured before all other imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="rag-evaluation")

# ─────────────────────────────────────────────────────────────────────────────
import json
import pandas as pd
import streamlit as st

from evaluation.rag_pipeline import run_pipeline, load_golden_dataset, summarize_pipeline
from evaluation.guardrails_evaluation import run_guardrails_eval, compute_guardrails_metrics
from evaluation.rag_metrics import run_metrics_sync

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG — Eval Suite",
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
    styler = df.style
    # pandas 3 removed applymap; older versions supported by Streamlit lack map.
    colorize = getattr(styler, "map", None)
    if colorize is None:
        colorize = styler.applymap
    styled = colorize(_color_score, subset=[metric_col]).format({metric_col: "{:.3f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_pipeline_coverage(dataset):
    coverage = summarize_pipeline(dataset)
    message = (
        f"Captured {coverage['completed']}/{coverage['total']} responses "
        f"({coverage['coverage']:.0%} coverage); {coverage['failed']} failed, "
        f"{coverage['pending']} pending."
    )
    if coverage["failed"] or coverage["pending"] or not coverage["total"]:
        st.warning(message + " Scores cover captured answers only; retry incomplete queries to restore coverage.")
    else:
        st.success(message)
    return coverage


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
st.title("🧪 RAG — Evaluation Suite")
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
    st.dataframe(df_golden, use_container_width=True, hide_index=True)
    st.caption(f"✅ {len(rag_rows)} golden RAG samples from 5 enterprise docs")

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
    st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)
    st.caption("6 guardrails test cases: 3 adversarial (should block) + 3 legit (should pass)")

    with st.expander("View raw rag_golden_dataset.json"):
        st.json(golden)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Pipeline
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Live Pipeline — Collect Real Responses")
    st.markdown(
        "Sends each golden question to your **running FastAPI app** (`localhost:8000/query`). "
        "Captures the actual response, retrieved contexts, and tool called. "
        "Full answers and evidence are retained for judging; tables show shortened previews."
    )
    st.info(
        "⚠️ Make sure your FastAPI backend is running first: `uvicorn rag.api:app --reload --port 8000`",
        icon="⚠️",
    )

    captured = st.session_state.enriched_dataset
    incomplete = summarize_pipeline(captured) if captured else None
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
        disabled=captured is None,
    )
    retry_pipeline_btn = col_p3.button(
        "↻ Retry incomplete RAG queries",
        width="stretch",
        disabled=not incomplete or not (incomplete["failed"] or incomplete["pending"]),
    )

    if reset_btn:
        st.session_state.pipeline_done = False
        st.session_state.enriched_dataset = None
        st.session_state.guardrails_results = None
        st.session_state.metric_results = None
        st.session_state.pipeline_rows = []
        st.rerun()

    if run_pipeline_btn or retry_pipeline_btn:
        st.session_state.metric_results = None
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
                    "Live Response (preview)": short_r if short_r else "⚠️ No response",
                    "Status": "✅ Captured" if stage == "done" else "❌ Request failed",
                })
                live_table_slot.dataframe(
                    pd.DataFrame(st.session_state.pipeline_rows),
                    use_container_width=True,
                    hide_index=True,
                )
                progress_bar.progress(
                    int(((i + 1) / total) * 100),
                    text=f"[{i+1}/{total}] {'Captured' if stage == 'done' else 'Request failed'}",
                )

        def checkpoint(dataset):
            # Keep successful captures available when a rerun interrupts collection.
            st.session_state.enriched_dataset = dataset

        with logfire.span("🚀 Streamlit — Run Pipeline Button"):
            enriched = run_pipeline(
                captured if retry_pipeline_btn else golden,
                progress_callback=pipeline_cb,
                retry_failed_only=retry_pipeline_btn,
                checkpoint_callback=checkpoint,
            )
            st.session_state.enriched_dataset = enriched

        progress_bar.progress(100, text="Response collection finished.")
        coverage = _render_pipeline_coverage(enriched)
        status_slot.info(f"{coverage['completed']} captured responses are available for scoring.")
        failed_rows = [
            {"Question": sample["question"], "Error": sample["pipeline_error"]}
            for sample in enriched["rag_samples"] if sample.get("pipeline_status") == "error"
        ]
        if failed_rows:
            st.dataframe(pd.DataFrame(failed_rows), use_container_width=True, hide_index=True)

        # ── Guardrails tests ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Guardrails Tests")
        g_progress = st.progress(0, text="Running guardrails tests...")
        g_status_slot = st.empty()

        def g_cb(i, total, input_text):
            g_progress.progress(
                int((i / total) * 100),
                text=f"[{i+1}/{total}] Testing: {input_text[:60]}...",
            )

        with logfire.span("🛡️ Streamlit — Guardrails Tests"):
            if retry_pipeline_btn and st.session_state.guardrails_results is not None:
                g_results = st.session_state.guardrails_results
            else:
                g_results = run_guardrails_eval(enriched["guardrails_samples"], progress_callback=g_cb)
            g_metrics = compute_guardrails_metrics(g_results)
            st.session_state.guardrails_results = g_results
            st.session_state.pipeline_done = True

        g_progress.progress(100, text="✅ Guardrails tests complete!")
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
        st.dataframe(pd.DataFrame(g_rows_live), use_container_width=True, hide_index=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Correct", f"{g_metrics['correct']}/{g_metrics['completed']}")
        mc2.metric("Precision", f"{g_metrics['precision']:.2f}")
        mc3.metric("Recall", f"{g_metrics['recall']:.2f}")
        mc4.metric("Accuracy", f"{g_metrics['accuracy']:.2f}")
        st.rerun()

    elif st.session_state.enriched_dataset is not None:
        _render_pipeline_coverage(st.session_state.enriched_dataset)

        resp_rows = []
        for s in st.session_state.enriched_dataset["rag_samples"]:
            resp_rows.append({
                "#": s["id"],
                "Domain": s["domain"].replace("_", " ").title(),
                "Question": s["question"][:60],
                "Live Response": s["actual_response"][:100] + "..." if len(s.get("actual_response","")) > 100 else s.get("actual_response",""),
                "Tool Called": s["actual_tools_called"][0] if s.get("actual_tools_called") else "—",
                "Contexts Retrieved": len(s.get("actual_contexts", [])),
                "Request Status": s.get("pipeline_status", "pending"),
                "Error": s.get("pipeline_error") or "",
            })
        st.dataframe(pd.DataFrame(resp_rows), use_container_width=True, hide_index=True)

        if st.session_state.guardrails_results:
            st.divider()
            st.subheader("Guardrails Results (from previous run)")
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
                })
            st.dataframe(pd.DataFrame(g_rows_prev), use_container_width=True, hide_index=True)
            gm = compute_guardrails_metrics(st.session_state.guardrails_results)
            if gm["errors"]:
                st.warning(f"{gm['errors']} guardrail requests failed. Metrics cover {gm['completed']} completed requests.")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Correct", f"{gm['correct']}/{gm['completed']}")
            mc2.metric("Precision", f"{gm['precision']:.2f}")
            mc3.metric("Recall", f"{gm['recall']:.2f}")
            mc4.metric("Accuracy", f"{gm['accuracy']:.2f}")

    if st.session_state.enriched_dataset is not None:
        st.download_button(
            "Download captured results (JSON)",
            data=json.dumps(st.session_state.enriched_dataset, indent=2),
            file_name="rag_evaluation_capture.json",
            mime="application/json",
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eval Metrics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Eval Metrics — RAGAS + Tool Correctness")

    if not st.session_state.pipeline_done:
        st.warning("⚠️ Complete Step 2 (Live Pipeline) first to collect responses.")
    else:
        coverage = _render_pipeline_coverage(st.session_state.enriched_dataset)
        st.markdown(
            "Runs all **6 metric experiments** on the stored responses. "
            "Samples are scored one at a time with 40s cooldowns between samples "
            "and 62s between experiments. Full answers and evidence are sent to the judge; "
            "token costs, provider quotas, and response lengths determine runtime."
        )
        st.info(
            "The judge uses `JUDGE_GROQ` when configured and otherwise `GROQ_API_KEY`. "
            "Use a separate judge key and suitable provider limits for the complete evidence.",
            icon="ℹ️",
        )

        run_metrics_btn = st.button(
            "▶️ Run Eval Metrics",
            type="primary",
            disabled=not coverage["completed"],
        )

        if run_metrics_btn:
            st.session_state.metric_results = None
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

            metric_results = {}
            try:
                with logfire.span("📊 Streamlit — Run Metrics Button"):
                    metric_results = run_metrics_sync(st.session_state.enriched_dataset, status_cb=status_cb)
                    st.session_state.metric_results = metric_results
            except Exception:
                logfire.exception("Evaluation judging failed")
                status_slot.error(
                    "Judging did not complete. Check judge credentials, provider limits, and logs. "
                    "Captured query results are retained; scoring can be retried."
                )
            else:
                status_slot.success(f"All 6 experiments complete on {coverage['completed']}/{coverage['total']} captured answers.")

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
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
