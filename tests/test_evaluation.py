"""Evaluation evidence, failure coverage, and recovery without provider calls."""
import asyncio
import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from threading import get_ident
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_optional(path, stubs):
    logger = ModuleType("logfire")
    logger.span = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    logger.info = Mock()
    logger.error = Mock()
    spec = importlib.util.spec_from_file_location("_evaluation_under_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"logfire": logger, **stubs}):
        spec.loader.exec_module(module)
    return module


def pipeline_module():
    requests = ModuleType("requests")
    requests.exceptions = SimpleNamespace(ConnectionError=ConnectionError)
    requests.post = Mock()
    module = load_optional("evaluation/rag_pipeline.py", {"requests": requests})
    module.time = SimpleNamespace(sleep=Mock())
    return module, requests


def response(answer, sources=None):
    result = Mock()
    result.json.return_value = {
        "answer": answer,
        "sources": sources or [],
        "thought_process": ["Intent: Technical"],
    }
    return result


def metrics_module():
    exports = {
        "pandas": ["DataFrame"], "openai": ["AsyncOpenAI"],
        "ragas.llms": ["llm_factory"], "ragas.embeddings": ["HuggingFaceEmbeddings"],
        "ragas.metrics.collections": [
            "Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall", "AnswerCorrectness",
        ],
    }
    stubs = {}
    for name, attributes in exports.items():
        stub = ModuleType(name)
        for attribute in attributes:
            setattr(stub, attribute, Mock())
        stubs[name] = stub
    return load_optional("evaluation/rag_metrics.py", stubs)


class EvaluationCaptureTests(unittest.TestCase):
    def test_capture_retains_claims_and_evidence_beyond_old_preview_limits(self):
        module, requests = pipeline_module()
        answer = "Supported opening. " * 30 + "A later incorrect claim must also be judged."
        sources = [f"Passage {index}: " + "evidence " * 100 for index in range(7)]
        requests.post.return_value = response(answer, sources)
        golden = {"rag_samples": [{"question": "How does this work?"}]}
        result = module.run_pipeline(golden)
        sample = result["rag_samples"][0]
        self.assertEqual(sample["actual_response"], answer)
        self.assertEqual(sample["actual_contexts"], sources)
        self.assertEqual(sample["pipeline_status"], "completed")
        self.assertIsNone(sample["pipeline_error"])
        self.assertEqual(result["pipeline_summary"]["coverage"], 1.0)
        self.assertNotIn("actual_response", golden["rag_samples"][0])

    def test_partial_failures_report_coverage_and_leave_resumable_checkpoints(self):
        module, requests = pipeline_module()
        requests.post.side_effect = [response("Captured answer"), ConnectionError("offline")]
        snapshots = []
        progress = Mock()
        result = module.run_pipeline(
            {"rag_samples": [{"question": "first"}, {"question": "second"}]},
            progress_callback=progress,
            checkpoint_callback=snapshots.append,
        )
        self.assertEqual(result["pipeline_summary"], {
            "total": 2, "completed": 1, "failed": 1, "pending": 0, "coverage": 0.5,
        })
        failed = result["rag_samples"][1]
        self.assertEqual(failed["actual_contexts"], [])
        self.assertEqual(failed["actual_response"], "")
        self.assertIn("Cannot reach the backend", failed["pipeline_error"])
        self.assertEqual(progress.call_args_list[-1].args[3], "error")
        self.assertEqual(snapshots[0]["pipeline_summary"]["pending"], 1)
        self.assertEqual(snapshots[1]["pipeline_summary"]["failed"], 1)
        snapshots[0]["rag_samples"][0]["actual_response"] = "changed snapshot"
        self.assertEqual(result["rag_samples"][0]["actual_response"], "Captured answer")

    def test_retry_preserves_completed_answers_and_requests_only_incomplete_samples(self):
        module, requests = pipeline_module()
        requests.post.return_value = response("Recovered answer")
        dataset = {"rag_samples": [
            {"question": "first", "actual_response": "Original answer", "actual_contexts": ["Original evidence"],
             "pipeline_status": "completed", "pipeline_error": None},
            {"question": "second", "actual_response": "", "pipeline_status": "error", "pipeline_error": "offline"},
            {"question": "third", "pipeline_status": "pending"},
        ]}
        result = module.run_pipeline(dataset, retry_failed_only=True)
        self.assertEqual(requests.post.call_count, 2)
        self.assertEqual([call.kwargs["json"]["q"] for call in requests.post.call_args_list], ["second", "third"])
        self.assertEqual(result["rag_samples"][0], dataset["rag_samples"][0])
        self.assertEqual(result["pipeline_summary"]["completed"], 3)
        self.assertTrue(all(sample["pipeline_error"] is None for sample in result["rag_samples"]))
        self.assertEqual(dataset["rag_samples"][1]["pipeline_status"], "error")
        module.time.sleep.assert_called_once()

    def test_invalid_api_answers_are_failures_instead_of_completed_samples(self):
        module, requests = pipeline_module()
        for answer in ("", None, {"unexpected": "object"}):
            with self.subTest(answer=answer):
                requests.post.return_value = response(answer)
                result = module.run_pipeline({"rag_samples": [{"question": "question"}]})
                self.assertEqual(result["pipeline_summary"]["failed"], 1)
                self.assertEqual(result["pipeline_summary"]["completed"], 0)

    def test_http_failure_reports_status_and_never_stale_successful_evidence(self):
        module, requests = pipeline_module()
        error = RuntimeError("provider credential must not appear in exported capture")
        error.response = SimpleNamespace(status_code=503)
        requests.post.return_value.raise_for_status.side_effect = error
        result = module.run_pipeline({"rag_samples": [{
            "question": "question", "actual_response": "Stale answer", "actual_contexts": ["Stale evidence"],
            "pipeline_status": "completed",
        }]})
        sample = result["rag_samples"][0]
        self.assertIn("HTTP 503", sample["pipeline_error"])
        self.assertNotIn("credential", sample["pipeline_error"])
        self.assertEqual(sample["actual_response"], "")
        self.assertEqual(sample["actual_contexts"], [])

    def test_http_success_with_error_status_is_not_a_completed_answer(self):
        module, requests = pipeline_module()
        requests.post.return_value = response("The provider failed.")
        requests.post.return_value.json.return_value["status"] = "error"
        result = module.run_pipeline({"rag_samples": [{"question": "question"}]})
        self.assertEqual(result["pipeline_summary"]["failed"], 1)
        self.assertEqual(result["rag_samples"][0]["actual_response"], "")


class EvaluationMetricsTests(unittest.TestCase):
    def test_judging_preserves_complete_live_evidence_and_excludes_failed_requests(self):
        module = metrics_module()
        evidence = ["Full evidence " * 100 for _ in range(5)]
        samples = module._prep_samples({"rag_samples": [
            {"actual_response": "Complete answer", "actual_contexts": evidence, "pipeline_status": "completed"},
            {"actual_response": "Stale answer", "pipeline_status": "error"},
            {"actual_response": "Pending answer", "pipeline_status": "pending"},
            {"actual_response": "No evidence", "actual_contexts": [], "relevant_contexts": ["Golden evidence"]},
        ]})
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["actual_contexts"], evidence)
        self.assertEqual(samples[1]["actual_contexts"], [])

    def test_cooldown_waits_for_remainder_and_short_intervals(self):
        module = metrics_module()
        for seconds, expected in ((62, [10] * 6 + [2]), (3, [3]), (0, [])):
            with self.subTest(seconds=seconds), patch.object(module.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                asyncio.run(module._cooldown(seconds, "test"))
                self.assertEqual([call.args[0] for call in sleep.call_args_list], expected)

    def test_missing_scores_raise_instead_of_silently_shortening_metric_coverage(self):
        module = metrics_module()
        with self.assertRaisesRegex(ValueError, "expected 2 scores, received 1"):
            module._score_df("faithfulness", [{"question": "one"}, {"question": "two"}], [SimpleNamespace(value=1)])

    def test_failed_only_capture_does_not_initialize_or_call_judge(self):
        module = metrics_module()
        with patch.object(module, "_build_judge") as build_judge:
            with self.assertRaisesRegex(ValueError, "Run Phase 1 first"):
                asyncio.run(module.run_all_metrics({"rag_samples": [{"pipeline_status": "error", "actual_response": ""}]}))
            build_judge.assert_not_called()

    def test_streamlit_bridge_isolates_judge_loop_and_delivers_progress_on_caller_thread(self):
        module = metrics_module()
        caller = get_ident()
        worker_threads = []
        callbacks = []

        async def judge(dataset, status_cb=None):
            worker_threads.append(get_ident())
            status_cb("first update")
            await asyncio.sleep(0)
            status_cb("second update")
            return {"finished": dataset["run_id"]}

        async def already_running_loop():
            return module.run_metrics_sync(
                {"run_id": "one"}, status_cb=lambda message: callbacks.append((get_ident(), message))
            )

        with patch.object(module, "run_all_metrics", side_effect=judge):
            result = asyncio.run(already_running_loop())
        self.assertEqual(result, {"finished": "one"})
        self.assertNotEqual(worker_threads, [caller])
        self.assertEqual(callbacks, [(caller, "first update"), (caller, "second update")])

    def test_streamlit_bridge_propagates_judge_failure_without_reporting_completion(self):
        module = metrics_module()

        async def judge(dataset, status_cb=None):
            status_cb("Scoring started")
            raise RuntimeError("judge unavailable")

        progress = Mock()
        with patch.object(module, "run_all_metrics", side_effect=judge):
            with self.assertRaisesRegex(RuntimeError, "judge unavailable"):
                module.run_metrics_sync({}, status_cb=progress)
        progress.assert_called_once_with("Scoring started")


class EvaluationGuardrailTests(unittest.TestCase):
    def test_unsuccessful_and_malformed_http_successes_never_count_as_passes(self):
        requests = ModuleType("requests")
        requests.exceptions = SimpleNamespace(ConnectionError=ConnectionError)
        requests.post = Mock()
        module = load_optional("evaluation/guardrails_evaluation.py", {"requests": requests})
        bodies = [
            {"answer": "The provider failed.", "status": "error", "thought_process": []},
            {}, [], {"answer": " ", "thought_process": []},
            {"answer": "Answer", "thought_process": "Intent: Guardrails Fired"},
        ]
        for body in bodies:
            with self.subTest(body=body):
                requests.post.return_value.json.return_value = body
                results = module.run_guardrails_eval([
                    {"id": "one", "input": "hello", "expected_blocked": False},
                ])
                self.assertEqual(results[0]["result"], "ERROR")
                self.assertIsNone(results[0]["actual_blocked"])
                metrics = module.compute_guardrails_metrics(results)
                self.assertEqual(metrics["completed"], 0)
                self.assertEqual(metrics["tn"], 0)
                self.assertEqual(metrics["errors"], 1)

    def test_valid_guardrail_outcomes_keep_all_four_confusion_matrix_categories(self):
        requests = ModuleType("requests")
        requests.exceptions = SimpleNamespace(ConnectionError=ConnectionError)
        requests.post = Mock()
        module = load_optional("evaluation/guardrails_evaluation.py", {"requests": requests})
        for blocked, expected, category in ((True, True, "TP"), (False, False, "TN"),
                                            (True, False, "FP"), (False, True, "FN")):
            with self.subTest(category=category):
                requests.post.return_value.json.return_value = {
                    "answer": "A valid response.",
                    "status": "Blocked by guardrails." if blocked else "complete",
                    "thought_process": ["Intent: Guardrails Fired"] if blocked else [],
                }
                results = module.run_guardrails_eval([
                    {"id": "one", "input": "question", "expected_blocked": expected},
                ])
                self.assertEqual(results[0]["result"], category)
                self.assertEqual(results[0]["actual_blocked"], blocked)


@unittest.skipUnless(
    importlib.util.find_spec("pandas") is not None and importlib.util.find_spec("streamlit") is not None,
    "Renderer integration requires the optional pandas and Streamlit packages.",
)
class EvaluationRendererTests(unittest.TestCase):
    def test_failed_judging_rerun_clears_previous_scores_and_summary(self):
        from streamlit.testing.v1 import AppTest

        source = (ROOT / "evaluation/rag_dashboard.py").read_text()
        tree = ast.parse(source)
        fragments = [
            "import pandas as pd\nimport streamlit as st\n"
            "from contextlib import nullcontext\nfrom types import SimpleNamespace",
            "logfire = SimpleNamespace(span=lambda *args, **kwargs: nullcontext(), "
            "exception=lambda *args, **kwargs: None)",
            "def _render_pipeline_coverage(dataset):\n    return {'completed': 1, 'total': 1}",
            "def run_metrics_sync(dataset, status_cb=None):\n    raise RuntimeError('judge unavailable')",
            "if 'metric_results' not in st.session_state:\n"
            "    st.session_state.metric_results = {'faithfulness': "
            "pd.DataFrame({'question': ['one'], 'faithfulness': [0.8]})}\n"
            "st.session_state.pipeline_done = True\n"
            "st.session_state.enriched_dataset = {'rag_samples': [{}]}\n"
            "st.session_state.guardrails_results = None\n"
            "tab3 = st.container()",
        ]
        helper_names = {"_badge", "_grade", "_color_score", "_render_metric_table"}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SCORE_COLORS" for target in node.targets
            ):
                fragments.append(ast.get_source_segment(source, node))
            elif isinstance(node, ast.FunctionDef) and node.name in helper_names:
                fragments.append(ast.get_source_segment(source, node))
            elif isinstance(node, ast.With) and any(
                isinstance(item.context_expr, ast.Name) and item.context_expr.id == "tab3"
                for item in node.items
            ):
                fragments.append(ast.get_source_segment(source, node))

        app = AppTest.from_string("\n\n".join(fragments), default_timeout=10).run()
        self.assertEqual([exception.value for exception in app.exception], [])
        self.assertTrue(app.session_state.metric_results)
        self.assertTrue(any(header.value == "Final Summary" for header in app.subheader))

        app.button[0].click().run()
        self.assertEqual([exception.value for exception in app.exception], [])
        self.assertTrue(any("Judging did not complete" in error.value for error in app.error))
        self.assertIsNone(app.session_state.metric_results)
        self.assertFalse(any(header.value == "Final Summary" for header in app.subheader))
        self.assertEqual(len(app.dataframe), 0)

    def test_metric_renderer_displays_results_with_installed_pandas_and_streamlit(self):
        from streamlit.testing.v1 import AppTest

        # Execute the real renderer through Streamlit without initializing unrelated
        # cloud SDKs, provider clients, or tracing exporters from the dashboard.
        source = (ROOT / "evaluation/rag_dashboard.py").read_text()
        tree = ast.parse(source)
        helper_names = {"_badge", "_grade", "_color_score", "_render_metric_table"}
        fragments = ["import pandas as pd\nimport streamlit as st"]
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SCORE_COLORS" for target in node.targets
            ):
                fragments.append(ast.get_source_segment(source, node))
            elif isinstance(node, ast.FunctionDef) and node.name in helper_names:
                fragments.append(ast.get_source_segment(source, node))
        fragments.append(
            '_render_metric_table(pd.DataFrame({"question": ["one", "two"], '
            '"faithfulness": [0.8, 0.4]}), "faithfulness", "Faithfulness")'
        )
        app = AppTest.from_string("\n\n".join(fragments), default_timeout=10).run()
        self.assertEqual([exception.value for exception in app.exception], [])
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(app.dataframe[0].value["faithfulness"].tolist(), [0.8, 0.4])


if __name__ == "__main__":
    unittest.main()
