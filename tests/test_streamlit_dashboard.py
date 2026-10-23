"""Optional Streamlit interaction regressions without provider calls."""
import builtins
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_AVAILABLE = all(importlib.util.find_spec(package) is not None for package in ("streamlit", "requests"))


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Install requirements-streamlit.txt for Streamlit tests")
class EvaluationDashboardTests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest

        self.app = AppTest.from_file(str(ROOT / "evaluation/rag_dashboard.py"), default_timeout=10)
        self.env = patch.dict(os.environ, {"JUDGE_GROQ": "", "LOGFIRE_TOKEN": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def assert_no_exception(self):
        self.assertEqual([item.value for item in self.app.exception], [])

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def run_pipeline(self):
        from evaluation.rag_pipeline import load_golden_dataset

        golden = load_golden_dataset()
        blocks = {sample["input"] for sample in golden["guardrails_samples"] if sample["expected_blocked"]}

        def response(*args, **kwargs):
            question = kwargs["json"]["q"]
            return Mock(json=Mock(return_value={
                "answer": "A response backed by the retrieved documents.",
                "sources": ["Retrieved source content"],
                "thought_process": ["Intent: Guardrails Fired" if question in blocks else "Local BM25 retrieval"],
            }))

        with patch("requests.get", return_value=Mock(json=Mock(return_value={"mode": "local"}))), \
                patch("requests.post", side_effect=response) as post, \
                patch("time.sleep") as sleep:
            self.button("▶️ Run Live Pipeline").click().run()
        self.assert_no_exception()
        self.assertTrue(self.app.session_state.pipeline_done)
        self.assertTrue(self.button("▶️ Run Live Pipeline").disabled)
        self.assertFalse(self.button("🔄 Reset & Re-run").disabled)
        self.assertTrue(any("Cloud safety rails are inactive" in item.value for item in self.app.info))
        self.assertEqual(post.call_count, len(golden["rag_samples"]) + len(golden["guardrails_samples"]))
        self.assertTrue(all(call.args[0].endswith("/query") for call in post.call_args_list))
        self.assertFalse(any(call.args[0] >= 2 for call in sleep.call_args_list))
        sessions = [call.kwargs["json"]["thread_id"] for call in post.call_args_list]
        self.assertEqual(len(sessions), len(set(sessions)))

    def test_corpus_opens_without_judge_or_tracing_packages(self):
        real_import = builtins.__import__

        def local_import(name, *args, **kwargs):
            if name == "evaluation.rag_metrics" or name.split(".")[0] in {"openai", "ragas", "logfire", "nest_asyncio"}:
                raise ModuleNotFoundError(name, name=name)
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=local_import):
            self.app.run()
        self.assert_no_exception()
        self.assertEqual(self.app.title[0].value, "RAG Evaluation")
        self.assertEqual(len(self.app.dataframe), 2)
        self.assertFalse(self.button("▶️ Run Live Pipeline").disabled)

    def test_live_pipeline_reset_and_repeat(self):
        self.app.run()
        self.run_pipeline()
        self.app.run()
        self.assert_no_exception()
        self.assertTrue(self.button("▶️ Run Live Pipeline").disabled)
        self.button("🔄 Reset & Re-run").click().run()
        self.assert_no_exception()
        self.assertFalse(self.app.session_state.pipeline_done)
        self.assertIsNone(self.app.session_state.enriched_dataset)
        self.run_pipeline()

    def test_offline_backend_reports_error_and_keeps_run_available(self):
        import requests

        self.app.run()
        with patch("requests.get", side_effect=requests.ConnectionError("offline")), patch("requests.post") as post:
            self.button("▶️ Run Live Pipeline").click().run()
        self.assert_no_exception()
        self.assertTrue(any("Cannot reach a healthy backend" in item.value for item in self.app.error))
        self.assertFalse(self.app.session_state.pipeline_done)
        self.assertFalse(self.button("▶️ Run Live Pipeline").disabled)
        post.assert_not_called()

    def test_failed_queries_are_reported_and_never_enable_judge(self):
        import requests

        self.app.run()
        with patch("requests.get", return_value=Mock(json=Mock(return_value={"mode": "local"}))), \
                patch("requests.post", side_effect=requests.ConnectionError("offline")), patch("time.sleep"):
            self.button("▶️ Run Live Pipeline").click().run()
        self.assert_no_exception()
        self.assertFalse(self.app.session_state.pipeline_done)
        self.assertTrue(any("responses collected" in item.value for item in self.app.warning))
        self.assertTrue(any("guardrail requests failed" in item.value for item in self.app.warning))
        self.assertFalse(any("Run Eval Metrics" in button.label for button in self.app.button))

    def test_judge_without_dedicated_key_warns_without_importing_provider(self):
        self.app.run()
        self.run_pipeline()
        with patch.dict(os.environ, {"GROQ_API_KEY": "production-key-must-not-be-used"}):
            self.button("▶️ Run Eval Metrics").click().run()
        self.assert_no_exception()
        self.assertTrue(any("Set JUDGE_GROQ" in item.value for item in self.app.warning))
        self.assertIsNone(self.app.session_state.metric_results)

    def test_judge_missing_packages_is_actionable(self):
        self.app.run()
        self.run_pipeline()
        real_import = builtins.__import__

        def missing_metrics(name, *args, **kwargs):
            if name == "evaluation.rag_metrics":
                raise ModuleNotFoundError("No module named ragas", name="ragas")
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"JUDGE_GROQ": "test-only"}), \
                patch("builtins.__import__", side_effect=missing_metrics):
            self.button("▶️ Run Eval Metrics").click().run()
        self.assert_no_exception()
        self.assertTrue(any("requirements-evaluation.txt" in item.value and "ragas" in item.value for item in self.app.warning))

    def test_judge_results_render_on_current_pandas_and_survive_rerun(self):
        import pandas as pd

        self.app.run()
        self.run_pipeline()
        metrics = ModuleType("evaluation.rag_metrics")

        async def mock_metrics(dataset, status_cb=None):
            return {name: pd.DataFrame([{"question": "Test question", name: 0.8}]) for name in (
                "faithfulness", "answer_relevancy", "context_precision", "context_recall",
                "answer_correctness", "tool_correctness",
            )}

        metrics.run_all_metrics = mock_metrics
        with patch.dict(os.environ, {"JUDGE_GROQ": "test-only"}), \
                patch.dict(sys.modules, {"evaluation.rag_metrics": metrics}):
            self.button("▶️ Run Eval Metrics").click().run()
        self.assert_no_exception()
        self.assertEqual(len(self.app.session_state.metric_results), 6)
        self.assertTrue(any(item.label == "Faithfulness" and item.value == "0.80" for item in self.app.metric))
        self.app.run()
        self.assert_no_exception()
        self.assertTrue(any("Metrics already computed" in item.value for item in self.app.success))

    def test_judge_runtime_failure_preserves_collected_pipeline(self):
        self.app.run()
        self.run_pipeline()
        metrics = ModuleType("evaluation.rag_metrics")

        async def failed_metrics(dataset, status_cb=None):
            raise RuntimeError("Judge quota unavailable")

        metrics.run_all_metrics = failed_metrics
        with patch.dict(os.environ, {"JUDGE_GROQ": "test-only"}), \
                patch.dict(sys.modules, {"evaluation.rag_metrics": metrics}):
            self.button("▶️ Run Eval Metrics").click().run()
        self.assert_no_exception()
        self.assertTrue(any("Judge quota unavailable" in item.value for item in self.app.error))
        self.assertTrue(self.app.session_state.pipeline_done)
        self.assertIsNone(self.app.session_state.metric_results)


if __name__ == "__main__":
    unittest.main()
