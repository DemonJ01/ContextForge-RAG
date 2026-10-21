"""Exercise cloud responder gating with stub SDKs; no optional packages or API calls."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch


class CloudResponderTests(unittest.TestCase):
    def setUp(self):
        logfire = ModuleType("logfire")
        logfire.info = Mock()
        logfire.warning = Mock()
        logfire.error = Mock()
        logfire.span = Mock(side_effect=lambda *args, **kwargs: nullcontext())
        gateway = ModuleType("rag.gateway")
        self.create = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A supported response."))]
        ))
        gateway.portkey_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        gateway.extract_cache_status = Mock(return_value="MISS")
        source = Path(__file__).resolve().parents[1] / "rag" / "agents" / "nodes" / "answer_generator.py"
        spec = importlib.util.spec_from_file_location("_rag_responder_under_test", source)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"logfire": logfire, "rag.gateway": gateway}):
            spec.loader.exec_module(self.module)

    @staticmethod
    def state(query="Kubernetes jobs", documents=None):
        return {
            "current_query": query,
            "documents": documents,
            "messages": [{"role": "user", "content": "How do jobs work?"}],
            "plan": ["Context Retrieved"],
        }

    def test_technical_empty_context_never_calls_gateway_or_checkpoints_unsupported_text(self):
        for documents in (None, [], ["", " \n\t"], ["CONTENT:   "]):
            with self.subTest(documents=documents):
                result = self.module.generate_node(self.state(documents=documents))
                self.assertEqual(result["status"], "no_answer")
                self.assertIn("source evidence", result["final_answer"])
                self.assertEqual(result["messages"], [{"role": "assistant", "content": result["final_answer"]}])
                self.assertIn("Insufficient source evidence", result["plan"][-1])
        self.create.assert_not_called()

    def test_conversational_flow_still_calls_gateway_without_document_sources(self):
        result = self.module.generate_node(self.state(query="CONVERSATIONAL", documents=[]))
        self.create.assert_called_once()
        self.assertEqual(result["status"], "Response generated.")
        self.assertEqual(result["final_answer"], "A supported response.")

    def test_technical_flow_with_evidence_still_calls_gateway(self):
        evidence = "CONTENT: A Kubernetes Job creates one or more Pods."
        result = self.module.generate_node(self.state(documents=[evidence]))
        self.create.assert_called_once()
        prompt = self.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn(evidence, prompt)
        self.assertEqual(result["final_answer"], "A supported response.")

    def test_context_rejected_by_budget_does_not_trigger_ungrounded_generation(self):
        result = self.module.generate_node(self.state(documents=["x" * 25001]))
        self.assertEqual(result["status"], "no_answer")
        self.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
