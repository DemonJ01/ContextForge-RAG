"""Safe legacy chat errors, with optional checks using Streamlit's real renderer."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import time
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from rag.client_config import QueryResponseError, query_response


ROOT = Path(__file__).resolve().parents[1]


def response(status, body):
    result = Mock(status_code=status)
    result.json.return_value = body
    return result


class QueryResponseTests(unittest.TestCase):
    def test_keeps_success_and_no_evidence_answers(self):
        for status in ("complete", "no_answer"):
            data = {"answer": "A usable answer", "status": status, "sources": []}
            self.assertIs(query_response(response(200, data)), data)

    def test_configuration_failure_preserves_actionable_safe_detail(self):
        detail = "Cloud mode is missing PORTKEY_API_KEY. Configure it or use RAG_MODE=local."
        with self.assertRaisesRegex(QueryResponseError, "PORTKEY_API_KEY.*RAG_MODE=local"):
            query_response(response(503, {"detail": detail}))

    def test_validation_failure_does_not_display_raw_validation_inputs(self):
        with self.assertRaisesRegex(QueryResponseError, "1 and 2,000 characters") as error:
            query_response(response(422, {"detail": [{"input": "PRIVATE_INPUT"}]}))
        self.assertNotIn("PRIVATE_INPUT", str(error.exception))

    def test_other_failures_never_display_raw_http_body(self):
        for status in (401, 500, 502, 503):
            with self.subTest(status=status):
                result = response(status, {"unexpected": "PRIVATE_BODY"})
                result.text = "PRIVATE_BODY"
                with self.assertRaisesRegex(QueryResponseError, f"HTTP {status}") as error:
                    query_response(result)
                self.assertNotIn("PRIVATE_BODY", str(error.exception))

    def test_rejects_unreadable_empty_and_error_answers(self):
        result = response(200, None)
        result.json.side_effect = ValueError("PRIVATE_PARSE_DETAILS")
        with self.assertRaisesRegex(QueryResponseError, "unreadable response") as error:
            query_response(result)
        self.assertNotIn("PRIVATE_PARSE_DETAILS", str(error.exception))
        for data in (None, [], {"answer": " "}, {"answer": "x", "status": "error"}):
            with self.subTest(data=data), self.assertRaisesRegex(QueryResponseError, "no usable answer"):
                query_response(response(200, data))


@unittest.skipUnless(importlib.util.find_spec("streamlit") is not None,
                     "optional Streamlit dependency is not installed")
class LegacyChatRenderingTests(unittest.TestCase):
    def run_chat(self, filename, result=None, error=None):
        from streamlit.testing.v1 import AppTest

        telemetry = ModuleType("logfire")
        telemetry.configure = telemetry.instrument_requests = Mock()
        telemetry.info = telemetry.error = telemetry.warn = telemetry.warning = Mock()
        telemetry.span = lambda *args, **kwargs: nullcontext()
        real_sleep = time.sleep
        # Speed only the app's presentation delay while retaining the test runner's waits.
        def animation_sleep(duration):
            real_sleep(0.0001 if duration == 0.005 else duration)

        with patch.dict(sys.modules, {"logfire": telemetry}), \
                patch("requests.post", return_value=result, side_effect=error), \
                patch("time.sleep", side_effect=animation_sleep):
            app = AppTest.from_file(str(ROOT / filename), default_timeout=20).run()
            app.chat_input[0].set_value("How do Kubernetes CronJobs work?").run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_success_renders_answer_and_sources_in_both_entrypoints(self):
        for filename in ("ui/rag_chat.py", "ui/rag_chat_cloud.py"):
            with self.subTest(filename=filename):
                app = self.run_chat(filename, response(200, {
                    "answer": "CronJobs create Jobs on a schedule. [1]", "status": "complete",
                    "sources": ["The supporting source passage."], "thought_process": [],
                }))
                self.assertEqual(len(app.error), 0)
                self.assertIn("The supporting source passage.", [entry.value for entry in app.info])
                self.assertEqual(app.session_state["messages"][-1]["role"], "assistant")

    def test_http_failures_are_actionable_in_both_entrypoints(self):
        cases = [(503, {"detail": "Configure PORTKEY_API_KEY or use RAG_MODE=local."}, "PORTKEY_API_KEY"),
                 (422, {"detail": [{"input": "PRIVATE_INPUT"}]}, "1 and 2,000 characters")]
        for filename in ("ui/rag_chat.py", "ui/rag_chat_cloud.py"):
            for status, body, expected in cases:
                with self.subTest(filename=filename, status=status):
                    app = self.run_chat(filename, response(status, body))
                    message = app.error[0].value
                    self.assertIn(expected, message)
                    self.assertNotIn("Offline", message)
                    self.assertNotIn("PRIVATE_INPUT", message)
                    self.assertEqual(app.session_state["messages"][-1]["role"], "user")

    def test_transport_errors_do_not_reveal_request_details(self):
        import requests

        for filename in ("ui/rag_chat.py", "ui/rag_chat_cloud.py"):
            for error, expected in ((requests.ConnectionError("PRIVATE_URL"), "Cannot reach the backend"),
                                    (requests.Timeout("PRIVATE_URL"), "search timed out")):
                with self.subTest(filename=filename, error=type(error).__name__):
                    app = self.run_chat(filename, error=error)
                    self.assertIn(expected, app.error[0].value)
                    self.assertNotIn("PRIVATE_URL", app.error[0].value)

    def test_no_evidence_response_is_not_labeled_conversational(self):
        app = self.run_chat("ui/rag_chat_cloud.py", response(200, {
            "answer": "I couldn't find enough evidence.", "status": "no_answer", "sources": [],
        }))
        self.assertEqual(len(app.error), 0)
        self.assertIn("No supporting evidence", app.caption[0].value)
        self.assertNotIn("conversational", app.caption[0].value)


if __name__ == "__main__":
    unittest.main()
