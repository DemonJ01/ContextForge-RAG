"""Optional Streamlit interaction tests; no provider credentials are needed."""
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


ROOT = Path(__file__).resolve().parent.parent


@unittest.skipIf(AppTest is None, "Install requirements-streamlit.txt to test the optional Streamlit clients")
class StreamlitClientTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"BACKEND_URL": "http://backend.test/"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def app(self, filename="rag_chat.py"):
        app = AppTest.from_file(str(ROOT / "ui" / filename)).run(timeout=10)
        self.assertEqual(len(app.exception), 0)
        return app

    def response(self, **overrides):
        data = {
            "answer": "CronJobs run on a schedule. [1]", "status": "complete", "mode": "local",
            "sources": ["CronJobs run on a schedule."],
            "citations": [{"filename": "cronjobs.docx", "excerpt": "CronJobs run on a schedule."}],
        }
        data.update(overrides)
        response = Mock()
        response.json.return_value = data
        return response

    def click(self, app, label):
        next(button for button in app.button if button.label == label).click().run(timeout=10)
        self.assertEqual(len(app.exception), 0)

    def test_both_entrypoints_start_without_secrets_or_cloud_dependencies(self):
        for filename in ("rag_chat.py", "rag_chat_cloud.py"):
            with self.subTest(filename=filename):
                app = self.app(filename)
                self.assertEqual(app.title[0].value, "RAG")
                self.assertTrue(any("http://backend.test" in caption.value for caption in app.caption))

    def test_repeated_questions_keep_sources_after_reruns(self):
        with patch("ui.streamlit_app.requests.post", return_value=self.response()) as post:
            app = self.app()
            for question in ("What are CronJobs?", "How do Kubernetes Jobs differ?", "How do queues work?"):
                app.chat_input[0].set_value(question).run(timeout=10)
                self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.chat_message), 6)
            self.assertEqual(len(app.expander), 3)
            self.assertEqual(post.call_count, 3)
            requests = [call.kwargs["json"] for call in post.call_args_list]
            self.assertEqual(len({request["thread_id"] for request in requests}), 1)
            self.assertTrue(all(request["scope"] == "reference" and request["top_k"] == 4 for request in requests))
            app.run(timeout=10)
            self.assertEqual(len(app.expander), 3)
            self.assertEqual(post.call_count, 3)
            self.assertEqual(sum(text.value == "CronJobs run on a schedule." for text in app.text), 3)

    def test_local_answers_and_excerpts_preserve_markup_literally(self):
        excerpt = "# Cron schedule\n* * * * *\n`kubectl get jobs`\n---\n# YAML comment"
        answer = f"Source excerpts:\n\n{excerpt} [1]"
        with patch("ui.streamlit_app.requests.post", return_value=self.response(
            answer=answer, sources=[excerpt], citations=[{"filename": "cron.yaml", "excerpt": excerpt}]
        )):
            app = self.app()
            app.chat_input[0].set_value("CronJobs").run(timeout=10)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual([text.value for text in app.text], [answer, excerpt])
            self.assertFalse(any("# Cron schedule" in markdown.value for markdown in app.markdown))
            app.run(timeout=10)
            self.assertEqual([text.value for text in app.text], [answer, excerpt])

    def test_scope_source_count_and_new_conversation(self):
        with patch("ui.streamlit_app.requests.post", return_value=self.response()) as post:
            app = self.app()
            app.selectbox[0].select("all").run()
            app.slider[0].set_value(2).run()
            app.chat_input[0].set_value("CronJobs").run(timeout=10)
            original_thread = post.call_args.kwargs["json"]["thread_id"]
            self.assertEqual(post.call_args.kwargs["json"]["scope"], "all")
            self.assertEqual(post.call_args.kwargs["json"]["top_k"], 2)
            self.click(app, "New conversation")
            self.assertEqual(len(app.chat_message), 0)
            self.assertNotEqual(app.session_state["session_id"], original_thread)
            app.chat_input[0].set_value("Jobs").run(timeout=10)
            self.assertNotEqual(post.call_args.kwargs["json"]["thread_id"], original_thread)

    def test_no_evidence_response_is_not_labelled_conversational(self):
        with patch("ui.streamlit_app.requests.post", return_value=self.response(
            answer="I couldn't find enough evidence.", status="no_answer", citations=[], sources=[]
        )):
            app = self.app()
            app.chat_input[0].set_value("What is the Moon made of?").run(timeout=10)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.expander), 0)
            self.assertTrue(any("No supporting sources" in caption.value for caption in app.caption))
            self.assertFalse(any("conversational" in caption.value.lower() for caption in app.caption))

    def test_failed_question_can_retry_with_its_original_scope(self):
        import requests

        with patch("ui.streamlit_app.requests.post", side_effect=[requests.Timeout(), self.response()]) as post:
            app = self.app()
            app.chat_input[0].set_value("CronJobs").run(timeout=10)
            self.assertEqual(len(app.error), 1)
            first_request = dict(post.call_args.kwargs["json"])
            app.selectbox[0].select("all").run()
            self.assertEqual(len(app.error), 1)
            self.click(app, "Retry")
            self.assertEqual(len(app.error), 0)
            self.assertEqual(len(app.chat_message), 2)
            self.assertEqual(post.call_args.kwargs["json"], first_request)
            self.assertEqual(len(app.expander), 1)

    def test_invalid_backend_response_shows_error_and_recovers(self):
        with patch("ui.streamlit_app.requests.post", side_effect=[self.response(answer=None), self.response()]):
            app = self.app()
            app.chat_input[0].set_value("CronJobs").run(timeout=10)
            self.assertEqual(len(app.exception), 0)
            self.assertIn("invalid response", app.error[0].value)
            self.click(app, "Retry")
            self.assertEqual(len(app.error), 0)
            self.assertEqual(len(app.chat_message), 2)

    def test_cloud_secrets_backend_takes_precedence(self):
        with patch("ui.streamlit_app.requests.post", return_value=self.response()) as post:
            app = AppTest.from_file(str(ROOT / "ui" / "rag_chat_cloud.py"))
            app.secrets["BACKEND_URL"] = " https://deployment.test/ "
            app.run(timeout=10)
            app.chat_input[0].set_value("CronJobs").run(timeout=10)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(post.call_args.args[0], "https://deployment.test/query")


if __name__ == "__main__":
    unittest.main()
