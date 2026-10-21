"""HTTP boundary tests; run with python -m unittest discover -s tests."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag.api import PROJECT_ROOT, create_app
from rag.runtime import RAGRuntime, RuntimeUnavailable


class FakeRuntime:
    mode = "local"

    def __init__(self):
        self.queries = []
        self.document_lookups = []

    def health(self):
        return {"status": "ready", "mode": self.mode, "document_count": 2, "chunk_count": 4}

    def documents(self, scope="all"):
        return [{"id": "abc123", "title": "Cron jobs", "scope": scope}]

    def document(self, document_id):
        self.document_lookups.append(document_id)
        return {"id": document_id, "content": "Run scheduled jobs."} if document_id == "abc123" else None

    def query(self, q, **kwargs):
        self.queries.append((q, kwargs))
        return {
            "answer": "Cron jobs run on a schedule. [1]", "status": "answered",
            "thought_process": ["Ranked document chunks"],
            "sources": ["Cron jobs run on a schedule."],
            "citations": [{"id": 1, "document_id": "abc123", "filename": "cronjobs.docx", "excerpt": "Cron jobs run on a schedule."}],
        }

    def graph(self):
        return {"mode": "local", "format": "mermaid", "diagram": "graph LR\nQuery --> Answer"}


class APIBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        web = self.root / "ui" / "web"
        web.mkdir(parents=True)
        (web / "index.html").write_text("<html><title>RAG</title></html>", encoding="utf-8")
        (web / "app.js").write_text("window.rag = true;", encoding="utf-8")
        self.runtime = FakeRuntime()
        self.client = TestClient(create_app(self.root, runtime=self.runtime))

    def tearDown(self):
        self.client.close()
        self.directory.cleanup()

    def test_home_assets_health_and_document_browsing(self):
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("text/html", home.headers["content-type"])
        self.assertIn("<title>RAG</title>", home.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/health").json()["document_count"], 2)
        self.assertEqual(self.client.get("/documents?scope=reference").json()["documents"][0]["scope"], "reference")
        self.assertEqual(self.client.get("/documents").json()["total"], 1)
        self.assertEqual(self.client.get("/documents/abc123").json()["content"], "Run scheduled jobs.")
        self.assertEqual(self.client.get("/graph").json()["format"], "mermaid")

    def test_query_keeps_legacy_sources_and_structured_citations(self):
        response = self.client.post("/query", json={"q": "  How do cron jobs run?  ", "scope": "all", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["question"], "How do cron jobs run?")
        self.assertEqual(data["mode"], "local")
        self.assertIsInstance(data["elapsed_ms"], float)
        self.assertIsInstance(data["sources"][0], str)
        self.assertEqual(data["citations"][0]["document_id"], "abc123")
        self.assertEqual(self.runtime.queries[0][1]["scope"], "all")
        self.assertEqual(self.runtime.queries[0][1]["top_k"], 2)
        self.assertEqual(self.runtime.queries[0][1]["thread_id"], data["thread_id"])

    def test_generated_threads_are_distinct_and_explicit_threads_are_preserved(self):
        first = self.client.post("/query", json={"q": "Hello"}).json()["thread_id"]
        second = self.client.post("/query", json={"q": "Hello"}).json()["thread_id"]
        self.assertNotEqual(first, second)
        explicit = self.client.post("/query", json={"q": "Hello", "thread_id": "session_123-safe"}).json()
        self.assertEqual(explicit["thread_id"], "session_123-safe")

    def test_invalid_requests_do_not_reach_engine(self):
        bad_requests = [
            {}, {"q": ""}, {"q": " \n\t "}, {"q": "x" * 2001}, {"q": 12},
            {"q": "hello", "scope": "private"}, {"q": "hello", "top_k": 0},
            {"q": "hello", "top_k": 9}, {"q": "hello", "top_k": True},
            {"q": "hello", "thread_id": "../another"},
            {"q": "hello", "thread_id": "a" * 129},
        ]
        for payload in bad_requests:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/query", json=payload).status_code, 422)
        self.assertEqual(self.client.get("/documents?scope=private").status_code, 422)
        self.assertEqual(self.runtime.queries, [])

    def test_unknown_or_unsafe_document_ids_never_become_paths(self):
        self.assertEqual(self.client.get("/documents/unknown").status_code, 404)
        for document_id in ("%2e%2e", "secret.env", "invalid:id", "x" * 129):
            with self.subTest(document_id=document_id):
                self.assertEqual(self.client.get("/documents/" + document_id).status_code, 404)
        self.assertEqual(self.runtime.document_lookups, ["unknown"])
        self.assertEqual(self.client.get("/static/../runtime.py").status_code, 404)

    def test_failures_have_error_statuses_without_exception_contents(self):
        with patch.object(self.runtime, "query", side_effect=RuntimeError("private provider diagnostic")):
            with self.assertLogs("rag.api", level="ERROR"):
                response = self.client.post("/query", json={"q": "hello"})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private provider diagnostic", response.text)
        with patch.object(self.runtime, "health", side_effect=RuntimeUnavailable("Choose a configured mode.")):
            unavailable = self.client.get("/health")
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["detail"], "Choose a configured mode.")


class RuntimeModeTests(unittest.TestCase):
    def test_local_mode_is_default_and_never_imports_cloud_modules(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = RAGRuntime(Path("."))
        local = FakeRuntime()
        runtime._local_engine = local
        with patch("rag.runtime.importlib.import_module", side_effect=AssertionError("Unexpected SDK import")):
            self.assertEqual(runtime.health()["mode"], "local")
            result = runtime.query("Hello", thread_id="abc")
        self.assertEqual(result["status"], "answered")

    def test_project_environment_file_selects_mode_without_overriding_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("RAG_MODE=cloud\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(RAGRuntime(root).mode, "cloud")
            with patch.dict(os.environ, {"RAG_MODE": "local"}, clear=True):
                self.assertEqual(RAGRuntime(root).mode, "local")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(RAGRuntime(root, mode="local").mode, "local")
                self.assertNotIn("RAG_MODE", os.environ)

    def test_empty_index_is_not_reported_as_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = RAGRuntime(Path(directory), mode="local")
            with TestClient(create_app(Path(directory), runtime=runtime)) as client:
                response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertIn("processed_documents", response.json()["detail"])

    def test_cloud_missing_credentials_returns_503_and_never_uses_local_query(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = RAGRuntime(Path("."), mode="cloud")
            runtime._local_engine = FakeRuntime()
            with patch("rag.runtime.importlib.import_module", side_effect=AssertionError("Unexpected SDK import")):
                with TestClient(create_app(runtime=runtime)) as client:
                    response = client.post("/query", json={"q": "How do jobs work?"})
                    health = client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(health.status_code, 503)
        self.assertIn("PORTKEY_API_KEY", response.json()["detail"])
        self.assertIn("RAG_MODE=local", response.json()["detail"])
        self.assertEqual(runtime._local_engine.queries, [])

    def test_cloud_legacy_flow_uses_request_thread_and_preserves_response(self):
        class Agent:
            def invoke(self, state, config):
                self.state = state
                self.config = config
                return {"final_answer": "A cloud answer.", "plan": ["Context Retrieved"], "documents": ["Evidence"], "status": "Response generated."}
        runtime = RAGRuntime(Path("."), mode="cloud")
        agent = Agent()
        runtime._cloud_components = (agent, lambda q: (False, None))
        with TestClient(create_app(runtime=runtime)) as client:
            response = client.post("/query", json={"q": "  How do jobs work?  ", "thread_id": "unique123"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "cloud")
        self.assertEqual(response.json()["sources"], ["Evidence"])
        self.assertEqual(agent.config["configurable"]["thread_id"], "unique123")
        self.assertEqual(agent.state["messages"], [{"role": "user", "content": "How do jobs work?"}])

    def test_cloud_technical_answer_requires_nonblank_source_evidence(self):
        class Agent:
            def __init__(self, documents):
                self.documents = documents
            def invoke(self, state, config):
                return {"current_query": "Kubernetes jobs", "final_answer": "Unsupported claim.",
                        "documents": self.documents, "plan": ["Context Retrieved"], "status": "Response generated."}
        for documents in ([], ["", " \n\t"], ["CONTENT:   "], None):
            with self.subTest(documents=documents):
                runtime = RAGRuntime(PROJECT_ROOT, mode="cloud")
                runtime._cloud_components = (Agent(documents), lambda q: (False, None))
                with TestClient(create_app(runtime=runtime)) as client:
                    response = client.post("/query", json={"q": "How do Kubernetes jobs work?"})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], "no_answer")
                self.assertNotIn("Unsupported claim.", data["answer"])
                self.assertEqual(data["sources"], [])
                self.assertEqual(data["citations"], [])
                self.assertIn("Insufficient source evidence", data["thought_process"][-1])

    def test_cloud_conversation_can_answer_without_document_sources(self):
        class Agent:
            def invoke(self, state, config):
                return {"current_query": "CONVERSATIONAL", "final_answer": "Hello!", "documents": [],
                        "plan": ["Retrieval: Skipped"], "status": "Response generated."}
        runtime = RAGRuntime(PROJECT_ROOT, mode="cloud")
        runtime._cloud_components = (Agent(), lambda q: (False, None))
        with TestClient(create_app(runtime=runtime)) as client:
            response = client.post("/query", json={"q": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Hello!")
        self.assertEqual(response.json()["status"], "Response generated.")
        self.assertEqual(response.json()["sources"], [])

    def test_empty_cloud_model_response_is_an_actionable_503(self):
        class Agent:
            def invoke(self, state, config):
                return {"current_query": "Kubernetes", "final_answer": " ", "documents": ["Evidence"], "plan": []}
        runtime = RAGRuntime(PROJECT_ROOT, mode="cloud")
        runtime._cloud_components = (Agent(), lambda q: (False, None))
        with TestClient(create_app(runtime=runtime)) as client:
            response = client.post("/query", json={"q": "Kubernetes"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("empty answer", response.json()["detail"])

    def test_unknown_mode_is_an_actionable_503(self):
        with TestClient(create_app(runtime=RAGRuntime(Path("."), mode="unexpected"))) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertIn("RAG_MODE", response.json()["detail"])


class RealEngineIntegrationTests(unittest.TestCase):
    def test_bundled_documents_are_usable_without_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = RAGRuntime(PROJECT_ROOT, mode="local")
            with TestClient(create_app(PROJECT_ROOT, runtime=runtime)) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                stats = health.json()
                self.assertEqual(stats["mode"], "local")
                self.assertFalse(stats["credentials_required"])
                self.assertGreater(stats["document_count"], 0)
                self.assertGreater(stats["chunk_count"], 0)
                listing = client.get("/documents?scope=reference").json()
                self.assertEqual(listing["total"], len(listing["documents"]))
                self.assertGreater(listing["total"], 0)
                reply = client.post("/query", json={"q": "What are Kubernetes CronJobs?", "top_k": 3})
                self.assertEqual(reply.status_code, 200)
                answer = reply.json()
                self.assertTrue(answer["sources"])
                self.assertTrue(answer["citations"])
                for citation in answer["citations"]:
                    detail = client.get("/documents/" + citation["document_id"])
                    self.assertEqual(detail.status_code, 200)
                    self.assertIn(citation["excerpt"], detail.json()["content"])



if __name__ == "__main__":
    unittest.main()
