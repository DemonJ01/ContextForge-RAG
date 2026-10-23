"""Offline regressions for optional cloud components; no provider calls needed."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def logger_stub():
    logger = ModuleType("logfire")
    for name in ("info", "warning", "error", "configure"):
        setattr(logger, name, Mock())
    logger.span = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    return logger


def load_module(path, stubs):
    spec = importlib.util.spec_from_file_location("_optional_under_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"logfire": logger_stub(), **stubs}):
        spec.loader.exec_module(module)
    return module


class CloudRetrievalTests(unittest.TestCase):
    def test_retriever_preserves_request_scope_and_evidence_limit(self):
        vector_store = ModuleType("rag.services.retrieval.vector_store")
        vector_store.search_rag_knowledge = Mock(return_value=[{"content": "reference evidence"}])
        reranker = ModuleType("rag.services.retrieval.reranker")
        reranker.rerank_documents = Mock(return_value=["reference evidence"])
        module = load_module("rag/agents/nodes/document_retriever.py", {
            vector_store.__name__: vector_store, reranker.__name__: reranker,
        })
        result = module.retrieve_node({"current_query": "CronJobs", "plan": [], "scope": "reference", "top_k": 2})
        vector_store.search_rag_knowledge.assert_called_once_with("CronJobs", limit=15, scope="reference")
        reranker.rerank_documents.assert_called_once_with("CronJobs", ["reference evidence"], top_n=2)
        self.assertEqual(result["documents"], ["CONTENT: reference evidence"])

    def test_qdrant_reference_scope_filters_the_ingested_source_type(self):
        qdrant = ModuleType("qdrant_client")
        client = Mock()
        client.query_points.return_value.points = []
        qdrant.QdrantClient = Mock(return_value=client)
        http = ModuleType("qdrant_client.http")
        http.models = SimpleNamespace(Filter=SimpleNamespace, FieldCondition=SimpleNamespace, MatchValue=SimpleNamespace)
        embeddings = ModuleType("rag.services.retrieval.embedding_service")
        embeddings.embed_query = Mock(return_value=[0.1, 0.2])
        settings = ModuleType("rag.settings")
        settings.settings = SimpleNamespace(QDRANT_URL="https://qdrant.example", QDRANT_API_KEY="", QDRANT_COLLECTION="rag")
        module = load_module("rag/services/retrieval/vector_store.py", {
            "qdrant_client": qdrant, "qdrant_client.http": http,
            embeddings.__name__: embeddings, settings.__name__: settings,
        })
        module.search_rag_knowledge("CronJobs", scope="reference")
        condition = client.query_points.call_args.kwargs["query_filter"].must[0]
        self.assertEqual(condition.key, "source_type")
        self.assertEqual(condition.match.value, "true")
        module.search_rag_knowledge("CronJobs", scope="all")
        self.assertIsNone(client.query_points.call_args.kwargs["query_filter"])
        client.query_points.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "Cloud document search failed"):
            module.search_rag_knowledge("CronJobs")


class CloudChunkingTests(unittest.TestCase):
    def test_long_pdf_paragraphs_are_bounded_without_losing_content(self):
        module = load_module("rag/ingestion/chunking/text_splitter.py", {})
        text = "A" * 4001 + "\n\n" + "B" * 13
        chunks = module.chunk_text(text, chunk_size=1500)
        self.assertTrue(all(0 < len(chunk) <= 1500 for chunk in chunks))
        self.assertEqual("".join(chunks).replace("\n", ""), text.replace("\n", ""))

    def test_short_paragraphs_pack_within_the_exact_limit(self):
        module = load_module("rag/ingestion/chunking/text_splitter.py", {})
        self.assertEqual(module.chunk_text("1234\n\n5678", chunk_size=10), ["1234\n\n5678"])
        self.assertEqual(module.chunk_text(" \n\t"), [])
        with self.assertRaises(ValueError):
            module.chunk_text("text", chunk_size=0)


class PdfLoaderTests(unittest.TestCase):
    def test_fallback_text_keeps_its_original_page_order(self):
        pypdf = ModuleType("pypdf")
        pypdf.PdfReader = Mock(return_value=SimpleNamespace(pages=[
            SimpleNamespace(extract_text=Mock(return_value="First page")),
            SimpleNamespace(extract_text=Mock(return_value="")),
            SimpleNamespace(extract_text=Mock(return_value="Last page")),
        ]))
        plumber = ModuleType("pdfplumber")
        plumber.open = Mock(return_value=nullcontext(SimpleNamespace(pages=[
            SimpleNamespace(extract_text=Mock(return_value="unused")),
            SimpleNamespace(extract_text=Mock(return_value="Middle page")),
            SimpleNamespace(extract_text=Mock(return_value="unused")),
        ])))
        module = load_module("rag/ingestion/loaders/pdf_loader.py", {"pypdf": pypdf})
        with patch.dict(sys.modules, {"pdfplumber": plumber}):
            self.assertEqual(module.parse_pdf("example.pdf"), "First page\nMiddle page\nLast page")


class EvaluationFailureTests(unittest.TestCase):
    def requests_stub(self):
        requests = ModuleType("requests")
        requests.exceptions = SimpleNamespace(ConnectionError=ConnectionError)
        requests.post = Mock(side_effect=ConnectionError("offline"))
        return requests

    def test_live_pipeline_failure_never_invents_retrieved_golden_context(self):
        requests = self.requests_stub()
        module = load_module("evaluation/rag_pipeline.py", {"requests": requests})
        golden = {"rag_samples": [{"question": "CronJobs?", "relevant_contexts": ["Expected answer evidence"]}]}
        result = module.run_pipeline(golden)
        self.assertEqual(result["rag_samples"][0]["actual_contexts"], [])
        self.assertEqual(result["rag_samples"][0]["actual_response"], "")
        self.assertNotIn("actual_response", golden["rag_samples"][0])

    def test_evaluation_runs_use_distinct_sessions(self):
        requests = self.requests_stub()
        module = load_module("evaluation/rag_pipeline.py", {"requests": requests})
        golden = {"rag_samples": [{"question": "CronJobs?"}]}
        module.run_pipeline(golden)
        module.run_pipeline(golden)
        sessions = [call.kwargs["json"]["thread_id"] for call in requests.post.call_args_list]
        self.assertNotEqual(sessions[0], sessions[1])
        self.assertEqual(module.detect_tool(["Searched 6 indexed documents using local BM25 retrieval."]), "retrieve_documents")

    def test_failed_guardrail_request_is_an_error_and_never_a_pass(self):
        requests = self.requests_stub()
        module = load_module("evaluation/guardrails_evaluation.py", {"requests": requests})
        results = module.run_guardrails_eval([{"id": "one", "input": "hello", "expected_blocked": False}])
        self.assertEqual(results[0]["result"], "ERROR")
        self.assertIsNone(results[0]["actual_blocked"])
        metrics = module.compute_guardrails_metrics(results)
        self.assertEqual(metrics["tn"], 0)
        self.assertEqual(metrics["completed"], 0)
        self.assertEqual(metrics["errors"], 1)
        metrics = module.compute_guardrails_metrics(results + [{"result": "TP"}])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["total"], 2)

    def test_metric_preparation_preserves_empty_live_retrieval(self):
        stubs = {}
        exports = {
            "pandas": ["DataFrame"], "openai": ["AsyncOpenAI"],
            "ragas": ["SingleTurnSample"], "ragas.llms": ["llm_factory"],
            "ragas.embeddings": ["HuggingFaceEmbeddings"],
            "ragas.metrics.collections": ["Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall", "AnswerCorrectness"],
        }
        for name, attributes in exports.items():
            module = ModuleType(name)
            for attribute in attributes:
                setattr(module, attribute, Mock())
            stubs[name] = module
        module = load_module("evaluation/rag_metrics.py", stubs)
        samples = module._prep_samples({"rag_samples": [{
            "actual_response": "No evidence", "actual_contexts": [],
            "relevant_contexts": ["Expected answer evidence"],
        }]})
        self.assertEqual(samples[0]["actual_contexts"], [])


class IngestionFailureTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        qdrant = ModuleType("qdrant_client")
        self.client = Mock()
        qdrant.QdrantClient = Mock(return_value=self.client)
        http = ModuleType("qdrant_client.http")
        http.models = SimpleNamespace(**{
            name: SimpleNamespace for name in (
                "PointStruct", "FilterSelector", "Filter", "FieldCondition", "MatchValue", "HasIdCondition"
            )
        })
        settings = ModuleType("rag.settings")
        settings.settings = SimpleNamespace(QDRANT_URL="https://qdrant.example", QDRANT_API_KEY="", QDRANT_COLLECTION="rag")
        self.embeddings = ModuleType("rag.services.retrieval.embedding_service")
        self.embeddings.embed_texts = Mock(return_value=[[0.1, 0.2]])
        self.embeddings.get_embedding_dim = Mock(return_value=2)
        stubs = {"qdrant_client": qdrant, "qdrant_client.http": http,
                 settings.__name__: settings, self.embeddings.__name__: self.embeddings}
        for name, function, result in [
            ("rag.ingestion.loaders.pdf_loader", "parse_pdf", "text"),
            ("rag.ingestion.loaders.html_loader", "parse_html", "text"),
            ("rag.ingestion.loaders.text_loader", "parse_text", "text"),
            ("rag.ingestion.chunking.text_splitter", "chunk_text", ["chunk one"]),
        ]:
            module = ModuleType(name)
            setattr(module, function, Mock(return_value=result))
            stubs[name] = module
        self.module = load_module("rag/ingestion/document_processor.py", stubs)
        self.module.PROCESSED_DATA_DIR = self.folder.name

    def test_repeated_ingestion_uses_stable_point_ids(self):
        self.module.process_file("example.txt", "example.txt", "true")
        first = self.client.upsert.call_args.kwargs["points"][0].id
        self.module.process_file("example.txt", "example.txt", "true")
        second = self.client.upsert.call_args.kwargs["points"][0].id
        self.assertEqual(first, second)

    def test_embedding_count_mismatch_and_upsert_failures_propagate(self):
        self.embeddings.embed_texts.return_value = []
        with self.assertRaisesRegex(ValueError, "Embedding count"):
            self.module.process_file("example.txt", "example.txt", "true")
        self.client.upsert.assert_not_called()
        self.client.delete.assert_not_called()
        self.embeddings.embed_texts.return_value = [[0.1, 0.2]]
        self.client.upsert.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            self.module.process_file("example.txt", "example.txt", "true")
        self.client.delete.assert_not_called()

    def test_reingestion_removes_stale_and_legacy_chunks_only_for_this_source(self):
        stored = {
            "legacy": SimpleNamespace(payload={"source": "example.txt", "source_type": "true"}),
            "other-document": SimpleNamespace(payload={"source": "other.txt", "source_type": "true"}),
            "other-type": SimpleNamespace(payload={"source": "example.txt", "source_type": "noisy"}),
        }

        def upsert(*, collection_name, points, wait):
            self.assertTrue(wait)
            stored.update({point.id: point for point in points})

        def cleanup(*, collection_name, points_selector, wait):
            self.assertTrue(wait)
            query_filter = points_selector.filter
            protected = set(query_filter.must_not[0].has_id)
            conditions = {condition.key: condition.match.value for condition in query_filter.must}
            for point_id, point in list(stored.items()):
                matches = all(point.payload.get(key) == value for key, value in conditions.items())
                if matches and point_id not in protected:
                    del stored[point_id]

        self.client.upsert.side_effect = upsert
        self.client.delete.side_effect = cleanup
        self.module.chunk_text.return_value = ["first", "second", "third"]
        self.embeddings.embed_texts.return_value = [[0.1, 0.2]] * 3
        self.module.process_file("example.txt", "example.txt", "true")
        self.assertNotIn("legacy", stored)
        self.assertEqual(len(stored), 5)
        self.module.chunk_text.return_value = ["replacement"]
        self.embeddings.embed_texts.return_value = [[0.1, 0.2]]
        self.module.process_file("example.txt", "example.txt", "true")
        self.assertEqual(len(stored), 3)
        self.assertIn("other-document", stored)
        self.assertIn("other-type", stored)
        remaining = [point for point in stored.values() if point.payload.get("text")]
        self.assertEqual([point.payload["text"] for point in remaining], ["replacement"])
        self.assertEqual([call[0] for call in self.client.method_calls], ["upsert", "delete", "upsert", "delete"])

    def test_obsolete_chunk_cleanup_failure_propagates(self):
        self.client.delete.side_effect = RuntimeError("cleanup unavailable")
        with self.assertRaisesRegex(RuntimeError, "cleanup unavailable"):
            self.module.process_file("example.txt", "example.txt", "true")
        self.client.upsert.assert_called_once()

    def test_invalid_source_fails_before_any_remote_wipe(self):
        with self.assertRaisesRegex(ValueError, "not a directory"):
            self.module.run_universal_ingestion(str(Path(self.folder.name) / "missing"), wipe=True)
        self.client.collection_exists.assert_not_called()
        self.client.delete_collection.assert_not_called()

    def test_existing_collection_dimension_mismatch_fails_before_indexing(self):
        self.client.collection_exists.return_value = True
        self.client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=768))
        ))
        with self.assertRaisesRegex(ValueError, "embedding dimensions"):
            self.module.run_universal_ingestion(self.folder.name)
        self.client.upsert.assert_not_called()
        self.client.delete_collection.assert_not_called()

    def test_embedding_initialization_failure_preserves_collection_when_wiping(self):
        self.embeddings.get_embedding_dim.side_effect = RuntimeError("embedding provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "embedding provider unavailable"):
            self.module.run_universal_ingestion(self.folder.name, wipe=True)
        self.client.collection_exists.assert_not_called()
        self.client.delete_collection.assert_not_called()
        self.client.create_collection.assert_not_called()
        self.client.upsert.assert_not_called()

    def test_mixed_directory_ingests_root_files_and_subdirectories(self):
        base = Path(self.folder.name)
        (base / "root.txt").write_text("root document")
        reference = base / "true_data"
        reference.mkdir()
        (reference / "reference.txt").write_text("reference document")
        self.client.collection_exists.return_value = True
        self.client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=2))
        ))
        with patch.object(self.module, "process_file") as process_file:
            self.module.run_universal_ingestion(self.folder.name)
        self.assertEqual({(call.args[1], call.args[2]) for call in process_file.call_args_list}, {
            ("root.txt", "general"), ("reference.txt", "true"),
        })

    def test_explicit_source_type_applies_to_subdirectories(self):
        base = Path(self.folder.name)
        nested = base / "documents"
        nested.mkdir()
        (nested / "reference.txt").write_text("reference document")
        self.client.collection_exists.return_value = True
        self.client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=2))
        ))
        with patch.object(self.module, "process_file") as process_file:
            self.module.run_universal_ingestion(self.folder.name, explicit_source_type="true")
        process_file.assert_called_once_with(str(nested / "reference.txt"), "reference.txt", "true")


if __name__ == "__main__":
    unittest.main()
