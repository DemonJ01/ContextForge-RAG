"""Embedding backend regressions with fake SDKs and no downloads or network."""

from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class EmbeddingBackendTests(unittest.TestCase):
    def setUp(self):
        logfire = ModuleType("logfire")
        logfire.info = Mock()
        logfire.warning = Mock()
        logfire.span = Mock(side_effect=lambda *args, **kwargs: nullcontext())
        google = ModuleType("google")
        genai = ModuleType("google.genai")
        self.genai_types = SimpleNamespace(
            Content=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            Part=SimpleNamespace(from_text=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))),
            EmbedContentConfig=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
        )
        genai.types = self.genai_types
        google.genai = genai
        self.embed = Mock(side_effect=lambda **kwargs: SimpleNamespace(embeddings=[
            SimpleNamespace(values=[float(index)] * 3072) for index, _ in enumerate(kwargs["contents"])
        ]))
        self.gemini = SimpleNamespace(models=SimpleNamespace(embed_content=self.embed))
        genai.Client = self.gemini_constructor = Mock(return_value=self.gemini)
        local_sdk = ModuleType("sentence_transformers")
        self.local = Mock()
        self.local.get_sentence_embedding_dimension.return_value = 768
        local_sdk.SentenceTransformer = self.local_constructor = Mock(return_value=self.local)
        settings = ModuleType("rag.settings")
        settings.settings = SimpleNamespace(GEMINI_API_KEY="fake-test-key")
        self.settings = settings.settings
        self.modules = patch.dict(sys.modules, {
            "logfire": logfire,
            "google": google,
            "google.genai": genai,
            "sentence_transformers": local_sdk,
            "rag.settings": settings,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.environment = patch.dict(os.environ, {
            "RAG_EMBEDDING_BACKEND": "gemini",
            "RAG_GEMINI_EMBEDDING_MODEL": "models/gemini-embedding-2-preview",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.module = self.load_service()

    def load_service(self):
        source = Path(__file__).resolve().parents[1] / "rag/services/retrieval/embedding_service.py"
        spec = importlib.util.spec_from_file_location("_rag_embeddings_under_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_gemini_failure_never_loads_another_backend(self):
        self.embed.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "backend was not changed"):
            self.module.embed_query("Kubernetes")
        self.local_constructor.assert_not_called()
        self.assertIsNone(self.module._active_model)

    def test_missing_key_fails_without_local_fallback(self):
        self.settings.GEMINI_API_KEY = ""
        with self.assertRaisesRegex(RuntimeError, "require GEMINI_API_KEY"):
            self.module.get_embedding_dim()
        self.gemini_constructor.assert_not_called()
        self.local_constructor.assert_not_called()

    def test_missing_gemini_sdk_reports_dependency_without_fallback(self):
        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            with self.assertRaisesRegex(RuntimeError, "require the google-genai package"):
                self.module.get_embedding_dim()
        self.local_constructor.assert_not_called()
        self.assertIsNone(self.module._active_model)

    def test_gemini_remains_selected_after_request_failure(self):
        self.assertEqual(self.module.get_embedding_dim(), 3072)
        self.embed.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "backend was not changed"):
            self.module.embed_query("Kubernetes")
        self.assertEqual(self.module._model_type, "gemini")
        self.assertEqual(self.module.get_embedding_dim(), 3072)
        self.local_constructor.assert_not_called()

    def test_default_model_retains_identity_without_unsupported_task_type(self):
        del os.environ["RAG_GEMINI_EMBEDDING_MODEL"]
        query = "  How do CronJobs work?\n"
        self.module.embed_query(query)
        self.module.embed_texts(["document one", "document two"])
        self.gemini_constructor.assert_called_once_with(api_key="fake-test-key")
        for call in self.embed.call_args_list:
            self.assertEqual(call.kwargs["model"], "models/gemini-embedding-2-preview")
            self.assertNotIn("config", call.kwargs)
        self.genai_types.EmbedContentConfig.assert_not_called()
        query_request = self.embed.call_args_list[1].kwargs
        self.assertEqual(query_request["contents"][0].parts[0].text, query)

    def test_explicit_stable_embedding_2_omits_task_type(self):
        os.environ["RAG_GEMINI_EMBEDDING_MODEL"] = "gemini-embedding-2"
        self.module.embed_query("query")
        self.module.embed_texts(["document"])
        for call in self.embed.call_args_list:
            self.assertEqual(call.kwargs["model"], "gemini-embedding-2")
            self.assertNotIn("config", call.kwargs)
        self.genai_types.EmbedContentConfig.assert_not_called()

    def test_embedding_001_uses_correct_query_and_document_task_types(self):
        os.environ["RAG_GEMINI_EMBEDDING_MODEL"] = "models/gemini-embedding-001"
        self.module.embed_query("query")
        self.module.embed_texts(["document one", "document two"])
        self.assertEqual(
            [call.kwargs["config"].task_type for call in self.embed.call_args_list],
            ["RETRIEVAL_QUERY", "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"],
        )
        for call in self.embed.call_args_list:
            self.assertEqual(call.kwargs["model"], "models/gemini-embedding-001")
        os.environ["RAG_GEMINI_EMBEDDING_MODEL"] = "gemini-embedding-2"
        self.module.embed_query("another query")
        self.assertEqual(self.embed.call_args.kwargs["model"], "models/gemini-embedding-001")

    def test_gemini_batches_use_separate_content_objects_for_each_document(self):
        texts = [f"document {index}" for index in range(51)]
        vectors = self.module.embed_texts(texts)
        self.assertEqual(len(vectors), 51)
        self.assertTrue(all(len(vector) == 3072 for vector in vectors))
        requests = [call.kwargs["contents"] for call in self.embed.call_args_list[1:]]
        self.assertEqual([len(contents) for contents in requests], [50, 1])
        self.assertEqual([content.parts[0].text for contents in requests for content in contents], texts)
        self.assertTrue(all(len(content.parts) == 1 for contents in requests for content in contents))
        self.assertEqual([vector[0] for vector in vectors], [float(index) for index in range(50)] + [0.0])
        self.local_constructor.assert_not_called()

    def test_aggregated_provider_result_is_rejected(self):
        self.module.get_embedding_dim()
        self.embed.side_effect = None
        self.embed.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * 3072)])
        with self.assertRaisesRegex(RuntimeError, "backend was not changed"):
            self.module.embed_texts(["document one", "document two"])
        self.local_constructor.assert_not_called()

    def test_mismatched_provider_vectors_are_rejected(self):
        self.assertEqual(self.module.get_embedding_dim(), 3072)
        self.embed.side_effect = None
        self.embed.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * 768)])
        with self.assertRaisesRegex(RuntimeError, "backend was not changed"):
            self.module.embed_texts(["text"])
        self.assertEqual(self.module._model_type, "gemini")
        self.local_constructor.assert_not_called()

    def test_explicit_local_backend_works_without_any_gemini_sdk(self):
        os.environ["RAG_EMBEDDING_BACKEND"] = "local"
        self.settings.GEMINI_API_KEY = ""
        self.local.encode.return_value = SimpleNamespace(tolist=lambda: [[0.2] * 768])
        with patch.dict(sys.modules, {"google": None, "google.genai": None, "langchain_google_genai": None}):
            module = self.load_service()
            self.assertEqual(module.get_embedding_dim(), 768)
            self.assertEqual(len(module.embed_texts(["text"])[0]), 768)
            self.local.encode.return_value = [SimpleNamespace(tolist=lambda: [0.2] * 768)]
            self.assertEqual(len(module.embed_query("query")), 768)
        self.gemini_constructor.assert_not_called()
        self.local_constructor.assert_called_once_with("all-mpnet-base-v2")
        os.environ["RAG_EMBEDDING_BACKEND"] = "gemini"
        self.assertEqual(module.get_embedding_dim(), 768)

    def test_unknown_model_fails_before_initialization_and_explains_reindexing(self):
        os.environ["RAG_GEMINI_EMBEDDING_MODEL"] = "automatic"
        with self.assertRaisesRegex(RuntimeError, "re-ingesting documents into a separate Qdrant collection"):
            self.module.get_embedding_dim()
        self.gemini_constructor.assert_not_called()
        self.local_constructor.assert_not_called()

    def test_unknown_backend_fails_before_model_loading(self):
        os.environ["RAG_EMBEDDING_BACKEND"] = "automatic"
        with self.assertRaisesRegex(RuntimeError, "must be either"):
            self.module.embed_query("Kubernetes")
        self.gemini_constructor.assert_not_called()
        self.local_constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
