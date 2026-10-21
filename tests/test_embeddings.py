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
        google = ModuleType("langchain_google_genai")
        self.gemini = Mock()
        self.gemini.embed_query.return_value = [0.1] * 3072
        self.gemini.embed_documents.side_effect = lambda texts: [[0.1] * 3072 for _ in texts]
        google.GoogleGenerativeAIEmbeddings = self.gemini_constructor = Mock(return_value=self.gemini)
        local_sdk = ModuleType("sentence_transformers")
        self.local = Mock()
        self.local.get_sentence_embedding_dimension.return_value = 768
        local_sdk.SentenceTransformer = self.local_constructor = Mock(return_value=self.local)
        settings = ModuleType("rag.settings")
        settings.settings = SimpleNamespace(GEMINI_API_KEY="fake-test-key")
        self.settings = settings.settings
        self.modules = patch.dict(sys.modules, {
            "logfire": logfire,
            "langchain_google_genai": google,
            "sentence_transformers": local_sdk,
            "rag.settings": settings,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.environment = patch.dict(os.environ, {"RAG_EMBEDDING_BACKEND": "gemini"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        source = Path(__file__).resolve().parents[1] / "rag/services/retrieval/embedding_service.py"
        spec = importlib.util.spec_from_file_location("_rag_embeddings_under_test", source)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_gemini_failure_never_loads_another_backend(self):
        self.gemini.embed_query.side_effect = RuntimeError("provider unavailable")
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

    def test_gemini_remains_selected_after_request_failure(self):
        self.assertEqual(self.module.get_embedding_dim(), 3072)
        self.gemini.embed_query.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "backend was not changed"):
            self.module.embed_query("Kubernetes")
        self.assertEqual(self.module._model_type, "gemini")
        self.assertEqual(self.module.get_embedding_dim(), 3072)
        self.local_constructor.assert_not_called()

    def test_gemini_batches_preserve_count_and_dimension(self):
        vectors = self.module.embed_texts(["text"] * 51)
        self.assertEqual(len(vectors), 51)
        self.assertTrue(all(len(vector) == 3072 for vector in vectors))
        self.assertEqual([len(call.args[0]) for call in self.gemini.embed_documents.call_args_list], [50, 1])
        self.local_constructor.assert_not_called()

    def test_mismatched_provider_vectors_are_rejected(self):
        self.assertEqual(self.module.get_embedding_dim(), 3072)
        self.gemini.embed_documents.side_effect = None
        self.gemini.embed_documents.return_value = [[0.1] * 768]
        with self.assertRaisesRegex(RuntimeError, "backend was not changed"):
            self.module.embed_texts(["text"])
        self.assertEqual(self.module._model_type, "gemini")
        self.local_constructor.assert_not_called()

    def test_explicit_local_backend_is_used_for_queries_and_documents(self):
        os.environ["RAG_EMBEDDING_BACKEND"] = "local"
        self.settings.GEMINI_API_KEY = ""
        self.local.encode.return_value = SimpleNamespace(
            tolist=lambda: [[0.2] * 768],
        )
        self.assertEqual(self.module.get_embedding_dim(), 768)
        self.assertEqual(len(self.module.embed_texts(["text"])[0]), 768)
        self.local.encode.return_value = [SimpleNamespace(tolist=lambda: [0.2] * 768)]
        self.assertEqual(len(self.module.embed_query("query")), 768)
        self.gemini_constructor.assert_not_called()
        self.local_constructor.assert_called_once_with("all-mpnet-base-v2")
        os.environ["RAG_EMBEDDING_BACKEND"] = "gemini"
        self.assertEqual(self.module.get_embedding_dim(), 768)

    def test_unknown_backend_fails_before_model_loading(self):
        os.environ["RAG_EMBEDDING_BACKEND"] = "automatic"
        with self.assertRaisesRegex(RuntimeError, "must be either"):
            self.module.embed_query("Kubernetes")
        self.gemini_constructor.assert_not_called()
        self.local_constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
