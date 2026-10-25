"""Cloud client startup contracts using SDK stubs; no provider requests."""
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rag.cloud_status import CloudIndexReadiness


ROOT = Path(__file__).resolve().parents[1]


def load_module(path, stubs=None):
    spec = importlib.util.spec_from_file_location("_cloud_startup_under_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs or {}):
        spec.loader.exec_module(module)
    return module


def load_settings(environment):
    with patch.dict(os.environ, environment, clear=True), patch("dotenv.load_dotenv"):
        return load_module("rag/settings.py")


def cloud_stubs(settings_module):
    logfire = ModuleType("logfire")
    for name in ("configure", "info", "warning", "error"):
        setattr(logfire, name, Mock())
    logfire.span = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    client = Mock()
    client.query_points.return_value.points = []
    client.get_collection.return_value = SimpleNamespace(
        status="green", points_count=2,
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=3072))),
    )
    qdrant = ModuleType("qdrant_client")
    qdrant.QdrantClient = Mock(return_value=client)
    http = ModuleType("qdrant_client.http")
    http.models = SimpleNamespace(Filter=SimpleNamespace, FieldCondition=SimpleNamespace, MatchValue=SimpleNamespace)
    embeddings = ModuleType("rag.services.retrieval.embedding_service")
    embeddings.embed_query = Mock(return_value=[0.1] * 3072)
    embeddings.embed_texts = Mock()
    embeddings.get_embedding_dim = Mock(return_value=3072)
    stubs = {
        "logfire": logfire, "qdrant_client": qdrant, "qdrant_client.http": http,
        "rag.settings": settings_module, embeddings.__name__: embeddings,
    }
    return stubs, qdrant.QdrantClient, client


class CollectionConfigurationTests(unittest.TestCase):
    def test_collection_setting_strips_whitespace_and_defaults_blank_values(self):
        for value, expected in ((" rag-team ", "rag-team"), (" \t\n", "rag"), ("", "rag"), (None, "rag")):
            with self.subTest(value=value):
                environment = {} if value is None else {"QDRANT_COLLECTION": value}
                self.assertEqual(load_settings(environment).settings.QDRANT_COLLECTION, expected)

    def test_health_and_queries_select_the_same_normalized_collection(self):
        environment = {
            "QDRANT_COLLECTION": " rag-team ",
            "QDRANT_CLUSTER_ENDPOINT": "https://private-qdrant.example",
            "QDRANT_API_KEY": "private-qdrant-key",
        }
        settings_module = load_settings(environment)
        stubs, _, client = cloud_stubs(settings_module)
        retrieval = load_module("rag/services/retrieval/vector_store.py", stubs)
        retrieval.search_rag_knowledge("CronJobs", scope="all")
        query_collection = client.query_points.call_args.kwargs["collection_name"]
        with patch.dict(sys.modules, stubs):
            CloudIndexReadiness().check(environment)
        health_collection = client.get_collection.call_args.kwargs["collection_name"]
        self.assertEqual(query_collection, "rag-team")
        self.assertEqual(health_collection, query_collection)

    def test_ingestion_uses_the_same_normalized_collection_setting(self):
        settings_module = load_settings({"QDRANT_COLLECTION": " rag-team "})
        stubs, _, client = cloud_stubs(settings_module)
        client.collection_exists.return_value = True
        for name, attribute in (
            ("rag.ingestion.loaders.pdf_loader", "parse_pdf"),
            ("rag.ingestion.loaders.html_loader", "parse_html"),
            ("rag.ingestion.loaders.text_loader", "parse_text"),
            ("rag.ingestion.chunking.text_splitter", "chunk_text"),
        ):
            module = ModuleType(name)
            setattr(module, attribute, Mock())
            stubs[name] = module
        ingestion = load_module("rag/ingestion/document_processor.py", stubs)
        with tempfile.TemporaryDirectory() as directory, patch.object(ingestion, "process_directory"):
            ingestion.run_universal_ingestion(directory, explicit_source_type="true")
        client.collection_exists.assert_called_once_with("rag-team")
        client.get_collection.assert_called_once_with("rag-team")


class RetrievalStartupTests(unittest.TestCase):
    def test_client_startup_disables_compatibility_probe_and_bounds_request_timeout(self):
        settings_module = load_settings({})
        stubs, factory, _ = cloud_stubs(settings_module)
        compatibility_probe = Mock(side_effect=AssertionError("Startup network probe"))
        def initialize_client(**options):
            if options.get("check_compatibility", True):
                compatibility_probe()
            if not 0 < options.get("timeout", float("inf")) <= 3:
                raise AssertionError("Retrieval requests must have a bounded timeout")
            return Mock()
        factory.side_effect = initialize_client
        load_module("rag/services/retrieval/vector_store.py", stubs)
        compatibility_probe.assert_not_called()

    def test_provider_timeout_does_not_reflect_endpoint_or_key_in_public_error(self):
        environment = {
            "QDRANT_CLUSTER_ENDPOINT": "https://private-qdrant.example",
            "QDRANT_API_KEY": "private-qdrant-key", "QDRANT_COLLECTION": " rag-team ",
        }
        stubs, _, client = cloud_stubs(load_settings(environment))
        client.query_points.side_effect = TimeoutError("https://private-qdrant.example private-qdrant-key")
        retrieval = load_module("rag/services/retrieval/vector_store.py", stubs)
        with self.assertRaisesRegex(RuntimeError, "Cloud document search failed") as raised:
            retrieval.search_rag_knowledge("CronJobs", scope="all")
        self.assertNotIn(environment["QDRANT_CLUSTER_ENDPOINT"], str(raised.exception))
        self.assertNotIn(environment["QDRANT_API_KEY"], str(raised.exception))
        self.assertEqual(client.query_points.call_args.kwargs["collection_name"], "rag-team")


if __name__ == "__main__":
    unittest.main()
