"""Offline diagnostics and cloud readiness checks; no provider calls or models."""
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from rag.api import create_app
from rag.cloud_status import CloudIndexReadiness, CloudIndexUnavailable, configuration_status
from rag.runtime import RAGRuntime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENT = {
    "PORTKEY_API_KEY": "private-portkey-value",
    "GROQ_API_KEY": "private-groq-value",
    "GEMINI_API_KEY": "private-gemini-value",
    "QDRANT_CLUSTER_ENDPOINT": "https://private-qdrant.example",
    "QDRANT_API_KEY": "private-qdrant-value",
}


def collection(*, size=3072, points=5, status="green"):
    return SimpleNamespace(
        status=status, points_count=points,
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=size))),
    )


class ConfigurationTests(unittest.TestCase):
    def test_endpoint_has_no_sdk_initialization_or_provider_requests_and_never_returns_secrets(self):
        with patch.dict(os.environ, ENVIRONMENT, clear=True):
            runtime = RAGRuntime(PROJECT_ROOT, mode="cloud")
            with TestClient(create_app(runtime=runtime)) as client:
                with (
                    patch.object(runtime, "cloud_components", side_effect=AssertionError("Cloud initialized")),
                    patch("rag.cloud_status.metadata.version", return_value="test-version"),
                    patch.object(socket.socket, "connect", side_effect=AssertionError("Network connection")),
                    patch("rag.cloud_status.importlib.import_module", side_effect=AssertionError("SDK imported")),
                ):
                    response = client.get("/configuration")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["configured_mode"], "cloud")
        self.assertTrue(data["cloud_configuration_ready"])
        self.assertEqual(data["provider_authentication"], "not_checked")
        self.assertEqual(data["remote_index"], "not_checked")
        self.assertEqual(data["embedding_model"], "models/gemini-embedding-2-preview")
        self.assertEqual(data["embedding_dimension"], 3072)
        for private_value in ENVIRONMENT.values():
            self.assertNotIn(private_value, response.text)

    def test_missing_variables_and_packages_are_names_with_actionable_steps(self):
        from importlib.metadata import PackageNotFoundError
        with patch("rag.cloud_status.metadata.version", side_effect=PackageNotFoundError):
            data = configuration_status("local", {})
        self.assertIn("GEMINI_API_KEY", data["required_environment_variables"])
        self.assertEqual(data["required_environment_variables"], data["missing_environment_variables"])
        self.assertIn("qdrant-client", data["missing_cloud_packages"])
        self.assertFalse(data["cloud_configuration_ready"])
        self.assertNotIn("keys are configured", data["authentication"])
        self.assertTrue(any("requirements-cloud.txt" in step for step in data["steps"]))
        self.assertTrue(any("RAG_MODE=cloud" in step for step in data["steps"]))

    def test_selector_values_are_allowlisted_and_local_embeddings_do_not_need_gemini_key(self):
        with patch("rag.cloud_status.metadata.version", return_value="test-version"):
            local = configuration_status("cloud", {"RAG_EMBEDDING_BACKEND": "local", **ENVIRONMENT})
            bad = configuration_status("private-mode", {
                "RAG_EMBEDDING_BACKEND": "gemini", "RAG_GEMINI_EMBEDDING_MODEL": "private-model-secret",
            })
            invalid_backend = configuration_status("cloud", {"RAG_EMBEDDING_BACKEND": "private-backend-secret"})
        self.assertNotIn("GEMINI_API_KEY", local["required_environment_variables"])
        self.assertEqual(local["embedding_dimension"], 768)
        self.assertEqual(local["embedding_model"], "all-mpnet-base-v2")
        self.assertNotIn("private-", json.dumps([bad, invalid_backend]))
        self.assertFalse(bad["cloud_configuration_ready"])

    def test_gemini_models_accept_optional_prefix_and_return_canonical_allowlisted_names(self):
        for model in ("gemini-embedding-2-preview", "gemini-embedding-2", "gemini-embedding-001"):
            for configured in (model, "models/" + model):
                with self.subTest(configured=configured), patch("rag.cloud_status.metadata.version", return_value="test-version"):
                    data = configuration_status("cloud", {**ENVIRONMENT, "RAG_GEMINI_EMBEDDING_MODEL": configured})
                self.assertTrue(data["cloud_configuration_ready"])
                self.assertEqual(data["embedding_model"], "models/" + model)

    def test_missing_genai_sdk_is_reported_even_when_legacy_langchain_sdk_is_installed(self):
        from importlib.metadata import PackageNotFoundError
        def installed_version(package):
            if package == "google-genai":
                raise PackageNotFoundError(package)
            return "test-version"
        with patch("rag.cloud_status.metadata.version", side_effect=installed_version) as versions:
            data = configuration_status("cloud", ENVIRONMENT)
        self.assertEqual(data["missing_cloud_packages"], ["google-genai"])
        self.assertFalse(data["cloud_configuration_ready"])
        self.assertIn("google-genai", [call.args[0] for call in versions.call_args_list])


class CloudIndexReadinessTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.get_collection.return_value = collection()
        self.factory = Mock(return_value=self.client)
        self.sdk = SimpleNamespace(QdrantClient=self.factory)
        self.sdk_patch = patch.dict("sys.modules", {"qdrant_client": self.sdk})
        self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)

    def runtime(self):
        runtime = RAGRuntime(PROJECT_ROOT, mode="cloud")
        runtime._cloud_components = (object(), lambda q: (False, None))
        runtime._local_engine = SimpleNamespace(health=lambda: {
            "document_count": 11, "reference_document_count": 6, "chunk_count": 100,
        })
        return runtime

    def test_health_checks_collection_and_keeps_remote_and_bundled_counts_distinct(self):
        with patch.dict(os.environ, {**ENVIRONMENT, "QDRANT_COLLECTION": "selected-index"}, clear=True):
            runtime = self.runtime()
            with TestClient(create_app(runtime=runtime)) as client:
                response = client.get("/health")
                cached_response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(cached_response.status_code, 200)
        data = response.json()
        self.assertEqual(data["remote_point_count"], 5)
        self.assertEqual(data["remote_vector_dimension"], 3072)
        self.assertIsNone(data["remote_document_count"])
        self.assertEqual(data["bundled_document_count"], 11)
        self.assertEqual(data["bundled_reference_document_count"], 6)
        self.assertEqual(data["bundled_chunk_count"], 100)
        self.assertNotIn("document_count", data)
        self.assertNotIn("chunk_count", data)
        self.assertEqual(data["provider_authentication"], "not_checked")
        self.client.get_collection.assert_called_once_with(collection_name="selected-index")
        self.assertEqual(self.factory.call_args.kwargs["timeout"], 3)
        self.assertFalse(self.factory.call_args.kwargs["check_compatibility"])
        self.client.close.assert_called_once()
        for private_value in ENVIRONMENT.values():
            self.assertNotIn(private_value, response.text)

    def test_empty_mismatched_unavailable_and_named_vector_collections_return_503(self):
        named = collection()
        named.config.params.vectors = {"named-vector": SimpleNamespace(size=3072)}
        for remote in (collection(points=0), collection(points=None), collection(size=768), collection(status="red"), named):
            with self.subTest(remote=remote):
                self.client.get_collection.return_value = remote
                with patch.dict(os.environ, ENVIRONMENT, clear=True):
                    with TestClient(create_app(runtime=self.runtime())) as client:
                        response = client.get("/health")
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("ready", response.json())

    def test_missing_and_unreachable_collection_errors_do_not_expose_provider_details(self):
        missing = RuntimeError("private remote response")
        missing.status_code = 404
        for failure, expected in ((missing, "does not exist"), (RuntimeError("private remote response"), "could not be checked")):
            with self.subTest(failure=failure):
                self.client.get_collection.side_effect = failure
                with patch.dict(os.environ, ENVIRONMENT, clear=True):
                    runtime = self.runtime()
                    with TestClient(create_app(runtime=runtime)) as client:
                        response = client.get("/health")
                        client.get("/health")
                self.assertEqual(response.status_code, 503)
                self.assertIn(expected, response.json()["detail"])
                self.assertNotIn("private remote response", response.text)
                for private_value in ENVIRONMENT.values():
                    self.assertNotIn(private_value, response.text)
        self.assertEqual(self.client.get_collection.call_count, 2)

    def test_cache_expires_and_local_backend_validates_768_dimensions_without_model_import(self):
        self.client.get_collection.return_value = collection(size=768)
        checker = CloudIndexReadiness(ttl_seconds=15)
        with patch("rag.cloud_status.monotonic", side_effect=[0, 5, 16, 16]):
            for _ in range(3):
                result = checker.check({**ENVIRONMENT, "RAG_EMBEDDING_BACKEND": "local"})
        self.assertEqual(result["remote_vector_dimension"], 768)
        self.assertEqual(self.client.get_collection.call_count, 2)

    def test_missing_connection_variables_fail_without_sdk_initialization(self):
        with self.assertRaises(CloudIndexUnavailable):
            CloudIndexReadiness().check({})
        self.factory.assert_not_called()

    def test_local_health_never_checks_remote_collection(self):
        runtime = RAGRuntime(PROJECT_ROOT, mode="local")
        with patch.object(runtime._cloud_readiness, "check", side_effect=AssertionError("Remote check in local mode")):
            self.assertEqual(runtime.health()["mode"], "local")
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
