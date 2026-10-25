"""Secret-free cloud setup diagnostics and bounded remote-index readiness checks.

Configuration inspection uses distribution metadata only. It never imports a
provider SDK, initializes an embedding model, or sends a provider request.
"""
from __future__ import annotations

import importlib
from importlib import metadata
import os
from threading import Lock
from time import monotonic


GEMINI_MODELS = (
    "models/gemini-embedding-2-preview",
    "models/gemini-embedding-2",
    "models/gemini-embedding-001",
)
_REQUIRED_ENV = ("PORTKEY_API_KEY", "GROQ_API_KEY", "QDRANT_CLUSTER_ENDPOINT", "QDRANT_API_KEY")
_CLOUD_PACKAGES = (
    "qdrant-client", "langgraph", "langchain-openai", "langchain-groq",
    "portkey-ai", "nemoguardrails", "flashrank", "logfire",
)


def embedding_configuration(environment=None):
    environment = os.environ if environment is None else environment
    backend = environment.get("RAG_EMBEDDING_BACKEND", "gemini").strip().lower()
    model = environment.get("RAG_GEMINI_EMBEDDING_MODEL", GEMINI_MODELS[0]).strip()
    model = "models/" + model.removeprefix("models/")
    if backend == "local":
        return {"backend": "local", "model": "all-mpnet-base-v2", "dimension": 768, "valid": True}
    if backend == "gemini":
        valid = model in GEMINI_MODELS
        # Return allowlisted values only: a mistaken secret in a selector must
        # never be reflected by the public configuration endpoint.
        return {"backend": "gemini", "model": model if valid else "unsupported", "dimension": 3072, "valid": valid}
    return {"backend": "unsupported", "model": "unsupported", "dimension": None, "valid": False}


def required_environment_variables(backend):
    return list(_REQUIRED_ENV + (("GEMINI_API_KEY",) if backend == "gemini" else ()))


def configuration_status(mode, environment=None):
    """Describe setup without returning values from credentials or endpoints."""
    environment = os.environ if environment is None else environment
    embedding = embedding_configuration(environment)
    required = required_environment_variables(embedding["backend"])
    missing = [name for name in required if not environment.get(name, "").strip()]
    packages = _CLOUD_PACKAGES + (("sentence-transformers",) if embedding["backend"] == "local" else ("google-genai",))
    missing_packages = []
    for package in packages:
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            missing_packages.append(package)
    steps = []
    configured_mode = mode if mode in {"local", "cloud"} else "unsupported"
    if configured_mode == "unsupported":
        steps.append("Set RAG_MODE to local or cloud, then restart the server.")
    if not embedding["valid"]:
        steps.append("Set RAG_EMBEDDING_BACKEND to gemini or local and select a supported RAG_GEMINI_EMBEDDING_MODEL.")
    if missing:
        steps.append("Configure the missing environment variables on the server, then restart it.")
    if missing_packages:
        steps.append("Install the optional cloud dependencies from requirements-cloud.txt in the server environment.")
    if configured_mode == "local":
        steps.append("To use cloud answers, set RAG_MODE=cloud on the server and restart it after completing cloud setup.")
    steps.append("Ingest documents into the selected Qdrant collection using the same embedding backend and model as queries.")
    steps.append("Check /health for remote index readiness; a successful cloud query is needed to verify model-provider access.")
    return {
        "configured_mode": configured_mode,
        "embedding_backend": embedding["backend"],
        "embedding_model": embedding["model"],
        "embedding_dimension": embedding["dimension"],
        "required_environment_variables": required,
        "missing_environment_variables": missing,
        "missing_cloud_packages": missing_packages,
        "cloud_configuration_ready": embedding["valid"] and not missing and not missing_packages,
        "provider_authentication": "not_checked",
        "remote_index": "not_checked",
        "authentication": "Provider API keys are configured on the server. This app has no user login.",
        "steps": steps,
    }


class CloudIndexUnavailable(RuntimeError):
    """A safe diagnostic that contains no provider response or endpoint value."""


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class CloudIndexReadiness:
    """Cache collection metadata briefly; never load or call an embedding model."""

    def __init__(self, ttl_seconds=15, timeout_seconds=3):
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._lock = Lock()
        self._cached = None

    def check(self, environment=None):
        environment = os.environ if environment is None else environment
        embedding = embedding_configuration(environment)
        if not embedding["valid"]:
            raise CloudIndexUnavailable("The configured embedding backend or model is unsupported. Check /configuration.")
        fingerprint = (
            environment.get("QDRANT_CLUSTER_ENDPOINT", ""), environment.get("QDRANT_API_KEY", ""),
            environment.get("QDRANT_COLLECTION", "").strip() or "rag", embedding["backend"], embedding["model"],
        )
        with self._lock:
            if self._cached is not None:
                key, checked_at, result, error = self._cached
                if key == fingerprint and monotonic() - checked_at < self.ttl_seconds:
                    if error:
                        raise CloudIndexUnavailable(error)
                    return dict(result)
            try:
                result = self._read_collection(fingerprint, embedding["dimension"])
            except CloudIndexUnavailable as exc:
                self._cached = (fingerprint, monotonic(), None, str(exc))
                raise
            self._cached = (fingerprint, monotonic(), result, None)
            return dict(result)

    def _read_collection(self, fingerprint, expected_dimension):
        endpoint, api_key, collection_name, _, _ = fingerprint
        if not endpoint.strip() or not api_key.strip():
            raise CloudIndexUnavailable("Qdrant connection variables are missing. Check /configuration.")
        client = None
        try:
            sdk = importlib.import_module("qdrant_client")
            client = sdk.QdrantClient(
                url=endpoint, api_key=api_key, timeout=self.timeout_seconds, check_compatibility=False,
            )
            collection = client.get_collection(collection_name=collection_name)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 404:
                message = "The selected Qdrant collection does not exist. Ingest documents before using cloud search."
            else:
                message = "The selected Qdrant collection could not be checked. Verify server credentials, connectivity, and collection access."
            raise CloudIndexUnavailable(message) from None
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        status = _field(collection, "status")
        status = getattr(status, "value", status)
        if status not in {"green", "yellow"}:
            raise CloudIndexUnavailable("The selected Qdrant collection is not available for search. Check its health before retrying.")
        vectors = _field(_field(_field(collection, "config"), "params"), "vectors")
        dimension = _field(vectors, "size")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension != expected_dimension:
            raise CloudIndexUnavailable(
                "The selected Qdrant collection does not match the configured embedding dimensions or unnamed-vector layout. "
                "Use the same embedding backend and model for ingestion and queries."
            )
        point_count = _field(collection, "points_count")
        if not isinstance(point_count, int) or isinstance(point_count, bool) or point_count < 1:
            raise CloudIndexUnavailable("The selected Qdrant collection has no indexed points. Ingest documents before using cloud search.")
        return {
            "remote_collection_status": "ready",
            "remote_point_count": point_count,
            "remote_vector_dimension": dimension,
            "remote_document_count": None,
            "provider_authentication": "not_checked",
            "remote_index_check_ttl_seconds": self.ttl_seconds,
        }
