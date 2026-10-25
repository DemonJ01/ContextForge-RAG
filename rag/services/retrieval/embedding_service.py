"""Stable, explicitly selected embeddings shared by ingestion and retrieval."""

import os
import time
from threading import Lock

import logfire
from rag.settings import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_LOCAL_DIM = 768  # all-mpnet-base-v2
_DEFAULT_GEMINI_MODEL = "models/gemini-embedding-2-preview"
_GEMINI_MODELS = {"gemini-embedding-2-preview", "gemini-embedding-2", "gemini-embedding-001"}

_active_model = None
_model_type: str | None = None
_model_lock = Lock()


def _validate_vectors(vectors, count, dimension):
    if len(vectors) != count or any(len(vector) != dimension for vector in vectors):
        raise RuntimeError(
            f"Embedding output must contain {count} vectors of dimension {dimension}. "
            "Use the same embedding backend and model for ingestion and retrieval."
        )
    return vectors


class _GeminiEmbeddings:
    """Text-only GenAI adapter that respects each Gemini model's API contract.

    Separate Content objects keep document embeddings separate. Passing a list
    of text parts in one Content would aggregate Embedding 2 inputs into one
    vector. Raw text is preserved so the default retains its existing space.
    """

    def __init__(self, model_name):
        # Import only when Gemini is selected. HuggingFace embeddings do not
        # need either the Google SDK or the LangChain Google integration.
        from google import genai
        from google.genai import types

        self.model_name = model_name
        self._types = types
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _embed(self, texts, task_type):
        request = {
            "model": self.model_name,
            "contents": [self._types.Content(parts=[self._types.Part.from_text(text=text)]) for text in texts],
        }
        if self.model_name.removeprefix("models/") == "gemini-embedding-001":
            request["config"] = self._types.EmbedContentConfig(task_type=task_type)
        # Embedding 2 rejects task_type. Do not introduce retrieval prompt
        # prefixes here: that would silently change existing indexed vectors.
        response = self._client.models.embed_content(**request)
        vectors = [list(embedding.values) for embedding in (response.embeddings or [])]
        return _validate_vectors(vectors, len(texts), _GEMINI_DIM)

    def embed_query(self, query):
        return self._embed([query], "RETRIEVAL_QUERY")[0]

    def embed_documents(self, texts):
        return self._embed(texts, "RETRIEVAL_DOCUMENT")


def _load_gemini():
    if not (settings.GEMINI_API_KEY or "").strip():
        raise RuntimeError("Gemini embeddings require GEMINI_API_KEY. Configure it before indexing or querying.")
    model_name = os.getenv("RAG_GEMINI_EMBEDDING_MODEL", _DEFAULT_GEMINI_MODEL).strip()
    if model_name.removeprefix("models/") not in _GEMINI_MODELS:
        raise RuntimeError(
            "RAG_GEMINI_EMBEDDING_MODEL must be gemini-embedding-2-preview, "
            "gemini-embedding-2, or gemini-embedding-001 (optionally prefixed with models/). "
            "Changing the model requires re-ingesting documents into a separate Qdrant collection."
        )
    try:
        model = _GeminiEmbeddings(model_name)
        _validate_vectors([model.embed_query("probe")], 1, _GEMINI_DIM)
        return model
    except ImportError as exc:
        raise RuntimeError(
            "Gemini embeddings require the google-genai package. Install requirements-cloud.txt; "
            "the embedding backend was not changed."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Gemini embeddings ({model_name}) could not be initialized. Check GEMINI_API_KEY, "
            "model access, and connectivity. The embedding backend was not changed. "
            "Changing RAG_GEMINI_EMBEDDING_MODEL requires re-ingesting documents into a separate Qdrant collection."
        ) from exc


def _load_local():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading local embeddings (all-mpnet-base-v2, 768-dim).")
    model = SentenceTransformer("all-mpnet-base-v2")
    if model.get_sentence_embedding_dimension() != _LOCAL_DIM:
        raise RuntimeError("The local embedding model must produce 768-dimensional vectors.")
    return model


def _init():
    """Select a backend once; failures must never change its vector space."""
    global _active_model, _model_type
    if _active_model is not None:
        return
    with _model_lock:
        if _active_model is not None:
            return
        backend = os.getenv("RAG_EMBEDDING_BACKEND", "gemini").strip().lower()
        if backend not in {"gemini", "local"}:
            raise RuntimeError("RAG_EMBEDDING_BACKEND must be either 'gemini' or 'local'.")
        model = _load_gemini() if backend == "gemini" else _load_local()
        # Publish the model last so callers using the fast path also see its backend.
        _model_type, _active_model = backend, model


def get_embedding_dim() -> int:
    """Return the dimension for the explicitly selected embedding backend."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _LOCAL_DIM


def _embed_batch(batch: list[str]) -> list[list[float]]:
    dimension = get_embedding_dim()
    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _validate_vectors(_active_model.embed_documents(batch), len(batch), dimension)
            except Exception as exc:
                err = str(exc).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(f"Gemini rate limit hit; retrying in {wait}s (attempt {attempt + 1}/4).")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        "Gemini embedding request failed. Check provider configuration and availability; "
                        "the embedding backend was not changed."
                    ) from exc
    return _validate_vectors(
        _active_model.encode(batch, show_progress_bar=False).tolist(), len(batch), dimension
    )


def embed_query(query: str) -> list[float]:
    dimension = get_embedding_dim()
    if _model_type == "gemini":
        try:
            vector = _active_model.embed_query(query)
        except Exception as exc:
            raise RuntimeError(
                "Gemini query embedding failed. Check provider configuration and availability; "
                "the embedding backend was not changed."
            ) from exc
    else:
        vector = _active_model.encode([query], show_progress_bar=False)[0].tolist()
    return _validate_vectors([vector], 1, dimension)[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
