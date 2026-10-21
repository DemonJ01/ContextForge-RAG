"""Stable, explicitly selected embeddings shared by ingestion and retrieval."""

import os
import time
from threading import Lock

import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rag.settings import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_LOCAL_DIM = 768  # all-mpnet-base-v2

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


def _load_gemini():
    if not (settings.GEMINI_API_KEY or "").strip():
        raise RuntimeError("Gemini embeddings require GEMINI_API_KEY. Configure it before indexing or querying.")
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        _validate_vectors([model.embed_query("probe")], 1, _GEMINI_DIM)
        return model
    except Exception as exc:
        raise RuntimeError(
            "Gemini embeddings could not be initialized. Check GEMINI_API_KEY, model access, "
            "and connectivity. The embedding backend was not changed."
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
