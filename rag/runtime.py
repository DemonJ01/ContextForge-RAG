"""Select the credential-free local engine or the explicitly enabled cloud flow."""
from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class RuntimeUnavailable(RuntimeError):
    """An actionable configuration/dependency error safe to return to clients."""


class RAGRuntime:
    def __init__(self, project_root: Path, mode: Optional[str] = None):
        self.project_root = Path(project_root).resolve()
        if mode is None:
            # Shell-provided values take precedence; resolve the file from the project,
            # independent of the directory Uvicorn was launched from.
            load_dotenv(self.project_root / ".env", override=False)
        self.mode = (mode if mode is not None else os.getenv("RAG_MODE", "local")).strip().lower()
        self._local_engine = None
        self._cloud_components = None
        self._local_lock = Lock()
        self._cloud_lock = Lock()

    def _check_mode(self):
        if self.mode not in {"local", "cloud"}:
            raise RuntimeUnavailable("RAG_MODE must be either 'local' or 'cloud'.")

    def local_engine(self):
        """The bundled document library also powers document browsing in cloud mode."""
        self._check_mode()
        if self._local_engine is None:
            with self._local_lock:
                if self._local_engine is None:
                    module = importlib.import_module("rag.local_rag")
                    self._local_engine = module.LocalRAG(self.project_root)
        return self._local_engine

    def cloud_components(self):
        self._check_mode()
        if self._cloud_components is not None:
            return self._cloud_components
        with self._cloud_lock:
            if self._cloud_components is not None:
                return self._cloud_components
            embedding_backend = os.getenv("RAG_EMBEDDING_BACKEND", "gemini").strip().lower()
            if embedding_backend not in {"gemini", "local"}:
                raise RuntimeUnavailable("RAG_EMBEDDING_BACKEND must be either 'gemini' or 'local'.")
            required = (
                "PORTKEY_API_KEY", "GROQ_API_KEY",
                "QDRANT_CLUSTER_ENDPOINT", "QDRANT_API_KEY",
            )
            if embedding_backend == "gemini":
                required += ("GEMINI_API_KEY",)
            missing = [name for name in required if not os.getenv(name, "").strip()]
            if missing:
                raise RuntimeUnavailable(
                    "Cloud mode requires configured environment variables: " + ", ".join(missing)
                    + ". Set RAG_MODE=local to use the bundled documents without credentials."
                )
            try:
                # Keep optional SDKs and their startup work entirely outside local mode.
                logfire = importlib.import_module("logfire")
                token = os.getenv("LOGFIRE_TOKEN")
                if token:
                    logfire.configure(token=token)
                else:
                    logfire.configure(send_to_logfire=False)
                workflow = importlib.import_module("rag.agents.workflow")
                safety = importlib.import_module("rag.guardrails")
                safety.initialize_rails()
                self._cloud_components = (workflow.rag_agent, safety.guard)
            except ImportError as exc:
                missing_module = getattr(exc, "name", None) or "an optional cloud package"
                raise RuntimeUnavailable(
                    "Cloud dependencies are unavailable (" + missing_module + "). "
                    "Install requirements-cloud.txt and configure the provider environment."
                ) from exc
            except Exception as exc:
                logger.exception("Cloud initialization failed")
                raise RuntimeUnavailable(
                    "Cloud initialization failed. Check provider configuration and server logs."
                ) from exc
            return self._cloud_components

    def health(self):
        self._check_mode()
        if self.mode == "cloud":
            self.cloud_components()
        stats = dict(self.local_engine().health())
        if stats.get("document_count", 0) < 1 or stats.get("chunk_count", 0) < 1:
            raise RuntimeUnavailable(
                "No indexed document text is available. Restore the bundled processed_documents "
                "folder or ingest documents before starting the application."
            )
        stats.update(status="ready", mode=self.mode)
        if self.mode == "cloud":
            stats.update(credentials_required=True, generation="cloud", retrieval="Qdrant + FlashRank")
        return stats

    def documents(self, scope="all"):
        return self.local_engine().documents(scope=scope)

    def document(self, document_id):
        return self.local_engine().document(document_id)

    def query(self, q, *, thread_id, scope="reference", top_k=4):
        self._check_mode()
        if self.mode == "local":
            return self.local_engine().query(q, scope=scope, top_k=top_k)
        agent, guard = self.cloud_components()
        rail_fired, rail_response = guard(q)
        if rail_fired:
            return {
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "citations": [],
            }
        initial_state = {
            "messages": [{"role": "user", "content": q}],
            "current_query": q,
            "scope": scope,
            "top_k": top_k,
            "documents": [],
            "plan": ["Start"],
            "status": "Initializing Graph...",
        }
        result = agent.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
        raw_documents = result.get("documents")
        sources = []
        for document in raw_documents if isinstance(raw_documents, (list, tuple)) else []:
            if not isinstance(document, str):
                continue
            text = document.strip()
            if text.startswith("CONTENT:"):
                text = text[len("CONTENT:"):].strip()
            if text:
                sources.append(document)
        trace = result.get("plan", [])
        if result.get("current_query") != "CONVERSATIONAL" and not sources:
            # Retrieval failures in the legacy flow can be represented as an empty
            # list. Never publish a technical synthesis without supporting text.
            return {
                "answer": "I couldn't find enough source evidence to answer this question. Please try another question or check the cloud document index.",
                "thought_process": trace + ["Insufficient source evidence. Technical answer withheld."],
                "status": "no_answer",
                "sources": [],
                "citations": [],
            }
        answer = result.get("final_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeUnavailable(
                "The cloud model returned an empty answer. Please retry or check the provider status."
            )
        return {
            "answer": answer,
            "thought_process": trace,
            "status": result.get("status", "Response generated."),
            "sources": sources,
            "citations": [],
        }

    def graph(self):
        self._check_mode()
        if self.mode == "cloud":
            agent, _ = self.cloud_components()
            # Return source directly; rendering PNG would require a remote diagram service.
            return {"mode": "cloud", "format": "mermaid", "diagram": agent.get_graph().draw_mermaid()}
        return {
            "mode": "local", "format": "mermaid",
            "diagram": "graph LR\nQuestion --> BM25[Rank document chunks]\nBM25 --> Evidence[Select supporting excerpts]\nEvidence --> Answer[Answer with citations]",
        }
