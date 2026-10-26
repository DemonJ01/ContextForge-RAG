"""Select the credential-free local engine or the explicitly enabled cloud flow."""
from __future__ import annotations

import importlib
import logging
import os
import re
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Optional

from dotenv import load_dotenv
from rag.cloud_status import (
    CloudIndexReadiness, CloudIndexUnavailable, configuration_status,
    embedding_configuration, required_environment_variables,
)
from rag.local_rag import normalize_question

logger = logging.getLogger(__name__)

_LOCAL_CONTEXT_LIMIT = 256
_LOCAL_CONTEXT_TTL = 30 * 60
_LOCAL_FOLLOWUP = re.compile(
    r"^\s*(?:(?:please|can you|could you)\s+)?(?:"
    # Match true subject pronouns in yes/no questions. Demonstratives such as
    # 'this' can introduce an explicit subject ('Does this HPA ...'), so leave
    # those questions to ordinary retrieval rather than substituting context.
    r"(?:do|does|did|can|could|should|will|would|is|are|was|were|has|have|had|must|may|might)\s+"
    r"(?P<yesno>it|they)\b|"
    r"(?:how|why|when|where)\s+(?:do|does|did|can|could|should|will|would|is|are)\s+"
    # A demonstrative followed by a named subject ('this HPA') must retain
    # that subject. Resolve standalone demonstratives with generic predicates.
    r"(?P<subject>it|they|(?:this|that|these|those)(?=\s+(?:"
    r"work(?!\s+queues?\b)|run|scale|fail|differ|behave|operate|function|help|handle|support|use|"
    r"affect|change|compare|relate|interact|recover|start|stop|restart|"
    r"used|configured|deployed|scheduled|managed|different)\b|\s*(?:[?!.]|$)))\b|"
    r"(?:what|which)\s+(?:is|are|was|were)\s+(?P<possessive>its|their)\b|"
    r"(?:what|which)\s+(?:is|are|was|were)\s+(?P<referent>it|they|this|that|these|those)"
    r"(?=\s*(?:[?!.]|$))|"
    r"(?:what|how)\s+about\s+(?P<about>it|its|them|their)\b|"
    r"(?:explain|describe|summarize|summarise|tell me(?: more)? about)\s+"
    r"(?P<object>it|its|them|their)\b)",
    re.IGNORECASE,
)


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
        self._cloud_readiness = CloudIndexReadiness()
        self._local_lock = Lock()
        self._cloud_lock = Lock()
        self._context_lock = Lock()
        self._local_context: OrderedDict[tuple[str, str], tuple[float, str]] = OrderedDict()

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
            embedding = embedding_configuration()
            if not embedding["valid"]:
                raise RuntimeUnavailable(
                    "RAG_EMBEDDING_BACKEND must be either 'gemini' or 'local', and "
                    "RAG_GEMINI_EMBEDDING_MODEL must select a supported model. Check /configuration."
                )
            required = required_environment_variables(embedding["backend"])
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

    def configuration(self):
        return configuration_status(self.mode)

    def health(self):
        self._check_mode()
        if self.mode == "cloud":
            self.cloud_components()
            try:
                remote = self._cloud_readiness.check()
            except CloudIndexUnavailable as exc:
                raise RuntimeUnavailable(str(exc)) from None
            bundled = dict(self.local_engine().health())
            return {
                "status": "ready", "mode": "cloud", "credentials_required": True,
                "generation": "cloud", "retrieval": "Qdrant + FlashRank",
                "bundled_document_count": bundled.get("document_count", 0),
                "bundled_reference_document_count": bundled.get("reference_document_count", 0),
                "bundled_chunk_count": bundled.get("chunk_count", 0),
                **remote,
            }
        stats = dict(self.local_engine().health())
        if stats.get("document_count", 0) < 1 or stats.get("chunk_count", 0) < 1:
            raise RuntimeUnavailable(
                "No indexed document text is available. Restore the bundled processed_documents "
                "folder or ingest documents before starting the application."
            )
        stats.update(status="ready", mode=self.mode)
        return stats

    def documents(self, scope="all"):
        return self.local_engine().documents(scope=scope)

    def document(self, document_id):
        return self.local_engine().document(document_id)

    def _local_query(self, q, *, thread_id, scope, top_k):
        engine = self.local_engine()
        # Store only the last supported question, never document text or generated
        # answers. A scope change must not reuse evidence from a different scope.
        key = (thread_id, scope) if isinstance(thread_id, str) and 0 < len(thread_id) <= 128 else None
        question = str(q).strip()
        followup = _LOCAL_FOLLOWUP.match(question)
        with self._context_lock:
            now = monotonic()
            for expired in [context_key for context_key, (timestamp, _) in self._local_context.items()
                            if now - timestamp >= _LOCAL_CONTEXT_TTL]:
                del self._local_context[expired]
            previous = self._local_context.get(key) if key else None
            resolved = question
            if followup:
                if previous is None:
                    result = engine.query("", scope=scope, top_k=top_k)
                    result["answer"] = "Please name the subject of your question so I can find supporting document excerpts."
                    result.update(status="no_answer", sources=[], citations=[])
                    result["thought_process"] = ["No supported conversation subject is available in this thread and search scope."]
                    return result
                start, end = followup.span(followup.lastgroup)
                resolved = question[:start] + "(" + previous[1] + ")" + question[end:]
            result = engine.query(resolved, scope=scope, top_k=top_k)
            if followup and previous:
                result["thought_process"] = [
                    "Resolved the follow-up using the previous supported question in this thread and search scope."
                ] + list(result.get("thought_process", []))
            if key:
                if result.get("citations") and result.get("status") != "no_answer" and len(question) <= 2000:
                    # Repeated follow-ups retain their original anchor, so the
                    # stored question and future retrieval requests cannot grow.
                    # Retain subject and intent, but remove request wrappers
                    # before embedding the anchor inside a later follow-up.
                    anchor = previous[1] if followup and previous else normalize_question(question)
                    self._local_context[key] = (now, anchor)
                    self._local_context.move_to_end(key)
                    while len(self._local_context) > _LOCAL_CONTEXT_LIMIT:
                        self._local_context.popitem(last=False)
                else:
                    # An unsupported independent question changes the topic too;
                    # a later pronoun must not silently revive an older subject.
                    self._local_context.pop(key, None)
            return result

    def query(self, q, *, thread_id, scope="reference", top_k=4):
        self._check_mode()
        if self.mode == "local":
            return self._local_query(q, thread_id=thread_id, scope=scope, top_k=top_k)
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
        # LangGraph checkpoints include conversation history and retrieved text.
        # Keep each search scope in its own history, including conversational
        # follow-ups that the planner routes around retrieval. The public API
        # continues to return the caller's original thread ID.
        checkpoint_id = "rag-cloud-" + sha256(f"{scope}\0{thread_id}".encode("utf-8")).hexdigest()
        result = agent.invoke(initial_state, config={"configurable": {"thread_id": checkpoint_id}})
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
