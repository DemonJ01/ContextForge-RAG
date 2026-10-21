"""Credential-free retrieval over the project's saved document text.

The local mode uses BM25 ranking and quotes source excerpts. It deliberately
does not synthesize an answer with a language model or modify saved documents.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


_STOPWORDS = frozenset("""
a an and are as at be been being by can could did do does for from had has
have how i if in into is it its me my of on or our please should show some
that the their them then there these they this those to use using was we
were what when where which who why will with would you your explain tell
about describe compare comparison difference differences versus vs between
example examples give get need want find help work works
main specific detail details valid one two three four five six seven eight
nine ten summarize summary list provide supported supports
field fields value values
after before once because while until all any other such command commands setup
data information
""".split())
_ALIASES = {
    "monitoring": "monitor", "monitored": "monitor",
    "scheduling": "schedule", "scheduled": "schedule",
    "recurring": "schedule", "recurrence": "schedule",
    "queues": "queue", "workqueue": "queue",
    "autoscalers": "autoscaler", "autoscaling": "autoscale",
    "scaling": "scale", "scaled": "scale",
    "managing": "manage", "management": "manage",
    "starting": "start", "started": "start",
    "installing": "install", "installed": "install", "installation": "install",
    "checking": "check", "checks": "check", "verifying": "verify",
    "processes": "process", "processing": "process", "processed": "process",
}
_EXPANSIONS = {
    "hpa": {"horizontal", "autoscale", "replica"},
    "vpa": {"vertical", "autoscale", "resource"},
    "cronjob": {"cron", "schedule"},
    "monitor": {"status", "log"},
    "queue": {"worker", "redis"},
}
_MAX_CHUNK_CHARS = 1300
_CHUNK_OVERLAP = 160


def _tokenize(text: str) -> list[str]:
    tokens = []
    text = re.sub(r"\bset\s+up\b", "setup", text, flags=re.IGNORECASE)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in _STOPWORDS:
            continue
        token = _ALIASES.get(token, token)
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif (len(token) > 3 and token.endswith("s")
              and not token.endswith(("ss", "us"))
              and token not in {"kubernetes", "redis", "https", "this"}):
            token = token[:-1]
        if token not in _STOPWORDS and len(token) > 1:
            tokens.append(token)
    return tokens


def _split_text(text: str) -> list[str]:
    """Bound even oversized parser chunks while retaining readable boundaries."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + _MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            # Prefer a paragraph, line or sentence break in the last third.
            lower = start + 850
            choices = [text.rfind(separator, lower, end)
                       for separator in ("\n\n", "\n", ". ", " ")]
            boundary = next((point for point in choices if point >= lower), -1)
            if boundary != -1:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        next_start = max(start + 1, end - _CHUNK_OVERLAP)
        # Avoid starting in the middle of a token when overlapping chunks.
        space = text.find(" ", next_start, min(end, next_start + 50))
        start = space + 1 if space != -1 else next_start
    return chunks


@dataclass(frozen=True)
class _Chunk:
    document_id: str
    index: int
    text: str
    terms: Counter


class LocalRAG:
    """An immutable in-memory lexical index built from processed JSON files."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self._documents: dict[str, dict] = {}
        self._chunks: list[_Chunk] = []
        self._document_terms: dict[str, set[str]] = {}
        self._document_vocabulary: dict[str, set[str]] = {}
        self._frequencies: Counter = Counter()
        for path in sorted((self.project_root / "processed_documents").glob("*/*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            filename = str(data["filename"])
            source_type = str(data.get("source_type", path.parent.name))
            raw_chunks = data.get("chunks", [])
            if not isinstance(raw_chunks, list) or not all(isinstance(c, str) for c in raw_chunks):
                raise ValueError(f"Invalid text chunks in {path.name}")
            document_id = hashlib.sha256(f"{source_type}/{filename}".encode()).hexdigest()[:16]
            title = Path(filename).stem.replace("_", " ").strip().title()
            pieces = [piece for raw in raw_chunks for piece in _split_text(raw)]
            self._documents[document_id] = {
                "id": document_id,
                "title": title,
                "filename": filename,
                "source_type": source_type,
                "chunk_count": len(pieces),
                "format": Path(filename).suffix.lstrip(".").lower(),
                "content": "\n\n".join(raw_chunks),
            }
            first_line = next((line.strip() for raw in raw_chunks for line in raw.splitlines()
                               if line.strip()), "")
            # The source heading often carries useful words omitted by a
            # concise filename (for example, 'Jobs and CronJobs').
            self._document_terms[document_id] = set(_tokenize(title + " " + first_line))
            self._document_vocabulary[document_id] = set(_tokenize("\n".join(raw_chunks)))
            for index, text in enumerate(pieces):
                terms = Counter(_tokenize(text))
                self._chunks.append(_Chunk(document_id, index, text, terms))
                self._frequencies.update(terms.keys())
        self._average_length = (
            sum(sum(chunk.terms.values()) for chunk in self._chunks) / len(self._chunks)
            if self._chunks else 1.0
        )

    def health(self) -> dict:
        reference_count = sum(d["source_type"] == "true" for d in self._documents.values())
        return {
            "document_count": len(self._documents),
            "chunk_count": len(self._chunks),
            "reference_document_count": reference_count,
            "reference_count": reference_count,
            "noisy_document_count": len(self._documents) - reference_count,
            "retrieval": "BM25",
            "generation": "extractive",
            "credentials_required": False,
        }

    def _in_scope(self, document: dict, scope: str) -> bool:
        if scope not in {"reference", "all", "noisy"}:
            raise ValueError("scope must be reference, all, or noisy")
        return scope == "all" or document["source_type"] == ("true" if scope == "reference" else "noisy")

    def documents(self, scope: str = "all") -> list[dict]:
        return [{k: v for k, v in d.items() if k != "content"}
                for d in self._documents.values() if self._in_scope(d, scope)]

    def document(self, document_id: str) -> dict | None:
        document = self._documents.get(document_id)
        return dict(document) if document else None

    def _idf(self, token: str) -> float:
        count = self._frequencies.get(token, 0)
        return math.log(1 + (len(self._chunks) - count + 0.5) / (count + 0.5))

    def _score(self, chunk: _Chunk, terms: set[str], expanded: set[str]) -> float:
        length = sum(chunk.terms.values())
        score = 0.0
        for token in terms | expanded:
            frequency = chunk.terms.get(token, 0)
            if frequency:
                denominator = frequency + 1.5 * (0.25 + 0.75 * length / self._average_length)
                weight = 1.0 if token in terms else 0.22
                score += weight * self._idf(token) * frequency * 2.5 / denominator
        # A modest title boost disambiguates job monitoring from generic Jobs.
        score += sum(self._idf(t) for t in terms & self._document_terms[chunk.document_id])
        # Prefer explanatory headings when the source explicitly defines a
        # queried concept, rather than repeated mentions in an introduction.
        definition_terms = set()
        for heading in re.findall(r"(?im)^what (?:is|are) ([^?\n]{1,80})", chunk.text):
            definition_terms.update(_tokenize(heading))
        score += 1.2 * sum(self._idf(t) for t in terms & definition_terms)
        return score

    def query(self, q: str, scope: str = "reference", top_k: int = 4) -> dict:
        q = str(q).strip()
        top_k = max(1, min(8, int(top_k)))
        documents = self.documents(scope)
        allowed = {d["id"] for d in documents}
        terms = set(_tokenize(q))
        expanded = set().union(*(_EXPANSIONS.get(t, set()) for t in terms)) if terms else set()
        candidates = [c for c in self._chunks if c.document_id in allowed]
        vocabulary = set().union(*(c.terms.keys() for c in candidates)) if candidates else set()
        known = terms & vocabulary
        trace = [f"Searched {len(documents)} indexed documents using local BM25 retrieval."]
        no_answer = {
            "answer": "I couldn't find enough evidence in the indexed documents to answer this question. Try a question about Kubernetes Jobs, CronJobs, autoscaling, monitoring, work queues, or Databricks jobs.",
            "status": "no_answer", "thought_process": trace + ["No sufficiently matching source text was found."],
            "sources": [], "citations": [],
        }
        # Unknown subject words must not be silently discarded in favor of
        # generic overlaps. Intent words and inflections are normalized above;
        # local mode conservatively declines unfamiliar content vocabulary.
        if not terms or known != terms:
            return no_answer
        term_weight = sum(self._idf(t) for t in known)
        supported_documents = {
            document_id for document_id in allowed
            if sum(self._idf(t) for t in known & self._document_vocabulary[document_id])
            / max(term_weight, 1e-9) >= 0.8
        }
        ranked = sorted(
            ((self._score(c, known, expanded), c) for c in candidates
             if c.document_id in supported_documents
             and sum(self._idf(t) for t in known & c.terms.keys()) / max(term_weight, 1e-9) >= 0.28),
            key=lambda item: (-item[0], item[1].document_id, item[1].index),
        )
        if not ranked:
            return no_answer
        # Require one bounded passage, with its actual source heading, to
        # anchor the question's subject. Scattered matches in unrelated parts
        # of a document are not evidence of a relationship between the words.
        if not any(known <= (set(chunk.terms) | self._document_terms[chunk.document_id])
                   for _, chunk in ranked):
            return no_answer
        selected: list[tuple[float, _Chunk]] = []
        covered: set[str] = set()
        pool = ranked[:max(40, top_k * 8)]
        while pool and len(selected) < top_k:
            def selection_score(item):
                score, chunk = item
                overlap = max((len(set(chunk.terms) & set(other.terms)) /
                               max(1, len(set(chunk.terms) | set(other.terms)))
                               for _, other in selected), default=0.0)
                new_terms = (known & chunk.terms.keys()) - covered
                return score * (1 - 0.55 * overlap) + 1.5 * sum(self._idf(t) for t in new_terms)
            best = max(pool, key=selection_score)
            pool.remove(best)
            if selected and best[0] < ranked[0][0] * 0.48:
                continue
            if any(best[1].text == previous.text for _, previous in selected):
                continue
            selected.append(best)
            covered.update(known & best[1].terms.keys())
        if not selected:
            return no_answer
        citations = []
        for number, (score, chunk) in enumerate(selected, 1):
            document = self._documents[chunk.document_id]
            citations.append({
                "id": f"{chunk.document_id}-{chunk.index}",
                "document_id": chunk.document_id,
                "title": document["title"], "filename": document["filename"],
                "excerpt": chunk.text,
                "score": round(score, 4), "chunk_index": chunk.index,
                "source_type": document["source_type"],
            })
        return {
            "answer": "Source excerpts from the indexed documents:\n\n" + "\n\n".join(
                f"{citation['excerpt']} [{number}]" for number, citation in enumerate(citations, 1)),
            "status": "complete",
            "thought_process": trace + [f"Selected {len(citations)} matching excerpts. The answer quotes saved document text; no language model was used."],
            "sources": [c["excerpt"] for c in citations],
            "citations": citations,
        }
