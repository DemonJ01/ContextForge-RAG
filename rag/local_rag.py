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
about describe compare comparison difference differences differ differs different versus vs between
example examples give get need want find help work works briefly clarify
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
    "failing": "fail", "failed": "fail", "failure": "fail", "failures": "fail",
    "troubleshooting": "troubleshoot", "debug": "troubleshoot", "debugging": "troubleshoot",
    "running": "run", "handling": "handle", "statuses": "status",
    "configuring": "configure", "configured": "configure", "configuration": "configure",
    "creating": "create", "created": "create", "creation": "create",
    "viewing": "view", "following": "follow", "retried": "retry", "retrying": "retry",
    "completed": "complete", "completing": "complete", "completion": "complete",
    "finished": "complete", "finishing": "complete",
    "k8s": "kubernetes",
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
_DEFINITION_HEADING = re.compile(r"(?im)^what (?:is|are) ([^?\n]{1,80})")
_DEFINITION_SENTENCE = re.compile(
    r"(?im)^(?:a |an |the )?([a-z0-9][a-z0-9 ()/'_-]{0,79}?)\s+(?:is|are)\s+")
_DEFINITION_DASH = re.compile(
    r"(?im)^([a-z0-9][a-z0-9 ()/'_-]{0,79}?)\s*[-–—]\s+\S")
_SUBJECT_HEADING = re.compile(
    r"(?m)^([A-Za-z][A-Za-z0-9 /_-]{0,79})[ \t]*\n([^\n]+)")


def _tokenize(text: str, *, split_identifiers: bool = False) -> list[str]:
    tokens = []
    text = re.sub(r"\bset\s+up\b", "setup", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:every\s+day|once\s+(?:a|per)\s+day)\b", "daily", text, flags=re.IGNORECASE)
    # Apply the same spelling and acronym normalization to source text and
    # queries. Full names must still identify the specific autoscaler.
    for pattern, replacement in (
        (r"\bcron\s*jobs?\b", "cronjob"),
        (r"\bhorizontal\s+pod\s+autoscal(?:ing|ers?)\b", "hpa"),
        (r"\bvertical\s+pod\s+autoscal(?:ing|ers?)\b", "vpa"),
        (r"\btime\s+zone\b", "timezone"),
        (r"\bcommand[ -]+line\b", "cli"),
    ):
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Keep exact API field names searchable and also index their component
    # words, so 'concurrency policies' can find 'concurrencyPolicy: Forbid'.
    words = []
    for word in re.findall(r"[A-Za-z0-9]+", text):
        words.append(word.lower())
        parts = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", word).lower().split()
        if split_identifiers and len(parts) > 1:
            words.extend(parts)
    for token in words:
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


def normalize_question(question: str) -> str:
    """Normalize request wrappers while retaining its subject and intent."""
    subject = re.sub(r"^\s*(?:(?:can|could|would|will) you\s+)?(?:please\s+)?", "", question,
                     flags=re.IGNORECASE)
    subject = re.sub(r"^(?:walk|talk) me through\b", "explain", subject, flags=re.IGNORECASE)
    subject = re.sub(
        r"^(?:give me\s+)?(?:a\s+)?(?:(?:quick|brief|short|simple)\s+)?"
        r"(?:overview|summary|explanation|introduction)\s+(?:of|to|about)\b",
        "explain", subject, flags=re.IGNORECASE,
    )
    # These are definition requests, with the same subject as 'what is X'.
    subject = re.sub(r"^what does (.+?) (?:stand for|mean|do)\s*[?!.]*$", r"what is \1",
                     subject, flags=re.IGNORECASE)
    subject = re.sub(r"^what is the purpose of\b", "what is", subject, flags=re.IGNORECASE)
    subject = re.sub(
        r"\bin (?:(?:simple|plain) (?:terms|words|language|english)|layman(?:['’]s)? terms)\b",
        "", subject, flags=re.IGNORECASE,
    )
    return subject.strip()


def _query_subject(question: str) -> str:
    """Extract search terms without discarding unfamiliar subject words."""
    subject = normalize_question(question)
    # A quantity question about retries still needs 'retry', the job and any
    # failure qualifiers; the generic 'many times' should not become evidence.
    subject = re.sub(r"^how many (?:times|retries)\b",
                     lambda match: "retry" if "retries" in match[0].lower() else "",
                     subject, flags=re.IGNORECASE)
    subject = re.sub(r"\bwhat happens?\b", "", subject, flags=re.IGNORECASE)
    if re.match(r"\s*how\b", subject, flags=re.IGNORECASE):
        operation = re.fullmatch(
            r"\s*how\s+(?:do|does|did|can|could|will|would)\s+"
            r"(?!i\b|you\b|we\b)(.+?)\s+(?:function|operate)\s*[?!.]*\s*",
            subject, flags=re.IGNORECASE,
        )
        if operation and not re.search(r"\b(?:a|an|the|sql)\s*$", operation[1], re.IGNORECASE):
            subject = operation[1]
        subject = re.sub(
            r"\b(?:run|running|configure|configuring|configured|set|setting|handle|handling)\b",
            "", subject, flags=re.IGNORECASE,
        )
    return subject


def _one_edit_apart(left: str, right: str) -> bool:
    """Accept one insertion, deletion, substitution or adjacent transposition."""
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        mismatches = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
        return (len(mismatches) == 1 or
                len(mismatches) == 2 and mismatches[1] == mismatches[0] + 1
                and left[mismatches[0]] == right[mismatches[1]]
                and left[mismatches[1]] == right[mismatches[0]])
    shorter, longer = sorted((left, right), key=len)
    index = next((i for i, (a, b) in enumerate(zip(shorter, longer)) if a != b),
                 len(shorter))
    return shorter[index:] == longer[index + 1:]


def _definition_start(text: str, terms: set[str], *, headings_only: bool = False) -> int | None:
    """Locate a heading that defines the whole subject, allowing its platform name."""
    patterns = ((_DEFINITION_HEADING,) if headings_only else
                (_DEFINITION_HEADING, _DEFINITION_SENTENCE, _DEFINITION_DASH))
    for pattern in patterns:
        for match in pattern.finditer(text):
            heading_terms = _heading_terms(match.group(1))
            if pattern is _DEFINITION_DASH and not heading_terms <= terms | {"kubernetes", "databricks"}:
                continue
            # Parser chunks can end with a heading whose definition starts in
            # the next chunk. An isolated heading is not a useful definition.
            if (text[match.end():].strip("? \t\r\n") and terms & heading_terms
                    and terms <= heading_terms | {"kubernetes", "databricks"}):
                return match.start()
    for match in _SUBJECT_HEADING.finditer(text):
        heading_terms = _heading_terms(match[1])
        # A short named heading followed by a full explanatory sentence is
        # useful evidence; an agenda or a list of component names is not.
        if (terms & heading_terms and terms <= heading_terms | {"kubernetes", "databricks"}
                and heading_terms <= terms | {"kube"}
                and len(re.findall(r"[A-Za-z0-9]+", match[2])) >= 9
                and re.match(r"(?:acts?\b|manages?\b|ensures?\b|provides?\b|"
                             r"automatically\b|[A-Za-z0-9_-]+\s+(?:is|are|monitors?)\b)",
                             match[2].strip(), re.IGNORECASE)):
            return match.start()
    return None


def _heading_terms(text: str) -> set[str]:
    terms = set(_tokenize(text, split_identifiers=True))
    for word in re.findall(r"[A-Za-z0-9]+", text):
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", word)
        if spaced != word:
            parts = set(_tokenize(spaced))
            if parts and parts != {word.lower()}:
                terms.difference_update(_tokenize(word, split_identifiers=True))
                terms.update(parts)
    return terms


def _field_value_matches(text: str, field_terms: set[str], value: str) -> bool:
    """Find a value attached to its field, including a short prose value list."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        line_terms = set(_tokenize(line, split_identifiers=True))
        if not field_terms <= line_terms:
            continue
        if value in line_terms:
            return True
        # A populated YAML assignment belongs only to its own value. A named
        # heading can introduce explanatory values on the following lines.
        if (re.search(r":\s*\S", line) or len(line_terms) > 8):
            continue
        for following in lines[index + 1:index + 8]:
            if not following.strip() or re.match(r"\s*\d+\.\s", following):
                break
            if value in _tokenize(following, split_identifiers=True):
                return True
    return False


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
        identifier_parts: dict[str, set[tuple[str, ...]]] = {}
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
            self._document_vocabulary[document_id] = set(_tokenize(
                "\n".join(raw_chunks), split_identifiers=True))
            for word in re.findall(r"[A-Za-z0-9]+", "\n".join(raw_chunks)):
                spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", word)
                if spaced != word:
                    parts = tuple(_tokenize(spaced))
                    if (parts and parts != (word.lower(),)
                            and (len(parts) > 1 or parts[0] in {"hpa", "vpa"})):
                        identifier_parts.setdefault(word.lower(), set()).add(parts)
            for index, text in enumerate(pieces):
                terms = Counter(_tokenize(text, split_identifiers=True))
                self._chunks.append(_Chunk(document_id, index, text, terms))
                self._frequencies.update(terms.keys())
        self._identifier_parts = {word: next(iter(parts))
                                  for word, parts in identifier_parts.items() if len(parts) == 1}
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

    def _score(self, chunk: _Chunk, terms: set[str], expanded: set[str],
               prefer_definition: bool = False, prefer_quantity: bool = False) -> float:
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
        for heading in _DEFINITION_HEADING.findall(chunk.text):
            definition_terms.update(_tokenize(heading, split_identifiers=True))
        score += 1.2 * sum(self._idf(t) for t in terms & definition_terms)
        if prefer_definition and _definition_start(chunk.text, terms) is not None:
            score += 3 * sum(self._idf(t) for t in terms)
        if prefer_quantity:
            # A retry-count question needs an explanation containing a count,
            # rather than an incidental 'this won't retry' in unrelated YAML.
            # Match the remaining subject in the same line as the number.
            quantity_terms = terms - self._document_terms[chunk.document_id]
            if quantity_terms and any(
                re.search(r"\b\d+\b", line)
                and quantity_terms <= set(_tokenize(line, split_identifiers=True))
                for line in chunk.text.splitlines()
            ):
                score += 2 * sum(self._idf(t) for t in quantity_terms)
        return score

    def query(self, q: str, scope: str = "reference", top_k: int = 4) -> dict:
        q = str(q).strip()
        top_k = max(1, min(8, int(top_k)))
        documents = self.documents(scope)
        allowed = {d["id"] for d in documents}
        subject = _query_subject(q)
        candidates = [c for c in self._chunks if c.document_id in allowed]
        vocabulary = set().union(*(c.terms.keys() for c in candidates)) if candidates else set()
        spelling_terms = set().union(*(self._document_terms[d] for d in allowed)) if allowed else set()
        corrections = []

        def correct_topic(match):
            word = match.group()
            if len(word) < 6 or set(_tokenize(word)) <= vocabulary:
                return word
            matches = [term for term in spelling_terms if len(term) >= 6
                       and _one_edit_apart(word.lower(), term)]
            # Limit correction to an unambiguous indexed topic, never an
            # arbitrary similar word anywhere in the corpus.
            if len(matches) == 1:
                corrections.append(f'Matched "{word}" to indexed topic "{matches[0]}".')
                return matches[0]
            return word

        subject = re.sub(r"[A-Za-z0-9]+", correct_topic, subject)
        terms = set(_tokenize(subject, split_identifiers=True))
        original_terms = set(terms)
        # Exact known identifiers and their complete word forms are equivalent
        # subjects. Keep the original vocabulary gate before this equivalence,
        # so invented fields cannot borrow their component words as evidence.
        for word, parts in self._identifier_parts.items():
            if word in terms and word in vocabulary:
                # Splitting can itself expose a normalized full name, such as
                # HorizontalPodAutoscaler -> horizontal pod autoscaler -> HPA.
                original = next((source for source in re.findall(r"[A-Za-z0-9]+", subject)
                                 if source.lower() == word), word)
                terms.difference_update(_tokenize(original, split_identifiers=True))
                terms.update(parts)
        field_values = []
        # Preserve connector positions: 'ConfigMap and Secret' compares two
        # subjects, whereas 'restartPolicy Always' asks for a field's value.
        query_words = [tuple(_tokenize(word)) for word in re.findall(r"[A-Za-z0-9]+", subject)]
        for word, parts in self._identifier_parts.items():
            if word not in vocabulary or len(parts) < 2:
                continue
            for form in ((word,), parts):
                for index in range(len(query_words) - len(form)):
                    if tuple(query_words[index:index + len(form)]) == tuple((part,) for part in form):
                        following = query_words[index + len(form)]
                        if len(following) != 1:
                            continue
                        value = following[0]
                        if value not in terms - set(parts) or value in {"kubernetes", "databricks", "job", "cronjob"}:
                            continue
                        relation = (set(parts), value)
                        if relation not in field_values:
                            field_values.append(relation)
        prefer_definition = bool(re.search(
            r"\b(?:what (?:is|are)|explain|clarify|describe)\b", subject, re.IGNORECASE))
        prefer_how_definition = bool(re.fullmatch(
            r"how\s+(?:do|does|did|can|could|will|would)\s+(?!i\b|you\b|we\b)"
            r"[A-Za-z0-9][A-Za-z0-9 /_-]*?\s+(?:work|function|operate)\s*[?!.]*",
            normalize_question(q), re.IGNORECASE))
        prefer_definition = prefer_definition or prefer_how_definition
        prefer_quantity = bool(re.search(r"\bhow many\b", q, re.IGNORECASE))
        expanded = set().union(*(_EXPANSIONS.get(t, set()) for t in terms)) if terms else set()
        known = terms & vocabulary
        trace = [f"Searched {len(documents)} indexed documents using local BM25 retrieval.",
                 *corrections]
        no_answer = {
            "answer": "I couldn't find enough evidence in the indexed documents to answer this question. Try a question about Kubernetes Jobs, CronJobs, autoscaling, monitoring, work queues, or Databricks jobs.",
            "status": "no_answer", "thought_process": trace + ["No sufficiently matching source text was found."],
            "sources": [], "citations": [],
        }
        # Unknown subject words must not be silently discarded in favor of
        # generic overlaps. Intent words and inflections are normalized above;
        # local mode conservatively declines unfamiliar content vocabulary.
        if not terms or not original_terms <= vocabulary or known != terms:
            return no_answer
        if prefer_how_definition:
            definition_documents = {
                chunk.document_id for chunk in candidates
                if _definition_start(chunk.text, known, headings_only=True) is not None
            }
            if definition_documents:
                candidates = [chunk for chunk in candidates if chunk.document_id in definition_documents]
        term_weight = sum(self._idf(t) for t in known)
        supported_documents = {
            document_id for document_id in allowed
            if sum(self._idf(t) for t in known & self._document_vocabulary[document_id])
            / max(term_weight, 1e-9) >= 0.8
        }
        ranked = sorted(
            ((self._score(c, known, expanded, prefer_definition, prefer_quantity), c) for c in candidates
             if c.document_id in supported_documents
             and all(_field_value_matches(c.text, field, value) for field, value in field_values)
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
                # For a definition request, lead with an explicit definition
                # heading before diversifying the supporting passages. Broad
                # incidental mentions can otherwise win the coverage bonus.
                starts_with_definition = (
                    prefer_definition and not selected
                    and _definition_start(chunk.text, known, headings_only=True) is not None
                )
                return (starts_with_definition,
                        score * (1 - 0.55 * overlap) + 1.5 * sum(self._idf(t) for t in new_terms))
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
            start = _definition_start(chunk.text, known) if prefer_definition else None
            excerpt = chunk.text[start:] if start is not None else chunk.text
            citations.append({
                "id": f"{chunk.document_id}-{chunk.index}",
                "document_id": chunk.document_id,
                "title": document["title"], "filename": document["filename"],
                "excerpt": excerpt,
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
