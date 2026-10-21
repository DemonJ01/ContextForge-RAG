# RAG improvement status

The default integrated local app is verified with a fresh installation, API tests, and live browser checks. See [runtime verification](rag_runtime_verification.md) for measured results and the limits of that verification.

## Completed repairs

- The API serves the UI and backend together without requiring cloud SDKs or keys in local mode.
- Questions, scope, result limits, and thread IDs are validated. Requests without a thread ID receive a unique session, and failures return HTTP error statuses.
- Local retrieval returns bounded excerpts with citations that resolve to the indexed source text. Unsupported questions return no evidence.
- The library, conversation history, source dialogs, retry controls, and responsive layouts are connected to the running API. Results completed while browsing the library appear when returning to chat.
- Cloud requests carry scope and top-k through retrieval. Reference scope filters remote points by `source_type=true`, and provider failures surface as errors.
- Cloud embeddings use an explicit, stable backend. Ingestion and query processes must select the same model and collection; provider failures cannot silently change vector spaces.
- Ingestion bounds oversized paragraphs, validates embedding counts and collection dimensions, propagates failures, and uses stable IDs. After successful indexing, scoped cleanup removes obsolete chunks and legacy IDs for the same document.
- Ingestion validates the source directory and embedding provider before an explicit `--wipe` can delete the remote index.
- Evaluation uses fresh sessions and preserves empty live retrieval. Failed guardrail requests are reported as errors instead of passes, and dashboard metrics show completed/error counts.
- Python clients share `BACKEND_URL`, local runtime dependencies are pinned, and environment/credential files remain excluded from Git.

## Remaining optional cloud improvements

1. **Verify live services and installation.** Provider calls, cloud SDK installation, model downloads, remote ingestion, and judge-based evaluation require a configured environment. Current cloud regressions use mocked SDKs. The Docker daemon must also be running before container verification.

2. **Preserve structured cloud source metadata.** The cloud retriever currently passes text through reranking and generation. Carry source filenames, document/chunk IDs, and scores into structured citations, as the local engine already does. The browser identifies remote text as retrieved passages; its document library contains bundled sources.

3. **Persist and bound cloud conversations.** LangGraph memory is process-local. Add persistent storage and ownership checks when adding authentication. Bound or summarize retained history and use a token budget for prompts; the existing generation context cap measures characters.

4. **Record an index manifest.** Collection dimension checks and fixed backends prevent automatic cross-model fallback, but an explicit manifest should also record model/version identity. Handle removed source files and concurrent re-ingestion jobs explicitly; per-document upsert and cleanup are not an atomic transaction.

5. **Improve extraction coverage.** Add OCR for scanned PDFs and verify page order when the fallback parser recovers blank intermediate pages. Image-only content cannot be searched by the current text index. Keep the provenance map when normalizing filenames.

6. **Strengthen guardrails and evaluation.** Prefer structured guardrail outcomes over response substring detection, test output filtering, and extend the golden dataset to follow-ups, conflicting sources, adversarial content, and failures. Evaluation currently truncates answers and contexts to reduce judge cost; make budgets configurable and expose truncation in reports.

7. **Broaden local language coverage.** Local mode is extractive BM25 search. Its conservative vocabulary checks can decline uncommon paraphrases; it does not synthesize model answers or reason over conversational history.

These follow-ups extend optional integrations and operational guarantees. They are not claims that real cloud services have been tested.
