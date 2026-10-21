# RAG improvement recommendations

These recommendations come from reading the application, UI, evaluation code, notebooks, documentation, datasets, and extracted corpus text. The initial rename updated project identity, file references, and notebook hygiene. A later runtime repair added the integrated web interface and a credential-free local evidence engine. See [runtime verification](rag_runtime_verification.md) for current behavior. The recommendations below primarily describe the original cloud pipeline and evaluation tools.

## Runtime repair status

The default app now starts without cloud SDKs or credentials, serves its frontend and API from one process, validates questions, generates a fresh thread ID when absent, returns real HTTP errors, and preserves local source metadata/citations. Local search chunks are bounded in memory; original processed files are unchanged. Optional clients share backend configuration. The cloud/evaluation limitations below remain where specifically noted.

## Highest priority

1. **Isolate conversation memory.** [`rag/api.py`](../rag/api.py) gives requests without an ID the shared `default_user` thread; [`workflow.py`](../rag/agents/workflow.py) checkpoints by that ID. Generate a unique session ID when absent, reject invalid IDs, and enforce ownership when adding authentication. Verify that two independent clients cannot read each other's history. `MemorySaver` is process-local; use persistent storage if conversations must survive restarts.

2. **Make embedding models consistent.** [`embedding_service.py`](../rag/services/retrieval/embedding_service.py) chooses between Gemini's 3072-dimensional vectors and a 768-dimensional local model independently in each process. Store the model ID, dimensions, and version with each collection; validate compatibility at startup and fail clearly on a mismatch. Do not silently query one model's vectors with another model.

3. **Make evaluation failures visible.** [`rag_pipeline.py`](../evaluation/rag_pipeline.py) substitutes golden contexts when a live call fails; [`rag_metrics.py`](../evaluation/rag_metrics.py) also substitutes them for empty retrieval. [`guardrails_evaluation.py`](../evaluation/guardrails_evaluation.py) treats connection errors as unblocked requests. Keep actual retrieval results, distinguish failed calls from passed tests, and prevent unavailable services from producing successful-looking scores.

4. **Preserve evidence and handle insufficient context.** [`document_retriever.py`](../rag/agents/nodes/document_retriever.py) discards source filenames and scores. Keep document IDs, filenames, page/chunk identifiers, and scores through reranking and generation; return structured citations to the UI. Add an explicit unsupported-answer outcome when context does not contain the answer. Test missing, conflicting, and adversarial source material.

## Retrieval and ingestion

5. **Enforce chunk budgets.** [`text_splitter.py`](../rag/ingestion/chunking/text_splitter.py) accepts a long paragraph without splitting it. Existing processed chunks reach 62,725 characters despite the intended 1,500-character budget. Split oversized blocks, add deliberate overlap, and enforce a token budget for generation. Rebuild affected chunks and vectors after changing the policy.

6. **Make ingestion repeatable and auditable.** [`document_processor.py`](../rag/ingestion/document_processor.py) creates random point IDs, so repeated runs duplicate documents. Use stable IDs based on source/content/chunk version, replace changed documents, and report per-file failures with a nonzero exit status. Validate vector counts before indexing and handle deletions explicitly.

7. **Measure document extraction coverage.** Add page-level text coverage and OCR support for scanned PDFs. Preserve page order when the PDF fallback recovers an intermediate page. Three image-heavy corpus PDFs provide very little extractable text; a plain text scan cannot inspect words inside those images. Use the provenance map to retain original document titles after filename normalization.

## API, guardrails, and UI

8. **Validate requests and return useful HTTP errors.** Bound question length, reject blank input, validate thread IDs, and return appropriate HTTP status codes. [`rag/api.py`](../rag/api.py) currently wraps backend errors in HTTP 200 responses. Make the chat interface check HTTP status and handle empty answers consistently.

9. **Use explicit guardrail outcomes and readiness.** [`safety.py`](../rag/guardrails/safety.py) detects guardrail decisions with response substrings and skips checks when initialization is missing. Prefer structured outcomes, report initialization status, and test bypasses and false positives. Classification uses a remote Groq model; documentation should describe that data flow accurately. Output filtering is not currently applied after answer generation.

10. **Share backend configuration.** The cloud chat and evaluation clients hardcode localhost. Read `BACKEND_URL` consistently in [`rag_chat_cloud.py`](../ui/rag_chat_cloud.py), the live evaluation pipeline, and guardrail evaluation. Surface connection failures and deployment readiness clearly.

11. **Bound retained history and prompt size.** Planning and generation currently concatenate all prior messages. Limit or summarize history and use a token budget in place of the current 25,000-character generation-context cap so longer conversations have predictable latency and cost.

## Evaluation and maintenance

12. **Use fresh evaluation sessions and full evidence.** Evaluation thread IDs repeat across runs, allowing previous history to affect later scores. Generate a run ID plus sample ID. Preserve full answers and retrieved chunks: answers currently stop at 300 characters, and metric preparation reduces evidence further. Make judge budgets configurable and report truncation explicitly.

13. **Broaden the golden dataset.** Add no-answer, conversational, follow-up, conflicting-source, prompt-injection, and failure cases. Check tool arguments as well as tool names; the current tool score is a Jaccard calculation over names. Add useful offline regression checks and a separate credential-backed integration/evaluation job.

14. **Make dependencies reproducible.** Pin or lock local dependencies, remove duplicate package declarations, and align the API dependency set with actual imports. The API imports `langchain_google_genai`, while its container dependency list omits the direct package. The local embedding fallback needs sentence-transformers, which the API set intentionally excludes. Verify installation and container startup in CI using the supported Python version.

15. **Align guides with code and measurements.** Several guides overstate memory persistence, citation guarantees, guardrail determinism, caching behavior, and truncation accuracy. Some evaluation examples have inconsistent rate limits and batching descriptions. Document actual behavior and use measured latency/accuracy results. Keep exact-match and semantic caching examples distinct.

16. **Finish credential hygiene.** Embedded notebook API-key defaults and saved outputs have been removed. Rotate any previously embedded real keys at the provider and use environment variables or a secrets manager. Keep executed notebook outputs out of published examples when they expose account details.

Suggested implementation order: conversation isolation, embedding compatibility, evaluation failure accounting, citation/evidence preservation, then chunking and repeatable ingestion. Each change should include a focused regression check that demonstrates the original failure.
