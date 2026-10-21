# RAG runtime verification

Verified again on 21 October 2026 using Python 3.12.14 and the project's `.venv`.

## Original failure and repair

Importing the supplied API failed with `ModuleNotFoundError: logfire`. Cloud SDKs and provider credentials were required during module import, and the UI required a separate Streamlit process. No project `.env` or provider credentials were present.

The default runtime now serves the web interface and API together and uses a local, immutable document index. Optional cloud imports are lazy and cloud mode is explicit. The local path does not call external services. It returns actual extracted evidence rather than simulating model-generated answers.

## Automated checks

- All **54 retrieval/API/cloud-boundary tests pass** with `python -m unittest discover -s tests -v`.
- An additional 48 concurrent API requests across six supported/unsupported topics returned the expected status, distinct session IDs, and citations that resolve exactly to the source document endpoint.
- A clean installation of `requirements-dev.txt` succeeds. Dependency compatibility checks pass.
- All 15 existing golden questions retrieve their expected leading source document, and all 3 suggested UI questions work.
- Unsupported subject questions return `no_answer` without citations. Tests include unknown terms mixed with familiar Kubernetes terms.
- Citation excerpts resolve verbatim to the source returned by the document endpoint.
- Search chunks stay within 1,300 characters; original processed documents are not rewritten.
- Technical cloud generation is withheld when retrieval has no evidence. This is tested with stubbed SDK calls; it does not claim real provider verification.
- Cloud scope and top-k reach the retriever; reference queries filter the remote `source_type=true` payload.
- Embedding backend selection is explicit and stable, with vector count/dimension checks and no automatic model switch on provider failure.
- Ingestion bounds oversized paragraphs, rejects mismatched collection dimensions, propagates provider failures, and replaces obsolete chunks only after successful indexing. Mocked tests cover shrinking documents, legacy random IDs, and unrelated source preservation.
- Ingestion validates the source directory and initializes the embedding provider before `--wipe` can delete an existing remote collection. A regression test confirms a provider initialization failure preserves it.
- Evaluation preserves empty retrieval, creates fresh sessions, and reports failed guardrail requests separately from completed scores.

## Live Chromium checks

The actual Uvicorn server was exercised at `http://127.0.0.1:8765`. The latest run used the project's virtual environment; earlier verification also used a fresh environment and launched from outside the project directory.

1. Health and real index counts display correctly.
2. A question reaches the real API and produces source-linked evidence.
3. A citation opens its actual source document.
4. Copy answer places the returned answer on the test browser clipboard.
5. Library filtering, title search, document inspection, and empty-filter recovery work.
6. New conversations, switching history, and reload persistence work.
7. An unsupported question displays the backend's no-evidence response.
8. A simulated HTTP 503 displays an error; retry reaches the real backend successfully.
9. Layout has no horizontal overflow at widths **320, 375, 768, 1024, and 1440 pixels**. Mobile navigation/evidence drawers support Escape and focus management.
10. The reduced-motion preference is respected.
11. Injected HTML-like answer/source strings render as inert text.
12. Clearing saved history removes only the test browser's conversations.
13. An answer received while browsing the library appears when returning to the conversation; its old loading state is removed.
14. Cloud-mode scope controls remain usable and send the chosen scope/result limit. This contract check uses intercepted cloud health/query responses and does not call providers.

No page JavaScript errors or failed network requests occurred during the passing browser run. Error and hostile-content checks use intercepted responses; normal question/document requests use the real backend. Detailed results are in [rag_browser_validation.json](rag_browser_validation.json).

The primary, secondary, and muted text tokens meet at least 4.5:1 contrast against the app's principal surfaces. Viewport screenshots are used because the installed Chromium/Playwright combination emits a spurious media-query event during full-page capture, closing the evidence rail. Ordinary source interactions and responsive transitions passed.

## Run the checks

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

For optional browser verification, start the app separately, then run:

```bash
python -m pip install -r requirements-browser.txt
python -m playwright install chromium
python tests/browser_smoke.py
```

Set `RAG_BASE_URL` for a different port and `RAG_BROWSER_OUTPUT` for a different screenshot/report folder.

## Verification boundaries

Real Groq/Portkey/Gemini/Qdrant calls and judge-based evaluation were not tested because provider credentials were not supplied. The environment contains local runtime/browser dependencies; cloud SDK/model installation was not exercised. The cloud pipeline still needs its configured remote index and integrations. Local mode provides extractive evidence search, not language-model synthesis or conversational reasoning. The integrated UI browses saved document text; document ingestion is a command-line workflow.

The Docker build was not run because the configured local Docker daemon socket was absent. The Dockerfile and build context were updated to include the integrated UI and saved document text. No external deployment was performed.

## Hive verification record

```json
{
  "assumptions": "A working credential-free local document workspace is the default; external provider services remain optional and require configuration.",
  "short_plan": "Reproduce startup failure, isolate optional cloud imports, implement actual local evidence retrieval, connect and redesign the web UI, verify APIs and live browser interactions.",
  "success_criteria": "One process serves the UI and backend; queries return actual source evidence; unsupported questions do not claim evidence; library, history, errors, and responsive controls operate on real data.",
  "verification_result": "53 automated tests passed in a fresh environment; 14 live Chromium checks passed; all 15 existing golden source-topic checks passed; installed runtime dependencies are compatible. Cloud behavior is tested with mocked SDKs, not live providers. Real cloud calls, cloud SDK installation, and Docker build remain explicitly unverified."
}
```
