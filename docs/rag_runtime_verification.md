# RAG runtime verification

Verified on 21 October 2026 using Python 3.12.14 and a separate minimal environment.

## Original failure and repair

Importing the supplied API failed with `ModuleNotFoundError: logfire`. Cloud SDKs and provider credentials were required during module import, and the UI required a separate Streamlit process. No project `.env` or provider credentials were present.

The default runtime now serves the web interface and API together and uses a local, immutable document index. Optional cloud imports are lazy and cloud mode is explicit. The local path does not call external services. It returns actual extracted evidence rather than simulating model-generated answers.

## Automated checks

- All **30 retrieval/API/cloud-boundary tests pass** with `python -m unittest discover -s tests -v`.
- A clean installation of `requirements-dev.txt` succeeds. Dependency compatibility checks pass.
- All 15 existing golden questions retrieve their expected leading source document, and all 3 suggested UI questions work.
- Unsupported subject questions return `no_answer` without citations. Tests include unknown terms mixed with familiar Kubernetes terms.
- Citation excerpts resolve verbatim to the source returned by the document endpoint.
- Search chunks stay within 1,300 characters; original processed documents are not rewritten.
- Technical cloud generation is withheld when retrieval has no evidence. This is tested with stubbed SDK calls; it does not claim real provider verification.

## Live Chromium checks

The actual Uvicorn server was exercised at `http://127.0.0.1:8000`.

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

Real Groq/Portkey/Gemini/Qdrant calls were not tested because provider credentials were not supplied. The cloud pipeline still needs its configured remote index and integrations. Local mode provides extractive evidence search, not language-model synthesis or conversational reasoning.

The Docker build was not run because the configured local Docker daemon socket was absent. The Dockerfile and build context were updated to include the integrated UI and saved document text. No external deployment was performed.

## Hive verification record

```json
{
  "assumptions": "A working credential-free local document workspace is the default; external provider services remain optional and require configuration.",
  "short_plan": "Reproduce startup failure, isolate optional cloud imports, implement actual local evidence retrieval, connect and redesign the web UI, verify APIs and live browser interactions.",
  "success_criteria": "One process serves the UI and backend; queries return actual source evidence; unsupported questions do not claim evidence; library, history, errors, and responsive controls operate on real data.",
  "verification_result": "30 automated tests passed; 12 live Chromium checks passed; all 15 existing golden source-topic checks passed; installed runtime dependencies are compatible. Real cloud calls and Docker build remain explicitly unverified."
}
```
