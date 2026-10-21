# RAG

A document workspace with a connected chat interface, searchable library, and inspectable source evidence. FastAPI serves the frontend and backend together, so one process runs the application.

The default **local mode works without API keys**. It ranks the included document text with BM25 and returns cited source excerpts. It does not run a language model. The optional cloud mode uses the original LangGraph, Groq/Portkey, Gemini, Qdrant, FlashRank, and NeMo pipeline.

## Start

Use Python 3.11 or later. From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_rag.py
```

Open [RAG](http://127.0.0.1:8000). On Windows, use `python` and activate with `.venv\Scripts\Activate.ps1` in PowerShell. Use `python run_rag.py --port 8080` to choose another port.

No `.env`, cloud service, ingestion run, frontend build, or Streamlit server is needed for local mode. The frontend makes same-origin requests to `/query`, `/health`, and `/documents`.

## Use the workspace

- Ask about CronJobs, autoscaling, job monitoring, Redis work queues, or Databricks jobs.
- Read numbered evidence excerpts and open their source documents.
- Use **Knowledge library** to browse and search the indexed document text.
- Choose reference documents or the full index when asking a question.
- Start a new conversation to keep questions separate. Browser history is stored locally; the local engine treats each question independently.

The saved index contains **11 documents**: 6 reference documents and 5 distractors. Original source files remain in `data/`; local mode searches the text already saved in `processed_documents/`. Oversized saved chunks are split into bounded search chunks in memory. Unmatched questions return a no-evidence response.

## Optional cloud pipeline

```bash
python -m pip install -r requirements-cloud.txt
```

Create a private `.env` file in the project root, set `RAG_MODE=cloud`, and configure Groq, Portkey, Gemini, and Qdrant credentials using the [configuration guide](docs/environment_variables.md). The Portkey slug settings must match integrations in your account. `RAG_EMBEDDING_BACKEND` defaults to `gemini` (3072 dimensions); explicitly choose `local` for all-mpnet-base-v2 embeddings (768 dimensions). Use the same backend for ingestion and queries; provider failures stop the operation instead of changing models. Ingest documents into the selected Qdrant collection before querying it:

```bash
python -m rag.ingestion.document_processor data
python run_rag.py
```

`QDRANT_COLLECTION` defaults to `rag`. Set it to an existing collection name to reuse that collection, matching its embedding model and dimensions. The optional ingestion `--wipe` flag deletes and recreates the selected remote collection after validating the source directory and initializing the embedding provider. Cloud configuration failures return HTTP 503; the app does not silently switch to local answers.

Cloud requests honor `scope` and `top_k`. Reference scope selects remote points with `source_type=true`; use the full index for other source types. Re-ingestion replaces chunks for the same filename and source type, including obsolete chunks when a document shrinks. Failed embedding, indexing, and cleanup operations stop ingestion with an error. The bundled local index is loaded at startup, so restart the app after regenerating its processed documents.

The default Docker image supports local mode. A cloud image needs the optional cloud dependencies added explicitly.

## API

| Route | Purpose |
|---|---|
| `GET /` | RAG web workspace |
| `GET /health` | Runtime mode and actual index counts |
| `GET /documents?scope=all` | Indexed document metadata |
| `GET /documents/{id}` | Extracted source text |
| `POST /query` | Answer, evidence, timing, and session ID |
| `GET /graph` | Workflow diagram source |
| `GET /docs` | Interactive API documentation |

```json
{"q": "How do Kubernetes CronJobs work?", "scope": "reference", "top_k": 4}
```

An optional `thread_id` keeps cloud conversations associated with a session. If it is omitted, the API generates a fresh ID. Questions must contain 1–2,000 characters. Local answers quote the indexed source text and return structured `citations`, plus plain-text `sources` for legacy clients.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The tests exercise retrieval relevance, no-evidence answers, chunk limits, source integrity, session IDs, request validation, HTTP failures, static assets, and frontend/backend contracts.

## Container

```bash
docker build -t rag .
docker run --rm -p 8080:8080 rag
```

## Optional evaluation tools

```bash
python -m pip install -r requirements-evaluation.txt
streamlit run evaluation/rag_dashboard.py
```

These tools use `BACKEND_URL` to connect to the running API. Judge-based evaluation needs provider credentials. Legacy chat scripts remain in `ui/rag_chat.py` and `ui/rag_chat_cloud.py`; the integrated web workspace is the primary interface.

## Files and guides

```text
rag/api.py                    HTTP routes and integrated frontend
rag/runtime.py                Explicit local/cloud runtime selection
rag/local_rag.py              Credential-free document index and evidence retrieval
ui/web/                       Responsive workspace HTML, CSS, JavaScript
processed_documents/          Saved extracted document text
run_rag.py                    Local launcher
requirements.txt              Minimal working runtime
requirements-cloud.txt        Optional cloud integration dependencies
requirements-evaluation.txt   Optional evaluation and Streamlit dependencies
tests/                        Retrieval and API regression tests
```

[Configuration](docs/environment_variables.md) · [Cloud architecture](docs/architecture.md)
