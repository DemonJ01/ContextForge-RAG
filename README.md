# ContextForge RAG

**Your sources. A clearer perspective.**

ContextForge RAG is a document workspace for asking questions, browsing a knowledge library, and inspecting the source passages behind an answer. A single FastAPI process serves the web interface and API.

Start locally with the included documents and **no API keys**. Local mode uses BM25 keyword retrieval to return cited excerpts; the optional cloud pipeline adds semantic retrieval and language-model answers.

![ContextForge RAG workspace showing a Kubernetes CronJobs question, source excerpts, and the Evidence panel](https://github.com/user-attachments/assets/ff6cd969-5967-4890-ad17-bae16611a624)

[Quick start](#quick-start) · [Workspace](#using-the-workspace) · [Cloud setup](#optional-cloud-pipeline) · [API](#api) · [Documentation](#documentation)

## Features

- **Inspect the evidence:** open numbered source passages and their extracted documents in local mode.
- **Browse the library:** filter documents by title or filename and read their saved text.
- **Choose a search scope:** query reference documents or the entire index.
- **Keep conversations organized:** create separate chats, retain drafts, and revisit history saved in your browser.
- **Enable cloud RAG:** use LangGraph, Groq through Portkey, Qdrant, FlashRank, and NeMo guardrails with your own provider configuration.
- **Ingest documents from the command line:** process PDF, HTML, TXT, DOCX, and PPTX files with the optional cloud dependencies.

## Quick start

Requires **Python 3.11 or later** and Git to clone the repository.

### macOS / Linux

```bash
git clone https://github.com/DemonJ01/ContextForge-RAG.git
cd ContextForge-RAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_rag.py
```

### Windows PowerShell

```powershell
git clone https://github.com/DemonJ01/ContextForge-RAG.git
cd ContextForge-RAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_rag.py
```

If you already have the project, start from its directory and skip the clone steps. Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)**. To use another port, run `python run_rag.py --port 8080`.

The default local mode needs no `.env` file, ingestion run, frontend build, or Streamlit server. If your environment already sets `RAG_MODE=cloud`, change it to `local` for this quick start.

## Using the workspace

1. Start a **New conversation** and ask a question, such as **“How do Kubernetes CronJobs work?”**
2. Choose **Reference documents** or **Entire library** before sending your question.
3. Read the response and inspect the **Evidence** panel to check the retrieved passages.
4. Open **Knowledge library** to filter the document list and read extracted source text.

The bundled library contains **11 documents: 6 reference documents and 5 distractors**. Example topics include Kubernetes CronJobs, pod autoscaling, job monitoring, Redis work queues, and Databricks jobs. Original files live in `data/`; local retrieval searches the text saved in `processed_documents/`.

Conversation history and drafts are stored in your browser. Short local follow-ups can reuse the last supported subject within the same session and search scope. That server-side context is held in memory, expires after 30 minutes, and clears on restart.

### Local and cloud modes

| | Local mode (default) | Cloud mode (optional) |
|---|---|---|
| Retrieval | BM25 keyword ranking over saved document text | Embeddings, Qdrant vector search, and FlashRank reranking |
| Answers | Extracted source passages; no language model | LangGraph workflow with Groq through Portkey |
| Setup | Minimal Python dependencies; no provider keys | Cloud dependencies, provider credentials, and an ingested collection |
| Document browser | Bundled extracted library | Bundled extracted library; cloud queries search the remote collection |

Local retrieval recognizes common question phrasing, autoscaler names, and API field spellings, but still relies on word matching. Unfamiliar paraphrases can produce weak matches or no evidence. A completed response means passages were found; check whether they answer the whole question. The saved documents may describe older software defaults.

The web workspace currently supports reading and querying documents. Add files through the ingestion command below. Restart the application after regenerating saved document text so it reloads the local index.

## Optional cloud pipeline

Activate your virtual environment and install the additional dependencies:

```bash
python -m pip install -r requirements-cloud.txt
```

Create a private `.env` file in the project root. Set `RAG_MODE=cloud` and configure:

| Setting | Purpose |
|---|---|
| `GROQ_API_KEY` | Input guardrail classification |
| `PORTKEY_API_KEY` | Planner and answer-generation gateway |
| `PORTKEY_PRIMARY_SLUG`, `PORTKEY_FALLBACK_SLUG` | Integration names in your Portkey account; defaults are `rag` and `rag-fallback` |
| `QDRANT_CLUSTER_ENDPOINT`, `QDRANT_API_KEY` | Qdrant connection |
| `QDRANT_COLLECTION` | Collection name; defaults to `rag` |
| `RAG_EMBEDDING_BACKEND` | `gemini` (default, 3072 dimensions) or `local` (768 dimensions) |
| `GEMINI_API_KEY` | Required for Gemini embeddings |

See the [environment-variable reference](docs/environment_variables.md) for model settings and optional tracing. Explicit `local` embeddings use `sentence-transformers/all-mpnet-base-v2`, which may download on first use. This still uses the cloud pipeline for generation and Qdrant retrieval.

Ingest the bundled source files, then start the application:

```bash
python -m rag.ingestion.document_processor data
python run_rag.py
```

For your own reference documents, pass a directory and the `true` source label:

```bash
python -m rag.ingestion.document_processor /path/to/documents true
```

Use the same embedding model and backend for ingestion and queries. Choose a separate collection and re-ingest when changing models. Embedding failures stop the operation instead of silently switching models.

Normal ingestion replaces chunks for the same filename and source type, including obsolete chunks after a successful upsert. The optional **`--wipe` flag deletes and recreates the selected Qdrant collection**. Reference scope selects remote points labeled `source_type="true"`; use `scope="all"` for other source types.

Open the sidebar connection dialog or request `/configuration` for setup diagnostics without exposing credentials or contacting providers. Cloud `/health` also checks Qdrant readiness; run a real query to verify model-provider access. Missing cloud configuration returns HTTP 503 instead of switching to local answers.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Web workspace |
| `GET` | `/health` | Runtime mode and index readiness |
| `GET` | `/configuration` | Setup diagnostics without credential values |
| `GET` | `/documents?scope=all` | Bundled document metadata |
| `GET` | `/documents/{id}` | Extracted text for a bundled document |
| `POST` | `/query` | Answer, evidence, timing, and session ID |
| `GET` | `/graph` | Workflow diagram source |

Example request from macOS or Linux:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"q":"How do Kubernetes CronJobs work?","scope":"reference","top_k":4}'
```

Questions must contain **1–2,000 characters** after trimming. `scope` accepts `reference` (default) or `all`; `top_k` accepts integers from **1 to 8** (default: 4). Supply the same optional `thread_id` to associate follow-up questions with a session. Omit it to receive a new ID.

Local responses include structured `citations` and plain-text `sources`. Cloud evidence is returned through `sources`.

## Docker

```bash
docker build -t contextforge-rag .
docker run --rm -p 127.0.0.1:8080:8080 contextforge-rag
```

Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)**. The default image runs local mode and includes the bundled index. A cloud image requires the optional cloud dependencies and provider configuration.

## Development and tests

With your virtual environment active, install the test dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The suite covers retrieval relevance, follow-ups, no-evidence responses, source integrity, request validation, runtime errors, static assets, and API contracts.

With Node.js installed, run the frontend state regressions for conversation drafts, navigation, retries, and interrupted requests:

```bash
node --test tests/test_web_state.cjs
```

### Optional evaluation tools

```bash
python -m pip install -r requirements-evaluation.txt
streamlit run evaluation/rag_dashboard.py
```

Start the API separately. These tools use `BACKEND_URL` to reach it; judge-based evaluation also requires provider credentials. Legacy Streamlit chat clients remain in `ui/rag_chat.py` and `ui/rag_chat_cloud.py`.

## Project structure

```text
rag/
  api.py                     HTTP routes and integrated frontend
  runtime.py                 Local/cloud runtime selection
  local_rag.py               BM25 retrieval and source evidence
  agents/                    LangGraph cloud workflow
  ingestion/                 Document loaders, chunking, and indexing
  services/retrieval/        Embeddings, vector search, and reranking
ui/web/                      Workspace HTML, CSS, and JavaScript
data/                        Original source documents
processed_documents/         Saved extracted text and chunk metadata
evaluation/                  Evaluation pipeline and dashboard
tests/                       Python and frontend regression tests
docs/                        Setup and architecture guides
run_rag.py                   Application launcher
requirements.txt             Minimal local runtime
requirements-cloud.txt       Optional cloud dependencies
requirements-evaluation.txt  Optional evaluation dependencies
```

## Documentation

- [Environment variables](docs/environment_variables.md)
- [Architecture](docs/architecture.md)
- [System overview](docs/system_overview.md)
- [Document ingestion](docs/document_ingestion.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Evaluation pipeline](docs/rag_evaluation_pipeline.md)

## License

Released under the [MIT License](LICENSE).
