# Troubleshooting

## Local startup

Start the integrated app with `python run_rag.py` after installing `requirements.txt`. FastAPI serves the API and web interface together. Local mode does not require Logfire, LangSmith, cloud SDKs, or provider credentials.

If a different runtime is selected by the shell or project `.env`, set `RAG_MODE=local` to use the saved document index. A shell value takes precedence over `.env`.

## Cloud initialization and logging

Cloud modules are imported only after `RAGRuntime.cloud_components()` validates the configuration. That method configures Logfire before importing the workflow and guardrails. The ingestion command has its own logging initialization.

Calls to Logfire before configuration may produce this warning:

```text
No logs or spans will be created until logfire.configure() has been called.
```

If cloud spans are missing, check that the entry point configured logging before importing modules that emit spans. Also check the token and the dashboard used by that process. Without `LOGFIRE_TOKEN`, the API intentionally uses `send_to_logfire=False`; local mode does not initialize Logfire at all.

A cloud HTTP 503 can also indicate missing environment values, optional packages, or failed provider initialization. The error response gives a configuration-level message; server logs hold the underlying exception. The runtime keeps cloud failures as errors rather than silently returning local answers.

## First-query model loading

FlashRank and the embedding model are initialized on first use. First-query work may include model downloads, so a fast API startup does not imply the first cloud query will have the same latency as later queries.

The reranker first tries `/tmp/flashrank` for its cache. Check write permissions and download access if initialization fails. A ranking failure falls back to the original Qdrant order; it does not bypass embedding or database failures.

## Embedding and collection mismatch

Use the same `RAG_EMBEDDING_BACKEND` for ingestion and querying. Gemini uses 3072-dimensional vectors; explicit local embeddings use 768 dimensions. An existing Qdrant collection must match the selected backend. Select a separately indexed collection when changing models.

See the [configuration guide](environment_variables.md) for required provider settings and the [ingestion guide](document_ingestion.md) for collection replacement behavior.
