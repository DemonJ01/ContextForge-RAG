# RAG environment variables

Local mode works without a `.env` file or provider keys. To configure the optional cloud mode, copy `.env.example` to `.env` in the project root and set `RAG_MODE=cloud`. [`rag/settings.py`](../rag/settings.py) loads environment values with `python-dotenv` and `os.getenv`; it currently uses a plain settings class without schema validation.

| Variable | Purpose | Default or example |
| `RAG_MODE` | Selects the credential-free extractive engine or configured cloud pipeline | `local` or `cloud` |
|---|---|---|
| `GROQ_API_KEY` | Direct Groq calls used by guardrail classification and notebook examples | Required for guardrails |
| `GROQ_FALLBACK_API_KEY` | Secondary Groq credential used by notebook examples | Optional for the API; configure fallback credentials in Portkey |
| `PORTKEY_API_KEY` | Authenticates the planner and answer generator with Portkey | Required |
| `PORTKEY_PRIMARY_SLUG` | Primary Groq integration name in your Portkey account | `rag` |
| `PORTKEY_FALLBACK_SLUG` | Fallback Groq integration name in your Portkey account | `rag-fallback` |
| `PORTKEY_CONFIG_ID` | Saved configuration used by an optional gateway notebook experiment | Your own saved configuration ID |
| `RAG_EMBEDDING_BACKEND` | Cloud pipeline embedding backend, fixed for both ingestion and retrieval | `gemini` (3072 dimensions) or explicit `local` (all-mpnet-base-v2, 768 dimensions) |
| `GEMINI_API_KEY` | Gemini document and query embeddings | Required when `RAG_EMBEDDING_BACKEND=gemini` |
| `QDRANT_CLUSTER_ENDPOINT` | Qdrant server URL | `https://your-cluster.cloud.qdrant.io:6333` |
| `QDRANT_API_KEY` | Qdrant access token | Required for authenticated servers |
| `QDRANT_COLLECTION` | Collection used by both ingestion and retrieval | `rag` |
| `LOGFIRE_TOKEN` | Logfire tracing credential | Your project token |
| `LANGSMITH_TRACING` | Enables LangSmith tracing | `true` |
| `LANGSMITH_ENDPOINT` | LangSmith endpoint | `https://api.smith.langchain.com` |
| `LANGSMITH_API_KEY` | LangSmith credential | Your project key |
| `LANGSMITH_PROJECT` | Groups LangSmith traces | `rag` |
| `BACKEND_URL` | Backend address used by the local Streamlit chat | `http://localhost:8000` |
| `JUDGE_GROQ` | Groq judge key used for evaluation | Separate evaluation credential |
| `NVIDIA_API_KEY` | Optional NVIDIA guardrail notebook experiment | Your NVIDIA credential |

The slugs must match integrations configured in your Portkey account. Portkey uses credentials stored with those integrations; the API does not forward `GROQ_FALLBACK_API_KEY` to Portkey. Set the slug variables to your existing integration names when reusing an account.

The default Qdrant collection is `rag`. Set `QDRANT_COLLECTION` to the current collection name if you want to reuse existing vectors. Match its embedding model and dimensions. Repository renaming changes local files; it does not rename, delete, or migrate remote resources. Build a new index explicitly if you choose a new collection.

Embedding failures stop ingestion or retrieval; the app never switches automatically to another embedding model. Use the same `RAG_EMBEDDING_BACKEND` for both processes. To switch backends, select a separate Qdrant collection and ingest it with the chosen backend before querying. Explicit local embeddings require the optional cloud dependencies and may download the sentence-transformer model on first use. This setting is independent of `RAG_MODE=local`, whose bundled BM25 engine requires no embedding model.

The integrated web interface uses same-origin API requests and needs no `BACKEND_URL`. The optional Streamlit and evaluation clients read this variable to reach the running API. Both runtimes honor scope and top-k settings; cloud reference scope filters remote points with `source_type=true`. The document browser always shows the bundled extracted library.

Keep `.env` out of version control. Avoid saving credentials or account configuration in notebook outputs. A separate evaluation key helps isolate workloads; verify the provider's actual account-level quotas before assuming separate keys have independent limits.
