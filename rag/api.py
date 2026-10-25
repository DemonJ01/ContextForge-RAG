"""RAG web application and HTTP API, runnable without cloud credentials."""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from rag.runtime import RAGRuntime, RuntimeUnavailable

logger = logging.getLogger(__name__)
Scope = Literal["all", "reference"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=2000)
    thread_id: Optional[str] = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    scope: Scope = "reference"
    top_k: int = Field(default=4, ge=1, le=8, strict=True)

    @field_validator("q", mode="before")
    @classmethod
    def trim_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class QueryResponse(BaseModel):
    question: str
    thread_id: str
    mode: str
    elapsed_ms: float
    answer: str
    status: str
    thought_process: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)


def create_app(project_root: Optional[Path] = None, runtime: Optional[RAGRuntime] = None) -> FastAPI:
    project_root = Path(project_root or PROJECT_ROOT).resolve()
    runtime = runtime if runtime is not None else RAGRuntime(project_root)
    app = FastAPI(title="RAG", description="Search documents and explore cited evidence.", version="1.0.0")
    app.state.runtime = runtime
    web_dir = project_root / "ui" / "web"
    app.mount("/static", StaticFiles(directory=str(web_dir), check_dir=False), name="static")

    def execute(operation):
        try:
            return operation()
        except RuntimeUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("RAG request failed")
            raise HTTPException(
                status_code=500, detail="The request could not be completed. Check the server logs."
            ) from exc

    @app.get("/", include_in_schema=False)
    def home():
        index = web_dir / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=503, detail="The web interface is missing. Restore ui/web/index.html.")
        return FileResponse(index, media_type="text/html")

    @app.get("/health")
    def health():
        return execute(runtime.health)

    @app.get("/configuration")
    def configuration():
        return execute(runtime.configuration)

    @app.get("/documents")
    def documents(scope: Scope = "all"):
        items = execute(lambda: runtime.documents(scope=scope))
        return {"documents": items, "total": len(items)}

    @app.get("/documents/{document_id}")
    def document(document_id: str):
        # Document IDs are opaque hashes looked up by the engine, never filesystem paths.
        if not document_id or len(document_id) > 128 or not all(
            character.isascii() and (character.isalnum() or character in "_-") for character in document_id
        ):
            raise HTTPException(status_code=404, detail="Document not found.")
        result = execute(lambda: runtime.document(document_id))
        if result is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return result

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        start = perf_counter()
        thread_id = request.thread_id or uuid4().hex
        result = execute(lambda: runtime.query(
            request.q, thread_id=thread_id, scope=request.scope, top_k=request.top_k
        ))
        return {
            **result, "question": request.q, "thread_id": thread_id, "mode": runtime.mode,
            "elapsed_ms": round((perf_counter() - start) * 1000, 2),
        }

    @app.get("/graph")
    def graph():
        return execute(runtime.graph)

    return app


app = create_app()
