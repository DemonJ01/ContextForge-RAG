"""Connection settings and response validation for the optional Python API clients."""
from pathlib import Path
import os

from dotenv import load_dotenv


class QueryResponseError(ValueError):
    """A user-safe error returned while validating the RAG API response."""


def backend_url() -> str:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return (os.getenv("BACKEND_URL") or "http://127.0.0.1:8000").strip().rstrip("/")


def query_response(response) -> dict:
    """Validate API results without displaying raw HTTP bodies or request errors."""
    if response.status_code == 422:
        raise QueryResponseError("Keep your question between 1 and 2,000 characters, then try again.")
    if response.status_code != 200:
        if response.status_code == 503:
            try:
                body = response.json()
            except ValueError:
                body = None
            # RuntimeUnavailable details are deliberately safe, actionable API messages.
            detail = body.get("detail") if isinstance(body, dict) else None
            if isinstance(detail, str) and detail.strip():
                raise QueryResponseError(detail.strip()[:500])
        raise QueryResponseError(
            f"The backend could not complete this search (HTTP {response.status_code}). "
            "Please try again or check the server logs."
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise QueryResponseError("The backend returned an unreadable response. Please try again.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str) \
            or not data["answer"].strip() or data.get("status") == "error":
        raise QueryResponseError("The backend returned no usable answer. Please try again.")
    return data
