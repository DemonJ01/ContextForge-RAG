"""Connection settings shared by the optional Python API clients."""
from pathlib import Path
import os

from dotenv import load_dotenv


def backend_url() -> str:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return (os.getenv("BACKEND_URL") or "http://127.0.0.1:8000").strip().rstrip("/")
