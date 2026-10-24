"""Start the integrated RAG application from any working directory."""
from pathlib import Path
import argparse
import os
import sys


def main() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("ContextForge-RAG requires Python 3.11 or later.")
    parser = argparse.ArgumentParser(description="Run the ContextForge-RAG workspace")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("Install dependencies first: python -m pip install -r requirements.txt")
    os.chdir(Path(__file__).resolve().parent)
    uvicorn.run("rag.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
