"""Streamlit entry point supporting BACKEND_URL in deployment secrets."""
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.streamlit_app import render_app


render_app(use_secrets=True)
