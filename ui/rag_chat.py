"""Run with: streamlit run ui/rag_chat.py."""
from pathlib import Path
import sys


# Streamlit may start with only the script's directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.streamlit_app import render_app


render_app()
