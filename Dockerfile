FROM python:3.12-slim-bookworm

WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY rag/ ./rag/
COPY ui/web/ ./ui/web/
COPY processed_documents/ ./processed_documents/

ENV RAG_MODE=local
EXPOSE 8080
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8080"]
