import os
import sys
import uuid
import json
import logfire
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag.settings import settings
from rag.services.retrieval.embedding_service import embed_texts, get_embedding_dim
from rag.ingestion.loaders.pdf_loader import parse_pdf
from rag.ingestion.loaders.html_loader import parse_html
from rag.ingestion.loaders.text_loader import parse_text
from rag.ingestion.chunking.text_splitter import chunk_text

logfire.configure(service_name="rag-ingestion")

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = str(PROJECT_ROOT / "processed_documents")

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_documents/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → save locally → embed → index in Qdrant."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from rag.ingestion.loaders.office_loader import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                if len(embeddings) != len(chunks):
                    raise ValueError("Embedding count does not match the document chunk count.")
                points = [
                    models.PointStruct(
                        # Re-running ingestion replaces existing chunk vectors
                        # instead of accumulating duplicates for every run.
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag:{source_type}:{filename}:{index}")),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for index, (chunk, vector) in enumerate(zip(chunks, embeddings))
                ]

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                    wait=True,
                )
                # Remove obsolete tail chunks and legacy random-ID vectors only
                # after the replacement vectors have been accepted. Restrict the
                # cleanup to this source, preserving all newly written IDs.
                qdrant_client.delete(
                    collection_name=settings.QDRANT_COLLECTION,
                    points_selector=models.FilterSelector(filter=models.Filter(
                        must=[
                            models.FieldCondition(key="source", match=models.MatchValue(value=filename)),
                            models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type)),
                        ],
                        must_not=[models.HasIdCondition(has_id=[point.id for point in points])],
                    )),
                    wait=True,
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            raise


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    # Validate the source before allowing --wipe to mutate the remote collection.
    if not os.path.isdir(base_dir):
        raise ValueError(f"Ingestion source is not a directory: {base_dir}")
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):
        # Initialize and probe embeddings before --wipe can remove a working
        # index. Missing credentials or an unavailable model must leave it intact.
        dim = get_embedding_dim()

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )
        else:
            vectors = qdrant_client.get_collection(settings.QDRANT_COLLECTION).config.params.vectors
            stored_dim = getattr(vectors, "size", None)
            if stored_dim != dim:
                raise ValueError(
                    "The Qdrant collection does not match the active embedding dimensions. "
                    "Select a compatible collection or explicitly rebuild the collection with --wipe."
                )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        base_name = os.path.basename(os.path.normpath(base_dir)).lower()
        source_type = explicit_source_type or (
            "true" if "true" in base_name
            else "noisy" if "noisy" in base_name
            else "general"
        )
        # Root-level documents must not disappear just because the directory
        # also contains source folders.
        process_directory(base_dir, source_type)
        for subdir in subdirs:
            source_type = explicit_source_type or (
                "true" if "true" in subdir.lower()
                else "noisy" if "noisy" in subdir.lower()
                else subdir
            )
            process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    # Usage:
    #   python -m rag.ingestion.document_processor data --wipe
    #   python -m rag.ingestion.document_processor data/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else str(PROJECT_ROOT / "data")
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completed.")
