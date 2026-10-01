import sys
import json
import time
from pathlib import Path

# Add backend directory to sys.path so app modules can be imported
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import settings
from app.document_metadata import normalize_document_academic_years
from app.embeddings import get_embedding_manager
from app.vector_store import get_vector_store

def load_and_validate_chunks(chunks_file: Path):
    """Load chunks and validate required schema fields."""
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found at: {chunks_file}")

    print(f"[BuildDB] Reading chunks from: {chunks_file}")
    with open(chunks_file, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    print(f"[BuildDB] Loaded {len(raw_chunks)} raw chunks from file.")

    required_keys = [
        "chunk_id",
        "document",
        "document_type",
        "page",
        "chunk_number_on_page",
        "academic_year",
        "text"
    ]

    valid_chunks = []
    skipped = 0

    for idx, c in enumerate(raw_chunks):
        missing = [k for k in required_keys if k not in c]
        if missing:
            print(f"[BuildDB] Warning: Chunk at index {idx} missing fields {missing}, skipping.")
            skipped += 1
            continue

        text = c.get("text", "")
        if not text or not str(text).strip():
            print(f"[BuildDB] Warning: Chunk {c.get('chunk_id', idx)} has empty text, skipping.")
            skipped += 1
            continue

        normalized_chunk = {
            "chunk_id": str(c["chunk_id"]).strip(),
            "document": str(c["document"]).strip(),
            "document_type": str(c["document_type"]).strip(),
            "page": int(c["page"]),
            "chunk_number_on_page": int(c["chunk_number_on_page"]),
            "academic_year": str(c.get("academic_year", "")).strip(),
            "text": str(c["text"]).strip()
        }
        if c.get("document_academic_year"):
            normalized_chunk["document_academic_year"] = c["document_academic_year"]
        if isinstance(c.get("document_metadata"), dict):
            normalized_chunk["document_metadata"] = c["document_metadata"]
        valid_chunks.append(normalized_chunk)

    normalized_chunks, document_years, changed_count = normalize_document_academic_years(valid_chunks)
    print(f"[BuildDB] Validation complete: {len(normalized_chunks)} valid chunks, {skipped} skipped.")
    print(f"[BuildDB] Applied document-level academic years to {changed_count} chunks across {len(document_years)} documents.")
    return normalized_chunks

def build_vector_database(batch_size: int = 64):
    """Generate embeddings and upsert into ChromaDB."""
    start_time = time.time()
    chunks = load_and_validate_chunks(settings.CHUNKS_FILE)

    if not chunks:
        print("[BuildDB] No valid chunks to process. Exiting.")
        return

    print("\n" + "=" * 60)
    print("STEP 2: GENERATING EMBEDDINGS & POPULATING CHROMADB")
    print(f"Persist Directory: {settings.CHROMA_PERSIST_DIR}")
    print(f"Collection Name:   {settings.COLLECTION_NAME}")
    print(f"Embedding Model:   {settings.EMBEDDING_MODEL_NAME}")
    print("=" * 60 + "\n")

    embedding_mgr = get_embedding_manager()
    vector_store = get_vector_store()

    total_chunks = len(chunks)
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "document": c["document"],
            "document_type": c["document_type"],
            "page": c["page"],
            "academic_year": c["academic_year"],
            "chunk_number_on_page": c["chunk_number_on_page"]
        }
        for c in chunks
    ]

    print(f"[BuildDB] Generating embeddings for {total_chunks} chunks in batches of {batch_size}...")
    all_embeddings = []

    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        batch_texts = texts[i:end]
        print(f"[BuildDB] Embedding batch {i + 1} to {end} / {total_chunks}...", end="", flush=True)
        batch_emb = embedding_mgr.embed_texts(batch_texts, batch_size=batch_size, show_progress_bar=False)
        all_embeddings.extend(batch_emb)
        print(" Done.")

    print(f"[BuildDB] Embedding complete. Generated {len(all_embeddings)} embedding vectors.")

    print(f"[BuildDB] Upserting records into ChromaDB (collection: '{settings.COLLECTION_NAME}')...")
    vector_store.upsert_chunks(
        chunk_ids=chunk_ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
        batch_size=100
    )

    final_count = vector_store.count()
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("BUILD VECTOR DATABASE COMPLETE")
    print(f"Total Chunks in ChromaDB: {final_count}")
    print(f"Total Time Elapsed:       {duration:.2f} seconds")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    build_vector_database()
