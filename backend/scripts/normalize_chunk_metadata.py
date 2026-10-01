import argparse
import json
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import settings
from app.document_metadata import normalize_document_academic_years
from app.vector_store import get_vector_store


def metadata_for_chunk(chunk):
    return {
        "chunk_id": chunk["chunk_id"],
        "document": chunk["document"],
        "document_type": chunk["document_type"],
        "page": int(chunk["page"]),
        "academic_year": chunk["academic_year"],
        "chunk_number_on_page": int(chunk["chunk_number_on_page"]),
    }


def normalize_existing_metadata(apply_changes: bool = False) -> int:
    chunks_file = settings.CHUNKS_FILE
    original_chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    normalized_chunks, document_years, changed_count = normalize_document_academic_years(original_chunks)

    print(f"Chunks: {len(original_chunks)} | Documents: {len(document_years)} | Academic-year fields changed: {changed_count}")
    for document, academic_year in sorted(document_years.items()):
        print(f"{document}: {academic_year or '(blank)'}")

    if not apply_changes:
        print("Dry run only. Pass --apply to update chunks.json and existing Chroma metadata.")
        return changed_count

    vector_store = get_vector_store()
    if vector_store.count() != len(normalized_chunks):
        raise RuntimeError(
            f"Chunk/index count mismatch ({len(normalized_chunks)} chunks, {vector_store.count()} indexed); refusing metadata update."
        )

    vector_store.update_chunk_metadata(
        chunk_ids=[chunk["chunk_id"] for chunk in normalized_chunks],
        metadatas=[metadata_for_chunk(chunk) for chunk in normalized_chunks],
    )

    temp_file = chunks_file.with_suffix(chunks_file.suffix + ".tmp")
    temp_file.write_text(json.dumps(normalized_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_file.replace(chunks_file)

    sample_id = next(
        chunk["chunk_id"]
        for chunk in normalized_chunks
        if chunk["document"] == "Prospectus 2025-26.pdf" and chunk["page"] == 15
    )
    indexed = vector_store.collection.get(ids=[sample_id], include=["metadatas"])
    indexed_year = indexed["metadatas"][0].get("academic_year", "")
    if indexed_year != "2025-26":
        raise RuntimeError(f"Chroma metadata verification failed: expected 2025-26, got {indexed_year!r}")

    print(f"Updated {len(normalized_chunks)} Chroma metadata records without regenerating embeddings.")
    print(f"Updated chunk metadata at: {chunks_file}")
    print(f"Verified {sample_id}: academic_year={indexed_year}")
    return changed_count


def main():
    parser = argparse.ArgumentParser(description="Normalize academic_year to document-level metadata.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update chunks.json and existing Chroma metadata. Without this flag, only report planned changes.",
    )
    args = parser.parse_args()
    normalize_existing_metadata(apply_changes=args.apply)


if __name__ == "__main__":
    main()
