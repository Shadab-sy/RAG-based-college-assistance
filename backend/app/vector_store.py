from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings

class VectorStoreManager:
    _instance: Optional["VectorStoreManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, persist_dir: Optional[Path] = None, collection_name: Optional[str] = None):
        if self._initialized:
            return
        self.persist_dir = Path(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._get_or_create_collection()
        self._initialized = True

    def _get_or_create_collection(self):
        # We specify cosine space for normalized embeddings
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": settings.DISTANCE_METRIC}
        )

    def count(self) -> int:
        """Return the number of items stored in the collection."""
        return self.collection.count()

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 100
    ):
        """
        Upsert chunks safely into ChromaDB in batches.
        Safe against re-runs (will update existing chunk_ids without creating duplicates).
        """
        total = len(chunk_ids)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            b_ids = chunk_ids[i:end_idx]
            b_emb = embeddings[i:end_idx]
            b_docs = documents[i:end_idx]
            b_meta = metadatas[i:end_idx]

            # Ensure all metadata values are valid ChromaDB types (str, int, float, bool)
            clean_meta = []
            for m in b_meta:
                clean_m = {}
                for k, v in m.items():
                    if v is None:
                        clean_m[k] = ""
                    elif isinstance(v, (str, int, float, bool)):
                        clean_m[k] = v
                    else:
                        clean_m[k] = str(v)
                clean_meta.append(clean_m)

            self.collection.upsert(
                ids=b_ids,
                embeddings=b_emb,
                documents=b_docs,
                metadatas=clean_meta
            )
            print(f"[VectorStore] Upserted batch {i + 1} to {end_idx} of {total} chunks.")

    def update_chunk_metadata(
        self,
        chunk_ids: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """Update existing chunk metadata without replacing stored embeddings or documents."""
        if len(chunk_ids) != len(metadatas):
            raise ValueError("chunk_ids and metadatas must have matching lengths")

        for start in range(0, len(chunk_ids), batch_size):
            end = min(start + batch_size, len(chunk_ids))
            clean_metadatas = []
            for metadata in metadatas[start:end]:
                clean_metadatas.append({
                    key: "" if value is None else value if isinstance(value, (str, int, float, bool)) else str(value)
                    for key, value in metadata.items()
                })
            self.collection.update(
                ids=chunk_ids[start:end],
                metadatas=clean_metadatas,
            )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the collection by embedding vector.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def reset_collection(self):
        """Delete and recreate the current collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._get_or_create_collection()

def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()
