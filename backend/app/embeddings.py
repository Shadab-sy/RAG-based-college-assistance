from typing import List, Optional, Any
import numpy as np
from app.config import settings

class EmbeddingManager:
    """
    Embedding manager providing 384-dimensional dense semantic embeddings
    using all-MiniLM-L6-v2 via high-performance ONNX / sentence-transformers.
    """
    _instance: Optional["EmbeddingManager"] = None
    _embed_fn: Any = None
    _initialized: bool = False

    def __new__(cls) -> "EmbeddingManager":
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: Optional[str] = None):
        if self._initialized:
            return
        self.model_name: str = model_name or settings.EMBEDDING_MODEL_NAME
        self._init_model()
        self._initialized = True

    def _init_model(self) -> None:
        """
        Initialize the embedding engine. Prefers local ONNX all-MiniLM-L6-v2
        for fast, lightweight CPU execution without external dependencies.
        """
        try:
            from chromadb.utils import embedding_functions
            self._embed_fn = embedding_functions.DefaultEmbeddingFunction()
            # Warm up with a test embedding
            _ = self._embed_fn(["MHSSCE Embedding Initialization"])
            print(f"[EmbeddingManager] Initialized native ONNX {self.model_name} successfully.")
        except Exception as onnx_err:
            try:
                import importlib
                st_module = importlib.import_module("sentence_transformers")
                st_class = getattr(st_module, "SentenceTransformer")
                self._embed_fn = st_class(self.model_name)
                print(f"[EmbeddingManager] Initialized SentenceTransformer {self.model_name}.")
            except Exception as st_err:
                raise RuntimeError(
                    f"Failed to initialize embedding model '{self.model_name}'. "
                    f"ONNX error: {onnx_err} | SentenceTransformers error: {st_err}"
                )

    def embed_texts(self, texts: List[str], batch_size: int = 64, show_progress_bar: bool = False) -> List[List[float]]:
        """
        Generate normalized embeddings for a list of texts in batches.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            if hasattr(self._embed_fn, "encode"):
                batch_emb = self._embed_fn.encode(
                    batch,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).tolist()
            else:
                # ChromaDB DefaultEmbeddingFunction (ONNX all-MiniLM-L6-v2)
                raw_emb = self._embed_fn(batch)
                batch_emb = []
                for vec in raw_emb:
                    arr = np.array(vec, dtype=np.float32)
                    norm = float(np.linalg.norm(arr))
                    if norm > 0.0:
                        arr = arr / norm
                    batch_emb.append(arr.tolist())

            all_embeddings.extend(batch_emb)

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate normalized embedding for a single query.
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty.")

        res = self.embed_texts([clean_query], batch_size=1)
        return res[0]

    @property
    def dimension(self) -> int:
        return settings.EMBEDDING_DIMENSION

def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()
