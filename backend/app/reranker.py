from functools import lru_cache
from threading import Lock
from time import perf_counter
from typing import Any, List, Optional, Sequence, TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from app.retrieval import RetrievedChunk


class CrossEncoderReranker:
    """Lazily load a sentence-transformers Cross-Encoder and rerank existing chunks."""

    def __init__(self, model_name: Optional[str] = None, batch_size: Optional[int] = None):
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL_NAME
        self.batch_size = batch_size or settings.CROSS_ENCODER_BATCH_SIZE
        self._model: Any = None
        self._load_error: Optional[Exception] = None
        self.last_inference_latency_ms: Optional[float] = None
        self._load_lock = Lock()

    def _load_model(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    if self._load_error is not None:
                        raise RuntimeError(f"Cross-Encoder model {self.model_name!r} could not be loaded.") from self._load_error
                    try:
                        from sentence_transformers import CrossEncoder

                        self._model = CrossEncoder(self.model_name)
                    except Exception as error:
                        self._load_error = error
                        raise RuntimeError(f"Cross-Encoder model {self.model_name!r} could not be loaded.") from error
        return self._model

    @staticmethod
    def _score_value(value: Any) -> float:
        if hasattr(value, "tolist"):
            value = value.tolist()
        while isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Cross-Encoder returned an empty score.")
            value = value[-1]
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def rerank(
        self,
        query: str,
        candidates: Sequence["RetrievedChunk"],
        top_k: Optional[int] = None,
    ) -> List["RetrievedChunk"]:
        """Score (query, chunk text) pairs and order by CE score, then hybrid score."""
        if not candidates:
            return []

        model = self._load_model()
        pairs = [(query, candidate.text) for candidate in candidates]
        started_at = perf_counter()
        raw_scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        self.last_inference_latency_ms = (perf_counter() - started_at) * 1000
        scores = [self._score_value(score) for score in raw_scores]
        if len(scores) != len(candidates):
            raise ValueError(
                f"Cross-Encoder returned {len(scores)} scores for {len(candidates)} candidates."
            )

        reranked = [
            candidate.model_copy(update={"cross_encoder_score": score})
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(
            key=lambda chunk: (chunk.cross_encoder_score, chunk.final_score),
            reverse=True,
        )
        return reranked[:top_k] if top_k is not None else reranked


@lru_cache(maxsize=1)
def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """Return one lazily initialized reranker for this process."""
    return CrossEncoderReranker()
