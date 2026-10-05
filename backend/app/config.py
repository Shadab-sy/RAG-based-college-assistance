from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Base paths
    APP_DIR: Path = Path(__file__).resolve().parent
    BACKEND_DIR: Path = APP_DIR.parent
    PROJECT_ROOT: Path = BACKEND_DIR.parent

    # Data paths
    CHUNKS_FILE: Path = PROJECT_ROOT / "MHSSCE_CHUNKS" / "chunks.json"
    CHROMA_PERSIST_DIR: Path = BACKEND_DIR / "chroma_db"

    # Vector DB & Embeddings
    COLLECTION_NAME: str = "mhssce_college_knowledge"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    DISTANCE_METRIC: str = "cosine"  # cosine distance for normalized embeddings

    # Retrieval parameters
    DEFAULT_TOP_K: int = 5
    CANDIDATE_POOL_SIZE: int = 25  # Number of candidate chunks retrieved before hybrid reranking
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_BATCH_SIZE: int = 32
    # Experimental only; production RAG uses the validated hybrid ranking.
    CROSS_ENCODER_ENABLED: bool = False
    CROSS_ENCODER_ALLOW_HYBRID_FALLBACK: bool = True

    # OpenRouter generation; the key is optional at startup and checked lazily.
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = Field(
        default="qwen/qwen3.8-27b:free",
        validation_alias="OPENROUTER_MODEL",
    )
    OPENROUTER_FALLBACK_MODELS: str = Field(default="", validation_alias="OPENROUTER_FALLBACK_MODELS")
    OPENROUTER_MAX_EVIDENCE_CHUNKS: int = Field(default=5, validation_alias="OPENROUTER_MAX_EVIDENCE_CHUNKS")
    OPENROUTER_MAX_CHUNK_CHARS: int = Field(default=1800, validation_alias="OPENROUTER_MAX_CHUNK_CHARS")
    OPENROUTER_MAX_OUTPUT_TOKENS: int = Field(default=4096, validation_alias="OPENROUTER_MAX_OUTPUT_TOKENS")
    OPENROUTER_TEMPERATURE: float = Field(default=0.2, validation_alias="OPENROUTER_TEMPERATURE")

    # Hybrid Retrieval Weights (Normalized sum = 1.0)
    # final_score = W_SEMANTIC * semantic + W_LEXICAL * lexical + W_METADATA * metadata
    HYBRID_W_SEMANTIC: float = 0.60
    HYBRID_W_LEXICAL: float = 0.25
    HYBRID_W_METADATA: float = 0.15

    # PERSON_ROLE scoring weights; the base hybrid relevance remains intact as a component.
    ROLE_W_BASE_RELEVANCE: float = 0.51
    ROLE_W_ENTITY: float = 0.14
    ROLE_W_PROXIMITY: float = 0.09
    ROLE_W_DEPARTMENT: float = 0.10
    ROLE_W_QUALIFIER: float = 0.06
    ROLE_W_TEMPORAL: float = 0.06
    ROLE_W_SOURCE_AUTHORITY: float = 0.04
    TEMPORAL_W_BASE_RELEVANCE: float = 0.90
    TEMPORAL_W_MATCH: float = 0.10

    # Threshold for flagging out-of-domain / insufficient context
    SIMILARITY_THRESHOLD: float = 0.35

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        arbitrary_types_allowed=True,
    )

settings = Settings()
