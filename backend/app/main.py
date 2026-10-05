from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.config import settings
from app.retrieval import retrieve_documents, retrieve_with_evaluation, RetrievalResponse
from app.rag_pipeline import RAGPipeline, ChatAnswerResponse
from app.llm import LLMConfigurationError, OpenRouterAPIError
from app.vector_store import get_vector_store

app = FastAPI(
    title="MHSSCE RAG Assistant Backend API",
    description="Backend retrieval and RAG API for M. H. Saboo Siddik College of Engineering Knowledge Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

pipeline = RAGPipeline()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

@app.get("/")
def read_root():
    vector_store = get_vector_store()
    chunk_count = vector_store.count()
    return {
        "project": "MHSSCE RAG College Knowledge Assistant",
        "college": "M. H. Saboo Siddik College of Engineering, Mumbai",
        "status": "online",
        "collection": settings.COLLECTION_NAME,
        "indexed_chunks": chunk_count,
        "embedding_model": settings.EMBEDDING_MODEL_NAME
    }

@app.get("/health")
def health_check():
    vector_store = get_vector_store()
    return {
        "status": "healthy",
        "chroma_db_indexed_chunks": vector_store.count()
    }


@app.post("/api/ask", response_model=ChatAnswerResponse, response_model_exclude_none=True)
def ask(
    request: AskRequest,
    top_k: int = Query(default=settings.DEFAULT_TOP_K, ge=1, le=20),
    include_scores: bool = Query(default=False, description="Include internal retrieval scores for debugging"),
):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return pipeline.ask(question, top_k=top_k, include_scores=include_scores)
    except LLMConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OpenRouterAPIError as error:
        raise HTTPException(status_code=error.http_status, detail=str(error)) from error

@app.get("/retrieve", response_model=RetrievalResponse)
def retrieve(
    query: str = Query(..., description="User search query"),
    top_k: int = Query(default=settings.DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to retrieve")
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return pipeline.retrieve(query=query, top_k=top_k)

@app.get("/prepare-rag-prompt")
def prepare_prompt(
    query: str = Query(..., description="User search query"),
    top_k: int = Query(default=settings.DEFAULT_TOP_K, ge=1, le=20)
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return pipeline.prepare_llm_input(query=query, top_k=top_k)
