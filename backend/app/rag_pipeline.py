import logging
from typing import Callable, List, Dict, Any, Mapping, Optional, Sequence
from pydantic import BaseModel, Field
from app.config import settings
from app.retrieval import retrieve_documents, retrieve_with_evaluation, RetrievedChunk, RetrievalResponse
from app.llm import GroundingValidationError, generate_grounded_answer

logger = logging.getLogger(__name__)

INSUFFICIENT_ANSWER = "The available MHSSCE documents do not contain enough information to answer this question."


class ChatSource(BaseModel):
    document: str
    page: int
    academic_year: str = ""
    chunk_id: str
    document_type: str = ""
    retrieval_score: Optional[float] = None


class ChatAnswerResponse(BaseModel):
    question: str
    answer: str
    status: str
    sources: List[ChatSource] = Field(default_factory=list)
    conflicting_evidence: bool = False
    requires_personal_context: bool = False
    confidence_note: Optional[str] = None
    generation: str = "abstention"

SYSTEM_PROMPT = """You are the official AI Knowledge Assistant for M. H. Saboo Siddik College of Engineering (MHSSCE), Mumbai.

STRICT GROUNDING RULES:
1. Answer strictly based on the provided context retrieved from the official MHSSCE documents.
2. DO NOT use general world knowledge or hallucinate college-specific facts (e.g. fees, syllabus, admission rules, committee members).
3. If the retrieved documents do NOT contain sufficient information to answer the question, you MUST clearly state:
   "The requested information was not found in the official MHSSCE college knowledge base."
4. Always cite the exact source document and page number for every factual claim made in your response:
   Example: [Source: fee structure 2026-27.pdf, Page 1]
5. Preserve original numbers, academic years, and names exactly as written in the text.
"""

def format_retrieved_context(chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a clear, referenced context block for an LLM prompt.
    """
    if not chunks:
        return "No relevant context found in the college knowledge base."

    context_parts = []
    for idx, chunk in enumerate(chunks, 1):
        header = f"[Document {idx}: {chunk.document} | Page: {chunk.page} | Type: {chunk.document_type}"
        if chunk.academic_year:
            header += f" | AY: {chunk.academic_year}"
        header += f" | Similarity: {chunk.similarity_score:.4f}]"

        context_parts.append(f"{header}\n{chunk.text.strip()}\n")

    return "\n---\n".join(context_parts)

def build_grounded_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    """
    Build the complete prompt containing system instructions, context, and query.
    """
    formatted_context = format_retrieved_context(chunks)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT FROM MHSSCE KNOWLEDGE BASE ---\n"
        f"{formatted_context}\n"
        f"--- END OF CONTEXT ---\n\n"
        f"User Question: {query}\n\n"
        f"Answer:"
    )
    return prompt

class RAGPipeline:
    def __init__(
        self,
        retriever: Optional[Callable[..., RetrievalResponse]] = None,
        answer_generator: Optional[Callable[..., str]] = None,
    ):
        self._retriever = retriever or retrieve_with_evaluation
        self._answer_generator = answer_generator or generate_grounded_answer

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        """
        Run retrieval and assess context sufficiency.
        """
        return self._retriever(query=query, top_k=top_k)

    @staticmethod
    def _source_package(response: RetrievalResponse, include_scores: bool = False) -> tuple[List[ChatSource], List[Dict[str, Any]]]:
        sources: List[ChatSource] = []
        evidence: List[Dict[str, Any]] = []
        for index, chunk in enumerate(response.results[: settings.OPENROUTER_MAX_EVIDENCE_CHUNKS], 1):
            source_id = f"S{index}"
            sources.append(ChatSource(
                document=chunk.document,
                page=chunk.page,
                academic_year=chunk.academic_year,
                chunk_id=chunk.chunk_id,
                document_type=chunk.document_type,
                retrieval_score=chunk.final_score if include_scores else None,
            ))
            evidence.append({
                "source_id": source_id,
                "document": chunk.document,
                "document_type": chunk.document_type,
                "page": chunk.page,
                "academic_year": chunk.academic_year,
                "chunk_id": chunk.chunk_id,
                "retrieval_score": chunk.final_score,
                "text": chunk.text,
            })
        return sources, evidence

    @staticmethod
    def _personal_context_abstention(sources: Sequence[ChatSource]) -> str:
        answer = (
            "The retrieved scholarship document is relevant to scholarship information, but it does not establish "
            "enough applicable eligibility criteria or your student-specific details to decide whether you qualify."
        )
        if sources:
            source = sources[0]
            answer += f" [Source: {source.document}, p. {source.page}]"
        return answer

    @staticmethod
    def _conflict_abstention() -> str:
        return (
            "The retrieved MHSSCE documents contain conflicting references. "
            "I can’t resolve them from the supplied evidence; see the listed sources and pages."
        )

    def ask(self, question: str, top_k: int = settings.DEFAULT_TOP_K, include_scores: bool = False) -> ChatAnswerResponse:
        """Retrieve, validate, and optionally generate a strictly evidence-grounded answer."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        retrieval = self.retrieve(question, top_k=top_k)
        sources, evidence = self._source_package(retrieval, include_scores=include_scores)
        answerability = {
            "status": retrieval.status,
            "answerability_score": retrieval.answerability_score,
            "relevance_score": retrieval.relevance_score,
            "has_sufficient_context": retrieval.has_sufficient_context,
            "requires_personal_context": retrieval.requires_personal_context,
            "temporal_score": retrieval.temporal_score,
            "confidence_note": retrieval.confidence_note,
            "normalized_query": retrieval.normalized_query,
        }
        conflict_information = {
            "conflicting_evidence": retrieval.conflicting_evidence,
            "records": retrieval.role_evidence,
            "note": retrieval.confidence_note if retrieval.conflicting_evidence else None,
        }

        if retrieval.status == "INSUFFICIENT_KNOWLEDGE":
            answer = (
                self._personal_context_abstention(sources)
                if retrieval.requires_personal_context
                else INSUFFICIENT_ANSWER
            )
            return ChatAnswerResponse(
                question=question,
                answer=answer,
                status=retrieval.status,
                sources=sources,
                conflicting_evidence=retrieval.conflicting_evidence,
                requires_personal_context=retrieval.requires_personal_context,
                confidence_note=retrieval.confidence_note,
                generation="abstention",
            )

        if retrieval.status not in {"SUFFICIENT", "CONFLICTING_EVIDENCE"} or not evidence:
            return ChatAnswerResponse(
                question=question,
                answer=INSUFFICIENT_ANSWER,
                status="INSUFFICIENT_KNOWLEDGE",
                sources=sources,
                confidence_note=retrieval.confidence_note,
            )

        try:
            answer = self._answer_generator(
                question=question,
                evidence=evidence,
                answerability=answerability,
                conflict_information=conflict_information,
            )
        except GroundingValidationError as error:
            logger.warning("Grounded response rejected: %s", error)
            return ChatAnswerResponse(
                question=question,
                answer=self._conflict_abstention() if retrieval.conflicting_evidence else INSUFFICIENT_ANSWER,
                status=retrieval.status if retrieval.conflicting_evidence else "INSUFFICIENT_KNOWLEDGE",
                sources=sources,
                conflicting_evidence=retrieval.conflicting_evidence,
                confidence_note="Generated text failed citation validation; returned a safe abstention.",
                generation="abstention",
            )

        return ChatAnswerResponse(
            question=question,
            answer=answer,
            status=retrieval.status,
            sources=sources,
            conflicting_evidence=retrieval.conflicting_evidence,
            requires_personal_context=retrieval.requires_personal_context,
            confidence_note=retrieval.confidence_note,
            generation="openrouter",
        )

    def prepare_llm_input(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieves context and prepares the grounded prompt and citations.
        Ready to be connected to any LLM generator in future milestones.
        """
        response = self.retrieve(query=query, top_k=top_k)
        prompt = build_grounded_prompt(query=query, chunks=response.results)
        
        citations = [
            {
                "document": c.document,
                "page": c.page,
                "document_type": c.document_type,
                "academic_year": c.academic_year,
                "similarity_score": c.similarity_score
            }
            for c in response.results
        ]

        return {
            "query": query,
            "has_sufficient_context": response.has_sufficient_context,
            "confidence_note": response.confidence_note,
            "prompt": prompt,
            "citations": citations,
            "chunks": [c.dict() for c in response.results]
        }
