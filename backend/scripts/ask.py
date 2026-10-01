import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.retrieval import format_evidence_excerpt, retrieve_with_evaluation
from app.rag_pipeline import RAGPipeline
from app.config import settings

pipeline = RAGPipeline()

def process_query(query: str, top_k: int = settings.DEFAULT_TOP_K):
    clean_query = query.strip()
    if not clean_query:
        return

    print("\n" + "=" * 75)
    print(f"QUESTION: \"{clean_query}\"")
    print("=" * 75)

    response = retrieve_with_evaluation(clean_query, top_k=top_k)

    print(f"DETECTED INTENT:       {response.intent}")
    print(f"NORMALIZED QUERY:      {response.normalized_query}")
    print(f"STATUS:                {response.status}")
    print(f"ANSWERABILITY SCORE:   {response.answerability_score:.4f}")
    print(f"TOP COMPOSITE SCORE:   {response.top_composite_score:.4f}")
    print(f"TEMPORAL SCORE:        {response.temporal_score:.4f}")
    print(f"CROSS-ENCODER:         {response.reranker_note}")
    print(f"CONFLICTING EVIDENCE:  {'YES' if response.conflicting_evidence else 'NO'}")
    if response.requires_personal_context:
        print("PERSONAL CONTEXT:      Required; not provided")
    print(f"NOTE:                  {response.confidence_note}")
    if not response.results:
        print("\n[!] No matching records found in the MHSSCE knowledge base.")
        return

    print("\nTOP RETRIEVED SOURCES:")
    print("-" * 75)

    for idx, chunk in enumerate(response.results, 1):
        ay_str = f" | AY: {chunk.academic_year}" if chunk.academic_year else ""
        cross_encoder_score = (
            f"{chunk.cross_encoder_score:.4f} (raw)"
            if chunk.cross_encoder_score is not None
            else "N/A"
        )
        print(f"[{idx}] {chunk.document} (Page {chunk.page}){ay_str}")
        print(f"    Category: {chunk.document_type}")
        print(
            "    Scores: "
            f"[Semantic: {chunk.semantic_score:.4f} | Keyword: {chunk.lexical_score:.4f} | "
            f"Entity: {chunk.entity_score:.4f} | Proximity: {chunk.proximity_score:.4f} | "
            f"Department: {chunk.department_score:.4f} | Qualifier: {chunk.role_qualifier_score:.4f} | "
            f"Temporal: {chunk.temporal_score:.4f} | "
            f"Authority: {chunk.source_authority_score:.4f} | Metadata: {chunk.metadata_score:.4f} | "
            f"Hybrid Final: {chunk.final_score:.4f} | Cross-Encoder: {cross_encoder_score}]"
        )
        if chunk.matched_name or chunk.matched_role:
            print(f"    Matched role: {chunk.matched_name} — {chunk.matched_role}")
        print(f"    Excerpt:")
        print(f"    \"{format_evidence_excerpt(clean_query, chunk.text)}\"")
        print()

    print("=" * 75 + "\n")

def interactive_mode(top_k: int = settings.DEFAULT_TOP_K):
    print("=" * 75)
    print("  M. H. SABOO SIDDIK COLLEGE OF ENGINEERING (MHSSCE)")
    print("  RAG Knowledge Assistant - Interactive Terminal")
    print("=" * 75)
    print("  Indexed Documents: 18 official PDFs (448 pages, 763 chunks)")
    print("  Type your question below.")
    print("  Type 'exit', 'quit', or 'q' to end the session.")
    print("=" * 75 + "\n")

    while True:
        try:
            query = input("Ask MHSSCE >> ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break
            process_query(query, top_k=top_k)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

def main():
    parser = argparse.ArgumentParser(description="Ask questions to the MHSSCE Knowledge Assistant via CLI")
    parser.add_argument("query", nargs="*", help="Optional question string. If omitted, starts interactive session.")
    parser.add_argument("-k", "--top-k", type=int, default=settings.DEFAULT_TOP_K, help=f"Number of retrieved chunks (default: {settings.DEFAULT_TOP_K})")
    args = parser.parse_args()

    if args.query:
        full_query = " ".join(args.query)
        process_query(full_query, top_k=args.top_k)
    else:
        interactive_mode(top_k=args.top_k)

if __name__ == "__main__":
    main()
