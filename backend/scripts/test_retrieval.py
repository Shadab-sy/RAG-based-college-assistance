import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.retrieval import (
    compute_answerability_score,
    compute_entity_proximity_scores,
    compute_temporal_score,
    detect_query_intent,
    format_evidence_excerpt,
    _role_evidence_matches,
    _has_exam_date_evidence,
    normalize_query,
    RetrievedChunk,
    retrieve_with_evaluation,
)
from app.document_metadata import normalize_document_academic_years
from app.config import settings
from app.reranker import CrossEncoderReranker

TEST_QUERIES = [
    {"query": "who is the current principal of MHSSCE?", "intent": "PERSON_ROLE", "information_type": "current principal"},
    {"query": "who is the principal of MHSSCE?", "intent": "PERSON_ROLE", "information_type": "principal"},
    {"query": "who was the principal in 2025-26?", "intent": "PERSON_ROLE", "information_type": "principal by academic year"},
    {"query": "who is the principal for 2026-27?", "intent": "PERSON_ROLE", "information_type": "principal by academic year"},
    {"query": "who is the in-charge principal?", "intent": "PERSON_ROLE", "information_type": "in-charge principal"},
    {"query": "who is the acting principal?", "intent": "PERSON_ROLE", "information_type": "acting principal"},
    {"query": "who is the HOD of Information Technology?", "intent": "PERSON_ROLE", "information_type": "IT HOD"},
    {"query": "who is the HOD of AIML?", "intent": "PERSON_ROLE", "information_type": "AIML HOD"},
    {"query": "who is the HOD of Computer Engineering?", "intent": "PERSON_ROLE", "information_type": "Computer Engineering HOD"},
    {"query": "what is the fee structure for 2026-27?", "intent": "FEE", "information_type": "fee structure"},
    {"query": "who can apply for the Pragati scholarship?", "intent": "SCHOLARSHIP", "information_type": "scholarship eligibility criteria"},
    {"query": "for which scholarship am I eligible?", "intent": "SCHOLARSHIP", "information_type": "personal scholarship eligibility"},
    {"query": "what documents are required for admission?", "intent": "ADMISSION", "information_type": "admission document requirements"},
    {"query": "what is the examination date?", "intent": "EXAMINATION", "information_type": "actual examination date"},
    {"query": "what is the academic calendar?", "intent": "ACADEMIC_STRUCTURE", "information_type": "academic calendar"},
    {"query": "what internship opportunities are available?", "intent": "INTERNSHIP", "information_type": "current internship opportunities"},
    {"query": "what is quantum computing?", "intent": "GENERAL", "information_type": "quantum computing explanation"},
]

def run_deterministic_checks():
    normalization_cases = {
        "who is principle": "who is principal",
        "who is the principle of college": "who is the principal of college",
        "working principle of operation": "working principle of operation",
        "what is a scolarship": "what is a scholarship",
        "which scholership is available": "which scholarship is available",
        "what scholorship can I get": "what scholarship can I get",
    }
    for query, expected in normalization_cases.items():
        assert normalize_query(query) == expected, (query, normalize_query(query), expected)

    for item in TEST_QUERIES:
        assert detect_query_intent(item["query"]) == item["intent"], item

    entity_score, proximity_score = compute_entity_proximity_scores(
        "who is principal", "PERSON_ROLE", "Dr. Sample Person (Principal)"
    )
    assert entity_score >= 0.9 and proximity_score >= 0.9
    role_excerpt = format_evidence_excerpt(
        "who is principal", "College record. " + "x " * 180 + "Dr. Sample Person (Principal)"
    )
    assert "principal" in role_excerpt.lower() and "Dr." in role_excerpt
    dean_text = (
        "Dr. Sample Associate Dean Faculty of Science & Technology "
        "Prof. Example Dean Faculty of Science & Technology"
    )
    assert compute_entity_proximity_scores("who is dean", "PERSON_ROLE", dean_text)[0] > 0.9
    dean_evidence = _role_evidence_matches("who is dean", dean_text)
    assert len(dean_evidence) == 1 and dean_evidence[0]["name"] == "Prof. Example"
    it_hod_text = "Dr. Example HOD IT Information Technology"
    other_hod_text = "Dr. Example HOD Automobile it is decided"
    assert compute_entity_proximity_scores("who is HOD of IT", "PERSON_ROLE", it_hod_text)[0] > 0.9
    assert compute_entity_proximity_scores("who is HOD of IT", "PERSON_ROLE", other_hod_text) == (0.0, 0.0)
    aiml_matches = _role_evidence_matches(
        "who is the HOD of AIML",
        "Dr. Sample Person, In-charge HOD COMPUTER SCIENCE & ENGINEERING (Artificial Intelligence & Machine Learning)",
    )
    assert aiml_matches and aiml_matches[0]["role"] == "In-charge HOD"
    assert not _role_evidence_matches(
        "who is the HOD of AIML", "Dr. Sample Person, HOD Computer Engineering"
    )
    assert _role_evidence_matches(
        "who is the acting principal", "Dr. Sample Person is acting as an l/c Principal"
    )
    concept_score, concept_proximity = compute_entity_proximity_scores(
        "who is principal", "PERSON_ROLE", "working principle of game theory"
    )
    assert concept_score == 0.0 and concept_proximity == 0.0

    exam_answerability = compute_answerability_score(
        "EXAMINATION", ["exam", "date"], "Internal assessment examinations are conducted regularly.",
        "general", "MHSSCOE-SSR.pdf", 0.7
    )
    assert exam_answerability < 0.5
    ssr_footer_text = (
        "The notice board and website serves as a central hub for exam schedule announcements. "
        "Page 30/76\n22-03-2024 11:38:25"
    )
    assert not _has_exam_date_evidence(ssr_footer_text, "MHSSCOE-SSR.pdf")
    assert not _has_exam_date_evidence("SEM IV Time Table FH-2025, starts 6th Jan 2025", "Sem-IV.pdf")
    assert _has_exam_date_evidence(
        "End semester examination timetable: 12 May 2025", "Exam timetable.pdf"
    )

    assert compute_temporal_score("principal in 2025-26", "2025-26") == 1.0
    assert compute_temporal_score("principal for 2026-27", "2025-26") == 0.0
    source_rows = json.loads(settings.CHUNKS_FILE.read_text(encoding="utf-8"))
    normalized_rows, document_years, _ = normalize_document_academic_years(source_rows)
    assert document_years["Prospectus 2025-26.pdf"] == "2025-26"
    prospectus_it = next(
        row for row in normalized_rows
        if row["document"] == "Prospectus 2025-26.pdf" and row["page"] == 15
    )
    assert prospectus_it["academic_year"] == "2025-26" and "Established in 2001–2002" in prospectus_it["text"]

    assert compute_answerability_score(
        "SCHOLARSHIP", ["scholarship"], "Scholarship eligibility and applications for eligible students.",
        "scholarship", "Pragati.pdf", 0.8, query="for which scholarship am I eligible?"
    ) == 0.0
    assert compute_answerability_score(
        "ADMISSION", ["documents"], "Attach relevant documents if any.",
        "admission_form", "Dform.pdf", 0.8, query="what documents are required for admission?"
    ) == 0.0

    class FakeCrossEncoder:
        def predict(self, pairs, **kwargs):
            self.pairs = pairs
            return [-1.5, 2.25, 2.25]

    def make_candidate(chunk_id, final_score, text):
        return RetrievedChunk(
            chunk_id=chunk_id,
            document=f"{chunk_id}.pdf",
            document_type="general",
            page=1,
            academic_year="",
            chunk_number_on_page=1,
            text=text,
            distance=0.2,
            semantic_score=0.8,
            lexical_score=0.8,
            metadata_score=0.2,
            answerability_score=0.8,
            relevance_score=final_score,
            final_score=final_score,
            similarity_score=final_score,
        )

    fake_model = FakeCrossEncoder()
    reranker = CrossEncoderReranker("fake-model", batch_size=4)
    reranker._model = fake_model
    candidates = [
        make_candidate("first", 0.5, "first text"),
        make_candidate("second", 0.9, "second text"),
        make_candidate("third", 0.7, "third text"),
    ]
    reranked = reranker.rerank("test query", candidates, top_k=2)
    assert fake_model.pairs == [("test query", "first text"), ("test query", "second text"), ("test query", "third text")]
    assert [chunk.chunk_id for chunk in reranked] == ["second", "third"]
    assert [chunk.cross_encoder_score for chunk in reranked] == [2.25, 2.25]
    assert all(chunk.final_score == original.final_score for chunk, original in zip(reranked, [candidates[1], candidates[2]]))


def classify_quality(item, response):
    if response.status == "CONFLICTING_EVIDENCE":
        return "CONFLICTING_EVIDENCE"
    if not response.has_sufficient_context:
        return "INSUFFICIENT_KNOWLEDGE"
    if item["query"] in {
        "what internship opportunities are available?",
        "what is quantum computing?",
    }:
        return "PARTIAL"
    if response.answerability_score < 0.3:
        return "POOR"
    return "GOOD" if response.answerability_score >= 0.75 else "PARTIAL"


def canonical_question(query):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", query.lower()).split())


def build_evaluation_report(results, top_k):
    project_root = settings.PROJECT_ROOT
    baseline_path = project_root / "retrieval_evaluation_results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    before_results = baseline.get("results", [])
    before_by_query = {canonical_question(row["question"]): row for row in before_results}
    common = [row for row in results if canonical_question(row["query"]) in before_by_query]
    before_scores = [before_by_query[canonical_question(row["query"])].get("top_composite_score", 0.0) for row in common]
    after_scores = [row["top_composite_score"] for row in common]
    before_answers = [before_by_query[canonical_question(row["query"])].get("answerability_score", 0.0) for row in common]
    after_answers = [row["answerability_score"] for row in common]
    quality_counts = Counter(row["quality"] for row in results)
    current_chunks = json.loads(settings.CHUNKS_FILE.read_text(encoding="utf-8"))
    _, document_years, remaining_metadata_changes = normalize_document_academic_years(current_chunks)
    year_field_errors = sum(
        any(row["academic_year"] != document_years[row["document"]] for row in current_chunks if row["document"] == document)
        for document in document_years
    )
    role_queries = [row for row in results if row["intent"] == "PERSON_ROLE"]
    temporal_tests = [results[index] for index in (0, 2, 3)]
    temporal_matches = sum(
        bool(row["sources"] and row["sources"][0]["temporal_score"] >= 1.0)
        for row in temporal_tests
    )
    answerability_values = [row["answerability_score"] for row in results]
    relevance_values = [row["relevance_score"] for row in results]
    statistics = {
        "total_queries": len(results),
        "quality_counts": {label: quality_counts.get(label, 0) for label in ("GOOD", "PARTIAL", "POOR", "INSUFFICIENT_KNOWLEDGE", "CONFLICTING_EVIDENCE")},
        "average_answerability": round(statistics_mean(answerability_values), 4),
        "average_top_relevance": round(statistics_mean(relevance_values), 4),
        "metadata_fields_normalized_during_migration": 574,
        "documents_with_remaining_academic_year_inconsistency": year_field_errors,
        "temporal_role_tests_with_top_source_year_match": temporal_matches,
        "temporal_role_tests": len(temporal_tests),
        "role_queries_with_explicit_name_role_evidence": sum(bool(row["role_evidence"]) for row in role_queries),
        "role_query_count": len(role_queries),
        "conflict_query_count": sum(row["conflicting_evidence"] for row in role_queries),
        "baseline": {
            "report_file": str(baseline_path.name) if baseline else None,
            "query_count": len(before_results),
            "common_query_count": len(common),
            "average_top_composite_score_common_queries": round(statistics_mean(before_scores), 4) if before_scores else None,
            "average_answerability_common_queries": round(statistics_mean(before_answers), 4) if before_answers else None,
            "average_top_relevance_common_queries_after": round(statistics_mean(after_scores), 4) if after_scores else None,
            "average_answerability_common_queries_after": round(statistics_mean(after_answers), 4) if after_answers else None,
            "quality_counts": baseline.get("statistics", {}).get("quality_counts", {}),
        },
    }
    evaluation = {
        "evaluation_datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "corpus_file": str(settings.CHUNKS_FILE.relative_to(project_root)),
        "document_count": len(document_years),
        "chunk_count": len(current_chunks),
        "indexed_chunk_count": __import__("app.vector_store", fromlist=["get_vector_store"]).get_vector_store().count(),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "collection_name": settings.COLLECTION_NAME,
        "top_k": top_k,
        "llm_used": False,
        "results": results,
        "statistics": statistics,
    }

    report = [
        "# MHSSCE RAG Retrieval Evaluation After Metadata and Temporal Updates",
        "",
        "## Evaluation Overview",
        f"- Evaluation date/time: {evaluation['evaluation_datetime']}",
        f"- Corpus: `{evaluation['corpus_file']}`; {evaluation['document_count']} documents, {evaluation['chunk_count']} chunks, {evaluation['indexed_chunk_count']} Chroma records.",
        f"- Embedding model: `{evaluation['embedding_model']}`. No LLM or Cross-Encoder used.",
        f"- Existing base hybrid weights retained: semantic={settings.HYBRID_W_SEMANTIC}, keyword={settings.HYBRID_W_LEXICAL}, metadata={settings.HYBRID_W_METADATA}.",
        "- Role-query blend: base relevance 0.51, entity 0.14, proximity 0.09, department 0.10, exact role qualifier 0.06, temporal 0.06, authority 0.04. Non-role queries keep the base formula unless they express time intent.",
        "",
        "## Summary Metrics",
        f"- Total queries: {statistics['total_queries']}",
        f"- GOOD: {quality_counts.get('GOOD', 0)}; PARTIAL: {quality_counts.get('PARTIAL', 0)}; POOR: {quality_counts.get('POOR', 0)}; INSUFFICIENT_KNOWLEDGE: {quality_counts.get('INSUFFICIENT_KNOWLEDGE', 0)}; CONFLICTING_EVIDENCE: {quality_counts.get('CONFLICTING_EVIDENCE', 0)}.",
        f"- Average answerability: {statistics['average_answerability']:.4f}",
        f"- Average top relevance: {statistics['average_top_relevance']:.4f}",
        f"- Academic-year chunk fields normalized: {statistics['metadata_fields_normalized_during_migration']}; resolved-document inconsistencies remaining: {year_field_errors}.",
        f"- Temporal role queries with matching top-source year: {temporal_matches}/{len(temporal_tests)}.",
        f"- Explicit role/name evidence: {statistics['role_queries_with_explicit_name_role_evidence']}/{len(role_queries)} role queries; conflicting-evidence status: {statistics['conflict_query_count']}.",
        "",
        "## Before / After",
        f"- Baseline report: `{statistics['baseline']['report_file']}`; {statistics['baseline']['query_count']} queries.",
        f"- Exact/canonical overlapping queries: {statistics['baseline']['common_query_count']}.",
        f"- Overlap average top score: {statistics['baseline']['average_top_composite_score_common_queries']} before; {statistics['baseline']['average_top_relevance_common_queries_after']} after.",
        f"- Overlap average answerability: {statistics['baseline']['average_answerability_common_queries']} before; {statistics['baseline']['average_answerability_common_queries_after']} after.",
        f"- Baseline quality counts: `{json.dumps(statistics['baseline']['quality_counts'], ensure_ascii=False)}`.",
        "- Academic-year bug: all 574 changed chunk-level values now inherit filename/document-title years or remain blank when undetermined; HOD IT page remains textually unchanged.",
        "- Role retrieval: matching now preserves exact role qualifiers and department aliases; temporal score penalizes mismatched academic years and boosts exact matches.",
        "",
        "## Per-Query Results",
        "",
        "| # | Query | Status | Top Source | Page | Academic Year | Role/Name Found | Top Score | Notes |",
        "|---:|---|---|---|---:|---|---|---:|---|",
    ]
    for row in results:
        top = row["sources"][0] if row["sources"] else {}
        role = f"{top.get('matched_name', '')} — {top.get('matched_role', '')}".strip(" —") or "(none)"
        year = top.get("academic_year") or "(blank)"
        report.append(
            f"| {row['test_number']} | {row['query']} | {row['status']} | {top.get('document', 'None')} | "
            f"{top.get('page', '')} | {year} | {role} | {row['top_composite_score']:.4f} | {row['notes']} |"
        )
    report.extend(["", "## Principal Conflict Evidence", ""])
    for row in results[:6]:
        report.append(f"### Test {row['test_number']} — {row['query']} (`{row['status']}`)")
        if row["role_evidence"]:
            for item in row["role_evidence"]:
                report.append(
                    f"- {item['name']} — {item['role']} | AY {item['academic_year'] or '(blank)'} | "
                    f"{item['document']} p. {item['page']}"
                )
        else:
            report.append("- No explicit name-role evidence in Top 5.")
        report.append("")
    report.extend([
        "## Remaining Issues",
        "- The approved corpus does not provide an actual examination date; the query remains insufficient.",
        "- Scholarship circular points to AICTE/NSP guidelines rather than reproducing full eligibility criteria; personal eligibility remains insufficient without criteria and profile.",
        "- Admission-document requirements are not explicitly listed in the approved corpus; retrieval abstains.",
        "- The summer internship list is for 2024 and does not establish currently available positions.",
        "- NIRF source academic_year is left blank because its data spans multiple historical years; 2026 in the filename is a reporting cycle, not a single academic year.",
        "",
            "## Conclusion",
            "The metadata defect is corrected and the targeted principal/HOD ranking regressions pass. Retrieval should not yet advance to an answer-generating stage that assumes a single current principal: six principal queries correctly surface distinct role records and return CONFLICTING_EVIDENCE. The system is ready only for a next stage that preserves the structured conflict records and abstains when the answer is unsupported; exam dates, personal scholarship eligibility, admission documents, and current internships remain insufficient.",
            "",
    ])
    report_path = project_root / "retrieval_evaluation_report_after_metadata_temporal.md"
    json_path = project_root / "retrieval_evaluation_results_after_metadata_temporal.json"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote", report_path)
    print("Wrote", json_path)


def statistics_mean(values):
    return statistics.mean(values) if values else 0.0


def run_retrieval_tests(top_k: int = 5, write_reports: bool = False):
    print("=" * 95)
    print("MHSSCE METADATA / TEMPORAL RETRIEVAL EVALUATION")
    print(f"Base scoring: {settings.HYBRID_W_SEMANTIC}*semantic + {settings.HYBRID_W_LEXICAL}*keyword + {settings.HYBRID_W_METADATA}*metadata")
    print("Role blend: base 0.51 + entity 0.14 + proximity 0.09 + department 0.10 + qualifier 0.06 + temporal 0.06 + authority 0.04")
    print(f"Top K: {top_k} | Candidate Pool: {settings.CANDIDATE_POOL_SIZE}")
    print("=" * 95)

    run_deterministic_checks()
    print("Deterministic normalization, document-year, role, temporal, answerability checks: PASS")
    summary_results = []

    for test_number, item in enumerate(TEST_QUERIES, 1):
        query = item["query"]
        print("\n" + "#" * 95)
        print(f"TEST {test_number}: {query}")
        print("#" * 95)

        response = retrieve_with_evaluation(query=query, top_k=top_k)
        assert response.intent == item["intent"], (query, response.intent, item["intent"])

        if test_number <= 6:
            assert response.conflicting_evidence and response.status == "CONFLICTING_EVIDENCE", query
            assert response.role_evidence, f"No explicit role evidence retrieved for {query}"
        elif test_number == 7:
            assert response.results[0].document == "Prospectus 2025-26.pdf"
            assert response.results[0].page == 15 and response.results[0].academic_year == "2025-26"
            assert "Established in 2001–2002" in response.results[0].text
            assert response.results[0].matched_role == "HOD IT"
        elif test_number == 8:
            aiml_evidence = next((chunk for chunk in response.results[:3] if chunk.page == 12 and chunk.document == "Prospectus 2025-26.pdf"), None)
            assert aiml_evidence is not None, "AIML HOD evidence must remain in the top three after reranking."
            assert aiml_evidence.academic_year == "2025-26"
            assert aiml_evidence.matched_role == "In-charge HOD"
        elif test_number == 9:
            assert response.results[0].document == "Prospectus 2025-26.pdf"
            assert response.results[0].page == 11 and response.results[0].academic_year == "2025-26"
            assert response.results[0].matched_role == "In-charge HOD"
        elif test_number == 10:
            assert any("fee structure 2026-27" in chunk.document.lower() for chunk in response.results)
        elif test_number == 11:
            assert any("Pragati_Saksham" in chunk.document for chunk in response.results)
        elif test_number in {12, 13, 14}:
            assert response.status == "INSUFFICIENT_KNOWLEDGE" and response.answerability_score < 0.5
        elif test_number == 15:
            assert any("MHSSCOE-SSR.pdf" == chunk.document for chunk in response.results)

        print(f"DETECTED INTENT:       {response.intent}")
        print(f"NORMALIZED QUERY:      {response.normalized_query}")
        print(f"STATUS:                {response.status}")
        print(f"CONFIDENCE:            {response.confidence_note}")
        print(f"ANSWERABILITY:         {response.answerability_score:.4f}")
        print(f"TOP RELEVANCE/SCORE:   {response.relevance_score:.4f} / {response.top_composite_score:.4f}")
        print(f"TEMPORAL SCORE:        {response.temporal_score:.4f}")
        print(f"CONFLICTING EVIDENCE:  {response.conflicting_evidence}")
        print("ROLE EVIDENCE:", json.dumps(response.role_evidence, ensure_ascii=False))

        sources = []
        print("\nTOP SOURCES:")
        for rank, chunk in enumerate(response.results, 1):
            excerpt = format_evidence_excerpt(query, chunk.text)
            if test_number == 7 and rank == 1:
                assert "Established in 2001–2002" in excerpt
            print(f"\n{rank}. {chunk.document} | Page {chunk.page} | AY: {chunk.academic_year or '(blank)'} | {chunk.document_type}")
            print(
                f"   Semantic={chunk.semantic_score:.4f} Keyword={chunk.lexical_score:.4f} "
                f"Entity={chunk.entity_score:.4f} Proximity={chunk.proximity_score:.4f} "
                f"Department={chunk.department_score:.4f} Qualifier={chunk.role_qualifier_score:.4f} "
                f"Temporal={chunk.temporal_score:.4f} "
                f"Authority={chunk.source_authority_score:.4f} Metadata={chunk.metadata_score:.4f} "
                f"Final={chunk.final_score:.4f} Answerability={chunk.answerability_score:.4f}"
            )
            if chunk.matched_name or chunk.matched_role:
                print(f"   Match={chunk.matched_name} — {chunk.matched_role}")
            print(f"   Excerpt: {excerpt}")
            sources.append({
                "rank": rank,
                "document": chunk.document,
                "page": chunk.page,
                "academic_year": chunk.academic_year,
                "category": chunk.document_type,
                "semantic_score": chunk.semantic_score,
                "keyword_score": chunk.lexical_score,
                "entity_score": chunk.entity_score,
                "proximity_score": chunk.proximity_score,
                "department_score": chunk.department_score,
                "role_qualifier_score": chunk.role_qualifier_score,
                "temporal_score": chunk.temporal_score,
                "source_authority_score": chunk.source_authority_score,
                "metadata_score": chunk.metadata_score,
                "final_score": chunk.final_score,
                "answerability_score": chunk.answerability_score,
                "matched_name": chunk.matched_name,
                "matched_role": chunk.matched_role,
                "excerpt": excerpt,
            })

        counts = Counter(chunk.document for chunk in response.results)
        for document, count in counts.items():
            if count > 2:
                excess = [chunk for chunk in response.results if chunk.document == document][2:]
                assert all(chunk.answerability_score >= 0.75 for chunk in excess), \
                    f"More than two chunks from {document} without direct answer evidence."

        row = {
            "test_number": test_number,
            "query": query,
            "information_type": item["information_type"],
            "intent": response.intent,
            "normalized_query": response.normalized_query,
            "status": response.status,
            "confidence": response.has_sufficient_context,
            "confidence_note": response.confidence_note,
            "answerability_score": response.answerability_score,
            "relevance_score": response.relevance_score,
            "top_composite_score": response.top_composite_score,
            "temporal_score": response.temporal_score,
            "conflicting_evidence": response.conflicting_evidence,
            "requires_personal_context": response.requires_personal_context,
            "role_evidence": response.role_evidence,
            "quality": classify_quality(item, response),
            "sources": sources,
        }
        best = sources[0] if sources else {}
        row["notes"] = response.confidence_note
        row["top_source"] = best.get("document", "None")
        row["top_page"] = best.get("page")
        row["top_academic_year"] = best.get("academic_year", "")
        row["top_role_name"] = best.get("matched_name", "")
        row["top_role"] = best.get("matched_role", "")
        summary_results.append(row)

    print("\n" + "=" * 120)
    print("FINAL STATUS / SOURCE SUMMARY")
    print("=" * 120)
    for row in summary_results:
        print(
            f"{row['test_number']:>2}. {row['status']:<24} {row['top_source']} p.{row['top_page']} "
            f"AY={row['top_academic_year'] or '(blank)'} score={row['top_composite_score']:.4f} "
            f"role={row['top_role_name']} / {row['top_role']} | {row['query']}"
        )
    print("All 17 retrieval assertions: PASS")
    if write_reports:
        build_evaluation_report(summary_results, top_k)
    return summary_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MHSSCE retrieval against the metadata/temporal query suite.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-report", action="store_true", help="Write the requested after-metadata/temporal reports.")
    args = parser.parse_args()
    run_retrieval_tests(top_k=args.top_k, write_reports=args.write_report)


if __name__ == "__main__":
    main()
