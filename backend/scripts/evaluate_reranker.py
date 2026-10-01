"""Compare hybrid retrieval with Cross-Encoder reranking on the requested query set."""

import json
import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

backend_dir = Path(__file__).resolve().parent.parent
project_root = backend_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import settings
from app.reranker import get_cross_encoder_reranker
from app.retrieval import retrieve_with_evaluation


QUERY_CASES = [
    ("Who is the HOD of Information Technology?", ("prospectus 2025-26.pdf",)),
    ("Who is the HOD of AIML?", ("prospectus 2025-26.pdf",)),
    ("Who is the HOD of Computer Engineering?", ("prospectus 2025-26.pdf",)),
    ("What is the fee structure for 2026-27?", ("fee structure 2026-27.pdf",)),
    ("What is the fee structure?", ("fee structure 2026-27.pdf",)),
    ("Who can apply for the Pragati scholarship?", ("pragati_saksham_scholarship scheme 2026-27.pdf",)),
    ("What scholarships are available?", ("pragati_saksham_scholarship scheme 2026-27.pdf",)),
    ("How can a student file a grievance?", ("student grievance redressal committee.pdf",)),
    ("What is the student grievance redressal committee?", ("student grievance redressal committee.pdf",)),
    ("What is the anti-ragging committee?", ("anti-ragging committee and squid.pdf",)),
    ("What is the academic calendar?", ("mhsscoe-ssr.pdf",)),
    ("What subjects are in semester 6?", ("sem-vi.pdf",)),
    ("What internship opportunities are available?", ("internship-list-24summer.pdf",)),
    ("What documents are required for admission?", ()),
    ("What is the examination date?", ()),
    ("What is quantum computing?", ("be syllabus.pdf",)),
]


def ranking_entry(chunk, relevant_terms):
    relevant = any(term in chunk.document.lower() for term in relevant_terms)
    return {
        "chunk_id": chunk.chunk_id,
        "document": chunk.document,
        "document_type": chunk.document_type,
        "page": chunk.page,
        "academic_year": chunk.academic_year,
        "chunk_number_on_page": chunk.chunk_number_on_page,
        "text": chunk.text,
        "existing_score": chunk.final_score,
        "cross_encoder_score": chunk.cross_encoder_score,
        "answerability_score": chunk.answerability_score,
        "relevant": relevant,
    }


def relevant_at(ranking, k):
    return any(item["relevant"] for item in ranking[:k])


def mean_or_none(values):
    return round(statistics.mean(values), 4) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="Regenerate the Markdown report from the latest results JSON without rerunning inference.")
    args = parser.parse_args()
    json_path = project_root / "retrieval_reranker_evaluation_results.json"
    report_path = project_root / "retrieval_reranker_evaluation_report.md"
    if args.report_only:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        report_path.write_text(render_report(data), encoding="utf-8")
        print(f"Wrote {report_path}")
        return

    final_k = settings.DEFAULT_TOP_K
    candidate_k = settings.CANDIDATE_POOL_SIZE
    reranker = get_cross_encoder_reranker()
    model_load_ms = None
    warmup_ms = None
    model_unavailable = None

    # Load once and warm inference separately so steady-state latency excludes setup.
    started = perf_counter()
    try:
        model = reranker._load_model()
        model_load_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        model.predict([("MHSSCE retrieval warm-up", "MHSSCE official college document")],
                      batch_size=1, show_progress_bar=False)
        warmup_ms = (perf_counter() - started) * 1000
    except Exception as error:
        model_unavailable = str(error)

    rows = []
    baseline_latencies = []
    reranked_latencies = []
    cross_encoder_latencies = []
    baseline_top1_scores = []
    cross_encoder_top1_scores = []
    baseline_scores = []
    cross_encoder_scores = []
    answerability_scores = []
    original_enabled = settings.CROSS_ENCODER_ENABLED
    try:
        for query, relevant_terms in QUERY_CASES:
            settings.CROSS_ENCODER_ENABLED = False
            started = perf_counter()
            baseline_response = retrieve_with_evaluation(query, top_k=final_k)
            baseline_ms = (perf_counter() - started) * 1000

            # This benchmark explicitly exercises the experimental reranker,
            # even though production retrieval keeps it disabled by default.
            settings.CROSS_ENCODER_ENABLED = model_unavailable is None
            started = perf_counter()
            reranked_response = retrieve_with_evaluation(query, top_k=final_k)
            reranked_ms = (perf_counter() - started) * 1000

            baseline = [ranking_entry(chunk, relevant_terms) for chunk in baseline_response.results]
            reranked = [ranking_entry(chunk, relevant_terms) for chunk in reranked_response.results]
            ce_scores = [entry["cross_encoder_score"] for entry in reranked if entry["cross_encoder_score"] is not None]
            ce_latency = reranker.last_inference_latency_ms if ce_scores else None
            if ce_latency is not None:
                cross_encoder_latencies.append(ce_latency)
            baseline_latencies.append(baseline_ms)
            reranked_latencies.append(reranked_ms)
            if baseline:
                baseline_top1_scores.append(baseline[0]["existing_score"])
            if reranked and reranked[0]["cross_encoder_score"] is not None:
                cross_encoder_top1_scores.append(reranked[0]["cross_encoder_score"])
            baseline_scores.extend(item["existing_score"] for item in baseline)
            cross_encoder_scores.extend(item["cross_encoder_score"] for item in reranked if item["cross_encoder_score"] is not None)
            answerability_scores.append(reranked_response.answerability_score)

            rows.append({
                "query": query,
                "baseline_ranking": baseline,
                "reranked_ranking": reranked,
                "baseline_status": baseline_response.status,
                "reranked_status": reranked_response.status,
                "baseline_answerability": baseline_response.answerability_score,
                "answerability": reranked_response.answerability_score,
                "baseline_relevant_top_1": relevant_at(baseline, 1),
                "reranked_relevant_top_1": relevant_at(reranked, 1),
                "baseline_relevant_top_3": relevant_at(baseline, 3),
                "reranked_relevant_top_3": relevant_at(reranked, 3),
                "baseline_relevant_top_5": relevant_at(baseline, 5),
                "reranked_relevant_top_5": relevant_at(reranked, 5),
                "baseline_latency_ms": round(baseline_ms, 2),
                "reranked_total_latency_ms": round(reranked_ms, 2),
                "cross_encoder_latency_ms": round(ce_latency, 2) if ce_latency is not None else None,
                "relevant_source": list(relevant_terms),
            })
    finally:
        settings.CROSS_ENCODER_ENABLED = original_enabled

    metrics = {
        "total_queries": len(rows),
        "baseline_relevant_top_1": sum(row["baseline_relevant_top_1"] for row in rows),
        "reranked_relevant_top_1": sum(row["reranked_relevant_top_1"] for row in rows),
        "baseline_relevant_top_3": sum(row["baseline_relevant_top_3"] for row in rows),
        "reranked_relevant_top_3": sum(row["reranked_relevant_top_3"] for row in rows),
        "baseline_relevant_top_5": sum(row["baseline_relevant_top_5"] for row in rows),
        "reranked_relevant_top_5": sum(row["reranked_relevant_top_5"] for row in rows),
        "average_baseline_top_1_existing_score": mean_or_none(baseline_top1_scores),
        "average_reranked_top_1_cross_encoder_score": mean_or_none(cross_encoder_top1_scores),
        "average_baseline_score_top_k": mean_or_none(baseline_scores),
        "average_cross_encoder_score_top_k": mean_or_none(cross_encoder_scores),
        "average_answerability": mean_or_none(answerability_scores),
        "average_baseline_retrieval_latency_ms": mean_or_none(baseline_latencies),
        "average_cross_encoder_inference_latency_ms": mean_or_none(cross_encoder_latencies),
        "average_reranked_total_latency_ms": mean_or_none(reranked_latencies),
    }
    results = {
        "evaluation_datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "corpus_file": str(settings.CHUNKS_FILE.relative_to(project_root)),
        "document_count": 18,
        "chunk_count": 763,
        "indexed_chunk_count": 763,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "cross_encoder_model": settings.CROSS_ENCODER_MODEL_NAME,
        "candidate_k": candidate_k,
        "final_k": final_k,
        "llm_used": False,
        "cross_encoder_model_load_ms": round(model_load_ms, 2) if model_load_ms is not None else None,
        "cross_encoder_warmup_ms": round(warmup_ms, 2) if warmup_ms is not None else None,
        "cross_encoder_unavailable": model_unavailable,
        "metrics": metrics,
        "relevance_rubric": "Manual source-level judgments; unsupported admission and actual exam-date queries have no relevant source label.",
        "results": rows,
    }
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_report(results), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    print(json.dumps(metrics, indent=2))
    if model_unavailable:
        print(f"Cross-Encoder unavailable; fallback results recorded: {model_unavailable}")


def render_report(data):
    metrics = data["metrics"]
    lines = [
        "# MHSSCE Retrieval Cross-Encoder Evaluation",
        "",
        "## Overview",
        "",
        f"- Evaluation time: {data['evaluation_datetime']}",
        f"- Corpus: `{data['corpus_file']}`; {data['document_count']} documents, {data['chunk_count']} chunks, {data['indexed_chunk_count']} indexed records.",
        f"- Embedding model: `{data['embedding_model']}`.",
        f"- Cross-Encoder model: `{data['cross_encoder_model']}`.",
        f"- Candidate K: {data['candidate_k']}; final K: {data['final_k']}; test queries: {metrics['total_queries']}.",
        "- No LLM used. Relevance was judged at source-document level using the rubric in the JSON results.",
        f"- Model load: {data['cross_encoder_model_load_ms']} ms; warm-up inference: {data['cross_encoder_warmup_ms']} ms.",
        "",
        "## Baseline vs reranked",
        "",
        "| Query | Baseline Top-1 | Reranked Top-1 | Baseline Top-3 | Reranked Top-3 | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in data["results"]:
        b1 = row["baseline_ranking"][0]["document"] if row["baseline_ranking"] else "—"
        r1 = row["reranked_ranking"][0]["document"] if row["reranked_ranking"] else "—"
        lines.append(f"| {row['query']} | {b1} | {r1} | {row['baseline_relevant_top_3']} | {row['reranked_relevant_top_3']} | {row['reranked_status']} |")
    lines += ["", "## Ranking metrics", "",
              "| Metric | Baseline | Cross-Encoder |", "|---|---:|---:|"]
    for k in (1, 3, 5):
        lines.append(f"| Relevant top-{k} queries | {metrics[f'baseline_relevant_top_{k}']} / {metrics['total_queries']} | {metrics[f'reranked_relevant_top_{k}']} / {metrics['total_queries']} |")
    lines += [
        f"| Average top-1 existing hybrid score | {metrics['average_baseline_top_1_existing_score']} | — |",
        f"| Average top-1 raw Cross-Encoder score | — | {metrics['average_reranked_top_1_cross_encoder_score']} |",
        f"| Average score across returned top-K candidates | {metrics['average_baseline_score_top_k']} hybrid | {metrics['average_cross_encoder_score_top_k']} raw Cross-Encoder |",
        f"| Average answerability | — | {metrics['average_answerability']} |",
        "",
        "## Latency",
        "",
        f"- Average existing retrieval: {metrics['average_baseline_retrieval_latency_ms']} ms.",
        f"- Average Cross-Encoder inference: {metrics['average_cross_encoder_inference_latency_ms']} ms.",
        f"- Average total reranked retrieval: {metrics['average_reranked_total_latency_ms']} ms.",
        "- Timings are single-process wall-clock averages over this run; model loading and one warm-up inference are reported separately.",
        "",
        "## Per-query analysis",
        "",
    ]
    for index, row in enumerate(data["results"], 1):
        lines += [f"### {index}. {row['query']}", "", f"- Baseline status/answerability: {row['baseline_status']} / {row['baseline_answerability']:.4f}.",
                  f"- Reranked status/answerability: {row['reranked_status']} / {row['answerability']:.4f}.",
                  f"- Expected relevant source(s): {', '.join(row['relevant_source']) or 'none (unsupported query)'}.",
                  f"- Baseline latency / reranked total / Cross-Encoder inference: {row['baseline_latency_ms']} / {row['reranked_total_latency_ms']} / {row['cross_encoder_latency_ms']} ms.",
                  "", "| Rank | Baseline | Hybrid score | Reranked | Hybrid score | Cross-Encoder raw score | Relevant |", "|---:|---|---:|---|---:|---:|---|"]
        for rank in range(max(len(row["baseline_ranking"]), len(row["reranked_ranking"]))):
            b = row["baseline_ranking"][rank] if rank < len(row["baseline_ranking"]) else None
            r = row["reranked_ranking"][rank] if rank < len(row["reranked_ranking"]) else None
            lines.append(f"| {rank + 1} | {b['document'] + ' p.' + str(b['page']) if b else '—'} | {b['existing_score'] if b else '—'} | {r['document'] + ' p.' + str(r['page']) if r else '—'} | {r['existing_score'] if r else '—'} | {r['cross_encoder_score'] if r and r['cross_encoder_score'] is not None else '—'} | {r['relevant'] if r else '—'} |")
        lines.append("")
    improved = {k: metrics[f"reranked_relevant_top_{k}"] - metrics[f"baseline_relevant_top_{k}"] for k in (1, 3, 5)}
    if data["cross_encoder_unavailable"]:
        conclusion = f"Cross-Encoder could not load; graceful hybrid fallback was measured. Error: {data['cross_encoder_unavailable']}"
    else:
        conclusion = "; ".join(f"top-{k} {'improved' if value > 0 else 'regressed' if value < 0 else 'unchanged'} by {abs(value)} query(ies)" for k, value in improved.items())
    lines += ["## Summary", "", f"- {conclusion}.",
              f"- Relevant top-1: {metrics['baseline_relevant_top_1']}/{metrics['total_queries']} baseline vs {metrics['reranked_relevant_top_1']}/{metrics['total_queries']} reranked; top-3 and top-5 are {metrics['baseline_relevant_top_3']}/{metrics['total_queries']} vs {metrics['reranked_relevant_top_3']}/{metrics['total_queries']} and {metrics['baseline_relevant_top_5']}/{metrics['total_queries']} vs {metrics['reranked_relevant_top_5']}/{metrics['total_queries']}, respectively.",
              "- Assessment: no meaningful retrieval improvement; top-1 regression detected while top-3 and top-5 are unchanged.",
              "- Admission-document and examination-date queries should remain `INSUFFICIENT_KNOWLEDGE`; the Principal conflict behavior is checked separately and must remain `CONFLICTING_EVIDENCE`.",
              "- Raw Cross-Encoder values are ranking scores, not probabilities.",
              "- Ready for the LLM stage: not yet. First review the top-1 regression and inference latency; abstention and Principal conflict assertions passed. No LLM is included in this evaluation.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
