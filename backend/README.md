# MHSSCE RAG-Based College Knowledge Assistant — Backend

An offline-first, grounded Retrieval-Augmented Generation (RAG) backend engineered for **M. H. Saboo Siddik College of Engineering (MHSSCE), Byculla, Mumbai**.

This system indexes official college documents (syllabi, fee structures, scholarship circulars, grievance committees, AICTE EOA reports, NAAC SSR, NIRF reports) and provides high-precision semantic retrieval with document and page-level citations.

---

## 1. Project Architecture

The architecture follows a modular, decoupled pipeline designed for reproducibility, high retrieval accuracy, and zero hallucination:

```
[Raw PDFs] (MHSSCE_Documents / MHSSCE_RAG_FINAL)
     │
     ▼
[Text Extraction & Page-Aware OCR] (MHSSCE_CLEAN_TEXT)
     │
     ▼
[Semantic Chunking] (MHSSCE_CHUNKS/chunks.json)
     │  (763 validated chunks across 18 documents, page-aware)
     │
     ▼
[Embedding Generation Engine] (sentence-transformers/all-MiniLM-L6-v2)
     │  (384-dimensional dense normalized embeddings)
     │
     ▼
[Local Vector Database] (ChromaDB with Cosine Metric HNSW)
     │  (backend/chroma_db/)
     │
     ▼
[Hybrid Retrieval] (app/retrieval.py)
     │  (validated hybrid ranking; experimental Cross-Encoder is off by default)
     │
     ▼
[Answerability / Conflict Checks] (app/retrieval.py)
     │
     ├── insufficient evidence -> controlled abstention
     └── supported or conflicting evidence -> Gemini grounded generation (app/llm.py)
             │
             ▼
[API & Testing Interface] (FastAPI app/main.py)
```

---

## 2. Data Flow

1. **Document Corpus**: 18 approved official documents spanning 448 pages (16 extracted natively, 2 extracted via Tesseract OCR for scanned circulars: `fee structure 2026-27.pdf` and `Pragati_Saksham_scholarship scheme 2026-27.pdf`).
2. **Chunking**: Chunks are stored in `MHSSCE_CHUNKS/chunks.json` with strict schema:
   - `chunk_id`: Unique identifier (e.g. `MHSSCE_00001` to `MHSSCE_00763`)
   - `document`: Filename of the source PDF
   - `document_type`: Categorization (e.g. `fee_structure`, `scholarship`, `syllabus`, `grievance`, `anti_ragging`, `admission_form`, etc.)
   - `page`: 1-indexed PDF page number
   - `chunk_number_on_page`: Index of chunk within that page
   - `academic_year`: Preserved academic year (e.g. `2026-27`, `2025-26`, `2023-2024`)
   - `text`: Cleaned text content
3. **Embedding Vectorization**: Each chunk's text is passed through `all-MiniLM-L6-v2` in batches, producing unit-normalized 384-dimensional vectors.
4. **Vector Storage**: Vectors, raw text, and metadata are indexed in persistent ChromaDB collections with an HNSW index using cosine space.
5. **Retrieval**: When a query is received, it is embedded using the exact same model, compared against the index, and the top-$k$ nearest neighbors are returned alongside cosine similarity scores and page references.

---

## 3. How Embeddings Work

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Mechanism**: The model maps variable-length texts into a dense 384-dimensional semantic vector space. Semantically related statements (e.g., "tuition fees" and "fee structure") map to vectors that are in close proximity.
- **Normalization**: Vectors are $L_2$-normalized ($||\mathbf{v}||_2 = 1$). For normalized vectors, the cosine similarity is the inner product:
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$$
  ChromaDB calculates cosine distance as:
  $$\text{Distance} = 1 - \text{Cosine Similarity}$$
  We convert distance back to similarity via `similarity = 1.0 - distance` so scores range from 0.0 (unrelated) to 1.0 (exact match).

---

## 4. Why ChromaDB is Used

1. **Embedded & Local**: Runs in-process without requiring an external Docker container or dedicated cloud database service.
2. **Persistence**: Saves vectors, documents, and rich metadata directly to the local directory `backend/chroma_db/`.
3. **Native Metadata Filtering**: Allows querying by metadata fields (e.g. filtering specifically for `document_type="fee_structure"` or `academic_year="2026-27"`).
4. **HNSW Indexing**: Uses Hierarchical Navigable Small World graphs for sub-millisecond approximate nearest neighbor search.
5. **Safe Idempotent Operations**: Built-in `.upsert()` prevents duplicate records when ingestion scripts are re-run.

---

## 5. Folder Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── config.py           # Paths, thresholds, model configuration
│   ├── embeddings.py       # SentenceTransformer wrapper & batching
│   ├── vector_store.py     # ChromaDB client & persistent collection manager
│   ├── retrieval.py        # Top-k vector retrieval & confidence evaluation
│   ├── reranker.py         # Lazy, process-wide Cross-Encoder reranker
│   ├── llm.py              # Grounded Gemini generation and citation validation
│   ├── rag_pipeline.py     # Retrieval, answerability, and grounded answer orchestration
│   └── main.py             # FastAPI REST endpoints
│
├── chroma_db/              # Persistent ChromaDB vector storage (auto-generated)
├── data/                   # Optional local data cache
├── scripts/
│   ├── build_vector_database.py # Ingestion & embedding pipeline
│   ├── test_retrieval.py        # Retrieval evaluation suite
│   └── evaluate_reranker.py     # Baseline vs Cross-Encoder ranking evaluation
│
├── requirements.txt        # Python dependency manifest
└── README.md               # Backend documentation
```

---

## 6. How to Build the Vector Database

Run the ingestion script from the repository root or backend directory:

```bash
python backend/scripts/build_vector_database.py
```

What the script does:
1. Validates all 763 chunks in `MHSSCE_CHUNKS/chunks.json`.
2. Loads `sentence-transformers/all-MiniLM-L6-v2`.
3. Computes embeddings in batches of 64.
4. Upserts vectors and metadata into `backend/chroma_db/`.
5. Prints elapsed time and total indexed chunk count.

The script is safe to re-run at any time without duplicating entries.

---

## 7. How to Test Retrieval & Ask Questions

### Option A: Interactive CLI Terminal (Ask Questions Continuously)

Run the interactive CLI tool:

```bash
python backend/scripts/ask.py
```

You can also ask single questions directly:

```bash
python backend/scripts/ask.py "What is the fee for OBC students in 2026-27?"
python backend/scripts/ask.py "Who can apply for the Pragati scholarship?"
```

### Option B: Retrieval Evaluation Suite

Execute the benchmark evaluation test suite:

```bash
python backend/scripts/test_retrieval.py
```

This validates metadata and temporal retrieval, Principal conflicts, HOD evidence, supported answers, and abstention cases.

### Option C: Baseline vs Cross-Encoder Evaluation

Run the 16-query non-Principal ranking evaluation and write the comparison report and JSON results to the project root:

```bash
python backend/scripts/evaluate_reranker.py
```

The evaluator records baseline and reranked top-five results, relevance-at-1/3/5, answerability, and approximate retrieval / Cross-Encoder latency. It excludes Principal queries from the ranking metric suite; the retrieval suite continues to assert that all six Principal queries retain `CONFLICTING_EVIDENCE`.


---

## 8. Gemini Setup and Chat API

Install the backend requirements, then paste your Gemini key into `backend/.env` on the `GEMINI_API_KEY=` line. The settings loader reads that file regardless of the current working directory. The key is read lazily, so retrieval and the API can start without it; supported questions that require generation return a clear `503` configuration error until it is set. `backend/.env` is ignored by Git; use `backend/.env.example` as the safe template.

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn app.main:app --app-dir backend --reload
```

`GEMINI_MODEL` is set to `gemini-3.8-flash` in the template and can be changed there. To retain the experimental Cross-Encoder path for a benchmark, set `RAG_CROSS_ENCODER_ENABLED=true` in the process environment; production retrieval uses hybrid ranking by default.

Ask a question with `POST /api/ask`:

```json
{"question": "Who is the HOD of Information Technology?"}
```

The response includes `answer`, retrieval `status`, and `sources` containing document, page, academic year, and chunk ID. Internal scores are omitted unless `include_scores=true` is passed as a query parameter. Unsupported questions abstain without calling Gemini; conflicting records are sent with their evidence and must be described without selecting a winner. Gemini answers use citations generated from retrieval source IDs, so document names and page numbers come from metadata.

Run the offline grounded-answer checks without a Gemini key:

```bash
python backend/scripts/test_llm.py
```

## 9. Current Limitations & Next Steps

1. **OCR Artifacts**: Scanned PDFs (`fee structure 2026-27.pdf` and `Pragati_Saksham_scholarship scheme 2026-27.pdf`) contain occasional character errors from OCR (e.g., `Refino.` or formatting noise in fee tables). While semantic retrieval finds these pages reliably, tabular fee parsing benefits from metadata-assisted table extraction.
2. **Keyword vs. Semantic Specificity**: Very specific course code searches (e.g., "CSC701") may benefit from hybrid BM25 + dense retrieval.
3. **Gemini Grounding**: Citation markers are checked against retrieved source metadata, but factual grounding still depends on the model following the evidence instructions; the pipeline safely abstains when citations are missing or invalid.
