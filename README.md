# MHSSCE Knowledge Assistant

An offline-first retrieval-augmented generation (RAG) backend for searching official M. H. Saboo Siddik College of Engineering documents. The backend indexes document chunks in a local ChromaDB database and exposes retrieval and answer APIs.

## Project layout

- `backend/app/` contains configuration, embeddings, vector storage, retrieval, answer generation, and API code.
- `backend/scripts/` contains the database builder, CLI, and retrieval evaluation tools.
- `backend/requirements.txt` lists Python dependencies; `backend/README.md` documents the backend architecture and evaluation behavior.
- `MHSSCE_Documents/` contains documents organized by subject.
- `MHSSCE_RAG_FINAL/` contains the approved source PDFs used for the current corpus.
- `MHSSCE_CLEAN_TEXT/` contains extracted text and extraction metadata.
- `MHSSCE_CHUNKS/` contains the chunk data used for indexing and its chunking report.
- `MHSSCE_RAG_Candidates/` contains documents awaiting review; candidate files are not automatically part of the indexed corpus.
- `backend/chroma_db/` is the generated local vector database.
- Root-level `retrieval_*` files are saved evaluation reports and results; they are retained as project artifacts.

## Setup (PowerShell)

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

Copy `backend/.env.example` to `backend/.env` and set `GEMINI_API_KEY` if generated answers are needed. Retrieval and evaluation can run without the key.

## Common commands

Build or refresh the local vector database:

```powershell
python backend\scripts\build_vector_database.py
```

Start the API:

```powershell
python -m uvicorn app.main:app --app-dir backend --reload
```

Run the retrieval evaluation suite:

```powershell
python backend\scripts\test_retrieval.py
```

Run the reranker comparison:

```powershell
python backend\scripts\evaluate_reranker.py
```

See [backend/README.md](backend/README.md) for the API details, data flow, and evaluation options. The vector database and Uvicorn logs are local runtime artifacts; evaluation reports and results are kept in the project root.
