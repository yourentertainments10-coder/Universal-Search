# AI Universal Personal Search

Local-first, privacy-focused search across photos, screenshots, PDFs and documents.
Nothing is uploaded; the server binds to `127.0.0.1` only.

## Roadmap
1. Local web prototype (Python, FastAPI, HTML/CSS/JS) - **Milestone 1 done**
2. Search engine: OCR, text extraction, SQLite FTS5, incremental indexing
3. Semantic image search (local embeddings)
4. Native Android app (Kotlin, Jetpack Compose)

## Run (Windows)
```
run.bat
```
Then open http://127.0.0.1:8000, enter a folder path and click Scan.

Manual alternative:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
