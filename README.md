# AI Universal Personal Search

Local-first, privacy-focused search across photos, screenshots, PDFs and documents.
Nothing is uploaded; the server binds to `127.0.0.1` only.

## Roadmap
1. Local web prototype (Python 3.10+, FastAPI, HTML/CSS/JS) - **Milestone 1 done**
   - **Milestone 2 done:** SQLite index (`data/index.db`), incremental rescans, deleted files drop out
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

## Index behaviour
- Files are stored as metadata only (path, size, modified time). Originals are never copied or moved.
- A rescan adds new files, updates changed ones (size or modified time differs) and removes deleted ones. Repeated scans never create duplicates.
- If a folder can't be read (permissions, unplugged drive), its files are left in the index instead of being removed.
- Delete `data/index.db` to reset the index. Your original files are not affected.

## Tests
```
python -m unittest discover -s tests -v
```
