# AI Universal Personal Search

Local-first, privacy-focused search across photos, screenshots, PDFs and documents.
Nothing is uploaded; the server binds to `127.0.0.1` only.

## Roadmap
1. Local web prototype (Python 3.10+, FastAPI, HTML/CSS/JS) - **Milestone 1 done**
   - **Milestone 2 done:** SQLite index (`data/index.db`), incremental rescans, deleted files drop out
   - **Milestone 3 done:** background text extraction (TXT, MD, DOCX, PDF) and OCR (images, scanned PDFs)
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

## OCR setup (needed for images and scanned PDFs)
Text files, DOCX and PDFs with a text layer work out of the box. For images and scanned PDFs install one OCR engine:

- **Option A (pip only):** `pip install -r requirements-ocr.txt` inside the virtual environment (RapidOCR).
- **Option B:** install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki) and `pip install pytesseract`.

Restart the server afterwards, then press **Retry failed** to process files that were waiting for OCR.
Optional environment variables: `OCR_ENGINE` (`rapidocr` or `tesseract`), `TESSERACT_CMD` (path to tesseract.exe), `OCR_LANG` (default `eng`).

## Index behaviour
- Files are stored as metadata only (path, size, modified time). Originals are never copied or moved.
- A rescan adds new files, updates changed ones (size or modified time differs) and removes deleted ones. Repeated scans never create duplicates.
- If a folder can't be read (permissions, unplugged drive), its files are left in the index instead of being removed.
- After a scan, a background worker extracts text from every new or changed file. Cheap files (text, DOCX) go first, images last. Use **Pause** to stop it; it resumes where it left off.
- File statuses: `waiting`, `text ready`, `failed`, `needs OCR` (no OCR engine installed), `skipped` (for example HEIC). The **Text** button on a card shows what was extracted.
- If a file changes while it is being processed it is processed again, so stale text is never kept.
- Python virtual environments, `site-packages`, `node_modules` and similar folders are skipped.
- Delete `data/index.db` to reset the index. Your original files are not affected.

## Tests
```
python -m unittest discover -s tests -v
```
