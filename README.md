# .\.venv\Scripts\Activate.ps1 
# AI Universal Personal Search

Local-first, privacy-focused search across photos, screenshots, PDFs and documents.
Nothing is uploaded; the server binds to `127.0.0.1` only.

## Roadmap
1. Local web prototype (Python 3.10+, FastAPI, HTML/CSS/JS) - **Milestone 1 done**
   - **Milestone 2 done:** SQLite index (`data/index.db`), incremental rescans, deleted files drop out
   - **Milestone 3 done:** background text extraction (TXT, MD, DOCX, PDF) and OCR (images, scanned PDFs)
   - **Milestone 4 done:** ranked keyword and phrase search, type and date filters, snippets, thumbnails, open original / show in folder
   - **Milestone 5 done:** retries for temporary failures, index reset, folder management, missing-file handling, offline and privacy checks
   - **Milestone 6 done:** search by meaning for images (local CLIP model), fused with keyword search
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

## Searching
Type in the search box; results update as you type.
- Several words must all appear: `salary calculation`
- `"quoted words"` must appear together, in that order: `"sin 3x"`
- The word you are still typing matches as a prefix (`trigon` finds *trigonometry*). Add a space after it to match the exact word.
- Filename matches rank above matches inside the text. Accents are ignored (`cafe` finds *café*).
- Use the type buttons, the From / To dates (file modified date) and the sort menu to narrow results.
- **Open** opens the original file, **Show in folder** selects it in Explorer, **Text** shows what was extracted.
- Files appear in search by name as soon as they are scanned, and by content once indexing has processed them.
- Previews are cached in `data/thumbs/`; deleting `data/` resets everything (your original files are never touched).

## Search by meaning (optional)
Finds photos by what they show, even when no text is visible: `college farewell group photo`, `dog on a beach`, `whiteboard diagram`.
It uses a local CLIP model (OpenAI CLIP ViT-B/32, MIT licence) run with ONNX Runtime. Photos and queries never leave this computer.

1. Install the extra packages inside `.venv`: `pip install -r requirements-semantic.txt`
2. Download the model once (about 606 MB; `CLIP_VARIANT=small` gives a 154 MB, faster, slightly less accurate version): press **Download model** in *Folders and indexing*, or run `python -m app.semantic download`. This is the only network use in the whole app, it needs your click, and it resumes if interrupted. The files are pinned to a fixed revision and stored in `data/models/`.
3. Scan a folder. Images are embedded in the background after text indexing finishes (one image takes roughly 0.05 to 0.3 s on a typical PC). Progress and Pause are in the same panel.
4. A *Search mode* menu appears next to the filters: **Text and meaning** (default, results from both are merged and files found both ways rank first), **Text only**, **Meaning only (images)**. Meaning matches show a `looks like it 0.27` badge.

Measure it on your own photos before trusting it:
```
python -m app.semantic check
python -m app.semantic check --folder D:\Photos --query "a dog" --query "people at a party"
```
This prints a sanity check, milliseconds per image, memory use, and the best matches with their scores.

Tuning (environment variables): `SEMANTIC_MIN_SIMILARITY` (default 0.20, raise it if unrelated photos appear), `SEMANTIC_MARGIN` (0.07, how close to the best match a result must be), `SEMANTIC_TOP_K` (60), `CLIP_THREADS`, `CLIP_VARIANT`.
Limits: images only (not PDFs), English queries work best, and the default thresholds are starting values that have not yet been calibrated on real photos.

## Keeping the index healthy
- **Folders:** the *Folders and indexing* section lists every scanned folder with its file count. **Rescan** refreshes one, **Rescan all** refreshes every reachable one, **Remove** forgets a folder (the files on disk are never touched).
- **Unplugged drives:** if a folder is unavailable, rescanning leaves its files in the index. They stay searchable, are marked *missing*, and recover when the drive is back.
- **Deleted files:** rescanning removes files that no longer exist. Until then they show as *missing* and cannot be opened.
- **Temporary failures** (for example a file open in another program) are retried automatically after 10 seconds, 1 minute and 5 minutes before the file is marked failed. Corrupt files fail immediately. **Retry failed** puts everything back in the queue.
- **Find problems:** use the *Any status* menu next to the filters to list failed, waiting or needs-OCR files; **Text** shows the error.
- **Reset search index** deletes the database and previews after a confirmation. It never deletes your files.
- Privacy and offline behaviour are covered by tests: the pipeline is run with network access blocked, the page loads nothing from the internet, and the server only listens on `127.0.0.1`.

## OCR setup (needed for images and scanned PDFs)
Text files, DOCX and PDFs with a text layer work out of the box. For images and scanned PDFs install one OCR engine:

- **Option A (pip only):** `pip install -r requirements-ocr.txt` inside the virtual environment (RapidOCR, installs `rapidocr` + `onnxruntime`).
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
- Search uses SQLite FTS5 (included with Python from python.org). Existing indexes upgrade automatically on first start.
- Delete `data/index.db` to reset the index. Your original files are not affected.

## Tests
```
python -m unittest discover -s tests -v
```
