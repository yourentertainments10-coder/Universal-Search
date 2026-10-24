"""AI Universal Personal Search - local web prototype.

Run bound to 127.0.0.1 only; nothing leaves the machine.
Milestone 2: files are stored in SQLite and rescans are incremental.
Milestone 3: background text extraction (text, DOCX, PDF) and OCR (images, scanned PDFs).
Milestone 4: ranked keyword/phrase search with filters, snippets, thumbnails.
Milestone 5: retries, index reset, folder management, missing-file handling.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, search, thumbs
from .indexer import forget_folder, sync_folder
from .scanner import scan_folder
from .worker import IndexWorker, retry_failed

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Universal Personal Search")
worker = IndexWorker()


class ScanRequest(BaseModel):
    folder: str


def _indexed_file(path: str, must_exist: bool = True):
    """Only allow access to files in the index (blocks arbitrary path access)."""
    with db.connect() as conn:
        row = db.find_file(conn, path)
    if row is None:
        raise HTTPException(status_code=404, detail="File is not in the index")
    if must_exist and not os.path.isfile(row["path"]):
        raise HTTPException(
            status_code=404,
            detail="This file no longer exists on disk (or its drive is unplugged). Rescan its folder to update the index.",
        )
    return row


def _scan_and_sync(folder: str) -> dict:
    try:
        result = scan_folder(folder)
    except (NotADirectoryError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    with db.connect() as conn:
        summary = sync_folder(conn, result)
        thumbs.cleanup(conn)  # drop previews of files that were removed or changed
    return summary


@app.post("/api/scan")
def api_scan(req: ScanRequest):
    summary = _scan_and_sync(req.folder)
    worker.start()  # extract text from new and changed files in the background
    return summary


@app.post("/api/rescan")
def api_rescan():
    """Rescan every known folder. Folders that are unavailable (unplugged drive) are left untouched."""
    with db.connect() as conn:
        roots = db.list_roots(conn)
    out = []
    for r in roots:
        if not os.path.isdir(r["path"]):
            out.append({"root": r["path"], "unavailable": True})
            continue
        try:
            out.append(_scan_and_sync(r["path"]))
        except HTTPException as e:
            out.append({"root": r["path"], "error": e.detail})
    worker.start()
    return {"roots": out}


@app.get("/api/roots")
def api_roots():
    with db.connect() as conn:
        roots = db.list_roots(conn)
    for r in roots:
        r["available"] = os.path.isdir(r["path"])
    return {"roots": roots}


@app.post("/api/roots/remove")
def api_roots_remove(req: ScanRequest):
    """Stop indexing a folder and forget everything indexed under it. Files on disk are untouched."""
    with db.connect() as conn:
        removed = forget_folder(conn, req.folder)
        thumbs.cleanup(conn)
    return {"removed": removed}


@app.get("/api/search")
def api_search(
    q: str = Query("", max_length=300),
    type: str | None = Query(None, pattern="^(image|pdf|document)$"),
    date_from: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    sort: str = Query("relevance", pattern="^(relevance|newest|oldest)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, pattern="^(pending|done|failed|needs_ocr|skipped)$"),
):
    try:
        with db.connect() as conn:
            return search.run_search(conn, q, type, date_from, date_to, sort, limit, offset, status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/files")
def api_files(
    type: str | None = Query(None, pattern="^(image|pdf|document)$"),
    limit: int = Query(300, ge=1, le=2000),
):
    with db.connect() as conn:
        files = db.list_files(conn, type, limit)
        total = db.count_files(conn, type)
        root = db.last_root(conn)
    return {"root": root, "count": total, "files": files}


@app.get("/api/index/status")
def api_index_status():
    return worker.status()


@app.post("/api/index/start")
def api_index_start():
    worker.start()
    return worker.status()


@app.post("/api/index/pause")
def api_index_pause():
    worker.pause()
    return worker.status()


@app.post("/api/index/retry")
def api_index_retry():
    """Re-queue failed files and files that were waiting for an OCR engine."""
    with db.connect() as conn:
        n = retry_failed(conn)
    worker.start()
    return {"requeued": n}


@app.post("/api/index/reset")
def api_index_reset():
    """Delete the whole search index and all previews. Original files are never touched."""
    if not worker.stop_and_wait(60):
        raise HTTPException(status_code=409, detail="Indexing is still stopping. Try again in a moment.")
    with db.connect() as conn:
        db.reset_index(conn)
    thumbs.clear()
    return {"reset": True}


@app.get("/api/content")
def api_content(path: str, max_chars: int = Query(20000, ge=100, le=200000)):
    row = _indexed_file(path, must_exist=False)  # stored text stays readable even if the file is gone
    with db.connect() as conn:
        content = db.get_content(conn, row["id"], max_chars)
    return {"name": row["name"], "status": row["status"], "content": content}


@app.get("/api/preview")
def api_preview(path: str):
    row = _indexed_file(path)
    if row["type"] != "image":
        raise HTTPException(status_code=400, detail="Preview only for images")
    return FileResponse(row["path"])


@app.get("/api/thumb")
def api_thumb(path: str):
    row = _indexed_file(path)
    if row["type"] != "image":
        raise HTTPException(status_code=404, detail="Only images have previews")
    thumb = thumbs.get_thumbnail(row)
    if thumb is None:
        raise HTTPException(status_code=415, detail="No preview available")
    return FileResponse(thumb, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.post("/api/reveal")
def api_reveal(path: str):
    """Show the original file in Windows Explorer."""
    row = _indexed_file(path)
    if sys.platform != "win32":
        raise HTTPException(status_code=501, detail="Show in folder is implemented for Windows only")
    subprocess.Popen(["explorer", f"/select,{row['path']}"])  # noqa: S603 - path comes from the index
    return {"revealed": row["name"]}


@app.post("/api/open")
def api_open(path: str):
    """Open the original file in the system's default viewer (Windows)."""
    row = _indexed_file(path)
    if sys.platform != "win32":
        raise HTTPException(status_code=501, detail="Open is implemented for Windows only")
    os.startfile(row["path"])  # noqa: S606 - path is validated against the index
    return {"opened": row["name"]}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
