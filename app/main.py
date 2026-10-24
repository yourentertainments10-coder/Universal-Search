"""AI Universal Personal Search - local web prototype.

Run bound to 127.0.0.1 only; nothing leaves the machine.
Milestone 2: files are stored in SQLite and rescans are incremental.
Milestone 3: background text extraction (text, DOCX, PDF) and OCR (images, scanned PDFs).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db
from .indexer import sync_folder
from .scanner import scan_folder
from .worker import IndexWorker, retry_failed

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Universal Personal Search")
worker = IndexWorker()


class ScanRequest(BaseModel):
    folder: str


def _indexed_file(path: str):
    """Only allow access to files in the index (blocks arbitrary path access)."""
    with db.connect() as conn:
        row = db.find_file(conn, path)
    if row is None or not os.path.isfile(row["path"]):
        raise HTTPException(status_code=404, detail="File not found in index")
    return row


@app.post("/api/scan")
def api_scan(req: ScanRequest):
    try:
        result = scan_folder(req.folder)
    except (NotADirectoryError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    with db.connect() as conn:
        summary = sync_folder(conn, result)
    worker.start()  # extract text from new and changed files in the background
    return summary


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


@app.get("/api/content")
def api_content(path: str, max_chars: int = Query(20000, ge=100, le=200000)):
    row = _indexed_file(path)
    with db.connect() as conn:
        content = db.get_content(conn, row["id"], max_chars)
    return {"name": row["name"], "status": row["status"], "content": content}


@app.get("/api/preview")
def api_preview(path: str):
    row = _indexed_file(path)
    if row["type"] != "image":
        raise HTTPException(status_code=400, detail="Preview only for images")
    return FileResponse(row["path"])


@app.post("/api/open")
def api_open(path: str):
    """Open the original file in the system's default viewer (Windows)."""
    row = _indexed_file(path)
    if sys.platform != "win32":
        raise HTTPException(status_code=501, detail="Open is implemented for Windows only")
    os.startfile(row["path"])  # noqa: S606 - path is validated against the index
    return {"opened": row["name"]}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
