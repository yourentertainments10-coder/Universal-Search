"""AI Universal Personal Search - local web prototype.

Run bound to 127.0.0.1 only; nothing leaves the machine.
Milestone 2: files are stored in SQLite and rescans are incremental.
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

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Universal Personal Search")


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
        return sync_folder(conn, result)


@app.get("/api/files")
def api_files(type: str | None = Query(None, pattern="^(image|pdf|document)$")):
    with db.connect() as conn:
        files = db.list_files(conn, type)
        root = db.last_root(conn)
    return {"root": root, "count": len(files), "files": files}


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
