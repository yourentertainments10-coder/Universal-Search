"""AI Universal Personal Search - local web prototype (Milestone 1).

Run bound to 127.0.0.1 only; nothing leaves the machine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .scanner import FileInfo, scan_folder

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Universal Personal Search")

# In-memory state for Milestone 1 (replaced by SQLite in Milestone 2).
state: dict = {"root": None, "files": {}}


class ScanRequest(BaseModel):
    folder: str


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _known_file(path: str) -> FileInfo:
    """Only allow access to files found by the last scan (blocks path traversal)."""
    info = state["files"].get(_key(path))
    if info is None:
        raise HTTPException(status_code=404, detail="File not in scanned folder")
    return info


@app.post("/api/scan")
def api_scan(req: ScanRequest):
    try:
        files = scan_folder(req.folder)
    except (NotADirectoryError, PermissionError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    state["root"] = req.folder
    state["files"] = {_key(f.path): f for f in files}
    return {"root": req.folder, "count": len(files)}


@app.get("/api/files")
def api_files(type: str | None = Query(None, pattern="^(image|pdf|document)$")):
    files = [f.to_dict() for f in state["files"].values()]
    if type:
        files = [f for f in files if f["type"] == type]
    files.sort(key=lambda f: f["modified"], reverse=True)
    return {"root": state["root"], "count": len(files), "files": files}


@app.get("/api/preview")
def api_preview(path: str):
    info = _known_file(path)
    if info.type != "image":
        raise HTTPException(status_code=400, detail="Preview only for images")
    return FileResponse(info.path)


@app.post("/api/open")
def api_open(path: str):
    """Open the original file in the system's default viewer (Windows)."""
    info = _known_file(path)
    if sys.platform != "win32":
        raise HTTPException(status_code=501, detail="Open is implemented for Windows only")
    os.startfile(info.path)  # noqa: S606 - path is validated against scan results
    return {"opened": info.name}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
