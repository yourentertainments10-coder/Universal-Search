"""Local file scanner: discovers supported files and reads basic metadata."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic"}
PDF_EXT = {".pdf"}
DOC_EXT = {".txt", ".md", ".docx"}

SUPPORTED = IMAGE_EXT | PDF_EXT | DOC_EXT


def file_type(ext: str) -> str:
    ext = ext.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in PDF_EXT:
        return "pdf"
    return "document"


@dataclass
class FileInfo:
    path: str
    name: str
    ext: str
    type: str
    size: int
    modified: str  # ISO 8601

    def to_dict(self) -> dict:
        return asdict(self)


def scan_folder(root: str | Path) -> list[FileInfo]:
    """Recursively scan `root` and return supported files. Never modifies files."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    results: list[FileInfo] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip hidden folders
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED:
                continue
            full = Path(dirpath) / fname
            try:
                st = full.stat()
            except OSError:
                continue  # unreadable or deleted mid-scan
            results.append(
                FileInfo(
                    path=str(full),
                    name=fname,
                    ext=ext,
                    type=file_type(ext),
                    size=st.st_size,
                    modified=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                )
            )
    results.sort(key=lambda f: f.modified, reverse=True)
    return results
