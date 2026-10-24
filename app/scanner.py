"""Local file scanner: discovers supported files and reads basic metadata."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
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


def path_key(p: str | Path) -> str:
    """Canonical form used to compare paths (case-insensitive on Windows)."""
    return os.path.normcase(os.path.abspath(str(p)))


@dataclass
class FileInfo:
    path: str
    name: str
    ext: str
    type: str
    size: int
    mtime_ns: int
    modified: str  # ISO 8601

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    root: str
    files: list[FileInfo] = field(default_factory=list)
    # Folders the scan could not read. Files under them must NOT be treated as deleted.
    failed_dirs: list[str] = field(default_factory=list)


def scan_folder(root: str | Path) -> ScanResult:
    """Recursively scan `root` and return supported files. Never modifies files."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    result = ScanResult(root=str(root))

    def on_error(err: OSError) -> None:
        if err.filename:
            result.failed_dirs.append(str(err.filename))

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error):
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
            result.files.append(
                FileInfo(
                    path=str(full),
                    name=fname,
                    ext=ext,
                    type=file_type(ext),
                    size=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                    modified=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                )
            )
    result.files.sort(key=lambda f: f.modified, reverse=True)
    return result
