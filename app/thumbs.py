"""Small cached JPEG previews so result lists do not load full-size photos."""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

from PIL import Image

from . import db
from .extractors import _prepare_image

THUMB_SIZE = 360


def thumbs_dir() -> Path:
    return db.db_path().parent / "thumbs"


def thumb_path(row) -> Path:
    # The modified time is part of the name, so an edited image gets a fresh preview.
    return thumbs_dir() / f"{row['id']}_{row['mtime_ns']}.jpg"


def get_thumbnail(row, size: int = THUMB_SIZE) -> Path | None:
    """Return the cached preview for an image row, creating it if needed. None if unreadable."""
    out = thumb_path(row)
    if out.is_file():
        return out
    try:
        with Image.open(row["path"]) as img:
            img.draft("RGB", (size * 2, size * 2))  # fast, low-resolution JPEG decode
            img = _prepare_image(img)
        img.thumbnail((size, size))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        img.save(tmp, "JPEG", quality=80)
        tmp.replace(out)  # atomic: readers never see a half-written file
        return out
    except Exception:
        return None


def cleanup(conn) -> int:
    """Delete previews of images that were removed from the index or have since changed."""
    folder = thumbs_dir()
    if not folder.is_dir():
        return 0
    valid = {f"{r['id']}_{r['mtime_ns']}.jpg" for r in conn.execute("SELECT id, mtime_ns FROM files WHERE type = 'image'")}
    removed = 0
    for p in folder.iterdir():
        if p.name.endswith(".tmp") or p.name in valid:
            continue  # leave in-progress writes and current previews alone
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def clear() -> None:
    shutil.rmtree(thumbs_dir(), ignore_errors=True)
