"""Small cached JPEG previews so result lists do not load full-size photos."""
from __future__ import annotations

import os
import threading
from pathlib import Path

from PIL import Image

from . import db
from .extractors import _prepare_image

THUMB_SIZE = 360


def thumb_path(row) -> Path:
    # The modified time is part of the name, so an edited image gets a fresh preview.
    return db.db_path().parent / "thumbs" / f"{row['id']}_{row['mtime_ns']}.jpg"


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
