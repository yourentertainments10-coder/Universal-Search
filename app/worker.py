"""Background processing of files whose status is 'pending'.

Statuses: pending -> done | failed | needs_ocr | skipped
A file is marked done only if it did not change while being processed.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections import Counter
from datetime import datetime

from . import db, extractors, ocr

log = logging.getLogger("universal_search")

_UNSET = object()
BATCH = 25
# cheap text files first, then PDFs, then images (OCR is the slow part)
ORDER = "CASE type WHEN 'document' THEN 0 WHEN 'pdf' THEN 1 ELSE 2 END, id"


def _store(conn: sqlite3.Connection, row, status: str, ex, error: str | None, note: str | None) -> None:
    with conn:
        cur = conn.execute(
            "UPDATE files SET status = ? WHERE id = ? AND mtime_ns = ?",
            (status, row["id"], row["mtime_ns"]),
        )
        if not cur.rowcount:
            return  # file changed or was removed meanwhile; it will be processed again
        conn.execute(
            """INSERT INTO content (file_id, text, method, engine, pages, note, error, extracted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(file_id) DO UPDATE SET
                 text=excluded.text, method=excluded.method, engine=excluded.engine,
                 pages=excluded.pages, note=excluded.note, error=excluded.error,
                 extracted_at=excluded.extracted_at""",
            (
                row["id"],
                ex.text if ex else "",
                ex.method if ex else "",
                ex.engine if ex else None,
                ex.pages if ex else None,
                note if note is not None else (ex.note if ex else None),
                error,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def process_one(conn: sqlite3.Connection, row, engine) -> str:
    status, ex, error, note = "done", None, None, None
    try:
        if not os.path.isfile(row["path"]):
            raise FileNotFoundError("file no longer exists")
        ex = extractors.extract(row["path"], row["type"], row["ext"], engine)
        if ex.needs_ocr:
            status = "needs_ocr"
    except extractors.Skipped as e:
        status, note = "skipped", str(e)
    except Exception as e:  # one bad file must never stop the whole run
        status, error = "failed", f"{type(e).__name__}: {e}"[:500]
        log.warning("Failed to process %s: %s", row["path"], error)
    _store(conn, row, status, ex, error, note)
    return status


def process_pending(stop: threading.Event | None = None, on_file=None, engine=_UNSET) -> Counter:
    """Process every pending file. Safe to call again; it only touches pending files."""
    if engine is _UNSET:
        engine = ocr.get_engine()
    counts: Counter = Counter()
    with db.connect() as conn:
        while not (stop and stop.is_set()):
            rows = conn.execute(
                f"SELECT id, path, type, ext, mtime_ns FROM files WHERE status = 'pending' ORDER BY {ORDER} LIMIT ?",
                (BATCH,),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                if stop and stop.is_set():
                    break
                if on_file:
                    on_file(os.path.basename(row["path"]))
                counts[process_one(conn, row, engine)] += 1
    return counts


def retry_failed(conn: sqlite3.Connection) -> int:
    """Put failed and needs_ocr files back in the queue."""
    with conn:
        cur = conn.execute("UPDATE files SET status = 'pending' WHERE status IN ('failed', 'needs_ocr')")
    return cur.rowcount


class IndexWorker:
    """Runs process_pending on one background thread; can be paused."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._current: str | None = None
        self._last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self._stop.clear()
            self._last_error = None
            self._thread = threading.Thread(target=self._run, name="index-worker", daemon=True)
            self._thread.start()
            return True

    def pause(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            process_pending(self._stop, self._set_current)
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            log.exception("Index worker stopped by an error")
        finally:
            self._current = None

    def _set_current(self, name: str) -> None:
        self._current = name

    def status(self) -> dict:
        with db.connect() as conn:
            counts = db.status_counts(conn)
        return {
            "running": self.running,
            "pausing": self.running and self._stop.is_set(),
            "current": self._current,
            "counts": counts,
            "total": sum(counts.values()),
            "ocr": ocr.status(),
            "last_error": self._last_error,
        }
