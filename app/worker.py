"""Background processing of files whose status is 'pending'.

Statuses: pending -> done | failed | needs_ocr | skipped

Temporary problems (a file locked by another program, a busy disk) are retried
a few times with growing delays; permanent ones (corrupt file, unsupported
format) fail at once. A file is only marked done if it did not change while it
was being processed.
"""
from __future__ import annotations

import errno
import logging
import os
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timedelta

from . import db, extractors, ocr

log = logging.getLogger("universal_search")

_UNSET = object()
BATCH = 25
# cheap text files first, then PDFs, then images (OCR is the slow part)
ORDER = "CASE type WHEN 'document' THEN 0 WHEN 'pdf' THEN 1 ELSE 2 END, id"
RETRY_DELAYS = (10, 60, 300)  # seconds to wait before the 2nd, 3rd and 4th attempt
MAX_WAIT_SLICE = 30           # an idle worker re-checks for due retries at least this often (seconds)
TRANSIENT_ERRNOS = {errno.EBUSY, errno.EAGAIN, errno.EMFILE, errno.ENFILE, errno.ETIMEDOUT}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def is_transient(e: BaseException) -> bool:
    """True for problems that may go away on their own. Corrupt files never do."""
    if isinstance(e, (PermissionError, TimeoutError, BlockingIOError)):
        return True
    return isinstance(e, OSError) and e.errno in TRANSIENT_ERRNOS


def _store(conn: sqlite3.Connection, row, status: str, ex, error: str | None, note: str | None) -> None:
    with conn:
        cur = conn.execute(
            "UPDATE files SET status = ?, retry_after = NULL WHERE id = ? AND mtime_ns = ?",
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
                _now(),
            ),
        )


def _schedule_retry(conn: sqlite3.Connection, row, attempts: int, error: str) -> str:
    due = (datetime.now() + timedelta(seconds=RETRY_DELAYS[attempts - 1])).isoformat(timespec="seconds")
    with conn:
        conn.execute(
            "UPDATE files SET attempts = ?, retry_after = ? WHERE id = ? AND mtime_ns = ?",
            (attempts, due, row["id"], row["mtime_ns"]),
        )
    log.info("Will retry %s (attempt %d failed: %s)", row["path"], attempts, error)
    return "retry"


def process_one(conn: sqlite3.Connection, row, engine) -> str:
    status, ex, error, note = "done", None, None, None
    try:
        if not os.path.isfile(row["path"]):
            raise FileNotFoundError("file no longer exists (rescan its folder to update the index)")
        ex = extractors.extract(row["path"], row["type"], row["ext"], engine)
        if ex.needs_ocr:
            status = "needs_ocr"
    except extractors.Skipped as e:
        status, note = "skipped", str(e)
    except Exception as e:  # one bad file must never stop the whole run
        error = f"{type(e).__name__}: {e}"[:500]
        if is_transient(e):
            attempts = row["attempts"] + 1
            if attempts <= len(RETRY_DELAYS):
                return _schedule_retry(conn, row, attempts, error)
            error = f"{error} (gave up after {attempts} attempts)"[:500]
        status = "failed"
        log.warning("Failed to process %s: %s", row["path"], error)
    _store(conn, row, status, ex, error, note)
    return status


def process_pending(
    stop: threading.Event | None = None,
    on_file=None,
    engine=_UNSET,
    wait_for_retries: bool = False,
) -> Counter:
    """Process every pending file that is due. Safe to call again; it only touches pending files.

    With wait_for_retries=True it also sleeps until files scheduled for a retry are due
    (pausing still interrupts the wait).
    """
    if engine is _UNSET:
        engine = ocr.get_engine()
    counts: Counter = Counter()
    with db.connect() as conn:
        while not (stop and stop.is_set()):
            rows = conn.execute(
                "SELECT id, path, type, ext, mtime_ns, attempts FROM files "
                "WHERE status = 'pending' AND (retry_after IS NULL OR retry_after <= ?) "
                f"ORDER BY {ORDER} LIMIT ?",
                (_now(), BATCH),
            ).fetchall()
            if rows:
                for row in rows:
                    if stop and stop.is_set():
                        break
                    if on_file:
                        on_file(os.path.basename(row["path"]))
                    counts[process_one(conn, row, engine)] += 1
                continue
            if not wait_for_retries:
                break
            nxt = conn.execute(
                "SELECT MIN(retry_after) FROM files WHERE status = 'pending' AND retry_after IS NOT NULL"
            ).fetchone()[0]
            if nxt is None:
                break
            delay = max((datetime.fromisoformat(nxt) - datetime.now()).total_seconds(), 0.2)
            delay = min(delay, MAX_WAIT_SLICE)
            if stop:
                if stop.wait(delay):
                    break
            else:
                time.sleep(delay)
    return counts


def retry_failed(conn: sqlite3.Connection) -> int:
    """Put failed and needs_ocr files back in the queue with a fresh attempt count."""
    with conn:
        cur = conn.execute(
            "UPDATE files SET status = 'pending', attempts = 0, retry_after = NULL "
            "WHERE status IN ('failed', 'needs_ocr')"
        )
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

    def stop_and_wait(self, timeout: float = 60.0) -> bool:
        """Pause and wait for the thread to finish its current file. True if it stopped."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        return not self.running

    def _run(self) -> None:
        try:
            process_pending(self._stop, self._set_current, wait_for_retries=True)
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
            retrying = conn.execute(
                "SELECT COUNT(*) FROM files WHERE status = 'pending' AND retry_after IS NOT NULL"
            ).fetchone()[0]
        return {
            "running": self.running,
            "pausing": self.running and self._stop.is_set(),
            "current": self._current,
            "counts": counts,
            "retrying": retrying,
            "total": sum(counts.values()),
            "ocr": ocr.status(),
            "last_error": self._last_error,
        }
