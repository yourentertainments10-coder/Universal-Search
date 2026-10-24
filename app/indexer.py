"""Incremental indexing: bring the database in line with what a scan found."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime

from .scanner import ScanResult, path_key


def _dir_prefix(p: str) -> str:
    return path_key(p).rstrip(os.sep) + os.sep


def sync_folder(conn: sqlite3.Connection, result: ScanResult) -> dict:
    """Add new files, update changed ones, remove deleted ones. One transaction.

    Change detection: a file is "changed" when its size or modified time differs
    from the stored values. Changed and new files are marked status='pending' so
    later milestones (OCR, text extraction) know what to process.

    Files under folders the scan could not read are left untouched, so a
    permission error or unplugged drive never wipes the index.
    """
    prefix = _dir_prefix(result.root)
    skipped = tuple(_dir_prefix(d) for d in result.failed_dirs)
    now = datetime.now().isoformat(timespec="seconds")
    added = updated = unchanged = 0

    with conn:  # commits on success, rolls back on error
        existing = {
            r["path_key"]: r
            for r in conn.execute(
                "SELECT id, path_key, size, mtime_ns FROM files WHERE substr(path_key, 1, ?) = ?",
                (len(prefix), prefix),
            )
        }
        seen: set[str] = set()

        for f in result.files:
            key = path_key(f.path)
            seen.add(key)
            row = existing.get(key)
            if row is None:
                conn.execute(
                    """INSERT INTO files
                       (path, path_key, name, ext, type, size, mtime_ns, modified,
                        status, first_seen, indexed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                    (f.path, key, f.name, f.ext, f.type, f.size, f.mtime_ns, f.modified, now, now),
                )
                added += 1
            elif row["size"] != f.size or row["mtime_ns"] != f.mtime_ns:
                conn.execute(
                    """UPDATE files SET size=?, mtime_ns=?, modified=?, status='pending',
                       indexed_at=? WHERE id=?""",
                    (f.size, f.mtime_ns, f.modified, now, row["id"]),
                )
                conn.execute("DELETE FROM content WHERE file_id = ?", (row["id"],))
                updated += 1
            else:
                unchanged += 1

        stale = [
            (row["id"],)
            for key, row in existing.items()
            if key not in seen and not key.startswith(skipped)
        ]
        conn.executemany("DELETE FROM files WHERE id = ?", stale)

        conn.execute(
            """INSERT INTO roots (path_key, path, last_scan) VALUES (?, ?, ?)
               ON CONFLICT(path_key) DO UPDATE SET path=excluded.path, last_scan=excluded.last_scan""",
            (path_key(result.root), result.root, now),
        )
        total = conn.execute(
            "SELECT COUNT(*) FROM files WHERE substr(path_key, 1, ?) = ?", (len(prefix), prefix)
        ).fetchone()[0]

    return {
        "root": result.root,
        "count": total,
        "added": added,
        "updated": updated,
        "unchanged": unchanged,
        "removed": len(stale),
        "skipped_folders": len(result.failed_dirs),
    }
