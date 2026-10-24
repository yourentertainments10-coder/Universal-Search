"""SQLite storage: connection handling, schema migrations and simple queries.

The index lives in data/index.db (override with UNIVERSAL_SEARCH_DB).
Only metadata is stored; original files are never copied or moved.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .scanner import path_key

# Migration N is applied when PRAGMA user_version < N. Append only; never edit old ones.
MIGRATIONS = [
    """
    CREATE TABLE files (
        id          INTEGER PRIMARY KEY,
        path        TEXT NOT NULL,
        path_key    TEXT NOT NULL UNIQUE,
        name        TEXT NOT NULL,
        ext         TEXT NOT NULL,
        type        TEXT NOT NULL,
        size        INTEGER NOT NULL,
        mtime_ns    INTEGER NOT NULL,
        modified    TEXT NOT NULL,
        status      TEXT NOT NULL DEFAULT 'pending',
        first_seen  TEXT NOT NULL,
        indexed_at  TEXT NOT NULL
    );
    CREATE INDEX idx_files_type ON files(type);
    CREATE TABLE roots (
        path_key  TEXT PRIMARY KEY,
        path      TEXT NOT NULL,
        last_scan TEXT NOT NULL
    );
    """,
    # 2: extracted text (Milestone 3). Rows disappear with their file.
    """
    CREATE TABLE content (
        file_id      INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
        text         TEXT NOT NULL DEFAULT '',
        method       TEXT NOT NULL DEFAULT '',
        engine       TEXT,
        pages        INTEGER,
        note         TEXT,
        error        TEXT,
        extracted_at TEXT NOT NULL
    );
    CREATE INDEX idx_files_status ON files(status);
    """,
]


def db_path() -> Path:
    env = os.environ.get("UNIVERSAL_SEARCH_DB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data" / "index.db"


def migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, sql in enumerate(MIGRATIONS, start=1):
        if version > current:
            conn.executescript(f"BEGIN;{sql}PRAGMA user_version={version};COMMIT;")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        migrate(conn)
        yield conn
    finally:
        conn.close()


def list_files(conn: sqlite3.Connection, type: str | None = None, limit: int | None = None) -> list[dict]:
    sql = "SELECT path, name, ext, type, size, modified, status FROM files"
    args: list = []
    if type:
        sql += " WHERE type = ?"
        args.append(type)
    sql += " ORDER BY modified DESC, name"
    if limit:
        sql += " LIMIT ?"
        args.append(limit)
    return [dict(r) for r in conn.execute(sql, args)]


def count_files(conn: sqlite3.Connection, type: str | None = None) -> int:
    if type:
        return conn.execute("SELECT COUNT(*) FROM files WHERE type = ?", (type,)).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]


def status_counts(conn: sqlite3.Connection) -> dict:
    counts = {"pending": 0, "done": 0, "failed": 0, "needs_ocr": 0, "skipped": 0}
    for r in conn.execute("SELECT status, COUNT(*) AS n FROM files GROUP BY status"):
        counts[r["status"]] = r["n"]
    return counts


def find_file(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, path, name, ext, type, status FROM files WHERE path_key = ?", (path_key(path),)
    ).fetchone()


def get_content(conn: sqlite3.Connection, file_id: int, max_chars: int) -> dict | None:
    r = conn.execute(
        "SELECT text, method, engine, pages, note, error, extracted_at FROM content WHERE file_id = ?",
        (file_id,),
    ).fetchone()
    if r is None:
        return None
    d = dict(r)
    d["length"] = len(d["text"])
    d["text"] = d["text"][:max_chars]
    return d


def last_root(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT path FROM roots ORDER BY last_scan DESC LIMIT 1").fetchone()
    return row["path"] if row else None
