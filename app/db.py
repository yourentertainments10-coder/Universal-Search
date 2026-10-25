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
    # 3: full-text search (Milestone 4). One FTS row per file (filename + extracted text),
    # kept in sync by triggers. The INSERT...SELECT backfills files indexed before this migration.
    """
    CREATE VIRTUAL TABLE files_fts USING fts5(name, text, tokenize='unicode61 remove_diacritics 2');
    INSERT INTO files_fts (rowid, name, text)
        SELECT f.id, f.name, COALESCE(c.text, '') FROM files f LEFT JOIN content c ON c.file_id = f.id;
    CREATE TRIGGER files_fts_ai AFTER INSERT ON files BEGIN
        INSERT INTO files_fts (rowid, name, text) VALUES (new.id, new.name, '');
    END;
    CREATE TRIGGER files_fts_ad AFTER DELETE ON files BEGIN
        DELETE FROM files_fts WHERE rowid = old.id;
    END;
    CREATE TRIGGER content_fts_ai AFTER INSERT ON content BEGIN
        DELETE FROM files_fts WHERE rowid = new.file_id;
        INSERT INTO files_fts (rowid, name, text) SELECT id, name, new.text FROM files WHERE id = new.file_id;
    END;
    CREATE TRIGGER content_fts_au AFTER UPDATE ON content BEGIN
        DELETE FROM files_fts WHERE rowid = new.file_id;
        INSERT INTO files_fts (rowid, name, text) SELECT id, name, new.text FROM files WHERE id = new.file_id;
    END;
    CREATE TRIGGER content_fts_ad AFTER DELETE ON content BEGIN
        DELETE FROM files_fts WHERE rowid = old.file_id;
        INSERT INTO files_fts (rowid, name, text) SELECT id, name, '' FROM files WHERE id = old.file_id;
    END;
    """,
    # 4: retry bookkeeping for temporary failures (Milestone 5)
    """
    ALTER TABLE files ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE files ADD COLUMN retry_after TEXT;
    """,
    # 5: image embeddings for semantic search (Milestone 6). A row with vector NULL records a failure.
    """
    CREATE TABLE embeddings (
        file_id    INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
        model      TEXT NOT NULL,
        mtime_ns   INTEGER NOT NULL,
        dim        INTEGER NOT NULL DEFAULT 0,
        vector     BLOB,
        error      TEXT,
        created_ns INTEGER NOT NULL
    );
    """,
]

FTS_MIGRATION = 3  # first migration that needs SQLite FTS5


def db_path() -> Path:
    env = os.environ.get("UNIVERSAL_SEARCH_DB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data" / "index.db"


def _require_fts5(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE temp.fts5_probe")
    except sqlite3.OperationalError as e:
        raise RuntimeError(
            "This Python's SQLite was built without FTS5, which search needs. "
            "Install Python from python.org (it includes FTS5)."
        ) from e


def migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < FTS_MIGRATION:
        _require_fts5(conn)
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
        "SELECT id, path, name, ext, type, status, mtime_ns FROM files WHERE path_key = ?", (path_key(path),)
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


def list_roots(conn: sqlite3.Connection) -> list[dict]:
    """Folders that have been scanned, with how many indexed files each holds."""
    out = []
    for r in conn.execute("SELECT path_key, path, last_scan FROM roots ORDER BY last_scan DESC"):
        prefix = r["path_key"].rstrip(os.sep) + os.sep
        n = conn.execute(
            "SELECT COUNT(*) FROM files WHERE substr(path_key, 1, ?) = ?", (len(prefix), prefix)
        ).fetchone()[0]
        out.append({"path": r["path"], "last_scan": r["last_scan"], "files": n})
    return out


def reset_index(conn: sqlite3.Connection) -> None:
    """Delete everything the app has indexed. Original files are never touched."""
    with conn:
        conn.execute("DELETE FROM files")  # cascades to content; triggers empty files_fts
        conn.execute("DELETE FROM roots")
    conn.execute("VACUUM")  # give the disk space back
