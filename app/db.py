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
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        migrate(conn)
        yield conn
    finally:
        conn.close()


def list_files(conn: sqlite3.Connection, type: str | None = None) -> list[dict]:
    sql = "SELECT path, name, ext, type, size, modified FROM files"
    args: tuple = ()
    if type:
        sql += " WHERE type = ?"
        args = (type,)
    sql += " ORDER BY modified DESC, name"
    return [dict(r) for r in conn.execute(sql, args)]


def find_file(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT path, name, ext, type FROM files WHERE path_key = ?", (path_key(path),)
    ).fetchone()


def last_root(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT path FROM roots ORDER BY last_scan DESC LIMIT 1").fetchone()
    return row["path"] if row else None
