"""Query parsing and ranked full-text search over files_fts (SQLite FTS5)."""
from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from datetime import date, timedelta

# Control characters wrap matches in snippets; the page turns them into <mark> safely.
MARK_START, MARK_END = "\x02", "\x03"
ELLIPSIS = "\u2026"
PHRASE = re.compile(r'"([^"]*)"')
MAX_TERMS = 20
NAME_WEIGHT, TEXT_WEIGHT = 8.0, 1.0  # a filename match outranks a body match


def tokenize(text: str) -> list[str]:
    """Split like FTS5's unicode61 tokenizer: letters, marks and digits form words."""
    tokens: list[str] = []
    current: list[str] = []
    for ch in text:
        if unicodedata.category(ch)[0] in "LMN":
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def build_match(q: str | None) -> str | None:
    """Turn what the user typed into a safe FTS5 query, or None if it has no words.

    - words are ANDed; "quoted words" must appear together in that order
    - the last word matches as a prefix while the user is still typing it
    - every token is quoted, so operators and punctuation can never break the query
    """
    q = q or ""
    parts: list[str] = []
    for m in PHRASE.finditer(q):
        toks = tokenize(m.group(1))[:MAX_TERMS]
        if toks:
            parts.append('"' + " ".join(toks) + '"')
    toks = tokenize(PHRASE.sub(" ", q))[:MAX_TERMS]
    still_typing = bool(q) and not q[-1].isspace() and q.rstrip()[-1:] != '"'
    for i, tok in enumerate(toks):
        is_last = i == len(toks) - 1
        parts.append(f'"{tok}"*' if is_last and still_typing and len(tok) >= 2 else f'"{tok}"')
    return " ".join(parts) or None


def _day(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"Invalid {label} date: {value!r} (use YYYY-MM-DD)") from None


def run_search(
    conn: sqlite3.Connection,
    q: str | None = None,
    type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort: str = "relevance",
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
) -> dict:
    match = build_match(q)
    where: list[str] = []
    args: list = []

    if match:
        source = "files_fts JOIN files f ON f.id = files_fts.rowid"
        where.append("files_fts MATCH ?")
        args.append(match)
    else:
        source = "files f"
    if type:
        where.append("f.type = ?")
        args.append(type)
    if status:
        where.append("f.status = ?")
        args.append(status)
    start = _day(date_from, "start") if date_from else None
    end = _day(date_to, "end") if date_to else None
    if start and end and start > end:
        raise ValueError("The start date is after the end date")
    if start:
        where.append("f.modified >= ?")
        args.append(start.isoformat())
    if end:
        where.append("f.modified < ?")  # exclusive next day keeps the whole end day
        args.append((end + timedelta(days=1)).isoformat())
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    if match and sort == "relevance":
        order = f"ORDER BY bm25(files_fts, {NAME_WEIGHT}, {TEXT_WEIGHT}), f.modified DESC"
    elif sort == "oldest":
        order = "ORDER BY f.modified ASC, f.name"
    else:
        order = "ORDER BY f.modified DESC, f.name"

    if match:
        select = (
            "SELECT f.path, f.name, f.ext, f.type, f.size, f.modified, f.status, "
            "highlight(files_fts, 0, ?, ?) AS name_hl, snippet(files_fts, 1, ?, ?, ?, 20) AS snippet"
        )
        select_args = [MARK_START, MARK_END, MARK_START, MARK_END, ELLIPSIS]
    else:
        select = "SELECT f.path, f.name, f.ext, f.type, f.size, f.modified, f.status, f.name AS name_hl, '' AS snippet"
        select_args = []

    total = conn.execute(f"SELECT COUNT(*) FROM {source}{clause}", args).fetchone()[0]
    rows = conn.execute(
        f"{select} FROM {source}{clause} {order} LIMIT ? OFFSET ?",
        select_args + args + [limit, offset],
    ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["snippet"] = (d["snippet"] or "").replace("\n", " ").strip()
        d["missing"] = not os.path.isfile(d["path"])  # deleted or on an unplugged drive
        results.append(d)

    pending = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'pending'").fetchone()[0]
    return {
        "query": q or "",
        "total": total,
        "offset": offset,
        "pending": pending,
        "results": results,
    }
