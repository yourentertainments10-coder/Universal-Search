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


def _filter_clauses(type, date_from, date_to, status) -> tuple[list[str], list]:
    """WHERE fragments (on files aliased f) for the type, status and date filters."""
    where: list[str] = []
    args: list = []
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
    return where, args


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
    extra_where, extra_args = _filter_clauses(type, date_from, date_to, status)
    where += extra_where
    args += extra_args
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


RRF_K = 60            # reciprocal rank fusion constant
KEYWORD_CANDIDATES = 200


def run_hybrid_search(
    conn: sqlite3.Connection,
    q: str,
    hits: list[tuple[int, float]],
    mode: str = "auto",
    type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort: str = "relevance",
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
) -> dict:
    """Merge keyword matches with images that look like the query (semantic `hits`).

    mode "auto" fuses both rankings (reciprocal rank fusion: a file found by both
    ranks highest); mode "semantic" returns only the meaning matches. The type, date
    and status filters apply to both.
    """
    filt_where, filt_args = _filter_clauses(type, date_from, date_to, status)
    filt = (" AND " + " AND ".join(filt_where)) if filt_where else ""

    similarity = dict(hits)
    sem_rank: list[int] = []
    if hits:
        marks = ",".join("?" * len(hits))
        allowed = {
            r[0]
            for r in conn.execute(f"SELECT f.id FROM files f WHERE f.id IN ({marks}){filt}", [h[0] for h in hits] + filt_args)
        }
        sem_rank = [fid for fid, _ in hits if fid in allowed]

    match = build_match(q)
    kw_rank: list[int] = []
    if match and mode != "semantic":
        kw_rank = [
            r[0]
            for r in conn.execute(
                "SELECT f.id FROM files_fts JOIN files f ON f.id = files_fts.rowid "
                f"WHERE files_fts MATCH ?{filt} ORDER BY bm25(files_fts, {NAME_WEIGHT}, {TEXT_WEIGHT}) LIMIT ?",
                [match] + filt_args + [KEYWORD_CANDIDATES],
            )
        ]

    score: dict[int, float] = {}
    for ranking in (kw_rank, sem_rank):
        for i, fid in enumerate(ranking, start=1):
            score[fid] = score.get(fid, 0.0) + 1.0 / (RRF_K + i)
    candidates = list(score)

    details: dict[int, dict] = {}
    if candidates:
        marks = ",".join("?" * len(candidates))
        for r in conn.execute(
            f"SELECT f.id, f.path, f.name, f.ext, f.type, f.size, f.modified, f.status FROM files f WHERE f.id IN ({marks})",
            candidates,
        ):
            details[r["id"]] = dict(r)

    if sort == "oldest":
        order = sorted(candidates, key=lambda i: (details[i]["modified"], details[i]["name"]))
    elif sort == "newest" or not candidates:
        order = sorted(candidates, key=lambda i: (details[i]["modified"], details[i]["name"]), reverse=True)
    else:
        order = sorted(candidates, key=lambda i: (-score[i], details[i]["modified"]), reverse=False)
    page = order[offset : offset + limit]

    marked: dict[int, tuple[str, str]] = {}
    page_kw = [i for i in page if i in set(kw_rank)]
    if match and page_kw:
        marks = ",".join("?" * len(page_kw))
        for r in conn.execute(
            "SELECT f.id, highlight(files_fts, 0, ?, ?) AS name_hl, snippet(files_fts, 1, ?, ?, ?, 20) AS snippet "
            f"FROM files_fts JOIN files f ON f.id = files_fts.rowid WHERE files_fts MATCH ? AND f.id IN ({marks})",
            [MARK_START, MARK_END, MARK_START, MARK_END, ELLIPSIS, match] + page_kw,
        ):
            marked[r["id"]] = (r["name_hl"], (r["snippet"] or "").replace("\n", " ").strip())

    kw_set, sem_set = set(kw_rank), set(sem_rank)
    results = []
    for fid in page:
        d = details[fid]
        d["name_hl"], d["snippet"] = marked.get(fid, (d["name"], ""))
        d["missing"] = not os.path.isfile(d["path"])
        d["source"] = "both" if fid in kw_set and fid in sem_set else ("keyword" if fid in kw_set else "meaning")
        d["similarity"] = round(similarity[fid], 3) if fid in sem_set else None
        results.append(d)

    pending = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'pending'").fetchone()[0]
    return {
        "query": q or "",
        "total": len(order),
        "offset": offset,
        "pending": pending,
        "semantic": True,
        "results": results,
    }
