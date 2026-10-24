"""Run with:  python -m unittest discover -s tests -v"""
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, extractors  # noqa: E402
from app.indexer import sync_folder  # noqa: E402
from app.scanner import scan_folder  # noqa: E402
from app.search import MARK_END, MARK_START, build_match, run_search  # noqa: E402
from app.worker import _store  # noqa: E402


class BuildMatchTests(unittest.TestCase):
    def test_words_are_anded_and_last_word_is_prefix(self):
        self.assertEqual(build_match("sin 3x"), '"sin" "3x"*')

    def test_trailing_space_means_exact_last_word(self):
        self.assertEqual(build_match("sin "), '"sin"')

    def test_single_letter_is_never_a_prefix(self):
        self.assertEqual(build_match("x"), '"x"')

    def test_quoted_phrase(self):
        self.assertEqual(build_match('"college farewell"'), '"college farewell"')
        self.assertEqual(build_match('photo "college farewell"'), '"college farewell" "photo"')

    def test_punctuation_is_just_a_separator(self):
        self.assertEqual(build_match("e-mail, sin^3(x)"), '"e" "mail" "sin" "3" "x"')

    def test_empty_and_symbols_only(self):
        for q in (None, "", "   ", '""', "***", "()", "-"):
            self.assertIsNone(build_match(q), repr(q))

    def test_operators_are_neutralised(self):
        self.assertEqual(build_match("a OR b NOT cc"), '"a" "OR" "b" "NOT" "cc"*')

    def test_devanagari_words_stay_whole(self):
        self.assertEqual(build_match("\u0938\u0942\u0924\u094d\u0930 "), '"\u0938\u0942\u0924\u094d\u0930"')


class SearchTests(unittest.TestCase):
    FILES = {
        # name: (text, modified date)
        "formulas.png": ("sin 3x formula trigonometry cheat", "2024-03-10"),
        "reverse.txt": ("3x sin written backwards", "2024-04-01"),
        "farewell_group_photo.jpg": ("", "2024-05-20"),
        "nimcet_notes.pdf": ("NIMCET mathematics probability notes", "2023-11-02"),
        "salary.txt": ("salary calculation 45000 monthly", "2025-01-15"),
        "caf\u00e9_menu.txt": ("coffee and tea", "2024-07-01"),
        "bond.txt": ("irrelevant words only", "2024-06-01"),
        "other.docx": ("bond bond bond bond bond", "2024-06-02"),
        "hindi.txt": ("\u0924\u094d\u0930\u093f\u0915\u094b\u0923\u092e\u093f\u0924\u093f \u0938\u0942\u0924\u094d\u0930", "2024-08-01"),
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.root = base / "files"
        self.root.mkdir()
        os.environ["UNIVERSAL_SEARCH_DB"] = str(base / "index.db")
        for name, (_, day) in self.FILES.items():
            p = self.root / name
            p.write_bytes(b"x")
            ts = datetime.fromisoformat(day + "T12:00:00").timestamp()
            os.utime(p, (ts, ts))
        self.scan()
        for name, (text, _) in self.FILES.items():
            if text:
                self.put_text(name, text)

    def scan(self):
        with db.connect() as conn:
            return sync_folder(conn, scan_folder(self.root))

    def put_text(self, name, text):
        with db.connect() as conn:
            row = conn.execute("SELECT id, mtime_ns FROM files WHERE name = ?", (name,)).fetchone()
            _store(conn, row, "done", extractors.Extraction(text, "text"), None, None)

    def search(self, q="", **kw):
        with db.connect() as conn:
            return run_search(conn, q, **kw)

    def names(self, q="", **kw):
        return [r["name"] for r in self.search(q, **kw)["results"]]

    def test_finds_by_content(self):
        self.assertEqual(self.names("mathematics"), ["nimcet_notes.pdf"])

    def test_finds_by_filename_before_any_text_is_extracted(self):
        self.assertEqual(self.names("farewell group"), ["farewell_group_photo.jpg"])

    def test_phrase_requires_word_order(self):
        self.assertEqual(set(self.names("sin 3x")), {"formulas.png", "reverse.txt"})
        self.assertEqual(self.names('"sin 3x"'), ["formulas.png"])

    def test_prefix_while_typing_but_not_after_space(self):
        self.assertEqual(self.names("trigon"), ["formulas.png"])
        self.assertEqual(self.names("trigon "), [])

    def test_filename_match_outranks_body_match(self):
        self.assertEqual(self.names("bond")[:2], ["bond.txt", "other.docx"])

    def test_diacritics_are_ignored(self):
        self.assertEqual(self.names("cafe"), ["caf\u00e9_menu.txt"])

    def test_devanagari_search(self):
        self.assertEqual(self.names("\u0938\u0942\u0924\u094d\u0930"), ["hindi.txt"])

    def test_type_filter(self):
        self.assertEqual(self.names("", type="pdf"), ["nimcet_notes.pdf"])
        self.assertEqual(set(self.names("sin", type="image")), {"formulas.png"})

    def test_date_filters_include_both_end_days(self):
        self.assertEqual(self.names("", date_from="2024-05-20", date_to="2024-06-01", sort="oldest"),
                         ["farewell_group_photo.jpg", "bond.txt"])
        self.assertEqual(self.names("notes", date_to="2023-11-02"), ["nimcet_notes.pdf"])
        self.assertEqual(self.names("salary", date_to="2024-12-31"), [])

    def test_no_query_lists_newest_first_and_sorting_works(self):
        self.assertEqual(self.names("")[0], "salary.txt")
        self.assertEqual(self.names("", sort="oldest")[0], "nimcet_notes.pdf")

    def test_snippet_marks_the_match(self):
        r = self.search("probability")["results"][0]
        self.assertIn(f"{MARK_START}probability{MARK_END}", r["snippet"])
        r = self.search("nimcet")["results"][0]
        self.assertIn(f"{MARK_START}nimcet{MARK_END}", r["name_hl"].lower())

    def test_pagination(self):
        page1 = self.search("", limit=4, offset=0)
        page2 = self.search("", limit=4, offset=4)
        self.assertEqual(page1["total"], len(self.FILES))
        self.assertEqual(len(page1["results"]), 4)
        self.assertFalse({r["name"] for r in page1["results"]} & {r["name"] for r in page2["results"]})

    def test_hostile_queries_never_raise(self):
        for q in ['"', "'", "a OR", "NEAR(a b)", "-x", "*", "(", "a:b", "name:formulas",
                  "AND", "\\", "'; DROP TABLE files; --", "%", "text:*", '"unterminated phrase']:
            self.search(q)
        self.assertEqual(self.search("")["total"], len(self.FILES))

    def test_invalid_dates_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.search("x", date_from="2024-13-45")
        with self.assertRaises(ValueError):
            self.search("x", date_from="2024-05-02", date_to="2024-05-01")

    def test_changed_file_old_text_stops_matching_until_reprocessed(self):
        p = self.root / "salary.txt"
        p.write_bytes(b"new longer content")
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 1000))
        self.assertEqual(self.scan()["updated"], 1)
        self.assertEqual(self.names("salary calculation"), [])  # stale text gone at once
        self.assertEqual(self.names("salary"), ["salary.txt"])  # still findable by name
        self.put_text("salary.txt", "brand new words")
        self.assertEqual(self.names("brand"), ["salary.txt"])

    def test_deleted_file_disappears_from_search(self):
        (self.root / "formulas.png").unlink()
        self.scan()
        self.assertEqual(self.names("trigonometry"), [])
        with db.connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]
            self.assertEqual(n, len(self.FILES) - 1)

    def test_pending_count_is_reported(self):
        # only the photo with no extracted text yet is still pending
        self.assertEqual(self.search("x")["pending"], 1)


class UpgradeTests(unittest.TestCase):
    def test_existing_milestone3_database_becomes_searchable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.db"
            os.environ["UNIVERSAL_SEARCH_DB"] = str(path)
            raw = sqlite3.connect(path)
            raw.executescript(f"BEGIN;{db.MIGRATIONS[0]}{db.MIGRATIONS[1]}PRAGMA user_version=2;COMMIT;")
            raw.execute("INSERT INTO files (id, path, path_key, name, ext, type, size, mtime_ns, modified, status,"
                        " first_seen, indexed_at) VALUES (1, '/a/old.png', '/a/old.png', 'old.png', '.png', 'image',"
                        " 1, 1, '2024-01-01T00:00:00', 'done', 'x', 'x')")
            raw.execute("INSERT INTO content (file_id, text, method, extracted_at) VALUES (1, 'legacy words', 'ocr', 'x')")
            raw.commit()
            raw.close()
            with db.connect() as conn:
                self.assertEqual([r["name"] for r in run_search(conn, "legacy")["results"]], ["old.png"])
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], len(db.MIGRATIONS))


if __name__ == "__main__":
    unittest.main()
