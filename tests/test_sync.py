"""Run with:  python -m unittest discover -s tests -v   (no extra packages needed)"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.indexer import sync_folder  # noqa: E402
from app.scanner import ScanResult, scan_folder  # noqa: E402


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.root = base / "files"
        (self.root / "sub").mkdir(parents=True)
        os.environ["UNIVERSAL_SEARCH_DB"] = str(base / "index.db")
        self.write("a.png", "aaa")
        self.write("sub/b.pdf", "bbb")
        self.write("notes.txt", "ccc")
        self.write("ignore.exe", "zzz")

    def write(self, rel, text):
        p = self.root / rel
        p.write_text(text)
        return p

    def sync(self, root=None):
        with db.connect() as conn:
            return sync_folder(conn, scan_folder(root or self.root))

    def names(self):
        with db.connect() as conn:
            return sorted(f["name"] for f in db.list_files(conn))

    def test_first_scan_adds_supported_files_only(self):
        r = self.sync()
        self.assertEqual((r["added"], r["count"]), (3, 3))
        self.assertEqual(self.names(), ["a.png", "b.pdf", "notes.txt"])

    def test_rescan_creates_no_duplicates(self):
        self.sync()
        r = self.sync()
        self.assertEqual((r["added"], r["updated"], r["removed"], r["unchanged"]), (0, 0, 0, 3))
        self.assertEqual(len(self.names()), 3)

    def test_modified_file_is_detected_and_marked_pending(self):
        self.sync()
        with db.connect() as conn:
            conn.execute("UPDATE files SET status='done'")
            conn.commit()
        p = self.write("a.png", "changed content")
        future = time.time() + 100
        os.utime(p, (future, future))
        r = self.sync()
        self.assertEqual((r["updated"], r["unchanged"]), (1, 2))
        with db.connect() as conn:
            status = {x["name"]: x["status"] for x in conn.execute("SELECT name, status FROM files")}
        self.assertEqual(status, {"a.png": "pending", "b.pdf": "done", "notes.txt": "done"})

    def test_deleted_file_drops_out(self):
        self.sync()
        (self.root / "sub" / "b.pdf").unlink()
        r = self.sync()
        self.assertEqual(r["removed"], 1)
        self.assertEqual(self.names(), ["a.png", "notes.txt"])

    def test_index_survives_reconnect(self):
        self.sync()
        self.assertEqual(len(self.names()), 3)  # new connection each call

    def test_unreadable_folder_does_not_wipe_its_files(self):
        self.sync()
        (self.root / "sub" / "b.pdf").unlink()  # looks deleted, but folder was unreadable
        result = scan_folder(self.root)
        result.failed_dirs.append(str(self.root / "sub"))
        with db.connect() as conn:
            r = sync_folder(conn, result)
        self.assertEqual(r["removed"], 0)
        self.assertIn("b.pdf", self.names())

    def test_nested_root_scan_does_not_duplicate_or_remove_outer_files(self):
        self.sync()
        r = self.sync(self.root / "sub")
        self.assertEqual((r["added"], r["removed"], r["count"]), (0, 0, 1))
        self.assertEqual(len(self.names()), 3)

    def test_sibling_folder_with_same_prefix_is_not_touched(self):
        other = self.root.parent / "files2"
        other.mkdir()
        (other / "z.txt").write_text("z")
        self.sync(other)
        self.sync()  # scanning "files" must not treat "files2" as inside it
        self.assertIn("z.txt", self.names())

    def test_missing_root_raises_and_changes_nothing(self):
        self.sync()
        with self.assertRaises(NotADirectoryError):
            scan_folder(self.root / "nope")
        self.assertEqual(len(self.names()), 3)

    def test_schema_version_is_set(self):
        with db.connect() as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], len(db.MIGRATIONS))


if __name__ == "__main__":
    unittest.main()
