"""Milestone 5: retries, resets, folder management, missing files, offline behaviour."""
import importlib.util
import os
import re
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, extractors, ocr, thumbs, worker  # noqa: E402
from app.indexer import forget_folder, sync_folder  # noqa: E402
from app.scanner import scan_folder  # noqa: E402
from app.search import run_search  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL_EXTRACT = extractors.extract


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "files"
        (self.root / "sub").mkdir(parents=True)
        os.environ["UNIVERSAL_SEARCH_DB"] = str(self.base / "index.db")

    def write(self, rel, text="hello world"):
        p = self.root / rel
        p.write_text(text)
        return p

    def scan(self, folder=None):
        with db.connect() as conn:
            return sync_folder(conn, scan_folder(folder or self.root))

    def row(self, name):
        with db.connect() as conn:
            return conn.execute("SELECT * FROM files WHERE name = ?", (name,)).fetchone()

    def names(self, q="", **kw):
        with db.connect() as conn:
            return sorted(r["name"] for r in run_search(conn, q, limit=100, **kw)["results"])


class RetryTests(Base):
    def setUp(self):
        super().setUp()
        self.write("a.txt", "retry me please")
        self.scan()
        patcher = mock.patch.object(worker, "RETRY_DELAYS", (0, 0, 0))
        patcher.start()
        self.addCleanup(patcher.stop)

    def flaky(self, failures, exc):
        state = {"n": 0}

        def fake(path, ftype, ext, engine):
            state["n"] += 1
            if state["n"] <= failures:
                raise exc
            return REAL_EXTRACT(path, ftype, ext, engine)

        return mock.patch.object(extractors, "extract", fake), state

    def test_classification(self):
        self.assertTrue(worker.is_transient(PermissionError("locked")))
        self.assertTrue(worker.is_transient(TimeoutError()))
        self.assertTrue(worker.is_transient(OSError(16, "busy")))
        self.assertFalse(worker.is_transient(FileNotFoundError()))
        self.assertFalse(worker.is_transient(ValueError("corrupt")))
        self.assertFalse(worker.is_transient(OSError("cannot identify image file")))  # like PIL's error

    def test_temporary_failure_is_retried_and_then_succeeds(self):
        patch, state = self.flaky(2, PermissionError("file is open in another program"))
        with patch:
            counts = worker.process_pending(engine=None, wait_for_retries=True)
        self.assertEqual((counts["retry"], counts["done"], state["n"]), (2, 1, 3))
        r = self.row("a.txt")
        self.assertEqual((r["status"], r["attempts"], r["retry_after"]), ("done", 2, None))
        self.assertEqual(self.names("retry"), ["a.txt"])

    def test_gives_up_after_the_last_attempt(self):
        patch, state = self.flaky(99, PermissionError("still locked"))
        with patch:
            counts = worker.process_pending(engine=None, wait_for_retries=True)
        self.assertEqual((counts["retry"], counts["failed"], state["n"]), (3, 1, 4))
        with db.connect() as conn:
            err = conn.execute("SELECT error FROM content").fetchone()["error"]
        self.assertIn("gave up after 4 attempts", err)
        self.assertEqual(self.row("a.txt")["status"], "failed")

    def test_permanent_error_is_not_retried(self):
        patch, state = self.flaky(99, ValueError("corrupt"))
        with patch:
            counts = worker.process_pending(engine=None, wait_for_retries=True)
        self.assertEqual((counts["failed"], counts["retry"], state["n"]), (1, 0, 1))

    def test_file_waiting_for_its_retry_is_left_pending_without_busy_looping(self):
        with mock.patch.object(worker, "RETRY_DELAYS", (3600, 3600, 3600)):
            patch, state = self.flaky(99, PermissionError("locked"))
            with patch:
                started = time.time()
                counts = worker.process_pending(engine=None)  # not waiting
        self.assertLess(time.time() - started, 5)
        self.assertEqual((counts["retry"], state["n"]), (1, 1))
        r = self.row("a.txt")
        self.assertEqual((r["status"], r["attempts"]), ("pending", 1))
        self.assertIsNotNone(r["retry_after"])
        self.assertEqual(sum(worker.process_pending(engine=None).values()), 0)  # not due yet

    def test_pause_interrupts_the_wait_for_a_retry(self):
        import threading

        with mock.patch.object(worker, "RETRY_DELAYS", (3600, 3600, 3600)):
            patch, _ = self.flaky(99, PermissionError("locked"))
            stop = threading.Event()
            threading.Timer(0.5, stop.set).start()
            with patch:
                started = time.time()
                worker.process_pending(stop, engine=None, wait_for_retries=True)
        self.assertLess(time.time() - started, 10)

    def test_changed_file_and_manual_retry_reset_the_attempt_count(self):
        with db.connect() as conn:
            conn.execute("UPDATE files SET attempts = 2, retry_after = '2999-01-01T00:00:00'")
            conn.commit()
        p = self.root / "a.txt"
        p.write_text("changed content now")
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 500))
        self.scan()
        r = self.row("a.txt")
        self.assertEqual((r["attempts"], r["retry_after"]), (0, None))
        with db.connect() as conn:
            conn.execute("UPDATE files SET status = 'failed', attempts = 3")
            conn.commit()
            self.assertEqual(worker.retry_failed(conn), 1)
        self.assertEqual(self.row("a.txt")["attempts"], 0)

    def test_vanished_file_fails_with_a_clear_message_and_scan_removes_it(self):
        (self.root / "a.txt").unlink()
        worker.process_pending(engine=None)
        with db.connect() as conn:
            err = conn.execute("SELECT error FROM content").fetchone()["error"]
        self.assertIn("no longer exists", err)
        self.scan()
        self.assertEqual(self.names(), [])


class FolderManagementTests(Base):
    def setUp(self):
        super().setUp()
        self.write("top.txt", "alpha")
        self.write("sub/inner.txt", "beta")
        sibling = self.base / "files2"
        sibling.mkdir()
        (sibling / "other.txt").write_text("gamma")
        self.scan()
        self.scan(sibling)
        worker.process_pending(engine=None)

    def test_forget_folder_removes_only_that_folder(self):
        with db.connect() as conn:
            removed = forget_folder(conn, str(self.root))
        self.assertEqual(removed, 2)
        self.assertEqual(self.names(), ["other.txt"])  # sibling "files2" is untouched
        self.assertEqual(self.names("alpha"), [])
        with db.connect() as conn:
            self.assertEqual([r["path"] for r in db.list_roots(conn)], [str(self.base / "files2")])
        self.assertTrue((self.root / "top.txt").exists())  # originals are never deleted

    def test_forget_subfolder_keeps_the_rest(self):
        with db.connect() as conn:
            forget_folder(conn, str(self.root / "sub"))
        self.assertEqual(self.names(), ["other.txt", "top.txt"])

    def test_list_roots_counts_files(self):
        with db.connect() as conn:
            roots = {r["path"]: r["files"] for r in db.list_roots(conn)}
        self.assertEqual(roots, {str(self.root): 2, str(self.base / "files2"): 1})

    def test_reset_clears_everything_but_not_the_originals(self):
        with db.connect() as conn:
            db.reset_index(conn)
            for table in ("files", "content", "roots", "files_fts"):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0, table)
        self.assertEqual(self.names("alpha"), [])
        self.assertTrue((self.root / "top.txt").exists())
        self.assertEqual(self.scan()["added"], 2)  # and indexing works again afterwards

    def test_unplugged_folder_keeps_its_files_flagged_as_missing(self):
        away = self.base / "files2"
        moved = self.base / "files2_unplugged"
        away.rename(moved)
        self.scan()  # rescanning the other folder must not touch it
        with db.connect() as conn:
            res = run_search(conn, "gamma")["results"]
        self.assertEqual([(r["name"], r["missing"]) for r in res], [("other.txt", True)])
        moved.rename(away)
        with db.connect() as conn:
            self.assertFalse(run_search(conn, "gamma")["results"][0]["missing"])

    def test_status_filter(self):
        with db.connect() as conn:
            conn.execute("UPDATE files SET status = 'failed' WHERE name = 'top.txt'")
            conn.commit()
        self.assertEqual(self.names("", status="failed"), ["top.txt"])
        self.assertEqual(len(self.names("", status="done")), 2)


class ThumbCleanupTests(Base):
    def test_cleanup_keeps_current_previews_and_removes_orphans(self):
        from PIL import Image

        Image.new("RGB", (40, 40), "red").save(self.root / "a.png")
        self.scan()
        with db.connect() as conn:
            row = conn.execute("SELECT id, path, mtime_ns FROM files").fetchone()
            good = thumbs.get_thumbnail(row)
            self.assertTrue(good.is_file())
            orphan = thumbs.thumbs_dir() / "999_1.jpg"
            orphan.write_bytes(b"x")
            in_progress = thumbs.thumbs_dir() / "5_5.jpg.123.tmp"
            in_progress.write_bytes(b"x")
            self.assertEqual(thumbs.cleanup(conn), 1)
        self.assertTrue(good.exists())
        self.assertFalse(orphan.exists())
        self.assertTrue(in_progress.exists())
        thumbs.clear()
        self.assertFalse(thumbs.thumbs_dir().exists())

    def test_edited_image_gets_a_new_preview_and_the_old_one_is_cleaned(self):
        from PIL import Image

        p = self.root / "a.png"
        Image.new("RGB", (40, 40), "red").save(p)
        self.scan()
        with db.connect() as conn:
            old = thumbs.get_thumbnail(conn.execute("SELECT id, path, mtime_ns FROM files").fetchone())
        Image.new("RGB", (60, 60), "blue").save(p)
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 500))
        self.scan()
        with db.connect() as conn:
            new = thumbs.get_thumbnail(conn.execute("SELECT id, path, mtime_ns FROM files").fetchone())
            thumbs.cleanup(conn)
        self.assertNotEqual(old, new)
        self.assertTrue(new.exists())
        self.assertFalse(old.exists())


class WorkerThreadTests(Base):
    def test_background_thread_processes_everything_and_stops(self):
        for i in range(6):
            self.write(f"f{i}.txt", f"document number {i}")
        self.scan()
        w = worker.IndexWorker()
        self.assertTrue(w.start())
        self.assertFalse(w.start())  # already running (or just finished): never two workers at once
        deadline = time.time() + 10
        while w.running and time.time() < deadline:
            time.sleep(0.05)
        st = w.status()
        self.assertFalse(st["running"])
        self.assertEqual((st["counts"]["done"], st["counts"]["pending"], st["retrying"]), (6, 0, 0))
        self.assertTrue(w.stop_and_wait(1))

    def test_stop_and_wait_pauses_a_long_run(self):
        for i in range(40):
            self.write(f"f{i}.txt", "x" * 100)
        self.scan()
        gate = __import__("threading").Event()

        def slow(path, ftype, ext, engine):
            gate.wait(0.05)
            return REAL_EXTRACT(path, ftype, ext, engine)

        w = worker.IndexWorker()
        with mock.patch.object(extractors, "extract", slow):
            w.start()
            time.sleep(0.3)
            self.assertTrue(w.stop_and_wait(10))
        self.assertGreater(w.status()["counts"]["pending"], 0)  # stopped early, work preserved
        w.start()  # and it can resume
        deadline = time.time() + 20
        while w.running and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(w.status()["counts"]["pending"], 0)


class OfflineAndPrivacyTests(Base):
    def test_whole_pipeline_works_with_the_network_blocked(self):
        calls = []

        def blocked(*a, **k):
            calls.append(a)
            raise OSError("network access attempted")

        self.write("notes.txt", "offline notes about probability")
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (700, 160), "white")
        ImageDraw.Draw(img).text((20, 50), "Offline OCR test 2024", fill="black")
        img.save(self.root / "shot.png")
        ocr.reset()
        self.addCleanup(ocr.reset)
        with mock.patch.object(socket.socket, "connect", blocked), \
                mock.patch.object(socket.socket, "connect_ex", blocked), \
                mock.patch.object(socket, "getaddrinfo", blocked), \
                mock.patch.object(socket, "create_connection", blocked):
            engine = ocr.get_engine()
            if importlib.util.find_spec("rapidocr") or importlib.util.find_spec("rapidocr_onnxruntime"):
                self.assertIsNotNone(
                    engine, "RapidOCR is installed but could not start without internet (its models may need a download)"
                )
            self.scan()
            worker.process_pending(engine=engine)
            self.assertEqual(self.names("probability"), ["notes.txt"])
        self.assertEqual(calls, [], "something tried to use the network")
        print(f"\n    offline check used OCR engine: {engine.name if engine else 'none installed'}")

    def test_web_page_loads_nothing_from_the_internet(self):
        for f in (ROOT / "static").iterdir():
            text = f.read_text(encoding="utf-8")
            self.assertEqual(re.findall(r"(?:https?:)?//[\w.-]+\.[a-z]{2,}", text), [], f.name)

    def test_server_is_started_on_the_loopback_address_only(self):
        bat = (ROOT / "run.bat").read_text()
        self.assertIn("--host 127.0.0.1", bat)
        self.assertNotIn("0.0.0.0", bat)


if __name__ == "__main__":
    unittest.main()
