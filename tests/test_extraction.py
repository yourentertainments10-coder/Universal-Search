"""Run with:  python -m unittest discover -s tests -v"""
import os
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app import db, extractors, ocr  # noqa: E402
from app.indexer import sync_folder  # noqa: E402
from app.scanner import scan_folder  # noqa: E402
from app.worker import process_pending, retry_failed  # noqa: E402

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/Library/Fonts/Arial.ttf",
]


class FakeEngine:
    name = "fake"

    def read(self, img):
        assert img.mode == "RGB"
        return "FAKE OCR TEXT"


def make_docx(path, paragraphs):
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", f"<w:document {ns}><w:body>{body}</w:body></w:document>")


def make_text_pdf(path, text):
    stream = f"BT /F1 18 Tf 50 700 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    Path(path).write_bytes(bytes(out))


def text_image(text, size=(900, 220)):
    font_path = next((p for p in FONT_PATHS if os.path.isfile(p)), None)
    img = Image.new("RGB", size, "white")
    if font_path:
        ImageDraw.Draw(img).text((30, 60), text, fill="black", font=ImageFont.truetype(font_path, 64))
    return img, font_path


class EnvironmentTests(unittest.TestCase):
    def test_required_packages_are_installed(self):
        import importlib.util

        for name in ("PIL", "pypdfium2"):
            self.assertIsNotNone(
                importlib.util.find_spec(name),
                f"{name} is missing. Activate the venv (.venv\\Scripts\\activate) and run: pip install -r requirements.txt",
            )


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_text_encodings(self):
        (self.dir / "u8.txt").write_bytes("caf\u00e9 formula".encode("utf-8"))
        (self.dir / "u16.txt").write_bytes("caf\u00e9 formula".encode("utf-16"))
        (self.dir / "w1252.txt").write_bytes("caf\u00e9 formula".encode("cp1252"))
        (self.dir / "bom.md").write_bytes(b"\xef\xbb\xbfcaf\xc3\xa9 formula")
        for name in ("u8.txt", "u16.txt", "w1252.txt", "bom.md"):
            ex = extractors.extract(str(self.dir / name), "document", Path(name).suffix, None)
            self.assertEqual(ex.text, "caf\u00e9 formula", name)

    def test_docx(self):
        make_docx(self.dir / "n.docx", ["NIMCET maths notes", "", "sin 3x formula"])
        ex = extractors.extract(str(self.dir / "n.docx"), "document", ".docx", None)
        self.assertEqual(ex.text, "NIMCET maths notes\nsin 3x formula")
        self.assertEqual(ex.method, "docx")

    def test_pdf_with_text_layer_needs_no_ocr(self):
        make_text_pdf(self.dir / "t.pdf", "Trigonometry formula sheet sin 3x")
        ex = extractors.extract(str(self.dir / "t.pdf"), "pdf", ".pdf", None)
        self.assertIn("Trigonometry formula sheet", ex.text)
        self.assertEqual((ex.method, ex.needs_ocr, ex.pages), ("pdf-text", False, 1))

    def test_scanned_pdf_uses_ocr_engine(self):
        text_image("x")[0].save(self.dir / "s.pdf", "PDF")
        ex = extractors.extract(str(self.dir / "s.pdf"), "pdf", ".pdf", FakeEngine())
        self.assertEqual((ex.text, ex.method, ex.engine), ("FAKE OCR TEXT", "pdf-ocr", "fake"))

    def test_scanned_pdf_without_engine_is_flagged(self):
        text_image("x")[0].save(self.dir / "s.pdf", "PDF")
        ex = extractors.extract(str(self.dir / "s.pdf"), "pdf", ".pdf", None)
        self.assertTrue(ex.needs_ocr)

    def test_image_ocr_and_transparent_png_is_flattened_on_white(self):
        Image.new("RGBA", (50, 50), (0, 0, 0, 0)).save(self.dir / "t.png")
        ex = extractors.extract(str(self.dir / "t.png"), "image", ".png", FakeEngine())  # asserts RGB
        self.assertEqual(ex.text, "FAKE OCR TEXT")

    def test_image_without_engine_is_flagged(self):
        Image.new("RGB", (20, 20)).save(self.dir / "a.png")
        self.assertTrue(extractors.extract(str(self.dir / "a.png"), "image", ".png", None).needs_ocr)

    def test_heic_is_skipped(self):
        with self.assertRaises(extractors.Skipped):
            extractors.extract("x.heic", "image", ".heic", FakeEngine())

    def test_real_tesseract_reads_an_image(self):
        engine = ocr._load_tesseract()
        img, font = text_image("NIMCET Maths")
        if engine is None or font is None:
            self.skipTest("Tesseract or a TrueType font is not available")
        img.save(self.dir / "r.png")
        ex = extractors.extract(str(self.dir / "r.png"), "image", ".png", engine)
        self.assertIn("NIMCET", ex.text.upper())

    def test_real_rapidocr_reads_an_image(self):
        engine = ocr._load_rapidocr()
        img, font = text_image("NIMCET Maths")
        if engine is None or font is None:
            self.skipTest("RapidOCR or a TrueType font is not available")
        img.save(self.dir / "r.png")
        ex = extractors.extract(str(self.dir / "r.png"), "image", ".png", engine)
        print(f"\n    RapidOCR read: {ex.text!r}")
        self.assertIn("NIMCET", ex.text.upper())

    def test_real_tesseract_reads_a_scanned_pdf(self):
        engine = ocr._load_tesseract()
        img, font = text_image("Trigonometry Sheet")
        if engine is None or font is None:
            self.skipTest("Tesseract or a TrueType font is not available")
        img.save(self.dir / "r.pdf", "PDF")
        ex = extractors.extract(str(self.dir / "r.pdf"), "pdf", ".pdf", engine)
        self.assertIn("TRIGONOMETRY", ex.text.upper())
        self.assertEqual(ex.method, "pdf-ocr")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.root = base / "files"
        self.root.mkdir()
        os.environ["UNIVERSAL_SEARCH_DB"] = str(base / "index.db")
        (self.root / "a.txt").write_text("hello trigonometry notes")
        make_docx(self.root / "b.docx", ["college farewell plan"])
        make_text_pdf(self.root / "c.pdf", "Bond prospectus interest rate table")
        Image.new("RGB", (60, 60), "white").save(self.root / "d.png")
        (self.root / "e.heic").write_bytes(b"not really heic")
        text_image("x")[0].save(self.root / "f.pdf", "PDF")
        (self.root / "bad.pdf").write_bytes(b"this is not a pdf")
        self.scan()

    def scan(self):
        with db.connect() as conn:
            return sync_folder(conn, scan_folder(self.root))

    def statuses(self):
        with db.connect() as conn:
            return {r["name"]: r["status"] for r in conn.execute("SELECT name, status FROM files")}

    def content(self, name):
        with db.connect() as conn:
            return conn.execute(
                "SELECT c.* FROM content c JOIN files f ON f.id = c.file_id WHERE f.name = ?", (name,)
            ).fetchone()

    def test_everything_starts_pending(self):
        self.assertEqual(set(self.statuses().values()), {"pending"})

    def test_process_with_engine(self):
        counts = process_pending(engine=FakeEngine())
        self.assertEqual(
            self.statuses(),
            {"a.txt": "done", "b.docx": "done", "c.pdf": "done", "d.png": "done",
             "e.heic": "skipped", "f.pdf": "done", "bad.pdf": "failed"},
        )
        self.assertEqual((counts["done"], counts["failed"], counts["skipped"]), (5, 1, 1))
        self.assertEqual(self.content("a.txt")["text"], "hello trigonometry notes")
        self.assertEqual(self.content("d.png")["text"], "FAKE OCR TEXT")
        self.assertEqual(self.content("f.pdf")["method"], "pdf-ocr")
        self.assertIn("PdfiumError", self.content("bad.pdf")["error"])
        self.assertIn("HEIC", self.content("e.heic")["note"])

    def test_no_engine_waits_then_retry_finishes(self):
        process_pending(engine=None)
        s = self.statuses()
        self.assertEqual((s["d.png"], s["f.pdf"], s["a.txt"], s["c.pdf"]), ("needs_ocr",) * 2 + ("done",) * 2)
        with db.connect() as conn:
            self.assertEqual(retry_failed(conn), 3)  # d.png, f.pdf, bad.pdf
        process_pending(engine=FakeEngine())
        s = self.statuses()
        self.assertEqual((s["d.png"], s["f.pdf"], s["bad.pdf"]), ("done", "done", "failed"))

    def test_second_run_does_nothing(self):
        process_pending(engine=FakeEngine())
        self.assertEqual(sum(process_pending(engine=FakeEngine()).values()), 0)

    def test_changed_file_loses_stale_text_and_is_reprocessed(self):
        process_pending(engine=FakeEngine())
        p = self.root / "a.txt"
        p.write_text("completely different words now")
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 100))
        r = self.scan()
        self.assertEqual(r["updated"], 1)
        self.assertIsNone(self.content("a.txt"))
        self.assertEqual(self.statuses()["a.txt"], "pending")
        process_pending(engine=FakeEngine())
        self.assertEqual(self.content("a.txt")["text"], "completely different words now")

    def test_deleted_file_takes_its_text_with_it(self):
        process_pending(engine=FakeEngine())
        (self.root / "a.txt").unlink()
        self.scan()
        with db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM content").fetchone()[0], 6)

    def test_file_changed_during_processing_is_not_marked_done(self):
        def sabotage(name):
            if name == "a.txt":
                p = self.root / "a.txt"
                p.write_text("edited while indexing")
                os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 500))
                self.scan()
        stop = threading.Event()
        # process only a.txt: pause right after it
        seen = []

        def on_file(name):
            seen.append(name)
            sabotage(name)
            if name == "a.txt":
                stop.set()

        process_pending(stop, on_file, engine=FakeEngine())
        self.assertEqual(self.statuses()["a.txt"], "pending")
        process_pending(engine=FakeEngine())
        self.assertEqual(self.content("a.txt")["text"], "edited while indexing")

    def test_pause_stops_before_next_file(self):
        stop = threading.Event()
        stop.set()
        self.assertEqual(sum(process_pending(stop, engine=FakeEngine()).values()), 0)

    def test_virtualenv_and_junk_folders_are_not_scanned(self):
        venv = self.root / "myenv"
        (venv / "Lib").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = x")
        (venv / "Lib" / "README.md").write_text("library readme")
        (self.root / "node_modules" / "pkg").mkdir(parents=True)
        (self.root / "node_modules" / "pkg" / "x.md").write_text("junk")
        (self.root / "keep").mkdir()
        (self.root / "keep" / "mine.md").write_text("my notes")
        names = {f.name for f in scan_folder(self.root).files}
        self.assertIn("mine.md", names)
        self.assertNotIn("README.md", names)
        self.assertNotIn("x.md", names)


if __name__ == "__main__":
    unittest.main()
