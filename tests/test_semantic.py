"""Milestone 6: semantic image search. The real CLIP model is replaced by stand-ins here;
run `python -m app.semantic check` to test the real model on your computer."""
import importlib.util
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
if HAVE_NUMPY:
    import numpy as np

from PIL import Image  # noqa: E402

from app import db, semantic  # noqa: E402
from app.indexer import sync_folder  # noqa: E402
from app.scanner import scan_folder  # noqa: E402
from app.search import MARK_START, run_hybrid_search, run_search  # noqa: E402

needs_numpy = unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "reddish": (200, 50, 50)}


class FakeEmbedder:
    """Stands in for CLIP: an image's vector is its average colour; text is understood if it names a colour."""

    def __init__(self, model_id="fake-1"):
        self.model_id = model_id

    def embed_image(self, img):
        v = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(axis=0)
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def embed_text(self, text):
        v = np.zeros(3, dtype=np.float32)
        for w in text.lower().split():
            if w in ("red", "green", "blue"):
                v += np.array([w == "red", w == "green", w == "blue"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else v


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "files"
        self.root.mkdir()
        os.environ["UNIVERSAL_SEARCH_DB"] = str(self.base / "index.db")
        semantic._cache.update(key=None, ids=None, mat=None)
        self.addCleanup(semantic.set_embedder, None)

    def image(self, name, color, day=None):
        p = self.root / name
        Image.new("RGB", (64, 48), COLORS.get(color, color)).save(p)
        if day:
            ts = time.mktime(time.strptime(day, "%Y-%m-%d")) + 43200
            os.utime(p, (ts, ts))
        return p

    def scan(self):
        with db.connect() as conn:
            return sync_folder(conn, scan_folder(self.root))

    def embed(self, embedder=None, **kw):
        return semantic.embed_pending(embedder=embedder or FakeEmbedder(), **kw)

    def hits(self, text, embedder=None):
        with db.connect() as conn:
            found = semantic.query_hits(conn, text, embedder or FakeEmbedder())
            names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM files")}
        return [(names[i], round(s, 2)) for i, s in found]


@needs_numpy
class PreprocessAndTokenTests(unittest.TestCase):
    def test_preprocess_shape_dtype_and_normalisation(self):
        arr = semantic.preprocess_image(Image.new("RGB", (300, 150), "white"))
        self.assertEqual((arr.shape, arr.dtype), ((1, 3, 224, 224), np.float32))
        expected = [(1 - m) / s for m, s in zip(semantic.CLIP_MEAN, semantic.CLIP_STD)]
        for c in range(3):
            self.assertAlmostEqual(float(arr[0, c].mean()), expected[c], places=4)

    def test_preprocess_centre_crops_wide_images(self):
        img = Image.new("RGB", (448, 224), "black")
        img.paste(Image.new("RGB", (224, 224), "white"), (112, 0))  # white only in the middle
        arr = semantic.preprocess_image(img)
        self.assertGreater(float(arr[0, 0].min()), 1.5)  # the cropped square is entirely white

    def test_preprocess_handles_tiny_and_non_rgb_images(self):
        for img in (Image.new("L", (10, 10), 128), Image.new("RGBA", (5, 400), (0, 0, 0, 255))):
            self.assertEqual(semantic.preprocess_image(img).shape, (1, 3, 224, 224))

    class Tok:
        def __init__(self): self.calls = []
        def no_padding(self): self.calls.append("no_padding")
        def no_truncation(self): self.calls.append("no_truncation")
        def token_to_id(self, t): return {"<|startoftext|>": 49406, "<|endoftext|>": 49407}.get(t)
        def encode(self, text, add_special_tokens=True):
            assert add_special_tokens is False, "special tokens must be added by us, not the tokenizer"
            return type("Enc", (), {"ids": [100 + i for i in range(len(text.split()))]})()

    def test_encode_text_adds_markers_pads_and_masks(self):
        ids, mask = semantic.encode_text(self.Tok(), "a dog on grass")
        self.assertEqual(len(ids), 77)
        self.assertEqual(ids[:6], [49406, 100, 101, 102, 103, 49407])
        self.assertEqual(ids[6:], [0] * 71)
        self.assertEqual((sum(mask), mask[5], mask[6]), (6, 1, 0))

    def test_encode_text_truncates_but_keeps_the_end_marker(self):
        ids, mask = semantic.encode_text(self.Tok(), " ".join(["w"] * 200))
        self.assertEqual((len(ids), ids[0], ids[-1], sum(mask)), (77, 49406, 49407, 77))

    def test_encode_text_falls_back_to_standard_marker_ids(self):
        tok = self.Tok()
        tok.token_to_id = lambda t: None
        ids, _ = semantic.encode_text(tok, "x")
        self.assertEqual((ids[0], ids[2]), (49406, 49407))


class IO:
    def __init__(self, name, type_="tensor(float)"):
        self.name, self.type = name, type_


class FakeSession:
    def __init__(self, inputs, outputs, fn):
        self._in, self._out, self._fn, self.feeds = inputs, outputs, fn, []

    def get_inputs(self): return [IO(n, t) for n, t in self._in]
    def get_outputs(self): return [IO(n) for n in self._out]

    def run(self, names, feed):
        self.feeds.append(feed)
        return [self._fn(names[0], feed)]


@needs_numpy
class ClipEmbedderPlumbingTests(unittest.TestCase):
    def make(self, vision, text):
        emb = semantic.ClipEmbedder("small")
        paths = {}

        def fake_session(path):
            return vision if "vision" in Path(path).name else text

        patches = [
            mock.patch.object(emb, "_session", side_effect=fake_session),
            mock.patch.object(emb, "_load_tokenizer", return_value=PreprocessAndTokenTests.Tok()),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return emb

    def vision_fn(self, name, feed):
        px = feed["pixel_values"]
        if name == "image_embeds":
            return np.full((1, 512), float(px.mean()) + 5.0, dtype=np.float32)
        return np.zeros((1, 50, 768), dtype=np.float32)

    def test_image_embedding_picks_the_embedding_output_and_is_unit_length(self):
        vision = FakeSession([("pixel_values", "tensor(float)")], ["last_hidden_state", "image_embeds"], self.vision_fn)
        emb = self.make(vision, None)
        v = emb.embed_image(Image.new("RGB", (80, 60), "gray"))
        self.assertEqual(v.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)
        self.assertEqual(vision.feeds[0]["pixel_values"].shape, (1, 3, 224, 224))

    def test_half_precision_model_gets_half_precision_input(self):
        vision = FakeSession([("pixel_values", "tensor(float16)")], ["image_embeds"], self.vision_fn)
        self.make(vision, None).embed_image(Image.new("RGB", (30, 30)))
        self.assertEqual(vision.feeds[0]["pixel_values"].dtype, np.float16)

    def test_text_embedding_feeds_ids_and_mask_by_name(self):
        text = FakeSession(
            [("input_ids", "tensor(int64)"), ("attention_mask", "tensor(int64)")], ["text_embeds"],
            lambda n, f: np.ones((1, 512), dtype=np.float32),
        )
        v = self.make(None, text).embed_text("a dog on grass")
        feed = text.feeds[0]
        self.assertEqual((feed["input_ids"].shape, feed["input_ids"].dtype), ((1, 77), np.int64))
        self.assertEqual(int(feed["attention_mask"].sum()), 6)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_text_model_without_attention_mask_input_still_works(self):
        text = FakeSession([("input_ids", "tensor(int64)")], ["text_embeds"], lambda n, f: np.ones((1, 8), np.float32))
        self.make(None, text).embed_text("x")
        self.assertEqual(list(text.feeds[0]), ["input_ids"])

    def test_unknown_text_input_is_a_clear_error(self):
        text = FakeSession([("mystery", "tensor(int64)")], ["text_embeds"], lambda n, f: None)
        with self.assertRaisesRegex(RuntimeError, "mystery"):
            self.make(None, text).embed_text("x")

    def test_wrong_output_shape_is_a_clear_error(self):
        vision = FakeSession([("pixel_values", "tensor(float)")], ["last_hidden_state"], self.vision_fn)
        with self.assertRaisesRegex(RuntimeError, "expected one embedding vector"):
            self.make(vision, None).embed_image(Image.new("RGB", (30, 30)))


class RangeHandler(BaseHTTPRequestHandler):
    files: dict = {}
    honour_range = True
    seen_ranges: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        data = self.files.get(self.path)
        if data is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        start = 0
        if rng and self.honour_range:
            self.seen_ranges.append(rng)
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_error(416)
                return
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])


class DownloadTests(Base):
    TINY = (
        semantic.ModelFile("vision_model.onnx", "onnx/vision_model.onnx", 2000),
        semantic.ModelFile("text_model.onnx", "onnx/text_model.onnx", 1500),
        semantic.ModelFile("tokenizer.json", "tokenizer.json", 100),
    )

    def setUp(self):
        super().setUp()
        RangeHandler.files = {
            "/onnx/vision_model.onnx": bytes(range(256)) * 10,
            "/onnx/text_model.onnx": bytes(range(255, -1, -1)) * 8,
            "/tokenizer.json": b"{" + b"x" * 200 + b"}",
        }
        RangeHandler.honour_range = True
        RangeHandler.seen_ranges = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        for p in (
            mock.patch.object(semantic, "BASE_URL", f"http://127.0.0.1:{server.server_port}/"),
            mock.patch.dict(semantic.VARIANTS, {"full": self.TINY}),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.dir = semantic.models_dir("full")

    def test_fresh_download_installs_everything_atomically(self):
        calls = []
        self.assertFalse(semantic.is_installed("full"))
        semantic.download_model("full", lambda *a: calls.append(a))
        self.assertTrue(semantic.is_installed("full"))
        self.assertEqual((self.dir / "vision_model.onnx").read_bytes(), RangeHandler.files["/onnx/vision_model.onnx"])
        self.assertEqual([p.name for p in self.dir.glob("*.part")], [])
        self.assertEqual(calls[-1][1:3], (2, 3))  # last file: index 2 of 3

    def test_interrupted_download_resumes_from_the_part_file(self):
        full = RangeHandler.files["/onnx/vision_model.onnx"]
        self.dir.mkdir(parents=True)
        (self.dir / "vision_model.onnx.part").write_bytes(full[:1000])
        semantic.download_model("full")
        self.assertEqual((self.dir / "vision_model.onnx").read_bytes(), full)
        self.assertEqual(RangeHandler.seen_ranges, ["bytes=1000-"])

    def test_server_that_ignores_range_restarts_cleanly(self):
        RangeHandler.honour_range = False
        full = RangeHandler.files["/onnx/vision_model.onnx"]
        self.dir.mkdir(parents=True)
        (self.dir / "vision_model.onnx.part").write_bytes(b"garbage" * 50)
        semantic.download_model("full")
        self.assertEqual((self.dir / "vision_model.onnx").read_bytes(), full)

    def test_already_complete_part_file_is_finalised(self):
        full = RangeHandler.files["/onnx/vision_model.onnx"]
        self.dir.mkdir(parents=True)
        (self.dir / "vision_model.onnx.part").write_bytes(full)  # complete, but never renamed
        semantic.download_model("full")
        self.assertEqual((self.dir / "vision_model.onnx").read_bytes(), full)

    def test_truncated_download_is_rejected_and_leaves_nothing_behind(self):
        RangeHandler.files["/onnx/vision_model.onnx"] = b"short"
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            semantic.download_model("full")
        self.assertFalse((self.dir / "vision_model.onnx").exists())
        self.assertFalse((self.dir / "vision_model.onnx.part").exists())
        self.assertFalse(semantic.is_installed("full"))

    def test_missing_file_on_the_server_is_an_error_not_an_install(self):
        del RangeHandler.files["/tokenizer.json"]
        with self.assertRaises(Exception):
            semantic.download_model("full")
        self.assertFalse(semantic.is_installed("full"))

    def test_background_downloader_reports_progress_and_errors(self):
        d = semantic.Downloader()
        self.assertTrue(d.start("full"))
        for _ in range(100):
            if not d.running:
                break
            time.sleep(0.05)
        st = d.status()
        self.assertEqual((st["running"], st["error"], st["files"]), (False, None, 3))
        self.assertTrue(semantic.is_installed("full"))
        RangeHandler.files.clear()
        (self.dir / "tokenizer.json").unlink()
        d.start("full")
        for _ in range(100):
            if not d.running:
                break
            time.sleep(0.05)
        self.assertIn("404", d.status()["error"] or "")

    def test_a_too_small_file_does_not_count_as_installed(self):
        semantic.download_model("full")
        (self.dir / "text_model.onnx").write_bytes(b"x")
        self.assertFalse(semantic.is_installed("full"))


@needs_numpy
class EmbeddingStoreTests(Base):
    def setUp(self):
        super().setUp()
        self.image("red.png", "red")
        self.image("reddish.png", "reddish")
        self.image("green.png", "green")
        self.image("blue.png", "blue")
        (self.root / "notes.txt").write_text("red notes")
        (self.root / "broken.png").write_bytes(b"not an image")
        (self.root / "phone.heic").write_bytes(b"heic bytes")
        self.scan()

    def test_only_images_are_embedded_and_bad_files_are_recorded_not_retried(self):
        self.assertEqual(self.embed(), {"embedded": 4, "failed": 2})
        self.assertEqual(self.embed(), {"embedded": 0, "failed": 0})  # nothing left, no retry loop
        with db.connect() as conn:
            c = semantic.counts(conn, "fake-1")
            errors = {r["name"]: r["error"] for r in conn.execute(
                "SELECT f.name, e.error FROM embeddings e JOIN files f ON f.id = e.file_id WHERE e.vector IS NULL")}
        self.assertEqual(c, {"images": 6, "embedded": 4, "failed": 2})
        self.assertIn("HEIC", errors["phone.heic"])
        self.assertTrue(errors["broken.png"])

    def test_query_ranks_matches_and_drops_the_rest(self):
        self.embed()
        self.assertEqual(self.hits("red"), [("red.png", 1.0), ("reddish.png", 0.94)])
        self.assertEqual(self.hits("blue"), [("blue.png", 1.0)])
        self.assertEqual(self.hits("purple unicorn"), [])  # nothing close enough: no results rather than noise

    def test_margin_trims_weaker_matches(self):
        self.embed()
        with mock.patch.object(semantic, "MARGIN", 0.01):
            self.assertEqual(self.hits("red"), [("red.png", 1.0)])

    def test_edited_image_is_embedded_again(self):
        self.embed()
        p = self.root / "red.png"
        Image.new("RGB", (64, 48), COLORS["blue"]).save(p)
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 500))
        self.assertEqual(self.scan()["updated"], 1)
        with db.connect() as conn:
            self.assertEqual(semantic.counts(conn, "fake-1")["embedded"], 3)  # old vector dropped at once
        self.embed()
        self.assertEqual({n for n, _ in self.hits("blue")}, {"blue.png", "red.png"})

    def test_deleted_image_disappears_with_its_vector(self):
        self.embed()
        (self.root / "green.png").unlink()
        self.scan()
        self.assertNotIn("green.png", [n for n, _ in self.hits("green")])
        with db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM embeddings WHERE vector IS NOT NULL").fetchone()[0], 3)

    def test_stale_vector_is_ignored_even_if_it_was_stored_late(self):
        self.embed()
        with db.connect() as conn:  # the file changed after its vector was computed
            conn.execute("UPDATE files SET mtime_ns = mtime_ns + 1 WHERE name = 'red.png'")
            conn.commit()
        self.assertEqual([n for n, _ in self.hits("red")], ["reddish.png"])

    def test_switching_model_recomputes_everything(self):
        self.embed()
        self.assertEqual(self.embed(FakeEmbedder("fake-2")), {"embedded": 4, "failed": 2})
        self.assertEqual(self.hits("red", FakeEmbedder("fake-2"))[0][0], "red.png")

    def test_retry_failed_requeues_only_failures(self):
        self.embed()
        with db.connect() as conn:
            self.assertEqual(semantic.retry_failed(conn), 2)
        self.assertEqual(self.embed(), {"embedded": 0, "failed": 2})

    def test_new_images_are_picked_up_by_the_cached_matrix(self):
        self.embed()
        self.assertEqual(self.hits("green"), [("green.png", 1.0)])
        self.image("another_green.png", "green")
        self.scan()
        self.embed()
        self.assertEqual({n for n, _ in self.hits("green")}, {"green.png", "another_green.png"})

    def test_worker_waits_while_text_indexing_is_busy_then_runs(self):
        gate = threading.Event()
        semantic.set_embedder(FakeEmbedder())
        w = semantic.SemanticWorker(busy=lambda: not gate.is_set())
        self.assertTrue(w.start())
        time.sleep(0.4)
        with db.connect() as conn:
            self.assertEqual(semantic.counts(conn, "fake-1")["embedded"], 0)  # still waiting its turn
        gate.set()
        for _ in range(100):
            if not w.running:
                break
            time.sleep(0.05)
        st = w.status()
        self.assertEqual((st["running"], st["embedded"], st["failed"], st["images"]), (False, 4, 2, 6))

    def test_pause_stops_the_worker_without_losing_work(self):
        semantic.set_embedder(FakeEmbedder())
        w = semantic.SemanticWorker()
        stop = threading.Event()
        stop.set()
        self.assertEqual(semantic.embed_pending(stop, embedder=FakeEmbedder()), {"embedded": 0, "failed": 0})
        self.assertTrue(w.stop_and_wait(1))

    def test_worker_refuses_to_start_when_the_model_is_not_ready(self):
        w = semantic.SemanticWorker()
        with mock.patch.object(semantic, "ready", return_value=False):
            self.assertFalse(w.start())


@needs_numpy
class HybridSearchTests(Base):
    def setUp(self):
        super().setUp()
        self.image("red_car.png", "red", "2024-03-01")     # keyword (name) AND meaning
        self.image("fire.png", "red", "2024-06-01")        # meaning only: the name says nothing about colour
        self.image("sky.png", "blue", "2024-04-01")        # unrelated
        (self.root / "red_notes.txt").write_text("my red balloon notes")  # keyword only (a document)
        ts = time.mktime(time.strptime("2024-05-01", "%Y-%m-%d")) + 43200
        os.utime(self.root / "red_notes.txt", (ts, ts))
        self.scan()
        from app.worker import _store
        from app import extractors

        with db.connect() as conn:
            row = conn.execute("SELECT id, mtime_ns FROM files WHERE name = 'red_notes.txt'").fetchone()
            _store(conn, row, "done", extractors.Extraction("my red balloon notes", "text"), None, None)
        self.embed()

    def search(self, q="red", mode="auto", **kw):
        with db.connect() as conn:
            found = semantic.query_hits(conn, q, FakeEmbedder())
            return run_hybrid_search(conn, q, found, mode, **kw)

    def names(self, res):
        return [r["name"] for r in res["results"]]

    def test_files_found_both_ways_rank_first_and_sources_are_labelled(self):
        res = self.search()
        by_name = {r["name"]: r for r in res["results"]}
        self.assertEqual(res["results"][0]["name"], "red_car.png")
        self.assertEqual({n: r["source"] for n, r in by_name.items()},
                         {"red_car.png": "both", "fire.png": "meaning", "red_notes.txt": "keyword"})
        self.assertEqual((by_name["fire.png"]["similarity"], by_name["red_notes.txt"]["similarity"]), (1.0, None))
        self.assertNotIn("sky.png", by_name)
        self.assertEqual(res["total"], 3)

    def test_keyword_hits_are_highlighted_and_meaning_hits_are_not(self):
        by_name = {r["name"]: r for r in self.search()["results"]}
        self.assertIn(MARK_START, by_name["red_car.png"]["name_hl"])
        self.assertIn(MARK_START, by_name["red_notes.txt"]["snippet"])
        self.assertEqual((by_name["fire.png"]["name_hl"], by_name["fire.png"]["snippet"]), ("fire.png", ""))

    def test_meaning_only_mode_excludes_keyword_only_files(self):
        self.assertEqual(set(self.names(self.search(mode="semantic"))), {"red_car.png", "fire.png"})

    def test_filters_apply_to_both_kinds_of_match(self):
        self.assertEqual(self.names(self.search(type="document")), ["red_notes.txt"])
        self.assertEqual(set(self.names(self.search(type="image"))), {"red_car.png", "fire.png"})
        self.assertEqual(set(self.names(self.search(date_from="2024-05-15"))), {"fire.png"})
        self.assertEqual(set(self.names(self.search(date_to="2024-03-31"))), {"red_car.png"})

    def test_date_sorting_overrides_relevance(self):
        self.assertEqual(self.names(self.search(sort="newest")), ["fire.png", "red_notes.txt", "red_car.png"])
        self.assertEqual(self.names(self.search(sort="oldest")), ["red_car.png", "red_notes.txt", "fire.png"])

    def test_pagination(self):
        first, second = self.search(limit=2), self.search(limit=2, offset=2)
        self.assertEqual((len(first["results"]), len(second["results"]), first["total"]), (2, 1, 3))
        self.assertFalse(set(self.names(first)) & set(self.names(second)))

    def test_no_meaning_matches_in_semantic_mode_gives_an_empty_result(self):
        with db.connect() as conn:
            res = run_hybrid_search(conn, "purple", [], "semantic")
        self.assertEqual((res["total"], res["results"]), (0, []))

    def test_invalid_dates_still_raise(self):
        with self.assertRaises(ValueError):
            self.search(date_from="2024-02-31")

    def test_plain_keyword_search_is_unchanged_without_hits(self):
        with db.connect() as conn:
            self.assertEqual({r["name"] for r in run_search(conn, "red")["results"]}, {"red_car.png", "red_notes.txt"})


@unittest.skipUnless(HAVE_NUMPY and importlib.util.find_spec("fastapi"), "needs numpy and fastapi")
class ApiSearchTests(Base):
    """Calls the real endpoint function, with every argument explicit (as FastAPI would pass them)."""

    def setUp(self):
        super().setUp()
        from app import main

        self.main = main
        self.image("red_car.png", "red")
        (self.root / "red_notes.txt").write_text("red notes")
        self.scan()
        semantic.set_embedder(FakeEmbedder())
        self.embed()

    def call(self, q="red", mode="auto"):
        return self.main.api_search(q=q, type=None, date_from=None, date_to=None, sort="relevance",
                                    limit=20, offset=0, status=None, mode=mode)

    def test_meaning_and_keyword_results_are_combined(self):
        res = self.call()
        self.assertEqual({r["name"]: r["source"] for r in res["results"]},
                         {"red_car.png": "both", "red_notes.txt": "keyword"})
        self.assertIsNone(res["semantic_error"])

    def test_a_broken_model_never_breaks_normal_search(self):
        class Broken(FakeEmbedder):
            def embed_text(self, text):
                raise RuntimeError("model file corrupt")

        semantic.set_embedder(Broken())
        res = self.call()
        self.assertEqual({r["name"] for r in res["results"]}, {"red_car.png", "red_notes.txt"})
        self.assertIn("RuntimeError: model file corrupt", res["semantic_error"])
        with self.assertRaises(self.main.HTTPException) as ctx:
            self.call(mode="semantic")
        self.assertEqual(ctx.exception.status_code, 503)

    def test_meaning_only_mode_without_a_model_explains_what_to_do(self):
        semantic.set_embedder(None)
        with mock.patch.object(semantic, "ready", return_value=False):
            with self.assertRaises(self.main.HTTPException) as ctx:
                self.call(mode="semantic")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("download the model", ctx.exception.detail)


class DependencyTests(unittest.TestCase):
    def test_semantic_module_imports_without_numpy_and_reports_missing_packages(self):
        self.assertIsInstance(semantic.missing_packages(), list)
        self.assertFalse(semantic.ready() and semantic.missing_packages())

    def test_model_pinned_to_a_fixed_revision_over_https(self):
        self.assertRegex(semantic.REVISION, r"^[0-9a-f]{40}$")
        self.assertTrue(semantic.BASE_URL.startswith("https://huggingface.co/"))
        self.assertIn(semantic.REVISION, semantic.BASE_URL)


if __name__ == "__main__":
    unittest.main()
