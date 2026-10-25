"""Semantic image search (Milestone 6): CLIP embeddings computed locally with ONNX Runtime.

Everything runs on this computer. The only network use is the one-time, user-started
download of the model files (about 606 MB, or 154 MB for CLIP_VARIANT=small) from
Hugging Face, pinned to a fixed revision. After that no network is needed.

Command line:
    python -m app.semantic download [--variant full|small]
    python -m app.semantic check [--folder PATH] [--query "a dog on a beach" ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import db

log = logging.getLogger("universal_search")

# ---- model files (verified on the model page; MIT licence, converted from OpenAI CLIP ViT-B/32) ----
REVISION = "b8b103f67eef16b29539311ff78dde6c4a754370"
BASE_URL = os.environ.get(
    "CLIP_BASE_URL", f"https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/{REVISION}/"
)


@dataclass(frozen=True)
class ModelFile:
    name: str        # file name on disk
    remote: str      # path under BASE_URL
    min_bytes: int   # anything smaller is a broken download


_TOKENIZER = ModelFile("tokenizer.json", "tokenizer.json", 1_500_000)
VARIANTS: dict[str, tuple[ModelFile, ModelFile, ModelFile]] = {
    "full": (
        ModelFile("vision_model.onnx", "onnx/vision_model.onnx", 300_000_000),
        ModelFile("text_model.onnx", "onnx/text_model.onnx", 220_000_000),
        _TOKENIZER,
    ),
    "small": (  # 8-bit quantized: about 4x smaller and faster, slightly less accurate
        ModelFile("vision_model_quantized.onnx", "onnx/vision_model_quantized.onnx", 75_000_000),
        ModelFile("text_model_quantized.onnx", "onnx/text_model_quantized.onnx", 55_000_000),
        _TOKENIZER,
    ),
}
APPROX_MB = {"full": 606, "small": 154}

# ---- search behaviour (starting values: tune with `python -m app.semantic check`) ----
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


MIN_SIMILARITY = _env_float("SEMANTIC_MIN_SIMILARITY", 0.20)  # CLIP text-image cosine floor
MARGIN = _env_float("SEMANTIC_MARGIN", 0.07)                   # keep hits within this of the best one
TOP_K = int(_env_float("SEMANTIC_TOP_K", 60))
CONTEXT = 77                                                   # CLIP text length
BOS_DEFAULT, EOS_DEFAULT = 49406, 49407
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def current_variant() -> str:
    v = os.environ.get("CLIP_VARIANT", "full").strip().lower()
    return v if v in VARIANTS else "full"


def model_id(variant: str) -> str:
    return f"clip-vit-b32-{variant}-{REVISION[:7]}"


def models_dir(variant: str) -> Path:
    return db.db_path().parent / "models" / f"clip-vit-b32-{variant}"


def is_installed(variant: str | None = None) -> bool:
    variant = variant or current_variant()
    d = models_dir(variant)
    return all((d / f.name).is_file() and (d / f.name).stat().st_size >= f.min_bytes for f in VARIANTS[variant])


def missing_packages() -> list[str]:
    return [m for m in ("numpy", "onnxruntime", "tokenizers") if importlib.util.find_spec(m) is None]


def _np():
    import numpy

    return numpy


# ====================================================================================
# Download (resumable, atomic)
# ====================================================================================

def _fetch(url: str, dest: Path, min_bytes: int, progress=None, timeout: float = 30.0) -> None:
    """Download `url` to `dest`. An interrupted download resumes from the .part file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "Universal-Search/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416 and existing >= min_bytes:  # the .part file is already the whole file
            os.replace(part, dest)
            return
        raise
    with resp:
        resumed = resp.status == 206
        length = int(resp.headers.get("Content-Length") or 0)
        done = existing if resumed else 0
        total = done + length
        with open(part, "ab" if resumed else "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if part.stat().st_size < min_bytes:
        size = part.stat().st_size
        part.unlink()
        raise RuntimeError(f"Download of {dest.name} was incomplete ({size} bytes). Please try again.")
    os.replace(part, dest)  # atomic: a half-downloaded model is never mistaken for a good one


def download_model(variant: str | None = None, progress=None) -> None:
    variant = variant or current_variant()
    files = VARIANTS[variant]
    for i, f in enumerate(files):
        target = models_dir(variant) / f.name
        if target.is_file() and target.stat().st_size >= f.min_bytes:
            continue
        log.info("Downloading %s", f.name)
        _fetch(BASE_URL + f.remote, target, f.min_bytes,
               (lambda d, t, name=f.name, idx=i: progress(name, idx, len(files), d, t)) if progress else None)


class Downloader:
    """Runs download_model on a background thread and reports progress."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state = {"file": None, "index": 0, "files": 0, "done": 0, "total": 0, "error": None}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, variant: str | None = None) -> bool:
        with self._lock:
            if self.running:
                return False
            self.state = {"file": None, "index": 0, "files": 0, "done": 0, "total": 0, "error": None}
            self._thread = threading.Thread(target=self._run, args=(variant,), name="model-download", daemon=True)
            self._thread.start()
            return True

    def _progress(self, name, index, files, done, total) -> None:
        self.state.update(file=name, index=index, files=files, done=done, total=total)

    def _run(self, variant) -> None:
        try:
            download_model(variant, self._progress)
        except Exception as e:  # network errors, disk full, ...
            self.state["error"] = f"{type(e).__name__}: {e}"
            log.warning("Model download failed: %s", self.state["error"])

    def status(self) -> dict:
        return {"running": self.running, **self.state}


downloader = Downloader()


# ====================================================================================
# The model
# ====================================================================================

def preprocess_image(img):
    """CLIP preprocessing: shortest side to 224 (bicubic), centre crop, normalise. Returns [1,3,224,224] float32."""
    np = _np()
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    scale = 224 / min(w, h)
    nw, nh = max(224, round(w * scale)), max(224, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - 224) // 2, (nh - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


def _token_id(tokenizer, token: str, default: int) -> int:
    try:
        value = tokenizer.token_to_id(token)
    except Exception:
        value = None
    return default if value is None else value


def encode_text(tokenizer, text: str) -> tuple[list[int], list[int]]:
    """Token ids and attention mask for CLIP: [start] text [end], zero padded to 77."""
    try:
        tokenizer.no_padding()
        tokenizer.no_truncation()
    except Exception:
        pass
    body = list(tokenizer.encode(text, add_special_tokens=False).ids)[: CONTEXT - 2]
    bos = _token_id(tokenizer, "<|startoftext|>", BOS_DEFAULT)
    eos = _token_id(tokenizer, "<|endoftext|>", EOS_DEFAULT)
    ids = [bos] + body + [eos]
    mask = [1] * len(ids) + [0] * (CONTEXT - len(ids))
    return ids + [0] * (CONTEXT - len(ids)), mask


def _normalise(vec):
    np = _np()
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class ClipEmbedder:
    """Turns images and text into vectors that can be compared (cosine similarity)."""

    def __init__(self, variant: str | None = None) -> None:
        self.variant = variant or current_variant()
        self.model_id = model_id(self.variant)
        self._lock = threading.Lock()
        self._vision = None
        self._text = None
        self._tokenizer = None

    # -- loading (each part is loaded only when first needed) --
    def _session(self, path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        threads = os.environ.get("CLIP_THREADS")
        if threads and threads.isdigit():
            opts.intra_op_num_threads = int(threads)
        return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])

    def _load_tokenizer(self):
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(models_dir(self.variant) / "tokenizer.json"))

    def _vision_session(self):
        with self._lock:
            if self._vision is None:
                self._vision = self._session(models_dir(self.variant) / VARIANTS[self.variant][0].name)
            return self._vision

    def _text_session(self):
        with self._lock:
            if self._text is None:
                self._text = self._session(models_dir(self.variant) / VARIANTS[self.variant][1].name)
            if self._tokenizer is None:
                self._tokenizer = self._load_tokenizer()
            return self._text, self._tokenizer

    # -- running --
    @staticmethod
    def _run(session, feed: dict, preferred: tuple[str, ...]):
        np = _np()
        names = [o.name for o in session.get_outputs()]
        name = next((p for p in preferred if p in names), names[0])
        arr = np.asarray(session.run([name], feed)[0], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise RuntimeError(
                f"Unexpected model output {name!r} with shape {tuple(arr.shape)}; expected one embedding vector."
            )
        return _normalise(arr)

    def embed_image(self, img):
        """`img` is an RGB PIL image. Returns a unit-length float32 vector."""
        session = self._vision_session()
        arr = preprocess_image(img)
        inp = session.get_inputs()[0]
        if "float16" in str(getattr(inp, "type", "")):
            arr = arr.astype(_np().float16)
        return self._run(session, {inp.name: arr}, ("image_embeds", "pooler_output"))

    def embed_text(self, text: str):
        np = _np()
        session, tokenizer = self._text_session()
        ids, mask = encode_text(tokenizer, text)
        feed = {}
        for inp in session.get_inputs():
            lname = inp.name.lower()
            if "attention" in lname:
                feed[inp.name] = np.array([mask], dtype=np.int64)
            elif "input_ids" in lname or lname == "ids":
                feed[inp.name] = np.array([ids], dtype=np.int64)
            else:
                raise RuntimeError(f"Unexpected text model input {inp.name!r}")
        return self._run(session, feed, ("text_embeds", "pooler_output"))


_override = None          # tests (and advanced users) can plug in another embedder
_embedder: ClipEmbedder | None = None
_embedder_lock = threading.Lock()


def set_embedder(embedder) -> None:
    global _override
    _override = embedder


def get_embedder():
    global _embedder
    if _override is not None:
        return _override
    with _embedder_lock:
        if _embedder is None or _embedder.variant != current_variant():
            _embedder = ClipEmbedder()
        return _embedder


def ready() -> bool:
    """True when searching by meaning can work right now."""
    return _override is not None or (is_installed() and not missing_packages())


# ====================================================================================
# Storing and querying embeddings
# ====================================================================================

_VALID = "e.model = ? AND e.mtime_ns = f.mtime_ns AND e.vector IS NOT NULL"  # ignores stale or failed rows


def store_embedding(conn, file_id: int, model: str, mtime_ns: int, vec=None, error: str | None = None) -> None:
    np = _np()
    blob = None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()
    dim = 0 if vec is None else int(np.asarray(vec).size)
    with conn:
        conn.execute(
            """INSERT INTO embeddings (file_id, model, mtime_ns, dim, vector, error, created_ns)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(file_id) DO UPDATE SET model=excluded.model, mtime_ns=excluded.mtime_ns,
                 dim=excluded.dim, vector=excluded.vector, error=excluded.error, created_ns=excluded.created_ns""",
            (file_id, model, mtime_ns, dim, blob, error, time.time_ns()),
        )


def counts(conn, model: str) -> dict:
    images = conn.execute("SELECT COUNT(*) FROM files WHERE type = 'image'").fetchone()[0]
    base = "SELECT COUNT(*) FROM embeddings e JOIN files f ON f.id = e.file_id WHERE e.model = ? AND e.mtime_ns = f.mtime_ns"
    embedded = conn.execute(base + " AND e.vector IS NOT NULL", (model,)).fetchone()[0]
    failed = conn.execute(base + " AND e.vector IS NULL", (model,)).fetchone()[0]
    return {"images": images, "embedded": embedded, "failed": failed}


_cache: dict = {"key": None, "ids": None, "mat": None}
_cache_lock = threading.Lock()


def _matrix(conn, model: str):
    """All valid image vectors as one matrix, rebuilt only when the stored embeddings change."""
    np = _np()
    n, newest = conn.execute(
        f"SELECT COUNT(*), COALESCE(MAX(e.created_ns), 0) FROM embeddings e JOIN files f ON f.id = e.file_id WHERE {_VALID}",
        (model,),
    ).fetchone()
    key = (model, n, newest)
    with _cache_lock:
        if _cache["key"] == key:
            return _cache["ids"], _cache["mat"]
    rows = conn.execute(
        f"SELECT e.file_id, e.vector FROM embeddings e JOIN files f ON f.id = e.file_id WHERE {_VALID} ORDER BY e.file_id",
        (model,),
    ).fetchall()
    if rows:
        ids = np.array([r["file_id"] for r in rows], dtype=np.int64)
        mat = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    else:
        ids, mat = np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    with _cache_lock:
        _cache.update(key=key, ids=ids, mat=mat)
    return ids, mat


def query_hits(conn, text: str, embedder=None, limit: int | None = None) -> list[tuple[int, float]]:
    """Images that look like `text`: [(file_id, similarity)], best first. Empty if nothing is close enough."""
    np = _np()
    embedder = embedder or get_embedder()
    ids, mat = _matrix(conn, embedder.model_id)
    if len(ids) == 0:
        return []
    q = embedder.embed_text(text)
    if mat.shape[1] != q.size:
        return []  # vectors from a different model; they will be recomputed
    scores = mat @ q
    order = np.argsort(-scores)[: limit or TOP_K]
    top = float(scores[order[0]])
    if top < MIN_SIMILARITY:
        return []
    floor = max(MIN_SIMILARITY, top - MARGIN)
    return [(int(ids[i]), float(scores[i])) for i in order if scores[i] >= floor]


# ====================================================================================
# Background embedding of images
# ====================================================================================

BATCH = 16


def embed_one(conn, embedder, row) -> bool:
    """Embed one image row. Always records the outcome, so a bad file is not retried in a loop."""
    from PIL import Image

    from .extractors import _prepare_image

    try:
        if row["ext"].lower() == ".heic":
            raise ValueError("HEIC images are not supported yet")
        with Image.open(row["path"]) as img:
            img.draft("RGB", (448, 448))  # fast low-resolution JPEG decode
            img = _prepare_image(img)
        store_embedding(conn, row["id"], embedder.model_id, row["mtime_ns"], embedder.embed_image(img))
        return True
    except Exception as e:
        store_embedding(conn, row["id"], embedder.model_id, row["mtime_ns"], None, f"{type(e).__name__}: {e}"[:300])
        return False


def embed_pending(stop: threading.Event | None = None, on_file=None, embedder=None, busy=None) -> dict:
    """Embed every image that has no valid embedding yet. `busy()` True makes it wait (OCR runs first)."""
    embedder = embedder or get_embedder()
    result = {"embedded": 0, "failed": 0}
    with db.connect() as conn:
        while not (stop and stop.is_set()):
            if busy and busy():
                if stop:
                    stop.wait(1.0)
                else:
                    time.sleep(1.0)
                continue
            rows = conn.execute(
                """SELECT f.id, f.path, f.ext, f.mtime_ns FROM files f
                   LEFT JOIN embeddings e ON e.file_id = f.id AND e.model = ? AND e.mtime_ns = f.mtime_ns
                   WHERE f.type = 'image' AND e.file_id IS NULL ORDER BY f.id LIMIT ?""",
                (embedder.model_id, BATCH),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                if stop and stop.is_set():
                    break
                if on_file:
                    on_file(os.path.basename(row["path"]))
                result["embedded" if embed_one(conn, embedder, row) else "failed"] += 1
    return result


def retry_failed(conn) -> int:
    with conn:
        cur = conn.execute("DELETE FROM embeddings WHERE vector IS NULL")
    return cur.rowcount


class SemanticWorker:
    def __init__(self, busy=None) -> None:
        self._busy = busy
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._current: str | None = None
        self._last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self.running or not ready():
                return False
            self._stop.clear()
            self._last_error = None
            self._thread = threading.Thread(target=self._run, name="semantic-worker", daemon=True)
            self._thread.start()
            return True

    def pause(self) -> None:
        self._stop.set()

    def stop_and_wait(self, timeout: float = 60.0) -> bool:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)
        return not self.running

    def _run(self) -> None:
        try:
            embed_pending(self._stop, lambda n: setattr(self, "_current", n), busy=self._busy)
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            log.exception("Semantic worker stopped by an error")
        finally:
            self._current = None

    def status(self) -> dict:
        variant = current_variant()
        model = _override.model_id if _override is not None else model_id(variant)
        with db.connect() as conn:
            c = counts(conn, model)
        return {
            "variant": variant,
            "approx_mb": APPROX_MB[variant],
            "installed": is_installed(variant),
            "missing_packages": missing_packages(),
            "ready": ready(),
            "download": downloader.status(),
            "running": self.running,
            "pausing": self.running and self._stop.is_set(),
            "current": self._current,
            "last_error": self._last_error,
            **c,
        }


# ====================================================================================
# Command line: download the model, and measure it on this computer
# ====================================================================================

def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t) for n in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
                ]

            kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            c = Counters()
            c.cb = ctypes.sizeof(c)
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return c.WorkingSetSize / 1e6
        except Exception:
            return None
        return None
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return None


def _cli_download(args) -> int:
    variant = args.variant or current_variant()
    print(f"Downloading CLIP ViT-B/32 ({variant}, about {APPROX_MB[variant]} MB) to {models_dir(variant)}")

    def show(name, idx, n, done, total):
        pct = f"{done * 100 // total}%" if total else ""
        print(f"\r  [{idx + 1}/{n}] {name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB {pct}   ", end="", flush=True)

    download_model(variant, show)
    print("\nDone. Next: python -m app.semantic check")
    return 0


def _cli_check(args) -> int:
    from PIL import Image, ImageDraw

    from .extractors import _prepare_image

    missing = missing_packages()
    if missing:
        print("Missing packages:", ", ".join(missing), "\nRun: pip install -r requirements-semantic.txt")
        return 1
    variant = current_variant()
    if not is_installed(variant):
        print(f"The {variant} model is not downloaded yet. Run: python -m app.semantic download")
        return 1
    emb = ClipEmbedder(variant)
    t0 = time.time()
    emb.embed_text("warm up")
    emb.embed_image(Image.new("RGB", (64, 64)))
    print(f"Model: {emb.model_id}   load time {time.time() - t0:.1f}s   memory {_rss_mb() or float('nan'):.0f} MB")

    checker = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(checker)
    for r in range(8):
        for c in range(8):
            if (r + c) % 2:
                d.rectangle([c * 32, r * 32, c * 32 + 31, r * 32 + 31], fill="black")
    tests = [
        ("a solid red image", Image.new("RGB", (256, 256), (220, 30, 30))),
        ("a solid blue image", Image.new("RGB", (256, 256), (30, 60, 220))),
        ("a solid green image", Image.new("RGB", (256, 256), (30, 180, 60))),
        ("a black and white checkerboard", checker),
    ]
    vecs = [emb.embed_image(img) for _, img in tests]
    texts = [emb.embed_text(t) for t, _ in tests]
    print("\nSanity check (each text should match its own image best; scores are cosine similarity):")
    ok = 0
    for i, (label, _) in enumerate(tests):
        scores = [float(texts[i] @ v) for v in vecs]
        best = max(range(len(scores)), key=scores.__getitem__)
        ok += best == i
        print(f"  {label:34} " + "  ".join(f"{s:.2f}" for s in scores) + ("   OK" if best == i else "   WRONG"))
    print(f"  -> {ok}/{len(tests)} correct. Typical good matches score about 0.25-0.35 (MIN_SIMILARITY is {MIN_SIMILARITY}).")

    if args.folder:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [p for p in sorted(Path(args.folder).rglob("*")) if p.suffix.lower() in exts][: args.limit]
        if not files:
            print(f"\nNo images found in {args.folder}")
            return 0
        print(f"\nEmbedding {len(files)} images from {args.folder} ...")
        vecs, names, t0 = [], [], time.time()
        for p in files:
            try:
                with Image.open(p) as img:
                    img.draft("RGB", (448, 448))
                    vecs.append(emb.embed_image(_prepare_image(img)))
                names.append(p.name)
            except Exception as e:
                print(f"  skipped {p.name}: {e}")
        dt = time.time() - t0
        per = dt / max(len(vecs), 1)
        print(f"  {per * 1000:.0f} ms per image  ->  about {per * 1000 / 60:.1f} minutes per 1000 images   memory {_rss_mb() or float('nan'):.0f} MB")
        np = _np()
        mat = np.vstack(vecs)
        for q in args.query or ["a photo of people", "a screenshot", "a landscape"]:
            scores = mat @ emb.embed_text(q)
            top = np.argsort(-scores)[:5]
            print(f'\n  "{q}"')
            for i in top:
                mark = "*" if scores[i] >= MIN_SIMILARITY else " "
                print(f"   {mark} {scores[i]:.3f}  {names[i]}")
        print("\n  (* = above MIN_SIMILARITY and would be shown). Adjust SEMANTIC_MIN_SIMILARITY / SEMANTIC_MARGIN if needed.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.semantic", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download the model files (one time)")
    d.add_argument("--variant", choices=sorted(VARIANTS))
    d.set_defaults(fn=_cli_download)
    c = sub.add_parser("check", help="sanity-check accuracy, speed and memory on this computer")
    c.add_argument("--folder", help="also embed up to --limit images from this folder and test queries on them")
    c.add_argument("--limit", type=int, default=40)
    c.add_argument("--query", action="append", help="a text query to try (repeatable)")
    c.set_defaults(fn=_cli_check)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
