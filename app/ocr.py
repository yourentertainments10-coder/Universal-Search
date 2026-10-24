"""OCR engines, loaded lazily. Order: RapidOCR (pip only), then Tesseract.

Set OCR_ENGINE=rapidocr or OCR_ENGINE=tesseract to force one.
Set TESSERACT_CMD to the tesseract.exe path if it is not on PATH.
Set OCR_LANG (default "eng") to change the Tesseract language.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Callable

log = logging.getLogger("universal_search")

WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

HINT = (
    "No OCR engine found. Images and scanned PDFs are waiting until you install one: "
    "run pip install -r requirements-ocr.txt, or install Tesseract OCR."
)


class Engine:
    def __init__(self, name: str, read: Callable):
        self.name = name
        self._read = read

    def read(self, img) -> str:
        """`img` is an RGB PIL image. Returns the recognised text."""
        return self._read(img) or ""


def _load_rapidocr() -> Engine | None:
    try:
        import numpy as np

        try:
            from rapidocr_onnxruntime import RapidOCR  # classic package

            engine = RapidOCR()

            def read(img):
                result, _ = engine(np.array(img))
                return "\n".join(item[1] for item in result) if result else ""

        except ImportError:
            from rapidocr import RapidOCR  # newer package

            engine = RapidOCR()

            def read(img):
                out = engine(np.array(img))
                txts = getattr(out, "txts", None)
                return "\n".join(txts) if txts else ""

        return Engine("rapidocr", read)
    except ImportError:
        return None
    except Exception:
        log.exception("RapidOCR is installed but failed to start")
        return None


def _load_tesseract() -> Engine | None:
    try:
        import pytesseract
    except ImportError:
        return None
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd:
        cmd = next((p for p in WINDOWS_TESSERACT_PATHS if os.path.isfile(p)), None)
    if not cmd:
        return None
    pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        log.exception("Tesseract found at %s but could not be run", cmd)
        return None
    lang = os.environ.get("OCR_LANG", "eng")
    return Engine("tesseract", lambda img: pytesseract.image_to_string(img, lang=lang, timeout=180))


_LOADERS = {"rapidocr": _load_rapidocr, "tesseract": _load_tesseract}
_lock = threading.Lock()
_loaded = False
_engine: Engine | None = None


def get_engine() -> Engine | None:
    """Return the OCR engine, or None if none is installed. Loaded once."""
    global _loaded, _engine
    with _lock:
        if not _loaded:
            forced = os.environ.get("OCR_ENGINE", "").strip().lower()
            order = [forced] if forced in _LOADERS else list(_LOADERS)
            for name in order:
                _engine = _LOADERS[name]()
                if _engine:
                    break
            _loaded = True
        return _engine


def reset() -> None:
    """Forget the loaded engine (used by tests)."""
    global _loaded, _engine
    with _lock:
        _loaded, _engine = False, None


def status() -> dict:
    engine = get_engine()
    return {"available": engine is not None, "engine": engine.name if engine else None, "hint": None if engine else HINT}
