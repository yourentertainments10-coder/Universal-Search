"""Text extraction for each supported file type. Never modifies the original file."""
from __future__ import annotations

import codecs
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass

from PIL import Image, ImageOps

MAX_TEXT_CHARS = 2_000_000       # cap on stored text per file
MAX_TEXT_FILE_BYTES = 20_000_000  # largest .txt/.md read
MAX_PDF_PAGES = 500
MAX_OCR_PAGES = 50               # scanned PDF pages OCR'd per file
MIN_PAGE_TEXT = 20               # fewer characters than this = page has no usable text layer
OCR_RENDER_SCALE = 3.0           # PDF render scale for OCR (72 dpi x 3)
MAX_OCR_SIDE = 4000              # larger images are scaled down before OCR

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class Skipped(Exception):
    """File type we deliberately do not process (not an error)."""


@dataclass
class Extraction:
    text: str
    method: str                  # text | docx | pdf-text | pdf-ocr | pdf-mixed | ocr
    engine: str | None = None
    pages: int | None = None
    note: str | None = None
    needs_ocr: bool = False      # OCR was required but no engine is installed


def _clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return text[:MAX_TEXT_CHARS]


def read_text_file(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read(MAX_TEXT_FILE_BYTES + 1)
    if len(data) > MAX_TEXT_FILE_BYTES:
        data = data[: max(data.rfind(b"\n", 0, MAX_TEXT_FILE_BYTES), 0) or MAX_TEXT_FILE_BYTES]
    if data.startswith(codecs.BOM_UTF8):
        return data[3:].decode("utf-8", "replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", "replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", "replace")


def extract_docx(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        info = z.getinfo("word/document.xml")
        if info.file_size > 100_000_000:
            raise ValueError("document.xml is unreasonably large")
        root = ET.fromstring(z.read(info))
    lines = []
    for p in root.iter(W + "p"):
        buf = []
        for el in p.iter():
            if el.tag == W + "t" and el.text:
                buf.append(el.text)
            elif el.tag == W + "tab":
                buf.append("\t")
            elif el.tag in (W + "br", W + "cr"):
                buf.append("\n")
        line = "".join(buf)
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def _prepare_image(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > MAX_OCR_SIDE:
        img.thumbnail((MAX_OCR_SIDE, MAX_OCR_SIDE))
    return img


def extract_image(path: str, engine) -> Extraction:
    if engine is None:
        return Extraction("", "ocr", needs_ocr=True)
    with Image.open(path) as img:
        text = engine.read(_prepare_image(img))
    return Extraction(_clean(text), "ocr", engine=engine.name)


def extract_pdf(path: str, engine) -> Extraction:
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise RuntimeError(
            "pypdfium2 is not installed. Run: pip install -r requirements.txt (inside the .venv)"
        ) from e

    pdf = pdfium.PdfDocument(path)
    try:
        total = len(pdf)
        parts: list[str] = []
        text_pages = ocr_pages = missing_pages = 0
        note = None
        for i in range(min(total, MAX_PDF_PAGES)):
            page = pdf[i]
            try:
                textpage = page.get_textpage()
                try:
                    txt = textpage.get_text_range() or ""
                finally:
                    textpage.close()
                if len(txt.strip()) >= MIN_PAGE_TEXT:
                    parts.append(txt)
                    text_pages += 1
                elif engine is None:
                    missing_pages += 1
                elif ocr_pages >= MAX_OCR_PAGES:
                    note = f"OCR limited to the first {MAX_OCR_PAGES} scanned pages"
                else:
                    img = page.render(scale=OCR_RENDER_SCALE).to_pil()
                    parts.append(engine.read(_prepare_image(img)))
                    ocr_pages += 1
            finally:
                page.close()
        if total > MAX_PDF_PAGES:
            note = f"Only the first {MAX_PDF_PAGES} of {total} pages were read"
        needs_ocr = engine is None and missing_pages > 0 and text_pages == 0
        if engine is None and missing_pages > 0 and text_pages > 0:
            note = f"{missing_pages} page(s) have no text layer; install an OCR engine to read them"
        method = "pdf-text" if ocr_pages == 0 else ("pdf-ocr" if text_pages == 0 else "pdf-mixed")
        return Extraction(
            _clean("\n\n".join(parts)), method,
            engine=engine.name if ocr_pages else None, pages=total, note=note, needs_ocr=needs_ocr,
        )
    finally:
        pdf.close()


def extract(path: str, ftype: str, ext: str, engine) -> Extraction:
    """Dispatch on file type. `engine` is an ocr.Engine or None."""
    ext = ext.lower()
    if ext in (".txt", ".md"):
        return Extraction(_clean(read_text_file(path)), "text")
    if ext == ".docx":
        return Extraction(_clean(extract_docx(path)), "docx")
    if ext == ".pdf":
        return extract_pdf(path, engine)
    if ftype == "image":
        if ext == ".heic":
            raise Skipped("HEIC images are not supported yet")
        return extract_image(path, engine)
    raise Skipped("unsupported file type")
