"""Text extraction from .txt and .pdf files with language detection.

Step 3 of v7 plan — leaf module with no internal imports.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.constants import Language

import fitz  # PyMuPDF
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

# French stopwords (common function words — enough for detection, not exhaustive)
_FRENCH_STOPWORDS = frozenset({
    "le", "la", "les", "une", "pour", "dans", "avec", "que", "qui", "est",
})


def detect_doc_lang(filename: str, text: str) -> Language:
    """Heuristic language detection based on filename hint + character/word analysis.

    Returns ``Language.AR``, ``Language.FR``, or ``Language.EN``. Detection is
    COSMETIC only — ``lang`` is stored in payloads for display/analytics, never
    used for retrieval routing.
    """
    name_lower = filename.lower()

    if "ar" in name_lower or _ARABIC_RE.search(text):
        return Language.AR

    if "fr" in name_lower:
        return Language.FR
    text_lower = text.lower()
    for word in _FRENCH_STOPWORDS:
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            return Language.FR

    return Language.EN


def _is_garbled(text: str) -> bool:
    """Detect broken ToUnicode CMap extraction: newline-after-every-glyph pattern.

    A clean paragraph has very few newlines relative to letter count.  A
    PDF with a broken ToUnicode map returns one line per glyph, making
    the ratio of ``\\n`` to letter characters abnormally high (>0.15).
    """
    letter_count = sum(1 for c in text if c.isalpha())
    if letter_count == 0:
        return False
    return text.count("\n") / letter_count > 0.15


def _resolve_tessdata_dir() -> str:
    """Locate the tesseract ``tessdata`` directory via a portable fallback chain.

    Return value is guaranteed to be a non-empty string (the last fallback
    is the original hardcoded path).

    Resolution order
    ----------------
    1. ``TESSDATA_PREFIX`` environment variable (if set and the directory exists).
    2. Parse ``tesseract --print-parameters`` for ``tessdata_dir`` value.
    3. Common system paths (Homebrew on macOS, Linux package-manager paths).
    4. ``/tmp/tessdata`` (original hardcoded default).

    Note
    ----
    We use ``--tessdata-dir`` on the tesseract CLI rather than manipulating
    the env var, because the CLI flag takes precedence and is supported across
    all modern tesseract versions (>= 4.x).
    """
    env_tessdata = os.environ.get("TESSDATA_PREFIX", "")
    if env_tessdata and os.path.isdir(env_tessdata):
        return env_tessdata

    if shutil.which("tesseract") is not None:
        try:
            result = subprocess.run(
                ["tesseract", "--print-parameters"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                # Format: "tessdata_dir  <path>  N" (tabs between fields)
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].strip() == "tessdata_dir":
                    param_path = parts[1].strip()
                    if param_path and os.path.isdir(param_path):
                        return param_path
        except (subprocess.TimeoutExpired, OSError):
            pass

    COMMON_TESSDATA_PATHS: list[str] = [
        "/opt/homebrew/share/tessdata",       # Apple Silicon Homebrew
        "/usr/local/share/tessdata",           # Intel Homebrew
        "/usr/share/tesseract-ocr/5/tessdata",  # Ubuntu 22.04+
        "/usr/share/tesseract-ocr/4/tessdata",  # Ubuntu 20.04
        "/usr/share/tesseract-ocr/tessdata",    # Arch / Fedora / some distros
        "/usr/share/tessdata",                 # Generic Linux
    ]
    for candidate in COMMON_TESSDATA_PATHS:
        if os.path.isdir(candidate):
            return candidate

    print("[extract] WARNING: Could not locate tessdata directory — falling back to /tmp/tessdata.")
    return "/tmp/tessdata"


_TESSERACT_LANG_MAP = {
    "ar": "ara",
    "fr": "fra",
    "en": "eng",
}


def _ocr_lang_from_hint(filename: str) -> str:
    """Choose Tesseract ``-l`` argument from filename language hint.

    Uses the same heuristic as ``detect_doc_lang`` but returns a
    Tesseract-compatible language string.  When the filename contains no
    language hint we default to English (the most common case for a
    general RAG pipeline).

    For mixed-language documents (e.g. Moroccan legal French/Arabic) we
    pass the best-guess primary language with English as a fallback combo.
    """
    name_lower = filename.lower()
    if "ar" in name_lower:
        return "ara+eng"
    if "fr" in name_lower:
        return "fra+eng"
    return "eng"


def _extract_pdf_ocr(path: str, ocr_lang: str = "eng") -> str | None:
    """Render each PDF page to PNG and run tesseract OCR.

    Returns concatenated OCR text (``"\\n\\n"``-separated pages) on
    success, or *None* if the ``tesseract`` binary is unavailable.

    The tessdata directory is resolved via :func:`_resolve_tessdata_dir`
    and passed through the ``--tessdata-dir`` CLI flag.

    Args:
        path: Path to the PDF file.
        ocr_lang: Tesseract ``-l`` argument (e.g. ``"ara+eng"``, ``"fra+eng"``).
    """
    if shutil.which("tesseract") is None:
        print("[extract] WARNING: tesseract not found — OCR fallback unavailable.")
        return None

    tessdata_dir = _resolve_tessdata_dir()

    doc = fitz.open(path)
    pages_text: list[str] = []

    for page in doc:
        pix = page.get_pixmap(dpi=300)

        png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        png_path = png.name
        png.close()  # closed so we can save over it
        try:
            pix.save(png_path)

            # Tesseract writes to <out_base>.txt
            out_fd, out_base = tempfile.mkstemp()
            os.close(out_fd)
            result = subprocess.run(
                [
                    "tesseract", png_path, out_base,
                    "-l", ocr_lang,
                    "--tessdata-dir", tessdata_dir,
                ],
                capture_output=True,
            )

            if result.returncode != 0:
                stderr_bytes = result.stderr or b""
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip() or "unknown error"
                print(f"[extract] WARNING: tesseract failed on page {(page.number or 0) + 1}: {stderr_text}")
                pages_text.append("")
            else:
                txt_path = out_base + ".txt"
                if os.path.exists(txt_path):
                    with open(txt_path, encoding="utf-8") as f:
                        pages_text.append(f.read())
                    os.unlink(txt_path)
                else:
                    pages_text.append("")
        finally:
            if os.path.exists(png_path):
                os.unlink(png_path)

    doc.close()
    return "\n\n".join(pages_text)


def extract_text(path: str) -> tuple[str, Language]:
    """Extract text from a .txt or .pdf file and detect its language.

    Args:
        path: File path to extract from.

    Returns:
        ``(text, lang)`` tuple where *lang* is "ar", "fr", or "en".

    Raises:
        ValueError: If the file extension is not .txt or .pdf, or if the
            file cannot be read.
    """
    try:
        p = Path(path)
        suffix = p.suffix.lower()

        if suffix == ".txt":
            text = p.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            doc = fitz.open(path)
            text = "\n\n".join(page.get_text("text") for page in doc)
            doc.close()

            # Broken-ToUnicode CMap detection + OCR fallback
            if _is_garbled(text):
                print(f"[extract] WARNING: Garbled text detected in '{path}' — trying OCR fallback ...")
                ocr_lang = _ocr_lang_from_hint(p.name)
                ocr_text = _extract_pdf_ocr(path, ocr_lang)
                if ocr_text is not None:
                    text = ocr_text
                else:
                    print(
                        "[extract] WARNING: OCR fallback failed — keeping garbled extracted text."
                    )

            text = re.sub(r"\n{3,}", "\n\n", text)
        else:
            raise ValueError(f"Unsupported file type: {path}")
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}") from None
    except (PermissionError, OSError) as e:
        raise ValueError(f"Cannot read {path}: {e}") from e

    lang = detect_doc_lang(p.name, text)

    # Guard: empty extraction (e.g. scanned/image-only PDF)
    if not text.strip():
        print(f"[extract] WARNING: No text extracted from '{path}' (scanned or empty document).")
        return ("", lang)

    return (text, lang)
