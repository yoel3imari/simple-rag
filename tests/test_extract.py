"""Unit tests for src/extract.py — text extraction, language detection, OCR fallback."""

from __future__ import annotations

from pathlib import Path
from subprocess import TimeoutExpired
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.extract import (
    _extract_pdf_ocr,
    _is_garbled,
    _ocr_lang_from_hint,
    _resolve_tessdata_dir,
    detect_doc_lang,
    extract_text,
)


# ---------------------------------------------------------------------------
# A) detect_doc_lang
# ---------------------------------------------------------------------------

class TestDetectDocLang:
    """Heuristic language detection based on filename + text analysis."""

    def test_arabic_unicode_chars_in_text(self):
        """Text containing Arabic Unicode characters (U+0600–U+06FF) returns 'ar'."""
        assert detect_doc_lang("doc.pdf", "\u0633\u0644\u0627\u0645") == "ar"  # سلام

    def test_filename_contains_ar(self):
        """Filename containing substring 'ar' returns 'ar' regardless of text."""
        assert detect_doc_lang("ar_doc.pdf", "Hello world") == "ar"

    def test_filename_contains_fr(self):
        """Filename containing substring 'fr' returns 'fr' when no Arabic present."""
        assert detect_doc_lang("fr_doc.pdf", "Hello world") == "fr"

    def test_french_stopwords_in_text(self):
        """Text containing French stopwords (le, la, les, …) returns 'fr'."""
        assert detect_doc_lang("doc.pdf", "Le monde est beau aujourd'hui") == "fr"

    def test_french_stopwords_la_match(self):
        """The French stopword 'la' is correctly detected."""
        assert detect_doc_lang("doc.pdf", "la maison est grande") == "fr"

    def test_french_stopwords_les_match(self):
        """The French stopword 'les' is correctly detected."""
        assert detect_doc_lang("doc.pdf", "les enfants jouent") == "fr"

    def test_plain_english_returns_en(self):
        """Plain English text with English filename returns 'en'."""
        assert detect_doc_lang("doc.pdf", "Hello world, this is English") == "en"

    def test_empty_text_returns_en(self):
        """Empty/blank text with neutral filename returns 'en'."""
        assert detect_doc_lang("doc.pdf", "") == "en"

    def test_arabic_chars_override_filename_fr(self):
        """Arabic characters in text take priority over 'fr' in filename."""
        assert detect_doc_lang("fr_doc.pdf", "\u0645\u0631\u062d\u0628\u0627") == "ar"  # مرحبا

    def test_arabic_chars_override_french_stopwords(self):
        """Arabic characters in text take priority over French stopwords in text."""
        assert detect_doc_lang("doc.pdf", "\u0645\u0631\u062d\u0628\u0627 le monde") == "ar"

    def test_filename_ar_overrides_french_stopwords(self):
        """'ar' in filename takes priority over French stopwords in text."""
        assert detect_doc_lang("ar_doc.pdf", "Le monde est beau") == "ar"

    def test_french_stopword_at_word_boundary(self):
        """Stopword detection uses \\b boundaries so 'le' in 'lemon' does not match."""
        assert detect_doc_lang("doc.pdf", "lemon tea sugar") == "en"

    def test_mixed_case_filename(self):
        """Mixed-case filename is normalised before substring search."""
        assert detect_doc_lang("FR_DOC.PDF", "Hello world") == "fr"
        assert detect_doc_lang("AR_DOC.PDF", "Hello world") == "ar"

    def test_neutral_filename_no_hint_returns_en(self):
        """Filename without 'ar' or 'fr' substring returns 'en' for English text."""
        assert detect_doc_lang("notes.pdf", "Hello") == "en"


# ---------------------------------------------------------------------------
# B) _is_garbled
# ---------------------------------------------------------------------------

class TestIsGarbled:
    """Broken ToUnicode CMap detection via newline-per-glyph pattern."""

    def test_high_newline_ratio_returns_true(self):
        """Newlines / letters > 0.15 indicates garbled text."""
        text = "G\nA\nR\nB\nL\nE\nD\nT\ne\nx\nt\n"
        assert _is_garbled(text) is True

    def test_clean_paragraph_returns_false(self):
        """Clean paragraph text has newlines / letters << 0.15."""
        text = "Hello world. This is a clean paragraph with enough words."
        assert _is_garbled(text) is False

    def test_empty_string_returns_false(self):
        """Empty string has zero letter count → False."""
        assert _is_garbled("") is False

    def test_no_alphabetic_chars_returns_false(self):
        """Text without any alphabetic characters → False (letter_count == 0)."""
        assert _is_garbled("123\n456\n789\n") is False

    def test_boundary_below_ratio(self):
        """Ratio exactly 0.15 should be considered clean (not > 0.15)."""
        # 3 newlines / 20 letters = 0.15 → not > 0.15 → False
        text = "abcde\nfghij\nklmno\npqrst"
        assert _is_garbled(text) is False

    def test_boundary_above_ratio(self):
        """Ratio just above 0.15 should be detected as garbled."""
        # 4 newlines / 20 letters = 0.20 → should be True
        text = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt"
        # Add one more newline
        text += "\n"
        assert _is_garbled(text) is True

    def test_single_letter_with_newline(self):
        """Single letter with newline ratio is 1.0 > 0.15 → True."""
        assert _is_garbled("A\n") is True


# ---------------------------------------------------------------------------
# C) _ocr_lang_from_hint
# ---------------------------------------------------------------------------

class TestOcrLangFromHint:
    """Tesseract -l argument selection from filename hint."""

    def test_ar_filename_returns_ara_plus_eng(self):
        """Filename with 'ar' returns 'ara+eng'."""
        assert _ocr_lang_from_hint("ar_doc.pdf") == "ara+eng"

    def test_fr_filename_returns_fra_plus_eng(self):
        """Filename with 'fr' returns 'fra+eng'."""
        assert _ocr_lang_from_hint("fr_doc.pdf") == "fra+eng"

    def test_no_language_hint_returns_eng(self):
        """Filename without language hint defaults to 'eng'."""
        assert _ocr_lang_from_hint("doc.pdf") == "eng"

    def test_mixed_case_filename(self):
        """Case-insensitive matching for language hints."""
        assert _ocr_lang_from_hint("AR_DOC.PDF") == "ara+eng"
        assert _ocr_lang_from_hint("FR_DOC.PDF") == "fra+eng"

    def test_neutral_filename_returns_eng(self):
        """Filename without 'ar' or 'fr' returns 'eng'."""
        assert _ocr_lang_from_hint("notes.pdf") == "eng"


# ---------------------------------------------------------------------------
# D) _resolve_tessdata_dir
# ---------------------------------------------------------------------------

class TestResolveTessdataDir:
    """Tessdata directory resolution via fallback chain.

    Resolution order: TESSDATA_PREFIX env → tesseract --print-parameters
    → common paths → /tmp/tessdata
    """

    def test_env_var_exists(self, monkeypatch):
        """TESSDATA_PREFIX set and directory exists → use it."""
        monkeypatch.setenv("TESSDATA_PREFIX", "/custom/tessdata")
        with patch("os.path.isdir", return_value=True):
            result = _resolve_tessdata_dir()
        assert result == "/custom/tessdata"

    def test_env_var_dir_does_not_exist_falls_through(self, monkeypatch):
        """TESSDATA_PREFIX set but directory missing → continue chain."""
        monkeypatch.setenv("TESSDATA_PREFIX", "/nonexistent/tessdata")
        with (
            patch("os.path.isdir", side_effect=lambda p: p == "/from/tessdata"),
            patch("shutil.which", return_value="/usr/bin/tesseract"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value.stdout = "tessdata_dir\t/from/tessdata\tN\n"
            mock_run.return_value.stderr = ""
            result = _resolve_tessdata_dir()
        assert result == "/from/tessdata"

    def test_via_tesseract_print_parameters(self, monkeypatch):
        """TESSDATA_PREFIX unset, tesseract --print-parameters supplies path."""
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
        with (
            patch("os.path.isdir", side_effect=lambda p: p == "/opt/share/tessdata"),
            patch("shutil.which", return_value="/usr/bin/tesseract"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value.stdout = (
                "some_param\tval\tN\n"
                "tessdata_dir\t/opt/share/tessdata\tN\n"
                "other_param\tval2\tN\n"
            )
            mock_run.return_value.stderr = ""
            result = _resolve_tessdata_dir()
        assert result == "/opt/share/tessdata"

    def test_tesseract_timeout_falls_through(self, monkeypatch):
        """tesseract --print-parameters times out → continue to common paths."""
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
        with (
            patch("os.path.isdir", side_effect=lambda p: p == "/usr/share/tessdata"),
            patch("shutil.which", return_value="/usr/bin/tesseract"),
            patch("subprocess.run", side_effect=TimeoutExpired("cmd", 10)),
        ):
            result = _resolve_tessdata_dir()
        assert result == "/usr/share/tessdata"

    def test_tesseract_oserror_falls_through(self, monkeypatch):
        """tesseract binary fails to execute → continue to common paths."""
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
        with (
            patch("os.path.isdir", side_effect=lambda p: p == "/usr/local/share/tessdata"),
            patch("shutil.which", return_value="/usr/bin/tesseract"),
            patch("subprocess.run", side_effect=OSError("cannot execute")),
        ):
            result = _resolve_tessdata_dir()
        assert result == "/usr/local/share/tessdata"

    def test_common_paths_found(self, monkeypatch):
        """No env var, no tesseract → use first existing common path."""
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
        with (
            patch("os.path.isdir", side_effect=lambda p: p == "/usr/share/tessdata"),
            patch("shutil.which", return_value=None),
        ):
            result = _resolve_tessdata_dir()
        assert result == "/usr/share/tessdata"

    def test_fallback_to_tmp(self, monkeypatch):
        """Nothing works → return '/tmp/tessdata' as last resort."""
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
        with (
            patch("os.path.isdir", return_value=False),
            patch("shutil.which", return_value=None),
        ):
            result = _resolve_tessdata_dir()
        assert result == "/tmp/tessdata"

    def test_fallback_warning_printed(self, monkeypatch, capsys):
        """Fallback to /tmp prints a WARNING to stderr/stdout."""
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
        with (
            patch("os.path.isdir", return_value=False),
            patch("shutil.which", return_value=None),
        ):
            _resolve_tessdata_dir()
            captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "tmp/tessdata" in captured.out


# ---------------------------------------------------------------------------
# E) _extract_pdf_ocr
# ---------------------------------------------------------------------------

class TestExtractPdfOcr:
    """PDF→PNG→Tesseract OCR pipeline."""

    def test_tesseract_not_found_returns_none(self, capsys):
        """When tesseract binary is unavailable return None without crashing."""
        with patch("shutil.which", return_value=None):
            result = _extract_pdf_ocr("/p/file.pdf", "eng")
        assert result is None

    def test_tesseract_not_found_warning(self, capsys):
        """Warning message is printed when tesseract is not found."""
        with patch("shutil.which", return_value=None):
            _extract_pdf_ocr("/p/file.pdf", "eng")
            captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "tesseract not found" in captured.out

    def test_successful_ocr_two_pages(self):
        """Two-page PDF returns page texts joined by '\\n\\n'."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([mock_page, mock_page])

        png1, png2 = MagicMock(), MagicMock()
        png1.name = "/tmp/_test_p1.png"
        png2.name = "/tmp/_test_p2.png"

        with (
            patch("src.extract._resolve_tessdata_dir", return_value="/tmp/td"),
            patch("shutil.which", return_value="/usr/bin/tesseract"),
            patch("src.extract.fitz.open", return_value=mock_doc),
            patch("src.extract.subprocess.run") as mock_run,
            patch("src.extract.tempfile.NamedTemporaryFile") as mock_ntf,
            patch("src.extract.tempfile.mkstemp") as mock_mkstemp,
            patch("src.extract.os.close"),
            patch("src.extract.os.path.exists", return_value=True),
            patch("src.extract.os.unlink"),
            patch("builtins.open", mock_open(read_data="OCR page content")),
        ):
            mock_ntf.side_effect = [png1, png2]
            mock_mkstemp.side_effect = [(3, "/tmp/_b1"), (4, "/tmp/_b2")]
            mock_run.return_value.returncode = 0

            result = _extract_pdf_ocr("/p/file.pdf", "eng")
        assert result == "OCR page content\n\nOCR page content"

    def test_ocr_output_file_missing_yields_empty_page(self):
        """If tesseract succeeds but .txt output is missing, page is empty string."""
        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = MagicMock()
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([mock_page])

        with (
            patch("src.extract._resolve_tessdata_dir", return_value="/tmp/td"),
            patch("shutil.which", return_value="/usr/bin/tesseract"),
            patch("src.extract.fitz.open", return_value=mock_doc),
            patch("src.extract.subprocess.run") as mock_run,
            patch("src.extract.tempfile.NamedTemporaryFile") as mock_ntf,
            patch("src.extract.tempfile.mkstemp") as mock_mkstemp,
            patch("src.extract.os.close"),
            patch("src.extract.os.path.exists", return_value=False),
            patch("src.extract.os.unlink"),
        ):
            mock_ntf.return_value.name = "/tmp/_p.png"
            mock_mkstemp.return_value = (3, "/tmp/_b")
            mock_run.return_value.returncode = 0

            result = _extract_pdf_ocr("/p/file.pdf", "eng")
        assert result == ""

    def test_tesseract_nonzero_returncode(self, capsys):
        """Tesseract returning non-zero → page is empty string, warning printed."""
        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = MagicMock()
        mock_page.number = 0
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([mock_page])

        with (
            patch("src.extract._resolve_tessdata_dir", return_value="/tmp/td"),
            patch("shutil.which", return_value="/usr/bin/tesseract"),
            patch("src.extract.fitz.open", return_value=mock_doc),
            patch("src.extract.subprocess.run") as mock_run,
            patch("src.extract.tempfile.NamedTemporaryFile") as mock_ntf,
            patch("src.extract.tempfile.mkstemp") as mock_mkstemp,
            patch("src.extract.os.close"),
            patch("src.extract.os.path.exists", return_value=False),
            patch("src.extract.os.unlink"),
        ):
            mock_ntf.return_value.name = "/tmp/_p.png"
            mock_mkstemp.return_value = (3, "/tmp/_b")
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = b"tesseract error"

            result = _extract_pdf_ocr("/p/file.pdf", "eng")
            captured = capsys.readouterr()
        assert result == ""
        assert "WARNING" in captured.out
        assert "tesseract failed" in captured.out
        assert "page 1" in captured.out or "page 1" in captured.out


# ---------------------------------------------------------------------------
# F) extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    """Top-level text extraction dispatch (.txt / .pdf → (text, lang))."""

    def test_txt_file(self):
        """.txt file → read raw text + detect language."""
        with patch.object(Path, "read_text", return_value="Hello world"):
            text, lang = extract_text("/path/to/file.txt")
        assert text == "Hello world"
        assert lang == "en"

    def test_txt_file_french(self):
        """.txt file with French text → detected as 'fr'."""
        with patch.object(Path, "read_text", return_value="Le monde est beau"):
            text, lang = extract_text("/path/to/file.txt")
        assert lang == "fr"

    def test_txt_file_arabic(self):
        """.txt file with Arabic text → detected as 'ar'."""
        with patch.object(Path, "read_text", return_value="\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"):
            text, lang = extract_text("/path/to/file.txt")
        assert lang == "ar"

    def test_pdf_clean_text(self):
        """.pdf with clean text (not garbled) → returns extracted text directly."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Clean PDF text content"
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([mock_page])

        with patch("src.extract.fitz.open", return_value=mock_doc):
            text, lang = extract_text("/path/to/doc.pdf")
        assert "Clean PDF text content" in text
        assert lang == "en"

    def test_pdf_garbled_triggers_ocr(self):
        """Garbled PDF text → OCR fallback replaces extracted text."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "G\nA\nR\nB"  # will trigger garbled check
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([mock_page])

        with (
            patch("src.extract.fitz.open", return_value=mock_doc),
            patch("src.extract._is_garbled", return_value=True),
            patch("src.extract._extract_pdf_ocr", return_value="OCR result text"),
            patch("src.extract._ocr_lang_from_hint", return_value="eng"),
        ):
            text, lang = extract_text("/path/to/doc.pdf")
        assert text == "OCR result text"
        assert lang == "en"

    def test_pdf_garbled_ocr_fallback_fails(self):
        """Garbled + OCR returns None → keep garbled text with warning."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "G\nA\nR\nB"
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([mock_page])

        with (
            patch("src.extract.fitz.open", return_value=mock_doc),
            patch("src.extract._is_garbled", return_value=True),
            patch("src.extract._extract_pdf_ocr", return_value=None),
            patch("src.extract._ocr_lang_from_hint", return_value="eng"),
        ):
            text, lang = extract_text("/path/to/doc.pdf")
        # Should keep the original garbled text when OCR fails
        assert "G" in text
        assert "A" in text
        assert lang == "en"

    def test_pdf_empty_text_returns_empty(self):
        """Empty/scanned PDF with no text → returns ('', lang)."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([mock_page])

        with patch("src.extract.fitz.open", return_value=mock_doc):
            text, lang = extract_text("/path/to/doc.pdf")
        assert text == ""
        assert lang == "en"

    def test_unsupported_extension_raises_value_error(self):
        """File with unsupported extension (.doc) raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text("/path/to/file.doc")

    def test_unsupported_extension_no_ext(self):
        """File with no extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text("/path/to/file_without_ext")

    def test_pdf_multi_page_clean(self):
        """Multi-page PDF joins pages with double newline separator."""
        page1 = MagicMock()
        page1.get_text.return_value = "Page 1 content"
        page2 = MagicMock()
        page2.get_text.return_value = "Page 2 content"
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([page1, page2])

        with patch("src.extract.fitz.open", return_value=mock_doc):
            text, lang = extract_text("/path/to/doc.pdf")
        assert "Page 1 content" in text
        assert "Page 2 content" in text
        assert "\n\n" in text
        assert lang == "en"

    def test_pdf_clean_text_not_garbled(self):
        """PDF with clean text is NOT sent to OCR (verifying the real _is_garbled works)."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is clean readable text."
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([mock_page])

        with (
            patch("src.extract.fitz.open", return_value=mock_doc),
            patch("src.extract._extract_pdf_ocr") as mock_ocr,
        ):
            text, lang = extract_text("/path/to/doc.pdf")
        mock_ocr.assert_not_called()
        assert "clean readable text" in text
        assert lang == "en"
