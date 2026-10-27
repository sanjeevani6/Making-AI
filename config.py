"""Central configuration for the OCR project.

Everything that used to be hard-coded (Tesseract path, model names, ...)
lives here so it can be changed in one place or via environment variables.
"""
import os
import shutil

# --------------------------------------------------------------------------
# Tesseract
# --------------------------------------------------------------------------
_WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def find_tesseract():
    """Return the Tesseract executable path (env var > PATH > Windows default)."""
    env = os.environ.get("TESSERACT_CMD")
    if env and os.path.exists(env):
        return env
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    if os.path.exists(_WINDOWS_DEFAULT):
        return _WINDOWS_DEFAULT
    return "tesseract"  # let pytesseract raise a clear error


TESSERACT_CMD = find_tesseract()

# Tesseract language(s). Use e.g. "eng+hin" if you installed the Hindi data.
OCR_LANG = os.environ.get("OCR_LANG", "eng")

# --------------------------------------------------------------------------
# Printed text
# --------------------------------------------------------------------------
# If Tesseract's mean confidence is below this and EasyOCR is installed,
# EasyOCR is tried as well and the more confident result is kept.
EASYOCR_FALLBACK_BELOW = 75
EASYOCR_LANGS = os.environ.get("EASYOCR_LANGS", "en").split(",")

# --------------------------------------------------------------------------
# Handwriting (TrOCR)
# --------------------------------------------------------------------------
# "auto" -> large model when a CUDA GPU exists, base model on CPU.
HANDWRITING_MODEL = os.environ.get("HANDWRITING_MODEL", "auto")
HANDWRITING_BASE = "microsoft/trocr-base-handwritten"
HANDWRITING_LARGE = "microsoft/trocr-large-handwritten"
HANDWRITING_BATCH_SIZE = 8
HANDWRITING_NUM_BEAMS = 5
HANDWRITING_MAX_TOKENS = 96
# TrOCR squeezes every crop to 384x384, so extremely wide lines lose detail.
# Lines wider than this (width / line height) are split, but ONLY at real
# blank gaps between words - never through a word.
HANDWRITING_MAX_ASPECT = 18.0
