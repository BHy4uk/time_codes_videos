import re
import unicodedata


_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def _strip_diacritics(s: str) -> str:
    normalized = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(s: str) -> str:
    """Normalize text for robust, deterministic fuzzy matching.

    Steps:
    - lowercase
    - fold accented characters to ASCII equivalents
    - remove punctuation/symbols
    - normalize whitespace
    """

    s = _strip_diacritics(s).lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()
