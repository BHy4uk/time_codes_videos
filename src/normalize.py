import re


_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Normalize text for robust, deterministic fuzzy matching.

    Steps:
    - lowercase
    - remove punctuation/symbols
    - normalize whitespace
    """

    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()
