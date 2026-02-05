from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from .config import Rule
from .normalize import normalize_text


def _best_window_in_range(
    phrase_norm: str,
    phrase_len: int,
    words_norm: List[str],
    words_raw: List[str],
    words_start: List[float],
    search_start: int,
    search_end: int,
    threshold: int,
) -> Optional[Dict[str, Any]]:
    """Find best fuzzy match window of words within [search_start, search_end).

    Deterministic tie-break order:
    1) higher token_set_ratio
    2) higher ratio
    3) earlier word index
    4) shorter window length
    """

    if phrase_len <= 0:
        return None

    # window size range around phrase length
    min_len = max(3, int(round(phrase_len * 0.75)))
    max_len = max(min_len, int(round(phrase_len * 1.25)) + 2)

    best: Optional[Tuple[int, int, int, int, int]] = None
    best_rec: Optional[Dict[str, Any]] = None

    end_limit = min(len(words_norm), search_end)
    start_limit = max(0, search_start)

    for wlen in range(min_len, max_len + 1):
        if start_limit + wlen > end_limit:
            break

        for i in range(start_limit, end_limit - wlen + 1):
            window_norm = " ".join(words_norm[i : i + wlen])
            ts = int(fuzz.token_set_ratio(phrase_norm, window_norm))
            if ts < threshold:
                continue

            r = int(fuzz.ratio(phrase_norm, window_norm))

            key = (-ts, -r, i, wlen)
            if best is None or key < best:
                best = key
                best_rec = {
                    "word_index": i,
                    "word_len": wlen,
                    "token_set_ratio": ts,
                    "ratio": r,
                    "start": float(words_start[i]),
                    "matched_window_text": " ".join(words_raw[i : i + wlen]),
                }

    return best_rec


def resolve_phrase_start_times(
    rules: List[Rule],
    transcript: Dict[str, Any],
    similarity_threshold: int = 85,
) -> List[Dict[str, Any]]:
    """Resolve mapping.json phrases to start timestamps in the audio.

    New conceptual model (per your request):
    - mapping.json is the single source of truth and defines ORDER.
    - For each rule.text (phrase), we find when that phrase begins in the audio.
    - The resulting list is one phrase -> one resolved timestamp.

    Implementation details:
    - Uses faster-whisper word timestamps from `transcribe_audio()`.
    - Uses fuzzy matching to compensate for minor STT differences.
    - Enforces chronological order: phrase N is searched only after phrase N-1.

    Returns a list of "resolved phrases":
      {
        "index": 0,
        "image": "01.jpg",
        "text": "...",
        "effects": {...},
        "start": 12.34,
        "similarity": {"token_set_ratio": 92, "ratio": 90},
        "match": {"matched_window_text": "...", "word_index": 123, "word_len": 18}
      }

    If any phrase cannot be resolved above the threshold, raises ValueError.
    """

    words = transcript.get("words") or []
    if not isinstance(words, list) or not words:
        raise ValueError("Transcript must include non-empty 'words' with timestamps. Re-run transcription.")

    words_raw = [str(w.get("text", "")).strip() for w in words]
    words_norm = [normalize_text(w) for w in words_raw]
    words_start = [float(w.get("start", 0.0)) for w in words]

    segments = transcript.get("segments") or []

    # For speed: pre-normalize segments (used only as a coarse locator)
    seg_norm = []
    for s in segments:
        seg_norm.append(
            {
                "id": int(s.get("id", 0)),
                "word_start": int(s.get("word_start", 0)),
                "word_end": int(s.get("word_end", 0)),
                "text_norm": normalize_text(str(s.get("text", ""))),
            }
        )

    resolved: List[Dict[str, Any]] = []

    # enforce order by word index
    cursor_word_idx = 0

    for idx, rule in enumerate(rules):
        phrase_raw = rule.text
        phrase_norm = normalize_text(phrase_raw)
        phrase_len = len(phrase_norm.split())

        # 1) choose best coarse segment after cursor
        best_seg = None
        best_seg_score = -1
        for s in seg_norm:
            if s["word_end"] <= cursor_word_idx:
                continue
            score = int(fuzz.token_set_ratio(phrase_norm, s["text_norm"]))
            if score > best_seg_score:
                best_seg_score = score
                best_seg = s

        # 2) define search range (words)
        if best_seg is not None and best_seg_score >= max(40, similarity_threshold - 15):
            # Expand a bit to allow crossing boundaries.
            search_start = max(cursor_word_idx, best_seg["word_start"] - 20)
            search_end = min(len(words_norm), best_seg["word_end"] + 20)
        else:
            # fallback: search globally but still after cursor
            search_start = cursor_word_idx
            search_end = len(words_norm)

        best = _best_window_in_range(
            phrase_norm=phrase_norm,
            phrase_len=phrase_len,
            words_norm=words_norm,
            words_raw=words_raw,
            words_start=words_start,
            search_start=search_start,
            search_end=search_end,
            threshold=similarity_threshold,
        )

        if best is None:
            raise ValueError(
                f"Could not resolve phrase {idx} to a timestamp above threshold={similarity_threshold}. "
                f"Phrase: {phrase_raw!r}"
            )

        # Update cursor: next phrase search starts after this phrase start.
        cursor_word_idx = int(best["word_index"]) + 1

        resolved.append(
            {
                "index": idx,
                **asdict(rule),
                "start": float(best["start"]),
                "similarity": {
                    "token_set_ratio": int(best["token_set_ratio"]),
                    "ratio": int(best["ratio"]),
                },
                "match": {
                    "matched_window_text": best["matched_window_text"],
                    "word_index": int(best["word_index"]),
                    "word_len": int(best["word_len"]),
                },
            }
        )

    # Ensure increasing starts (strict order). If not, fail fast.
    for i in range(1, len(resolved)):
        if resolved[i]["start"] < resolved[i - 1]["start"]:
            raise ValueError(
                "Resolved phrase timestamps are not monotonic. "
                "This indicates ambiguous matching; adjust phrases or raise threshold."
            )

    return resolved
