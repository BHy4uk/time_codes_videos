from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from .config_schema import RuleV2 as Rule
from .normalize import normalize_text


_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _sentence_chunks(text: str) -> List[str]:
    return [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(text) if chunk.strip()]


def _has_strong_first_sentence_match(
    phrase_sentences: List[str],
    matched_window_text: str,
    similarity_threshold: int,
) -> bool:
    if len(phrase_sentences) <= 1:
        return True

    first_sentence_norm = phrase_sentences[0]
    first_sentence_tokens = first_sentence_norm.split()
    first_sentence_token_len = len(first_sentence_tokens)
    matched_window_norm = normalize_text(matched_window_text)
    candidate_prefix = " ".join(matched_window_norm.split()[: max(first_sentence_token_len + 1, 4)])
    first_sentence_score = max(
        int(fuzz.ratio(first_sentence_norm, candidate_prefix)),
        int(fuzz.partial_ratio(first_sentence_norm, candidate_prefix)),
        int(fuzz.token_set_ratio(first_sentence_norm, candidate_prefix)),
    )

    matched_tokens = matched_window_norm.split()
    leading_token_matches = 0
    for expected, actual in zip(first_sentence_tokens, matched_tokens):
        if expected != actual:
            break
        leading_token_matches += 1

    return (
        first_sentence_score >= max(75, similarity_threshold - 10)
        or leading_token_matches >= min(2, first_sentence_token_len)
    )


def _best_window_in_range(
    phrase_norm: str,
    phrase_len: int,
    words_norm: List[str],
    words_raw: List[str],
    words_start: List[float],
    search_start: int,
    search_end: int,
    threshold: int,
    phrase_sentences: Optional[List[str]] = None,
    allowed_start_indices: Optional[set[int]] = None,
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

    phrase_tokens = phrase_norm.split()
    phrase_head_len = min(4, len(phrase_tokens))
    phrase_head = " ".join(phrase_tokens[:phrase_head_len])

    # Window size range around phrase length.
    # ASR often compresses spans like "novecientos sesenta y uno" into a single
    # token such as "961", so we must allow meaningfully shorter windows than
    # the mapping phrase length. We still keep a bounded range to avoid a broad,
    # noisy search space.
    if phrase_len <= 2:
        min_len = 1
        max_len = phrase_len + 1
    else:
        min_len = max(1, int(round(phrase_len * 0.4)))
        max_len = max(min_len, int(round(phrase_len * 1.5)) + 3)

    end_limit = min(len(words_norm), search_end)
    start_limit = max(0, search_start)
    best_by_start: Dict[int, Dict[str, Any]] = {}

    for wlen in range(min_len, max_len + 1):
        if start_limit + wlen > end_limit:
            break

        for i in range(start_limit, end_limit - wlen + 1):
            window_tokens = words_norm[i : i + wlen]
            window_norm = " ".join(window_tokens)
            ts = int(fuzz.token_set_ratio(phrase_norm, window_norm))
            window_head_tokens = window_tokens[: min(max(phrase_head_len + 1, 4), len(window_tokens))]
            window_head = " ".join(window_head_tokens)
            head_score = max(
                int(fuzz.ratio(phrase_head, window_head)),
                int(fuzz.partial_ratio(phrase_head, window_head)),
                int(fuzz.token_set_ratio(phrase_head, window_head)),
            )

            # For mapping-based alignment, the beginning of the phrase is the
            # most important signal. ASR can badly distort later words, so we
            # keep candidates with a strong phrase-start match even when the
            # full-window token_set_ratio falls below the global threshold.
            if ts < threshold and head_score < max(80, threshold - 5):
                continue

            r = int(fuzz.ratio(phrase_norm, window_norm))

            rec = {
                "word_index": i,
                "word_len": wlen,
                "token_set_ratio": ts,
                "ratio": r,
                "start": float(words_start[i]),
                "matched_window_text": " ".join(words_raw[i : i + wlen]),
                "head_score": head_score,
                "boundary_aligned": allowed_start_indices is not None and i in allowed_start_indices,
            }

            if phrase_sentences and not _has_strong_first_sentence_match(
                phrase_sentences=phrase_sentences,
                matched_window_text=rec["matched_window_text"],
                similarity_threshold=threshold,
            ):
                continue

            window_key = (-r, -ts, abs(wlen - phrase_len), wlen)
            current = best_by_start.get(i)
            if current is None or window_key < current["window_key"]:
                rec["window_key"] = window_key
                best_by_start[i] = rec

    if not best_by_start:
        return None

    def start_key(rec: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
        return (
            0 if rec["boundary_aligned"] else 1,
            -int(rec["head_score"]),
            -int(rec["ratio"]),
            -int(rec["token_set_ratio"]),
            rec["word_index"],
            rec["word_len"],
        )

    best_rec = min(best_by_start.values(), key=start_key)
    best_rec.pop("window_key", None)
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
    allowed_start_indices: set[int] = {0}
    for index, word_text in enumerate(words_raw[:-1]):
        if word_text.rstrip().endswith((".", "!", "?")):
            allowed_start_indices.add(index + 1)

    segments = transcript.get("segments") or []

    # For speed: pre-normalize segments (used only as a coarse locator)
    # IMPORTANT: do NOT rely on Whisper segment `id` as a list index because we may skip empty segments.
    seg_norm = []
    for seg_i, s in enumerate(segments):
        seg_norm.append(
            {
                "seg_index": seg_i,
                "id": int(s.get("id", 0)),
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "word_start": int(s.get("word_start", 0)),
                "word_end": int(s.get("word_end", 0)),
                "text_norm": normalize_text(str(s.get("text", ""))),
            }
        )
        allowed_start_indices.add(int(s.get("word_start", 0)))

    resolved: List[Dict[str, Any]] = []

    # enforce order by word index
    cursor_word_idx = 0

    for idx, rule in enumerate(rules):
        phrase_raw = rule.text
        phrase_norm = normalize_text(phrase_raw)
        phrase_len = len(phrase_norm.split())
        phrase_sentences = [normalize_text(chunk) for chunk in _sentence_chunks(phrase_raw)]

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
            phrase_sentences=phrase_sentences,
            allowed_start_indices=allowed_start_indices,
        )

        # Deterministic anchoring fix for the FIRST phrase.
        # If the best fuzzy window starts inside the first segment, but the segment
        # itself appears to contain the beginning of the mapping phrase, we anchor
        # to segment_start instead of word-level offset.
        if idx == 0 and best is not None and best_seg is not None:
            seg_text_norm = best_seg.get("text_norm", "")
            # Condition A: window start is not at segment start
            window_inside_segment = int(best["word_index"]) > int(best_seg.get("word_start", 0))
            # Condition B: segment contains the beginning of the mapping phrase
            # Use a more robust phrase-beginning check:
            # - Take the first N words
            # - Also allow fuzzy match of that prefix against the segment
            head_n = max(2, min(8, len(phrase_norm.split())))
            phrase_head = " ".join(phrase_norm.split()[:head_n])
            segment_contains_phrase_head = bool(phrase_head) and (
                phrase_head in seg_text_norm or fuzz.partial_ratio(phrase_head, seg_text_norm) >= 90
            )
            # Condition C: guard against token_set_ratio passing while ratio is low
            guard_low_ratio = int(best["token_set_ratio"]) >= similarity_threshold and int(best["ratio"]) < 90

            if (window_inside_segment and segment_contains_phrase_head) or guard_low_ratio:
                # Anchor to segment start timestamp (not word timestamp)
                best["start"] = float(best_seg.get("start", best["start"]))
                best["anchored_to"] = "segment_start"
            else:
                best["anchored_to"] = "word_start"

        if best is None:
            raise ValueError(
                f"Could not resolve phrase {idx} to a timestamp above threshold={similarity_threshold}. "
                f"Phrase: {phrase_raw!r}"
            )

        # Update cursor: mapping phrases are atomic, so the next search begins
        # after the end of the matched window, not merely after its first word.
        cursor_word_idx = int(best["word_index"]) + int(best["word_len"])

        resolved.append(
            {
                "index": idx,
                **asdict(rule),
                "image": rule.asset,
                "start": float(best["start"]),
                "similarity": {
                    "token_set_ratio": int(best["token_set_ratio"]),
                    "ratio": int(best["ratio"]),
                },
                "match": {
                    "matched_window_text": best["matched_window_text"],
                    "word_index": int(best["word_index"]),
                    "word_len": int(best["word_len"]),
                    "anchored_to": best.get("anchored_to", "word_start"),
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
