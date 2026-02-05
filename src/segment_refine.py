from __future__ import annotations

import re
from typing import Any, Dict, List


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def refine_segments_sentence_split(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split Whisper segments into smaller sentence-like segments.

    Motivation:
    - Whisper segments often contain multiple sentences.
    - Our pipeline triggers images at segment.start.
    - If the matched phrase appears mid-segment, the image appears too early.

    This function deterministically splits each segment by punctuation boundaries
    and assigns approximate timestamps by proportional text length.

    Output segments keep fields: id,start,end,text.

    Determinism:
    - Splitting uses a fixed regex.
    - Timestamp allocation is purely proportional and deterministic.

    Note:
    - This is an approximation. For perfect alignment you would need
      word-level timestamps and phrase-localization.
    """

    refined: List[Dict[str, Any]] = []
    next_id = 0

    for seg in segments:
        text_raw = (seg.get("text") or "").strip()
        if not text_raw:
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= start:
            continue

        parts = [p.strip() for p in _SENT_SPLIT_RE.split(text_raw) if p.strip()]
        if len(parts) <= 1:
            refined.append({"id": next_id, "start": start, "end": end, "text": text_raw})
            next_id += 1
            continue

        total_len = sum(len(p) for p in parts)
        if total_len <= 0:
            refined.append({"id": next_id, "start": start, "end": end, "text": text_raw})
            next_id += 1
            continue

        dur = end - start
        cur_t = start

        # allocate times proportional to part length
        for i, p in enumerate(parts):
            frac = len(p) / total_len
            part_dur = dur * frac
            part_start = cur_t
            part_end = end if i == len(parts) - 1 else (cur_t + part_dur)

            # enforce monotonicity
            if part_end <= part_start:
                part_end = min(end, part_start + 1e-3)

            refined.append({"id": next_id, "start": part_start, "end": part_end, "text": p})
            next_id += 1
            cur_t = part_end

    # ensure sorted
    refined.sort(key=lambda s: (s["start"], s["id"]))
    return refined
