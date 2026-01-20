from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from .config import Rule
from .normalize import normalize_text


def _tie_break_key(rule: Rule) -> Tuple[str, str]:
    """Deterministic tie-breaker key.

    Only used when similarity scores are exactly equal for a segment.
    """

    return (normalize_text(rule.text), rule.image)


def match_segments_to_rules(
    segments: List[Dict[str, Any]],
    rules: List[Rule],
    similarity_threshold: int = 85,
) -> List[Dict[str, Any]]:
    """Match each transcription segment to at most one rule.

    Full phrase matching (Variant B): each config rule.text is an atomic unit.

    Constraints implemented:
    - One transcription segment triggers at most one image.
    - Each image triggers at most once, at earliest matching segment.
    - If a segment matches multiple rules above threshold: pick highest similarity.
      If exact tie: deterministic tie-breaker (normalized phrase, then image filename).
    - No keyword spotting, no partial triggering logic.

    Returns a list of match records:
      {
        "segment_id": int,
        "segment_start": float,
        "segment_end": float,
        "segment_text": str,
        "rule": {"image": str, "text": str},
        "similarity": int
      }

    Note: This function is deterministic given identical segments and rules.
    """

    norm_rules: List[Tuple[Rule, str]] = [(r, normalize_text(r.text)) for r in rules]

    triggered_images: set[str] = set()
    triggered_rules: set[str] = set()  # use normalized rule text as identity

    matches: List[Dict[str, Any]] = []

    for seg in segments:
        seg_text_raw = (seg.get("text") or "").strip()
        if not seg_text_raw:
            continue
        seg_text = normalize_text(seg_text_raw)

        best: Optional[Tuple[int, Rule]] = None  # (score, rule)

        for rule, rule_norm in norm_rules:
            if rule.image in triggered_images:
                continue
            if rule_norm in triggered_rules:
                continue

            # Robust fuzzy match compensating for STT imperfections
            score = int(fuzz.token_set_ratio(seg_text, rule_norm))
            if score < similarity_threshold:
                continue

            if best is None:
                best = (score, rule)
            else:
                best_score, best_rule = best
                if score > best_score:
                    best = (score, rule)
                elif score == best_score:
                    # Deterministic tie-breaker (NOT config order)
                    if _tie_break_key(rule) < _tie_break_key(best_rule):
                        best = (score, rule)

        if best is None:
            continue

        score, rule = best
        rule_norm = normalize_text(rule.text)

        triggered_images.add(rule.image)
        triggered_rules.add(rule_norm)

        matches.append(
            {
                "segment_id": int(seg.get("id")),
                "segment_start": float(seg.get("start")),
                "segment_end": float(seg.get("end")),
                "segment_text": seg_text_raw,
                "rule": asdict(rule),
                "similarity": score,
            }
        )

    # ensure chronological
    matches.sort(key=lambda m: (m["segment_start"], m["segment_id"]))
    return matches
