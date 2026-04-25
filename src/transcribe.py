from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from faster_whisper import WhisperModel


_SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"')\]\u00bb\u201d\u2019]+)?$")


def transcribe_audio(
    audio_path: str,
    model_size_or_path: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    vad_filter: bool = False,
    vad_min_silence_ms: int = 500,
) -> Dict[str, Any]:
    """Transcribe audio or video media with word-level timestamps using faster-whisper.

    IMPORTANT: For strict timing alignment, VAD is disabled by default because
    it can suppress early speech and shift the first segment/word timestamps later.

    Returns structured data:
      {
        "language": "en",
        "duration": <float seconds>,
        "segments": [
          {"id": 0, "start": 0.0, "end": 3.2, "text": "..."},
          ...
        ]
      }

    Determinism:
    - We use fixed decoding parameters.
    - We avoid temperature sampling.
    - We enable word timestamps for phrase start-time localization.
    """

    model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)

    # Fixed decode params for determinism
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        # IMPORTANT FOR TIMING: VAD can suppress early speech and shift the first
        # detected timestamps later. For strict phrase alignment, we default VAD OFF.
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": int(vad_min_silence_ms)},
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=None,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []

    word_cursor = 0
    for i, seg in enumerate(segments_iter):
        text = (seg.text or "").strip()
        if not text:
            continue

        seg_words = []
        if getattr(seg, "words", None) is not None:
            for w in seg.words:
                w_text = (getattr(w, "word", None) or getattr(w, "text", "") or "").strip()
                if not w_text:
                    continue
                w_start = float(getattr(w, "start", seg.start))
                w_end = float(getattr(w, "end", seg.end))
                seg_words.append({"text": w_text, "start": w_start, "end": w_end})

        seg_word_start = word_cursor
        for w in seg_words:
            words.append(w)
            word_cursor += 1
        seg_word_end = word_cursor

        segments.append(
            {
                "id": i,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
                "word_start": seg_word_start,
                "word_end": seg_word_end,
            }
        )

    duration = None
    if hasattr(info, "duration"):
        duration = float(getattr(info, "duration"))

    return {
        "language": getattr(info, "language", None),
        "duration": duration,
        "segments": segments,
        "words": words,
    }


def _is_sentence_end(token: str) -> bool:
    return bool(_SENTENCE_END_RE.search(token.strip()))


def _extract_segment_phrases(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Whisper segments into phrase timeline entries."""

    phrases: List[Dict[str, Any]] = []
    for index, segment in enumerate(transcript.get("segments") or []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if end < start:
            end = start

        phrases.append(
            {
                "index": index,
                "segment_id": int(segment.get("id", index)),
                "start": start,
                "end": end,
                "text": text,
            }
        )

    return phrases


def _extract_word_phrases(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Whisper word timestamps into one-entry-per-word timeline entries."""

    phrases: List[Dict[str, Any]] = []
    for index, word in enumerate(transcript.get("words") or []):
        text = str(word.get("text", "")).strip()
        if not text:
            continue

        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        if end < start:
            end = start

        phrases.append(
            {
                "index": len(phrases),
                "word_index": index,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    return phrases


def _extract_sentence_phrases(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Whisper word timestamps into sentence-level timeline entries."""

    words = transcript.get("words") or []
    if not isinstance(words, list) or not words:
        return _extract_segment_phrases(transcript)

    phrases: List[Dict[str, Any]] = []
    current_words: List[str] = []
    current_segment_ids: set[int] = set()
    sentence_start: Optional[float] = None
    sentence_end: Optional[float] = None

    segments = transcript.get("segments") or []
    segment_ranges: List[tuple[int, int, int]] = []
    for segment in segments:
        segment_ranges.append(
            (
                int(segment.get("id", 0)),
                int(segment.get("word_start", 0)),
                int(segment.get("word_end", 0)),
            )
        )

    def append_sentence() -> None:
        nonlocal current_words, current_segment_ids, sentence_start, sentence_end
        text = " ".join(current_words).strip()
        if not text or sentence_start is None or sentence_end is None:
            current_words = []
            current_segment_ids = set()
            sentence_start = None
            sentence_end = None
            return

        phrases.append(
            {
                "index": len(phrases),
                "start": sentence_start,
                "end": sentence_end,
                "text": text,
                "source_segment_ids": sorted(current_segment_ids),
            }
        )
        current_words = []
        current_segment_ids = set()
        sentence_start = None
        sentence_end = None

    for word_index, word in enumerate(words):
        text = str(word.get("text", "")).strip()
        if not text:
            continue

        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        if sentence_start is None:
            sentence_start = start
        sentence_end = end if end >= start else start
        current_words.append(text)

        for segment_id, word_start, word_end in segment_ranges:
            if word_start <= word_index < word_end:
                current_segment_ids.add(segment_id)

        if _is_sentence_end(text):
            append_sentence()

    append_sentence()
    return phrases


def extract_phrase_timeline(transcript: Dict[str, Any], split: str = "segments") -> List[Dict[str, Any]]:
    """Convert a transcript into timeline entries using the requested granularity."""

    if split == "segments":
        return _extract_segment_phrases(transcript)
    if split == "sentences":
        return _extract_sentence_phrases(transcript)
    if split == "words":
        return _extract_word_phrases(transcript)
    raise ValueError(f"Unsupported phrase split mode: {split}")
