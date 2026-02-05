from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from faster_whisper import WhisperModel


def transcribe_audio(
    audio_path: str,
    model_size_or_path: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe audio with word-level timestamps using faster-whisper.

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
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
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
