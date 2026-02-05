from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_audio_duration_seconds(audio_path: str) -> float:
    """Read duration using ffprobe for determinism."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        audio_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(p.stdout)
    dur = float(data["format"]["duration"])
    return dur


def build_timeline(
    matches: List[Dict[str, Any]],
    audio_path: str,
    fps: int = 30,
    matches_are_phrases: bool = False,
) -> Dict[str, Any]:
    """Create a non-overlapping, strictly chronological timeline.

    Timeline item format:
      {"start": 0.0, "end": 3.2, "image": "01.png", "source": {...}}

    Rules:
    - Each image starts at its matched segment_start
    - It remains until next image starts
    - Last image remains until audio end
    - Strictly chronological & non-overlapping (end is clamped)

    We also quantize timestamps to frame boundaries for deterministic FFmpeg rendering.
    """

    audio_dur = _get_audio_duration_seconds(audio_path)

    def q(t: float) -> float:
        # quantize to nearest frame boundary for stable results
        frame = 1.0 / float(fps)
        # round to nearest frame
        return round(t / frame) * frame

    items: List[Dict[str, Any]] = []

    if not matches:
        return {
            "audio": {"path": str(Path(audio_path)), "duration": audio_dur},
            "fps": fps,
            "items": items,
        }

    # Order is significant. In the new model `matches` is already in mapping.json order.
    for i, m in enumerate(matches):
        start_key = "start" if matches_are_phrases else "segment_start"
        start = q(float(m[start_key]))
        if start < 0:
            start = 0.0
        if start > audio_dur:
            continue

        next_start: Optional[float] = None
        if i + 1 < len(matches):
            next_start = q(float(matches[i + 1][start_key]))

        end = q(audio_dur) if next_start is None else next_start
        if end <= start:
            # enforce non-overlap by ensuring at least 1 frame
            end = min(q(start + (1.0 / fps)), q(audio_dur))

        if matches_are_phrases:
            items.append(
                {
                    "start": start,
                    "end": end,
                    "image": m["image"],
                    "effects": m.get("effects") or {},
                    "source": {
                        "phrase_index": m.get("index"),
                        "phrase_text": m.get("text"),
                        "similarity": m.get("similarity"),
                        "matched_window_text": (m.get("match") or {}).get("matched_window_text"),
                    },
                }
            )
        else:
            items.append(
                {
                    "start": start,
                    "end": end,
                    "image": m["rule"]["image"],
                    "effects": m["rule"].get("effects") or {},
                    "source": {
                        "segment_id": m["segment_id"],
                        "segment_text": m["segment_text"],
                        "similarity": m["similarity"],
                        "matched_text": m["rule"]["text"],
                    },
                }
            )

    # final clamp for safety
    for it in items:
        if it["end"] > audio_dur:
            it["end"] = q(audio_dur)

    # ensure strictly increasing starts (do NOT reorder in phrase mode)
    if not matches_are_phrases:
        items.sort(key=lambda x: x["start"])

    return {
        "audio": {"path": str(Path(audio_path)), "duration": audio_dur},
        "fps": fps,
        "items": items,
    }
