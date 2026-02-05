#!/usr/bin/env python3
"""CLI entry point.

Deterministic pipeline:
Audio -> timestamped transcription -> full-phrase fuzzy matching -> timeline -> rendered MP4

Example:
  python main.py --audio audio.mp3 --images ./images --config mapping.json --out ./out
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from src.config import load_config
from src.phrase_align import resolve_phrase_start_times
from src.render import render_video
from src.timeline import build_timeline
from src.transcribe import transcribe_audio


def _check_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Missing required dependency '{name}'.\n"
            f"Please install it and ensure it's on your PATH.\n"
            f"- macOS: brew install ffmpeg\n"
            f"- Ubuntu/Debian: sudo apt-get install ffmpeg\n"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a video by syncing images to spoken phrases.")

    p.add_argument("--audio", required=True, help="Path to audio file (.mp3 or .wav)")
    # In full pipeline mode these are required, but for --transcribe-only they are optional.
    p.add_argument("--images", required=False, help="Folder containing images referenced by config")
    p.add_argument("--config", required=False, help="JSON config mapping full phrases to images")
    p.add_argument("--out", default="./out", help="Output folder")

    p.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only produce segments.json (no config/images/ffmpeg required).",
    )

    p.add_argument("--model", default="base", help="faster-whisper model size or path (default: base)")
    p.add_argument("--device", default="cpu", help="Device for faster-whisper: cpu/cuda (default: cpu)")
    p.add_argument("--compute-type", default="int8", help="Compute type for faster-whisper (default: int8)")
    p.add_argument("--language", default=None, help="Force language code (e.g., en). Default: auto-detect")

    p.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    p.add_argument("--width", type=int, default=1920, help="Video width (default: 1920)")
    p.add_argument("--height", type=int, default=1080, help="Video height (default: 1080)")

    p.add_argument("--segments-json", default=None, help="Optional override path for segments.json")
    p.add_argument("--timeline-json", default=None, help="Optional override path for timeline.json")
    p.add_argument("--output-video", default=None, help="Optional override path for output.mp4")

    # Note: mapping.json is the single source of truth. We do NOT split/rewrite phrases.

    return p.parse_args()


def main() -> None:
    args = parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out)

    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    segments_json_path = Path(args.segments_json) if args.segments_json else (out_dir / "segments.json")
    timeline_json_path = Path(args.timeline_json) if args.timeline_json else (out_dir / "timeline.json")
    output_video_path = Path(args.output_video) if args.output_video else (out_dir / "output.mp4")

    # 1) Transcribe (always)
    transcription = transcribe_audio(
        audio_path=str(audio_path),
        model_size_or_path=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )

    raw_segments = transcription.get("segments") or []
    if args.no_sentence_refine:
        segments_for_matching = raw_segments
    else:
        segments_for_matching = refine_segments_sentence_split(raw_segments)

    # Write both raw and refined segments for transparency.
    out_transcription = dict(transcription)
    out_transcription["raw_segments"] = raw_segments
    out_transcription["segments"] = segments_for_matching

    segments_json_path.write_text(json.dumps(out_transcription, ensure_ascii=False, indent=2), encoding="utf-8")

    # If user only wants transcription, stop here.
    if args.transcribe_only:
        print("Done (transcribe-only)")
        print(f"- Segments: {segments_json_path}")
        return

    # For full pipeline we need config/images and ffmpeg.
    if not args.images:
        raise SystemExit("--images is required unless --transcribe-only is set")
    if not args.config:
        raise SystemExit("--config is required unless --transcribe-only is set")

    images_dir = Path(args.images)
    config_path = Path(args.config)

    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Images folder not found or not a directory: {images_dir}")
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    _check_binary("ffmpeg")
    _check_binary("ffprobe")

    # 2) Load config
    cfg = load_config(str(config_path))

    # 3) Match phrases
    matches = match_segments_to_rules(
        segments=segments_for_matching,
        rules=cfg.rules,
        similarity_threshold=cfg.matching.similarity_threshold,
    )

    # 4) Build timeline (non-overlapping, strictly chronological)
    timeline = build_timeline(
        matches=matches,
        audio_path=str(audio_path),
        fps=args.fps,
    )
    timeline_json_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) Render
    render_video(
        timeline=timeline,
        images_dir=str(images_dir),
        audio_path=str(audio_path),
        out_path=str(output_video_path),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    print("Done")
    print(f"- Segments: {segments_json_path}")
    print(f"- Timeline: {timeline_json_path}")
    print(f"- Video:    {output_video_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
