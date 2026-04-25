#!/usr/bin/env python3
"""Deterministic media production pipeline (CLI).

Pipeline stages:

prompts
  ↓
Gemini image generation
  ↓
generated
  ↓
upscale_queue
  ↓
Real-ESRGAN
  ↓
img
  ↓
mapping.json
  ↓
phrase alignment (timestamps)
  ↓
timeline
  ↓
render

Key principles:
- mapping.json is the single source of truth for ordering and phrase triggers.
- No auto-reordering of mapping rules.
- No heuristic timing redistribution.
- Fail fast on missing dependencies.
- Log each stage outputs.

Commands:
  python main.py generate --prompts ./prompts --count 3 --seed 42
  python main.py upscale --scale 4
    python main.py phrases --input ./audio.mp3 --out ./out --lang auto
    python main.py timeline --config ./config/mapping.json --audio ./audio.mp3 --lang auto
  python main.py render --config ./config/mapping.json --audio ./audio.mp3
    python main.py render --timeline ./out/timeline.json --audio ./audio.mp3

Environment:
- GOOGLE_API_KEY must be set for `generate`.
- FFmpeg/FFprobe must be installed for `render`.
- Real-ESRGAN executable must be installed for `upscale`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from src.config_loader import load_mapping_config
from src.env import get_env_path, load_env


LANG_CHOICES = ("en", "es", "auto")


def _check_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Missing required dependency '{name}'.\n"
            f"Please install it and ensure it's on your PATH.\n"
            f"- macOS: brew install ffmpeg\n"
            f"- Ubuntu/Debian: sudo apt-get install ffmpeg\n"
        )


def _resolve_transcription_language(lang: str | None) -> str | None:
    if lang in (None, "auto"):
        return None
    if lang not in {"en", "es"}:
        raise SystemExit(f"Unsupported language {lang!r}. Allowed values: en, es, auto.")
    return lang


def _build_timeline_artifacts(args: argparse.Namespace) -> tuple[object, dict, Path, Path]:
    try:
        from src.phrase_align import resolve_phrase_start_times
        from src.timeline import build_timeline
        from src.transcribe import transcribe_audio
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing Python dependency for `timeline`: "
            f"{e.name}\n"
            "Install render dependencies in your active environment and retry:\n"
            "  pip install rapidfuzz faster-whisper pillow"
        ) from e

    _check_binary("ffprobe")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    cfg = load_mapping_config(args.config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript = transcribe_audio(
        audio_path=str(audio_path),
        model_size_or_path=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=_resolve_transcription_language(args.lang),
        vad_filter=args.vad_filter,
        vad_min_silence_ms=args.vad_min_silence_ms,
    )

    resolved_phrases = resolve_phrase_start_times(
        rules=cfg.rules,
        transcript=transcript,
        similarity_threshold=cfg.matching.similarity_threshold,
    )

    (out_dir / "segments.json").write_text(
        json.dumps({"audio": str(audio_path), "phrases": resolved_phrases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    timeline = build_timeline(
        matches=resolved_phrases,
        audio_path=str(audio_path),
        fps=args.fps,
        matches_are_phrases=True,
    )
    timeline["render"] = {"on_short_video": cfg.render.on_short_video}

    (out_dir / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return cfg, timeline, audio_path, out_dir


def _cmd_phrases(args: argparse.Namespace) -> None:
    try:
        from src.transcribe import extract_phrase_timeline, transcribe_audio
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing Python dependency for `phrases`: "
            f"{e.name}\n"
            "Install transcription dependencies in your active environment and retry:\n"
            "  pip install faster-whisper"
        ) from e

    media_path = Path(args.input)
    if not media_path.exists():
        raise SystemExit(f"Input media file not found: {media_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript = transcribe_audio(
        audio_path=str(media_path),
        model_size_or_path=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=_resolve_transcription_language(args.lang),
        vad_filter=args.vad_filter,
        vad_min_silence_ms=args.vad_min_silence_ms,
    )
    phrases = extract_phrase_timeline(transcript)

    (out_dir / "phrases.json").write_text(
        json.dumps(
            {
                "input": str(media_path),
                "language": transcript.get("language"),
                "duration": transcript.get("duration"),
                "phrases": phrases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Done (phrases)")
    print(f"- Out: {out_dir}")
    print("- Wrote: phrases.json")


def _cmd_generate(args: argparse.Namespace) -> None:
    try:
        from src.gemini_generation import generate_images_from_prompts
    except ModuleNotFoundError as e:
        if e.name in {"google", "google.genai"}:
            raise SystemExit(
                "Missing optional dependency for `generate`: google-genai\n"
                "Install it in your environment and retry:\n"
                "  pip install google-genai"
            ) from e
        raise

    out_generated = Path(args.generated)
    out_queue = Path(args.upscale_queue)
    out_generated.mkdir(parents=True, exist_ok=True)
    out_queue.mkdir(parents=True, exist_ok=True)

    results = generate_images_from_prompts(
        prompts_dir=args.prompts,
        out_dir=str(out_generated),
        count=args.count,
        model=args.model,
        seed=args.seed,
        reference_image=args.reference,
    )

    # Copy generated images to upscale_queue (preserve names, no overwrite)
    copied = []
    for r in results:
        for f in r.output_files:
            src = Path(f)
            dst = out_queue / src.name
            if dst.exists():
                raise SystemExit(f"Refusing to overwrite in upscale_queue: {dst}")
            dst.write_bytes(src.read_bytes())
            copied.append(str(dst))

    print("Done (generate)")
    print(f"- Generated: {out_generated}")
    print(f"- Enqueued for upscale: {out_queue} ({len(copied)} files)")


def _cmd_upscale(args: argparse.Namespace) -> None:
    from src.upscale import upscale_queue

    # Resolve Real-ESRGAN executable path (optional env override)
    realesrgan_arg = args.realesrgan
    if not realesrgan_arg:
        env_p = get_env_path("REALESRGAN_PATH")
        if env_p:
            realesrgan_arg = str(env_p)

    outputs = upscale_queue(
        upscale_queue_dir=args.upscale_queue,
        upscaled_dir=args.upscaled,
        img_dir=args.img,
        scale=args.scale,
        realesrgan_exe=realesrgan_arg,
    )

    print("Done (upscale)")
    print(f"- Processed: {len(outputs)}")
    print(f"- Output img/: {args.img}")


def _cmd_timeline(args: argparse.Namespace) -> None:
    _build_timeline_artifacts(args)

    print("Done (timeline)")
    print(f"- Out: {args.out}")
    print("- Wrote: segments.json, timeline.json")


def _cmd_render(args: argparse.Namespace) -> None:
    try:
        from src.render import render_video
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing Python dependency for `render`: "
            f"{e.name}\n"
            "Install render dependencies in your active environment and retry:\n"
            "  pip install rapidfuzz faster-whisper pillow"
        ) from e

    _check_binary("ffmpeg")
    _check_binary("ffprobe")

    if not args.timeline and not args.config:
        raise SystemExit("render requires either --timeline or --config")

    if args.timeline:
        timeline_path = Path(args.timeline)
        if not timeline_path.exists():
            raise SystemExit(f"Timeline file not found: {timeline_path}")
        audio_path = Path(args.audio)
        if not audio_path.exists():
            raise SystemExit(f"Audio file not found: {audio_path}")
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        timeline = json.loads(timeline_path.read_text(encoding="utf-8-sig"))
        on_short_video = (
            ((timeline.get("render") or {}).get("on_short_video"))
            or (load_mapping_config(args.config).render.on_short_video if args.config else None)
            or "freeze"
        )
    else:
        cfg, timeline, audio_path, out_dir = _build_timeline_artifacts(args)
        on_short_video = cfg.render.on_short_video

    # 4) Render (supports image + video)
    render_video(
        timeline=timeline,
        assets_dir=args.assets,
        audio_path=str(audio_path),
        out_path=str(out_dir / "output.mp4"),
        width=args.width,
        height=args.height,
        fps=args.fps,
        on_short_video=on_short_video,
        debug=args.debug_render,
        work_dir=str(out_dir / "_render_work"),
    )

    print("Done (render)")
    print(f"- Out: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic media production pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # generate
    g = sub.add_parser("generate", help="Generate images from prompts using Gemini")
    g.add_argument("--prompts", default="./prompts", help="Folder containing .txt prompt files")
    g.add_argument("--generated", default="./generated", help="Output folder for raw generated images")
    g.add_argument("--upscale-queue", default="./upscale_queue", help="Folder to enqueue images for upscaling")
    g.add_argument("--count", type=int, required=True, help="Number of images per prompt")
    g.add_argument("--seed", type=int, default=None, help="Optional seed (deterministic if supported by model)")
    g.add_argument("--reference", default=None, help="Optional reference image path")
    g.add_argument(
        "--model",
        default="gemini-2.5-flash-image",
        help="Gemini image model id (AI Studio). Example: gemini-2.5-flash-image",
    )
    g.set_defaults(func=_cmd_generate)

    # upscale
    u = sub.add_parser("upscale", help="Upscale all images in upscale_queue using Real-ESRGAN")
    u.add_argument("--upscale-queue", default="./upscale_queue", help="Input folder")
    u.add_argument("--upscaled", default="./upscaled", help="Intermediate output folder")
    u.add_argument("--img", default="./img", help="Final images folder")
    u.add_argument("--scale", type=int, required=True, choices=[2, 4], help="Upscale factor (2 or 4)")
    u.add_argument(
        "--realesrgan",
        default=None,
        help="Optional path to realesrgan-ncnn-vulkan.exe (else uses REALESRGAN_PATH env or default path)",
    )
    u.set_defaults(func=_cmd_upscale)

    # phrases
    ph = sub.add_parser("phrases", help="Transcribe audio or video and write phrase timestamps without mapping")
    ph.add_argument("--input", required=True, help="Path to input audio or video file")
    ph.add_argument("--out", default="./out", help="Output folder")
    ph.add_argument("--model", default="base", help="faster-whisper model size or path (default: base)")
    ph.add_argument("--device", default="cpu", help="Device for faster-whisper: cpu/cuda (default: cpu)")
    ph.add_argument("--compute-type", default="int8", help="Compute type for faster-whisper (default: int8)")
    ph.add_argument(
        "--lang",
        "--language",
        dest="lang",
        choices=LANG_CHOICES,
        default="auto",
        help="Transcription language: en, es, or auto (default: auto)",
    )
    ph.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable VAD filter during transcription (may shift early timestamps). Default: OFF.",
    )
    ph.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="VAD min silence duration (ms) if VAD is enabled. Default: 500.",
    )
    ph.set_defaults(func=_cmd_phrases)

    # timeline
    t = sub.add_parser("timeline", help="Transcribe audio, resolve phrase starts, and write timeline artifacts")
    t.add_argument("--config", required=True, help="Path to mapping.json")
    t.add_argument("--audio", required=True, help="Path to audio file (.mp3 or .wav)")
    t.add_argument("--assets", default="./img", help="Assets folder (accepted for workflow symmetry; not used here)")
    t.add_argument("--out", default="./out", help="Output folder")
    t.add_argument("--model", default="base", help="faster-whisper model size or path (default: base)")
    t.add_argument("--device", default="cpu", help="Device for faster-whisper: cpu/cuda (default: cpu)")
    t.add_argument("--compute-type", default="int8", help="Compute type for faster-whisper (default: int8)")
    t.add_argument(
        "--lang",
        "--language",
        dest="lang",
        choices=LANG_CHOICES,
        default="auto",
        help="Transcription language: en, es, or auto (default: auto)",
    )
    t.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable VAD filter during transcription (may shift early timestamps). Default: OFF.",
    )
    t.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="VAD min silence duration (ms) if VAD is enabled. Default: 500.",
    )
    t.add_argument("--fps", type=int, default=30, help="Timeline FPS for deterministic quantization (default: 30)")
    t.set_defaults(func=_cmd_timeline)

    # render
    r = sub.add_parser("render", help="Render final video from mapping.json + audio")
    r.add_argument("--config", help="Path to mapping.json")
    r.add_argument("--timeline", help="Path to pre-generated timeline.json")
    r.add_argument("--audio", required=True, help="Path to audio file (.mp3 or .wav)")
    r.add_argument("--assets", default="./img", help="Assets folder containing images/videos referenced in mapping")
    r.add_argument("--out", default="./out", help="Output folder")

    r.add_argument("--model", default="base", help="faster-whisper model size or path (default: base)")
    r.add_argument("--device", default="cpu", help="Device for faster-whisper: cpu/cuda (default: cpu)")
    r.add_argument("--compute-type", default="int8", help="Compute type for faster-whisper (default: int8)")
    r.add_argument(
        "--lang",
        "--language",
        dest="lang",
        choices=LANG_CHOICES,
        default="auto",
        help="Transcription language when generating timeline internally: en, es, or auto (default: auto)",
    )

    # Timing-critical: VAD can shift the first detected timestamps later.
    r.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable VAD filter during transcription (may shift early timestamps). Default: OFF.",
    )
    r.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="VAD min silence duration (ms) if VAD is enabled. Default: 500.",
    )

    r.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    r.add_argument("--width", type=int, default=1920, help="Video width (default: 1920)")
    r.add_argument("--height", type=int, default=1080, help="Video height (default: 1080)")
    r.add_argument(
        "--debug-render",
        action="store_true",
        help="Verbose render logs (scene durations, frame counts, ffmpeg commands, ffprobe durations).",
    )
    r.set_defaults(func=_cmd_render)

    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
