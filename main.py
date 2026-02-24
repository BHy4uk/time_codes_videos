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
  python main.py render --config ./config/mapping.json --audio ./audio.mp3

Environment:
- GOOGLE_API_KEY must be set for `generate`.
- FFmpeg/FFprobe must be installed for `render`.
- Real-ESRGAN executable must be installed for `upscale`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.config_loader import load_mapping_config
from src.env import get_env_path, load_env
from src.gemini_generation import generate_images_from_prompts
from src.phrase_align import resolve_phrase_start_times
from src.render import render_video
from src.timeline import build_timeline
from src.transcribe import transcribe_audio
from src.upscale import upscale_queue


def _check_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Missing required dependency '{name}'.\n"
            f"Please install it and ensure it's on your PATH.\n"
            f"- macOS: brew install ffmpeg\n"
            f"- Ubuntu/Debian: sudo apt-get install ffmpeg\n"
        )


def _cmd_generate(args: argparse.Namespace) -> None:
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


def _cmd_render(args: argparse.Namespace) -> None:
    _check_binary("ffmpeg")
    _check_binary("ffprobe")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    cfg = load_mapping_config(args.config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Transcribe with word timestamps
    transcript = transcribe_audio(
        audio_path=str(audio_path),
        model_size_or_path=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )

    # 2) Resolve phrase start times strictly in mapping order
    resolved_phrases = resolve_phrase_start_times(
        rules=cfg.rules,
        transcript=transcript,
        similarity_threshold=cfg.matching.similarity_threshold,
    )

    # segments.json in the new model = resolved mapping phrases
    (out_dir / "segments.json").write_text(
        __import__("json").dumps({"audio": str(audio_path), "phrases": resolved_phrases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3) Build timeline strictly from phrase order
    timeline = build_timeline(
        matches=resolved_phrases,
        audio_path=str(audio_path),
        fps=args.fps,
        matches_are_phrases=True,
    )
    (out_dir / "timeline.json").write_text(
        __import__("json").dumps(timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4) Render (supports image + video)
    render_video(
        timeline=timeline,
        assets_dir=args.assets,
        audio_path=str(audio_path),
        out_path=str(out_dir / "output.mp4"),
        width=args.width,
        height=args.height,
        fps=args.fps,
        on_short_video=cfg.render.on_short_video,
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

    # render
    r = sub.add_parser("render", help="Render final video from mapping.json + audio")
    r.add_argument("--config", required=True, help="Path to mapping.json")
    r.add_argument("--audio", required=True, help="Path to audio file (.mp3 or .wav)")
    r.add_argument("--assets", default="./img", help="Assets folder containing images/videos referenced in mapping")
    r.add_argument("--out", default="./out", help="Output folder")

    r.add_argument("--model", default="base", help="faster-whisper model size or path (default: base)")
    r.add_argument("--device", default="cpu", help="Device for faster-whisper: cpu/cuda (default: cpu)")
    r.add_argument("--compute-type", default="int8", help="Compute type for faster-whisper (default: int8)")
    r.add_argument("--language", default=None, help="Force language code (e.g., en). Default: auto-detect")

    r.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    r.add_argument("--width", type=int, default=1920, help="Video width (default: 1920)")
    r.add_argument("--height", type=int, default=1080, help="Video height (default: 1080)")
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
