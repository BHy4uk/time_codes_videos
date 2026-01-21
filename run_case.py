#!/usr/bin/env python3
"""Windows helper runner for batch processing cases/blocks.

Workflow:
1) Ask for case id (e.g., case_002) and block id (e.g., block_00)
2) Ensure folders exist:
   - D:\State_51\cases\<case>\<block>\out
   - D:\State_51\cases\<case>\<block>\config
3) Ensure mapping.json exists (create a template if missing)
4) Run transcription-only
5) Ask for confirmation, then run full pipeline render

This script is optional convenience tooling. It does not change the deterministic
matching/rendering behavior in main.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(r"D:\State_51\cases")


def _prompt(label: str, example: str) -> str:
    while True:
        v = input(f"{label} (e.g., {example}): ").strip()
        if v:
            return v
        print("Value cannot be empty.")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ensure_mapping_json(config_dir: Path) -> Path:
    mapping_path = config_dir / "mapping.json"
    if mapping_path.exists() and mapping_path.stat().st_size > 0:
        return mapping_path

    template = {
        "rules": [
            {
                "image": "01.png",
                "text": "Full sentence or paragraph ...",
                "effects": {
                    "zoom": {"type": "in", "scale": 1.1, "duration": 4},
                    "motion": {"direction": "right", "intensity": 0.05},
                    "fade": {"type": "in", "duration": 1},
                    "focus": {
                        "source": {"width": 4000, "height": 3000},
                        "target": {"x": 1200, "y": 800, "width": 600, "height": 500},
                    },
                },
            }
        ],
        "matching": {"mode": "full_phrase", "similarity_threshold": 85},
    }

    mapping_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapping_path


def _run(cmd: list[str]) -> None:
    print("\nRUN:")
    print(" ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> None:
    print("=== State_51 Case Runner ===")
    case_id = _prompt("Enter case", "case_002")
    block_id = _prompt("Enter block", "block_00")

    case_dir = BASE_DIR / case_id / block_id
    audio_path = case_dir / "audio" / "audio.mp3"
    images_dir = case_dir / "images"
    out_dir = case_dir / "out"
    config_dir = case_dir / "config"

    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")
    if not images_dir.exists():
        print(f"Warning: images folder not found yet: {images_dir}")

    _ensure_dir(out_dir)
    _ensure_dir(config_dir)
    mapping_path = _ensure_mapping_json(config_dir)

    print("\nPaths:")
    print(f"- Audio:   {audio_path}")
    print(f"- Images:  {images_dir}")
    print(f"- Config:  {mapping_path}")
    print(f"- Out:     {out_dir}")

    # 1) Transcribe only
    print("\n[1/2] Transcription (transcribe-only) starting...")
    _run(
        [
            sys.executable,
            "main.py",
            "--audio",
            str(audio_path),
            "--out",
            str(out_dir),
            "--transcribe-only",
        ]
    )
    print("[1/2] Transcription complete.")
    print(f"Segments saved to: {out_dir / 'segments.json'}")

    # Ask for confirmation before render
    print("\nNow you can update mapping.json based on segments.json.")
    answer = input("Run full pipeline render now? (y/N): ").strip().lower()
    if answer not in ("y", "yes"):
        print("Stopped. You can run the render step later by re-running this script.")
        return

    # 2) Full pipeline
    print("\n[2/2] Full pipeline render starting...")
    _run(
        [
            sys.executable,
            "main.py",
            "--audio",
            str(audio_path),
            "--images",
            str(images_dir),
            "--config",
            str(mapping_path),
            "--out",
            str(out_dir),
        ]
    )
    print("[2/2] Render complete.")
    print(f"Video: {out_dir / 'output.mp4'}")


if __name__ == "__main__":
    main()
