#!/usr/bin/env python3
"""Windows helper runner for a single working folder.

Workflow:
1) Ask for a working folder name under D:\Youtube\Work (e.g., video)
2) Ensure ./out and ./config exist inside that folder
3) Ensure mapping.json exists (create a template if missing)
4) Run phrases extraction only
5) Ask for confirmation, then run timeline generation and render

This script is optional convenience tooling. It does not change the deterministic
matching/rendering behavior in main.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(r"D:\Youtube\Work")
REPO_ROOT = Path(__file__).resolve().parent


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
                "asset": "01.png",
                "type": "image",
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
    print("=== Working Folder Runner ===")
    folder_name = _prompt("Enter working folder", "video")

    work_dir = BASE_DIR / folder_name
    audio_path = work_dir / "audio" / "audio.mp3"
    assets_dir = work_dir / "img"
    out_dir = work_dir / "out"
    config_dir = work_dir / "config"

    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")
    if not assets_dir.exists():
        print(f"Warning: assets folder not found yet: {assets_dir}")

    _ensure_dir(out_dir)
    _ensure_dir(config_dir)
    mapping_path = _ensure_mapping_json(config_dir)

    print("\nPaths:")
    print(f"- Audio:   {audio_path}")
    print(f"- Assets:  {assets_dir}")
    print(f"- Config:  {mapping_path}")
    print(f"- Out:     {out_dir}")

    # 1) Extract phrases only
    print("\n[1/3] Phrase extraction starting...")
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "phrases",
            "--input",
            str(audio_path),
            "--out",
            str(out_dir),
        ]
    )
    print("[1/3] Phrase extraction complete.")
    print(f"Phrases saved to: {out_dir / 'phrases.json'}")

    # Ask for confirmation before timeline/render
    print("\nNow you can update mapping.json based on phrases.json.")
    answer = input("Run timeline and render now? (y/N): ").strip().lower()
    if answer not in ("y", "yes"):
        print("Stopped. You can run the timeline/render steps later by re-running this script.")
        return

    # 2) Timeline
    print("\n[2/3] Timeline generation starting...")
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "timeline",
            "--audio",
            str(audio_path),
            "--config",
            str(mapping_path),
            "--assets",
            str(assets_dir),
            "--out",
            str(out_dir),
        ]
    )
    print("[2/3] Timeline generation complete.")

    # 3) Render
    print("\n[3/3] Render starting...")
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "render",
            "--audio",
            str(audio_path),
            "--config",
            str(mapping_path),
            "--assets",
            str(assets_dir),
            "--out",
            str(out_dir),
        ]
    )
    print("[3/3] Render complete.")
    print(f"Video: {out_dir / 'output.mp4'}")


if __name__ == "__main__":
    main()
