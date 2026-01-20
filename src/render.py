from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict


def _ffmpeg_escape(path: str) -> str:
    # concat demuxer expects paths escaped by replacing single quotes
    return path.replace("'", "'\\''")


def render_video(
    timeline: Dict[str, Any],
    images_dir: str,
    audio_path: str,
    out_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> None:
    """Render MP4 with FFmpeg.

    Implementation uses the concat demuxer with per-image durations.

    - Images are scaled/padded to fullscreen (no distortion) using scale+pad.
    - Audio is mapped from input audio file.
    - Video ends exactly at audio end (shortest).
    """

    items = timeline.get("items") or []
    if not items:
        raise ValueError("Timeline has no items; nothing to render")

    images_dir_p = Path(images_dir)
    audio_p = Path(audio_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Build concat file
    with tempfile.TemporaryDirectory() as td:
        concat_path = Path(td) / "images.concat"
        lines = []
        for it in items:
            img = images_dir_p / it["image"]
            if not img.exists():
                raise FileNotFoundError(f"Image referenced by timeline not found: {img}")
            dur = float(it["end"]) - float(it["start"])
            if dur <= 0:
                continue
            # concat demuxer syntax
            lines.append(f"file '{_ffmpeg_escape(str(img))}'")
            lines.append(f"duration {dur:.6f}")

        # Per ffmpeg docs: last file line should be repeated without duration
        last_img = images_dir_p / items[-1]["image"]
        lines.append(f"file '{_ffmpeg_escape(str(last_img))}'")

        concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"format=yuv420p"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-safe",
            "0",
            "-f",
            "concat",
            "-i",
            str(concat_path),
            "-i",
            str(audio_p),
            "-r",
            str(fps),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out_p),
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                "FFmpeg failed.\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout: {p.stdout}\n"
                f"stderr: {p.stderr}\n"
            )

        # Emit a small render manifest for reproducibility/debugging
        manifest = {
            "ffmpeg_cmd": cmd,
            "timeline": timeline,
        }
        (out_p.parent / "render_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
