from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .effects import build_effects_filter


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

    Why this implementation (important):
    - On some platforms, FFmpeg's concat demuxer can behave inconsistently with still images
      + `duration` directives (especially on Windows / with CRLF files), producing a video
      that stops early.
    - To guarantee video length deterministically, we instead build the video from a set of
      `-loop 1 -t <duration> -i <image>` inputs and concatenate them via the `concat` filter.

    Result:
    - The video stream duration matches the sum of timeline item durations.
    - With `-shortest`, the final MP4 length matches the audio length (assuming timeline
      ends at audio end, which our timeline builder does).

    Images are scaled/padded to fullscreen (no distortion).
    """

    items = timeline.get("items") or []
    if not items:
        raise ValueError("Timeline has no items; nothing to render")

    images_dir_p = Path(images_dir)
    audio_p = Path(audio_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Validate images & build per-item durations
    input_args: List[str] = []
    durations: List[float] = []
    effects_debug: List[Dict[str, Any]] = []

    # We will build a per-input filter chain and then concat the processed streams.
    per_stream_filters: List[str] = []
    concat_inputs: List[str] = []

    input_index = 0
    for it in items:
        img = images_dir_p / it["image"]
        if not img.exists():
            raise FileNotFoundError(f"Image referenced by timeline not found: {img}")

        dur = float(it["end"]) - float(it["start"])
        if dur <= 0:
            continue

        durations.append(dur)

        # Input options apply to the *next* -i
        input_args.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-t",
                f"{dur:.6f}",
                "-i",
                str(img),
            ]
        )

        effects = it.get("effects") if isinstance(it.get("effects"), dict) else {}
        vf, dbg = build_effects_filter(effects=effects, width=width, height=height, fps=fps, duration=dur)
        effects_debug.append({"image": it.get("image"), "duration": dur, "effects": effects, "debug": dbg})

        per_stream_filters.append(f"[{input_index}:v]setpts=PTS-STARTPTS,{vf}[v{input_index}]")
        concat_inputs.append(f"[v{input_index}]")
        input_index += 1

    if not durations:
        raise ValueError("Timeline items have non-positive durations; nothing to render")

    n = len(durations)

    filter_complex = ";".join(per_stream_filters + [f"{''.join(concat_inputs)}concat=n={n}:v=1:a=0[vout]"])

    # Audio is the last input index
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *input_args,
        "-i",
        str(audio_p),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        f"{n}:a:0",
        "-r",
        str(fps),
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

    # Emit a render manifest for reproducibility/debugging
    manifest = {
        "ffmpeg_cmd": cmd,
        "timeline": timeline,
        "timeline_durations": [
            {
                "start": float(it["start"]),
                "end": float(it["end"]),
                "duration": float(it["end"]) - float(it["start"]),
            }
            for it in items
        ],
    }
    (out_p.parent / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
