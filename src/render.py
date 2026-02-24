from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .effects import build_effects_filter


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi")


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")


def render_video(
    timeline: Dict[str, Any],
    assets_dir: str,
    audio_path: str,
    out_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    on_short_video: str = "freeze",
) -> None:
    """Render MP4 with FFmpeg (supports image and video assets).

    Deterministic approach:
    - Each timeline item becomes one FFmpeg input.
    - Each input is processed into a fixed 1920x1080 stream (scale-to-fit + pad + SAR=1).
    - Effects are applied after normalization.
    - Streams are concatenated with `concat` filter.

    Asset behaviors:
    - Image: loop still frame; duration enforced via trim.
    - Video:
      - If longer than interval: trimmed.
      - If shorter:
        - on_short_video=freeze -> last frame frozen.
        - on_short_video=loop   -> loop video.

    Notes:
    - Requires FFmpeg installed.
    """

    if on_short_video not in ("freeze", "loop"):
        raise ValueError("on_short_video must be 'freeze' or 'loop'")

    items = timeline.get("items") or []
    if not items:
        raise ValueError("Timeline has no items; nothing to render")

    assets_p = Path(assets_dir)
    audio_p = Path(audio_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    input_args: List[str] = []
    durations: List[float] = []
    effects_debug: List[Dict[str, Any]] = []

    per_stream_filters: List[str] = []
    concat_inputs: List[str] = []

    input_index = 0

    for it in items:
        asset_name = it.get("asset") or it.get("image")
        if not asset_name:
            raise ValueError("Timeline item missing asset")

        asset_path = assets_p / str(asset_name)
        if not asset_path.exists():
            raise FileNotFoundError(f"Asset referenced by timeline not found: {asset_path}")

        dur = float(it["end"]) - float(it["start"])
        if dur <= 0:
            continue

        durations.append(dur)

        asset_arg = str(asset_path).replace("\\", "/")

        asset_type = str(it.get("type") or ("video" if _is_video(asset_path) else "image"))

        # Input args per asset type
        if asset_type == "image":
            if not _is_image(asset_path):
                raise ValueError(f"Asset declared as image but extension not supported: {asset_path}")
            input_args.extend(["-loop", "1", "-i", asset_arg])
        elif asset_type == "video":
            if not _is_video(asset_path):
                raise ValueError(f"Asset declared as video but extension not supported: {asset_path}")
            if on_short_video == "loop":
                # Loop video input indefinitely
                input_args.extend(["-stream_loop", "-1", "-i", asset_arg])
            else:
                input_args.extend(["-i", asset_arg])
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")

        effects = it.get("effects") if isinstance(it.get("effects"), dict) else {}

        # Provide original image size to normalize focus coords (only meaningful for images).
        orig_size: Optional[tuple[int, int]] = None
        if asset_type == "image":
            try:
                with Image.open(asset_path) as im:
                    orig_size = im.size
            except Exception:
                orig_size = None

        vf, dbg = build_effects_filter(
            effects=effects,
            width=width,
            height=height,
            fps=fps,
            duration=dur,
            source_size=orig_size,
        )

        effects_debug.append(
            {
                "asset": str(asset_name),
                "type": asset_type,
                "duration": dur,
                "effects": effects,
                "debug": dbg,
            }
        )

        # For video, ensure we have a video stream. Some videos may be variable fps.
        # We always normalize+fps+trim in build_effects_filter.

        per_stream_filters.append(f"[{input_index}:v]setpts=PTS-STARTPTS,{vf},setsar=1[v{input_index}]")
        concat_inputs.append(f"[v{input_index}]")
        input_index += 1

    if not durations:
        raise ValueError("Timeline items have non-positive durations; nothing to render")

    n = len(durations)
    filter_complex = ";".join(per_stream_filters + [f"{''.join(concat_inputs)}concat=n={n}:v=1:a=0[vout]"])

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *input_args,
        "-i",
        str(audio_p).replace("\\", "/"),
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
        str(out_p).replace("\\", "/"),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {p.stdout}\n"
            f"stderr: {p.stderr}\n"
        )

    manifest = {
        "ffmpeg_cmd": cmd,
        "timeline": timeline,
        "timeline_durations": [
            {
                "start": float(it["start"]),
                "end": float(it["end"]),
                "duration": float(it["end"]) - float(it["start"]),
            }
            for it in (timeline.get("items") or [])
        ],
        "effects_debug": effects_debug,
        "on_short_video": on_short_video,
    }

    (out_p.parent / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
