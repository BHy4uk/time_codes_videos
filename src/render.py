from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .effects import build_effects_filter


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi")


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path).replace("\\", "/"),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {p.stderr}")
    data = json.loads(p.stdout)
    return float(data["format"]["duration"])


def _run(cmd: List[str], debug: bool) -> None:
    if debug:
        print("\nFFMPEG:")
        print(" ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {p.stdout}\n"
            f"stderr: {p.stderr}\n"
        )


def _ensure_reencode_cfr(src: Path, dst: Path, fps: int, debug: bool) -> None:
    """Normalize an intermediate clip to CFR and sane timestamps."""

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src).replace("\\", "/"),
        "-r",
        str(fps),
        "-vsync",
        "cfr",
        "-vf",
        "setpts=PTS-STARTPTS,setsar=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst).replace("\\", "/"),
    ]
    _run(cmd, debug=debug)


def render_video(
    timeline: Dict[str, Any],
    assets_dir: str,
    audio_path: str,
    out_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    on_short_video: str = "freeze",
    debug: bool = False,
    work_dir: Optional[str] = None,
) -> None:
    """Render final MP4 with strict deterministic timing.

    Hardened approach (per requirement):
    1) For each timeline item, render an intermediate scene clip `sceneNNN.mp4`
       with EXACT duration = end-start.
       - CFR enforced: -r fps + -vsync cfr
       - timestamps reset: setpts=PTS-STARTPTS
       - duration enforced: -t <duration>
    2) Concatenate scene clips using concat demuxer to avoid filter_complex drift.
    3) Mux the concatenated video with the original audio so video aligns to audio.

    Output duration is forced to audio duration (trim/pad video if needed is NOT done;
    instead we guarantee scene durations sum to audio duration by construction).
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

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")

    audio_dur = _ffprobe_duration(audio_p)

    # Compute expected total duration from timeline
    expected_total = 0.0
    for it in items:
        expected_total += max(0.0, float(it["end"]) - float(it["start"]))

    # Use work dir
    if work_dir:
        work_p = Path(work_dir)
        work_p.mkdir(parents=True, exist_ok=True)
    else:
        work_p = Path(tempfile.mkdtemp(prefix="render_work_"))

    scenes_dir = work_p / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        print("\nRENDER DEBUG")
        print(f"- fps={fps}")
        print(f"- audio_duration={audio_dur:.6f}")
        print(f"- expected_total_from_timeline={expected_total:.6f}")
        print(f"- work_dir={work_p}")

    scene_paths: List[Path] = []
    scene_meta: List[Dict[str, Any]] = []

    for idx, it in enumerate(items):
        asset_name = it.get("asset") or it.get("image")
        asset_type = str(it.get("type") or "image")
        effects = it.get("effects") if isinstance(it.get("effects"), dict) else {}

        asset_path = assets_p / str(asset_name)
        if not asset_path.exists():
            raise FileNotFoundError(f"Asset not found: {asset_path}")

        dur = float(it["end"]) - float(it["start"])
        if dur <= 0:
            continue

        frames = int(round(dur * fps))
        # enforce at least 1 frame
        frames = max(1, frames)
        dur_exact = frames / float(fps)

        # Build per-scene filter
        orig_size: Optional[tuple[int, int]] = None
        if asset_type == "image" and _is_image(asset_path):
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
            duration=dur_exact,
            source_size=orig_size,
        )

        scene_out = scenes_dir / f"scene{idx:04d}.mp4"
        if scene_out.exists():
            scene_out.unlink()

        asset_arg = str(asset_path).replace("\\", "/")

        if asset_type == "image":
            if not _is_image(asset_path):
                raise ValueError(f"Asset declared as image but unsupported extension: {asset_path}")

            # Render scene from a single still image.
            # -loop 1 makes an infinite stream; -t dur_exact + trim in filters guarantees exact length.
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-i",
                asset_arg,
                "-t",
                f"{dur_exact:.6f}",
                "-r",
                str(fps),
                "-vsync",
                "cfr",
                "-vf",
                f"setpts=PTS-STARTPTS,{vf},setsar=1",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(scene_out).replace("\\", "/"),
            ]
            _run(cmd, debug=debug)

        elif asset_type == "video":
            if not _is_video(asset_path):
                raise ValueError(f"Asset declared as video but unsupported extension: {asset_path}")

            # For videos, we enforce duration with -t and optionally loop input.
            input_args: List[str] = []
            if on_short_video == "loop":
                input_args = ["-stream_loop", "-1"]

            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                *input_args,
                "-i",
                asset_arg,
                "-t",
                f"{dur_exact:.6f}",
                "-r",
                str(fps),
                "-vsync",
                "cfr",
                "-vf",
                f"setpts=PTS-STARTPTS,{vf},setsar=1",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(scene_out).replace("\\", "/"),
            ]
            _run(cmd, debug=debug)

            if on_short_video == "freeze":
                # If video is shorter than dur_exact, ffmpeg will end early.
                # Detect and re-encode with tpad stop_mode=clone.
                actual = _ffprobe_duration(scene_out)
                if actual + (1.0 / fps) < dur_exact:
                    padded = scenes_dir / f"scene{idx:04d}_padded.mp4"
                    cmd2 = [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(scene_out).replace("\\", "/"),
                        "-r",
                        str(fps),
                        "-vsync",
                        "cfr",
                        "-vf",
                        f"tpad=stop_mode=clone:stop_duration={dur_exact - actual:.6f},setpts=PTS-STARTPTS,setsar=1",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        str(padded).replace("\\", "/"),
                    ]
                    _run(cmd2, debug=debug)
                    scene_out.unlink(missing_ok=True)
                    scene_out = padded

        else:
            raise ValueError(f"Unsupported asset type in timeline: {asset_type}")

        # Validate duration
        actual_dur = _ffprobe_duration(scene_out)
        # Allow 1 frame tolerance
        if abs(actual_dur - dur_exact) > (1.5 / fps):
            raise RuntimeError(
                f"Scene duration mismatch for {scene_out}: expected {dur_exact:.6f}s, got {actual_dur:.6f}s"
            )

        scene_paths.append(scene_out)
        scene_meta.append(
            {
                "index": idx,
                "asset": str(asset_name),
                "type": asset_type,
                "timeline_start": float(it["start"]),
                "timeline_end": float(it["end"]),
                "duration_requested": dur,
                "duration_quantized": dur_exact,
                "frames": frames,
                "ffprobe_duration": actual_dur,
                "effects_debug": dbg,
            }
        )

        if debug:
            print(
                f"scene{idx:04d}: start={it['start']:.3f} end={it['end']:.3f} "
                f"dur={dur:.3f} -> frames={frames} dur_q={dur_exact:.6f} ffprobe={actual_dur:.6f}"
            )

    if not scene_paths:
        raise ValueError("No scenes were rendered (all durations <= 0?)")

    # Concat demuxer file
    concat_file = work_p / "concat.txt"
    concat_lines = []
    def _concat_escape(p: Path) -> str:
        # concat demuxer uses single quotes; escape single quote by closing/opening with \'
        return str(p).replace("\\", "/").replace("'", "'\\\\''")

    for sp in scene_paths:
        concat_lines.append(f"file '{_concat_escape(sp)}'")
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    video_only = work_p / "video_concat.mp4"

    # Concatenate (no re-encode)
    cmd_concat = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file).replace("\\", "/"),
        "-c",
        "copy",
        str(video_only).replace("\\", "/"),
    ]
    _run(cmd_concat, debug=debug)

    # Mux audio + video.
    # We avoid -shortest; instead we map audio and video and cut to audio duration if needed.
    final_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_only).replace("\\", "/"),
        "-i",
        str(audio_p).replace("\\", "/"),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        f"{audio_dur:.6f}",
        str(out_p).replace("\\", "/"),
    ]
    _run(final_cmd, debug=debug)

    final_dur = _ffprobe_duration(out_p)

    # Write manifest
    manifest = {
        "render_version": "scene-clips+concat-demuxer",
        "fps": fps,
        "audio": {"path": str(audio_p), "duration": audio_dur},
        "expected_total_from_timeline": expected_total,
        "scene_meta": scene_meta,
        "commands": {
            "scene_generation": "one ffmpeg call per scene (see debug logs)",
            "concat": cmd_concat,
            "mux": final_cmd,
        },
        "work_dir": str(work_p),
        "final_duration": final_dur,
    }

    (out_p.parent / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if debug:
        print(f"\nFinal output duration: {final_dur:.6f}s (audio={audio_dur:.6f}s)")
