from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from .logging_utils import append_jsonl, utc_ts, write_json


class UpscaleError(RuntimeError):
    pass


def _default_realesrgan_path() -> Optional[Path]:
    # As per your convention
    p = Path(r"C:\AI\RealESRGAN\realesrgan-ncnn-vulkan.exe")
    return p if p.exists() else None


def _resolve_realesrgan_exe(explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise UpscaleError(f"Real-ESRGAN executable not found at: {p}")
        return p

    default = _default_realesrgan_path()
    if default:
        return default

    which = shutil.which("realesrgan-ncnn-vulkan.exe") or shutil.which("realesrgan-ncnn-vulkan")
    if which:
        return Path(which)

    raise UpscaleError(
        "Real-ESRGAN not found. Provide --realesrgan path or install it at "
        "C:\\AI\\RealESRGAN\\realesrgan-ncnn-vulkan.exe"
    )


def upscale_queue(
    upscale_queue_dir: str,
    upscaled_dir: str,
    img_dir: str,
    scale: int,
    realesrgan_exe: Optional[str] = None,
    log_path: Optional[str] = None,
) -> List[str]:
    """Process all files in /upscale_queue using Real-ESRGAN CLI.

    - Preserves filenames.
    - Writes results to /upscaled.
    - Moves final images to /img.
    - Fails if output file would be overwritten.

    Determinism:
    - No randomness; same input -> same output.
    """

    if scale not in (2, 4):
        raise ValueError("--scale must be 2 or 4")

    exe = _resolve_realesrgan_exe(realesrgan_exe)

    queue_p = Path(upscale_queue_dir)
    upscaled_p = Path(upscaled_dir)
    img_p = Path(img_dir)

    queue_p.mkdir(parents=True, exist_ok=True)
    upscaled_p.mkdir(parents=True, exist_ok=True)
    img_p.mkdir(parents=True, exist_ok=True)

    inputs = sorted([p for p in queue_p.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")])
    if not inputs:
        return []

    out_files: List[str] = []
    run_log_path = log_path or str(upscaled_p / "upscale_log.jsonl")

    for inp in inputs:
        out_tmp = upscaled_p / inp.name
        if out_tmp.exists():
            raise UpscaleError(f"Refusing to overwrite existing upscaled file: {out_tmp}")

        cmd = [
            str(exe),
            "-i",
            str(inp),
            "-o",
            str(out_tmp),
            "-s",
            str(scale),
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise UpscaleError(
                "Real-ESRGAN failed.\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout: {p.stdout}\n"
                f"stderr: {p.stderr}\n"
            )

        # Move to img
        final_path = img_p / inp.name
        if final_path.exists():
            raise UpscaleError(f"Refusing to overwrite existing final image in img/: {final_path}")

        shutil.move(str(out_tmp), str(final_path))
        # Keep the original in queue (do not delete silently). Move it aside.
        processed_dir = queue_p / "processed"
        processed_dir.mkdir(exist_ok=True)
        shutil.move(str(inp), str(processed_dir / inp.name))

        out_files.append(str(final_path))

        append_jsonl(
            run_log_path,
            {
                "ts": utc_ts(),
                "input": str(inp),
                "output": str(final_path),
                "scale": scale,
                "exe": str(exe),
            },
        )

    write_json(upscaled_p / "upscale_summary.json", {"scale": scale, "processed": out_files})
    return out_files
