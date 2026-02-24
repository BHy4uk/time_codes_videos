from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types

from .env import require_env
from .logging_utils import append_jsonl, utc_ts, write_json


@dataclass(frozen=True)
class GenerationResult:
    prompt_file: str
    prompt_text: str
    output_files: List[str]
    model: str
    seed: Optional[int]
    reference_image: Optional[str]


def _scene_prefix_from_filename(name: str) -> str:
    stem = Path(name).stem
    # If prompt file name is like scene01.txt -> scene01
    return stem


def generate_images_from_prompts(
    prompts_dir: str,
    out_dir: str,
    count: int = 1,
    model: str = "gemini-2.5-flash-image",
    seed: Optional[int] = None,
    reference_image: Optional[str] = None,
    log_path: Optional[str] = None,
) -> List[GenerationResult]:
    """Generate images for each .txt prompt in prompts_dir.

    Uses official Google Gemini SDK (google-genai).

    Determinism:
    - If `seed` is provided, we pass it through config if supported.
    - We log model id, seed, prompt text, and output file paths.

    Output filenames:
      <scene>_<NN>.png
    Example:
      scene01_01.png
      scene01_02.png

    Raises RuntimeError on missing API key or API failures.
    """

    api_key = require_env("GOOGLE_API_KEY")
    _ = api_key  # used by SDK via env variable

    os.environ["GOOGLE_API_KEY"] = api_key

    prompts_p = Path(prompts_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if not prompts_p.exists() or not prompts_p.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_p}")

    prompt_files = sorted([p for p in prompts_p.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if not prompt_files:
        raise ValueError(f"No .txt prompt files found in: {prompts_p}")

    client = genai.Client(api_key=api_key)

    ref_obj = None
    if reference_image:
        ref_path = Path(reference_image)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        # The official docs accept PIL.Image objects in contents.
        # To keep dependencies minimal here, we pass bytes Part.
        mime = "image/png" if ref_path.suffix.lower() == ".png" else "image/jpeg"
        ref_bytes = ref_path.read_bytes()
        ref_obj = types.Part.from_bytes(data=ref_bytes, mime_type=mime)

    results: List[GenerationResult] = []

    run_log_path = log_path or str(out_p / "generation_log.jsonl")

    for pf in prompt_files:
        prompt_text = pf.read_text(encoding="utf-8").strip()
        if not prompt_text:
            continue

        scene_prefix = _scene_prefix_from_filename(pf.name)
        output_files: List[str] = []

        for i in range(1, count + 1):
            # Config: request image output
            cfg = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
            # seed support exists in docs; if SDK doesn't accept, we'll catch TypeError.
            if seed is not None:
                try:
                    cfg = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"], seed=int(seed))
                except TypeError:
                    # SDK/version mismatch: fail clearly (determinism requirement)
                    raise RuntimeError(
                        "The installed google-genai SDK does not support `seed` for image generation. "
                        "Upgrade google-genai or omit --seed."
                    )

            contents = [prompt_text]
            if ref_obj is not None:
                contents = [ref_obj, prompt_text]

            try:
                resp = client.models.generate_content(model=model, contents=contents, config=cfg)
            except Exception as e:
                raise RuntimeError(f"Gemini generation failed for {pf.name}: {e}")

            # Save first image part found
            saved = False
            for part in getattr(resp, "parts", []) or []:
                if getattr(part, "inline_data", None) is not None:
                    img = part.as_image()
                    out_name = f"{scene_prefix}_{i:02d}.png"
                    out_file = out_p / out_name
                    if out_file.exists():
                        raise FileExistsError(f"Refusing to overwrite existing file: {out_file}")
                    img.save(str(out_file))
                    output_files.append(str(out_file))
                    saved = True
                    break

            if not saved:
                # Provide helpful debugging info without dumping big payloads
                text_preview = getattr(resp, "text", None)
                raise RuntimeError(
                    f"Gemini did not return an image for {pf.name}. "
                    f"Text response: {text_preview!r}"
                )

        rec = {
            "ts": utc_ts(),
            "prompt_file": str(pf),
            "model": model,
            "seed": seed,
            "count": count,
            "reference_image": reference_image,
            "prompt_text": prompt_text,
            "outputs": output_files,
        }
        append_jsonl(run_log_path, rec)

        results.append(
            GenerationResult(
                prompt_file=str(pf),
                prompt_text=prompt_text,
                output_files=output_files,
                model=model,
                seed=seed,
                reference_image=reference_image,
            )
        )

    # Write a deterministic summary
    write_json(out_p / "generation_summary.json", {"model": model, "seed": seed, "results": [r.__dict__ for r in results]})

    return results
