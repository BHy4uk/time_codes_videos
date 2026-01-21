# time_codes_videos

Local, deterministic CLI tool to generate a video by synchronizing fullscreen images with spoken phrases in an audio file.

## What it does

Deterministic pipeline:

1. **Audio → transcription** (segment timestamps) using **faster-whisper**
2. **Full-phrase fuzzy matching** against configured phrases (token_set_ratio + normalization)
3. **Timeline generation** (strictly chronological, non-overlapping)
4. **FFmpeg rendering** to MP4 with the audio as background

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### System dependency

You must have `ffmpeg` and `ffprobe` installed and available on your PATH.

## Run

### 1) Transcribe only (generate `segments.json`)

Use this first if you haven’t built `mapping.json` yet:

```bash
python main.py \
  --audio ./audio.mp3 \
  --out ./out \
  --transcribe-only
```

### 2) Full pipeline (segments → timeline → mp4)

```bash
python main.py \
  --audio ./audio.mp3 \
  --images ./images \
  --config ./mapping.json \
  --out ./out
```

If the output video ends early (common on Windows with FFmpeg concat + still images),
this project uses a robust rendering approach (`-loop 1` inputs + `concat` filter) to
ensure video duration matches the audio/timeline.

Outputs:

- `out/segments.json` – transcription segments with start/end timestamps
- `out/timeline.json` – final image timeline
- `out/output.mp4` – rendered video
- `out/render_manifest.json` – ffmpeg command + embedded timeline (for reproducibility)

## Config format

See `example_config.json`:

```json
{
  "rules": [
    {
      "image": "01.png",
      "text": "Full sentence or paragraph ...",
      "effects": {
        "zoom": {"type": "in", "scale": 1.1, "duration": 4},
        "motion": {"direction": "right", "intensity": 0.05},
        "fade": {"type": "in", "duration": 1}
      }
    }
  ],
  "matching": {
    "mode": "full_phrase",
    "similarity_threshold": 85
  }
}
```

### Supported effects (all optional; can be combined)

Each `rules[]` item may include an `effects` object. Effects are **per scene** (per image occurrence in the final timeline).
Unknown effect keys are safely ignored.

#### 1) `zoom` (Ken Burns–style zoom)

```json
"zoom": { "type": "in", "scale": 1.1, "duration": 4 }
```

- `type`: `"in"` or `"out"`
- `scale`: target zoom scale (1.0 = no zoom). Recommended `1.03`–`1.12` for subtle motion.
- `duration`: seconds for the zoom ramp. If omitted, zoom ramps over the whole scene duration.

**Tip (anti-jitter):** the renderer generates each scene using a single still frame + `zoompan` for the entire scene duration to avoid zoom jitter on some systems.

#### 2) `motion` (pan / simulated camera movement)

```json
"motion": { "direction": "right", "intensity": 0.05 }
```

- `direction`: `"right" | "left" | "up" | "down"`
- `intensity`: how far to pan (0.0..0.5). Recommended `0.02`–`0.08`.

This is implemented together with `zoom` via FFmpeg `zoompan`.

#### 3) `fade` (fade in/out)

```json
"fade": { "type": "inout", "duration": 0.8 }
```

- `type`: `"in" | "out" | "inout"`
- `duration`: seconds for each fade. For `inout`, the same duration is used at start and end.

#### 4) `darken` (slight darkening)

```json
"darken": { "amount": 0.15 }
```

- `amount`: 0.0..1.0 (mapped to a small negative brightness). Recommended `0.05`–`0.25`.

#### 5) `vignette` (edge darkening)

```json
"vignette": { "angle": 0.55, "eval": "init" }
```

- `angle`: 0..1.57 radians (larger = stronger/closer vignette). Recommended `0.45`–`0.8`.
- `eval`: `"init"` (default) or `"frame"` (dynamic, slower).

Notes:

- Each `text` is treated as one **atomic phrase**.
- Each segment triggers **at most one** image.
- Each image triggers **only once**, at its earliest match.
- If a segment matches multiple rules, the **highest similarity** wins.
- Exact ties are broken deterministically (lexicographic by normalized phrase, then image filename).
