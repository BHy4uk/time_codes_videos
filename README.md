# time_codes_videos

Local, deterministic CLI tool to generate a video by synchronizing fullscreen images with spoken phrases in an audio file.

## What it does

Deterministic pipeline:

1. **Audio → transcription** (segment timestamps) using **faster-whisper**
2. **Full-phrase fuzzy matching** against configured phrases (token_set_ratio + normalization)
3. **Timeline generation** (strictly chronological, non-overlapping)
4. **FFmpeg rendering** to MP4 with the audio as background

## Install (Windows)

Open PowerShell in the project folder (where `main.py` is) and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

> If PowerShell blocks activation scripts, run once:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### System dependency (Windows)

You must have `ffmpeg` and `ffprobe` installed and available on your PATH.

Quick check:

```powershell
ffmpeg -version
ffprobe -version
```

## Run (Windows)

### 1) Transcribe only (generate `segments.json`)

Use this first if you haven’t built `mapping.json` yet:

```powershell
python main.py `
  --audio "./audio.mp3" `
  --out "./out" `
  --transcribe-only
```

### 2) Full pipeline (segments → timeline → mp4)

```powershell
python main.py `
  --audio "./audio.mp3" `
  --images "./images" `
  --config "./mapping.json" `
  --out "./out"
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
        "fade": {"type": "in", "duration": 1},
        "focus": {
          "source": { "width": 4000, "height": 3000 },
          "target": { "x": 1200, "y": 800, "width": 600, "height": 500 }
        }
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

**Tip (anti-jitter):** the renderer generates each scene using a single still frame + `zoompan` for the entire scene duration, and renders with oversampling + pixel-rounded pan coordinates to minimize jitter/stepping (especially noticeable on small zoom values).

#### 2) `motion` (pan / simulated camera movement)

```json
"motion": { "direction": "right", "intensity": 0.05 }
```

- `direction`: `"right" | "left" | "up" | "down"`
- `intensity`: how far to pan (0.0..0.5). Recommended `0.02`–`0.08`.

This is implemented together with `zoom` via FFmpeg `zoompan`.

#### 2b) `focus` (object-anchored push-in + pan, using manual coordinates)

Use this when you want the camera to push-in and pan toward a specific object/region.
Coordinates are measured in **absolute pixels on the ORIGINAL image** (e.g., from Photoshop).

```json
"focus": {
  "source": { "width": 4000, "height": 3000 },
  "target": { "x": 1200, "y": 800, "width": 600, "height": 500 }
}
```

How it works:
- Computes the target center:
  - `cx = x + width/2`, `cy = y + height/2`
- Normalizes `(cx,cy)` to `[0..1]` based on source image size
- During the zoom ramp, the camera pans smoothly from center to the target center:
  - `cx(t) = center + (target - center) * progress`
- Uses `zoompan` math to keep the view centered on the anchor:
  - `x = cx - (iw/zoom)/2`, `y = cy - (ih/zoom)/2`
- Clamps x/y so the crop window stays inside the image (no black borders).

Notes:
- If you omit `source`, the tool will try to read the image size automatically.
- `focus` can be used with `zoom` (recommended) and optionally with `fade/darken/vignette`.

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
