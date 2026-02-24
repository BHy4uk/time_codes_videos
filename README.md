# time_codes_videos

Local, deterministic CLI tool to generate a video by synchronizing fullscreen images with spoken phrases in an audio file.

## What it does

Deterministic pipeline:

1. **Prompts → Gemini image generation** (optional stage)
2. **Upscale** (optional stage via Real-ESRGAN)
3. **Audio → transcription** (word timestamps) using **faster-whisper**
4. **Phrase alignment**: mapping.json defines phrase order; transcription is used only to find each phrase start timestamp.
5. **Timeline generation**: asset N shows from phrase N start until phrase N+1 start; last asset until audio end.
6. **FFmpeg rendering** to MP4 with the audio as background

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

## Run (Windows) — block-based execution

This project is designed so that **each block folder is an independent production unit**.
You work inside a specific block directory (current working directory), while calling `main.py` from the repository using its full path.

### Directory layout (example)

Block folder:

```
D:\State_51\cases\case_007\block_00\
  audio\
    audio.mp3
  prompts\
  upscale_queue\
  img\
  config\
    mapping.json
  out\
```

Repository (tooling):

```
C:\Users\DZ\source\repos\videos_creations\time_codes_videos\
  main.py
  src\...
  requirements.txt
  .env
```

### Environment variables (.env)

The `.env` file stays in the **repository root** (NOT inside blocks):

`C:\Users\DZ\source\repos\videos_creations\time_codes_videos\.env`

It must contain:

```env
GOOGLE_API_KEY=...
REALESRGAN_PATH=C:\AI\RealESRGAN\realesrgan-ncnn-vulkan.exe
```

### Important clarifications

- **All relative paths** like `./prompts`, `./img`, `./config`, `./out` are resolved from your **current working directory** (the block folder).
- `main.py` is called using its **full path** from the repository.
- There is **no base-dir parameter**.
- The pipeline will fail fast if dependencies are missing (Gemini key, Real-ESRGAN, FFmpeg).

---

## Execution flow

### Step 1 — Navigate to the block directory

```powershell
cd D:\State_51\cases\case_007\block_00
```

### Step 2 — Generate images (Gemini)

```powershell
python C:\Users\DZ\source\repos\videos_creations\time_codes_videos\main.py generate --prompts "./prompts" --count 1 --seed 42
```

Outputs (in the block folder):
- `./generated`
- `./upscale_queue`

### Step 3 — Upscale (Real-ESRGAN)

```powershell
python C:\Users\DZ\source\repos\videos_creations\time_codes_videos\main.py upscale --scale 4
```

Outputs:
- `./img`

### Step 4 — Render

```powershell
python C:\Users\DZ\source\repos\videos_creations\time_codes_videos\main.py render --config "./config/mapping.json" --audio "./audio/audio.mp3" --assets "./img" --out "./out"
```

Output:
- `./out/output.mp4`

## Config format

### mapping.json schema (supports image OR video assets)

Each rule defines:
- `asset`: filename in your assets folder (e.g., `./img`)
- `type`: `"image"` or `"video"`
- `text`: phrase that marks the start of this asset
- `effects`: optional visual effects applied after normalization

Top-level `render` controls video-short behavior:

```json
"render": { "on_short_video": "freeze" }
```

See `example_config.json`:

```json
{
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
          "source": { "width": 4000, "height": 3000 },
          "target": { "x": 1200, "y": 800, "width": 600, "height": 500 }
        }
      }
    },
    {
      "asset": "intro.mp4",
      "type": "video",
      "text": "This is where the intro begins."
    }
  ],
  "matching": {
    "mode": "full_phrase",
    "similarity_threshold": 85
  },
  "render": {
    "on_short_video": "freeze"
  }
}
```

### Supported effects (all optional; can be combined)

Each `rules[]` item may include an `effects` object. Effects are **per scene** (per image occurrence in the final timeline).
Unknown effect keys are safely ignored.

**Important (normalization):** before applying any effects, every image is first normalized onto a fixed **1920×1080** canvas (scale-to-fit + pad). After that, all effects operate consistently with `iw=1920`, `ih=1080`.

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
