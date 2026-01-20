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
    {"image": "01.png", "text": "Full sentence or paragraph ..."}
  ],
  "matching": {
    "mode": "full_phrase",
    "similarity_threshold": 85
  }
}
```

Notes:

- Each `text` is treated as one **atomic phrase**.
- Each segment triggers **at most one** image.
- Each image triggers **only once**, at its earliest match.
- If a segment matches multiple rules, the **highest similarity** wins.
- Exact ties are broken deterministically (lexicographic by normalized phrase, then image filename).
