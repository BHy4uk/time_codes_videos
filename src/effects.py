from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))



def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def build_effects_filter(
    effects: Dict[str, Any],
    width: int,
    height: int,
    fps: int,
    duration: float,
    source_size: Optional[Tuple[int, int]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build an FFmpeg video filter chain implementing supported effects.

    Supported (optional/composable):
    - zoom: {"type": "in"|"out", "scale": 1.1..2.0, "duration": seconds(optional)}
    - motion: {"direction": "right"|"left"|"down"|"up", "intensity": 0.0..0.5}
    - fade: {"type": "in"|"out"|"inout", "duration": seconds}
    - darken: {"amount": 0.0..1.0}
    - vignette: {"angle": radians (0..PI/2), "eval": "init"|"frame"}

    Unknown keys are ignored.

    Returns: (filter_string_without_trailing_comma, debug_info)

    Notes:
    - We build effects on top of a base scale/pad.
    - zoom/motion are implemented using `zoompan`.
    - fade uses `fade` filter.
    - darken uses `eq=brightness=`.

    Determinism:
    - All expressions are deterministic functions of frame number/time.
    """

    debug: Dict[str, Any] = {"applied": []}

    # Base: keep aspect ratio, fit inside, pad to a canvas.
    # When using zoompan we render at an oversampled resolution to reduce
    # visible stepping/jitter from integer crop coordinates.


    chain = []

    # ---- zoom + motion (+ optional focus target) (zoompan) ----
    zoom_cfg = effects.get("zoom") if isinstance(effects.get("zoom"), dict) else None
    motion_cfg = effects.get("motion") if isinstance(effects.get("motion"), dict) else None
    focus_cfg = effects.get("focus") if isinstance(effects.get("focus"), dict) else None

    if zoom_cfg is not None or motion_cfg is not None or focus_cfg is not None:
        # Defaults
        zoom_type = (zoom_cfg or {}).get("type", "in")
        target_scale = _as_float((zoom_cfg or {}).get("scale", 1.1), 1.1)
        target_scale = _clamp(target_scale, 1.0, 3.0)

        motion_dir = (motion_cfg or {}).get("direction", "right")
        intensity = _as_float((motion_cfg or {}).get("intensity", 0.0), 0.0)
        intensity = _clamp(intensity, 0.0, 0.5)

        # Use full clip duration; allow zoom.duration to override the ramp time.
        zoom_ramp = _as_float((zoom_cfg or {}).get("duration", duration), duration)
        zoom_ramp = _clamp(zoom_ramp, 0.01, duration)

        total_frames = max(1, int(round(duration * fps)))
        ramp_frames = max(1, int(round(zoom_ramp * fps)))

        # z expression: linear ramp for first ramp_frames, then hold.
        # pzoom is available, but we use explicit formula based on output frame number (on).
        if zoom_type == "out":
            start_z = target_scale
            end_z = 1.0
        else:
            start_z = 1.0
            end_z = target_scale

        # Use on (output frame number, 1-based). Clamp after ramp.
        # z = start + (end-start) * min(on-1, ramp_frames-1)/(ramp_frames-1)
        if ramp_frames <= 1:
            z_expr = f"{end_z:.6f}"
        else:
            z_expr = (
                f"{start_z:.6f}+({end_z:.6f}-{start_z:.6f})*min(on-1\,{ramp_frames-1})/{ramp_frames-1}"
            )

        # motion x/y around centered crop window; offset within remaining margin.
        # zoompan uses x/y in input coords; window size is iw/zoom, ih/zoom.
        # max_x = iw - iw/zoom; max_y = ih - ih/zoom
        # base center: (iw - iw/zoom)/2 etc.
        # Round to whole pixels to avoid sub-pixel wobble caused by fractional
        # crop coordinates.
        x_center = "floor((iw-iw/zoom)/2)"
        y_center = "floor((ih-ih/zoom)/2)"

        # Offset by intensity * max_{x,y}, linearly over clip.
        progress = f"(on-1)/{max(1, total_frames-1)}"
        x_offset = "0"
        y_offset = "0"
        if intensity > 0:
            if motion_dir == "left":
                x_offset = f"-({intensity:.6f})*(iw-iw/zoom)*{progress}"
            elif motion_dir == "right":
                x_offset = f"({intensity:.6f})*(iw-iw/zoom)*{progress}"
            elif motion_dir == "up":
                y_offset = f"-({intensity:.6f})*(ih-ih/zoom)*{progress}"
            elif motion_dir == "down":
                y_offset = f"({intensity:.6f})*(ih-ih/zoom)*{progress}"

        x_expr = f"floor({x_center}+{x_offset})"
        y_expr = f"floor({y_center}+{y_offset})"

        # zoompan: generate the whole scene from a single input frame.
        # To reduce jitter, render at an oversampled size and then downscale.
        oversample = 2
        ow = width * oversample
        oh = height * oversample

        chain.append(
            f"scale={ow}:{oh}:force_original_aspect_ratio=increase"
        )
        chain.append(
            f"crop={ow}:{oh}"
        )
        chain.append(
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={total_frames}:s={ow}x{oh}:fps={fps}"
        )
        chain.append(
            f"scale={width}:{height}:flags=lanczos"
        )

        debug["applied"].append(
            {
                "type": "zoompan",
                "zoom": zoom_cfg or {},
                "motion": motion_cfg or {},
                "computed": {
                    "duration": duration,
                    "fps": fps,
                    "total_frames": total_frames,
                    "ramp_frames": ramp_frames,
                    "z_expr": z_expr,
                    "x_expr": x_expr,
                    "y_expr": y_expr,
                },
            }
        )

    # ---- fade ----
    fade_cfg = effects.get("fade") if isinstance(effects.get("fade"), dict) else None
    if fade_cfg is not None:
        fade_type = (fade_cfg.get("type") or "in").lower()
        fade_dur = float(fade_cfg.get("duration", 1.0))
        fade_dur = _clamp(fade_dur, 0.0, duration)

        if fade_type in ("in", "inout") and fade_dur > 0:
            chain.append(f"fade=t=in:st=0:d={fade_dur:.6f}")
        if fade_type in ("out", "inout") and fade_dur > 0:
            start_out = max(0.0, duration - fade_dur)
            chain.append(f"fade=t=out:st={start_out:.6f}:d={fade_dur:.6f}")

        debug["applied"].append({"type": "fade", "config": fade_cfg})

    # ---- darken ----
    darken_cfg = effects.get("darken") if isinstance(effects.get("darken"), dict) else None
    if darken_cfg is not None:
        amount = float(darken_cfg.get("amount", 0.0))
        amount = _clamp(amount, 0.0, 1.0)
        # brightness range is roughly [-1..1]; we'll map amount 0..1 -> 0..-0.3
        brightness = -0.3 * amount
        chain.append(f"eq=brightness={brightness:.6f}")
        debug["applied"].append({"type": "darken", "config": {"amount": amount, "brightness": brightness}})

    # ---- vignette ----
    vignette_cfg = effects.get("vignette") if isinstance(effects.get("vignette"), dict) else None
    if vignette_cfg is not None:
        angle = vignette_cfg.get("angle", None)
        eval_mode = vignette_cfg.get("eval", "init")
        if angle is None:
            # a gentle default
            angle = 0.6
        angle = float(angle)
        angle = _clamp(angle, 0.0, 1.57079632679)
        eval_mode = "frame" if str(eval_mode).lower() == "frame" else "init"
        chain.append(f"vignette=angle={angle}:eval={eval_mode}")
        debug["applied"].append({"type": "vignette", "config": {"angle": angle, "eval": eval_mode}})

    # Ensure deterministic FPS and exact scene duration for concat.
    chain.append(f"fps={fps}")
    chain.append(f"trim=duration={duration:.6f}")
    chain.append("format=yuv420p")

    return ",".join(chain), debug
