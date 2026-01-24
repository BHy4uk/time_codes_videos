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
    - zoom: {"type": "in"|"out", "scale": 1.0..3.0, "duration": seconds(optional)}
    - motion: {"direction": "right"|"left"|"down"|"up", "intensity": 0.0..0.5}
    - focus: {
        "source": {"width": <int>, "height": <int>} (optional),
        "target": {"x": <px>, "y": <px>, "width": <px>, "height": <px>}
      }
      `target` coords are absolute pixels measured on the ORIGINAL image (e.g., Photoshop).
      The center of this bbox becomes the anchor for zoom/pan.
    - fade: {"type": "in"|"out"|"inout", "duration": seconds}
    - darken: {"amount": 0.0..1.0}
    - vignette: {"angle": radians (0..PI/2), "eval": "init"|"frame"}

    Unknown keys are ignored.

    Returns: (filter_string, debug_info)

    Determinism:
    - fixed fps
    - expressions depend only on frame number
    - `trim=duration=...` ensures exact scene duration
    """

    debug: Dict[str, Any] = {"applied": []}

    zoom_cfg = effects.get("zoom") if isinstance(effects.get("zoom"), dict) else None
    motion_cfg = effects.get("motion") if isinstance(effects.get("motion"), dict) else None
    focus_cfg = effects.get("focus") if isinstance(effects.get("focus"), dict) else None

    use_zoompan = zoom_cfg is not None or motion_cfg is not None or focus_cfg is not None

    chain = []

    # ---------- ZOOMPAN (zoom + motion + focus) ----------
    if use_zoompan:
        zoom_type = (zoom_cfg or {}).get("type", "in")
        target_scale = _clamp(_as_float((zoom_cfg or {}).get("scale", 1.08), 1.08), 1.0, 3.0)

        motion_dir = (motion_cfg or {}).get("direction", "right")
        intensity = _clamp(_as_float((motion_cfg or {}).get("intensity", 0.0), 0.0), 0.0, 0.5)

        zoom_ramp = _clamp(_as_float((zoom_cfg or {}).get("duration", duration), duration), 0.01, duration)

        total_frames = max(1, int(round(duration * fps)))
        ramp_frames = max(1, int(round(zoom_ramp * fps)))

        if zoom_type == "out":
            start_z = target_scale
            end_z = 1.0
        else:
            start_z = 1.0
            end_z = target_scale

        if ramp_frames <= 1:
            z_expr = f"{end_z:.6f}"
            pan_progress = "1"
        else:
            z_expr = f"{start_z:.6f}+({end_z:.6f}-{start_z:.6f})*min(on-1\\,{ramp_frames-1})/{ramp_frames-1}"
            # pan progresses with the same ramp as zoom (smooth pan while zooming)
            pan_progress = f"min(on-1\\,{ramp_frames-1})/{ramp_frames-1}"

        # Focus target center in normalized coords
        # If not available, default is center of frame.
        target_nx: Optional[float] = None
        target_ny: Optional[float] = None
        if focus_cfg is not None:
            target = focus_cfg.get("target") if isinstance(focus_cfg.get("target"), dict) else None
            source_override = focus_cfg.get("source") if isinstance(focus_cfg.get("source"), dict) else None

            src_w = None
            src_h = None
            if source_override is not None:
                src_w = _as_int(source_override.get("width"), 0) or None
                src_h = _as_int(source_override.get("height"), 0) or None
            if (src_w is None or src_h is None) and source_size is not None:
                src_w, src_h = source_size

            if target is not None and src_w and src_h:
                tx = _as_float(target.get("x"), 0.0)
                ty = _as_float(target.get("y"), 0.0)
                tw = _as_float(target.get("width"), 0.0)
                th = _as_float(target.get("height"), 0.0)
                cx = tx + (tw / 2.0)
                cy = ty + (th / 2.0)

                # Normalize into the final 1920x1080 (width x height) canvas coordinates
                # produced by: scale=width:height:force_original_aspect_ratio=decrease + pad.
                s = min(width / float(src_w), height / float(src_h))
                new_w = float(src_w) * s
                new_h = float(src_h) * s
                pad_x = (width - new_w) / 2.0
                pad_y = (height - new_h) / 2.0

                target_cx_canvas = pad_x + (cx * s)
                target_cy_canvas = pad_y + (cy * s)

                target_nx = target_cx_canvas / float(width)
                target_ny = target_cy_canvas / float(height)

                debug["applied"].append(
                    {
                        "type": "focus",
                        "config": focus_cfg,
                        "computed": {
                            "source_size": [int(src_w), int(src_h)],
                            "target_center_source": [cx, cy],
                            "scale": s,
                            "pad": [pad_x, pad_y],
                            "target_center_canvas": [target_cx_canvas, target_cy_canvas],
                            "target_center_norm": [target_nx, target_ny],
                            "pan_progress": pan_progress,
                        },
                    }
                )

        # Determine (cx,cy) expressions in current frame coordinates.
        # We pan from center toward target during the zoom ramp.
        if target_nx is not None and target_ny is not None:
            cx_expr = f"iw/2+(({target_nx:.10f})*iw-iw/2)*({pan_progress})"
            cy_expr = f"ih/2+(({target_ny:.10f})*ih-ih/2)*({pan_progress})"
        else:
            cx_expr = "iw/2"
            cy_expr = "ih/2"

        # Base anchored pan (spec):
        # x = cx - (iw/zoom)/2
        # y = cy - (ih/zoom)/2
        x_unclamped = f"({cx_expr})-(iw/zoom)/2"
        y_unclamped = f"({cy_expr})-(ih/zoom)/2"

        x_base = f"max(0\\,min({x_unclamped}\\,iw-iw/zoom))"
        y_base = f"max(0\\,min({y_unclamped}\\,ih-ih/zoom))"

        # Optional extra motion offset (still deterministic)
        progress_full = f"(on-1)/{max(1, total_frames-1)}"
        x_offset = "0"
        y_offset = "0"
        if intensity > 0:
            if motion_dir == "left":
                x_offset = f"-({intensity:.6f})*(iw-iw/zoom)*{progress_full}"
            elif motion_dir == "right":
                x_offset = f"({intensity:.6f})*(iw-iw/zoom)*{progress_full}"
            elif motion_dir == "up":
                y_offset = f"-({intensity:.6f})*(ih-ih/zoom)*{progress_full}"
            elif motion_dir == "down":
                y_offset = f"({intensity:.6f})*(ih-ih/zoom)*{progress_full}"

        # Round to integer pixels to reduce micro-wobble.
        x_expr = f"floor(max(0\\,min(({x_base})+({x_offset})\\,iw-iw/zoom)))"
        y_expr = f"floor(max(0\\,min(({y_base})+({y_offset})\\,ih-ih/zoom)))"

        # Normalize image FIRST to a fixed 1920x1080 canvas.
        # After this point, zoompan/focus/fade all operate on a consistent iw/ih.
        chain.extend(
            [
                f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            ]
        )

        # Oversample to reduce visible stepping, then downscale.
        oversample = 2
        ow = width * oversample
        oh = height * oversample

        chain.extend(
            [
                f"scale={ow}:{oh}:flags=lanczos",
                f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={total_frames}:s={ow}x{oh}:fps={fps}",
                f"scale={width}:{height}:flags=lanczos",
            ]
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
                    "cx_expr": cx_expr,
                    "cy_expr": cy_expr,
                    "x_expr": x_expr,
                    "y_expr": y_expr,
                },
            }
        )

    else:
        # No zoompan: simple fit + pad
        chain.extend(
            [
                f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            ]
        )

    # ---------- FADE ----------
    fade_cfg = effects.get("fade") if isinstance(effects.get("fade"), dict) else None
    if fade_cfg is not None:
        fade_type = (fade_cfg.get("type") or "in").lower()
        fade_dur = _clamp(_as_float(fade_cfg.get("duration", 1.0), 1.0), 0.0, duration)

        if fade_type in ("in", "inout") and fade_dur > 0:
            chain.append(f"fade=t=in:st=0:d={fade_dur:.6f}")
        if fade_type in ("out", "inout") and fade_dur > 0:
            start_out = max(0.0, duration - fade_dur)
            chain.append(f"fade=t=out:st={start_out:.6f}:d={fade_dur:.6f}")

        debug["applied"].append({"type": "fade", "config": fade_cfg})

    # ---------- DARKEN ----------
    darken_cfg = effects.get("darken") if isinstance(effects.get("darken"), dict) else None
    if darken_cfg is not None:
        amount = _clamp(_as_float(darken_cfg.get("amount", 0.0), 0.0), 0.0, 1.0)
        brightness = -0.3 * amount
        chain.append(f"eq=brightness={brightness:.6f}")
        debug["applied"].append({"type": "darken", "config": {"amount": amount, "brightness": brightness}})

    # ---------- VIGNETTE ----------
    vignette_cfg = effects.get("vignette") if isinstance(effects.get("vignette"), dict) else None
    if vignette_cfg is not None:
        angle = _clamp(_as_float(vignette_cfg.get("angle", 0.6), 0.6), 0.0, 1.57079632679)
        eval_mode = "frame" if str(vignette_cfg.get("eval", "init")).lower() == "frame" else "init"
        chain.append(f"vignette=angle={angle}:eval={eval_mode}")
        debug["applied"].append({"type": "vignette", "config": {"angle": angle, "eval": eval_mode}})

    # Ensure deterministic FPS and exact scene duration for concat.
    chain.append(f"fps={fps}")
    chain.append(f"trim=duration={duration:.6f}")
    chain.append("format=yuv420p")

    return ",".join(chain), debug
