from __future__ import annotations

import json
from typing import Any, Dict, List

from pathlib import Path

from .config_schema import AppConfigV2, MatchingConfig, RenderConfig, RuleV2


def load_mapping_config(config_path: str) -> AppConfigV2:
    """Load mapping.json with support for image/video assets.

    Schema (required):

    {
      "rules": [
        {
          "asset": "01.jpg",
          "type": "image",   // or "video"
          "text": "Phrase that marks the start",
          "effects": {...}    // optional
        }
      ],
      "matching": {"mode": "full_phrase", "similarity_threshold": 85},
      "render": {"on_short_video": "freeze" | "loop"}
    }

    Backward compatible:
    - If a rule has `image` instead of `asset`, we treat it as type=image.
    """

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    data: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8-sig"))

    rules_raw = data.get("rules")
    if not isinstance(rules_raw, list) or not rules_raw:
        raise ValueError("Config must contain non-empty 'rules' list")

    rules: List[RuleV2] = []
    for i, r in enumerate(rules_raw):
        if not isinstance(r, dict):
            raise ValueError(f"Rule at index {i} must be an object")

        # Back-compat: image -> asset
        asset = r.get("asset")
        rtype = r.get("type")
        if asset is None and isinstance(r.get("image"), str):
            asset = r.get("image")
            rtype = "image"

        text = r.get("text")
        effects = r.get("effects")

        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(f"Rule at index {i} is missing non-empty 'asset'")
        if rtype not in ("image", "video"):
            raise ValueError(f"Rule at index {i} has invalid 'type' (must be 'image' or 'video')")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Rule at index {i} is missing non-empty 'text'")
        if effects is not None and not isinstance(effects, dict):
            raise ValueError(f"Rule at index {i} has invalid 'effects' (must be an object)")

        rules.append(RuleV2(asset=asset.strip(), type=rtype, text=text.strip(), effects=effects))

    matching_raw = data.get("matching") if isinstance(data.get("matching"), dict) else {}
    mode = matching_raw.get("mode", "full_phrase")
    threshold = matching_raw.get("similarity_threshold", 85)
    if mode != "full_phrase":
        raise ValueError("Only matching.mode='full_phrase' is supported")
    if not isinstance(threshold, int) or threshold < 0 or threshold > 100:
        raise ValueError("matching.similarity_threshold must be an integer 0..100")

    render_raw = data.get("render") if isinstance(data.get("render"), dict) else {}
    on_short_video = render_raw.get("on_short_video", "freeze")
    if on_short_video not in ("freeze", "loop"):
        raise ValueError("render.on_short_video must be 'freeze' or 'loop'")

    return AppConfigV2(
        rules=rules,
        matching=MatchingConfig(mode=mode, similarity_threshold=threshold),
        render=RenderConfig(on_short_video=on_short_video),
    )
