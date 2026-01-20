import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Rule:
    image: str
    text: str


@dataclass(frozen=True)
class MatchingConfig:
    mode: str = "full_phrase"
    similarity_threshold: int = 85


@dataclass(frozen=True)
class AppConfig:
    rules: List[Rule]
    matching: MatchingConfig


def load_config(config_path: str) -> AppConfig:
    """Load JSON configuration.

    Expected shape:
    {
      "rules": [{"image": "01.png", "text": "..."}, ...],
      "matching": {"mode": "full_phrase", "similarity_threshold": 85}
    }
    """

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    data: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))

    rules_raw = data.get("rules")
    if not isinstance(rules_raw, list) or not rules_raw:
        raise ValueError("Config must contain non-empty 'rules' list")

    rules: List[Rule] = []
    for i, r in enumerate(rules_raw):
        if not isinstance(r, dict):
            raise ValueError(f"Rule at index {i} must be an object")
        image = r.get("image")
        text = r.get("text")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"Rule at index {i} is missing non-empty 'image'")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Rule at index {i} is missing non-empty 'text'")
        rules.append(Rule(image=image.strip(), text=text.strip()))

    matching_raw: Optional[Dict[str, Any]] = data.get("matching") if isinstance(data.get("matching"), dict) else None
    mode = (matching_raw or {}).get("mode", "full_phrase")
    threshold = (matching_raw or {}).get("similarity_threshold", 85)

    if mode != "full_phrase":
        raise ValueError("Only matching.mode='full_phrase' is supported")
    if not isinstance(threshold, int) or threshold < 0 or threshold > 100:
        raise ValueError("matching.similarity_threshold must be an integer 0..100")

    return AppConfig(rules=rules, matching=MatchingConfig(mode=mode, similarity_threshold=threshold))
