from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


AssetType = Literal["image", "video"]
OnShortVideo = Literal["freeze", "loop"]


@dataclass(frozen=True)
class AssetSpec:
    asset: str
    type: AssetType


@dataclass(frozen=True)
class RuleV2:
    asset: str
    type: AssetType
    text: str
    effects: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RenderConfig:
    on_short_video: OnShortVideo = "freeze"


@dataclass(frozen=True)
class MatchingConfig:
    mode: str = "full_phrase"
    similarity_threshold: int = 85


@dataclass(frozen=True)
class AppConfigV2:
    rules: List[RuleV2]
    matching: MatchingConfig
    render: RenderConfig
