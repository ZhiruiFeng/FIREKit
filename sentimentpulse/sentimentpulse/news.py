"""News/event pipeline: NewsItem model, symbol tagging, deduplication."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator


class NewsItem(BaseModel):
    """A single news document."""

    id: str
    timestamp: datetime
    headline: str
    body: str = ""
    symbol: str | None = None
    source: str = "unknown"

    @field_validator("headline")
    @classmethod
    def _headline_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("headline must not be empty")
        return v

    @property
    def text(self) -> str:
        """Headline + body, the unit of scoring."""
        return f"{self.headline}\n{self.body}".strip()


class AliasMap(BaseModel):
    """Entity-to-symbol alias map, e.g. {"AAPL": ["apple", "apple inc"]}."""

    aliases: dict[str, list[str]] = Field(default_factory=dict)

    def compiled(self) -> dict[str, re.Pattern[str]]:
        patterns: dict[str, re.Pattern[str]] = {}
        for symbol, names in self.aliases.items():
            terms = sorted({symbol.lower(), *[n.lower() for n in names]}, key=len, reverse=True)
            joined = "|".join(re.escape(t) for t in terms)
            patterns[symbol] = re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)
        return patterns


def tag_symbols(text: str, alias_map: AliasMap) -> list[str]:
    """Symbols whose name/aliases appear in the text (word-boundary match)."""
    return sorted(sym for sym, pat in alias_map.compiled().items() if pat.search(text))


def tag_items(items: list[NewsItem], alias_map: AliasMap) -> list[NewsItem]:
    """Assign symbols to untagged items; items matching several symbols are
    duplicated once per symbol; items matching none are dropped."""
    tagged: list[NewsItem] = []
    for item in items:
        if item.symbol is not None:
            tagged.append(item)
            continue
        for sym in tag_symbols(item.text, alias_map):
            tagged.append(item.model_copy(update={"symbol": sym}))
    return tagged


def _dedup_key(item: NewsItem) -> tuple[str | None, str]:
    normalized = re.sub(r"[^a-z0-9 ]+", "", item.headline.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return (item.symbol, normalized)


def deduplicate(items: list[NewsItem], window: timedelta = timedelta(days=1)) -> list[NewsItem]:
    """Drop near-duplicate items: same symbol + normalized headline within
    ``window`` of an earlier item. Keeps the earliest; preserves time order."""
    last_seen: dict[tuple[str | None, str], datetime] = {}
    kept: list[NewsItem] = []
    for item in sorted(items, key=lambda x: x.timestamp):
        key = _dedup_key(item)
        prev = last_seen.get(key)
        if prev is not None and item.timestamp - prev <= window:
            continue
        last_seen[key] = item.timestamp
        kept.append(item)
    return kept
