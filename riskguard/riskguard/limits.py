"""Exposure limit engine: clip/rescale proposed weights and report violations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExposureLimits:
    """Limit configuration. ``None`` disables a check.

    - ``max_position``: cap on each |weight|.
    - ``max_gross``: cap on sum of |weights| (gross exposure).
    - ``max_net``: cap on |sum of weights| (net exposure).
    - ``max_sector``: cap on gross exposure per sector (requires a sector map).
    """

    max_position: float | None = 0.10
    max_gross: float | None = 1.0
    max_net: float | None = 1.0
    max_sector: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_position", "max_gross", "max_net", "max_sector"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive or None, got {value}")


@dataclass(frozen=True)
class Violation:
    """One limit breach and the corrective action taken."""

    rule: str  # "max_position" | "max_sector" | "max_gross" | "max_net"
    scope: str  # asset name, sector name, or "portfolio"
    value: float  # offending exposure before correction
    limit: float
    action: str  # human-readable description of the fix


@dataclass(frozen=True)
class LimitResult:
    """Adjusted weights plus the violations report."""

    weights: dict[str, float]
    violations: list[Violation] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        """True when the proposed weights required no adjustment."""
        return len(self.violations) == 0

    @property
    def gross_exposure(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    @property
    def net_exposure(self) -> float:
        return sum(self.weights.values())


class LimitEngine:
    """Apply exposure limits to proposed portfolio weights.

    Order of enforcement: per-position cap -> sector caps -> gross cap ->
    net cap. Each correction is recorded as a :class:`Violation`. Later
    steps only ever scale weights *down*, so earlier limits stay satisfied.
    """

    def __init__(self, limits: ExposureLimits | None = None):
        self.limits = limits or ExposureLimits()

    def apply(
        self,
        weights: Mapping[str, float],
        sectors: Mapping[str, str] | None = None,
    ) -> LimitResult:
        """Return clipped/rescaled weights and a violations report."""
        lim = self.limits
        adjusted = {k: float(v) for k, v in weights.items()}
        violations: list[Violation] = []

        # 1. Per-position cap (preserve sign).
        if lim.max_position is not None:
            for asset, w in adjusted.items():
                if abs(w) > lim.max_position + 1e-12:
                    capped = lim.max_position if w > 0 else -lim.max_position
                    violations.append(
                        Violation(
                            rule="max_position",
                            scope=asset,
                            value=w,
                            limit=lim.max_position,
                            action=f"clipped {asset} from {w:+.4f} to {capped:+.4f}",
                        )
                    )
                    adjusted[asset] = capped

        # 2. Sector gross caps (scale every member of the offending sector).
        if lim.max_sector is not None and sectors is not None:
            sector_gross: dict[str, float] = {}
            for asset, w in adjusted.items():
                sector = sectors.get(asset, "unassigned")
                sector_gross[sector] = sector_gross.get(sector, 0.0) + abs(w)
            for sector, gross in sorted(sector_gross.items()):
                if gross > lim.max_sector + 1e-12:
                    scale = lim.max_sector / gross
                    for asset in adjusted:
                        if sectors.get(asset, "unassigned") == sector:
                            adjusted[asset] *= scale
                    violations.append(
                        Violation(
                            rule="max_sector",
                            scope=sector,
                            value=gross,
                            limit=lim.max_sector,
                            action=f"scaled sector '{sector}' by {scale:.4f}",
                        )
                    )

        # 3. Gross exposure cap (scale entire book).
        if lim.max_gross is not None:
            gross = sum(abs(w) for w in adjusted.values())
            if gross > lim.max_gross + 1e-12:
                scale = lim.max_gross / gross
                adjusted = {k: w * scale for k, w in adjusted.items()}
                violations.append(
                    Violation(
                        rule="max_gross",
                        scope="portfolio",
                        value=gross,
                        limit=lim.max_gross,
                        action=f"scaled all weights by {scale:.4f}",
                    )
                )

        # 4. Net exposure cap (scale entire book; keeps gross compliant).
        if lim.max_net is not None:
            net = sum(adjusted.values())
            if abs(net) > lim.max_net + 1e-12:
                scale = lim.max_net / abs(net)
                adjusted = {k: w * scale for k, w in adjusted.items()}
                violations.append(
                    Violation(
                        rule="max_net",
                        scope="portfolio",
                        value=net,
                        limit=lim.max_net,
                        action=f"scaled all weights by {scale:.4f}",
                    )
                )

        return LimitResult(weights=adjusted, violations=violations)
