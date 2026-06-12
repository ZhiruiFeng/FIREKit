"""Synthetic news + price generation with known ground truth.

The generator produces headline/body text from templates with a *known*
polarity label, so scorer accuracy is measurable. Synthetic prices are built
so that news polarity has injected forward predictive power (sentiment on day
t moves returns on t+1 and t+2), letting the event-study pipeline recover a
real effect.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from sentimentpulse.news import AliasMap, NewsItem

_ADJECTIVES = [
    "Apex", "Blue", "Cedar", "Delta", "Ember", "Falcon", "Granite", "Harbor",
    "Iron", "Juniper", "Keystone", "Lumen", "Meridian", "North", "Orchid",
    "Pioneer", "Quartz", "Ridge", "Summit", "Titan", "Umber", "Vantage",
    "Willow", "Xenon", "Yield", "Zephyr",
]
_NOUNS = [
    "Dynamics", "Systems", "Industries", "Holdings", "Labs", "Networks",
    "Energy", "Logistics", "Materials", "Robotics", "Analytics", "Capital",
    "Foods", "Mobility", "Health",
]

# (template, ground-truth polarity). {c} = company name.
POSITIVE_TEMPLATES: list[str] = [
    "{c} reports record quarterly profit and beats expectations",
    "{c} raises full-year guidance on strong demand",
    "{c} shares surge after earnings top forecasts",
    "{c} announces major buyback as growth accelerates",
    "{c} wins lucrative contract, boosting revenue outlook",
    "{c} posts robust margin improvement and solid cash flow",
    "Analysts upgrade {c} citing impressive momentum",
    "{c} delivers exceptional results, profit climbs to record high",
    "{c} expansion into new markets drives strong growth",
    "{c} rebounds sharply as recovery gains strength",
    "{c} launches innovative product to upbeat reviews",
    "{c} outperforms peers with very strong subscriber gains",
]
NEGATIVE_TEMPLATES: list[str] = [
    "{c} misses earnings estimates as sales decline",
    "{c} cuts guidance, warns of weak demand",
    "{c} shares plunge on disappointing quarterly loss",
    "{c} announces layoffs amid deteriorating conditions",
    "{c} faces lawsuit over alleged fraud",
    "Regulators open investigation into {c} accounting practices",
    "{c} suffers setback as flagship product recall widens",
    "{c} downgraded as margin pressure worsens",
    "{c} profit slumps on rising costs and supply problems",
    "{c} delays product launch, citing serious quality failures",
    "{c} struggles with shrinking market share and falling prices",
    "{c} warns of severe headwinds, outlook turns bleak",
]
NEUTRAL_TEMPLATES: list[str] = [
    "{c} to present at industry conference next month",
    "{c} schedules quarterly earnings call",
    "{c} appoints new board member",
    "{c} announces date of annual shareholder meeting",
    "{c} files routine quarterly report with regulators",
    "{c} completes previously announced office relocation",
    "{c} updates corporate website and brand identity",
    "{c} confirms participation in technology summit",
]
# Hard cases: ground truth differs from naive bag-of-words polarity.
HARD_POSITIVE_TEMPLATES: list[str] = [
    "{c} says recession fears overblown, demand has not weakened",
    "{c} avoids feared losses as turnaround stays on track",
]
HARD_NEGATIVE_TEMPLATES: list[str] = [
    "{c} growth story unravels, gains prove short lived",
    "{c} fails to deliver promised profit improvement",
]

POSITIVE_BODIES = [
    "Management highlighted strong execution and healthy demand across segments.",
    "The company expects continued improvement and raised its outlook.",
]
NEGATIVE_BODIES = [
    "Management warned that conditions remain difficult and risks persist.",
    "The company lowered its outlook citing weak demand and cost pressure.",
]
NEUTRAL_BODIES = [
    "Further details will be provided in due course.",
    "The event is open to institutional investors.",
]


def generate_universe(n_symbols: int = 20) -> tuple[list[str], AliasMap]:
    """Deterministic fictional universe: tickers plus company-name aliases."""
    if n_symbols < 1 or n_symbols > len(_ADJECTIVES) * len(_NOUNS):
        raise ValueError(f"n_symbols must be in [1, {len(_ADJECTIVES) * len(_NOUNS)}]")
    symbols: list[str] = []
    aliases: dict[str, list[str]] = {}
    for i in range(n_symbols):
        adj = _ADJECTIVES[i % len(_ADJECTIVES)]
        noun = _NOUNS[(i * 7) % len(_NOUNS)]
        name = f"{adj} {noun}"
        ticker = (adj[:2] + noun[:2]).upper() + (str(i) if i >= len(_ADJECTIVES) else "")
        symbols.append(ticker)
        aliases[ticker] = [name, adj]
    return symbols, AliasMap(aliases=aliases)


def company_name(symbol: str, alias_map: AliasMap) -> str:
    """Longest alias = the full fictional company name."""
    return max(alias_map.aliases[symbol], key=len)


def generate_news(
    symbols: list[str],
    alias_map: AliasMap,
    n_items: int = 2000,
    start: str = "2024-01-01",
    n_days: int = 504,
    seed: int = 21,
    hard_fraction: float = 0.08,
) -> tuple[list[NewsItem], list[int]]:
    """Generate seeded synthetic news with ground-truth polarity labels.

    Returns:
        (items, truths) where truths[i] in {-1, 0, +1} is the true polarity
        of items[i]. Polarity mix is roughly 40% positive / 40% negative /
        20% neutral; ``hard_fraction`` of polar items use adversarial
        templates whose naive bag-of-words polarity is misleading.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    items: list[NewsItem] = []
    truths: list[int] = []

    for i in range(n_items):
        symbol = symbols[int(rng.integers(len(symbols)))]
        name = company_name(symbol, alias_map)
        u = rng.random()
        if u < 0.4:
            truth = 1
            if rng.random() < hard_fraction:
                template = HARD_POSITIVE_TEMPLATES[int(rng.integers(len(HARD_POSITIVE_TEMPLATES)))]
                body = ""
            else:
                template = POSITIVE_TEMPLATES[int(rng.integers(len(POSITIVE_TEMPLATES)))]
                body = POSITIVE_BODIES[int(rng.integers(len(POSITIVE_BODIES)))]
        elif u < 0.8:
            truth = -1
            if rng.random() < hard_fraction:
                template = HARD_NEGATIVE_TEMPLATES[int(rng.integers(len(HARD_NEGATIVE_TEMPLATES)))]
                body = ""
            else:
                template = NEGATIVE_TEMPLATES[int(rng.integers(len(NEGATIVE_TEMPLATES)))]
                body = NEGATIVE_BODIES[int(rng.integers(len(NEGATIVE_BODIES)))]
        else:
            truth = 0
            template = NEUTRAL_TEMPLATES[int(rng.integers(len(NEUTRAL_TEMPLATES)))]
            body = NEUTRAL_BODIES[int(rng.integers(len(NEUTRAL_BODIES)))]

        day = dates[int(rng.integers(len(dates)))]
        ts = day + timedelta(
            hours=int(rng.integers(8, 20)), minutes=int(rng.integers(60))
        )
        items.append(
            NewsItem(
                id=f"news-{i:05d}",
                timestamp=ts.to_pydatetime(),
                headline=template.format(c=name),
                body=body,
                symbol=symbol,
                source="synthetic",
            )
        )
        truths.append(truth)
    order = np.argsort([it.timestamp for it in items], kind="stable")
    return [items[j] for j in order], [truths[j] for j in order]


def generate_prices(
    symbols: list[str],
    dates: pd.DatetimeIndex,
    items: list[NewsItem],
    truths: list[int],
    seed: int = 22,
    impact: float = 0.006,
    daily_vol: float = 0.012,
) -> pd.DataFrame:
    """Synthetic close prices where news polarity predicts forward returns.

    Net polarity of a symbol's news on day t adds ``impact * 0.7`` to its
    return on t+1 and ``impact * 0.3`` on t+2 (clipped at +/-3 net items),
    on top of seeded Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    sym_idx = {s: j for j, s in enumerate(symbols)}
    net = np.zeros((len(dates), len(symbols)))
    date_idx = {d: i for i, d in enumerate(dates)}
    for item, truth in zip(items, truths, strict=True):
        day = pd.Timestamp(item.timestamp).normalize()
        i = date_idx.get(day)
        j = sym_idx.get(item.symbol) if item.symbol else None
        if i is not None and j is not None:
            net[i, j] += truth
    net = np.clip(net, -3, 3)

    rets = rng.normal(0.0, daily_vol, size=(len(dates), len(symbols)))
    rets[1:] += impact * 0.7 * net[:-1]
    rets[2:] += impact * 0.3 * net[:-2]
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=pd.DatetimeIndex(dates, name="date"),
        columns=pd.Index(symbols, name="symbol"),
    )
    return close
