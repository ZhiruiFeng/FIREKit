"""Bundled financial sentiment lexicon (Loughran-McDonald style).

A curated subset of finance-oriented sentiment word lists written for this
package (not the original LM lists). Categories: positive, negative,
uncertainty, litigious, plus modifier lists (negators, intensifiers,
diminishers) used by the scorer.

All entries are lowercase single tokens.
"""

from __future__ import annotations

POSITIVE: frozenset[str] = frozenset({
    # performance / results
    "achieve", "achieved", "achieves", "advantage", "advantages", "beat", "beats",
    "benefit", "benefited", "benefits", "boost", "boosted", "boosts", "breakthrough",
    "buyback", "deliver", "delivered", "delivers", "exceed", "exceeded", "exceeds",
    "excellent", "exceptional", "expand", "expanded", "expansion", "gain", "gained",
    "gains", "grew", "grow", "growing", "growth", "high", "higher", "improve",
    "improved", "improvement", "improves", "increase", "increased", "increases",
    "innovative", "leading", "lucrative", "milestone", "momentum", "opportunity",
    "opportunities", "optimistic", "outperform", "outperformed", "outperforms",
    "positive", "profit", "profitability", "profitable", "profits", "progress",
    "raise", "raised", "raises", "rally", "rallied", "rebound", "rebounded",
    "record", "recover", "recovered", "recovery", "resilient", "reward", "rewarded",
    "robust", "soar", "soared", "soars", "solid", "strength", "strengthen",
    "strengthened", "strong", "stronger", "succeed", "succeeded", "success",
    "successful", "surge", "surged", "surges", "surpass", "surpassed", "tailwind",
    "tailwinds", "top", "topped", "tops", "upbeat", "upgrade", "upgraded",
    "upgrades", "upside", "upturn", "win", "winner", "winning", "wins",
    # quality / favorable conditions
    "attractive", "best", "bullish", "confident", "efficient", "encouraging",
    "favorable", "good", "great", "healthy", "impressive", "outstanding",
    "promising", "prosperous", "stable", "superior", "thrive", "thriving",
    "valuable", "vibrant",
})

NEGATIVE: frozenset[str] = frozenset({
    # performance / results
    "abandon", "abandoned", "adverse", "adversely", "bankrupt", "bankruptcy",
    "breach", "breached", "collapse", "collapsed", "collapses", "concern",
    "concerned", "concerning", "concerns", "contraction", "crash", "crashed",
    "crisis", "cut", "cuts", "damage", "damaged", "damages", "decline", "declined",
    "declines", "declining", "decrease", "decreased", "decreases", "deficit",
    "delay", "delayed", "delays", "deteriorate", "deteriorated", "deteriorating",
    "deterioration", "disappoint", "disappointed", "disappointing", "disappoints",
    "downgrade", "downgraded", "downgrades", "downturn", "drop", "dropped",
    "drops", "fail", "failed", "failing", "fails", "failure", "fall", "fallen",
    "falling", "falls", "fell", "halt", "halted", "headwind", "headwinds", "hurt",
    "impair", "impaired", "impairment", "insolvency", "insolvent", "lawsuit",
    "layoff", "layoffs", "lose", "loses", "losing", "loss", "losses", "lost",
    "low", "lower", "lowered", "miss", "missed", "misses", "negative", "plummet",
    "plummeted", "plunge", "plunged", "plunges", "poor", "pressure", "pressured",
    "problem", "problems", "recall", "recalled", "recalls", "recession", "risk",
    "risks", "risky", "selloff", "setback", "setbacks", "shortfall", "shrink",
    "shrinking", "shut", "shutdown", "slash", "slashed", "slow", "slowdown",
    "slowed", "slower", "slump", "slumped", "slumps", "struggle", "struggled",
    "struggles", "struggling", "suspend", "suspended", "suspension", "threat",
    "threats", "tumble", "tumbled", "tumbles", "turmoil", "underperform",
    "underperformed", "underperforms", "unprofitable", "warn", "warned",
    "warning", "warnings", "weak", "weaken", "weakened", "weakness", "worse",
    "worsen", "worsened", "worst", "writedown", "writeoff",
    # quality / unfavorable conditions
    "bad", "bearish", "bleak", "costly", "difficult", "dismal", "gloomy",
    "grim", "terrible", "troubled", "troubling", "uncompetitive", "unfavorable",
    "volatile",
})

UNCERTAINTY: frozenset[str] = frozenset({
    "almost", "ambiguity", "ambiguous", "anticipate", "anticipated", "anticipates",
    "appear", "appeared", "appears", "approximate", "approximately", "assume",
    "assumed", "assumes", "assumption", "assumptions", "believe", "believed",
    "believes", "cautious", "cautiously", "could", "depend", "depending",
    "depends", "doubt", "doubtful", "doubts", "estimate", "estimated", "estimates",
    "exposure", "fluctuate", "fluctuating", "fluctuation", "fluctuations",
    "indefinite", "indeterminate", "intangible", "may", "maybe", "might",
    "nearly", "occasionally", "pending", "perhaps", "possibility", "possible",
    "possibly", "precaution", "predict", "predicted", "prediction", "predicts",
    "preliminary", "presumably", "probable", "probably", "reassess", "reconsider",
    "revise", "revised", "roughly", "seem", "seemed", "seems", "sometime",
    "sometimes", "somewhat", "speculate", "speculation", "speculative", "tentative",
    "tentatively", "uncertain", "uncertainties", "uncertainty", "unclear",
    "unconfirmed", "undecided", "undetermined", "unexpected", "unknown",
    "unpredictable", "unproven", "unsettled", "unspecified", "untested", "variable",
    "variability", "vague",
})

LITIGIOUS: frozenset[str] = frozenset({
    "allegation", "allegations", "allege", "alleged", "allegedly", "alleges",
    "antitrust", "appeal", "appealed", "arbitration", "attorney", "attorneys",
    "claimant", "claimants", "complaint", "complaints", "contractual", "convict",
    "convicted", "conviction", "court", "courts", "defendant", "defendants",
    "deposition", "fine", "fined", "fines", "fraud", "fraudulent", "indict",
    "indicted", "indictment", "infringe", "infringed", "infringement", "injunction",
    "investigate", "investigated", "investigates", "investigation", "investigations",
    "judicial", "jury", "lawsuit", "lawsuits", "legal", "liability", "litigate",
    "litigated", "litigation", "misconduct", "negligence", "negligent", "penalties",
    "penalty", "plaintiff", "plaintiffs", "plea", "pleaded", "probe", "prosecute",
    "prosecuted", "prosecution", "regulator", "regulators", "regulatory", "ruling",
    "sanction", "sanctioned", "sanctions", "settle", "settled", "settlement",
    "settlements", "subpoena", "sue", "sued", "sues", "testimony", "tribunal",
    "verdict", "violate", "violated", "violation", "violations",
})

NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "nor", "without", "cannot", "can't",
    "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
    "weren't", "hardly", "barely", "fails", "failed",
})

INTENSIFIERS: frozenset[str] = frozenset({
    "very", "extremely", "highly", "significantly", "substantially", "sharply",
    "dramatically", "remarkably", "exceptionally", "considerably", "strongly",
    "massively", "hugely", "vastly", "deeply", "severely",
})

DIMINISHERS: frozenset[str] = frozenset({
    "slightly", "somewhat", "marginally", "mildly", "modestly", "moderately",
    "fractionally", "partially",
})


def lexicon_summary() -> dict[str, int]:
    """Word counts per category (used in docs/tests/demo)."""
    return {
        "positive": len(POSITIVE),
        "negative": len(NEGATIVE),
        "uncertainty": len(UNCERTAINTY),
        "litigious": len(LITIGIOUS),
        "negators": len(NEGATORS),
        "intensifiers": len(INTENSIFIERS),
        "diminishers": len(DIMINISHERS),
    }
