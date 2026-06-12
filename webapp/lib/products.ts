export type Layer = "foundation" | "intelligence" | "portfolio-risk" | "execution";

export interface Product {
  slug: string;
  name: string;
  tagline: string;
  blurb: string;
  version: string;
  layer: Layer;
  icon: string;
  accent: string; // tailwind-compatible hex accent
  features: string[];
  docFile: string; // file in content/products/
  pkg: string; // python package directory in the monorepo
  manualSection: string; // heading prefix of its section in MANUAL.md, e.g. "4.9"
}

export const LAYERS: Record<Layer, { label: string; description: string }> = {
  foundation: {
    label: "Foundation",
    description: "Data and backtesting infrastructure every other product builds on.",
  },
  intelligence: {
    label: "Intelligence",
    description: "Alpha research, ML signals, LLM sentiment, and RL agents.",
  },
  "portfolio-risk": {
    label: "Portfolio & Risk",
    description: "Turn signals into sized, risk-managed portfolios.",
  },
  execution: {
    label: "Execution",
    description: "Route portfolio decisions into orders and fills.",
  },
};

export const PRODUCTS: Product[] = [
  {
    slug: "vectorforge",
    name: "VectorForge",
    tagline: "High-performance backtesting engine with hybrid vectorized/event-driven architecture",
    blurb:
      "The foundational backtesting core of FIREKit. A dual-mode design pairs vectorized array simulation (up to 1000x faster than event-driven frameworks) for rapid research with an event-driven mode that mirrors live trading for production validation. v0.3.0 adds full multi-asset portfolio mode with cross-sectional signals and rebalancing.",
    version: "0.3.0",
    layer: "foundation",
    icon: "⚒️",
    accent: "#f97316",
    features: [
      "Hybrid architecture: vectorized mode for research, event-driven mode for production parity",
      "Multi-asset portfolio backtests with cross-sectional signals and rebalancing logic",
      "Walk-forward optimization, grid search, and purged K-fold validation",
      "Built-in protection against lookahead, survivorship, and data-snooping bias",
      "Portfolio analytics: Sharpe, drawdown, turnover, HHI concentration, diversification ratio",
      "JAX integration path for GPU acceleration and gradient-based optimization",
    ],
    docFile: "01_vectorforge.md",
    pkg: "vectorforge",
    manualSection: "4.9",
  },
  {
    slug: "datastream",
    name: "DataStream",
    tagline: "Unified data pipeline: sources, Parquet store, quality engine, point-in-time universe",
    blurb:
      "The single source of market data for the ecosystem. One canonical OHLCV schema across pluggable sources, a Parquet-backed local store with caching, a data-quality engine that detects and repairs bad data, and point-in-time universe membership that eliminates survivorship bias.",
    version: "0.1.0",
    layer: "foundation",
    icon: "🌊",
    accent: "#0ea5e9",
    features: [
      "Canonical long-format OHLCV schema shared by every FIREKit product",
      "Pluggable sources: seeded synthetic generator, CSV/Parquet files, API connector designs (Alpaca, Polygon, Binance, ...)",
      "ParquetStore: compressed columnar storage with read caching",
      "Quality engine: detects and repairs duplicates, bad prices, and OHLC violations; flags outliers and gaps",
      "PointInTimeUniverse prevents survivorship and lookahead bias in research",
    ],
    docFile: "02_datastream.md",
    pkg: "datastream",
    manualSection: "4.1",
  },
  {
    slug: "alphalab",
    name: "AlphaLab",
    tagline: "Factor mining workbench: 20-factor zoo (incl. Alpha101), IC/quantile evaluation",
    blurb:
      "A systematic alpha-discovery workbench. Ships a 20-factor zoo spanning momentum, reversal, volatility, volume, technical, beta, and Alpha101 formulaic alphas, plus a rigorous evaluation stack: IC and rank-IC time series, quantile portfolios, decay analysis, and factor-correlation redundancy checks.",
    version: "0.1.0",
    layer: "intelligence",
    icon: "🧪",
    accent: "#a855f7",
    features: [
      "20-factor built-in zoo including five WorldQuant Alpha101 formulaic alphas",
      "Cross-sectional and time-series operator library: cs_rank, ts_corr, delta, delay, ts_std, ...",
      "IC / rank-IC evaluation with t-stats, IR, and rolling analysis",
      "Quantile portfolio spreads and factor decay analysis",
      "Factor correlation matrix to prune redundant signals",
      "Custom Factor base class for rapid prototyping of new alphas",
    ],
    docFile: "03_alphalab.md",
    pkg: "alphalab",
    manualSection: "4.2",
  },
  {
    slug: "signalml",
    name: "SignalML",
    tagline: "ML signal hub: model zoo, purged walk-forward, ensembles, registry",
    blurb:
      "Production-grade machine learning for trading signals. A leakage-free feature pipeline feeds a model zoo through a purged walk-forward engine, ensembles combine per-date predictions, and a registry persists every trained model with metadata.",
    version: "0.1.0",
    layer: "intelligence",
    icon: "🤖",
    accent: "#22c55e",
    features: [
      "Feature pipeline: multi-horizon returns, realized vol, RSI, MA gaps, volume features — lookahead-free by construction",
      "Model zoo: gradient boosting, random forest, and ridge baselines behind one SignalModel interface",
      "Walk-forward engine with purged gap (gap ≥ label horizon) to prevent label leakage",
      "Ensembles: per-date z-score mean, rank mean, and IC-proportional weighting",
      "Model registry with pickle + JSON metadata persistence",
      "Evaluation: OOS IC, rank IC, hit rate, decile spread, feature importances",
    ],
    docFile: "04_signalml.md",
    pkg: "signalml",
    manualSection: "4.4",
  },
  {
    slug: "sentimentpulse",
    name: "SentimentPulse",
    tagline: "Financial sentiment: lexicon scorer, pluggable LLM providers, shock detection",
    blurb:
      "Extracts tradable sentiment from financial text. A bundled Loughran-McDonald-style lexicon with negation and intensifier handling powers an offline scorer, while a provider abstraction makes real LLMs pluggable. Aggregation produces per-symbol daily sentiment indices, momentum, and z-score shock detection.",
    version: "0.1.0",
    layer: "intelligence",
    icon: "📰",
    accent: "#eab308",
    features: [
      "Bundled financial lexicon (~470 words) with negators, intensifiers, and diminishers",
      "Rule-based scorer with exact, unit-tested negation handling (\"not good\" flips polarity)",
      "SentimentProvider abstraction: offline LexiconProvider today, real LLM providers pluggable",
      "News pipeline: entity-to-symbol tagging via alias maps, time-windowed deduplication",
      "Per-symbol daily sentiment index with exponential decay, momentum, and shock detection",
      "Event-study evaluation of forward returns after sentiment shocks",
    ],
    docFile: "05_sentimentpulse.md",
    pkg: "sentimentpulse",
    manualSection: "4.3",
  },
  {
    slug: "deeptrader",
    name: "DeepTrader",
    tagline: "RL trading agents: trading env, Q-learning + REINFORCE, OOS evaluation",
    blurb:
      "Reinforcement-learning trading agents with honest evaluation. A gym-like trading environment with transaction costs hosts tabular Q-learning and REINFORCE agents, trained on seeded regime-switching markets and evaluated strictly out-of-sample against buy-and-hold, SMA, and random baselines.",
    version: "0.1.0",
    layer: "intelligence",
    icon: "🧠",
    accent: "#ec4899",
    features: [
      "Gym-like TradingEnv: reset / step with reward = position x next return − cost x |Δposition|",
      "Agents: tabular Q-learning (epsilon-greedy with decay) and linear softmax REINFORCE",
      "Baselines: buy-and-hold, SMA crossover, and random agents for honest comparison",
      "Seeded regime-switching AR(1) market generator with genuinely learnable structure",
      "Strict out-of-sample evaluation with train/test environment separation",
      "Transaction-cost sensitivity analysis at 0 / 5 / 10 bps",
    ],
    docFile: "06_deeptrader.md",
    pkg: "deeptrader",
    manualSection: "4.5",
  },
  {
    slug: "portfolioengine",
    name: "PortfolioEngine",
    tagline: "Allocation: min-var/max-Sharpe/risk parity/HRP optimizers, efficient frontier",
    blurb:
      "Portfolio construction with six optimizers behind one allocate(mu, cov) interface: equal weight, inverse volatility, minimum variance, max Sharpe, exact risk parity, and Lopez de Prado's hierarchical risk parity. Plus shrinkage covariance estimators, efficient frontiers, and a rolling rebalancing backtest harness.",
    version: "0.1.0",
    layer: "portfolio-risk",
    icon: "⚖️",
    accent: "#14b8a6",
    features: [
      "Six optimizers: EqualWeight, InverseVolatility, MinimumVariance, MaxSharpe, RiskParity, HierarchicalRiskParity",
      "Exact equal-risk-contribution risk parity (Spinu cyclical coordinate descent, ~1e-10 precision)",
      "Covariance estimation: sample, Ledoit-Wolf shrinkage, EWMA (RiskMetrics-style)",
      "Efficient frontier via target-return sweep with warm starts",
      "Constraints: long-only, per-asset max weight, sector caps",
      "run_backtest: rolling re-estimation plus periodic rebalancing across all optimizers",
    ],
    docFile: "09_portfolioengine.md",
    pkg: "portfolioengine",
    manualSection: "4.6",
  },
  {
    slug: "riskguard",
    name: "RiskGuard",
    tagline: "Position sizing (Kelly), vol targeting, circuit breaker, VaR/CVaR, limits",
    blurb:
      "The risk layer that keeps strategies alive. Kelly-criterion position sizing (binary, moment-based, and multi-asset), volatility targeting with no lookahead, a stateful drawdown circuit breaker, VaR/CVaR in three flavors, and a limit engine enforcing position, sector, gross, and net caps.",
    version: "0.1.0",
    layer: "portfolio-risk",
    icon: "🛡️",
    accent: "#ef4444",
    features: [
      "Kelly sizing: binary outcome, continuous mu/sigma^2 approximation, fractional scaling, multi-asset",
      "VolatilityTargeter scales exposure to an annualized vol target from lagged realized vol",
      "DrawdownCircuitBreaker cuts exposure on drawdown breach, re-enters after recovery",
      "VaR and CVaR: historical, Gaussian, and Cornish-Fisher, plus rolling versions",
      "LimitEngine: per-position, sector, gross, and net exposure caps with violation reports",
      "build_risk_report assembles every metric in a single call",
    ],
    docFile: "08_riskguard.md",
    pkg: "riskguard",
    manualSection: "4.7",
  },
  {
    slug: "executioncore",
    name: "ExecutionCore",
    tagline: "Order management: paper broker, TWAP/VWAP, implementation shortfall analytics",
    blurb:
      "Production-grade order management and execution. A validated order state machine, a deterministic paper broker with bar-by-bar matching, TWAP/VWAP slicing algorithms, realistic slippage and commission models, pre-trade risk validators, and Perold implementation-shortfall analytics.",
    version: "0.1.0",
    layer: "execution",
    icon: "⚡",
    accent: "#6366f1",
    features: [
      "Full order lifecycle state machine: NEW → ACCEPTED → PARTIALLY_FILLED → FILLED / CANCELLED / REJECTED",
      "Market, limit, stop, and stop-limit orders with DAY / GTC / IOC / FOK time-in-force",
      "Deterministic PaperBroker with synthetic intraday feed and volume-participation caps",
      "TWAP and VWAP parent-order slicing algorithms",
      "Slippage (fixed bps, square-root impact) and commission (per-share, %, tiered) models",
      "Pre-trade validators and Perold implementation shortfall (delay / trading / opportunity)",
    ],
    docFile: "07_executioncore.md",
    pkg: "executioncore",
    manualSection: "4.8",
  },
];

export const PRODUCT_BY_SLUG = new Map(PRODUCTS.map((p) => [p.slug, p]));

export const LAYER_ORDER: Layer[] = ["foundation", "intelligence", "portfolio-risk", "execution"];
