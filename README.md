# FIREKit

**Build a set of toolkits for financial freedom**

FIREKit is a comprehensive ecosystem of interconnected tools for building AI-powered quantitative trading systems. Designed for individual developers with ML backgrounds, it democratizes algorithmic trading by combining open-source frameworks, modern AI capabilities, and institutional-grade architecture.

## Ecosystem Overview

```
                                    ┌───────────────────┐
                                    │   PORTFOLIO       │
                                    │   DASHBOARD       │
                                    └────────┬──────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
          ┌─────────▼─────────┐    ┌─────────▼─────────┐    ┌─────────▼─────────┐
          │   PortfolioEngine │    │    RiskGuard      │    │   ExecutionCore   │
          │   (Allocation)    │◄──►│   (Risk Mgmt)     │◄──►│   (Live Trading)  │
          └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
                    │                        │                        │
                    └────────────┬───────────┴────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │       SignalML          │
                    │    (Model Ensemble)     │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   SentimentPulse  │  │    DeepTrader     │  │     AlphaLab      │
│   (LLM Analysis)  │  │   (RL Agents)     │  │  (Factor Mining)  │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      DataStream         │
                    │    (Data Pipeline)      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      VectorForge        │
                    │   (Backtesting Core)    │
                    └─────────────────────────┘
```

## Products

| Product | Description | Status | Code | Documentation |
|---------|-------------|--------|------|---------------|
| **VectorForge** | High-performance backtesting engine with hybrid vectorized/event-driven architecture + multi-asset portfolio mode | ✅ v0.3.0 | [vectorforge/](vectorforge/) | [View](docs/products/01_vectorforge.md) |
| **DataStream** | Unified data pipeline: sources, Parquet store, quality engine, point-in-time universe | ✅ v0.1.0 | [datastream/](datastream/) | [View](docs/products/02_datastream.md) |
| **AlphaLab** | Factor mining workbench: 20-factor zoo (incl. Alpha101), IC/quantile evaluation | ✅ v0.1.0 | [alphalab/](alphalab/) | [View](docs/products/03_alphalab.md) |
| **SignalML** | ML signal hub: model zoo, purged walk-forward, ensembles, registry | ✅ v0.1.0 | [signalml/](signalml/) | [View](docs/products/04_signalml.md) |
| **SentimentPulse** | Financial sentiment: lexicon scorer, pluggable LLM providers, shock detection | ✅ v0.1.0 | [sentimentpulse/](sentimentpulse/) | [View](docs/products/05_sentimentpulse.md) |
| **DeepTrader** | RL trading agents: trading env, Q-learning + REINFORCE, OOS evaluation | ✅ v0.1.0 | [deeptrader/](deeptrader/) | [View](docs/products/06_deeptrader.md) |
| **ExecutionCore** | Order management: paper broker, TWAP/VWAP, implementation shortfall analytics | ✅ v0.1.0 | [executioncore/](executioncore/) | [View](docs/products/07_executioncore.md) |
| **RiskGuard** | Position sizing (Kelly), vol targeting, circuit breaker, VaR/CVaR, limits | ✅ v0.1.0 | [riskguard/](riskguard/) | [View](docs/products/08_riskguard.md) |
| **PortfolioEngine** | Allocation: min-var/max-Sharpe/risk parity/HRP optimizers, efficient frontier | ✅ v0.1.0 | [portfolioengine/](portfolioengine/) | [View](docs/products/09_portfolioengine.md) |

Every product is an installable Python package with its own test suite and a
deterministic demo pipeline that feeds the **[FIREKit Hub](hub/index.html)** —
a self-contained dashboard visualizing results from all nine products.

See the full [Ecosystem Overview](docs/ECOSYSTEM_OVERVIEW.md) for architecture details and integration patterns.

## Quick Start

```bash
# Run every product's test suite
for d in vectorforge datastream alphalab signalml sentimentpulse \
         deeptrader executioncore riskguard portfolioengine; do
  (cd "$d" && python3 -m pytest tests -q)
done

# Run all product demos and build the hub dashboard
python3 run_all.py

# View the hub (or just open hub/index.html in a browser)
python3 -m http.server -d hub 8080
```

## Roadmap

### Phase 1: Foundation (Months 1-3)
1. **VectorForge**: Build your first backtest
2. **DataStream**: Set up data ingestion (Alpaca, Polygon)
3. **RiskGuard**: Implement position sizing

### Phase 2: Intelligence (Months 4-6)
4. **AlphaLab**: Research alpha factors
5. **SignalML**: Train ML models
6. **SentimentPulse**: Add LLM sentiment

### Phase 3: Execution (Months 7-9)
7. **ExecutionCore**: Connect to paper trading
8. **PortfolioEngine**: Deploy allocation

### Phase 4: Advanced AI (Months 10-12)
9. **DeepTrader**: Train RL agents
10. Multi-strategy deployment

## Technology Stack

| Category | Technologies |
|----------|--------------|
| **Core** | Python 3.11+, Rust (PyO3), JAX, NumPy, Pandas, Polars |
| **ML/AI** | LightGBM, XGBoost, PyTorch, Stable-Baselines3, LangChain |
| **Financial NLP** | FinBERT, FinGPT, GPT-4, Claude |
| **Data** | Parquet, TimescaleDB, Redis, DuckDB |
| **Trading** | Alpaca, IBKR, CCXT, NautilusTrader |

## Key Features

- **1000x Faster Backtesting**: Vectorized operations with JAX/NumPy
- **Bias Prevention**: Point-in-time data, purged cross-validation
- **Production-Ready**: Identical backtest and live trading code
- **Cost Optimized**: Smart data source routing, model selection
- **Risk First**: Kelly sizing, circuit breakers, gradual derisking

## Target Metrics

| Metric | Target |
|--------|--------|
| Backtest Speed | 1M+ trades/sec |
| Signal IC | >0.03 |
| Sharpe Ratio | >1.5 |
| Max Drawdown | <20% |
| System Uptime | 99.9% |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ZhiruiFeng/FIREKit.git
cd FIREKit

# Install core dependencies
pip install numpy pandas polars scipy scikit-learn pydantic pyarrow pyyaml pytest

# Run the whole ecosystem and open the hub dashboard
python3 run_all.py
python3 -m http.server -d hub 8080   # then open http://localhost:8080
```

## Documentation

- [Ecosystem Overview](docs/ECOSYSTEM_OVERVIEW.md) - Architecture and integration
- [VectorForge](docs/products/01_vectorforge.md) - Backtesting engine
- [DataStream](docs/products/02_datastream.md) - Data pipeline
- [AlphaLab](docs/products/03_alphalab.md) - Factor research
- [SignalML](docs/products/04_signalml.md) - ML models
- [SentimentPulse](docs/products/05_sentimentpulse.md) - LLM analysis
- [DeepTrader](docs/products/06_deeptrader.md) - RL agents
- [ExecutionCore](docs/products/07_executioncore.md) - Live trading
- [RiskGuard](docs/products/08_riskguard.md) - Risk management
- [PortfolioEngine](docs/products/09_portfolioengine.md) - Portfolio optimization

## Recommended Learning Path

1. **Read**: "Advances in Financial Machine Learning" by Lopez de Prado
2. **Build**: Start with VectorForge backtesting
3. **Data**: Connect DataStream to Alpaca (free)
4. **Research**: Explore factors with AlphaLab
5. **Train**: Build models with SignalML
6. **Trade**: Go live with ExecutionCore

## Cost-Optimized Data Stack

| Budget | Stocks | Crypto | Alternative |
|--------|--------|--------|-------------|
| $0/mo | Alpha Vantage Free + yfinance | CoinGecko Demo + Binance | SEC EDGAR |
| $10-50/mo | Alpaca ($9) + Polygon ($29) | CoinGecko + exchanges | Finnhub |
| $100-300/mo | Polygon Advanced ($199) | CoinGecko Pro | Benzinga |

## Contributing

See individual product documentation for contribution guidelines.

## License

MIT License - See LICENSE for details.

---

**FIREKit**: From backtest to production, build your path to financial independence.
