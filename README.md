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

| Product | Description | Priority | Documentation |
|---------|-------------|----------|---------------|
| **VectorForge** | High-performance backtesting engine with hybrid vectorized/event-driven architecture | P0 | [View](docs/products/01_vectorforge.md) |
| **DataStream** | Unified data pipeline for stocks, crypto, and alternative data | P0 | [View](docs/products/02_datastream.md) |
| **AlphaLab** | Factor mining and feature engineering workbench with Alpha101 | P1 | [View](docs/products/03_alphalab.md) |
| **SignalML** | ML model hub for signal generation (LightGBM, LSTM, TFT) | P1 | [View](docs/products/04_signalml.md) |
| **SentimentPulse** | LLM-powered sentiment analysis (FinBERT, GPT-4, FinGPT) | P1 | [View](docs/products/05_sentimentpulse.md) |
| **DeepTrader** | Reinforcement learning trading agents (PPO, SAC via FinRL) | P2 | [View](docs/products/06_deeptrader.md) |
| **ExecutionCore** | Live trading and order management (Alpaca, IBKR, CCXT) | P1 | [View](docs/products/07_executioncore.md) |
| **RiskGuard** | Position sizing (Kelly) and risk management | P0 | [View](docs/products/08_riskguard.md) |
| **PortfolioEngine** | Asset allocation and portfolio optimization | P2 | [View](docs/products/09_portfolioengine.md) |

See the full [Ecosystem Overview](docs/ECOSYSTEM_OVERVIEW.md) for architecture details and integration patterns.

## Quick Start

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
git clone https://github.com/your-org/FIREKit.git
cd FIREKit

# Install dependencies
pip install -r requirements.txt

# Start with VectorForge backtesting
python -m firekit.vectorforge.examples.quickstart
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
