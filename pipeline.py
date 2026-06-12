#!/usr/bin/env python3
"""
FIREKit end-to-end pipeline: every product working together.

Chains all nine FIREKit products into one deterministic research-to-execution
workflow on a shared synthetic universe:

    1. DataStream        generate a 30-symbol universe, clean it
    2. AlphaLab          rank the 20-factor zoo on the cleaned panel
    3. SentimentPulse    score synthetic news into a daily sentiment index
    4. SignalML          walk-forward ML ensemble (price + sentiment features)
    5. PortfolioEngine   risk-parity allocation over the top-ranked names
    6. RiskGuard         vol targeting, circuit breaker, VaR/CVaR report
    7. ExecutionCore     paper-trade the first rebalance (TWAP-free market orders)
    8. DeepTrader        RL agent benchmark on the strategy's equity curve
    9. VectorForge       independent backtest cross-check of the ML strategy

Writes hub/data/pipeline.json (hub schema v1) so the run shows up on the
FIREKit Hub dashboard, and prints a stage-by-stage report.

Usage:
    python3 pipeline.py            # full run (~1-2 minutes)
    python3 pipeline.py --fast     # smaller universe / fewer windows (~30s)
"""

# ruff: noqa: E402  (product imports must follow the sys.path setup below)
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRODUCT_DIRS = [
    "vectorforge", "datastream", "alphalab", "signalml", "sentimentpulse",
    "deeptrader", "executioncore", "riskguard", "portfolioengine",
]
for _d in PRODUCT_DIRS:
    sys.path.insert(0, str(ROOT / _d))

import numpy as np
import pandas as pd

# --- product imports (one block per product, in pipeline order) -------------
from datastream import QualityEngine, SyntheticSource

from alphalab import FactorZoo, Panel

from sentimentpulse import (
    AliasMap,
    LexiconProvider,
    daily_scores,
    generate_news,
    scores_frame,
    sentiment_index,
)

from signalml import (
    EnsembleCombiner,
    GradientBoostingSignalModel,
    RidgeSignalModel,
    WalkForwardEngine,
    build_dataset,
    ic_weights,
    information_coefficient,
    rank_ic,
)

from portfolioengine import Constraints, RiskParity, ledoit_wolf_cov

from riskguard import (
    DrawdownCircuitBreaker,
    VolatilityTargeter,
    build_risk_report,
    max_drawdown as rg_max_drawdown,
    sharpe_ratio as rg_sharpe,
)

from executioncore import (
    FixedBpsSlippage,
    MaxNotional,
    MaxOrderSize,
    Order,
    OrderManager,
    OrderSide,
    PaperBroker,
    PerShareCommission,
    PriceFeed,
    fill_rate,
    run_session,
    slippage_bps,
    synthetic_intraday_feed,
)

from deeptrader import (
    BuyAndHoldAgent,
    Discretizer,
    QLearningAgent,
    evaluate as dt_evaluate,
    train as dt_train,
    train_test_split_envs,
)

from vectorforge import (
    MissingDataPolicy,
    PortfolioData,
    TargetWeights,
    VectorizedBacktester,
)

HUB_DATA = ROOT / "hub" / "data"
CAPITAL = 1_000_000.0


def banner(stage: int, product: str, title: str) -> None:
    print(f"\n[{stage}/9] {product:<16} {title}")


def downsample(x: list, *series: list, limit: int = 500) -> tuple[list, ...]:
    """Stride-sample parallel arrays so each has at most `limit` points."""
    step = max(1, math.ceil(len(x) / limit))
    return tuple([s[::step] for s in (x, *series)])


def mask_return_outliers(panel: Panel, threshold: float = 0.40) -> Panel:
    """Replace OHLCV cells on days whose close-to-close move exceeds
    ``threshold`` with NaN (treated as missing data downstream)."""
    spikes = panel.close.pct_change().abs() > threshold
    fields = {
        name: getattr(panel, name).mask(spikes)
        for name in ("open", "high", "low", "close", "volume")
    }
    return Panel(**fields)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="smaller universe, fewer windows")
    args = parser.parse_args()

    t0 = time.perf_counter()
    n_symbols = 20 if args.fast else 30
    start, end = ("2022-01-03", "2023-12-29") if args.fast else ("2021-01-04", "2023-12-29")
    horizon = 5
    top_k = 8
    rebalance_every = 21

    # ------------------------------------------------------------ 1 DataStream
    banner(1, "DataStream", "generate + clean the universe")
    source = SyntheticSource(n_symbols=n_symbols, start=start, end=end, seed=11, issue_rate=0.01)
    raw = source.fetch()
    clean, quality = QualityEngine().run(raw)
    print(f"      {len(raw):,} raw rows -> {len(clean):,} clean rows, "
          f"{quality.n_issues} issues ({quality.n_repaired} repaired), "
          f"quality score {quality.score:.1f}/100")

    # QualityEngine repairs structural issues but only *flags* return
    # outliers; mask them here (close jumps beyond +/-40% are synthetic
    # corruption, ~15 sigma in this universe) so they don't pollute returns.
    panel = mask_return_outliers(Panel.from_long(clean), threshold=0.40)

    # -------------------------------------------------------------- 2 AlphaLab
    banner(2, "AlphaLab", "rank the factor zoo")
    zoo = FactorZoo.default().evaluate(panel, horizon=horizon, n_quantiles=5)
    best = zoo.best
    top_factors = zoo.to_frame().head(5)
    print(f"      {len(zoo.evaluations)} factors on {panel.shape[1]} symbols x "
          f"{panel.shape[0]} days; best: {best.name} "
          f"(rank IC {best.rank_ic.mean:+.4f}, IR {best.rank_ic.ir:+.2f})")

    # -------------------------------------------------------- 3 SentimentPulse
    banner(3, "SentimentPulse", "score news into a daily sentiment index")
    symbols = panel.symbols
    alias_map = AliasMap(aliases={s: [f"{s} Corp", f"{s} Holdings Incorporated"] for s in symbols})
    items, truths = generate_news(
        symbols, alias_map, n_items=1500, start=start, n_days=len(panel.dates), seed=21
    )
    provider = LexiconProvider()
    results = provider.score_batch([f"{it.headline} {it.body}".strip() for it in items])
    senti = sentiment_index(daily_scores(scores_frame(items, results), dates=panel.dates))
    accuracy = float(np.mean([np.sign(r.score) == t for r, t in zip(results, truths)]))
    print(f"      {len(items)} news items scored (lexicon accuracy {accuracy:.0%}); "
          f"index covers {senti.notna().any(axis=1).sum()} days x {senti.shape[1]} symbols")

    # -------------------------------------------------------------- 4 SignalML
    banner(4, "SignalML", "walk-forward ML ensemble (price + sentiment features)")
    close = panel.close.ffill()
    volume = panel.volume.fillna(0.0)
    X, y = build_dataset(close, volume, horizon=horizon)
    senti_long = senti.stack()
    senti_long.index = senti_long.index.set_names(X.index.names)
    X = X.assign(senti_idx=senti_long.reindex(X.index).fillna(0.0))
    engine = WalkForwardEngine(
        {
            "ridge": lambda: RidgeSignalModel(alpha=10.0),
            "hist_gbdt": lambda: GradientBoostingSignalModel(max_iter=60, importance_sample=300),
        },
        train_size=252, test_size=21, gap=horizon,
    )
    wf = engine.run(X, y)
    weights_by_model = ic_weights(wf.predictions, wf.actuals)
    signal = EnsembleCombiner("zscore_mean", weights=weights_by_model).combine(wf.predictions)
    ens_ic = information_coefficient(signal, wf.actuals)
    ens_rank_ic = rank_ic(signal, wf.actuals)
    print(f"      {len(wf.windows)} windows, {len(X.columns)} features; ensemble OOS "
          f"IC {ens_ic:+.4f}, rank IC {ens_rank_ic:+.4f} "
          f"(weights {', '.join(f'{k}={v:.2f}' for k, v in weights_by_model.items())})")

    # ------------------------------------------------------- 5 PortfolioEngine
    banner(5, "PortfolioEngine", f"risk-parity allocation over top-{top_k} names")
    returns_wide = close.pct_change()
    pred_dates = signal.index.get_level_values("date").unique().sort_values()
    rebal_dates = list(pred_dates[::rebalance_every])
    optimizer = RiskParity(Constraints(long_only=True, max_weight=0.25))
    schedule: dict[pd.Timestamp, dict[str, float]] = {}
    for rd in rebal_dates:
        cross = signal.xs(rd, level="date").dropna()
        picks = list(cross.nlargest(top_k).index)
        hist = returns_wide[picks].loc[:rd].dropna().tail(126)
        w = optimizer.allocate(None, ledoit_wolf_cov(hist))
        schedule[rd] = dict(zip(picks, (float(v) for v in w)))
    print(f"      {len(schedule)} rebalances every {rebalance_every} trading days; "
          f"last holdings: {', '.join(sorted(schedule[rebal_dates[-1]]))}")

    # daily weight matrix: weights decided at rebalance date rd apply from rd+1
    weight_frame = pd.DataFrame(0.0, index=panel.dates, columns=symbols)
    for rd in rebal_dates:  # chronological, later rebalances overwrite
        weight_frame.loc[rd:, :] = 0.0
        for s, v in schedule[rd].items():
            weight_frame.loc[rd:, s] = v
    held = weight_frame.shift(1).fillna(0.0)
    strat_returns = (held * returns_wide.fillna(0.0)).sum(axis=1).loc[rebal_dates[0]:]

    # -------------------------------------------------------------- 6 RiskGuard
    banner(6, "RiskGuard", "vol targeting + circuit breaker + VaR/CVaR")
    vt = VolatilityTargeter(target_vol=0.10, window=20, max_leverage=1.5).apply(strat_returns)
    breaker = DrawdownCircuitBreaker(max_drawdown=0.12, reentry_drawdown=0.04)
    protection = breaker.apply(vt.scaled_returns)
    last_weights = schedule[rebal_dates[-1]]
    risk_rets = returns_wide[list(last_weights)].dropna().tail(252)
    risk_report = build_risk_report(risk_rets, last_weights, target_vol=0.10)
    raw_equity = (1.0 + strat_returns).cumprod()
    vt_equity = (1.0 + vt.scaled_returns).cumprod()
    prot_equity = (1.0 + protection.filtered_returns).cumprod()
    print(f"      raw Sharpe {rg_sharpe(strat_returns):.2f} / maxDD {rg_max_drawdown(raw_equity):.1%}"
          f"  ->  vol-targeted Sharpe {rg_sharpe(vt.scaled_returns):.2f}"
          f"  ->  protected maxDD {rg_max_drawdown(prot_equity):.1%}"
          f" ({protection.n_triggers} circuit-breaker triggers)")

    # ---------------------------------------------------------- 7 ExecutionCore
    banner(7, "ExecutionCore", "paper-trade the first rebalance")
    first_rd = rebal_dates[0]
    first_weights = schedule[first_rd]
    feed_bars = {}
    orders: list[Order] = []
    for i, (sym, w) in enumerate(sorted(first_weights.items())):
        px = float(close.loc[first_rd, sym])
        qty = max(1.0, round(w * CAPITAL / px))
        feed_bars[sym] = synthetic_intraday_feed(
            sym, n_bars=390, start_price=px, seed=100 + i
        ).bars(sym)
        orders.append(Order(symbol=sym, side=OrderSide.BUY, quantity=qty))
    broker = PaperBroker(
        PriceFeed(feed_bars),
        slippage=FixedBpsSlippage(2.0),
        commission=PerShareCommission(0.005),
        participation_cap=0.25,
    )
    oms = OrderManager(
        broker, cash=CAPITAL,
        validators=(MaxOrderSize(100_000), MaxNotional(0.6 * CAPITAL)),
    )
    # stagger submissions through the morning, 5 bars apart
    run_session(broker, submissions={i * 5: [o] for i, o in enumerate(orders)}, oms=oms)
    all_orders = list(oms.orders.values())
    fr = fill_rate(all_orders)
    slips = [s for s in (slippage_bps(o) for o in all_orders) if s is not None]
    avg_slip = float(np.mean(slips)) if slips else 0.0
    snap = oms.snapshot()
    print(f"      {len(orders)} orders, fill rate {fr:.1%}, avg slippage "
          f"{avg_slip:+.1f} bps, commission ${oms.total_commission:,.0f}, "
          f"end-of-day equity ${snap.equity:,.0f}")

    # ------------------------------------------------------------- 8 DeepTrader
    banner(8, "DeepTrader", "RL timing benchmark on the strategy equity curve")
    eq_prices = (raw_equity * 100.0).to_numpy()
    train_env, test_env = train_test_split_envs(eq_prices, cost_bps=5.0)
    disc = Discretizer(n_bins=[3, 3, 3, 5, 3]).fit(train_env.observations)
    agent = QLearningAgent(disc, alpha=0.1, gamma=0.5, seed=1)
    dt_train(agent, train_env, episodes=40)
    rl_eval = dt_evaluate(test_env, agent)
    bh_eval = dt_evaluate(test_env, BuyAndHoldAgent())
    print(f"      Q-learning OOS return {rl_eval.total_return:+.1%} "
          f"vs buy-and-hold {bh_eval.total_return:+.1%} on the strategy curve")

    # ------------------------------------------------------------ 9 VectorForge
    banner(9, "VectorForge", "independent backtest cross-check")
    frames = {
        s: pd.DataFrame(
            {
                "Open": panel.open[s], "High": panel.high[s], "Low": panel.low[s],
                "Close": panel.close[s], "Volume": panel.volume[s],
            },
            index=panel.dates,
        ).dropna()
        for s in symbols
    }
    pdata = PortfolioData.from_dict(frames).align(policy=MissingDataPolicy.FORWARD_FILL)
    w_matrix = np.zeros((len(pdata.symbols), len(pdata.dates)))
    held_aligned = held.reindex(pdata.dates).fillna(0.0)
    for i, s in enumerate(pdata.symbols):
        w_matrix[i, :] = held_aligned[s].to_numpy()
    backtester = VectorizedBacktester()
    vf_result = backtester.run_portfolio(
        strategy=TargetWeights.from_array(w_matrix, list(pdata.symbols), pdata.dates),
        data=pdata,
        initial_capital=CAPITAL,
    )
    vf_bench = backtester.run_portfolio(
        strategy=TargetWeights.equal_weight(list(pdata.symbols), pdata.dates),
        data=pdata,
        initial_capital=CAPITAL,
    )
    print(f"      ML strategy: total return {vf_result.total_return:+.1%}, "
          f"Sharpe {vf_result.sharpe_ratio:.2f}, maxDD {vf_result.max_drawdown:.1%}"
          f"  |  equal-weight benchmark: {vf_bench.total_return:+.1%}, "
          f"Sharpe {vf_bench.sharpe_ratio:.2f}")

    # ----------------------------------------------------------- hub JSON + exit
    elapsed = time.perf_counter() - t0
    payload = build_hub_payload(
        quality=quality, best=best, top_factors=top_factors, accuracy=accuracy,
        ens_ic=ens_ic, ens_rank_ic=ens_rank_ic, n_windows=len(wf.windows),
        schedule=schedule, strat_returns=strat_returns,
        raw_equity=raw_equity, vt_equity=vt_equity, prot_equity=prot_equity,
        protection=protection, risk_report=risk_report,
        fr=fr, avg_slip=avg_slip, commission=oms.total_commission,
        rl_eval=rl_eval, bh_eval=bh_eval,
        vf_result=vf_result, vf_bench=vf_bench, elapsed=elapsed,
    )
    HUB_DATA.mkdir(parents=True, exist_ok=True)
    out = HUB_DATA / "pipeline.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nPipeline complete in {elapsed:.1f}s -> {out.relative_to(ROOT)}")
    print("Rebuild the dashboard bundle with: python3 run_all.py --bundle-only")
    return 0


def build_hub_payload(**kw) -> dict:
    """Assemble the hub schema v1 JSON document for the pipeline run."""
    quality, best, risk_report = kw["quality"], kw["best"], kw["risk_report"]
    vf_result, vf_bench = kw["vf_result"], kw["vf_bench"]
    prot_equity = kw["prot_equity"]

    dates = [d.strftime("%Y-%m-%d") for d in kw["raw_equity"].index]
    x, raw_s, vt_s, prot_s = downsample(
        dates,
        [round(float(v), 4) for v in kw["raw_equity"]],
        [round(float(v), 4) for v in kw["vt_equity"]],
        [round(float(v), 4) for v in prot_equity],
    )

    bench_curve = (vf_bench.equity_curve / vf_bench.equity_curve.iloc[0]).reindex(
        kw["raw_equity"].index
    ).ffill().fillna(1.0)
    _, bench_s = downsample(dates, [round(float(v), 4) for v in bench_curve])

    var_95 = risk_report.to_dict()["var"]["historical"]["0.95"]
    cvar_95 = risk_report.to_dict()["cvar"]["historical"]["0.95"]

    top_rows = [
        [str(name), round(float(r["rank_ic_mean"]), 4), round(float(r["rank_ic_ir"]), 2),
         round(float(r["t_stat"]), 2), round(float(r["turnover"]), 3)]
        for name, r in kw["top_factors"].iterrows()
    ]

    return {
        "schema_version": 1,
        "product": "pipeline",
        "title": "End-to-End Pipeline",
        "tagline": "All nine products chained: data -> factors -> sentiment -> ML -> "
                   "allocation -> risk -> execution -> RL -> independent backtest",
        "version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "summary_metrics": [
            {"label": "Data Quality", "value": round(quality.score, 1), "unit": "/100"},
            {"label": "Best Factor Rank IC", "value": round(best.rank_ic.mean, 4), "unit": ""},
            {"label": "Ensemble OOS Rank IC", "value": round(kw["ens_rank_ic"], 4), "unit": ""},
            {"label": "Strategy Sharpe (vol-targeted)",
             "value": round(float(rg_sharpe_safe(kw["vt_equity"])), 2), "unit": ""},
            {"label": "Protected Max Drawdown",
             "value": round(float(rg_max_drawdown(prot_equity)) * 100, 1), "unit": "%"},
            {"label": "VaR 95 (hist)", "value": round(float(var_95) * 100, 2), "unit": "%"},
            {"label": "Fill Rate", "value": round(kw["fr"] * 100, 1), "unit": "%"},
            {"label": "VectorForge Sharpe", "value": round(vf_result.sharpe_ratio, 2), "unit": ""},
        ],
        "charts": [
            {
                "id": "equity",
                "title": "Strategy equity: raw vs vol-targeted vs protected vs benchmark",
                "type": "line",
                "x_label": "date",
                "y_label": "growth of $1",
                "x": x,
                "series": [
                    {"name": "raw", "data": raw_s},
                    {"name": "vol-targeted", "data": vt_s},
                    {"name": "protected", "data": prot_s},
                    {"name": "equal-weight bench", "data": bench_s},
                ],
            },
        ],
        "tables": [
            {
                "id": "stages",
                "title": "Stage-by-stage results",
                "columns": ["#", "Product", "Result"],
                "rows": [
                    [1, "DataStream", f"quality {quality.score:.1f}/100, "
                                      f"{quality.n_issues} issues, {quality.n_repaired} repaired"],
                    [2, "AlphaLab", f"best factor {best.name} "
                                    f"(rank IC {best.rank_ic.mean:+.4f}, IR {best.rank_ic.ir:+.2f})"],
                    [3, "SentimentPulse", f"lexicon accuracy {kw['accuracy']:.0%} on 1500 items"],
                    [4, "SignalML", f"{kw['n_windows']} walk-forward windows, "
                                    f"ensemble OOS IC {kw['ens_ic']:+.4f}"],
                    [5, "PortfolioEngine", f"{len(kw['schedule'])} risk-parity rebalances"],
                    [6, "RiskGuard", f"VaR95 {float(var_95):.2%}, CVaR95 {float(cvar_95):.2%}, "
                                     f"{kw['protection'].n_triggers} breaker triggers"],
                    [7, "ExecutionCore", f"fill rate {kw['fr']:.1%}, "
                                         f"slippage {kw['avg_slip']:+.1f} bps, "
                                         f"commission ${kw['commission']:,.0f}"],
                    [8, "DeepTrader", f"Q-learning OOS {kw['rl_eval'].total_return:+.1%} vs "
                                      f"buy-hold {kw['bh_eval'].total_return:+.1%}"],
                    [9, "VectorForge", f"strategy Sharpe {vf_result.sharpe_ratio:.2f} vs "
                                       f"benchmark {vf_bench.sharpe_ratio:.2f}"],
                ],
            },
            {
                "id": "top_factors",
                "title": "AlphaLab top factors (by |rank-IC IR|)",
                "columns": ["Factor", "Rank IC", "IR", "t-stat", "Turnover"],
                "rows": top_rows,
            },
        ],
        "notes": f"Deterministic synthetic run in {kw['elapsed']:.1f}s. The sentiment "
                 "feature exercises the SentimentPulse -> SignalML plumbing; on synthetic "
                 "data news does not drive prices, so its weight is learned near zero.",
    }


def rg_sharpe_safe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    return rg_sharpe(rets) if len(rets) > 2 else 0.0


if __name__ == "__main__":
    sys.exit(main())
