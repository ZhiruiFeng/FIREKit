# FIREKit 使用手册

> English version: [MANUAL.md](MANUAL.md)

FIREKit 是一个由九个可安装 Python 产品组成的 AI 量化交易工具生态，外加一个
自包含的可视化仪表盘（**Hub**）和两个顶层入口脚本（`validate_all.py`、
`pipeline.py`）将所有产品串联起来。

本手册涵盖：环境搭建、日常工作流、附带可运行代码的逐产品指南、端到端
流水线详解，以及故障排查。

---

## 1. 生态总览

| 层级 | 产品 | 功能 |
|------|------|------|
| 数据 | **DataStream** | 数据源、Parquet 存储、质量引擎、时点宇宙（point-in-time universe） |
| 研究 | **AlphaLab** | 20 因子库（含 Alpha101）、IC / 分位数评估 |
| 研究 | **SentimentPulse** | 金融词典打分器、可插拔 LLM 提供方、情绪冲击检测 |
| 信号 | **SignalML** | 模型库、带 purge 间隔的滚动验证、模型集成、模型注册表 |
| 信号 | **DeepTrader** | 类 gym 交易环境、Q-learning + REINFORCE 智能体、样本外评估 |
| 配置 | **PortfolioEngine** | 最小方差 / 最大夏普 / 风险平价 / HRP 优化器、有效前沿 |
| 风控 | **RiskGuard** | Kelly 仓位、波动率目标、熔断器、VaR/CVaR、敞口限制 |
| 执行 | **ExecutionCore** | 订单生命周期、模拟券商、TWAP/VWAP、执行损耗分析 |
| 回测 | **VectorForge** | 向量化 / 事件驱动混合回测引擎，支持多资产组合模式 |

数据自底向上流动：DataStream 喂给研究层（AlphaLab、SentimentPulse），研究层
喂给信号层（SignalML、DeepTrader），信号层再进入配置（PortfolioEngine）、
风控（RiskGuard）与执行（ExecutionCore）。VectorForge 独立验证任何策略。

每个产品的目录结构完全一致：

```
<product>/
  <product>/        # Python 包
    demo.py         # 确定性演示，输出 hub/data/<product>.json
  tests/            # pytest 测试套件
  pyproject.toml
  README.md         # 该产品的完整 API 文档
```

---

## 2. 安装

要求：**Python 3.11+**。所有产品共享同一套依赖（无需逐个安装产品）：

```bash
git clone https://github.com/ZhiruiFeng/FIREKit.git
cd FIREKit
pip install numpy pandas polars scipy scikit-learn pydantic pyarrow pyyaml pytest
```

无需 `pip install -e`；顶层脚本会自动处理路径，每个产品都可在自己的目录
内直接运行。

---

## 3. 核心工作流

### 3.1 一条命令验证全部

```bash
python3 validate_all.py
```

它分五层执行，对每项检查打印 PASS/FAIL（全程约 2 分钟，退出码 0 = 全绿）：

1. **environment** — Python 版本 + 必需第三方包
2. **smoke** — 每个产品都能导入并报告版本号
3. **tests** — 每个产品的 pytest 套件（约 1000 个测试）
4. **demos** — `run_all.py`：9 个演示 + Hub JSON 模式校验 + 打包
5. **pipeline** — `pipeline.py --fast`：跨产品集成运行

常用参数：

```bash
python3 validate_all.py --skip-tests                 # 快速健全性检查
python3 validate_all.py --only signalml,riskguard    # 只跑指定产品（第 2-3 层）
python3 validate_all.py --skip-demos --skip-pipeline # 只跑测试
```

任何改动前后、每次提交前都建议运行。

### 3.2 运行演示并打开仪表盘

```bash
python3 run_all.py                     # 全部产品演示 + 打包
python3 run_all.py --only alphalab     # 单个产品的演示
python3 run_all.py --bundle-only       # 仅用已有 JSON 重建 products.js
python3 -m http.server -d hub 8080     # 打开 http://localhost:8080
```

每个演示都是确定性的（固定随机种子），并按 `hub/SCHEMA.md` 约定输出
`hub/data/<product>.json`。Hub（`hub/index.html`）是零依赖原生 JS，
直接用 `file://` 打开也可以。

### 3.3 运行端到端流水线

```bash
python3 pipeline.py          # 完整运行（30 个标的、3 年数据，约 15 秒）
python3 pipeline.py --fast   # 精简运行（约 7 秒）
```

它在同一个合成股票宇宙上串联**全部九个产品**，并在 Hub 上新增第十个面板
（"End-to-End Pipeline"）。逐阶段详解见第 5 节。

### 3.4 测试单个产品

```bash
cd signalml && python3 -m pytest tests -q
cd signalml && python3 -m signalml.demo
```

---

## 4. 逐产品指南

每小节给出标准用法。所有代码片段都可在对应产品目录内直接运行（或像
`pipeline.py` 那样把产品目录加入 `sys.path`）。

### 4.1 DataStream — 数据管道

```python
from datastream import SyntheticSource, ParquetStore, QualityEngine, PointInTimeUniverse

source = SyntheticSource(n_symbols=30, start="2021-01-04", end="2023-12-29",
                         seed=11, issue_rate=0.01)
raw = source.fetch()                      # 长表：timestamp, symbol, OHLCV

store = ParquetStore("./market_data")     # 按 symbol 分区，zstd 压缩
store.write(raw)

clean, report = QualityEngine().run(store.read_many())
print(report.score, report.issue_counts())   # 质量得分 0-100，按类型统计问题

universe = PointInTimeUniverse()
universe.add("SYM001", "2021-01-04")          # 加入日期（退出日期可选）
survivorship_safe = universe.filter_frame(clean)
```

要点：
- 标准格式是**长表** DataFrame（`timestamp, symbol, open, high, low,
  close, volume`）；`normalize_frame` 负责归一化。
- 质量引擎会**修复**重复、NaN、非正价格和 OHLC violations，但对收益率
  异常值和数据缺口只做**标记** —— 被标记的异常值需自行处理（参见
  `pipeline.py` 中的 `mask_return_outliers`）。
- 真实数据用 `FileSource` 加载 CSV/Parquet 目录。

### 4.2 AlphaLab — 因子研究

```python
from alphalab import FactorZoo, Panel, Momentum, make_synthetic_panel

panel = Panel.from_long(clean_long_frame)       # 或 make_synthetic_panel(...)
report = FactorZoo.default().evaluate(panel, horizon=5, n_quantiles=5)

print(report.to_frame().head(10))     # 按 |rank-IC IR| 排序
best = report.best                    # FactorEvaluation
print(best.name, best.rank_ic.mean, best.rank_ic.ir, best.turnover)
print(report.top_correlated_pairs(3))  # 冗余因子候选

mom = Momentum(lookback=126).compute(panel)     # 单因子，dates x symbols 宽表
```

要点：`Panel` 是五张对齐的宽表（日期 × 标的）；因子值是宽表，可作为
SignalML 的特征，也可作为 VectorForge 的信号。

### 4.3 SentimentPulse — 新闻情绪

```python
from sentimentpulse import (LexiconProvider, AliasMap, scores_frame,
                            daily_scores, sentiment_index, detect_shocks, event_study)

provider = LexiconProvider()                    # 内置金融词典
results = provider.score_batch(headlines)       # 得分范围 [-1, 1]

frame = scores_frame(items, results)            # items: list[NewsItem]
daily = daily_scores(frame, dates=trading_days) # 宽表：date x symbol
index = sentiment_index(daily, halflife=5.0)    # 指数衰减加权情绪指数
shocks = detect_shocks(daily)                   # z-score 突变
study = event_study(shocks, close_prices)       # 冲击后的平均前向收益
```

要点：提供方可插拔（`SentimentProvider` 抽象基类）——对接真实 LLM 只需
实现 `score_batch`。`AliasMap` 把公司名/别名映射到代码，用于给未标注
新闻打标。

### 4.4 SignalML — 机器学习信号

```python
from signalml import (build_dataset, WalkForwardEngine, RidgeSignalModel,
                      GradientBoostingSignalModel, ic_weights, EnsembleCombiner,
                      summarize, ModelRegistry)

X, y = build_dataset(close_wide, volume_wide, horizon=5)   # (date, symbol) 索引

engine = WalkForwardEngine(
    {"ridge": lambda: RidgeSignalModel(alpha=10.0),
     "gbdt": lambda: GradientBoostingSignalModel()},
    train_size=252, test_size=21, gap=5,        # purge 间隔杜绝标签泄漏
)
result = engine.run(X, y)                       # 拼接后的样本外预测

weights = ic_weights(result.predictions, result.actuals)
signal = EnsembleCombiner("zscore_mean", weights=weights).combine(result.predictions)
print(summarize(signal, result.actuals))        # ic, rank_ic, hit_rate, spread

ModelRegistry("./models").save(result.models["ridge"], metadata={"ic": 0.02})
```

要点：所有数据都以 `(date, symbol)` MultiIndex 索引；`gap` 必须 ≥ 标签
预测期（horizon）以防泄漏；额外特征（比如情绪指数）直接作为 `X` 的新列
加入即可。

### 4.5 DeepTrader — 强化学习智能体

```python
from deeptrader import (regime_switching_series, train_test_split_envs,
                        Discretizer, QLearningAgent, train, evaluate, cost_sensitivity)

series = regime_switching_series(n=2600, seed=42)
train_env, test_env = train_test_split_envs(series.prices, cost_bps=5.0)

disc = Discretizer(n_bins=[3, 3, 3, 5, 3]).fit(train_env.observations)
agent = QLearningAgent(disc, alpha=0.1, gamma=0.5, seed=1)
train(agent, train_env, episodes=60)

result = evaluate(test_env, agent)     # 净值曲线、夏普、最大回撤、胜率
table = cost_sensitivity(test_env, agent)   # 0/5/10 bps 成本敏感性
```

要点：智能体的动作是仓位 {-1, 0, +1}；务必在样本外区间与内置基线
（`BuyAndHoldAgent`、`SMACrossoverAgent`、`RandomAgent`）对比 ——
表格型 RL 在噪声价格上经常跑输。

### 4.6 PortfolioEngine — 资产配置

```python
from portfolioengine import (RiskParity, MaxSharpe, Constraints,
                             ledoit_wolf_cov, efficient_frontier, run_backtest, make_universe)

cov = ledoit_wolf_cov(returns_window)          # 收缩估计，输出 np.ndarray
w = RiskParity(Constraints(long_only=True, max_weight=0.25)).allocate(None, cov)
w2 = MaxSharpe(Constraints(long_only=True)).allocate(mu, cov)   # 需要 mu

frontier = efficient_frontier(mu, cov, n_points=30)
result = run_backtest(returns, optimizers=None, lookback=252,
                      rebalance_every=21, cost_bps=5.0)   # 滚动对比
```

要点：所有优化器共享 `allocate(mu, cov) -> weights` 接口；解析型优化器
忽略 `mu`。基于 SLSQP 的优化器（MinVar、MaxSharpe）可精确执行行业上限
约束；启发式优化器通过截断再分配执行 `max_weight`。

### 4.7 RiskGuard — 风险管理

```python
from riskguard import (VolatilityTargeter, DrawdownCircuitBreaker,
                       kelly_from_moments, build_risk_report)

vt = VolatilityTargeter(target_vol=0.10, window=20, max_leverage=1.5)
scaled = vt.apply(strategy_returns)            # .scaled_returns, .exposure

breaker = DrawdownCircuitBreaker(max_drawdown=0.12, reentry_drawdown=0.04)
protected = breaker.apply(scaled.scaled_returns)   # .filtered_returns, .n_triggers

size = kelly_from_moments(mean=0.001, variance=0.0004, fraction=0.5)

report = build_risk_report(asset_returns, weights, target_vol=0.10)
print(report.to_dict()["var"]["historical"]["0.95"])
```

要点：波动率目标器与熔断器都是滞后一期生效（无前视偏差）；
`build_risk_report` 一次调用汇总 VaR/CVaR（3 种方法 × 2 个置信度）、
Kelly 建议、波动率目标敞口与限额检查。

### 4.8 ExecutionCore — 订单执行

```python
from executioncore import (PaperBroker, OrderManager, Order, OrderSide, PriceFeed,
                           FixedBpsSlippage, PerShareCommission, MaxOrderSize,
                           MaxNotional, TWAP, AlgoExecutor, run_session,
                           fill_rate, slippage_bps, synthetic_intraday_feed)

feed = synthetic_intraday_feed("ACME", n_bars=390, start_price=100.0, seed=7)
broker = PaperBroker(feed, slippage=FixedBpsSlippage(2.0),
                     commission=PerShareCommission(0.005), participation_cap=0.25)
oms = OrderManager(broker, cash=1_000_000,
                   validators=(MaxOrderSize(10_000), MaxNotional(500_000)))

order = Order(symbol="ACME", side=OrderSide.BUY, quantity=5_000)
run_session(broker, submissions={0: [order]}, oms=oms)

print(fill_rate(oms.orders.values()), slippage_bps(order), oms.snapshot().equity)
```

要点：成交严格遵循因果律（订单绝不会在提交时已知的 K 线上成交）；
`participation_cap` 产生真实的部分成交；定时执行用
`AlgoExecutor(TWAP(...), ...)`，TWAP 与 VWAP 的对比用 `algo_comparison`。

### 4.9 VectorForge — 回测

```python
from vectorforge import (PortfolioData, MissingDataPolicy, VectorizedBacktester,
                         TargetWeights, CrossSectionalSignal, Rebalancer)

data = PortfolioData.from_dict({sym: ohlcv_df for sym, ohlcv_df in frames.items()})
data = data.align(policy=MissingDataPolicy.FORWARD_FILL)

# 方式 A：内置横截面信号
momentum = CrossSectionalSignal.momentum(lookback=126)
weights = momentum.generate(data).top_percentile(20)

# 方式 B：自定义权重表（n_symbols x n_dates 数组）
weights = TargetWeights.from_array(w_matrix, list(data.symbols), data.dates)

result = VectorizedBacktester().run_portfolio(
    strategy=weights, data=data, initial_capital=1_000_000)
print(result.total_return, result.sharpe_ratio, result.max_drawdown)
print(result.equity_curve, result.turnover_history)
```

要点：VectorForge 是**独立裁判** —— 无论权重来自 ML、RL 还是优化器，
`run_portfolio` 都从原始价格重新计算净值曲线。流水线正是用它来交叉
验证手工计算的收益。

---

## 5. 端到端流水线逐阶段详解

`pipeline.py` 是产品集成的参考实现 —— 自己写跨产品代码时请先读它。
各阶段的输入与输出：

| # | 产品 | 输入 | 输出 |
|---|------|------|------|
| 1 | DataStream | 固定种子合成数据源（1% 故意污染） | 清洗后的长表 + 质量得分 |
| 2 | AlphaLab | `Panel.from_long(clean)`（已屏蔽异常值） | 20 因子排名报告 |
| 3 | SentimentPulse | 同一批标的的合成新闻 | 每日情绪指数（日期 × 标的） |
| 4 | SignalML | 价量特征 **+ 情绪列**，ridge + GBDT 滚动验证 | 样本外集成信号 |
| 5 | PortfolioEngine | 每 21 天按信号选前 8 名，Ledoit-Wolf 协方差 | 风险平价权重表 |
| 6 | RiskGuard | 每日策略收益（权重滞后 1 天） | 波动率目标化 + 熔断后的收益、VaR/CVaR 报告 |
| 7 | ExecutionCore | 首次调仓以市价单进入日内模拟交易 | 成交率、滑点、佣金 |
| 8 | DeepTrader | 以策略自身净值曲线作为交易环境 | RL 择时基准 vs 买入持有 |
| 9 | VectorForge | 同一份权重表 + 原始 OHLCV | 独立净值曲线、夏普、最大回撤 |

两个值得照搬到真实系统的细节：

- **全程无前视偏差**：信号使用 ≥ horizon 的 purge 间隔；第 t 天决定的
  权重从 t+1 天开始产生收益；波动率目标与熔断器滞后一期；模拟券商
  不会在提交 K 线上成交。
- **独立验证**：第 9 阶段从价格重新计算绩效，必须与第 6 阶段手工计算
  的数字一致（实际误差仅在舍入级别）—— 如果不一致，说明有 bug。

绩效数字预期是*平淡的*：宇宙是接近零漂移的 GBM，诚实的流水线只会产出
接近持平的策略。重点是管道本身，不是 alpha。

---

## 6. Hub 仪表盘

- `hub/index.html` + `hub/app.js`：零依赖渲染器，读取
  `hub/data/products.js`。
- `hub/SCHEMA.md`：每个演示输出的 JSON 契约（schema v1）——
  `summary_metrics`、`charts`（≤500 个点、≤5 条序列）、`tables`、
  `notes`；不允许 NaN/Infinity。
- `run_all.py` 在打包前会按 schema 校验每个 `hub/data/*.json`，
  违规会显式报错。

添加自定义面板：按 schema 写一个 JSON 文件放进 `hub/data/`，然后执行
`python3 run_all.py --bundle-only`。

---

## 7. 开发工作流

1. 开分支，在对应产品的包内做修改。
2. 在 `<product>/tests/` 中新增/调整测试（项目宪章要求测试先行；
   覆盖率门槛 70%）。
3. `cd <product> && python3 -m pytest tests -q`
4. `python3 validate_all.py` —— 五层全绿。
5. 改了演示或 hub schema 的话，肉眼检查仪表盘：
   `python3 run_all.py && python3 -m http.server -d hub 8080`。
6. 提交。规格驱动的功能开发使用 `speckit.*` 技能和 `specs/<feature>/`
   文档；`VALIDATION.yaml` 定义 VectorForge 的 CI 契约。

---

## 8. 故障排查

| 现象 | 原因 / 解决 |
|------|------------|
| `ModuleNotFoundError: numpy` 等 | 按第 2 节安装依赖。 |
| 自己的脚本里 `ModuleNotFoundError: <product>` | 产品包未 pip 安装；在产品目录内运行，或把产品目录加入 `sys.path`（参见 `pipeline.py` 开头）。 |
| 演示报图表长度错误 | Hub schema 限制图表最多 500 个 x 点 —— 需降采样（参见 `pipeline.py` 的 `downsample()`）。 |
| `run_all.py` 报 `non-finite literal` | JSON 里有 NaN/Infinity；换成 `null` 或丢弃这些点。 |
| 合成数据上回测收益离谱 | 未修复的收益率异常值 —— 质量引擎只做标记；需屏蔽尖刺（参见 `pipeline.py` 的 `mask_return_outliers`）。 |
| SignalML 演示偏慢（约 35 秒） | GBDT 模型的置换重要性开销大；像 `pipeline.py` 那样调小 `max_iter` / `importance_sample`。 |
| Hub 页面空白 | 缺少 `hub/data/products.js` —— 先运行 `python3 run_all.py`。 |
| 滚动验证 IC 好得反常 | 检查 `WalkForwardEngine` 的 `gap >= horizon`；间隔太小会泄漏标签。 |

---

## 9. 延伸阅读

- 产品深入文档：各 `<product>/README.md` 与 `docs/products/*.md`
- 架构与路线图：`docs/ECOSYSTEM_OVERVIEW.md`
- VectorForge 生态指南（EN/中文）：`docs/guides/`
- Hub 数据契约：`hub/SCHEMA.md`
