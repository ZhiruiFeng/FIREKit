# VectorForge：为什么我们需要它以及如何在 FIREKit 中使用

## 简介

VectorForge 是 FIREKit 生态系统核心的基础回测引擎。本指南解释了 VectorForge 存在的原因、它解决的问题，以及它如何与其他 FIREKit 产品集成。

## 为什么我们需要 VectorForge？

### 问题：传统回测系统已经过时

量化交易面临四个关键挑战，而现有工具无法解决：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        传统回测的痛点                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  🐌 速度：事件驱动框架需要数小时才能完成参数扫描                              │
│  🎭 准确性：研究代码与生产代码不同，导致偏差                                  │
│  🔮 偏差：前瞻偏差和幸存者偏差会破坏回测结果                                  │
│  🔧 复杂性：机构级工具的学习曲线陡峭                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 1. 速度问题

传统的事件驱动回测非常缓慢：

| 操作 | 传统方式 | VectorForge | 提升倍数 |
|------|---------|-------------|----------|
| 10年日线回测 | 2.3 秒 | 0.05 秒 | **46倍** |
| 1000参数扫描 | 38 分钟 | 1.2 秒 | **1917倍** |
| 蒙特卡洛10k路径 | 6+ 小时 | 8 秒 | **2700倍** |

当测试1000个参数组合需要38分钟时，你无法快速迭代。研究成为瓶颈。

#### 2. 准确性问题

大多数回测工具只能优化速度或准确性，迫使你做出选择：

```
传统方法：
┌────────────────┐     ┌────────────────┐
│   研究代码      │ ──▶ │   生产代码      │  ← 需要重写代码
│  (向量化)      │     │  (事件驱动)     │  ← 不同的逻辑路径
└────────────────┘     └────────────────┘  ← 引入 bug
```

这导致"部署漂移" - 在研究中有效的策略在生产中失败，因为代码根本不同。

#### 3. 偏差问题

回测偏差会摧毁收益：

- **前瞻偏差**：意外使用未来数据（例如，用明天的收盘价决定今天的交易）
- **幸存者偏差**：只测试仍然存在的股票（忽略破产公司）
- **数据窥探**：对历史数据过度拟合参数

一个因前瞻偏差而有50%夏普比率的策略，在实盘中夏普比率将为0%。

#### 4. 复杂性问题

QuantConnect、Zipline 或 Backtrader 等机构级工具学习曲线陡峭，且与现代机器学习工作流（JAX、PyTorch、LLM）集成不佳。

### 解决方案：VectorForge 的混合架构

VectorForge 通过双模式设计解决所有四个问题：

```
┌─────────────────────────────────────────────────────────────────┐
│                        VectorForge                               │
├────────────────────────────┬────────────────────────────────────┤
│        向量化模式           │         事件驱动模式                │
├────────────────────────────┼────────────────────────────────────┤
│  ✓ NumPy/JAX 操作          │  ✓ 基于队列的架构                   │
│  ✓ 并行参数扫描            │  ✓ 真实的市场模拟                   │
│  ✓ 100万+交易/秒           │  ✓ 滑点和手续费模型                 │
│  ✓ GPU 加速               │  ✓ 订单簿动态                       │
│  ✓ 快速研究迭代            │  ✓ 与实盘交易代码完全一致            │
└────────────────────────────┴────────────────────────────────────┘

                    ↓ HybridRunner ↓

        快速研究 → 准确验证 → 自信部署
```

## VectorForge 如何融入 FIREKit 生态系统

### 架构位置

VectorForge 位于 FIREKit 金字塔的底部，提供所有其他产品构建所依赖的基本回测能力：

```
┌─────────────────────────────────────────────────────────────────┐
│                     投资组合仪表板                                │  ← 可视化
├─────────────────────────────────────────────────────────────────┤
│  PortfolioEngine  │   RiskGuard   │   ExecutionCore              │  ← 部署层
├─────────────────────────────────────────────────────────────────┤
│  SignalML  │  AlphaLab  │  SentimentPulse  │  DeepTrader         │  ← 智能层
├─────────────────────────────────────────────────────────────────┤
│                        DataStream                                │  ← 数据层
├─────────────────────────────────────────────────────────────────┤
│                       VectorForge ★                              │  ← 回测核心
└─────────────────────────────────────────────────────────────────┘
```

### 生态系统中的数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         完整的 FIREKit 数据流                             │
└──────────────────────────────────────────────────────────────────────────┘

  外部 API (Alpaca, Polygon, CoinGecko)
           │
           ▼
  ┌─────────────────┐
  │   DataStream    │ ──── 清洗、标准化的 OHLCV 数据
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │   VectorForge   │ ──── 使用这些数据回测策略
  └────────┬────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│AlphaLab │ │SignalML │ ──── 生成信号/特征
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
  ┌─────────────────┐
  │   RiskGuard     │ ──── 应用仓位管理和风险控制
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  ExecutionCore  │ ──── 在生产环境中执行交易
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ PortfolioEngine │ ──── 管理多策略投资组合
  └─────────────────┘
```

## 集成示例

### 1. DataStream → VectorForge（数据到回测）

DataStream 提供 VectorForge 使用的清洗、标准化数据：

```python
from datastream import DataStream
from vectorforge import VectorizedBacktester, MomentumStrategy

# 从 DataStream 加载清洗后的数据
data = DataStream.load(
    symbol='AAPL',
    start='2020-01-01',
    end='2024-01-01',
    adjust='split_and_dividend'  # 时点调整
)

# 使用 VectorForge 回测
backtester = VectorizedBacktester()
strategy = MomentumStrategy(lookback=20)
results = backtester.run(strategy, data, initial_capital=100000)

print(results.summary())
# 总收益率: 45.2%
# 夏普比率: 1.35
# 最大回撤: -12.4%
```

### 2. VectorForge + SignalML（机器学习驱动的策略）

在 VectorForge 回测中使用 SignalML 的机器学习模型：

```python
from signalml import load_ensemble
from vectorforge import EventDrivenBacktester, BaseStrategy

# 加载预训练的机器学习模型
model = load_ensemble("momentum_classifier")

class MLStrategy(BaseStrategy):
    """使用机器学习预测生成交易信号的策略。"""

    def __init__(self, model, threshold=0.6):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def on_bar(self, event):
        # 从当前K线提取特征
        features = self.extract_features(event.bar)

        # 获取机器学习预测
        prob_up = self.model.predict_proba(features)[0, 1]

        # 根据预测生成订单
        if prob_up > self.threshold and self.position <= 0:
            return self.create_order('BUY', quantity=100)
        elif prob_up < (1 - self.threshold) and self.position >= 0:
            return self.create_order('SELL', quantity=abs(self.position))

        return None

# 回测机器学习策略
backtester = EventDrivenBacktester(
    slippage_model='volume_dependent',
    commission_model='tiered'
)
results = backtester.run(MLStrategy(model), data)
```

### 3. VectorForge + AlphaLab（因子研究）

使用 AlphaLab 研究 alpha 因子，然后用 VectorForge 回测：

```python
from alphalab import FactorLibrary, Alpha101
from vectorforge import VectorizedBacktester

# 生成 alpha 因子
factor_lib = FactorLibrary()
momentum_factor = factor_lib.momentum(lookback=20)
mean_reversion_factor = Alpha101.alpha_042()  # 日内反转

# 组合因子
combined_alpha = 0.6 * momentum_factor + 0.4 * mean_reversion_factor

# 创建基于因子的策略
class FactorStrategy(BaseStrategy):
    def generate_signals(self, close, **kwargs):
        alpha = combined_alpha.compute(close)
        # 做多前10%，做空后10%
        return np.where(alpha > np.percentile(alpha, 90), 1,
                       np.where(alpha < np.percentile(alpha, 10), -1, 0))

# 快速参数扫描
from vectorforge import HybridRunner

runner = HybridRunner()
param_results = runner.run_batch(
    strategy_class=FactorStrategy,
    param_grid={
        'lookback': range(10, 60, 5),
        'threshold': [0.5, 0.6, 0.7, 0.8]
    },
    data=data
)
```

### 4. VectorForge + RiskGuard（风险控制回测）

在回测过程中应用风险管理：

```python
from vectorforge import EventDrivenBacktester
from riskguard import RiskManager

# 配置风险管理器
risk_manager = RiskManager(
    max_position_pct=0.10,      # 单个持仓最大10%
    max_drawdown=0.20,          # 回撤达20%时停止交易
    daily_loss_limit=0.03,      # 日亏损最大3%
    portfolio_heat=0.02         # 每笔交易组合风险2%
)

# 带风险控制的回测
backtester = EventDrivenBacktester()
backtester.set_risk_manager(risk_manager)

results = backtester.run(strategy, data)

# 结果现在包含风险调整后的指标
print(f"风险调整夏普: {results.risk_adjusted_sharpe}")
print(f"风控触发次数: {results.risk_stops}")
```

### 5. VectorForge → ExecutionCore（从研究到生产）

相同的策略代码在 VectorForge（回测）和 ExecutionCore（实盘）中都能工作：

```python
from vectorforge import EventDrivenBacktester
from executioncore import LiveExecutor

# 只需定义策略一次
class MyStrategy(BaseStrategy):
    def __init__(self, fast_period=10, slow_period=30):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_bar(self, event):
        fast_ma = self.data.close[-self.fast_period:].mean()
        slow_ma = self.data.close[-self.slow_period:].mean()

        if fast_ma > slow_ma and self.position <= 0:
            return self.create_order('BUY', quantity=100)
        elif fast_ma < slow_ma and self.position >= 0:
            return self.create_order('SELL', quantity=abs(self.position))
        return None

# 步骤1：使用 VectorForge 回测
backtester = EventDrivenBacktester()
backtest_results = backtester.run(MyStrategy(), historical_data)

if backtest_results.sharpe > 1.5:
    # 步骤2：使用 ExecutionCore 模拟交易
    executor = LiveExecutor(broker='alpaca', mode='paper')
    executor.run(MyStrategy(), live_data_stream)

    # 步骤3：使用相同的代码进行实盘交易
    executor = LiveExecutor(broker='alpaca', mode='live')
    executor.run(MyStrategy(), live_data_stream)
```

## 完整工作流程

以下是 VectorForge 如何融入完整的量化交易工作流程：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    使用 FIREKit 的完整量化工作流程                         │
└──────────────────────────────────────────────────────────────────────────┘

阶段1：数据准备
─────────────
DataStream → 加载和清洗历史数据
           → 处理公司行为（拆股、分红）
           → 创建时点宇宙（无幸存者偏差）

阶段2：研究（快速迭代）                              ← VectorForge
────────────────────                                  向量化模式
VectorForge (向量化) → 几秒内测试1000个参数
                     → 探索因子组合
                     → 找到有前景的策略候选

阶段3：验证（生产精度）                              ← VectorForge
────────────────────                                  事件驱动模式
VectorForge (事件驱动) → 使用真实执行验证
                       → 模拟滑点和手续费
                       → 滚动前向优化

阶段4：风险评估
─────────────
RiskGuard → 使用凯利公式确定仓位
          → 设置回撤限制
          → 配置熔断机制

阶段5：部署
─────────
ExecutionCore → 先进行模拟交易
              → 观察2-4周
              → 自信地上线实盘

阶段6：监控
─────────
PortfolioEngine → 跟踪多策略组合
                → 按需再平衡
                → 生成业绩报告
```

## 最佳实践

### 1. 始终使用混合方法

```python
from vectorforge import HybridRunner

runner = HybridRunner()

# 快速：测试1000个参数组合
params = runner.run_batch(strategy_class, param_grid, data)

# 准确：用真实执行验证前10个
for param in params[:10]:
    validated = runner.validate(strategy_class(**param), data)
```

### 2. 启用偏差保护

```python
from vectorforge.data import DataGuard, PointInTimeUniverse

# 防止前瞻偏差
guarded_data = DataGuard(data, current_idx)

# 防止幸存者偏差
universe = PointInTimeUniverse.sp500(date='2020-01-01')
```

### 3. 使用滚动前向优化

```python
from vectorforge.optimization import WalkForwardOptimizer

wfo = WalkForwardOptimizer(
    train_period=252,  # 1年训练期
    test_period=63,    # 1季度测试期
    anchored=False     # 滚动窗口
)

results = wfo.run(strategy_class, param_grid, data)
print(f"平均退化率: {results.avg_degradation:.2%}")
```

### 4. 使回测与生产执行匹配

```python
# 在回测和实盘中使用相同的执行模型
execution_config = {
    'slippage_model': 'volume_dependent',
    'slippage_bps': 5,
    'commission_model': 'ibkr_tiered',
}

# 回测
backtester = EventDrivenBacktester(**execution_config)

# 实盘（相同配置）
executor = LiveExecutor(broker='ibkr', **execution_config)
```

## 总结

VectorForge 对 FIREKit 至关重要，因为它：

1. **实现快速研究**：1000倍加速让你快速迭代
2. **确保生产精度**：事件驱动模式完全模拟实盘交易
3. **防止代价高昂的偏差**：内置防止前瞻偏差和幸存者偏差的保护
4. **无缝集成**：与 DataStream、SignalML、RiskGuard 和 ExecutionCore 完美配合
5. **支持现代机器学习**：JAX/GPU 加速支持基于机器学习的策略

没有 VectorForge，你无法在冒真金白银的风险之前验证策略。它是使 FIREKit 中其他一切成为可能的基础。

## 下一步

1. **安装 VectorForge**：`pip install vectorforge`
2. **运行你的第一个回测**：参见[快速入门指南](../vectorforge/README.md)
3. **连接 DataStream**：设置数据接入
4. **探索 AlphaLab**：研究 alpha 因子
5. **训练 SignalML 模型**：构建机器学习驱动的策略
6. **使用 ExecutionCore 部署**：自信地上线实盘

---

*详细的 API 文档，请参见 [VectorForge 技术规格](../products/01_vectorforge.md)*
