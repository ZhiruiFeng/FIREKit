"""ExecutionCore: order management, paper-broker simulation, and execution analytics."""

from executioncore.algos import TWAP, VWAP, AlgoExecutor, ExecutionAlgo, run_session
from executioncore.analytics import (
    ShortfallReport,
    algo_comparison,
    executor_shortfall,
    fill_rate,
    implementation_shortfall,
    order_shortfall,
    slippage_bps,
)
from executioncore.broker import Broker, OrderEvent, PaperBroker
from executioncore.costs import (
    CommissionModel,
    FixedBpsSlippage,
    NoCommission,
    NoSlippage,
    PercentageCommission,
    PerShareCommission,
    SlippageModel,
    TieredCommission,
    VolumeImpactSlippage,
)
from executioncore.market import Bar, PriceFeed, synthetic_intraday_feed, u_shaped_volume_profile
from executioncore.oms import OrderManager, PortfolioSnapshot, Position
from executioncore.orders import (
    Fill,
    InvalidTransitionError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
    TimeInForce,
)
from executioncore.risk import (
    MaxNotional,
    MaxOrderSize,
    PriceBand,
    RiskCheckResult,
    RiskContext,
    RiskValidator,
)

__version__ = "0.1.0"

__all__ = [
    "TWAP",
    "VWAP",
    "AlgoExecutor",
    "Bar",
    "Broker",
    "CommissionModel",
    "ExecutionAlgo",
    "Fill",
    "FixedBpsSlippage",
    "InvalidTransitionError",
    "MaxNotional",
    "MaxOrderSize",
    "NoCommission",
    "NoSlippage",
    "Order",
    "OrderEvent",
    "OrderManager",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "OrderValidationError",
    "PaperBroker",
    "PercentageCommission",
    "PerShareCommission",
    "PortfolioSnapshot",
    "Position",
    "PriceBand",
    "PriceFeed",
    "RiskCheckResult",
    "RiskContext",
    "RiskValidator",
    "ShortfallReport",
    "SlippageModel",
    "TieredCommission",
    "TimeInForce",
    "VolumeImpactSlippage",
    "algo_comparison",
    "executor_shortfall",
    "fill_rate",
    "implementation_shortfall",
    "order_shortfall",
    "run_session",
    "slippage_bps",
    "synthetic_intraday_feed",
    "u_shaped_volume_profile",
]
