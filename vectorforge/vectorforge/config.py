"""
VectorForge Configuration Module

Defines configuration schemas and defaults for the backtesting engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class EngineMode(str, Enum):
    """Backtesting engine mode."""
    VECTORIZED = "vectorized"
    EVENT_DRIVEN = "event_driven"
    HYBRID = "hybrid"


class Backend(str, Enum):
    """Computation backend for vectorized mode."""
    NUMPY = "numpy"
    JAX = "jax"
    NUMBA = "numba"


class Device(str, Enum):
    """Computation device."""
    CPU = "cpu"
    GPU = "gpu"


class QueueType(str, Enum):
    """Event queue type for event-driven mode."""
    FIFO = "fifo"
    PRIORITY = "priority"


class TimeResolution(str, Enum):
    """Time resolution for event-driven simulation."""
    SECOND = "second"
    MINUTE = "minute"
    DAILY = "daily"


class SlippageModelType(str, Enum):
    """Slippage model types."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLUME_DEPENDENT = "volume_dependent"
    ALMGREN_CHRISS = "almgren_chriss"


class CommissionModelType(str, Enum):
    """Commission model types."""
    ZERO = "zero"
    FIXED = "fixed"
    PER_SHARE = "per_share"
    PERCENTAGE = "percentage"
    TIERED = "tiered"


class CrossValidationMethod(str, Enum):
    """Cross-validation methods."""
    PURGED_KFOLD = "purged_kfold"
    COMBINATORIAL_PURGED = "combinatorial_purged"
    WALK_FORWARD = "walk_forward"


class VectorizedConfig(BaseModel):
    """Configuration for vectorized backtesting mode."""
    backend: Backend = Backend.JAX
    device: Device = Device.CPU
    precision: str = "float32"
    parallel_strategies: int = Field(default=1, ge=1, description="Number of strategies to test in parallel")


class EventDrivenConfig(BaseModel):
    """Configuration for event-driven backtesting mode."""
    queue_type: QueueType = QueueType.PRIORITY
    time_resolution: TimeResolution = TimeResolution.MINUTE
    process_after_close: bool = True


class SlippageConfig(BaseModel):
    """Slippage model configuration."""
    model: SlippageModelType = SlippageModelType.VOLUME_DEPENDENT
    base_bps: float = Field(default=5.0, ge=0, description="Base slippage in basis points")
    impact_factor: float = Field(default=0.1, ge=0, description="Market impact factor")


class CommissionConfig(BaseModel):
    """Commission model configuration."""
    model: CommissionModelType = CommissionModelType.TIERED
    min_commission: float = Field(default=1.0, ge=0, description="Minimum commission per order")
    per_share: float = Field(default=0.005, ge=0, description="Commission per share")
    max_pct: float = Field(default=0.01, ge=0, le=1, description="Maximum commission as percentage of trade")


class ExecutionConfig(BaseModel):
    """Execution simulation configuration."""
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    commission: CommissionConfig = Field(default_factory=CommissionConfig)
    market_impact: bool = True
    fill_ratio: float = Field(default=0.95, ge=0, le=1, description="Expected fill ratio")


class WalkForwardConfig(BaseModel):
    """Walk-forward validation configuration."""
    train_period: int = Field(default=252, ge=1, description="Training period in bars")
    test_period: int = Field(default=63, ge=1, description="Testing period in bars")
    min_trades: int = Field(default=30, ge=1, description="Minimum trades for valid period")
    anchored: bool = False


class CrossValidationConfig(BaseModel):
    """Cross-validation configuration."""
    method: CrossValidationMethod = CrossValidationMethod.PURGED_KFOLD
    n_splits: int = Field(default=5, ge=2, description="Number of CV splits")
    embargo_period: int = Field(default=5, ge=0, description="Embargo period between train/test")


class ValidationConfig(BaseModel):
    """Validation configuration."""
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    cross_validation: CrossValidationConfig = Field(default_factory=CrossValidationConfig)


class VectorForgeConfig(BaseModel):
    """Main VectorForge configuration."""
    mode: EngineMode = EngineMode.HYBRID
    default_capital: float = Field(default=100000.0, ge=0, description="Default starting capital")
    currency: str = "USD"
    vectorized: VectorizedConfig = Field(default_factory=VectorizedConfig)
    event_driven: EventDrivenConfig = Field(default_factory=EventDrivenConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VectorForgeConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    @classmethod
    def default(cls) -> "VectorForgeConfig":
        """Create default configuration."""
        return cls()
