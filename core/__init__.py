"""
Trading System Core Module
"""

from .coinbase_client import (
    CoinbaseClient,
    SimulatedCoinbaseClient,
    CoinbaseWebSocket,
    OrderResponse,
    Position,
    Candle
)
from .risk_manager import (
    RiskManager,
    RiskMetrics,
    RiskLevel,
    TrailingStopManager,
    DailyLossTracker,
    DrawdownProtection,
    TakeProfitManager
)
from .position_sizer import AdaptivePositionSizer
from .liquidation_guard import LiquidationGuard, LiquidationRisk
from .signal_aggregator import (
    SignalAggregator,
    AggregatedSignal,
    AggregationMethod
)
from .execution_engine import (
    ExecutionEngine,
    ExecutionResult,
    Trade,
    TradeType,
    OrderStatus
)
from .portfolio_manager import (
    PortfolioManager,
    PortfolioSnapshot,
    PerformanceMetrics
)
from .scheduler import (
    DualLoopScheduler,
    SchedulerConfig,
    LoopType,
    LoopStats
)
from .trading_system import (
    TradingSystem,
    run_trading_system
)
from .dynamic_symbols import (
    DynamicSymbolScanner,
    SymbolManager,
    SymbolScore,
    ScanConfig,
    ScanMetric
)
from .copy_trade_fetcher import (
    CopyTradeAggregator,
    TraderPosition,
    TraderProfile,
    SimulatedDataSource,
    ManualDataSource
)

__all__ = [
    # Client
    "CoinbaseClient",
    "SimulatedCoinbaseClient",
    "CoinbaseWebSocket",
    "OrderResponse",
    "Position",
    "Candle",
    # Risk Management
    "RiskManager",
    "RiskMetrics",
    "RiskLevel",
    "TrailingStopManager",
    "DailyLossTracker",
    "DrawdownProtection",
    "TakeProfitManager",
    # Position Sizing
    "AdaptivePositionSizer",
    # Liquidation
    "LiquidationGuard",
    "LiquidationRisk",
    # Signal Aggregation
    "SignalAggregator",
    "AggregatedSignal",
    "AggregationMethod",
    # Execution
    "ExecutionEngine",
    "ExecutionResult",
    "Trade",
    "TradeType",
    "OrderStatus",
    # Portfolio
    "PortfolioManager",
    "PortfolioSnapshot",
    "PerformanceMetrics",
    # Scheduler
    "DualLoopScheduler",
    "SchedulerConfig",
    "LoopType",
    "LoopStats",
    # Dynamic Symbols
    "DynamicSymbolScanner",
    "SymbolManager",
    "SymbolScore",
    "ScanConfig",
    "ScanMetric",
    # Copy Trading
    "CopyTradeAggregator",
    "TraderPosition",
    "TraderProfile",
    "SimulatedDataSource",
    "ManualDataSource",
    # Main System
    "TradingSystem",
    "run_trading_system"
]
