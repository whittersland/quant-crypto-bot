"""
Trading Strategies Module

Contains:
- Base strategy framework
- 5 Standard strategies (momentum, mean reversion, volatility, volume, trend)
- 3 Novel dual-condition strategies (confluence, cross-asset, temporal)
- 1 Quantum-inspired strategy (quantum trader scoring)
"""

from .base_strategy import (
    BaseStrategy,
    DualConditionStrategy,
    StrategyType,
    TimeFrame,
    StrategyState,
    ConditionResult,
    StrategyRegistry
)
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .volatility_breakout import VolatilityBreakoutStrategy
from .volume_analysis import VolumeAnalysisStrategy
from .trend_following import TrendFollowingStrategy

# Novel dual-condition strategies
from .confluence_convergence import ConfluenceConvergenceStrategy
from .cross_asset_resonance import CrossAssetResonanceStrategy
from .temporal_alignment import TemporalAlignmentStrategy

# Quantum-inspired copy trading strategy
from .quantum_trader_scoring import (
    QuantumTraderScoring,
    TraderState,
    EntanglementPair,
    create_sample_traders
)

__all__ = [
    # Base
    "BaseStrategy",
    "DualConditionStrategy",
    "StrategyType",
    "TimeFrame",
    "StrategyState",
    "ConditionResult",
    "StrategyRegistry",
    # Standard strategies
    "MomentumStrategy",
    "MeanReversionStrategy",
    "VolatilityBreakoutStrategy",
    "VolumeAnalysisStrategy",
    "TrendFollowingStrategy",
    # Novel dual-condition strategies
    "ConfluenceConvergenceStrategy",
    "CrossAssetResonanceStrategy",
    "TemporalAlignmentStrategy",
    # Quantum strategy
    "QuantumTraderScoring",
    "TraderState",
    "EntanglementPair",
    "create_sample_traders"
]


def create_all_strategies(
    symbols: list,
    config: dict = None,
    include_quantum: bool = True
) -> StrategyRegistry:
    """
    Factory function to create all strategies with default or custom config
    
    Args:
        symbols: List of trading symbols
        config: Optional configuration dict
        include_quantum: Whether to include quantum trader scoring
    
    Returns:
        StrategyRegistry with all strategies registered
    """
    config = config or {}
    registry = StrategyRegistry()
    
    # Standard strategies
    registry.register(MomentumStrategy(
        symbols=symbols,
        weight=config.get("momentum", {}).get("weight", 0.12)
    ))
    
    registry.register(MeanReversionStrategy(
        symbols=symbols,
        weight=config.get("mean_reversion", {}).get("weight", 0.12)
    ))
    
    registry.register(VolatilityBreakoutStrategy(
        symbols=symbols,
        weight=config.get("volatility_breakout", {}).get("weight", 0.12)
    ))
    
    registry.register(VolumeAnalysisStrategy(
        symbols=symbols,
        weight=config.get("volume_analysis", {}).get("weight", 0.08)
    ))
    
    registry.register(TrendFollowingStrategy(
        symbols=symbols,
        weight=config.get("trend_following", {}).get("weight", 0.12)
    ))
    
    # Novel dual-condition strategies
    registry.register(ConfluenceConvergenceStrategy(
        symbols=symbols,
        weight=config.get("confluence_convergence", {}).get("weight", 0.10)
    ))
    
    registry.register(CrossAssetResonanceStrategy(
        symbols=symbols,
        weight=config.get("cross_asset_resonance", {}).get("weight", 0.10)
    ))
    
    registry.register(TemporalAlignmentStrategy(
        symbols=symbols,
        weight=config.get("temporal_alignment", {}).get("weight", 0.10)
    ))
    
    # Quantum trader scoring (highest weight - copy trading signal)
    if include_quantum:
        quantum_strategy = QuantumTraderScoring(
            symbols=symbols,
            min_traders_for_signal=config.get("quantum", {}).get("min_traders", 3),
            collapse_threshold=config.get("quantum", {}).get("collapse_threshold", 0.6),
            weight=config.get("quantum", {}).get("weight", 0.14)  # High weight for copy trading
        )
        registry.register(quantum_strategy)
    
    return registry
