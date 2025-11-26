"""
Base Strategy Module
Abstract base class for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

from ..utils.indicators import Signal, MarketData, TechnicalIndicators

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy classification types"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    DUAL_CONDITION = "dual_condition"  # Novel strategies requiring multiple confirmations
    COMPOSITE = "composite"


class TimeFrame(Enum):
    """Supported timeframes"""
    M1 = "ONE_MINUTE"
    M5 = "FIVE_MINUTE"
    M15 = "FIFTEEN_MINUTE"
    M30 = "THIRTY_MINUTE"
    H1 = "ONE_HOUR"
    H2 = "TWO_HOUR"
    H6 = "SIX_HOUR"
    D1 = "ONE_DAY"


@dataclass
class StrategyState:
    """Tracks strategy internal state"""
    last_signal: Optional[Signal] = None
    last_signal_time: Optional[datetime] = None
    signals_generated: int = 0
    successful_signals: int = 0
    total_pnl: float = 0.0
    active: bool = True
    cooldown_until: Optional[datetime] = None
    custom_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionResult:
    """Result of a single condition check"""
    name: str
    met: bool
    value: Any
    threshold: Any
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    
    All strategies must implement:
    - analyze(): Generate trading signals
    - get_required_history(): Specify data requirements
    
    Optional overrides:
    - validate_signal(): Additional signal validation
    - on_trade_opened(): Called when trade opens
    - on_trade_closed(): Called when trade closes
    """
    
    def __init__(
        self,
        name: str,
        strategy_type: StrategyType,
        symbols: List[str],
        weight: float = 1.0,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        self.name = name
        self.strategy_type = strategy_type
        self.symbols = symbols
        self.weight = weight
        self.enabled = enabled
        self.params = params or {}
        
        # State tracking per symbol
        self.states: Dict[str, StrategyState] = {
            symbol: StrategyState() for symbol in symbols
        }
        
        # Performance tracking
        self.total_signals = 0
        self.winning_signals = 0
        self.losing_signals = 0
        
        logger.info(f"Strategy initialized: {name} ({strategy_type.value})")
    
    @abstractmethod
    def analyze(self, symbol: str, market_data: MarketData) -> Optional[Signal]:
        """
        Analyze market data and generate trading signal
        
        Args:
            symbol: Trading symbol
            market_data: Market data with pre-calculated indicators
        
        Returns:
            Signal if conditions met, None otherwise
        """
        pass
    
    @abstractmethod
    def get_required_history(self) -> int:
        """
        Get minimum candles required for analysis
        
        Returns:
            Number of candles needed
        """
        pass
    
    def get_required_timeframes(self) -> List[TimeFrame]:
        """
        Get required timeframes for multi-timeframe strategies
        
        Override in subclasses that need multiple timeframes
        """
        return [TimeFrame.M1]
    
    def validate_signal(self, signal: Signal, market_data: MarketData) -> bool:
        """
        Additional signal validation
        
        Override to add custom validation logic
        """
        # Basic validation
        if signal.confidence < 0.05:
            return False
        
        if signal.direction == "NEUTRAL":
            return False
        
        return True
    
    def generate_signal(
        self,
        symbol: str,
        market_data: MarketData
    ) -> Optional[Signal]:
        """
        Main entry point for signal generation
        
        Handles state management and validation
        """
        if not self.enabled:
            return None
        
        state = self.states.get(symbol)
        if not state:
            state = StrategyState()
            self.states[symbol] = state
        
        if not state.active:
            return None
        
        # Check cooldown
        if state.cooldown_until and datetime.now(timezone.utc) < state.cooldown_until:
            return None
        
        # Check data requirements
        if len(market_data.candles) < self.get_required_history():
            logger.debug(f"{self.name}: Insufficient history for {symbol}")
            return None
        
        try:
            # Generate signal
            signal = self.analyze(symbol, market_data)
            
            if signal and self.validate_signal(signal, market_data):
                state.last_signal = signal
                state.last_signal_time = datetime.now(timezone.utc)
                state.signals_generated += 1
                self.total_signals += 1
                
                logger.info(
                    f"{self.name} signal: {symbol} {signal.direction} "
                    f"confidence={signal.confidence:.3f}"
                )
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"{self.name} error analyzing {symbol}: {e}")
            return None
    
    def on_trade_opened(self, symbol: str, entry_price: float, size: float) -> None:
        """Called when a trade is opened based on this strategy's signal"""
        state = self.states.get(symbol)
        if state:
            state.custom_state["open_trade"] = {
                "entry_price": entry_price,
                "size": size,
                "opened_at": datetime.now(timezone.utc).isoformat()
            }
    
    def on_trade_closed(self, symbol: str, exit_price: float, pnl: float) -> None:
        """Called when a trade is closed"""
        state = self.states.get(symbol)
        if state:
            state.total_pnl += pnl
            
            if pnl >= 0:
                state.successful_signals += 1
                self.winning_signals += 1
            else:
                self.losing_signals += 1
            
            state.custom_state.pop("open_trade", None)
    
    def set_cooldown(self, symbol: str, minutes: int) -> None:
        """Set cooldown period for a symbol"""
        from datetime import timedelta
        
        state = self.states.get(symbol)
        if state:
            state.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            logger.debug(f"{self.name}: {symbol} cooldown set for {minutes} minutes")
    
    def disable(self) -> None:
        """Disable the strategy"""
        self.enabled = False
        logger.info(f"Strategy disabled: {self.name}")
    
    def enable(self) -> None:
        """Enable the strategy"""
        self.enabled = True
        logger.info(f"Strategy enabled: {self.name}")
    
    def reset_state(self, symbol: str = None) -> None:
        """Reset strategy state"""
        if symbol:
            self.states[symbol] = StrategyState()
        else:
            for sym in self.symbols:
                self.states[sym] = StrategyState()
    
    def get_performance(self) -> Dict:
        """Get strategy performance metrics"""
        win_rate = (
            self.winning_signals / self.total_signals * 100 
            if self.total_signals > 0 else 0
        )
        
        total_pnl = sum(s.total_pnl for s in self.states.values())
        
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "enabled": self.enabled,
            "weight": self.weight,
            "total_signals": self.total_signals,
            "winning_signals": self.winning_signals,
            "losing_signals": self.losing_signals,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "symbols": {
                symbol: {
                    "signals": state.signals_generated,
                    "successful": state.successful_signals,
                    "pnl": state.total_pnl,
                    "active": state.active
                }
                for symbol, state in self.states.items()
            }
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.strategy_type.value})"


class DualConditionStrategy(BaseStrategy):
    """
    Base class for dual-condition (novel) strategies
    
    These strategies require multiple independent conditions to be met
    simultaneously before generating a signal.
    """
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        min_conditions: int = 2,
        weight: float = 1.0,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        super().__init__(
            name=name,
            strategy_type=StrategyType.DUAL_CONDITION,
            symbols=symbols,
            weight=weight,
            enabled=enabled,
            params=params
        )
        
        self.min_conditions = min_conditions
        self.condition_history: Dict[str, List[ConditionResult]] = {
            symbol: [] for symbol in symbols
        }
    
    @abstractmethod
    def evaluate_conditions(
        self, 
        symbol: str, 
        market_data: MarketData
    ) -> List[ConditionResult]:
        """
        Evaluate all conditions for the strategy
        
        Returns:
            List of ConditionResult objects
        """
        pass
    
    def analyze(self, symbol: str, market_data: MarketData) -> Optional[Signal]:
        """
        Analyze by evaluating all conditions
        
        Signal is generated only when minimum conditions are met
        and they agree on direction
        """
        conditions = self.evaluate_conditions(symbol, market_data)
        
        if not conditions:
            return None
        
        # Store for debugging
        self.condition_history[symbol] = conditions
        
        # Count conditions met per direction
        long_conditions = [c for c in conditions if c.met and c.direction == "LONG"]
        short_conditions = [c for c in conditions if c.met and c.direction == "SHORT"]
        
        long_score = sum(c.weight for c in long_conditions)
        short_score = sum(c.weight for c in short_conditions)
        
        # Determine direction
        if len(long_conditions) >= self.min_conditions and long_score > short_score:
            direction = "LONG"
            met_conditions = long_conditions
            score = long_score
        elif len(short_conditions) >= self.min_conditions and short_score > long_score:
            direction = "SHORT"
            met_conditions = short_conditions
            score = short_score
        else:
            return None
        
        # Calculate confidence based on conditions met and their weights
        max_possible_score = sum(c.weight for c in conditions)
        confidence = score / max_possible_score if max_possible_score > 0 else 0
        
        # Boost confidence if all conditions agree
        if len(met_conditions) == len(conditions):
            confidence = min(1.0, confidence * 1.2)
        
        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "conditions_met": len(met_conditions),
                "total_conditions": len(conditions),
                "condition_details": [
                    {
                        "name": c.name,
                        "met": c.met,
                        "direction": c.direction,
                        "value": str(c.value),
                        "threshold": str(c.threshold)
                    }
                    for c in conditions
                ]
            }
        )
    
    def get_condition_summary(self, symbol: str) -> Dict:
        """Get summary of last condition evaluation"""
        conditions = self.condition_history.get(symbol, [])
        
        return {
            "total_conditions": len(conditions),
            "met_conditions": len([c for c in conditions if c.met]),
            "long_conditions": len([c for c in conditions if c.met and c.direction == "LONG"]),
            "short_conditions": len([c for c in conditions if c.met and c.direction == "SHORT"]),
            "details": [
                {
                    "name": c.name,
                    "met": c.met,
                    "direction": c.direction
                }
                for c in conditions
            ]
        }


class StrategyRegistry:
    """
    Registry for managing multiple strategies
    """
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.weights: Dict[str, float] = {}
    
    def register(self, strategy: BaseStrategy) -> None:
        """Register a strategy"""
        self.strategies[strategy.name] = strategy
        self.weights[strategy.name] = strategy.weight
        logger.info(f"Registered strategy: {strategy.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a strategy"""
        if name in self.strategies:
            del self.strategies[name]
            del self.weights[name]
            logger.info(f"Unregistered strategy: {name}")
    
    def get(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name"""
        return self.strategies.get(name)
    
    def get_all(self) -> List[BaseStrategy]:
        """Get all registered strategies"""
        return list(self.strategies.values())
    
    def get_enabled(self) -> List[BaseStrategy]:
        """Get all enabled strategies"""
        return [s for s in self.strategies.values() if s.enabled]
    
    def get_by_type(self, strategy_type: StrategyType) -> List[BaseStrategy]:
        """Get strategies by type"""
        return [
            s for s in self.strategies.values() 
            if s.strategy_type == strategy_type
        ]
    
    def generate_all_signals(
        self, 
        symbol: str, 
        market_data: MarketData
    ) -> List[Signal]:
        """Generate signals from all enabled strategies"""
        signals = []
        
        for strategy in self.get_enabled():
            if symbol in strategy.symbols:
                signal = strategy.generate_signal(symbol, market_data)
                if signal:
                    signals.append(signal)
        
        return signals
    
    def get_weights(self) -> Dict[str, float]:
        """Get normalized strategy weights"""
        enabled_weights = {
            name: weight 
            for name, weight in self.weights.items()
            if self.strategies[name].enabled
        }
        
        total = sum(enabled_weights.values())
        if total > 0:
            return {k: v / total for k, v in enabled_weights.items()}
        
        return enabled_weights
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for all strategies"""
        return {
            name: strategy.get_performance()
            for name, strategy in self.strategies.items()
        }
    
    def disable_all(self) -> None:
        """Disable all strategies"""
        for strategy in self.strategies.values():
            strategy.disable()
    
    def enable_all(self) -> None:
        """Enable all strategies"""
        for strategy in self.strategies.values():
            strategy.enable()
