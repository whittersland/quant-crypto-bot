"""
Momentum Strategy
Multi-timeframe momentum analysis with rate of change confirmation
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import numpy as np

from .base_strategy import BaseStrategy, StrategyType, TimeFrame
from ..utils.indicators import Signal, MarketData, TechnicalIndicators


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy
    
    Analyzes momentum across multiple lookback periods to identify
    strong directional moves. Combines:
    - Price momentum (rate of change)
    - RSI momentum
    - MACD momentum
    
    Signal generated when all momentum indicators align.
    """
    
    def __init__(
        self,
        symbols: List[str],
        lookback_periods: List[int] = None,
        rsi_period: int = 14,
        rsi_threshold_high: float = 60,
        rsi_threshold_low: float = 40,
        weight: float = 0.15,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        super().__init__(
            name="momentum",
            strategy_type=StrategyType.MOMENTUM,
            symbols=symbols,
            weight=weight,
            enabled=enabled,
            params=params or {}
        )
        
        self.lookback_periods = lookback_periods or [3, 5, 10, 20]
        self.rsi_period = rsi_period
        self.rsi_threshold_high = rsi_threshold_high
        self.rsi_threshold_low = rsi_threshold_low
    
    def get_required_history(self) -> int:
        return max(self.lookback_periods) + 50  # Extra for indicator warmup
    
    def analyze(self, symbol: str, market_data: MarketData) -> Optional[Signal]:
        closes = market_data.indicators.get("close", [])
        
        if len(closes) < self.get_required_history():
            return None
        
        # Calculate momentum across all periods
        momentum_signals = []
        
        for period in self.lookback_periods:
            roc = TechnicalIndicators.rate_of_change(closes, period)
            current_roc = roc[-1] if roc and not np.isnan(roc[-1]) else 0
            
            if current_roc > 1.0:  # >1% gain
                momentum_signals.append(1)
            elif current_roc < -1.0:  # >1% loss
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
        
        # Check RSI
        rsi = market_data.get_indicator("rsi")
        rsi_signal = 0
        
        if rsi:
            if rsi > self.rsi_threshold_high:
                rsi_signal = 1  # Bullish momentum
            elif rsi < self.rsi_threshold_low:
                rsi_signal = -1  # Bearish momentum
        
        # Check MACD
        macd_hist = market_data.get_indicator("macd_histogram")
        macd_hist_prev = market_data.get_indicator("macd_histogram", 1)
        macd_signal = 0
        
        if macd_hist and macd_hist_prev:
            if macd_hist > 0 and macd_hist > macd_hist_prev:
                macd_signal = 1
            elif macd_hist < 0 and macd_hist < macd_hist_prev:
                macd_signal = -1
        
        # Combine signals
        total_bullish = sum(1 for s in momentum_signals if s > 0)
        total_bearish = sum(1 for s in momentum_signals if s < 0)
        
        # Add RSI and MACD
        if rsi_signal > 0:
            total_bullish += 1
        elif rsi_signal < 0:
            total_bearish += 1
        
        if macd_signal > 0:
            total_bullish += 1
        elif macd_signal < 0:
            total_bearish += 1
        
        total_indicators = len(self.lookback_periods) + 2  # +2 for RSI and MACD
        
        # Determine direction and confidence
        if total_bullish >= total_indicators * 0.6:
            direction = "LONG"
            confidence = total_bullish / total_indicators
        elif total_bearish >= total_indicators * 0.6:
            direction = "SHORT"
            confidence = total_bearish / total_indicators
        else:
            return None
        
        # Boost confidence if all align
        if total_bullish == total_indicators or total_bearish == total_indicators:
            confidence = min(1.0, confidence * 1.15)
        
        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "momentum_bullish": total_bullish,
                "momentum_bearish": total_bearish,
                "rsi": rsi,
                "macd_histogram": macd_hist,
                "lookback_periods": self.lookback_periods
            }
        )
