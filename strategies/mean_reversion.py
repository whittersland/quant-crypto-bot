"""
Mean Reversion Strategy
Identifies overbought/oversold conditions using Bollinger Bands and RSI
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import numpy as np

from .base_strategy import BaseStrategy, StrategyType
from ..utils.indicators import Signal, MarketData, TechnicalIndicators


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy
    
    Identifies mean reversion opportunities using:
    - Bollinger Bands for price extremes
    - RSI for momentum confirmation
    - Stochastic for timing
    
    Enters when price is at band extremes with confirming indicators.
    """
    
    def __init__(
        self,
        symbols: List[str],
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        stoch_oversold: float = 20,
        stoch_overbought: float = 80,
        weight: float = 0.15,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        super().__init__(
            name="mean_reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbols=symbols,
            weight=weight,
            enabled=enabled,
            params=params or {}
        )
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
    
    def get_required_history(self) -> int:
        return max(self.bb_period, self.rsi_period) + 30
    
    def analyze(self, symbol: str, market_data: MarketData) -> Optional[Signal]:
        closes = market_data.indicators.get("close", [])
        
        if len(closes) < self.get_required_history():
            return None
        
        current_price = closes[-1]
        
        # Get Bollinger Bands
        bb_upper = market_data.get_indicator("bb_upper")
        bb_lower = market_data.get_indicator("bb_lower")
        bb_middle = market_data.get_indicator("bb_middle")
        
        if not all([bb_upper, bb_lower, bb_middle]):
            return None
        
        # Calculate price position relative to bands
        band_width = bb_upper - bb_lower
        if band_width == 0:
            return None
        
        # Position: -1 = at lower band, 0 = middle, +1 = at upper band
        band_position = (current_price - bb_middle) / (band_width / 2)
        
        # Get RSI
        rsi = market_data.get_indicator("rsi")
        
        # Get Stochastic
        stoch_k = market_data.get_indicator("stoch_k")
        stoch_d = market_data.get_indicator("stoch_d")
        
        # Scoring for mean reversion signals
        long_score = 0
        short_score = 0
        
        # Bollinger Band signals
        if current_price <= bb_lower:
            long_score += 2  # Strong oversold
        elif band_position < -0.8:
            long_score += 1  # Near lower band
        
        if current_price >= bb_upper:
            short_score += 2  # Strong overbought
        elif band_position > 0.8:
            short_score += 1  # Near upper band
        
        # RSI signals
        if rsi:
            if rsi <= self.rsi_oversold:
                long_score += 2
            elif rsi <= self.rsi_oversold + 10:
                long_score += 1
            
            if rsi >= self.rsi_overbought:
                short_score += 2
            elif rsi >= self.rsi_overbought - 10:
                short_score += 1
        
        # Stochastic signals
        if stoch_k and stoch_d:
            if stoch_k <= self.stoch_oversold and stoch_d <= self.stoch_oversold:
                long_score += 1
                # Bonus for bullish crossover
                stoch_k_prev = market_data.get_indicator("stoch_k", 1)
                stoch_d_prev = market_data.get_indicator("stoch_d", 1)
                if stoch_k_prev and stoch_d_prev:
                    if stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev:
                        long_score += 1
            
            if stoch_k >= self.stoch_overbought and stoch_d >= self.stoch_overbought:
                short_score += 1
                # Bonus for bearish crossover
                stoch_k_prev = market_data.get_indicator("stoch_k", 1)
                stoch_d_prev = market_data.get_indicator("stoch_d", 1)
                if stoch_k_prev and stoch_d_prev:
                    if stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev:
                        short_score += 1
        
        # Determine direction
        max_score = 7  # Maximum possible score
        
        if long_score >= 3 and long_score > short_score:
            direction = "LONG"
            confidence = min(1.0, long_score / max_score)
        elif short_score >= 3 and short_score > long_score:
            direction = "SHORT"
            confidence = min(1.0, short_score / max_score)
        else:
            return None
        
        # Reduce confidence if conflicting signals
        if long_score > 0 and short_score > 0:
            confidence *= 0.8
        
        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "band_position": band_position,
                "rsi": rsi,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "long_score": long_score,
                "short_score": short_score,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "current_price": current_price
            }
        )
