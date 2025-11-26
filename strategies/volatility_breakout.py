"""
Volatility Breakout Strategy
Identifies breakouts from volatility compression using ATR and Bollinger Band squeeze
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import numpy as np

from .base_strategy import BaseStrategy, StrategyType
from ..utils.indicators import Signal, MarketData, TechnicalIndicators


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Breakout Strategy
    
    Identifies breakout opportunities using:
    - ATR for volatility measurement
    - Bollinger Band width for squeeze detection
    - Price breakouts from consolidation
    
    Enters when volatility expands after compression period.
    """
    
    def __init__(
        self,
        symbols: List[str],
        atr_period: int = 14,
        bb_period: int = 20,
        squeeze_percentile: float = 20,  # BB width below this percentile = squeeze
        breakout_multiplier: float = 1.5,  # ATR multiplier for breakout confirmation
        lookback_for_squeeze: int = 50,
        weight: float = 0.15,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        super().__init__(
            name="volatility_breakout",
            strategy_type=StrategyType.VOLATILITY,
            symbols=symbols,
            weight=weight,
            enabled=enabled,
            params=params or {}
        )
        
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.squeeze_percentile = squeeze_percentile
        self.breakout_multiplier = breakout_multiplier
        self.lookback_for_squeeze = lookback_for_squeeze
    
    def get_required_history(self) -> int:
        return self.lookback_for_squeeze + self.bb_period + 10
    
    def _calculate_bb_width(self, market_data: MarketData) -> List[float]:
        """Calculate Bollinger Band width history"""
        closes = market_data.indicators.get("close", [])
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            closes, self.bb_period, 2.0
        )
        
        widths = []
        for i in range(len(bb_upper)):
            if bb_middle[i] and bb_middle[i] > 0:
                width = (bb_upper[i] - bb_lower[i]) / bb_middle[i]
                widths.append(width)
            else:
                widths.append(np.nan)
        
        return widths
    
    def _is_in_squeeze(self, bb_widths: List[float]) -> bool:
        """Check if currently in volatility squeeze"""
        valid_widths = [w for w in bb_widths[-self.lookback_for_squeeze:] if not np.isnan(w)]
        
        if len(valid_widths) < 10:
            return False
        
        current_width = valid_widths[-1]
        threshold = np.percentile(valid_widths, self.squeeze_percentile)
        
        return current_width <= threshold
    
    def _detect_breakout(
        self, 
        market_data: MarketData,
        atr: float
    ) -> tuple:
        """
        Detect if price is breaking out
        
        Returns:
            (is_breakout, direction, strength)
        """
        closes = market_data.indicators.get("close", [])
        highs = market_data.indicators.get("high", [])
        lows = market_data.indicators.get("low", [])
        
        if len(closes) < 10:
            return False, None, 0
        
        current_close = closes[-1]
        prev_close = closes[-2]
        
        # Calculate recent range
        recent_high = max(highs[-10:-1])
        recent_low = min(lows[-10:-1])
        
        # Breakout threshold based on ATR
        breakout_threshold = atr * self.breakout_multiplier
        
        # Check for upside breakout
        if current_close > recent_high + breakout_threshold * 0.5:
            move = current_close - prev_close
            strength = move / atr if atr > 0 else 0
            return True, "LONG", strength
        
        # Check for downside breakout
        if current_close < recent_low - breakout_threshold * 0.5:
            move = prev_close - current_close
            strength = move / atr if atr > 0 else 0
            return True, "SHORT", strength
        
        return False, None, 0
    
    def analyze(self, symbol: str, market_data: MarketData) -> Optional[Signal]:
        closes = market_data.indicators.get("close", [])
        
        if len(closes) < self.get_required_history():
            return None
        
        # Get ATR
        atr = market_data.get_indicator("atr")
        if not atr or atr <= 0:
            return None
        
        # Calculate BB width
        bb_widths = self._calculate_bb_width(market_data)
        
        # Check for squeeze
        was_in_squeeze = self._is_in_squeeze(bb_widths[:-1])  # Previous state
        current_squeeze = self._is_in_squeeze(bb_widths)
        
        # Detect breakout
        is_breakout, direction, strength = self._detect_breakout(market_data, atr)
        
        if not is_breakout:
            return None
        
        # Calculate confidence
        confidence = 0.0
        
        # Base confidence from breakout strength
        confidence += min(0.4, strength * 0.2)
        
        # Bonus for breaking out of squeeze
        if was_in_squeeze and not current_squeeze:
            confidence += 0.3
        elif was_in_squeeze:
            confidence += 0.15
        
        # Check volume confirmation
        volume = market_data.get_indicator("volume")
        volume_sma = market_data.get_indicator("volume_sma")
        
        if volume and volume_sma and volume > volume_sma * 1.5:
            confidence += 0.2  # High volume confirmation
        elif volume and volume_sma and volume > volume_sma:
            confidence += 0.1
        
        # Check ADX for trend strength
        adx = market_data.get_indicator("adx")
        if adx and adx > 25:
            confidence += 0.1
        
        confidence = min(1.0, confidence)
        
        if confidence < 0.15:
            return None
        
        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "atr": atr,
                "breakout_strength": strength,
                "was_in_squeeze": was_in_squeeze,
                "current_bb_width": bb_widths[-1] if bb_widths else None,
                "volume_ratio": volume / volume_sma if volume and volume_sma else None,
                "adx": adx
            }
        )
