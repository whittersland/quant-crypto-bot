"""
Trend Following Strategy
Identifies and follows established trends using multiple EMAs and ADX
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import numpy as np

from .base_strategy import BaseStrategy, StrategyType
from ..utils.indicators import Signal, MarketData, TechnicalIndicators


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy
    
    Identifies strong trends using:
    - EMA alignment (fast > medium > slow for uptrend)
    - ADX for trend strength
    - Price position relative to EMAs
    - Pullback entries in trend direction
    
    Enters on pullbacks within established trends.
    """
    
    def __init__(
        self,
        symbols: List[str],
        ema_fast: int = 9,
        ema_medium: int = 21,
        ema_slow: int = 50,
        adx_period: int = 14,
        adx_threshold: float = 25,  # ADX > 25 indicates strong trend
        pullback_threshold: float = 0.3,  # How close to EMA for pullback entry
        weight: float = 0.15,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        super().__init__(
            name="trend_following",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbols=symbols,
            weight=weight,
            enabled=enabled,
            params=params or {}
        )
        
        self.ema_fast = ema_fast
        self.ema_medium = ema_medium
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.pullback_threshold = pullback_threshold
    
    def get_required_history(self) -> int:
        return self.ema_slow + self.adx_period + 20
    
    def _check_ema_alignment(self, market_data: MarketData) -> tuple:
        """
        Check EMA alignment for trend direction
        
        Returns:
            (trend_direction, alignment_strength)
        """
        ema_fast = market_data.get_indicator("ema_9")
        ema_medium = market_data.get_indicator("ema_21")
        ema_slow = market_data.get_indicator("ema_50")
        
        if not all([ema_fast, ema_medium, ema_slow]):
            return None, 0
        
        # Uptrend: fast > medium > slow
        if ema_fast > ema_medium > ema_slow:
            # Calculate alignment strength
            spread = (ema_fast - ema_slow) / ema_slow if ema_slow != 0 else 0
            return "LONG", min(1.0, spread * 20)
        
        # Downtrend: fast < medium < slow
        if ema_fast < ema_medium < ema_slow:
            spread = (ema_slow - ema_fast) / ema_slow if ema_slow != 0 else 0
            return "SHORT", min(1.0, spread * 20)
        
        return None, 0
    
    def _check_trend_strength(self, market_data: MarketData) -> tuple:
        """
        Check trend strength using ADX
        
        Returns:
            (is_strong_trend, adx_value)
        """
        adx = market_data.get_indicator("adx")
        
        if not adx:
            return False, 0
        
        return adx >= self.adx_threshold, adx
    
    def _check_pullback(
        self,
        current_price: float,
        ema_fast: float,
        ema_medium: float,
        trend_direction: str
    ) -> tuple:
        """
        Check if price is at a pullback level
        
        Returns:
            (is_pullback, pullback_quality)
        """
        if trend_direction == "LONG":
            # For uptrend, pullback is when price pulls back toward EMAs
            ema_zone_top = ema_fast
            ema_zone_bottom = ema_medium
            
            if current_price <= ema_zone_top and current_price >= ema_zone_bottom * 0.99:
                # Price is in the EMA zone - good pullback
                distance_to_fast = (ema_fast - current_price) / ema_fast if ema_fast != 0 else 0
                quality = 1.0 - min(1.0, abs(distance_to_fast) * 10)
                return True, quality
            elif current_price < ema_zone_bottom and current_price > ema_medium * 0.97:
                # Price slightly below medium EMA - deeper pullback
                return True, 0.7
        
        elif trend_direction == "SHORT":
            # For downtrend, pullback is when price rallies toward EMAs
            ema_zone_bottom = ema_fast
            ema_zone_top = ema_medium
            
            if current_price >= ema_zone_bottom and current_price <= ema_zone_top * 1.01:
                distance_to_fast = (current_price - ema_fast) / ema_fast if ema_fast != 0 else 0
                quality = 1.0 - min(1.0, abs(distance_to_fast) * 10)
                return True, quality
            elif current_price > ema_zone_top and current_price < ema_medium * 1.03:
                return True, 0.7
        
        return False, 0
    
    def _check_price_momentum(self, market_data: MarketData, direction: str) -> float:
        """
        Check if recent price action confirms trend
        
        Returns:
            Momentum score (0-1)
        """
        closes = market_data.indicators.get("close", [])
        
        if len(closes) < 5:
            return 0
        
        # Check last few candles
        recent_change = (closes[-1] - closes[-3]) / closes[-3] if closes[-3] != 0 else 0
        
        if direction == "LONG" and recent_change > 0:
            return min(1.0, recent_change * 20)
        elif direction == "SHORT" and recent_change < 0:
            return min(1.0, abs(recent_change) * 20)
        
        return 0
    
    def analyze(self, symbol: str, market_data: MarketData) -> Optional[Signal]:
        closes = market_data.indicators.get("close", [])
        
        if len(closes) < self.get_required_history():
            return None
        
        current_price = closes[-1]
        
        # Check EMA alignment
        trend_direction, alignment_strength = self._check_ema_alignment(market_data)
        
        if not trend_direction:
            return None
        
        # Check trend strength
        is_strong_trend, adx = self._check_trend_strength(market_data)
        
        # Get EMA values for pullback check
        ema_fast = market_data.get_indicator("ema_9")
        ema_medium = market_data.get_indicator("ema_21")
        
        if not ema_fast or not ema_medium:
            return None
        
        # Check for pullback entry
        is_pullback, pullback_quality = self._check_pullback(
            current_price, ema_fast, ema_medium, trend_direction
        )
        
        # Check momentum
        momentum_score = self._check_price_momentum(market_data, trend_direction)
        
        # Calculate confidence
        confidence = 0.0
        
        # EMA alignment contributes base confidence
        confidence += alignment_strength * 0.3
        
        # ADX trend strength
        if is_strong_trend:
            confidence += 0.25
            if adx > 35:
                confidence += 0.1  # Very strong trend bonus
        else:
            confidence += (adx / self.adx_threshold) * 0.15 if adx else 0
        
        # Pullback quality
        if is_pullback:
            confidence += pullback_quality * 0.25
        else:
            # Not at pullback - reduce confidence but still valid trend entry
            confidence *= 0.7
        
        # Momentum confirmation
        confidence += momentum_score * 0.1
        
        confidence = min(1.0, confidence)
        
        if confidence < 0.20:
            return None
        
        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=trend_direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "ema_alignment": alignment_strength,
                "adx": adx,
                "is_strong_trend": is_strong_trend,
                "is_pullback": is_pullback,
                "pullback_quality": pullback_quality,
                "momentum_score": momentum_score,
                "ema_fast": ema_fast,
                "ema_medium": ema_medium,
                "current_price": current_price
            }
        )
