"""
Volume Analysis Strategy
Identifies trading opportunities based on volume patterns and anomalies
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import numpy as np

from .base_strategy import BaseStrategy, StrategyType
from ..utils.indicators import Signal, MarketData, TechnicalIndicators


class VolumeAnalysisStrategy(BaseStrategy):
    """
    Volume Analysis Strategy
    
    Analyzes volume patterns to identify:
    - Volume spikes (institutional activity)
    - Volume divergences
    - OBV trends
    - Volume-price confirmation
    
    Higher confidence when volume confirms price action.
    """
    
    def __init__(
        self,
        symbols: List[str],
        volume_ma_period: int = 20,
        spike_threshold: float = 2.0,  # Volume > 2x average
        divergence_lookback: int = 10,
        weight: float = 0.10,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        super().__init__(
            name="volume_analysis",
            strategy_type=StrategyType.VOLUME,
            symbols=symbols,
            weight=weight,
            enabled=enabled,
            params=params or {}
        )
        
        self.volume_ma_period = volume_ma_period
        self.spike_threshold = spike_threshold
        self.divergence_lookback = divergence_lookback
    
    def get_required_history(self) -> int:
        return self.volume_ma_period + self.divergence_lookback + 20
    
    def _detect_volume_spike(
        self, 
        current_volume: float, 
        volume_sma: float
    ) -> tuple:
        """
        Detect volume spike
        
        Returns:
            (is_spike, spike_magnitude)
        """
        if volume_sma <= 0:
            return False, 0
        
        ratio = current_volume / volume_sma
        is_spike = ratio >= self.spike_threshold
        
        return is_spike, ratio
    
    def _detect_volume_divergence(
        self,
        closes: List[float],
        volumes: List[float]
    ) -> tuple:
        """
        Detect volume-price divergence
        
        Returns:
            (has_divergence, divergence_type, strength)
            divergence_type: "bullish" or "bearish"
        """
        if len(closes) < self.divergence_lookback or len(volumes) < self.divergence_lookback:
            return False, None, 0
        
        recent_closes = closes[-self.divergence_lookback:]
        recent_volumes = volumes[-self.divergence_lookback:]
        
        # Calculate price trend
        price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] if recent_closes[0] != 0 else 0
        
        # Calculate volume trend
        first_half_vol = np.mean(recent_volumes[:len(recent_volumes)//2])
        second_half_vol = np.mean(recent_volumes[len(recent_volumes)//2:])
        vol_change = (second_half_vol - first_half_vol) / first_half_vol if first_half_vol != 0 else 0
        
        # Bullish divergence: Price down, volume decreasing (selling exhaustion)
        if price_change < -0.01 and vol_change < -0.1:
            strength = abs(price_change) * abs(vol_change) * 10
            return True, "bullish", min(1.0, strength)
        
        # Bearish divergence: Price up, volume decreasing (buying exhaustion)
        if price_change > 0.01 and vol_change < -0.1:
            strength = abs(price_change) * abs(vol_change) * 10
            return True, "bearish", min(1.0, strength)
        
        return False, None, 0
    
    def _analyze_obv_trend(self, market_data: MarketData) -> tuple:
        """
        Analyze OBV trend
        
        Returns:
            (trend_direction, trend_strength)
            trend_direction: "bullish", "bearish", or "neutral"
        """
        obv = market_data.indicators.get("obv", [])
        
        if len(obv) < 20:
            return "neutral", 0
        
        # Calculate OBV moving averages
        obv_short = np.mean(obv[-5:])
        obv_long = np.mean(obv[-20:])
        
        if obv_long == 0:
            return "neutral", 0
        
        trend_ratio = (obv_short - obv_long) / abs(obv_long)
        
        if trend_ratio > 0.05:
            return "bullish", min(1.0, trend_ratio * 5)
        elif trend_ratio < -0.05:
            return "bearish", min(1.0, abs(trend_ratio) * 5)
        
        return "neutral", 0
    
    def _check_volume_confirmation(
        self,
        price_direction: str,
        current_volume: float,
        volume_sma: float,
        obv_trend: str
    ) -> float:
        """
        Check if volume confirms price direction
        
        Returns:
            Confirmation score (0-1)
        """
        score = 0.0
        
        # High volume in direction of move
        if current_volume > volume_sma:
            score += 0.3
        if current_volume > volume_sma * 1.5:
            score += 0.2
        
        # OBV trend confirmation
        if price_direction == "LONG" and obv_trend == "bullish":
            score += 0.3
        elif price_direction == "SHORT" and obv_trend == "bearish":
            score += 0.3
        
        return min(1.0, score)
    
    def analyze(self, symbol: str, market_data: MarketData) -> Optional[Signal]:
        closes = market_data.indicators.get("close", [])
        volumes = market_data.indicators.get("volume", [])
        
        if len(closes) < self.get_required_history():
            return None
        
        current_volume = volumes[-1] if volumes else 0
        volume_sma = market_data.get_indicator("volume_sma")
        
        if not volume_sma or volume_sma <= 0:
            return None
        
        # Check for volume spike
        is_spike, spike_magnitude = self._detect_volume_spike(current_volume, volume_sma)
        
        # Check for divergence
        has_divergence, divergence_type, div_strength = self._detect_volume_divergence(
            closes, volumes
        )
        
        # Analyze OBV trend
        obv_trend, obv_strength = self._analyze_obv_trend(market_data)
        
        # Determine price direction from recent candles
        price_change = (closes[-1] - closes[-3]) / closes[-3] if closes[-3] != 0 else 0
        
        if price_change > 0.005:  # >0.5% up
            price_direction = "LONG"
        elif price_change < -0.005:  # >0.5% down
            price_direction = "SHORT"
        else:
            price_direction = None
        
        # Build signal
        direction = None
        confidence = 0.0
        
        # High volume spike with price direction
        if is_spike and price_direction:
            direction = price_direction
            confidence += 0.3 + min(0.2, (spike_magnitude - self.spike_threshold) * 0.1)
        
        # Divergence signal
        if has_divergence:
            if divergence_type == "bullish":
                if direction == "LONG":
                    confidence += div_strength * 0.3
                elif direction is None:
                    direction = "LONG"
                    confidence += div_strength * 0.4
            elif divergence_type == "bearish":
                if direction == "SHORT":
                    confidence += div_strength * 0.3
                elif direction is None:
                    direction = "SHORT"
                    confidence += div_strength * 0.4
        
        # OBV trend
        if obv_trend == "bullish" and direction in ["LONG", None]:
            if direction is None:
                direction = "LONG"
            confidence += obv_strength * 0.2
        elif obv_trend == "bearish" and direction in ["SHORT", None]:
            if direction is None:
                direction = "SHORT"
            confidence += obv_strength * 0.2
        
        if not direction or confidence < 0.15:
            return None
        
        # Add confirmation bonus
        confirmation = self._check_volume_confirmation(
            direction, current_volume, volume_sma, obv_trend
        )
        confidence = min(1.0, confidence + confirmation * 0.15)
        
        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "volume_spike": is_spike,
                "spike_magnitude": spike_magnitude,
                "divergence": divergence_type,
                "divergence_strength": div_strength,
                "obv_trend": obv_trend,
                "obv_strength": obv_strength,
                "volume_ratio": current_volume / volume_sma if volume_sma else 0
            }
        )
