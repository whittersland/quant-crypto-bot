"""
Confluence Convergence Strategy (Novel Dual-Condition)
Requires simultaneous agreement across multiple independent indicator families
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import numpy as np

from .base_strategy import DualConditionStrategy, ConditionResult
from ..utils.indicators import Signal, MarketData, TechnicalIndicators


class ConfluenceConvergenceStrategy(DualConditionStrategy):
    """
    Confluence Convergence Strategy
    
    NOVEL DUAL-CONDITION APPROACH:
    This strategy requires multiple INDEPENDENT indicator families to 
    simultaneously confirm the same direction. Unlike simple multi-indicator
    strategies, this enforces that signals must come from different 
    analytical approaches:
    
    Indicator Families:
    1. TREND Family: EMA alignment, ADX, price vs SMA
    2. MOMENTUM Family: RSI, MACD, Rate of Change
    3. VOLATILITY Family: Bollinger position, ATR breakout
    4. VOLUME Family: Volume spike, OBV trend
    
    Signal only generated when at least 3 of 4 families agree.
    This creates a robust "dual-condition" (actually multi-condition) 
    confirmation system.
    """
    
    def __init__(
        self,
        symbols: List[str],
        min_families_required: int = 3,
        weight: float = 0.10,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        super().__init__(
            name="confluence_convergence",
            symbols=symbols,
            min_conditions=min_families_required,
            weight=weight,
            enabled=enabled,
            params=params or {}
        )
        
        self.min_families_required = min_families_required
    
    def get_required_history(self) -> int:
        return 60  # Need enough for all indicator families
    
    def _evaluate_trend_family(self, market_data: MarketData) -> ConditionResult:
        """Evaluate trend indicator family"""
        score = 0
        direction = "NEUTRAL"
        details = {}
        
        # EMA alignment
        ema_9 = market_data.get_indicator("ema_9")
        ema_21 = market_data.get_indicator("ema_21")
        ema_50 = market_data.get_indicator("ema_50")
        
        if ema_9 and ema_21 and ema_50:
            if ema_9 > ema_21 > ema_50:
                score += 1
                direction = "LONG"
                details["ema_alignment"] = "bullish"
            elif ema_9 < ema_21 < ema_50:
                score += 1
                direction = "SHORT"
                details["ema_alignment"] = "bearish"
        
        # ADX trend strength
        adx = market_data.get_indicator("adx")
        if adx and adx > 25:
            score += 0.5
            details["adx"] = adx
        
        # Price vs SMA
        current_price = market_data.current_price
        sma_20 = market_data.get_indicator("sma_20")
        
        if sma_20:
            if current_price > sma_20 * 1.01:
                if direction == "LONG":
                    score += 0.5
                elif direction == "NEUTRAL":
                    direction = "LONG"
                    score += 0.5
                details["price_vs_sma"] = "above"
            elif current_price < sma_20 * 0.99:
                if direction == "SHORT":
                    score += 0.5
                elif direction == "NEUTRAL":
                    direction = "SHORT"
                    score += 0.5
                details["price_vs_sma"] = "below"
        
        # Family is confirmed if score >= 1.5
        is_met = score >= 1.5
        
        return ConditionResult(
            name="trend_family",
            met=is_met,
            value=score,
            threshold=1.5,
            direction=direction,
            weight=1.0,
            metadata=details
        )
    
    def _evaluate_momentum_family(self, market_data: MarketData) -> ConditionResult:
        """Evaluate momentum indicator family"""
        score = 0
        direction = "NEUTRAL"
        details = {}
        
        # RSI
        rsi = market_data.get_indicator("rsi")
        if rsi:
            details["rsi"] = rsi
            if rsi > 55:
                score += 0.7
                direction = "LONG"
            elif rsi < 45:
                score += 0.7
                direction = "SHORT"
        
        # MACD
        macd = market_data.get_indicator("macd")
        macd_signal = market_data.get_indicator("macd_signal")
        macd_hist = market_data.get_indicator("macd_histogram")
        
        if macd and macd_signal:
            if macd > macd_signal and macd_hist and macd_hist > 0:
                if direction in ["LONG", "NEUTRAL"]:
                    score += 0.7
                    direction = "LONG"
                details["macd"] = "bullish"
            elif macd < macd_signal and macd_hist and macd_hist < 0:
                if direction in ["SHORT", "NEUTRAL"]:
                    score += 0.7
                    direction = "SHORT"
                details["macd"] = "bearish"
        
        # Rate of Change
        closes = market_data.indicators.get("close", [])
        if len(closes) >= 10:
            roc = (closes[-1] - closes[-10]) / closes[-10] * 100 if closes[-10] != 0 else 0
            details["roc_10"] = roc
            
            if roc > 1.5:
                if direction in ["LONG", "NEUTRAL"]:
                    score += 0.6
                    if direction == "NEUTRAL":
                        direction = "LONG"
            elif roc < -1.5:
                if direction in ["SHORT", "NEUTRAL"]:
                    score += 0.6
                    if direction == "NEUTRAL":
                        direction = "SHORT"
        
        is_met = score >= 1.3
        
        return ConditionResult(
            name="momentum_family",
            met=is_met,
            value=score,
            threshold=1.3,
            direction=direction,
            weight=1.0,
            metadata=details
        )
    
    def _evaluate_volatility_family(self, market_data: MarketData) -> ConditionResult:
        """Evaluate volatility indicator family"""
        score = 0
        direction = "NEUTRAL"
        details = {}
        
        current_price = market_data.current_price
        
        # Bollinger Band position
        bb_upper = market_data.get_indicator("bb_upper")
        bb_lower = market_data.get_indicator("bb_lower")
        bb_middle = market_data.get_indicator("bb_middle")
        
        if bb_upper and bb_lower and bb_middle:
            band_width = bb_upper - bb_lower
            if band_width > 0:
                position = (current_price - bb_middle) / (band_width / 2)
                details["bb_position"] = position
                
                # Upper half of bands = bullish bias
                if position > 0.3:
                    score += 0.7
                    direction = "LONG"
                # Lower half of bands = bearish bias
                elif position < -0.3:
                    score += 0.7
                    direction = "SHORT"
        
        # ATR breakout detection
        atr = market_data.get_indicator("atr")
        closes = market_data.indicators.get("close", [])
        
        if atr and len(closes) >= 5:
            recent_move = abs(closes[-1] - closes[-5])
            if recent_move > atr * 1.5:
                score += 0.6
                details["atr_breakout"] = True
                
                if closes[-1] > closes[-5]:
                    if direction in ["LONG", "NEUTRAL"]:
                        direction = "LONG"
                else:
                    if direction in ["SHORT", "NEUTRAL"]:
                        direction = "SHORT"
        
        # Volatility expansion
        if bb_upper and bb_lower:
            bb_width_pct = (bb_upper - bb_lower) / bb_middle * 100 if bb_middle else 0
            details["bb_width_pct"] = bb_width_pct
            
            # Wide bands indicate volatility - adds to signal if direction confirmed
            if bb_width_pct > 4 and direction != "NEUTRAL":
                score += 0.4
        
        is_met = score >= 1.0
        
        return ConditionResult(
            name="volatility_family",
            met=is_met,
            value=score,
            threshold=1.0,
            direction=direction,
            weight=1.0,
            metadata=details
        )
    
    def _evaluate_volume_family(self, market_data: MarketData) -> ConditionResult:
        """Evaluate volume indicator family"""
        score = 0
        direction = "NEUTRAL"
        details = {}
        
        volume = market_data.get_indicator("volume")
        volume_sma = market_data.get_indicator("volume_sma")
        
        # Volume spike
        if volume and volume_sma and volume_sma > 0:
            volume_ratio = volume / volume_sma
            details["volume_ratio"] = volume_ratio
            
            if volume_ratio > 1.5:
                score += 0.6
                details["volume_spike"] = True
        
        # OBV trend
        obv = market_data.indicators.get("obv", [])
        if len(obv) >= 10:
            obv_short = np.mean(obv[-5:])
            obv_long = np.mean(obv[-10:])
            
            if obv_long != 0:
                obv_trend = (obv_short - obv_long) / abs(obv_long)
                details["obv_trend"] = obv_trend
                
                if obv_trend > 0.02:
                    score += 0.7
                    direction = "LONG"
                elif obv_trend < -0.02:
                    score += 0.7
                    direction = "SHORT"
        
        # Volume confirms price direction
        closes = market_data.indicators.get("close", [])
        if len(closes) >= 3 and volume and volume_sma:
            price_up = closes[-1] > closes[-3]
            high_volume = volume > volume_sma
            
            if price_up and high_volume:
                if direction in ["LONG", "NEUTRAL"]:
                    score += 0.5
                    direction = "LONG"
                details["volume_confirms"] = "bullish"
            elif not price_up and high_volume:
                if direction in ["SHORT", "NEUTRAL"]:
                    score += 0.5
                    direction = "SHORT"
                details["volume_confirms"] = "bearish"
        
        is_met = score >= 1.0
        
        return ConditionResult(
            name="volume_family",
            met=is_met,
            value=score,
            threshold=1.0,
            direction=direction,
            weight=1.0,
            metadata=details
        )
    
    def evaluate_conditions(
        self, 
        symbol: str, 
        market_data: MarketData
    ) -> List[ConditionResult]:
        """Evaluate all indicator families"""
        return [
            self._evaluate_trend_family(market_data),
            self._evaluate_momentum_family(market_data),
            self._evaluate_volatility_family(market_data),
            self._evaluate_volume_family(market_data)
        ]
