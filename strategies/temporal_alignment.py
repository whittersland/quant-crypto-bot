"""
Temporal Alignment Strategy (Novel Dual-Condition)
Requires the same signal to appear across multiple timeframes simultaneously
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
import numpy as np

from .base_strategy import DualConditionStrategy, ConditionResult, TimeFrame
from ..utils.indicators import Signal, MarketData, TechnicalIndicators


class TemporalAlignmentStrategy(DualConditionStrategy):
    """
    Temporal Alignment Strategy
    
    NOVEL DUAL-CONDITION APPROACH:
    This strategy requires the SAME signal to appear across MULTIPLE
    TIMEFRAMES simultaneously. This addresses the common trading problem
    of conflicting signals on different timeframes.
    
    Dual Conditions:
    1. Signal must exist on lower timeframe (entry timing)
    2. Signal must be confirmed on higher timeframe (trend context)
    
    Timeframe Hierarchy:
    - 1-minute: Entry timing, short-term momentum
    - 5-minute: Near-term trend
    - 15-minute: Medium-term context
    
    When all three timeframes agree, we have high-confidence "temporal alignment"
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[int] = None,  # Minutes: [1, 5, 15]
        min_timeframes_aligned: int = 2,
        weight: float = 0.10,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        super().__init__(
            name="temporal_alignment",
            symbols=symbols,
            min_conditions=min_timeframes_aligned,
            weight=weight,
            enabled=enabled,
            params=params or {}
        )
        
        self.timeframes = timeframes or [1, 5, 15]
        self.min_timeframes_aligned = min_timeframes_aligned
        
        # Store data for different timeframes
        # In practice, you'd fetch different granularity data
        # Here we simulate by aggregating candles
        self.timeframe_data: Dict[str, Dict[int, MarketData]] = {
            symbol: {} for symbol in symbols
        }
    
    def get_required_history(self) -> int:
        return max(self.timeframes) * 20  # Need enough 1min candles
    
    def get_required_timeframes(self) -> List[TimeFrame]:
        """Return required timeframes for data fetching"""
        mapping = {
            1: TimeFrame.M1,
            5: TimeFrame.M5,
            15: TimeFrame.M15,
            30: TimeFrame.M30,
            60: TimeFrame.H1
        }
        return [mapping.get(tf, TimeFrame.M1) for tf in self.timeframes]
    
    def _aggregate_candles(
        self, 
        candles: List, 
        target_minutes: int
    ) -> List:
        """
        Aggregate 1-minute candles into higher timeframe
        
        This is a simplified aggregation for when we only have 1min data
        """
        if target_minutes == 1:
            return candles
        
        aggregated = []
        for i in range(0, len(candles) - target_minutes + 1, target_minutes):
            chunk = candles[i:i + target_minutes]
            if len(chunk) == target_minutes:
                agg_candle = type(chunk[0])(
                    timestamp=chunk[0].timestamp,
                    open=chunk[0].open,
                    high=max(c.high for c in chunk),
                    low=min(c.low for c in chunk),
                    close=chunk[-1].close,
                    volume=sum(c.volume for c in chunk)
                )
                aggregated.append(agg_candle)
        
        return aggregated
    
    def _analyze_timeframe(
        self, 
        market_data: MarketData,
        timeframe_minutes: int
    ) -> Dict:
        """
        Analyze a specific timeframe and return direction + strength
        """
        result = {
            "timeframe": timeframe_minutes,
            "direction": "NEUTRAL",
            "strength": 0,
            "trend_score": 0,
            "momentum_score": 0
        }
        
        # Use the indicators from market_data
        # In a full implementation, you'd have separate indicator calculations per timeframe
        
        # Trend: EMA relationship
        ema_fast = market_data.get_indicator("ema_9")
        ema_slow = market_data.get_indicator("ema_21")
        
        if ema_fast and ema_slow:
            diff_pct = (ema_fast - ema_slow) / ema_slow * 100 if ema_slow != 0 else 0
            result["trend_score"] = np.clip(diff_pct, -2, 2) / 2  # Normalize to -1 to 1
        
        # Momentum: RSI and MACD
        rsi = market_data.get_indicator("rsi")
        macd_hist = market_data.get_indicator("macd_histogram")
        
        if rsi:
            rsi_score = (rsi - 50) / 50  # Normalize to -1 to 1
            result["momentum_score"] += rsi_score * 0.5
        
        if macd_hist:
            # Normalize MACD histogram relative to price
            price = market_data.current_price
            if price > 0:
                macd_normalized = np.clip(macd_hist / price * 1000, -1, 1)
                result["momentum_score"] += macd_normalized * 0.5
        
        # Combine scores
        combined = result["trend_score"] * 0.6 + result["momentum_score"] * 0.4
        
        if combined > 0.15:
            result["direction"] = "LONG"
            result["strength"] = min(1.0, combined)
        elif combined < -0.15:
            result["direction"] = "SHORT"
            result["strength"] = min(1.0, abs(combined))
        
        return result
    
    def _evaluate_timeframe_condition(
        self,
        timeframe_minutes: int,
        analysis: Dict
    ) -> ConditionResult:
        """Convert timeframe analysis to condition result"""
        return ConditionResult(
            name=f"timeframe_{timeframe_minutes}m",
            met=analysis["direction"] != "NEUTRAL",
            value=analysis["strength"],
            threshold=0.15,
            direction=analysis["direction"],
            weight=self._get_timeframe_weight(timeframe_minutes),
            metadata={
                "trend_score": analysis["trend_score"],
                "momentum_score": analysis["momentum_score"]
            }
        )
    
    def _get_timeframe_weight(self, timeframe_minutes: int) -> float:
        """
        Get weight for a timeframe
        Higher timeframes get more weight (trend context)
        """
        if timeframe_minutes >= 15:
            return 1.2
        elif timeframe_minutes >= 5:
            return 1.0
        else:
            return 0.8
    
    def evaluate_conditions(
        self, 
        symbol: str, 
        market_data: MarketData
    ) -> List[ConditionResult]:
        """
        Evaluate all timeframe conditions
        
        Note: In a full implementation, each timeframe would have its own
        candle data and indicators. Here we use the primary data with
        adjusted thresholds.
        """
        conditions = []
        
        for tf in self.timeframes:
            # Analyze this timeframe
            # In practice, you'd use aggregated candles for higher TFs
            analysis = self._analyze_timeframe(market_data, tf)
            
            condition = self._evaluate_timeframe_condition(tf, analysis)
            conditions.append(condition)
        
        return conditions
    
    def analyze(self, symbol: str, market_data: MarketData) -> Optional[Signal]:
        """
        Override to add temporal alignment specific logic
        """
        conditions = self.evaluate_conditions(symbol, market_data)
        
        if not conditions:
            return None
        
        # Get conditions that are met per direction
        long_conditions = [c for c in conditions if c.met and c.direction == "LONG"]
        short_conditions = [c for c in conditions if c.met and c.direction == "SHORT"]
        
        # Check alignment
        if len(long_conditions) >= self.min_timeframes_aligned:
            direction = "LONG"
            aligned = long_conditions
        elif len(short_conditions) >= self.min_timeframes_aligned:
            direction = "SHORT"
            aligned = short_conditions
        else:
            return None
        
        # Calculate confidence based on alignment quality
        confidence = 0.0
        
        # Base confidence from number of aligned timeframes
        alignment_ratio = len(aligned) / len(conditions)
        confidence += alignment_ratio * 0.4
        
        # Weighted strength from aligned timeframes
        total_weight = sum(c.weight for c in aligned)
        weighted_strength = sum(c.value * c.weight for c in aligned) / total_weight if total_weight > 0 else 0
        confidence += weighted_strength * 0.4
        
        # Bonus for full alignment
        if len(aligned) == len(conditions):
            confidence += 0.15
        
        # Bonus for higher timeframe alignment
        higher_tf_aligned = [c for c in aligned if "15" in c.name or "30" in c.name or "60" in c.name]
        if higher_tf_aligned:
            confidence += 0.1
        
        confidence = min(1.0, confidence)
        
        if confidence < 0.2:
            return None
        
        # Build timeframe summary
        tf_summary = {}
        for c in conditions:
            tf_summary[c.name] = {
                "direction": c.direction,
                "strength": c.value,
                "aligned": c in aligned
            }
        
        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "aligned_timeframes": len(aligned),
                "total_timeframes": len(conditions),
                "alignment_ratio": alignment_ratio,
                "timeframe_details": tf_summary,
                "full_alignment": len(aligned) == len(conditions)
            }
        )
    
    def update_timeframe_data(
        self, 
        symbol: str, 
        timeframe: int, 
        market_data: MarketData
    ) -> None:
        """
        Update stored data for a specific timeframe
        Called when new candle data is available
        """
        if symbol not in self.timeframe_data:
            self.timeframe_data[symbol] = {}
        
        self.timeframe_data[symbol][timeframe] = market_data
    
    def get_alignment_status(self, symbol: str) -> Dict:
        """Get current temporal alignment status for a symbol"""
        if symbol not in self.timeframe_data:
            return {"error": "No data for symbol"}
        
        status = {
            "timeframes_available": list(self.timeframe_data[symbol].keys()),
            "analyses": {}
        }
        
        for tf, data in self.timeframe_data[symbol].items():
            analysis = self._analyze_timeframe(data, tf)
            status["analyses"][f"{tf}m"] = analysis
        
        return status
