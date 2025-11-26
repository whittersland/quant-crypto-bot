"""
Utility functions and technical indicators
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal from a strategy"""
    strategy: str
    symbol: str
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TechnicalIndicators:
    """Technical analysis indicators"""
    
    @staticmethod
    def sma(data: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        if len(data) < period:
            return [np.nan] * len(data)
        
        result = [np.nan] * (period - 1)
        for i in range(period - 1, len(data)):
            result.append(np.mean(data[i - period + 1:i + 1]))
        return result
    
    @staticmethod
    def ema(data: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if len(data) < period:
            return [np.nan] * len(data)
        
        multiplier = 2 / (period + 1)
        result = [np.nan] * (period - 1)
        result.append(np.mean(data[:period]))
        
        for i in range(period, len(data)):
            ema_val = (data[i] - result[-1]) * multiplier + result[-1]
            result.append(ema_val)
        
        return result
    
    @staticmethod
    def rsi(data: List[float], period: int = 14) -> List[float]:
        """Relative Strength Index"""
        if len(data) < period + 1:
            return [np.nan] * len(data)
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        result = [np.nan] * period
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100 - (100 / (1 + rs)))
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                result.append(100.0)
            else:
                rs = avg_gain / avg_loss
                result.append(100 - (100 / (1 + rs)))
        
        return result
    
    @staticmethod
    def bollinger_bands(
        data: List[float], 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands - returns (upper, middle, lower)"""
        middle = TechnicalIndicators.sma(data, period)
        
        upper = []
        lower = []
        
        for i in range(len(data)):
            if i < period - 1:
                upper.append(np.nan)
                lower.append(np.nan)
            else:
                std = np.std(data[i - period + 1:i + 1])
                upper.append(middle[i] + std_dev * std)
                lower.append(middle[i] - std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def atr(
        highs: List[float], 
        lows: List[float], 
        closes: List[float], 
        period: int = 14
    ) -> List[float]:
        """Average True Range"""
        if len(highs) < 2:
            return [np.nan] * len(highs)
        
        true_ranges = [highs[0] - lows[0]]
        
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            true_ranges.append(tr)
        
        return TechnicalIndicators.ema(true_ranges, period)
    
    @staticmethod
    def macd(
        data: List[float], 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[List[float], List[float], List[float]]:
        """MACD - returns (macd_line, signal_line, histogram)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = [
            f - s if not (np.isnan(f) or np.isnan(s)) else np.nan 
            for f, s in zip(ema_fast, ema_slow)
        ]
        
        # Filter out NaN values for signal line calculation
        valid_macd = [m for m in macd_line if not np.isnan(m)]
        signal_line_valid = TechnicalIndicators.ema(valid_macd, signal)
        
        # Pad signal line to match original length
        signal_line = [np.nan] * (len(macd_line) - len(signal_line_valid)) + signal_line_valid
        
        histogram = [
            m - s if not (np.isnan(m) or np.isnan(s)) else np.nan
            for m, s in zip(macd_line, signal_line)
        ]
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[List[float], List[float]]:
        """Stochastic Oscillator - returns (%K, %D)"""
        k_values = []
        
        for i in range(len(closes)):
            if i < k_period - 1:
                k_values.append(np.nan)
            else:
                highest_high = max(highs[i - k_period + 1:i + 1])
                lowest_low = min(lows[i - k_period + 1:i + 1])
                
                if highest_high == lowest_low:
                    k_values.append(50.0)
                else:
                    k = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)
                    k_values.append(k)
        
        d_values = TechnicalIndicators.sma(k_values, d_period)
        
        return k_values, d_values
    
    @staticmethod
    def adx(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> List[float]:
        """Average Directional Index"""
        if len(highs) < period + 1:
            return [np.nan] * len(highs)
        
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)
            
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)
        
        atr_values = TechnicalIndicators.atr(highs, lows, closes, period)
        
        plus_di = []
        minus_di = []
        
        smoothed_plus = sum(plus_dm[:period])
        smoothed_minus = sum(minus_dm[:period])
        
        for i in range(len(plus_dm)):
            if i < period - 1:
                plus_di.append(np.nan)
                minus_di.append(np.nan)
            else:
                if i >= period:
                    smoothed_plus = smoothed_plus - (smoothed_plus / period) + plus_dm[i]
                    smoothed_minus = smoothed_minus - (smoothed_minus / period) + minus_dm[i]
                
                atr_val = atr_values[i + 1] if i + 1 < len(atr_values) else atr_values[-1]
                
                if atr_val and atr_val > 0:
                    plus_di.append(100 * smoothed_plus / period / atr_val)
                    minus_di.append(100 * smoothed_minus / period / atr_val)
                else:
                    plus_di.append(np.nan)
                    minus_di.append(np.nan)
        
        dx = []
        for p, m in zip(plus_di, minus_di):
            if np.isnan(p) or np.isnan(m) or (p + m) == 0:
                dx.append(np.nan)
            else:
                dx.append(100 * abs(p - m) / (p + m))
        
        adx = TechnicalIndicators.ema([d for d in dx if not np.isnan(d)], period)
        
        # Pad to original length
        result = [np.nan] * (len(highs) - len(adx)) + adx
        return result
    
    @staticmethod
    def volume_sma(volumes: List[float], period: int = 20) -> List[float]:
        """Volume Simple Moving Average"""
        return TechnicalIndicators.sma(volumes, period)
    
    @staticmethod
    def obv(closes: List[float], volumes: List[float]) -> List[float]:
        """On-Balance Volume"""
        if len(closes) < 2:
            return volumes[:1] if volumes else [0]
        
        obv = [volumes[0]]
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        
        return obv
    
    @staticmethod
    def momentum(data: List[float], period: int = 10) -> List[float]:
        """Price Momentum"""
        result = [np.nan] * period
        
        for i in range(period, len(data)):
            result.append(data[i] - data[i - period])
        
        return result
    
    @staticmethod
    def rate_of_change(data: List[float], period: int = 10) -> List[float]:
        """Rate of Change (ROC)"""
        result = [np.nan] * period
        
        for i in range(period, len(data)):
            if data[i - period] != 0:
                result.append(100 * (data[i] - data[i - period]) / data[i - period])
            else:
                result.append(np.nan)
        
        return result


class MarketData:
    """Market data container and processor"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.candles = []
        self.indicators = {}
    
    def update(self, candles: List) -> None:
        """Update with new candle data"""
        self.candles = candles
        self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Pre-calculate common indicators"""
        if not self.candles:
            return
        
        closes = [c.close for c in self.candles]
        highs = [c.high for c in self.candles]
        lows = [c.low for c in self.candles]
        volumes = [c.volume for c in self.candles]
        
        self.indicators = {
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": volumes,
            "sma_20": TechnicalIndicators.sma(closes, 20),
            "sma_50": TechnicalIndicators.sma(closes, 50),
            "ema_9": TechnicalIndicators.ema(closes, 9),
            "ema_21": TechnicalIndicators.ema(closes, 21),
            "ema_50": TechnicalIndicators.ema(closes, 50),
            "rsi": TechnicalIndicators.rsi(closes, 14),
            "atr": TechnicalIndicators.atr(highs, lows, closes, 14),
            "volume_sma": TechnicalIndicators.volume_sma(volumes, 20),
        }
        
        bb = TechnicalIndicators.bollinger_bands(closes, 20, 2.0)
        self.indicators["bb_upper"] = bb[0]
        self.indicators["bb_middle"] = bb[1]
        self.indicators["bb_lower"] = bb[2]
        
        macd = TechnicalIndicators.macd(closes)
        self.indicators["macd"] = macd[0]
        self.indicators["macd_signal"] = macd[1]
        self.indicators["macd_histogram"] = macd[2]
        
        stoch = TechnicalIndicators.stochastic(highs, lows, closes)
        self.indicators["stoch_k"] = stoch[0]
        self.indicators["stoch_d"] = stoch[1]
        
        self.indicators["adx"] = TechnicalIndicators.adx(highs, lows, closes)
        self.indicators["obv"] = TechnicalIndicators.obv(closes, volumes)
        self.indicators["momentum"] = TechnicalIndicators.momentum(closes, 10)
        self.indicators["roc"] = TechnicalIndicators.rate_of_change(closes, 10)
    
    @property
    def current_price(self) -> float:
        """Get current (most recent) price"""
        return self.candles[-1].close if self.candles else 0.0
    
    def get_indicator(self, name: str, offset: int = 0) -> Optional[float]:
        """Get indicator value at offset from current (0 = current, 1 = previous, etc.)"""
        if name not in self.indicators:
            return None
        
        idx = -1 - offset
        if abs(idx) > len(self.indicators[name]):
            return None
        
        return self.indicators[name][idx]


def calculate_position_size(
    capital: float,
    min_size: float,
    max_size: float,
    confidence: float,
    risk_percent: float = 0.02
) -> float:
    """
    Calculate position size based on confidence and risk parameters
    
    Args:
        capital: Available trading capital
        min_size: Minimum position size
        max_size: Maximum position size
        confidence: Signal confidence (0.0 to 1.0)
        risk_percent: Max risk per trade as percentage of capital
    
    Returns:
        Position size in USD
    """
    # Scale position between min and max based on confidence
    size_range = max_size - min_size
    base_size = min_size + (size_range * confidence)
    
    # Apply risk limit
    max_risk_size = capital * risk_percent
    
    return min(base_size, max_risk_size, max_size)


def aggregate_signals(signals: List[Signal], weights: Dict[str, float] = None) -> Signal:
    """
    Aggregate multiple signals into a single combined signal
    
    Uses weighted averaging of confidence scores
    """
    if not signals:
        return Signal(
            strategy="aggregated",
            symbol="",
            direction="NEUTRAL",
            confidence=0.0,
            timestamp=datetime.now()
        )
    
    if weights is None:
        weights = {s.strategy: 1.0 / len(signals) for s in signals}
    
    # Normalize weights
    total_weight = sum(weights.get(s.strategy, 0) for s in signals)
    if total_weight == 0:
        total_weight = 1
    
    # Calculate weighted direction scores
    long_score = 0.0
    short_score = 0.0
    
    for signal in signals:
        weight = weights.get(signal.strategy, 0) / total_weight
        
        if signal.direction == "LONG":
            long_score += signal.confidence * weight
        elif signal.direction == "SHORT":
            short_score += signal.confidence * weight
    
    # Determine final direction and confidence
    if long_score > short_score and long_score > 0.1:
        direction = "LONG"
        confidence = long_score - short_score * 0.5  # Reduce confidence if conflicting signals
    elif short_score > long_score and short_score > 0.1:
        direction = "SHORT"
        confidence = short_score - long_score * 0.5
    else:
        direction = "NEUTRAL"
        confidence = 0.0
    
    confidence = max(0.0, min(1.0, confidence))
    
    return Signal(
        strategy="aggregated",
        symbol=signals[0].symbol,
        direction=direction,
        confidence=confidence,
        timestamp=datetime.now(),
        metadata={
            "component_signals": len(signals),
            "long_score": long_score,
            "short_score": short_score
        }
    )
