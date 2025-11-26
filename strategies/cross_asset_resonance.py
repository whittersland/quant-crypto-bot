"""
Cross-Asset Resonance Strategy (Novel Dual-Condition)
Requires correlated signals across multiple crypto assets simultaneously
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
import numpy as np

from .base_strategy import BaseStrategy, StrategyType, ConditionResult
from ..utils.indicators import Signal, MarketData, TechnicalIndicators


class CrossAssetResonanceStrategy(BaseStrategy):
    """
    Cross-Asset Resonance Strategy
    
    NOVEL DUAL-CONDITION APPROACH:
    This strategy requires MULTIPLE ASSETS to show the same technical
    setup simultaneously. The idea is that when BTC, ETH, and SOL all
    show bullish signals at the same time, it indicates broad market
    strength rather than isolated asset movement.
    
    Dual Conditions:
    1. Multiple assets must show same directional signal
    2. The signals must occur within the same analysis window
    
    This is different from correlation-based trading - we're looking
    for INDEPENDENT but SIMULTANEOUS signals across assets.
    """
    
    def __init__(
        self,
        symbols: List[str],
        min_assets_aligned: int = 2,  # At least 2 of 3 must agree
        correlation_window: int = 20,
        signal_agreement_threshold: float = 0.6,
        weight: float = 0.10,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        super().__init__(
            name="cross_asset_resonance",
            strategy_type=StrategyType.DUAL_CONDITION,
            symbols=symbols,
            weight=weight,
            enabled=enabled,
            params=params or {}
        )
        
        self.min_assets_aligned = min_assets_aligned
        self.correlation_window = correlation_window
        self.signal_agreement_threshold = signal_agreement_threshold
        
        # Store market data for all assets
        self.asset_data: Dict[str, MarketData] = {}
        self.asset_signals: Dict[str, Dict] = {}
    
    def get_required_history(self) -> int:
        return 50
    
    def update_asset_data(self, symbol: str, market_data: MarketData) -> None:
        """Update stored market data for an asset"""
        self.asset_data[symbol] = market_data
        self.asset_signals[symbol] = self._analyze_single_asset(market_data)
    
    def _analyze_single_asset(self, market_data: MarketData) -> Dict:
        """
        Analyze a single asset and return its signal components
        
        Returns dict with direction scores for each indicator
        """
        result = {
            "trend_score": 0,  # -1 to +1
            "momentum_score": 0,
            "volatility_score": 0,
            "overall_direction": "NEUTRAL",
            "strength": 0
        }
        
        # Trend analysis
        ema_9 = market_data.get_indicator("ema_9")
        ema_21 = market_data.get_indicator("ema_21")
        
        if ema_9 and ema_21:
            if ema_9 > ema_21:
                result["trend_score"] = min(1.0, (ema_9 - ema_21) / ema_21 * 50)
            else:
                result["trend_score"] = max(-1.0, (ema_9 - ema_21) / ema_21 * 50)
        
        # Momentum analysis
        rsi = market_data.get_indicator("rsi")
        if rsi:
            # Normalize RSI to -1 to +1
            result["momentum_score"] = (rsi - 50) / 50
        
        # Volatility/breakout analysis
        closes = market_data.indicators.get("close", [])
        atr = market_data.get_indicator("atr")
        
        if len(closes) >= 10 and atr and atr > 0:
            move = (closes[-1] - closes[-10]) / closes[-10]
            result["volatility_score"] = np.clip(move / (atr / closes[-1] * 5), -1, 1)
        
        # Calculate overall direction
        total_score = (
            result["trend_score"] * 0.4 +
            result["momentum_score"] * 0.3 +
            result["volatility_score"] * 0.3
        )
        
        if total_score > 0.2:
            result["overall_direction"] = "LONG"
            result["strength"] = min(1.0, total_score)
        elif total_score < -0.2:
            result["overall_direction"] = "SHORT"
            result["strength"] = min(1.0, abs(total_score))
        else:
            result["overall_direction"] = "NEUTRAL"
            result["strength"] = 0
        
        return result
    
    def _check_asset_alignment(self) -> Tuple[str, float, List[str]]:
        """
        Check if multiple assets are aligned in the same direction
        
        Returns:
            (direction, confidence, aligned_assets)
        """
        if len(self.asset_signals) < 2:
            return "NEUTRAL", 0, []
        
        # Count directions
        long_assets = []
        short_assets = []
        
        for symbol, signal in self.asset_signals.items():
            if signal["overall_direction"] == "LONG":
                long_assets.append(symbol)
            elif signal["overall_direction"] == "SHORT":
                short_assets.append(symbol)
        
        # Check alignment
        if len(long_assets) >= self.min_assets_aligned:
            avg_strength = np.mean([
                self.asset_signals[s]["strength"] for s in long_assets
            ])
            return "LONG", avg_strength, long_assets
        
        if len(short_assets) >= self.min_assets_aligned:
            avg_strength = np.mean([
                self.asset_signals[s]["strength"] for s in short_assets
            ])
            return "SHORT", avg_strength, short_assets
        
        return "NEUTRAL", 0, []
    
    def _calculate_correlation_boost(self, aligned_assets: List[str]) -> float:
        """
        Calculate confidence boost based on asset correlations
        
        High correlation between aligned assets = stronger signal
        """
        if len(aligned_assets) < 2:
            return 0
        
        # Get closes for each aligned asset
        closes_list = []
        for symbol in aligned_assets:
            if symbol in self.asset_data:
                closes = self.asset_data[symbol].indicators.get("close", [])
                if len(closes) >= self.correlation_window:
                    closes_list.append(closes[-self.correlation_window:])
        
        if len(closes_list) < 2:
            return 0
        
        # Calculate returns
        returns_list = []
        for closes in closes_list:
            returns = np.diff(closes) / closes[:-1]
            returns_list.append(returns)
        
        # Calculate average correlation
        correlations = []
        for i in range(len(returns_list)):
            for j in range(i + 1, len(returns_list)):
                if len(returns_list[i]) == len(returns_list[j]):
                    corr = np.corrcoef(returns_list[i], returns_list[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        if not correlations:
            return 0
        
        avg_corr = np.mean(correlations)
        
        # High correlation = boost (market moving together)
        # But not too high (could be just BTC dominance)
        if 0.5 <= avg_corr <= 0.85:
            return 0.15
        elif avg_corr > 0.85:
            return 0.1  # Slightly reduced for very high correlation
        elif 0.3 <= avg_corr < 0.5:
            return 0.1
        
        return 0
    
    def analyze(self, symbol: str, market_data: MarketData) -> Optional[Signal]:
        """
        Analyze for cross-asset resonance
        
        This strategy is unique in that it uses data from ALL tracked assets,
        not just the one being analyzed. The signal is generated only when
        multiple assets show agreement.
        """
        # Update this asset's data
        self.update_asset_data(symbol, market_data)
        
        # Need data from multiple assets
        if len(self.asset_signals) < self.min_assets_aligned:
            return None
        
        # Check alignment
        direction, base_strength, aligned_assets = self._check_asset_alignment()
        
        if direction == "NEUTRAL":
            return None
        
        # Only generate signal for the symbol being analyzed if it's aligned
        if symbol not in aligned_assets:
            return None
        
        # Calculate confidence
        confidence = base_strength * 0.6
        
        # Boost for more assets aligned
        alignment_ratio = len(aligned_assets) / len(self.asset_signals)
        confidence += alignment_ratio * 0.2
        
        # Correlation boost
        corr_boost = self._calculate_correlation_boost(aligned_assets)
        confidence += corr_boost
        
        # Boost if this asset has strongest signal
        this_asset_strength = self.asset_signals[symbol]["strength"]
        max_strength = max(self.asset_signals[s]["strength"] for s in aligned_assets)
        
        if this_asset_strength == max_strength:
            confidence += 0.1
        
        confidence = min(1.0, confidence)
        
        if confidence < 0.2:
            return None
        
        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "aligned_assets": aligned_assets,
                "total_assets": len(self.asset_signals),
                "alignment_ratio": alignment_ratio,
                "this_asset_strength": this_asset_strength,
                "asset_scores": {
                    s: self.asset_signals[s] 
                    for s in aligned_assets
                }
            }
        )
    
    def get_resonance_report(self) -> Dict:
        """Get current cross-asset resonance status"""
        direction, strength, aligned = self._check_asset_alignment()
        
        return {
            "current_direction": direction,
            "alignment_strength": strength,
            "aligned_assets": aligned,
            "total_assets_tracked": len(self.asset_signals),
            "asset_details": self.asset_signals.copy()
        }
