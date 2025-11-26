"""
Signal Aggregation Engine
Combines signals from multiple strategies with weighted voting and conflict resolution
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import numpy as np
import logging

from ..utils.indicators import Signal

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Signal aggregation methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    UNANIMOUS = "unanimous"
    HIGHEST_CONFIDENCE = "highest_confidence"
    DUAL_CONFIRMATION = "dual_confirmation"


@dataclass
class AggregatedSignal:
    """Result of signal aggregation"""
    symbol: str
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float
    timestamp: datetime
    contributing_signals: List[Signal]
    aggregation_method: AggregationMethod
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act on"""
        return self.direction != "NEUTRAL" and self.confidence >= 0.15
    
    @property
    def signal_count(self) -> int:
        """Number of contributing signals"""
        return len(self.contributing_signals)


class SignalAggregator:
    """
    Aggregates signals from multiple strategies into unified trading decisions
    
    Features:
    - Multiple aggregation methods
    - Conflict detection and resolution
    - Confidence normalization
    - Dual-condition verification
    """
    
    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        min_confidence: float = 0.15,
        min_signals: int = 1,
        require_dual_confirmation: bool = True
    ):
        self.method = method
        self.min_confidence = min_confidence
        self.min_signals = min_signals
        self.require_dual_confirmation = require_dual_confirmation
        
        # Default weights (can be overridden)
        self.strategy_weights: Dict[str, float] = {}
        
        # Signal history for analysis
        self.signal_history: List[AggregatedSignal] = []
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set strategy weights for aggregation"""
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            self.strategy_weights = {k: v / total for k, v in weights.items()}
        else:
            self.strategy_weights = weights
    
    def aggregate(
        self, 
        signals: List[Signal],
        weights: Dict[str, float] = None
    ) -> Optional[AggregatedSignal]:
        """
        Aggregate multiple signals into a single decision
        
        Args:
            signals: List of signals from different strategies
            weights: Optional weight overrides
        
        Returns:
            AggregatedSignal or None if no actionable signal
        """
        if not signals:
            return None
        
        # Use provided weights or stored weights
        active_weights = weights or self.strategy_weights
        
        # Filter to same symbol (should already be, but safety check)
        symbol = signals[0].symbol
        signals = [s for s in signals if s.symbol == symbol]
        
        if len(signals) < self.min_signals:
            return None
        
        # Apply aggregation method
        if self.method == AggregationMethod.WEIGHTED_AVERAGE:
            result = self._weighted_average(signals, active_weights)
        elif self.method == AggregationMethod.MAJORITY_VOTE:
            result = self._majority_vote(signals, active_weights)
        elif self.method == AggregationMethod.UNANIMOUS:
            result = self._unanimous(signals)
        elif self.method == AggregationMethod.HIGHEST_CONFIDENCE:
            result = self._highest_confidence(signals)
        elif self.method == AggregationMethod.DUAL_CONFIRMATION:
            result = self._dual_confirmation(signals, active_weights)
        else:
            result = self._weighted_average(signals, active_weights)
        
        if result and result.confidence >= self.min_confidence:
            self.signal_history.append(result)
            return result
        
        return None
    
    def _weighted_average(
        self, 
        signals: List[Signal],
        weights: Dict[str, float]
    ) -> AggregatedSignal:
        """
        Weighted average aggregation
        
        Each signal's contribution is weighted by its strategy weight
        and individual confidence
        """
        long_score = 0.0
        short_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = weights.get(signal.strategy, 1.0 / len(signals))
            weighted_conf = signal.confidence * weight
            total_weight += weight
            
            if signal.direction == "LONG":
                long_score += weighted_conf
            elif signal.direction == "SHORT":
                short_score += weighted_conf
        
        # Normalize
        if total_weight > 0:
            long_score /= total_weight
            short_score /= total_weight
        
        # Determine direction
        if long_score > short_score and long_score > 0.1:
            direction = "LONG"
            # Confidence reduced by conflicting signals
            confidence = long_score - (short_score * 0.5)
        elif short_score > long_score and short_score > 0.1:
            direction = "SHORT"
            confidence = short_score - (long_score * 0.5)
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        
        confidence = max(0.0, min(1.0, confidence))
        
        return AggregatedSignal(
            symbol=signals[0].symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            contributing_signals=signals,
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            metadata={
                "long_score": long_score,
                "short_score": short_score,
                "total_weight": total_weight
            }
        )
    
    def _majority_vote(
        self, 
        signals: List[Signal],
        weights: Dict[str, float]
    ) -> AggregatedSignal:
        """
        Majority vote aggregation
        
        Direction determined by majority, confidence by agreement strength
        """
        long_votes = 0
        short_votes = 0
        long_confidence = []
        short_confidence = []
        
        for signal in signals:
            if signal.direction == "LONG":
                long_votes += 1
                long_confidence.append(signal.confidence)
            elif signal.direction == "SHORT":
                short_votes += 1
                short_confidence.append(signal.confidence)
        
        total_votes = long_votes + short_votes
        
        if total_votes == 0:
            return AggregatedSignal(
                symbol=signals[0].symbol,
                direction="NEUTRAL",
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                contributing_signals=signals,
                aggregation_method=AggregationMethod.MAJORITY_VOTE
            )
        
        if long_votes > short_votes:
            direction = "LONG"
            vote_ratio = long_votes / total_votes
            avg_confidence = np.mean(long_confidence) if long_confidence else 0
        elif short_votes > long_votes:
            direction = "SHORT"
            vote_ratio = short_votes / total_votes
            avg_confidence = np.mean(short_confidence) if short_confidence else 0
        else:
            direction = "NEUTRAL"
            vote_ratio = 0.5
            avg_confidence = 0
        
        # Confidence based on vote ratio and average signal confidence
        confidence = vote_ratio * 0.5 + avg_confidence * 0.5
        
        return AggregatedSignal(
            symbol=signals[0].symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            contributing_signals=signals,
            aggregation_method=AggregationMethod.MAJORITY_VOTE,
            metadata={
                "long_votes": long_votes,
                "short_votes": short_votes,
                "vote_ratio": vote_ratio
            }
        )
    
    def _unanimous(self, signals: List[Signal]) -> AggregatedSignal:
        """
        Unanimous agreement required
        
        All signals must agree on direction
        """
        directions = set(s.direction for s in signals if s.direction != "NEUTRAL")
        
        if len(directions) == 1:
            direction = directions.pop()
            confidence = np.mean([s.confidence for s in signals if s.direction == direction])
            # Boost for unanimity
            confidence = min(1.0, confidence * 1.2)
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        
        return AggregatedSignal(
            symbol=signals[0].symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            contributing_signals=signals,
            aggregation_method=AggregationMethod.UNANIMOUS,
            metadata={
                "unique_directions": len(directions),
                "is_unanimous": len(directions) == 1
            }
        )
    
    def _highest_confidence(self, signals: List[Signal]) -> AggregatedSignal:
        """
        Use highest confidence signal
        
        Winner-take-all approach
        """
        best_signal = max(signals, key=lambda s: s.confidence)
        
        return AggregatedSignal(
            symbol=signals[0].symbol,
            direction=best_signal.direction,
            confidence=best_signal.confidence,
            timestamp=datetime.now(timezone.utc),
            contributing_signals=[best_signal],
            aggregation_method=AggregationMethod.HIGHEST_CONFIDENCE,
            metadata={
                "winning_strategy": best_signal.strategy,
                "signals_considered": len(signals)
            }
        )
    
    def _dual_confirmation(
        self, 
        signals: List[Signal],
        weights: Dict[str, float]
    ) -> AggregatedSignal:
        """
        Dual confirmation aggregation
        
        Normally requires at least 2 different strategy types to agree.
        Exception: Single signals with very high confidence (0.80+) can pass.
        """
        # Group signals by strategy type (using strategy name prefix)
        strategy_groups: Dict[str, List[Signal]] = {}
        
        for signal in signals:
            # Extract strategy type from name
            strategy_type = signal.strategy.split("_")[0]
            if strategy_type not in strategy_groups:
                strategy_groups[strategy_type] = []
            strategy_groups[strategy_type].append(signal)
        
        # Check each direction
        long_groups = []
        short_groups = []
        highest_long_conf = 0.0
        highest_short_conf = 0.0
        
        for group_name, group_signals in strategy_groups.items():
            group_direction = None
            group_confidence = 0
            
            for s in group_signals:
                if s.direction in ["LONG", "SHORT"]:
                    if group_direction is None:
                        group_direction = s.direction
                        group_confidence = s.confidence
                    elif group_direction == s.direction:
                        group_confidence = max(group_confidence, s.confidence)
            
            if group_direction == "LONG":
                long_groups.append((group_name, group_confidence))
                highest_long_conf = max(highest_long_conf, group_confidence)
            elif group_direction == "SHORT":
                short_groups.append((group_name, group_confidence))
                highest_short_conf = max(highest_short_conf, group_confidence)
        
        # Standard dual confirmation: need at least 2 groups
        # Exception: Allow single group if confidence >= 0.80
        high_conf_threshold = 0.80
        
        if len(long_groups) >= 2:
            direction = "LONG"
            confidence = np.mean([c for _, c in long_groups])
            # Boost for dual confirmation
            confidence = min(1.0, confidence * (1 + 0.1 * len(long_groups)))
            confirming_groups = long_groups
            logger.info(f"Dual confirmation: LONG with {len(long_groups)} groups, conf={confidence:.3f}")
        elif len(short_groups) >= 2:
            direction = "SHORT"
            confidence = np.mean([c for _, c in short_groups])
            confidence = min(1.0, confidence * (1 + 0.1 * len(short_groups)))
            confirming_groups = short_groups
            logger.info(f"Dual confirmation: SHORT with {len(short_groups)} groups, conf={confidence:.3f}")
        elif len(long_groups) == 1 and highest_long_conf >= high_conf_threshold:
            # Single high-confidence signal exception
            direction = "LONG"
            confidence = highest_long_conf * 0.95  # Slight penalty for single source
            confirming_groups = long_groups
            logger.info(f"Single high-conf exception: LONG conf={confidence:.3f} (original={highest_long_conf:.3f})")
        elif len(short_groups) == 1 and highest_short_conf >= high_conf_threshold:
            direction = "SHORT"
            confidence = highest_short_conf * 0.95
            confirming_groups = short_groups
            logger.info(f"Single high-conf exception: SHORT conf={confidence:.3f} (original={highest_short_conf:.3f})")
        else:
            direction = "NEUTRAL"
            confidence = 0.0
            confirming_groups = []
            if long_groups or short_groups:
                logger.debug(f"No dual confirmation: LONG groups={len(long_groups)} (max conf={highest_long_conf:.3f}), SHORT groups={len(short_groups)} (max conf={highest_short_conf:.3f})")
        
        return AggregatedSignal(
            symbol=signals[0].symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            contributing_signals=signals,
            aggregation_method=AggregationMethod.DUAL_CONFIRMATION,
            metadata={
                "long_groups": len(long_groups),
                "short_groups": len(short_groups),
                "confirming_groups": [g[0] for g in confirming_groups],
                "is_dual_confirmed": len(confirming_groups) >= 2,
                "high_conf_exception": len(confirming_groups) == 1
            }
        )
    
    def detect_conflicts(self, signals: List[Signal]) -> Dict:
        """
        Detect and analyze conflicting signals
        
        Returns conflict analysis for logging/debugging
        """
        long_signals = [s for s in signals if s.direction == "LONG"]
        short_signals = [s for s in signals if s.direction == "SHORT"]
        neutral_signals = [s for s in signals if s.direction == "NEUTRAL"]
        
        has_conflict = len(long_signals) > 0 and len(short_signals) > 0
        
        conflict_strength = 0
        if has_conflict:
            long_strength = sum(s.confidence for s in long_signals)
            short_strength = sum(s.confidence for s in short_signals)
            total_strength = long_strength + short_strength
            
            if total_strength > 0:
                # Conflict strength = how balanced the opposing signals are
                conflict_strength = 1 - abs(long_strength - short_strength) / total_strength
        
        return {
            "has_conflict": has_conflict,
            "conflict_strength": conflict_strength,
            "long_count": len(long_signals),
            "short_count": len(short_signals),
            "neutral_count": len(neutral_signals),
            "long_strategies": [s.strategy for s in long_signals],
            "short_strategies": [s.strategy for s in short_signals]
        }
    
    def get_signal_summary(self, signals: List[Signal]) -> Dict:
        """Get summary of signals for logging"""
        if not signals:
            return {"count": 0}
        
        return {
            "count": len(signals),
            "symbol": signals[0].symbol,
            "strategies": [s.strategy for s in signals],
            "directions": {s.strategy: s.direction for s in signals},
            "confidences": {s.strategy: round(s.confidence, 3) for s in signals},
            "conflicts": self.detect_conflicts(signals)
        }
    
    def get_history_stats(self, last_n: int = 100) -> Dict:
        """Get statistics from signal history"""
        recent = self.signal_history[-last_n:] if self.signal_history else []
        
        if not recent:
            return {"count": 0}
        
        long_count = sum(1 for s in recent if s.direction == "LONG")
        short_count = sum(1 for s in recent if s.direction == "SHORT")
        avg_confidence = np.mean([s.confidence for s in recent])
        avg_signals = np.mean([s.signal_count for s in recent])
        
        return {
            "count": len(recent),
            "long_count": long_count,
            "short_count": short_count,
            "long_ratio": long_count / len(recent) if recent else 0,
            "avg_confidence": avg_confidence,
            "avg_contributing_signals": avg_signals
        }
