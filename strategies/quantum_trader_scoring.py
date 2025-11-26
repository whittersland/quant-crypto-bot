"""
Quantum Trader Scoring Strategy

Implements quantum-inspired concepts for aggregating copy trading signals:
- Superposition: Each trader exists in multiple success states simultaneously
- Entanglement: Correlated traders amplify signals, anti-correlated cancel
- Wave Function Collapse: Signal executes when observation exceeds threshold
- Heisenberg Uncertainty: Recent performance weighted more (lower time uncertainty)
- Quantum Interference: Conflicting signals interfere destructively
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import numpy as np
import logging
import math

from .base_strategy import BaseStrategy, StrategyType
from ..utils.indicators import Signal, MarketData

logger = logging.getLogger(__name__)


@dataclass
class TraderState:
    """Represents a trader's quantum state (superposition of success states)"""
    trader_id: str
    name: str
    
    # Performance metrics (observable states)
    win_rate: float = 0.5  # 0-1
    sharpe_ratio: float = 0.0  # Can be negative
    profit_factor: float = 1.0  # gross_profit / gross_loss
    avg_trade_pnl: float = 0.0
    total_trades: int = 0
    
    # Temporal data (for uncertainty calculation)
    last_trade_time: Optional[datetime] = None
    trading_days: int = 0
    
    # Current position/signal
    current_position: str = "NEUTRAL"  # LONG, SHORT, NEUTRAL
    position_symbol: Optional[str] = None
    position_size_pct: float = 0.0  # % of their portfolio
    position_confidence: float = 0.0
    
    # Quantum state properties
    probability_amplitude: complex = complex(1, 0)  # ψ
    coherence: float = 1.0  # How "pure" their state is (degrades with conflicting signals)
    
    def __post_init__(self):
        self._calculate_amplitude()
    
    def _calculate_amplitude(self):
        """Calculate probability amplitude from performance metrics"""
        # Amplitude based on composite score
        # Higher win rate, sharpe, profit factor = higher amplitude
        
        # Normalize metrics to 0-1 range
        win_score = self.win_rate  # Already 0-1
        
        # Sharpe typically -3 to +3, normalize to 0-1
        sharpe_score = min(1.0, max(0.0, (self.sharpe_ratio + 1) / 4))
        
        # Profit factor typically 0-3, normalize to 0-1
        pf_score = min(1.0, max(0.0, (self.profit_factor - 0.5) / 2))
        
        # Composite score
        composite = (win_score * 0.3 + sharpe_score * 0.4 + pf_score * 0.3)
        
        # Convert to amplitude (magnitude)
        magnitude = math.sqrt(composite)
        
        # Phase based on recent performance (profitable = 0, losing = π)
        phase = 0 if self.avg_trade_pnl >= 0 else math.pi / 2
        
        self.probability_amplitude = complex(
            magnitude * math.cos(phase),
            magnitude * math.sin(phase)
        )
    
    @property
    def probability(self) -> float:
        """Probability = |ψ|² (Born rule)"""
        return abs(self.probability_amplitude) ** 2


@dataclass
class EntanglementPair:
    """Represents entangled (correlated) trader pair"""
    trader_a_id: str
    trader_b_id: str
    correlation: float  # -1 to +1
    entanglement_strength: float  # 0 to 1
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_constructive(self) -> bool:
        """Positive correlation = constructive interference"""
        return self.correlation > 0.3
    
    @property
    def is_destructive(self) -> bool:
        """Negative correlation = destructive interference"""
        return self.correlation < -0.3


class QuantumTraderScoring(BaseStrategy):
    """
    Quantum-inspired copy trading signal aggregation
    
    Core Concepts:
    1. SUPERPOSITION: Each trader exists in multiple success states
       - State is weighted by win_rate, sharpe, recency, correlation
       - Superposition score = Σ(probability_amplitude²)
    
    2. ENTANGLEMENT: Correlated traders amplify/cancel signals
       - Traders become entangled if strategies correlate
       - Entangled pairs affect confidence calculation
    
    3. WAVE FUNCTION COLLAPSE: Signal executes when threshold exceeded
       - Observation (market check) collapses probability to definite state
       - Confidence = |Ψ|² where Ψ is combined wave function
    
    4. HEISENBERG UNCERTAINTY: Time-weighted performance
       - ΔE·Δt ≥ ℏ/2 (Energy-Time uncertainty analog)
       - Recent performance has lower uncertainty (higher weight)
       - Old data has higher uncertainty (lower weight)
    
    5. QUANTUM INTERFERENCE: Conflicting signals cancel
       - BUY + SELL from correlated traders = cancellation
       - Only constructive interference passes threshold
    """
    
    def __init__(
        self,
        symbols: List[str],
        min_traders_for_signal: int = 3,
        collapse_threshold: float = 0.6,  # Wave function collapse threshold
        uncertainty_halflife_days: float = 7.0,  # Heisenberg uncertainty decay
        entanglement_window_days: int = 30,  # Window for correlation calc
        weight: float = 0.15,
        enabled: bool = True,
        params: Dict[str, Any] = None
    ):
        super().__init__(
            name="quantum_trader_scoring",
            strategy_type=StrategyType.DUAL_CONDITION,  # Requires multiple confirmations
            symbols=symbols,
            weight=weight,
            enabled=enabled,
            params=params or {}
        )
        
        self.min_traders_for_signal = min_traders_for_signal
        self.collapse_threshold = collapse_threshold
        self.uncertainty_halflife_days = uncertainty_halflife_days
        self.entanglement_window_days = entanglement_window_days
        
        # Trader registry
        self.traders: Dict[str, TraderState] = {}
        
        # Entanglement matrix (trader pairs)
        self.entanglements: Dict[Tuple[str, str], EntanglementPair] = {}
        
        # Signal history per symbol
        self.trader_signals: Dict[str, List[Dict]] = {s: [] for s in symbols}
        
        # Wave function state per symbol
        self.wave_functions: Dict[str, complex] = {s: complex(0, 0) for s in symbols}
        
        logger.info(f"QuantumTraderScoring initialized with collapse_threshold={collapse_threshold}")
    
    def get_required_history(self) -> int:
        return 30  # Need some market data for context
    
    def register_trader(self, trader: TraderState) -> None:
        """Register a trader to the quantum system"""
        self.traders[trader.trader_id] = trader
        logger.info(f"Registered trader: {trader.name} (probability={trader.probability:.3f})")
    
    def update_trader(
        self,
        trader_id: str,
        win_rate: float = None,
        sharpe_ratio: float = None,
        profit_factor: float = None,
        avg_trade_pnl: float = None,
        current_position: str = None,
        position_symbol: str = None,
        position_size_pct: float = None
    ) -> None:
        """Update a trader's state"""
        if trader_id not in self.traders:
            logger.warning(f"Trader {trader_id} not registered")
            return
        
        trader = self.traders[trader_id]
        
        if win_rate is not None:
            trader.win_rate = win_rate
        if sharpe_ratio is not None:
            trader.sharpe_ratio = sharpe_ratio
        if profit_factor is not None:
            trader.profit_factor = profit_factor
        if avg_trade_pnl is not None:
            trader.avg_trade_pnl = avg_trade_pnl
        if current_position is not None:
            trader.current_position = current_position
            trader.last_trade_time = datetime.now(timezone.utc)
        if position_symbol is not None:
            trader.position_symbol = position_symbol
        if position_size_pct is not None:
            trader.position_size_pct = position_size_pct
        
        # Recalculate amplitude
        trader._calculate_amplitude()
    
    def record_trader_signal(
        self,
        trader_id: str,
        symbol: str,
        direction: str,
        confidence: float = 0.5
    ) -> None:
        """Record a signal from a trader"""
        if trader_id not in self.traders:
            return
        
        trader = self.traders[trader_id]
        trader.current_position = direction
        trader.position_symbol = symbol
        trader.position_confidence = confidence
        trader.last_trade_time = datetime.now(timezone.utc)
        
        # Add to signal history
        if symbol not in self.trader_signals:
            self.trader_signals[symbol] = []
        
        self.trader_signals[symbol].append({
            "trader_id": trader_id,
            "direction": direction,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc),
            "trader_probability": trader.probability
        })
        
        # Keep only recent signals (last hour)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        self.trader_signals[symbol] = [
            s for s in self.trader_signals[symbol]
            if s["timestamp"] > cutoff
        ]
    
    def _calculate_heisenberg_weight(self, trader: TraderState) -> float:
        """
        Calculate time-uncertainty weight (Heisenberg-inspired)
        
        Recent performance has lower time uncertainty = higher weight
        Older data has higher uncertainty = lower weight
        
        W(t) = exp(-t/τ) where τ is halflife
        """
        if not trader.last_trade_time:
            return 0.5  # Default weight for unknown recency
        
        now = datetime.now(timezone.utc)
        days_elapsed = (now - trader.last_trade_time).total_seconds() / 86400
        
        # Exponential decay
        decay_constant = math.log(2) / self.uncertainty_halflife_days
        weight = math.exp(-decay_constant * days_elapsed)
        
        return max(0.1, min(1.0, weight))
    
    def _calculate_entanglement(
        self,
        trader_a: TraderState,
        trader_b: TraderState
    ) -> EntanglementPair:
        """
        Calculate entanglement between two traders
        
        Traders are entangled if their signals correlate over time
        """
        pair_key = tuple(sorted([trader_a.trader_id, trader_b.trader_id]))
        
        # If we have cached entanglement, check if still valid
        if pair_key in self.entanglements:
            cached = self.entanglements[pair_key]
            age = (datetime.now(timezone.utc) - cached.last_updated).days
            if age < 1:  # Use cached if less than 1 day old
                return cached
        
        # Calculate correlation from signal history
        # (In production, this would use actual trade history)
        
        # Simplified: correlation based on current position agreement
        if trader_a.position_symbol == trader_b.position_symbol:
            if trader_a.current_position == trader_b.current_position:
                correlation = 0.7  # Same direction = positive correlation
            elif trader_a.current_position == "NEUTRAL" or trader_b.current_position == "NEUTRAL":
                correlation = 0.0  # One neutral = no correlation
            else:
                correlation = -0.7  # Opposite direction = negative correlation
        else:
            correlation = 0.0  # Different symbols = no correlation
        
        # Entanglement strength based on combined probability
        strength = math.sqrt(trader_a.probability * trader_b.probability)
        
        entanglement = EntanglementPair(
            trader_a_id=trader_a.trader_id,
            trader_b_id=trader_b.trader_id,
            correlation=correlation,
            entanglement_strength=strength
        )
        
        self.entanglements[pair_key] = entanglement
        return entanglement
    
    def _calculate_wave_function(self, symbol: str) -> Tuple[complex, Dict]:
        """
        Calculate combined wave function for a symbol
        
        Ψ_total = Σ(w_i * ψ_i) where w_i includes uncertainty and entanglement
        
        Returns:
            Tuple of (wave_function, metadata)
        """
        # Get traders with signals for this symbol
        relevant_traders = [
            t for t in self.traders.values()
            if t.position_symbol == symbol and t.current_position != "NEUTRAL"
        ]
        
        if len(relevant_traders) < self.min_traders_for_signal:
            return complex(0, 0), {"reason": "Insufficient traders", "count": len(relevant_traders)}
        
        # Separate by direction
        long_traders = [t for t in relevant_traders if t.current_position == "LONG"]
        short_traders = [t for t in relevant_traders if t.current_position == "SHORT"]
        
        # Calculate wave functions for each direction
        long_psi = complex(0, 0)
        short_psi = complex(0, 0)
        
        # Process LONG traders
        for trader in long_traders:
            # Heisenberg weight (time uncertainty)
            h_weight = self._calculate_heisenberg_weight(trader)
            
            # Add amplitude to long wave
            weighted_amp = trader.probability_amplitude * h_weight
            long_psi += weighted_amp
        
        # Process SHORT traders
        for trader in short_traders:
            h_weight = self._calculate_heisenberg_weight(trader)
            
            # Short signals have opposite phase
            weighted_amp = trader.probability_amplitude * h_weight * complex(-1, 0)
            short_psi += weighted_amp
        
        # Apply entanglement effects
        entanglement_boost = 0.0
        
        # Check all pairs within same direction (constructive)
        for i, t1 in enumerate(long_traders):
            for t2 in long_traders[i+1:]:
                ent = self._calculate_entanglement(t1, t2)
                if ent.is_constructive:
                    entanglement_boost += ent.entanglement_strength * 0.1
        
        for i, t1 in enumerate(short_traders):
            for t2 in short_traders[i+1:]:
                ent = self._calculate_entanglement(t1, t2)
                if ent.is_constructive:
                    entanglement_boost += ent.entanglement_strength * 0.1
        
        # Combine long and short (interference)
        total_psi = long_psi + short_psi
        
        # Apply entanglement boost to magnitude
        if entanglement_boost > 0:
            magnitude = abs(total_psi) * (1 + entanglement_boost)
            phase = np.angle(total_psi)
            total_psi = complex(magnitude * np.cos(phase), magnitude * np.sin(phase))
        
        # Store for reference
        self.wave_functions[symbol] = total_psi
        
        metadata = {
            "long_traders": len(long_traders),
            "short_traders": len(short_traders),
            "long_amplitude": abs(long_psi),
            "short_amplitude": abs(short_psi),
            "entanglement_boost": entanglement_boost,
            "total_amplitude": abs(total_psi),
            "phase": np.angle(total_psi)
        }
        
        return total_psi, metadata
    
    def _collapse_wave_function(
        self,
        symbol: str,
        psi: complex,
        metadata: Dict
    ) -> Tuple[str, float]:
        """
        Collapse wave function to definite state (LONG, SHORT, or NEUTRAL)
        
        Measurement causes collapse:
        - Probability = |Ψ|²
        - Direction from phase
        
        Returns:
            Tuple of (direction, confidence)
        """
        probability = abs(psi) ** 2
        phase = np.angle(psi)
        
        # Normalize probability to 0-1 range
        # (Can exceed 1 due to constructive interference)
        normalized_prob = min(1.0, probability)
        
        # Check threshold for collapse
        if normalized_prob < self.collapse_threshold:
            return "NEUTRAL", normalized_prob
        
        # Determine direction from phase
        # Phase near 0 = LONG (from long traders' positive amplitude)
        # Phase near π = SHORT (from short traders' negative amplitude)
        if abs(phase) < math.pi / 2:
            # More long contribution
            if metadata["long_amplitude"] > metadata["short_amplitude"]:
                direction = "LONG"
            else:
                direction = "NEUTRAL"  # Shouldn't happen but safety
        else:
            # More short contribution
            if metadata["short_amplitude"] > metadata["long_amplitude"]:
                direction = "SHORT"
            else:
                direction = "NEUTRAL"
        
        # Final confidence = probability with direction confidence
        if direction != "NEUTRAL":
            dominant = max(metadata["long_amplitude"], metadata["short_amplitude"])
            total = metadata["long_amplitude"] + metadata["short_amplitude"]
            direction_confidence = dominant / total if total > 0 else 0
            confidence = normalized_prob * direction_confidence
        else:
            confidence = 0.0
        
        return direction, confidence
    
    def analyze(self, symbol: str, market_data: MarketData) -> Optional[Signal]:
        """
        Analyze using quantum trader scoring
        
        Process:
        1. Gather trader states (superposition)
        2. Calculate wave function (interference)
        3. Apply entanglement effects
        4. Collapse wave function to signal
        """
        # Calculate combined wave function
        psi, metadata = self._calculate_wave_function(symbol)
        
        if abs(psi) == 0:
            return None
        
        # Collapse to definite state
        direction, confidence = self._collapse_wave_function(symbol, psi, metadata)
        
        if direction == "NEUTRAL" or confidence < 0.15:
            return None
        
        # Generate signal
        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "quantum_state": {
                    "wave_amplitude": abs(psi),
                    "wave_phase": np.angle(psi),
                    "probability": abs(psi) ** 2
                },
                "traders": {
                    "long_count": metadata["long_traders"],
                    "short_count": metadata["short_traders"],
                    "total_required": self.min_traders_for_signal
                },
                "interference": {
                    "long_amplitude": metadata["long_amplitude"],
                    "short_amplitude": metadata["short_amplitude"],
                    "entanglement_boost": metadata["entanglement_boost"]
                },
                "collapse_threshold": self.collapse_threshold
            }
        )
    
    def get_quantum_state_report(self, symbol: str = None) -> Dict:
        """Get current quantum state for debugging/monitoring"""
        report = {
            "traders_registered": len(self.traders),
            "entanglement_pairs": len(self.entanglements),
            "collapse_threshold": self.collapse_threshold,
            "uncertainty_halflife_days": self.uncertainty_halflife_days
        }
        
        # Per-trader states
        report["trader_states"] = {}
        for tid, trader in self.traders.items():
            report["trader_states"][tid] = {
                "name": trader.name,
                "probability": trader.probability,
                "amplitude": abs(trader.probability_amplitude),
                "phase": np.angle(trader.probability_amplitude),
                "current_position": trader.current_position,
                "position_symbol": trader.position_symbol,
                "heisenberg_weight": self._calculate_heisenberg_weight(trader)
            }
        
        # Per-symbol wave functions
        if symbol:
            symbols = [symbol]
        else:
            symbols = self.symbols
        
        report["wave_functions"] = {}
        for sym in symbols:
            psi, meta = self._calculate_wave_function(sym)
            direction, confidence = self._collapse_wave_function(sym, psi, meta)
            
            report["wave_functions"][sym] = {
                "amplitude": abs(psi),
                "phase": np.angle(psi),
                "probability": abs(psi) ** 2,
                "collapsed_direction": direction,
                "collapsed_confidence": confidence,
                "metadata": meta
            }
        
        return report


# Helper function to create sample traders for testing
def create_sample_traders(symbols: List[str]) -> List[TraderState]:
    """Create sample traders for testing"""
    return [
        TraderState(
            trader_id="trader_1",
            name="CryptoKing",
            win_rate=0.68,
            sharpe_ratio=1.8,
            profit_factor=2.1,
            avg_trade_pnl=45.0,
            total_trades=520
        ),
        TraderState(
            trader_id="trader_2",
            name="BTCWhale",
            win_rate=0.72,
            sharpe_ratio=2.1,
            profit_factor=2.5,
            avg_trade_pnl=82.0,
            total_trades=340
        ),
        TraderState(
            trader_id="trader_3",
            name="SolanaSniper",
            win_rate=0.61,
            sharpe_ratio=1.4,
            profit_factor=1.7,
            avg_trade_pnl=28.0,
            total_trades=890
        ),
        TraderState(
            trader_id="trader_4",
            name="DeFiMaster",
            win_rate=0.55,
            sharpe_ratio=0.9,
            profit_factor=1.3,
            avg_trade_pnl=15.0,
            total_trades=1200
        ),
        TraderState(
            trader_id="trader_5",
            name="MomentumTrader",
            win_rate=0.64,
            sharpe_ratio=1.6,
            profit_factor=1.9,
            avg_trade_pnl=35.0,
            total_trades=680
        ),
    ]
