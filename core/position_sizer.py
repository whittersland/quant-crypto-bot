"""
Adaptive Position Sizing Module
Implements dynamic position sizing that increases on profitable days
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class DailyRecord:
    """Record of a trading day"""
    date: str
    starting_capital: float
    ending_capital: float
    pnl: float
    pnl_percent: float
    trades: int
    profitable: bool


@dataclass
class SizingState:
    """Current position sizing state"""
    current_min: float
    current_max: float
    consecutive_profit_days: int
    last_adjustment_date: Optional[str]
    history: list = field(default_factory=list)


class AdaptivePositionSizer:
    """
    Adaptive position sizing that grows on profitable days
    
    Rules:
    - Base range: $15-$60 per trade
    - After 24 hours of profit: increase range by $5
    - On losing day: reset to base range
    - Max concurrent positions: 4
    """
    
    def __init__(
        self,
        base_min: float = 15.0,
        base_max: float = 60.0,
        increment: float = 5.0,
        max_concurrent: int = 4,
        profit_check_hours: int = 24,
        state_file: str = "sizing_state.json"
    ):
        self.base_min = base_min
        self.base_max = base_max
        self.increment = increment
        self.max_concurrent = max_concurrent
        self.profit_check_hours = profit_check_hours
        self.state_file = state_file
        
        # Current state
        self.current_min = base_min
        self.current_max = base_max
        self.consecutive_profit_days = 0
        self.last_check_time: Optional[datetime] = None
        self.last_check_capital: Optional[float] = None
        
        # Daily tracking
        self.day_start_capital: Optional[float] = None
        self.current_day: Optional[str] = None
        self.daily_history: list = []
        
        # Load saved state if exists
        self._load_state()
    
    def _load_state(self) -> None:
        """Load saved state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                self.current_min = data.get("current_min", self.base_min)
                self.current_max = data.get("current_max", self.base_max)
                self.consecutive_profit_days = data.get("consecutive_profit_days", 0)
                self.daily_history = data.get("history", [])
                
                last_check = data.get("last_check_time")
                if last_check:
                    self.last_check_time = datetime.fromisoformat(last_check)
                
                logger.info(f"Loaded sizing state: ${self.current_min}-${self.current_max}")
        except Exception as e:
            logger.warning(f"Could not load sizing state: {e}")
    
    def _save_state(self) -> None:
        """Save current state to file"""
        try:
            data = {
                "current_min": self.current_min,
                "current_max": self.current_max,
                "consecutive_profit_days": self.consecutive_profit_days,
                "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
                "history": self.daily_history[-30:]  # Keep last 30 days
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save sizing state: {e}")
    
    def initialize_day(self, capital: float) -> None:
        """Initialize tracking for a new day"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if self.current_day != today:
            # Record previous day if exists
            if self.current_day and self.day_start_capital:
                self._record_day(capital)
            
            self.current_day = today
            self.day_start_capital = capital
            logger.info(f"New trading day initialized: {today}, capital: ${capital:.2f}")
    
    def _record_day(self, ending_capital: float) -> None:
        """Record completed trading day"""
        if not self.day_start_capital:
            return
        
        pnl = ending_capital - self.day_start_capital
        pnl_percent = (pnl / self.day_start_capital) * 100 if self.day_start_capital > 0 else 0
        
        record = {
            "date": self.current_day,
            "starting_capital": self.day_start_capital,
            "ending_capital": ending_capital,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "profitable": pnl > 0,
            "sizing_range": f"${self.current_min}-${self.current_max}"
        }
        
        self.daily_history.append(record)
        logger.info(f"Day recorded: {self.current_day}, P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
    
    def check_and_adjust(self, current_capital: float) -> Tuple[bool, str]:
        """
        Check if 24 hours passed and adjust sizing if profitable
        
        Returns:
            Tuple of (was_adjusted, adjustment_message)
        """
        now = datetime.now(timezone.utc)
        
        # Initialize if first check
        if self.last_check_time is None:
            self.last_check_time = now
            self.last_check_capital = current_capital
            return False, "Initial check recorded"
        
        # Check if enough time has passed
        hours_elapsed = (now - self.last_check_time).total_seconds() / 3600
        
        if hours_elapsed < self.profit_check_hours:
            remaining = self.profit_check_hours - hours_elapsed
            return False, f"Next check in {remaining:.1f} hours"
        
        # Calculate profit/loss over period
        pnl = current_capital - self.last_check_capital
        profitable = pnl > 0
        
        # Update state
        self.last_check_time = now
        self.last_check_capital = current_capital
        
        if profitable:
            # Increase range
            self.consecutive_profit_days += 1
            old_min, old_max = self.current_min, self.current_max
            
            self.current_min += self.increment
            self.current_max += self.increment
            
            self._save_state()
            
            message = (
                f"Profitable period! Range increased: ${old_min}-${old_max} → "
                f"${self.current_min}-${self.current_max} "
                f"(Consecutive profit periods: {self.consecutive_profit_days})"
            )
            logger.info(message)
            return True, message
        
        else:
            # Reset to base
            old_min, old_max = self.current_min, self.current_max
            old_streak = self.consecutive_profit_days
            
            self.current_min = self.base_min
            self.current_max = self.base_max
            self.consecutive_profit_days = 0
            
            self._save_state()
            
            message = (
                f"Loss period - resetting range: ${old_min}-${old_max} → "
                f"${self.current_min}-${self.current_max} "
                f"(Streak ended at {old_streak} periods)"
            )
            logger.info(message)
            return True, message
    
    def calculate_size(
        self,
        confidence: float,
        available_capital: float,
        current_positions: int,
        risk_multiplier: float = 1.0
    ) -> Tuple[float, Dict]:
        """
        Calculate position size for a trade
        
        Position size scales with confidence:
        - 0.65 confidence (minimum threshold) → base_min size
        - 0.80 confidence → midpoint size
        - 0.95+ confidence → base_max size
        
        Args:
            confidence: Signal confidence (0.0 to 1.0)
            available_capital: Currently available capital
            current_positions: Number of open positions
            risk_multiplier: Risk-based multiplier (from risk manager)
        
        Returns:
            Tuple of (position_size, sizing_details)
        """
        # Check position limit
        if current_positions >= self.max_concurrent:
            return 0.0, {
                "reason": "Max concurrent positions reached",
                "max_positions": self.max_concurrent,
                "current_positions": current_positions
            }
        
        # Scale confidence to position size
        # Map confidence from threshold range (0.65-1.0) to size range (min-max)
        min_confidence = 0.65  # Our minimum threshold
        max_confidence = 0.95  # Treat anything above this as max confidence
        
        # Normalize confidence to 0-1 range within our thresholds
        normalized_conf = (confidence - min_confidence) / (max_confidence - min_confidence)
        normalized_conf = max(0.0, min(1.0, normalized_conf))  # Clamp to 0-1
        
        # Apply exponential scaling - high confidence gets disproportionately more
        # This makes 0.95+ confidence get ~2x the size of 0.65 confidence
        confidence_multiplier = normalized_conf ** 0.7  # Exponential curve
        
        # Calculate base size from confidence
        size_range = self.current_max - self.current_min
        base_size = self.current_min + (size_range * confidence_multiplier)
        
        # Apply risk multiplier
        adjusted_size = base_size * risk_multiplier
        
        # Ensure within bounds
        adjusted_size = max(self.current_min * risk_multiplier, adjusted_size)
        adjusted_size = min(self.current_max, adjusted_size)
        
        # Check against available capital (with leverage factor)
        leverage_factor = 10  # Match our 10x leverage setting
        max_from_capital = available_capital * leverage_factor * 0.9  # 90% of leveraged capital
        remaining_slots = self.max_concurrent - current_positions
        max_per_position = max_from_capital / remaining_slots if remaining_slots > 0 else 0
        
        final_size = min(adjusted_size, max_per_position)
        
        # Don't trade if size too small
        if final_size < self.current_min * 0.5:
            return 0.0, {
                "reason": "Insufficient capital for minimum position",
                "calculated_size": final_size,
                "minimum_required": self.current_min * 0.5
            }
        
        # Log confidence-based sizing
        logger.info(f"Position sizing: conf={confidence:.2f} → normalized={normalized_conf:.2f} → size=${final_size:.2f}")
        
        details = {
            "base_range": f"${self.current_min}-${self.current_max}",
            "confidence": confidence,
            "normalized_confidence": normalized_conf,
            "confidence_multiplier": confidence_multiplier,
            "confidence_size": base_size,
            "risk_multiplier": risk_multiplier,
            "adjusted_size": adjusted_size,
            "capital_limit": max_per_position,
            "final_size": final_size,
            "consecutive_profit_periods": self.consecutive_profit_days
        }
        
        return final_size, details
    
    def get_current_range(self) -> Tuple[float, float]:
        """Get current position size range"""
        return self.current_min, self.current_max
    
    def get_stats(self) -> Dict:
        """Get position sizing statistics"""
        return {
            "base_range": f"${self.base_min}-${self.base_max}",
            "current_range": f"${self.current_min}-${self.current_max}",
            "increment": self.increment,
            "consecutive_profit_periods": self.consecutive_profit_days,
            "total_increases": int((self.current_min - self.base_min) / self.increment),
            "max_concurrent_positions": self.max_concurrent,
            "profit_check_hours": self.profit_check_hours,
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "recent_history": self.daily_history[-7:]  # Last 7 days
        }
    
    def force_reset(self) -> None:
        """Force reset to base sizing (manual override)"""
        self.current_min = self.base_min
        self.current_max = self.base_max
        self.consecutive_profit_days = 0
        self._save_state()
        logger.info(f"Position sizing force reset to ${self.base_min}-${self.base_max}")
    
    def force_adjust(self, new_min: float, new_max: float) -> None:
        """Force specific sizing (manual override)"""
        self.current_min = new_min
        self.current_max = new_max
        self._save_state()
        logger.info(f"Position sizing manually set to ${new_min}-${new_max}")
