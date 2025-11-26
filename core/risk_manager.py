"""
Risk Management Module
Handles trailing stops, daily limits, drawdown protection, and position risk
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    HALT = "halt"


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    daily_pnl: float = 0.0
    daily_pnl_percent: float = 0.0
    total_drawdown: float = 0.0
    total_drawdown_percent: float = 0.0
    open_risk: float = 0.0
    risk_level: RiskLevel = RiskLevel.NORMAL
    positions_at_risk: List[str] = field(default_factory=list)
    can_trade: bool = True
    reason: str = ""


@dataclass
class StopLevel:
    """Stop loss level for a position"""
    symbol: str
    entry_price: float
    current_stop: float
    highest_price: float
    stop_type: str  # "trailing" or "fixed"
    triggered: bool = False


class TrailingStopManager:
    """Manages trailing stops for all positions"""
    
    def __init__(self, trailing_percent: float = 12.0):
        self.trailing_percent = trailing_percent
        self.stops: Dict[str, StopLevel] = {}
    
    def add_position(self, symbol: str, entry_price: float, side: str = "LONG") -> StopLevel:
        """Add a new position with trailing stop"""
        if side == "LONG":
            initial_stop = entry_price * (1 - self.trailing_percent / 100)
        else:
            initial_stop = entry_price * (1 + self.trailing_percent / 100)
        
        stop = StopLevel(
            symbol=symbol,
            entry_price=entry_price,
            current_stop=initial_stop,
            highest_price=entry_price,
            stop_type="trailing"
        )
        
        self.stops[symbol] = stop
        logger.info(f"Added trailing stop for {symbol}: entry={entry_price:.2f}, stop={initial_stop:.2f}")
        
        return stop
    
    def update_price(self, symbol: str, current_price: float, side: str = "LONG") -> Tuple[bool, Optional[float]]:
        """
        Update price and adjust trailing stop
        
        Returns:
            Tuple of (stop_triggered, new_stop_price)
        """
        if symbol not in self.stops:
            return False, None
        
        stop = self.stops[symbol]
        
        if side == "LONG":
            # Update highest price for long positions
            if current_price > stop.highest_price:
                stop.highest_price = current_price
                new_stop = current_price * (1 - self.trailing_percent / 100)
                
                if new_stop > stop.current_stop:
                    stop.current_stop = new_stop
                    logger.debug(f"Trailing stop raised for {symbol}: {stop.current_stop:.2f}")
            
            # Check if stop triggered
            if current_price <= stop.current_stop:
                stop.triggered = True
                logger.warning(f"STOP TRIGGERED for {symbol} at {current_price:.2f} (stop: {stop.current_stop:.2f})")
                return True, stop.current_stop
        
        else:  # SHORT
            # Update lowest price for short positions
            if current_price < stop.highest_price:
                stop.highest_price = current_price
                new_stop = current_price * (1 + self.trailing_percent / 100)
                
                if new_stop < stop.current_stop:
                    stop.current_stop = new_stop
                    logger.debug(f"Trailing stop lowered for {symbol}: {stop.current_stop:.2f}")
            
            # Check if stop triggered
            if current_price >= stop.current_stop:
                stop.triggered = True
                logger.warning(f"STOP TRIGGERED for {symbol} at {current_price:.2f} (stop: {stop.current_stop:.2f})")
                return True, stop.current_stop
        
        return False, stop.current_stop
    
    def remove_position(self, symbol: str) -> None:
        """Remove position from tracking"""
        if symbol in self.stops:
            del self.stops[symbol]
            logger.info(f"Removed trailing stop for {symbol}")
    
    def get_stop(self, symbol: str) -> Optional[StopLevel]:
        """Get current stop level for a position"""
        return self.stops.get(symbol)
    
    def get_all_stops(self) -> Dict[str, StopLevel]:
        """Get all current stops"""
        return self.stops.copy()


class DailyLossTracker:
    """Tracks daily P&L and enforces daily loss limits"""
    
    def __init__(self, initial_capital: float, daily_limit_percent: float = 15.0):
        self.initial_capital = initial_capital
        self.daily_limit_percent = daily_limit_percent
        self.daily_limit_amount = initial_capital * (daily_limit_percent / 100)
        
        self.day_start_capital = initial_capital
        self.current_capital = initial_capital
        self.realized_pnl_today = 0.0
        self.unrealized_pnl = 0.0
        self.last_reset = datetime.now(timezone.utc).date()
        
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
    
    def _check_day_reset(self) -> None:
        """Reset daily counters if new day"""
        today = datetime.now(timezone.utc).date()
        
        if today > self.last_reset:
            logger.info(f"New trading day - resetting daily counters. Previous day P&L: ${self.realized_pnl_today:.2f}")
            
            self.day_start_capital = self.current_capital
            self.daily_limit_amount = self.current_capital * (self.daily_limit_percent / 100)
            self.realized_pnl_today = 0.0
            self.trades_today = 0
            self.wins_today = 0
            self.losses_today = 0
            self.last_reset = today
    
    def record_trade(self, pnl: float) -> None:
        """Record a completed trade"""
        self._check_day_reset()
        
        self.realized_pnl_today += pnl
        self.current_capital += pnl
        self.trades_today += 1
        
        if pnl >= 0:
            self.wins_today += 1
        else:
            self.losses_today += 1
        
        logger.info(f"Trade recorded: P&L=${pnl:.2f}, Daily total=${self.realized_pnl_today:.2f}")
    
    def update_unrealized(self, unrealized: float) -> None:
        """Update unrealized P&L"""
        self._check_day_reset()
        self.unrealized_pnl = unrealized
    
    def update_capital(self, new_capital: float) -> None:
        """Update current capital"""
        self._check_day_reset()
        self.current_capital = new_capital
    
    @property
    def total_daily_pnl(self) -> float:
        """Total daily P&L including unrealized"""
        return self.realized_pnl_today + self.unrealized_pnl
    
    @property
    def daily_pnl_percent(self) -> float:
        """Daily P&L as percentage of day start capital"""
        if self.day_start_capital == 0:
            return 0.0
        return (self.total_daily_pnl / self.day_start_capital) * 100
    
    @property
    def remaining_daily_risk(self) -> float:
        """Remaining risk budget for the day"""
        return max(0, self.daily_limit_amount + self.total_daily_pnl)
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on daily limits"""
        self._check_day_reset()
        
        # Check if daily loss limit hit
        if self.total_daily_pnl <= -self.daily_limit_amount:
            return False, f"Daily loss limit reached: ${self.total_daily_pnl:.2f} (limit: -${self.daily_limit_amount:.2f})"
        
        # Warning if approaching limit
        if self.total_daily_pnl <= -self.daily_limit_amount * 0.8:
            logger.warning(f"Approaching daily loss limit: ${self.total_daily_pnl:.2f}")
        
        return True, "OK"
    
    def get_daily_stats(self) -> Dict:
        """Get daily trading statistics"""
        self._check_day_reset()
        
        win_rate = (self.wins_today / self.trades_today * 100) if self.trades_today > 0 else 0
        
        return {
            "date": self.last_reset.isoformat(),
            "trades": self.trades_today,
            "wins": self.wins_today,
            "losses": self.losses_today,
            "win_rate": win_rate,
            "realized_pnl": self.realized_pnl_today,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_daily_pnl,
            "pnl_percent": self.daily_pnl_percent,
            "remaining_risk": self.remaining_daily_risk
        }
    
    def reset(self) -> None:
        """Reset daily tracker - use after fixing erroneous P&L"""
        self.realized_pnl_today = 0.0
        self.unrealized_pnl = 0.0
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.last_reset = datetime.now(timezone.utc).date()  # Use .date() for consistency
        logger.info("Daily loss tracker reset")


class DrawdownProtection:
    """Monitors and enforces maximum drawdown limits"""
    
    def __init__(self, initial_capital: float, max_drawdown_percent: float = 25.0):
        self.initial_capital = initial_capital
        self.max_drawdown_percent = max_drawdown_percent
        self.max_drawdown_amount = initial_capital * (max_drawdown_percent / 100)
        
        self.peak_capital = initial_capital
        self.current_capital = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown_hit = 0.0
    
    def update(self, current_capital: float) -> Tuple[bool, float]:
        """
        Update capital and check drawdown
        
        Returns:
            Tuple of (is_within_limits, current_drawdown_percent)
        """
        self.current_capital = current_capital
        
        # Update peak if new high
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        # Calculate current drawdown
        self.current_drawdown = self.peak_capital - current_capital
        drawdown_percent = (self.current_drawdown / self.peak_capital) * 100 if self.peak_capital > 0 else 0
        
        # Track max drawdown hit
        if self.current_drawdown > self.max_drawdown_hit:
            self.max_drawdown_hit = self.current_drawdown
        
        # Check if limit exceeded
        if drawdown_percent >= self.max_drawdown_percent:
            logger.critical(f"MAX DRAWDOWN EXCEEDED: {drawdown_percent:.2f}% (limit: {self.max_drawdown_percent}%)")
            return False, drawdown_percent
        
        # Warnings at various levels
        if drawdown_percent >= self.max_drawdown_percent * 0.8:
            logger.warning(f"Drawdown at {drawdown_percent:.2f}% - approaching limit")
        elif drawdown_percent >= self.max_drawdown_percent * 0.5:
            logger.info(f"Drawdown at {drawdown_percent:.2f}%")
        
        return True, drawdown_percent
    
    @property
    def drawdown_percent(self) -> float:
        """Current drawdown as percentage"""
        if self.peak_capital == 0:
            return 0.0
        return (self.current_drawdown / self.peak_capital) * 100
    
    @property
    def remaining_drawdown_budget(self) -> float:
        """Remaining drawdown budget in dollars"""
        return max(0, self.max_drawdown_amount - self.current_drawdown)
    
    def get_stats(self) -> Dict:
        """Get drawdown statistics"""
        return {
            "initial_capital": self.initial_capital,
            "peak_capital": self.peak_capital,
            "current_capital": self.current_capital,
            "current_drawdown": self.current_drawdown,
            "current_drawdown_percent": self.drawdown_percent,
            "max_drawdown_hit": self.max_drawdown_hit,
            "max_allowed_percent": self.max_drawdown_percent,
            "remaining_budget": self.remaining_drawdown_budget
        }
    
    def reset(self, new_capital: Optional[float] = None):
        """
        Reset drawdown tracking - useful after fixing erroneous P&L calculations
        
        Args:
            new_capital: New capital to use (defaults to current_capital)
        """
        capital = new_capital or self.current_capital or self.initial_capital
        self.peak_capital = capital
        self.current_capital = capital
        self.current_drawdown = 0.0
        self.max_drawdown_hit = 0.0
        logger.info(f"Drawdown tracking reset. New capital baseline: ${capital:.2f}")


class TakeProfitManager:
    """Manages tiered take-profit levels"""
    
    def __init__(self, tiers: List[Dict] = None):
        """
        Initialize with take-profit tiers
        
        Args:
            tiers: List of dicts with 'threshold' (profit %) and 'take_percent' (% to close)
        """
        self.tiers = tiers or [
            {"threshold": 25, "take_percent": 30},
            {"threshold": 50, "take_percent": 40},
            {"threshold": 100, "take_percent": 50}
        ]
        
        # Sort tiers by threshold
        self.tiers = sorted(self.tiers, key=lambda x: x["threshold"])
        
        # Track which tiers have been hit per position
        self.position_tiers_hit: Dict[str, List[int]] = {}
    
    def add_position(self, symbol: str) -> None:
        """Track a new position"""
        self.position_tiers_hit[symbol] = []
    
    def check_take_profit(
        self, 
        symbol: str, 
        entry_price: float, 
        current_price: float,
        position_size: float,
        side: str = "LONG"
    ) -> Optional[Tuple[float, float]]:
        """
        Check if any take-profit tier is hit
        
        Returns:
            Tuple of (size_to_close, tier_threshold) if TP hit, None otherwise
        """
        if symbol not in self.position_tiers_hit:
            self.position_tiers_hit[symbol] = []
        
        # Calculate profit percentage
        if side == "LONG":
            profit_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_percent = ((entry_price - current_price) / entry_price) * 100
        
        # Check each tier
        for i, tier in enumerate(self.tiers):
            threshold = tier["threshold"]
            take_percent = tier["take_percent"]
            
            # Skip already-hit tiers
            if i in self.position_tiers_hit[symbol]:
                continue
            
            # Check if tier hit
            if profit_percent >= threshold:
                self.position_tiers_hit[symbol].append(i)
                
                # Calculate size to close
                size_to_close = position_size * (take_percent / 100)
                
                logger.info(f"Take-profit tier hit for {symbol}: {threshold}% profit, closing {take_percent}%")
                
                return size_to_close, threshold
        
        return None
    
    def remove_position(self, symbol: str) -> None:
        """Remove position from tracking"""
        if symbol in self.position_tiers_hit:
            del self.position_tiers_hit[symbol]
    
    def reset_position(self, symbol: str) -> None:
        """Reset tier tracking for a position"""
        self.position_tiers_hit[symbol] = []


class RiskManager:
    """
    Main risk management coordinator
    Combines all risk components into unified risk assessment
    """
    
    def __init__(
        self,
        initial_capital: float,
        trailing_stop_percent: float = 12.0,
        daily_loss_limit_percent: float = 15.0,
        max_drawdown_percent: float = 25.0,
        position_stop_loss_percent: float = 8.0,
        take_profit_tiers: List[Dict] = None
    ):
        self.initial_capital = initial_capital
        self.position_stop_loss_percent = position_stop_loss_percent
        
        # Initialize components
        self.trailing_stops = TrailingStopManager(trailing_stop_percent)
        self.daily_tracker = DailyLossTracker(initial_capital, daily_loss_limit_percent)
        self.drawdown = DrawdownProtection(initial_capital, max_drawdown_percent)
        self.take_profit = TakeProfitManager(take_profit_tiers)
        
        # Trading state
        self.is_halted = False
        self.halt_reason = ""
    
    def register_position(self, symbol: str, entry_price: float, side: str = "LONG") -> None:
        """Register a new position with all risk components"""
        self.trailing_stops.add_position(symbol, entry_price, side)
        self.take_profit.add_position(symbol)
        logger.info(f"Position registered: {symbol} @ {entry_price:.2f} ({side})")
    
    def get_stop_price(self, symbol: str) -> Optional[float]:
        """Get current stop price for a symbol"""
        stop = self.trailing_stops.get_stop(symbol)
        if stop:
            return stop.current_stop
        return None
    
    def close_position(self, symbol: str, pnl: float) -> None:
        """Close a position and record P&L"""
        self.trailing_stops.remove_position(symbol)
        self.take_profit.remove_position(symbol)
        self.daily_tracker.record_trade(pnl)
        logger.info(f"Position closed: {symbol}, P&L: ${pnl:.2f}")
    
    def update_prices(self, prices: Dict[str, float], sides: Dict[str, str] = None) -> List[str]:
        """
        Update prices for all tracked positions
        
        Returns:
            List of symbols that hit stop loss
        """
        if sides is None:
            sides = {}
        
        stopped_out = []
        
        for symbol, price in prices.items():
            side = sides.get(symbol, "LONG")
            triggered, _ = self.trailing_stops.update_price(symbol, price, side)
            
            if triggered:
                stopped_out.append(symbol)
        
        return stopped_out
    
    def check_take_profits(
        self, 
        positions: Dict[str, Dict]
    ) -> List[Tuple[str, float, float]]:
        """
        Check take-profit levels for all positions
        
        Args:
            positions: Dict of {symbol: {entry_price, current_price, size, side}}
        
        Returns:
            List of (symbol, size_to_close, threshold) for positions hitting TP
        """
        take_profits = []
        
        for symbol, pos in positions.items():
            result = self.take_profit.check_take_profit(
                symbol=symbol,
                entry_price=pos["entry_price"],
                current_price=pos["current_price"],
                position_size=pos["size"],
                side=pos.get("side", "LONG")
            )
            
            if result:
                size_to_close, threshold = result
                take_profits.append((symbol, size_to_close, threshold))
        
        return take_profits
    
    def update_capital(self, current_capital: float, unrealized_pnl: float = 0.0) -> None:
        """Update capital across all components"""
        self.daily_tracker.update_capital(current_capital)
        self.daily_tracker.update_unrealized(unrealized_pnl)
        
        within_limits, _ = self.drawdown.update(current_capital)
        
        if not within_limits and not self.is_halted:
            self.halt_trading("Maximum drawdown exceeded")
    
    def halt_trading(self, reason: str) -> None:
        """Halt all trading"""
        self.is_halted = True
        self.halt_reason = reason
        logger.critical(f"TRADING HALTED: {reason}")
    
    def resume_trading(self) -> None:
        """Resume trading (manual override)"""
        self.is_halted = False
        self.halt_reason = ""
        logger.info("Trading resumed")
    
    def reset_drawdown(self, new_capital: float = None) -> None:
        """
        Reset drawdown tracking - use after fixing erroneous P&L
        Also resumes trading if it was halted
        """
        self.drawdown.reset(new_capital)
        self.daily_tracker.reset()
        if self.is_halted:
            self.is_halted = False
            self.halt_reason = ""
            logger.info("Trading resumed after drawdown reset")
    
    def assess_risk(self) -> RiskMetrics:
        """Get comprehensive risk assessment"""
        # Check daily limits
        can_trade_daily, daily_reason = self.daily_tracker.can_trade()
        
        # Check drawdown
        within_drawdown, drawdown_pct = self.drawdown.update(self.daily_tracker.current_capital)
        
        # Determine risk level
        if self.is_halted or not within_drawdown:
            risk_level = RiskLevel.HALT
            can_trade = False
            reason = self.halt_reason or "Max drawdown exceeded"
        elif not can_trade_daily:
            risk_level = RiskLevel.HALT
            can_trade = False
            reason = daily_reason
        elif drawdown_pct >= 20 or self.daily_tracker.daily_pnl_percent <= -12:
            risk_level = RiskLevel.CRITICAL
            can_trade = True
            reason = "Critical risk level - reduce position sizes"
        elif drawdown_pct >= 15 or self.daily_tracker.daily_pnl_percent <= -10:
            risk_level = RiskLevel.HIGH
            can_trade = True
            reason = "High risk level"
        elif drawdown_pct >= 10 or self.daily_tracker.daily_pnl_percent <= -7:
            risk_level = RiskLevel.ELEVATED
            can_trade = True
            reason = "Elevated risk level"
        else:
            risk_level = RiskLevel.NORMAL
            can_trade = True
            reason = "Normal operations"
        
        # Find positions at risk (near stop loss)
        positions_at_risk = [
            symbol for symbol, stop in self.trailing_stops.stops.items()
            if stop.triggered
        ]
        
        return RiskMetrics(
            daily_pnl=self.daily_tracker.total_daily_pnl,
            daily_pnl_percent=self.daily_tracker.daily_pnl_percent,
            total_drawdown=self.drawdown.current_drawdown,
            total_drawdown_percent=drawdown_pct,
            open_risk=sum(
                abs(s.current_stop - s.highest_price) 
                for s in self.trailing_stops.stops.values()
            ),
            risk_level=risk_level,
            positions_at_risk=positions_at_risk,
            can_trade=can_trade,
            reason=reason
        )
    
    def get_position_risk_multiplier(self) -> float:
        """
        Get position size multiplier based on current risk level
        
        Returns:
            Multiplier between 0.0 and 1.0
        """
        metrics = self.assess_risk()
        
        multipliers = {
            RiskLevel.NORMAL: 1.0,
            RiskLevel.ELEVATED: 0.75,
            RiskLevel.HIGH: 0.5,
            RiskLevel.CRITICAL: 0.25,
            RiskLevel.HALT: 0.0
        }
        
        return multipliers.get(metrics.risk_level, 1.0)
    
    def get_full_report(self) -> Dict:
        """Get comprehensive risk report"""
        metrics = self.assess_risk()
        
        return {
            "risk_assessment": {
                "level": metrics.risk_level.value,
                "can_trade": metrics.can_trade,
                "reason": metrics.reason,
                "position_multiplier": self.get_position_risk_multiplier()
            },
            "daily_stats": self.daily_tracker.get_daily_stats(),
            "drawdown": self.drawdown.get_stats(),
            "active_stops": {
                symbol: {
                    "entry": stop.entry_price,
                    "current_stop": stop.current_stop,
                    "highest": stop.highest_price,
                    "triggered": stop.triggered
                }
                for symbol, stop in self.trailing_stops.stops.items()
            },
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason
        }
