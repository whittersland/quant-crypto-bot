"""
Portfolio Manager
Tracks overall portfolio state, capital allocation, and performance
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot"""
    timestamp: datetime
    total_capital: float
    available_capital: float
    allocated_capital: float
    unrealized_pnl: float
    realized_pnl_today: float
    open_positions: int
    drawdown_percent: float


@dataclass 
class PerformanceMetrics:
    """Portfolio performance metrics"""
    total_return: float
    total_return_percent: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    total_trades: int
    trading_days: int


class PortfolioManager:
    """
    Portfolio Manager
    
    Tracks:
    - Total capital and allocation
    - Performance metrics
    - Historical snapshots
    - Capital allocation per symbol
    """
    
    def __init__(
        self,
        initial_capital: float,
        leverage: float = 2.0,
        state_file: str = "portfolio_state.json"
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.state_file = state_file
        
        # Current state
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        self.allocated_capital = 0.0
        
        # Tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.peak_capital = initial_capital
        
        # Allocation per symbol
        self.allocations: Dict[str, float] = {}
        
        # History
        self.snapshots: List[PortfolioSnapshot] = []
        self.daily_pnl: List[Dict] = []
        self.trade_results: List[Dict] = []
        
        # Load saved state
        self._load_state()
    
    def _load_state(self) -> None:
        """Load saved portfolio state"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                self.current_capital = data.get("current_capital", self.initial_capital)
                self.available_capital = data.get("available_capital", self.initial_capital)
                self.allocated_capital = data.get("allocated_capital", 0.0)
                self.realized_pnl = data.get("realized_pnl", 0.0)
                self.peak_capital = data.get("peak_capital", self.initial_capital)
                self.allocations = data.get("allocations", {})
                self.daily_pnl = data.get("daily_pnl", [])[-30:]  # Keep last 30 days
                
                logger.info(f"Loaded portfolio state: capital=${self.current_capital:.2f}")
        except Exception as e:
            logger.warning(f"Could not load portfolio state: {e}")
    
    def _save_state(self) -> None:
        """Save portfolio state"""
        try:
            data = {
                "current_capital": self.current_capital,
                "available_capital": self.available_capital,
                "allocated_capital": self.allocated_capital,
                "realized_pnl": self.realized_pnl,
                "peak_capital": self.peak_capital,
                "allocations": self.allocations,
                "daily_pnl": self.daily_pnl[-30:],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save portfolio state: {e}")
    
    def allocate(self, symbol: str, amount: float) -> bool:
        """
        Allocate capital to a position
        
        Returns:
            True if allocation successful
        """
        if amount > self.available_capital:
            logger.warning(f"Insufficient capital for allocation: need ${amount:.2f}, have ${self.available_capital:.2f}")
            return False
        
        self.allocations[symbol] = self.allocations.get(symbol, 0) + amount
        self.allocated_capital += amount
        self.available_capital -= amount
        
        self._save_state()
        logger.debug(f"Allocated ${amount:.2f} to {symbol}")
        
        return True
    
    def deallocate(self, symbol: str, amount: Optional[float] = None) -> float:
        """
        Deallocate capital from a position
        
        Args:
            symbol: Symbol to deallocate from
            amount: Amount to deallocate (None = all)
        
        Returns:
            Amount deallocated
        """
        if symbol not in self.allocations:
            return 0.0
        
        current = self.allocations[symbol]
        to_deallocate = amount if amount is not None else current
        to_deallocate = min(to_deallocate, current)
        
        self.allocations[symbol] -= to_deallocate
        if self.allocations[symbol] <= 0:
            del self.allocations[symbol]
        
        self.allocated_capital -= to_deallocate
        self.available_capital += to_deallocate
        
        self._save_state()
        logger.debug(f"Deallocated ${to_deallocate:.2f} from {symbol}")
        
        return to_deallocate
    
    def record_trade_result(
        self,
        symbol: str,
        pnl: float,
        trade_type: str = "trade"
    ) -> None:
        """Record a trade result"""
        self.realized_pnl += pnl
        self.current_capital += pnl
        self.available_capital += pnl
        
        # Update peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Record
        self.trade_results.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "pnl": pnl,
            "type": trade_type,
            "capital_after": self.current_capital
        })
        
        self._save_state()
        logger.info(f"Trade result recorded: {symbol} P&L=${pnl:.2f}, capital=${self.current_capital:.2f}")
    
    def update_unrealized_pnl(self, unrealized: float) -> None:
        """Update unrealized P&L"""
        self.unrealized_pnl = unrealized
    
    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a point-in-time snapshot"""
        drawdown = 0.0
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_capital=self.current_capital,
            available_capital=self.available_capital,
            allocated_capital=self.allocated_capital,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl_today=self._get_today_pnl(),
            open_positions=len(self.allocations),
            drawdown_percent=drawdown
        )
        
        self.snapshots.append(snapshot)
        
        # Keep last 1000 snapshots
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-1000:]
        
        return snapshot
    
    def _get_today_pnl(self) -> float:
        """Get realized P&L for today"""
        today = datetime.now(timezone.utc).date()
        
        today_pnl = sum(
            t["pnl"] for t in self.trade_results
            if datetime.fromisoformat(t["timestamp"]).date() == today
        )
        
        return today_pnl
    
    def record_daily_close(self) -> None:
        """Record end-of-day summary"""
        today = datetime.now(timezone.utc).date().isoformat()
        
        daily_record = {
            "date": today,
            "capital": self.current_capital,
            "realized_pnl": self._get_today_pnl(),
            "trades": len([
                t for t in self.trade_results
                if t["timestamp"].startswith(today)
            ]),
            "drawdown": (self.peak_capital - self.current_capital) / self.peak_capital * 100 if self.peak_capital > 0 else 0
        }
        
        # Avoid duplicate entries
        if not self.daily_pnl or self.daily_pnl[-1]["date"] != today:
            self.daily_pnl.append(daily_record)
        else:
            self.daily_pnl[-1] = daily_record
        
        self._save_state()
    
    def calculate_performance(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trade_results:
            return PerformanceMetrics(
                total_return=0,
                total_return_percent=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                avg_trade_pnl=0,
                total_trades=0,
                trading_days=0
            )
        
        # Basic metrics
        total_return = self.current_capital - self.initial_capital
        total_return_percent = (total_return / self.initial_capital) * 100
        
        # Win rate
        wins = [t for t in self.trade_results if t["pnl"] >= 0]
        losses = [t for t in self.trade_results if t["pnl"] < 0]
        win_rate = len(wins) / len(self.trade_results) * 100 if self.trade_results else 0
        
        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade
        avg_trade = sum(t["pnl"] for t in self.trade_results) / len(self.trade_results)
        
        # Max drawdown from snapshots
        max_dd = 0.0
        peak = self.initial_capital
        for snap in self.snapshots:
            if snap.total_capital > peak:
                peak = snap.total_capital
            dd = (peak - snap.total_capital) / peak * 100
            max_dd = max(max_dd, dd)
        
        # Trading days
        unique_days = set(
            datetime.fromisoformat(t["timestamp"]).date()
            for t in self.trade_results
        )
        
        # Sharpe ratio (simplified - would need daily returns for proper calc)
        import numpy as np
        daily_returns = []
        for record in self.daily_pnl:
            if record.get("realized_pnl"):
                daily_returns.append(record["realized_pnl"])
        
        if len(daily_returns) > 1:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            total_return_percent=total_return_percent,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade,
            total_trades=len(self.trade_results),
            trading_days=len(unique_days)
        )
    
    def get_allocation_summary(self) -> Dict:
        """Get current allocation summary"""
        return {
            "total_capital": self.current_capital,
            "available_capital": self.available_capital,
            "allocated_capital": self.allocated_capital,
            "leverage": self.leverage,
            "effective_buying_power": self.available_capital * self.leverage,
            "utilization_percent": (self.allocated_capital / self.current_capital * 100) if self.current_capital > 0 else 0,
            "allocations": self.allocations.copy(),
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "drawdown_from_peak": (self.peak_capital - self.current_capital) / self.peak_capital * 100 if self.peak_capital > 0 else 0
        }
    
    def get_status(self) -> Dict:
        """Get comprehensive portfolio status"""
        performance = self.calculate_performance()
        
        return {
            "capital": {
                "initial": self.initial_capital,
                "current": self.current_capital,
                "available": self.available_capital,
                "allocated": self.allocated_capital,
                "peak": self.peak_capital
            },
            "pnl": {
                "realized": self.realized_pnl,
                "unrealized": self.unrealized_pnl,
                "total": self.realized_pnl + self.unrealized_pnl,
                "today": self._get_today_pnl()
            },
            "performance": {
                "total_return_percent": performance.total_return_percent,
                "win_rate": performance.win_rate,
                "profit_factor": performance.profit_factor,
                "sharpe_ratio": performance.sharpe_ratio,
                "max_drawdown": performance.max_drawdown,
                "total_trades": performance.total_trades
            },
            "positions": {
                "open": len(self.allocations),
                "symbols": list(self.allocations.keys())
            }
        }
    
    def reset(self, new_capital: Optional[float] = None) -> None:
        """Reset portfolio to initial state"""
        self.current_capital = new_capital or self.initial_capital
        self.available_capital = self.current_capital
        self.allocated_capital = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.peak_capital = self.current_capital
        self.allocations = {}
        self.snapshots = []
        self.trade_results = []
        
        self._save_state()
        logger.info(f"Portfolio reset to ${self.current_capital:.2f}")
