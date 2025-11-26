"""
Copy Trade Data Fetcher

Fetches top trader positions and signals from various sources.
Supports multiple data sources with fallback.

Sources:
- Manual input (for testing)
- Simulated top traders (for paper trading)
- Future: Bitget, Bybit, etc. APIs
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod
import asyncio
import aiohttp
import logging
import random

logger = logging.getLogger(__name__)


@dataclass
class TraderPosition:
    """A position held by a top trader"""
    trader_id: str
    trader_name: str
    symbol: str
    direction: str  # LONG, SHORT, NEUTRAL
    entry_price: float
    current_size: float
    size_percent: float  # % of their portfolio
    entry_time: datetime
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    
    # Trader stats
    trader_win_rate: float = 0.5
    trader_pnl_30d: float = 0.0
    trader_followers: int = 0


@dataclass
class TraderProfile:
    """Profile of a top trader"""
    trader_id: str
    name: str
    platform: str
    
    # Performance metrics
    win_rate: float = 0.5
    profit_factor: float = 1.0
    sharpe_ratio: float = 0.0
    total_pnl_30d: float = 0.0
    total_pnl_90d: float = 0.0
    avg_trade_duration_hours: float = 24.0
    
    # Social metrics
    followers: int = 0
    copiers: int = 0
    
    # Current state
    open_positions: int = 0
    total_trades: int = 0
    
    # Risk metrics
    max_drawdown: float = 0.0
    avg_position_size_pct: float = 0.0
    
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DataSource(ABC):
    """Abstract base class for copy trade data sources"""
    
    @abstractmethod
    async def get_top_traders(self, limit: int = 20) -> List[TraderProfile]:
        """Get list of top traders"""
        pass
    
    @abstractmethod
    async def get_trader_positions(self, trader_id: str) -> List[TraderPosition]:
        """Get current positions for a trader"""
        pass
    
    @abstractmethod
    async def get_all_positions(self, symbol: str = None) -> List[TraderPosition]:
        """Get all positions across all tracked traders"""
        pass


class SimulatedDataSource(DataSource):
    """
    Simulated copy trade data for testing/paper trading
    
    Generates realistic-looking trader data with positions
    that follow market trends with some noise.
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.traders: Dict[str, TraderProfile] = {}
        self.positions: Dict[str, List[TraderPosition]] = {}
        
        # Initialize simulated traders
        self._initialize_traders()
    
    def _initialize_traders(self):
        """Create simulated top traders"""
        trader_configs = [
            ("sim_trader_1", "CryptoAlpha", 0.72, 2.4, 1.9),
            ("sim_trader_2", "BTCMaximalist", 0.68, 1.8, 1.5),
            ("sim_trader_3", "AltcoinHunter", 0.61, 1.5, 1.2),
            ("sim_trader_4", "SwingMaster", 0.65, 1.7, 1.4),
            ("sim_trader_5", "ScalpKing", 0.58, 1.3, 1.1),
            ("sim_trader_6", "TrendFollower", 0.70, 2.0, 1.6),
            ("sim_trader_7", "MomentumPro", 0.63, 1.4, 1.3),
            ("sim_trader_8", "ValueInvestor", 0.75, 2.5, 2.0),
        ]
        
        for tid, name, win_rate, pf, sharpe in trader_configs:
            self.traders[tid] = TraderProfile(
                trader_id=tid,
                name=name,
                platform="simulated",
                win_rate=win_rate,
                profit_factor=pf,
                sharpe_ratio=sharpe,
                total_pnl_30d=random.uniform(500, 5000),
                followers=random.randint(100, 10000),
                copiers=random.randint(10, 500),
                total_trades=random.randint(50, 500)
            )
            self.positions[tid] = []
    
    async def get_top_traders(self, limit: int = 20) -> List[TraderProfile]:
        """Get simulated top traders"""
        traders = list(self.traders.values())
        # Sort by win rate * sharpe (composite score)
        traders.sort(key=lambda t: t.win_rate * t.sharpe_ratio, reverse=True)
        return traders[:limit]
    
    async def get_trader_positions(self, trader_id: str) -> List[TraderPosition]:
        """Get positions for a simulated trader"""
        return self.positions.get(trader_id, [])
    
    async def get_all_positions(self, symbol: str = None) -> List[TraderPosition]:
        """Get all positions"""
        all_pos = []
        for positions in self.positions.values():
            for pos in positions:
                if symbol is None or pos.symbol == symbol:
                    all_pos.append(pos)
        return all_pos
    
    def simulate_market_update(
        self,
        prices: Dict[str, float],
        market_bias: str = "NEUTRAL"
    ):
        """
        Update simulated positions based on market
        
        Args:
            prices: Current prices for each symbol
            market_bias: Overall market direction (LONG, SHORT, NEUTRAL)
        """
        # Clear old positions
        for tid in self.positions:
            self.positions[tid] = []
        
        # Generate new positions based on market bias
        for trader_id, trader in self.traders.items():
            # Each trader has a chance to have positions
            for symbol in self.symbols:
                if random.random() < 0.4:  # 40% chance of position per symbol
                    # Direction influenced by market bias and trader style
                    if market_bias == "LONG":
                        direction = "LONG" if random.random() < 0.7 else "SHORT"
                    elif market_bias == "SHORT":
                        direction = "SHORT" if random.random() < 0.7 else "LONG"
                    else:
                        direction = random.choice(["LONG", "SHORT"])
                    
                    # Add some contrarians
                    if "Value" in trader.name and random.random() < 0.3:
                        direction = "SHORT" if direction == "LONG" else "LONG"
                    
                    price = prices.get(symbol, 100)
                    
                    position = TraderPosition(
                        trader_id=trader_id,
                        trader_name=trader.name,
                        symbol=symbol,
                        direction=direction,
                        entry_price=price * random.uniform(0.95, 1.05),
                        current_size=random.uniform(0.01, 1.0),
                        size_percent=random.uniform(5, 25),
                        entry_time=datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 48)),
                        trader_win_rate=trader.win_rate,
                        trader_pnl_30d=trader.total_pnl_30d,
                        trader_followers=trader.followers
                    )
                    
                    self.positions[trader_id].append(position)


class ManualDataSource(DataSource):
    """
    Manual data source for testing
    
    Allows manually adding trader positions for testing quantum scoring
    """
    
    def __init__(self):
        self.traders: Dict[str, TraderProfile] = {}
        self.positions: List[TraderPosition] = []
    
    def add_trader(self, profile: TraderProfile):
        """Add a trader profile"""
        self.traders[profile.trader_id] = profile
    
    def add_position(self, position: TraderPosition):
        """Add a position"""
        self.positions.append(position)
    
    def clear_positions(self):
        """Clear all positions"""
        self.positions = []
    
    async def get_top_traders(self, limit: int = 20) -> List[TraderProfile]:
        return list(self.traders.values())[:limit]
    
    async def get_trader_positions(self, trader_id: str) -> List[TraderPosition]:
        return [p for p in self.positions if p.trader_id == trader_id]
    
    async def get_all_positions(self, symbol: str = None) -> List[TraderPosition]:
        if symbol:
            return [p for p in self.positions if p.symbol == symbol]
        return self.positions


class CopyTradeAggregator:
    """
    Aggregates copy trade data from multiple sources
    and feeds into the Quantum Trader Scoring system
    """
    
    def __init__(
        self,
        symbols: List[str],
        use_simulation: bool = True
    ):
        self.symbols = symbols
        self.use_simulation = use_simulation
        
        # Data sources
        self.sources: List[DataSource] = []
        
        if use_simulation:
            self.sim_source = SimulatedDataSource(symbols)
            self.sources.append(self.sim_source)
        
        # Aggregated data
        self.trader_profiles: Dict[str, TraderProfile] = {}
        self.all_positions: List[TraderPosition] = []
        
        # Update tracking
        self.last_update: Optional[datetime] = None
        self.update_interval_seconds: int = 60  # Update every minute
        
        logger.info(f"CopyTradeAggregator initialized (simulation={use_simulation})")
    
    def add_source(self, source: DataSource):
        """Add a data source"""
        self.sources.append(source)
    
    async def update(self, prices: Dict[str, float] = None) -> Dict:
        """
        Update data from all sources
        
        Args:
            prices: Current prices (needed for simulation)
        
        Returns:
            Update summary
        """
        # Update simulation if using it
        if self.use_simulation and prices:
            # Determine market bias from price changes
            # (In real implementation, this would come from actual market data)
            bias = "NEUTRAL"
            if prices:
                # Simple heuristic: if BTC up, bias is LONG
                btc_price = prices.get("BTC-USD", 0)
                if btc_price > 0:
                    bias = random.choice(["LONG", "LONG", "NEUTRAL", "SHORT"])
            
            self.sim_source.simulate_market_update(prices, bias)
        
        # Fetch from all sources
        all_traders = []
        all_positions = []
        
        for source in self.sources:
            try:
                traders = await source.get_top_traders(20)
                all_traders.extend(traders)
                
                positions = await source.get_all_positions()
                all_positions.extend(positions)
            except Exception as e:
                logger.warning(f"Error fetching from source: {e}")
        
        # Update aggregated data
        self.trader_profiles = {t.trader_id: t for t in all_traders}
        self.all_positions = all_positions
        self.last_update = datetime.now(timezone.utc)
        
        logger.info(f"Copy trade update: {len(all_traders)} traders, {len(all_positions)} positions")
        
        return {
            "traders_count": len(self.trader_profiles),
            "positions_count": len(self.all_positions),
            "positions": self.all_positions,  # Include actual positions!
            "symbols_with_positions": list(set(p.symbol for p in all_positions)),
            "timestamp": self.last_update.isoformat()
        }
    
    def get_positions_for_symbol(self, symbol: str) -> List[TraderPosition]:
        """Get all positions for a specific symbol"""
        return [p for p in self.all_positions if p.symbol == symbol]
    
    def get_signal_summary(self, symbol: str) -> Dict:
        """Get aggregated signal summary for a symbol"""
        positions = self.get_positions_for_symbol(symbol)
        
        if not positions:
            return {
                "symbol": symbol,
                "long_count": 0,
                "short_count": 0,
                "total_traders": 0,
                "consensus": "NEUTRAL",
                "avg_win_rate": 0
            }
        
        long_pos = [p for p in positions if p.direction == "LONG"]
        short_pos = [p for p in positions if p.direction == "SHORT"]
        
        # Weight by win rate
        long_weight = sum(p.trader_win_rate for p in long_pos)
        short_weight = sum(p.trader_win_rate for p in short_pos)
        
        if long_weight > short_weight * 1.2:
            consensus = "LONG"
        elif short_weight > long_weight * 1.2:
            consensus = "SHORT"
        else:
            consensus = "NEUTRAL"
        
        return {
            "symbol": symbol,
            "long_count": len(long_pos),
            "short_count": len(short_pos),
            "total_traders": len(positions),
            "consensus": consensus,
            "long_weight": long_weight,
            "short_weight": short_weight,
            "avg_win_rate": sum(p.trader_win_rate for p in positions) / len(positions),
            "top_long_traders": [
                {"name": p.trader_name, "win_rate": p.trader_win_rate}
                for p in sorted(long_pos, key=lambda x: x.trader_win_rate, reverse=True)[:3]
            ],
            "top_short_traders": [
                {"name": p.trader_name, "win_rate": p.trader_win_rate}
                for p in sorted(short_pos, key=lambda x: x.trader_win_rate, reverse=True)[:3]
            ]
        }
    
    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get signal summaries for all symbols"""
        return {symbol: self.get_signal_summary(symbol) for symbol in self.symbols}
    
    async def should_update(self) -> bool:
        """Check if it's time to update"""
        if not self.last_update:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self.last_update).total_seconds()
        return elapsed >= self.update_interval_seconds
