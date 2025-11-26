"""
Binance Futures Leaderboard Copy Trading
Fetches top traders from Binance leaderboard and their positions
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class PeriodType(Enum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    ALL = "ALL"


class SortType(Enum):
    ROI = "ROI"
    PNL = "PNL"


@dataclass
class BinanceTrader:
    """Represents a trader from Binance leaderboard"""
    encrypted_uid: str
    nickname: str
    rank: int
    pnl: float
    roi: float
    follower_count: int
    position_shared: bool
    win_rate: Optional[float] = None
    twitter_url: Optional[str] = None
    
    @property
    def trader_id(self) -> str:
        return self.encrypted_uid


@dataclass
class BinancePosition:
    """Represents a position from a Binance leaderboard trader"""
    symbol: str
    entry_price: float
    mark_price: float
    pnl: float
    roe: float  # Return on equity (ROI for position)
    amount: float
    leverage: int
    side: str  # "LONG" or "SHORT"
    update_time: datetime
    trader_id: str
    
    @classmethod
    def from_api(cls, data: Dict, trader_id: str) -> "BinancePosition":
        """Create from Binance API response"""
        # Determine side from amount sign
        amount = float(data.get("amount", 0))
        side = "LONG" if amount > 0 else "SHORT"
        
        return cls(
            symbol=data.get("symbol", ""),
            entry_price=float(data.get("entryPrice", 0)),
            mark_price=float(data.get("markPrice", 0)),
            pnl=float(data.get("pnl", 0)),
            roe=float(data.get("roe", 0)),
            amount=abs(amount),
            leverage=int(data.get("leverage", 1)),
            side=side,
            update_time=datetime.fromtimestamp(
                data.get("updateTimeStamp", 0) / 1000, 
                tz=timezone.utc
            ),
            trader_id=trader_id
        )


class BinanceLeaderboardSource:
    """
    Fetches copy trading data from Binance Futures Leaderboard
    
    This uses Binance's public leaderboard API to get:
    1. Top traders ranked by ROI or PNL
    2. Their open positions (if shared publicly)
    
    Note: This data comes from Binance, but we can use it to inform
    trades on Coinbase since the market movements are correlated.
    """
    
    BASE_URL = "https://www.binance.com/bapi/futures"
    
    def __init__(
        self,
        period: PeriodType = PeriodType.WEEKLY,
        sort_by: SortType = SortType.ROI,
        min_roi: float = 10.0,  # Minimum ROI % to consider
        min_followers: int = 100,
        max_traders: int = 20
    ):
        self.period = period
        self.sort_by = sort_by
        self.min_roi = min_roi
        self.min_followers = min_followers
        self.max_traders = max_traders
        
        self.traders: Dict[str, BinanceTrader] = {}
        self.positions: List[BinancePosition] = []
        self.last_update: Optional[datetime] = None
        
        logger.info(
            f"BinanceLeaderboard initialized: period={period.value}, "
            f"sort={sort_by.value}, min_roi={min_roi}%"
        )
    
    async def fetch_leaderboard(self) -> List[BinanceTrader]:
        """Fetch top traders from leaderboard"""
        url = f"{self.BASE_URL}/v2/public/future/leaderboard/searchLeaderboard"
        
        payload = {
            "isShared": True,  # Only traders sharing positions
            "limit": self.max_traders * 2,  # Fetch extra to filter
            "periodType": self.period.value,
            "sortType": self.sort_by.value,
            "tradeType": "PERPETUAL"
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=10) as resp:
                    if resp.status != 200:
                        logger.warning(f"Binance leaderboard API returned {resp.status}")
                        return []
                    
                    data = await resp.json()
                    
                    if not data.get("success"):
                        logger.warning(f"Binance API error: {data.get('message')}")
                        return []
                    
                    traders = []
                    for i, item in enumerate(data.get("data", [])):
                        trader = BinanceTrader(
                            encrypted_uid=item.get("encryptedUid", ""),
                            nickname=item.get("nickName", f"Trader_{i}"),
                            rank=i + 1,
                            pnl=float(item.get("pnl", 0)),
                            roi=float(item.get("roi", 0)) * 100,  # Convert to percentage
                            follower_count=int(item.get("followerCount", 0)),
                            position_shared=item.get("positionShared", False),
                            twitter_url=item.get("twitterUrl")
                        )
                        
                        # Filter by criteria
                        if (trader.roi >= self.min_roi and 
                            trader.follower_count >= self.min_followers and
                            trader.position_shared):
                            traders.append(trader)
                            
                            if len(traders) >= self.max_traders:
                                break
                    
                    logger.info(f"Fetched {len(traders)} qualified traders from Binance leaderboard")
                    return traders
                    
        except asyncio.TimeoutError:
            logger.warning("Binance leaderboard request timed out")
            return []
        except Exception as e:
            logger.warning(f"Error fetching Binance leaderboard: {e}")
            return []
    
    async def fetch_trader_positions(self, trader: BinanceTrader) -> List[BinancePosition]:
        """Fetch open positions for a specific trader"""
        url = f"{self.BASE_URL}/v1/public/future/leaderboard/getOtherPosition"
        
        payload = {
            "encryptedUid": trader.encrypted_uid,
            "tradeType": "PERPETUAL"
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=10) as resp:
                    if resp.status != 200:
                        return []
                    
                    data = await resp.json()
                    
                    if not data.get("success"):
                        return []
                    
                    positions = []
                    other_data = data.get("data", {}).get("otherPositionRetList", [])
                    
                    for pos_data in other_data:
                        try:
                            position = BinancePosition.from_api(pos_data, trader.encrypted_uid)
                            positions.append(position)
                        except Exception as e:
                            logger.debug(f"Error parsing position: {e}")
                    
                    return positions
                    
        except Exception as e:
            logger.debug(f"Error fetching positions for {trader.nickname}: {e}")
            return []
    
    async def update(self) -> Dict[str, Any]:
        """Update all trader data and positions"""
        # Fetch leaderboard
        traders = await self.fetch_leaderboard()
        
        if not traders:
            logger.warning("No traders fetched from Binance leaderboard")
            return {"traders": 0, "positions": 0}
        
        self.traders = {t.encrypted_uid: t for t in traders}
        
        # Fetch positions for each trader
        all_positions = []
        for trader in traders[:10]:  # Limit to top 10 to avoid rate limits
            positions = await self.fetch_trader_positions(trader)
            all_positions.extend(positions)
            await asyncio.sleep(0.2)  # Rate limit delay
        
        self.positions = all_positions
        self.last_update = datetime.now(timezone.utc)
        
        logger.info(f"Binance leaderboard update: {len(self.traders)} traders, {len(self.positions)} positions")
        
        return {
            "traders": len(self.traders),
            "positions": len(self.positions),
            "timestamp": self.last_update.isoformat()
        }
    
    def get_consensus(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading consensus for a symbol from top traders
        
        Returns:
            Dict with long_count, short_count, consensus direction, confidence
        """
        # Map Binance symbols to our format
        # Binance uses BTCUSDT, we use BTC-USD
        binance_symbol = symbol.replace("-USD", "USDT")
        
        relevant_positions = [
            p for p in self.positions 
            if p.symbol == binance_symbol
        ]
        
        if not relevant_positions:
            return {
                "symbol": symbol,
                "long_count": 0,
                "short_count": 0,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "positions": []
            }
        
        long_positions = [p for p in relevant_positions if p.side == "LONG"]
        short_positions = [p for p in relevant_positions if p.side == "SHORT"]
        
        long_count = len(long_positions)
        short_count = len(short_positions)
        total = long_count + short_count
        
        # Calculate weighted confidence based on trader performance
        long_weight = sum(
            self.traders.get(p.trader_id, BinanceTrader("", "", 0, 0, 0, 0, False)).roi
            for p in long_positions
        )
        short_weight = sum(
            self.traders.get(p.trader_id, BinanceTrader("", "", 0, 0, 0, 0, False)).roi
            for p in short_positions
        )
        
        if long_weight > short_weight:
            direction = "LONG"
            confidence = min(0.9, 0.5 + (long_count / total) * 0.4)
        elif short_weight > long_weight:
            direction = "SHORT"
            confidence = min(0.9, 0.5 + (short_count / total) * 0.4)
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        
        return {
            "symbol": symbol,
            "long_count": long_count,
            "short_count": short_count,
            "direction": direction,
            "confidence": confidence,
            "avg_leverage": sum(p.leverage for p in relevant_positions) / total if total > 0 else 0,
            "positions": [
                {
                    "trader": self.traders.get(p.trader_id, BinanceTrader("", "Unknown", 0, 0, 0, 0, False)).nickname,
                    "side": p.side,
                    "leverage": p.leverage,
                    "roe": p.roe * 100  # Convert to percentage
                }
                for p in relevant_positions[:5]  # Top 5 positions
            ]
        }


class CopyTradeAggregatorWithBinance:
    """
    Enhanced copy trade aggregator that uses Binance leaderboard data
    """
    
    def __init__(
        self,
        symbols: List[str],
        min_roi: float = 10.0,
        min_followers: int = 100,
        period: PeriodType = PeriodType.WEEKLY
    ):
        self.symbols = symbols
        self.binance = BinanceLeaderboardSource(
            period=period,
            min_roi=min_roi,
            min_followers=min_followers
        )
        self.last_update: Optional[datetime] = None
        
        logger.info(f"CopyTradeAggregator with Binance initialized for {symbols}")
    
    async def update(self, prices: Dict[str, float] = None) -> Dict[str, Any]:
        """Update copy trade data from Binance"""
        result = await self.binance.update()
        self.last_update = datetime.now(timezone.utc)
        
        # Get consensus for each symbol
        consensus = {}
        for symbol in self.symbols:
            consensus[symbol] = self.binance.get_consensus(symbol)
        
        result["consensus"] = consensus
        result["positions"] = self.binance.positions
        
        return result
    
    def get_signal_summary(self, symbol: str) -> Dict[str, Any]:
        """Get aggregated signal for a symbol"""
        return self.binance.get_consensus(symbol)


# For testing
async def test_binance_leaderboard():
    """Test the Binance leaderboard integration"""
    source = BinanceLeaderboardSource(
        period=PeriodType.WEEKLY,
        min_roi=5.0,
        min_followers=50
    )
    
    print("Fetching Binance leaderboard...")
    result = await source.update()
    print(f"Result: {result}")
    
    if source.traders:
        print(f"\nTop traders:")
        for uid, trader in list(source.traders.items())[:5]:
            print(f"  {trader.nickname}: ROI={trader.roi:.1f}%, followers={trader.follower_count}")
    
    if source.positions:
        print(f"\nSample positions:")
        for pos in source.positions[:5]:
            print(f"  {pos.symbol} {pos.side} @ {pos.entry_price:.2f}, leverage={pos.leverage}x")
    
    # Test consensus
    for symbol in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        consensus = source.get_consensus(symbol)
        print(f"\n{symbol} consensus: {consensus['direction']} (conf={consensus['confidence']:.2f})")
        print(f"  Long: {consensus['long_count']}, Short: {consensus['short_count']}")


if __name__ == "__main__":
    asyncio.run(test_binance_leaderboard())
