"""
MEXC Copy Trading Integration

Fetches top traders from MEXC Futures Copy Trading leaderboard.
Public API - no authentication required!

Endpoints (from browser DevTools):
- /api/v1/copytrading/leaderboard?category=ROI&page=1&pageSize=50
- /api/v1/copytrading/trader/detail?traderId=XXX
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class RankCategory(Enum):
    """MEXC leaderboard ranking categories"""
    ROI = "ROI"              # Top Return
    PNL = "PNL"              # Top Profit
    WIN_RATE = "WIN_RATE"    # Stable Win Rate
    LOW_DRAWDOWN = "LOW_DRAWDOWN"  # Lowest Drawdown


@dataclass
class MEXCTrader:
    """Represents a trader from MEXC leaderboard"""
    trader_id: str
    nickname: str
    avatar: str = ""
    roi: float = 0.0           # Return on investment %
    pnl: float = 0.0           # Total profit/loss
    win_rate: float = 0.0      # Win rate %
    drawdown: float = 0.0      # Max drawdown %
    follower_count: int = 0
    trade_count: int = 0
    rank: int = 0
    category: RankCategory = RankCategory.ROI


@dataclass 
class MEXCPosition:
    """Represents an open position from a MEXC trader"""
    symbol: str              # e.g., "BTC_USDT"
    side: str                # "LONG" or "SHORT"
    size: float              # Position size
    entry_price: float       # Entry price
    mark_price: float        # Current mark price
    leverage: int            # Leverage used
    pnl: float               # Unrealized PNL
    pnl_rate: float          # PNL percentage
    trader_id: str
    
    @property
    def normalized_symbol(self) -> str:
        """Convert MEXC symbol to our format (BTCUSDT -> BTC-USD)"""
        # Handle both formats: BTCUSDT and BTC_USDT
        symbol = self.symbol
        if "_USDT" in symbol:
            base = symbol.replace("_USDT", "")
        elif symbol.endswith("USDT"):
            base = symbol.replace("USDT", "")
        else:
            return symbol
        return f"{base}-USD"


class MEXCCopyTrading:
    """
    MEXC Copy Trading Data Source
    
    Fetches top traders and their positions from MEXC's public API.
    No authentication required!
    """
    
    # Use www.mexc.com for the web API (not contract.mexc.com)
    BASE_URL = "https://www.mexc.com/api/v1/copytrading"
    
    # Browser-like headers to avoid blocks
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.mexc.com/futures/copy-trade",
    }
    
    def __init__(
        self,
        symbols: List[str],
        min_win_rate: float = 50.0,
        min_roi: float = 10.0,
        max_drawdown: float = 30.0,
        top_n_traders: int = 20
    ):
        self.symbols = symbols
        self.min_win_rate = min_win_rate
        self.min_roi = min_roi
        self.max_drawdown = max_drawdown
        self.top_n_traders = top_n_traders
        
        self.traders: Dict[str, MEXCTrader] = {}
        self.positions: List[MEXCPosition] = []
        self.last_update: Optional[datetime] = None
        
        # Map our symbols to MEXC format
        self.symbol_map = {
            "BTC-USD": "BTCUSDT",
            "ETH-USD": "ETHUSDT", 
            "SOL-USD": "SOLUSDT",
            "DOGE-USD": "DOGEUSDT",
            "XRP-USD": "XRPUSDT",
            "ADA-USD": "ADAUSDT",
            "AVAX-USD": "AVAXUSDT",
            "LINK-USD": "LINKUSDT",
            "DOT-USD": "DOTUSDT",
            "MATIC-USD": "MATICUSDT",
            "SHIB-USD": "SHIBUSDT",
            "LTC-USD": "LTCUSDT",
        }
        
        # Reverse map for lookups
        self.reverse_symbol_map = {v: k for k, v in self.symbol_map.items()}
        
        logger.info(
            f"MEXCCopyTrading initialized: "
            f"min_win_rate={min_win_rate}%, min_roi={min_roi}%, max_drawdown={max_drawdown}%"
        )
    
    async def _fetch_json(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Fetch JSON from URL with error handling"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    headers=self.HEADERS, 
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        if resp.status == 404:
                            logger.warning(f"MEXC API returned 404 for {url} - endpoint may have changed")
                        elif resp.status == 403:
                            logger.warning(f"MEXC API returned 403 for {url} - blocked (likely US IP)")
                        else:
                            logger.warning(f"MEXC API returned {resp.status} for {url}")
                        return None
                    
                    data = await resp.json()
                    
                    # MEXC returns success in 'code' field (0 = success)
                    if data.get("code") != 0 and data.get("success") is not True:
                        logger.warning(f"MEXC API error: {data.get('msg', data.get('message', 'Unknown error'))}")
                        return None
                    
                    return data
                    
        except asyncio.TimeoutError:
            logger.warning(f"MEXC API timeout for {url}")
            return None
        except Exception as e:
            logger.warning(f"MEXC API error: {e}")
            return None
    
    async def fetch_leaderboard(self, category: RankCategory = RankCategory.ROI) -> List[MEXCTrader]:
        """
        Fetch top traders from MEXC leaderboard
        
        Args:
            category: Ranking category (ROI, PNL, WIN_RATE, LOW_DRAWDOWN)
        
        Returns:
            List of qualified traders
        """
        url = f"{self.BASE_URL}/leaderboard"
        params = {
            "category": category.value,
            "page": 1,
            "pageSize": 50
        }
        
        data = await self._fetch_json(url, params)
        if not data:
            return []
        
        traders = []
        page_data = data.get("data", {})
        rank_data = page_data.get("list", []) if isinstance(page_data, dict) else page_data
        
        for i, item in enumerate(rank_data):
            try:
                # Handle both percentage formats (0.71 or 71.3)
                roi_raw = float(item.get("roi", 0))
                roi = roi_raw if roi_raw > 2 else roi_raw * 100  # If < 2, assume decimal
                
                win_rate_raw = float(item.get("winRate", 0))
                win_rate = win_rate_raw if win_rate_raw > 2 else win_rate_raw * 100
                
                drawdown_raw = abs(float(item.get("maxDrawdown", 0)))
                drawdown = drawdown_raw if drawdown_raw > 2 else drawdown_raw * 100
                
                trader = MEXCTrader(
                    trader_id=str(item.get("traderId", "")),
                    nickname=item.get("nickname", f"Trader_{i}"),
                    avatar=item.get("avatar", ""),
                    roi=roi,
                    pnl=float(item.get("pnl", item.get("totalPnl", 0))),
                    win_rate=win_rate,
                    drawdown=drawdown,
                    follower_count=int(item.get("followers", item.get("followerCount", 0))),
                    trade_count=int(item.get("tradesCount", item.get("tradeCount", 0))),
                    rank=i + 1,
                    category=category
                )
                
                # Filter by criteria
                if self._qualifies(trader):
                    traders.append(trader)
                    
                    if len(traders) >= self.top_n_traders:
                        break
                        
            except Exception as e:
                logger.debug(f"Error parsing trader: {e}")
                continue
        
        logger.info(f"Fetched {len(traders)} qualified traders from MEXC ({category.value})")
        return traders
    
    def _qualifies(self, trader: MEXCTrader) -> bool:
        """Check if trader meets our criteria"""
        return (
            trader.win_rate >= self.min_win_rate and
            trader.roi >= self.min_roi and
            trader.drawdown <= self.max_drawdown
        )
    
    async def fetch_trader_positions(self, trader: MEXCTrader) -> List[MEXCPosition]:
        """
        Fetch open positions for a specific trader
        
        Args:
            trader: The trader to fetch positions for
            
        Returns:
            List of open positions
        """
        url = f"{self.BASE_URL}/trader/detail"
        params = {"traderId": trader.trader_id}
        
        data = await self._fetch_json(url, params)
        if not data:
            return []
        
        positions = []
        detail_data = data.get("data", {})
        
        # Try multiple possible field names for positions
        position_list = (
            detail_data.get("portfolio", []) or 
            detail_data.get("openPositions", []) or 
            detail_data.get("positions", [])
        )
        
        for pos_data in position_list:
            try:
                # Determine side from position type or sign
                pos_type = pos_data.get("side", pos_data.get("positionType", ""))
                if isinstance(pos_type, int):
                    side = "LONG" if pos_type == 1 else "SHORT"
                else:
                    side = "LONG" if "long" in str(pos_type).lower() else "SHORT"
                
                position = MEXCPosition(
                    symbol=pos_data.get("symbol", ""),
                    side=side,
                    size=float(pos_data.get("holdVol", pos_data.get("size", pos_data.get("weightPct", 0)))),
                    entry_price=float(pos_data.get("openAvgPrice", pos_data.get("entryPrice", 0))),
                    mark_price=float(pos_data.get("markPrice", pos_data.get("currentPrice", 0))),
                    leverage=int(pos_data.get("leverage", 1)),
                    pnl=float(pos_data.get("unrealisedPnl", pos_data.get("pnl", 0))),
                    pnl_rate=float(pos_data.get("pnlRate", 0)) * 100 if pos_data.get("pnlRate", 0) < 2 else float(pos_data.get("pnlRate", 0)),
                    trader_id=trader.trader_id
                )
                
                positions.append(position)
                
            except Exception as e:
                logger.debug(f"Error parsing position: {e}")
                continue
        
        return positions
    
    async def update(self) -> Dict[str, Any]:
        """
        Update all trader data and positions
        
        Fetches from multiple categories to get diverse trader types:
        - Top Return (high ROI)
        - Stable Win Rate (consistent)
        - Lowest Drawdown (safe)
        """
        all_traders = {}
        all_positions = []
        
        # Fetch from multiple categories
        categories = [
            RankCategory.ROI,
            RankCategory.WIN_RATE,
            RankCategory.LOW_DRAWDOWN
        ]
        
        for category in categories:
            traders = await self.fetch_leaderboard(category)
            
            for trader in traders:
                # Avoid duplicates, keep best ranked version
                if trader.trader_id not in all_traders:
                    all_traders[trader.trader_id] = trader
            
            await asyncio.sleep(0.3)  # Rate limit
        
        self.traders = all_traders
        
        # Fetch positions for top traders (limit to avoid rate limits)
        traders_to_check = list(all_traders.values())[:15]
        
        for trader in traders_to_check:
            try:
                positions = await self.fetch_trader_positions(trader)
                all_positions.extend(positions)
                await asyncio.sleep(0.2)  # Rate limit
            except Exception as e:
                logger.debug(f"Error fetching positions for {trader.nickname}: {e}")
        
        self.positions = all_positions
        self.last_update = datetime.now(timezone.utc)
        
        logger.info(
            f"MEXC copy trade update: {len(self.traders)} traders, "
            f"{len(self.positions)} positions"
        )
        
        return {
            "traders_count": len(self.traders),
            "positions_count": len(self.positions),
            "timestamp": self.last_update.isoformat()
        }
    
    def get_consensus(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading consensus for a symbol from top traders
        
        Args:
            symbol: Our symbol format (e.g., "BTC-USD")
            
        Returns:
            Dict with long_count, short_count, direction, confidence
        """
        # Convert to MEXC format
        mexc_symbol = self.symbol_map.get(symbol)
        if not mexc_symbol:
            # Try direct match
            mexc_symbol = symbol.replace("-USD", "_USDT")
        
        relevant_positions = [
            p for p in self.positions 
            if p.symbol == mexc_symbol
        ]
        
        if not relevant_positions:
            return {
                "symbol": symbol,
                "long_count": 0,
                "short_count": 0,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "avg_leverage": 0,
                "positions": []
            }
        
        long_positions = [p for p in relevant_positions if p.side == "LONG"]
        short_positions = [p for p in relevant_positions if p.side == "SHORT"]
        
        long_count = len(long_positions)
        short_count = len(short_positions)
        total = long_count + short_count
        
        # Weight by trader quality (win rate * ROI)
        def get_weight(pos: MEXCPosition) -> float:
            trader = self.traders.get(pos.trader_id)
            if trader:
                return (trader.win_rate / 100) * (1 + trader.roi / 100)
            return 1.0
        
        long_weight = sum(get_weight(p) for p in long_positions)
        short_weight = sum(get_weight(p) for p in short_positions)
        total_weight = long_weight + short_weight
        
        if total_weight == 0:
            return {
                "symbol": symbol,
                "long_count": long_count,
                "short_count": short_count,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "avg_leverage": 0,
                "positions": []
            }
        
        # Determine direction and confidence
        if long_weight > short_weight:
            direction = "LONG"
            confidence = long_weight / total_weight
            consensus_ratio = long_count / total
        elif short_weight > long_weight:
            direction = "SHORT"
            confidence = short_weight / total_weight
            consensus_ratio = short_count / total
        else:
            direction = "NEUTRAL"
            confidence = 0.5
            consensus_ratio = 0.5
        
        # Boost confidence if strong consensus
        if consensus_ratio >= 0.7:
            confidence = min(0.95, confidence * 1.1)
        
        # Calculate average leverage
        avg_leverage = sum(p.leverage for p in relevant_positions) / total if total > 0 else 0
        
        return {
            "symbol": symbol,
            "long_count": long_count,
            "short_count": short_count,
            "direction": direction,
            "confidence": round(confidence, 3),
            "consensus_ratio": round(consensus_ratio, 3),
            "avg_leverage": round(avg_leverage, 1),
            "positions": [
                {
                    "trader": self.traders.get(p.trader_id, MEXCTrader(p.trader_id, "Unknown")).nickname,
                    "side": p.side,
                    "leverage": p.leverage,
                    "pnl_rate": round(p.pnl_rate, 2)
                }
                for p in relevant_positions[:5]
            ]
        }
    
    def get_all_consensus(self) -> Dict[str, Dict]:
        """Get consensus for all tracked symbols"""
        return {symbol: self.get_consensus(symbol) for symbol in self.symbols}


class MEXCCopyTradeAggregator:
    """
    Wrapper that integrates MEXC copy trading with our signal system
    """
    
    def __init__(
        self,
        symbols: List[str],
        min_win_rate: float = 50.0,
        min_roi: float = 10.0,
        max_drawdown: float = 30.0
    ):
        self.symbols = symbols
        self.mexc = MEXCCopyTrading(
            symbols=symbols,
            min_win_rate=min_win_rate,
            min_roi=min_roi,
            max_drawdown=max_drawdown
        )
        
        logger.info(f"MEXCCopyTradeAggregator initialized for {symbols}")
    
    async def update(self, prices: Dict[str, float] = None) -> Dict[str, Any]:
        """Update copy trade data from MEXC"""
        result = await self.mexc.update()
        
        # Add consensus data
        result["consensus"] = self.mexc.get_all_consensus()
        result["positions"] = self.mexc.positions
        
        return result
    
    def get_signal_summary(self, symbol: str) -> Dict[str, Any]:
        """Get aggregated signal for a symbol"""
        return self.mexc.get_consensus(symbol)


# Test function
async def test_mexc_copy_trading():
    """Test the MEXC copy trading integration"""
    print("Testing MEXC Copy Trading API...")
    
    source = MEXCCopyTrading(
        symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
        min_win_rate=40.0,
        min_roi=5.0,
        max_drawdown=50.0
    )
    
    print("\n1. Fetching Top Return leaderboard...")
    traders = await source.fetch_leaderboard(RankCategory.ROI)
    
    if traders:
        print(f"   Found {len(traders)} traders")
        for t in traders[:5]:
            print(f"   - {t.nickname}: ROI={t.roi:.1f}%, WinRate={t.win_rate:.1f}%, DD={t.drawdown:.1f}%")
    else:
        print("   No traders found (API may be blocked from this network)")
    
    print("\n2. Full update with positions...")
    result = await source.update()
    print(f"   Traders: {result['traders_count']}, Positions: {result['positions_count']}")
    
    print("\n3. Consensus for symbols:")
    for symbol in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        consensus = source.get_consensus(symbol)
        print(f"   {symbol}: {consensus['direction']} (conf={consensus['confidence']:.2f})")
        print(f"      Long: {consensus['long_count']}, Short: {consensus['short_count']}")
    
    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(test_mexc_copy_trading())
