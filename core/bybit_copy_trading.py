"""
Bybit Copy Trading Integration

Fetches top traders and their positions from Bybit's copy trading leaderboard.
Public API - no authentication required!

Based on working scraper endpoints.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class BybitTrader:
    """Represents a trader from Bybit leaderboard"""
    leader_mark: str
    nickname: str
    roi: float = 0.0
    pnl: float = 0.0
    win_rate: float = 0.0
    follower_count: int = 0
    
    
@dataclass
class BybitPosition:
    """Represents an open position from a Bybit trader"""
    symbol: str
    side: str  # "Buy" or "Sell" -> normalized to "LONG" or "SHORT"
    size: float
    entry_price: float
    leverage: float
    leader_mark: str
    
    @property
    def normalized_side(self) -> str:
        """Convert Bybit side to standard LONG/SHORT"""
        if self.side and self.side.lower() in ["buy", "long"]:
            return "LONG"
        return "SHORT"
    
    @property
    def normalized_symbol(self) -> str:
        """Convert Bybit symbol to our format (BTCUSDT -> BTC-USD)"""
        symbol = self.symbol.upper() if self.symbol else ""
        if symbol.endswith("USDT"):
            base = symbol.replace("USDT", "")
            return f"{base}-USD"
        elif symbol.endswith("USD"):
            base = symbol.replace("USD", "")
            return f"{base}-USD"
        return symbol


class BybitCopyTrading:
    """
    Bybit Copy Trading Data Source
    
    Fetches top traders and their positions from Bybit's public API.
    Uses the exact endpoints from the working scraper.
    """
    
    BASE = "https://www.bybit.com"
    LEADERBOARD_URL = "/x-api/fapi/beehive/public/v1/common/get-leaderboard"
    POSITION_URL = "/x-api/fapi/beehive/public/v1/common/position/list"
    
    # Exact headers from working scraper
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/142.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Referer": "https://www.bybit.com/copyTrade/trade-center/detail",
        "lang": "en",
        "platform": "pc",
    }
    
    def __init__(
        self,
        symbols: List[str],
        min_roi: float = 10.0,
        max_leaders: int = 30
    ):
        self.symbols = symbols
        self.min_roi = min_roi
        self.max_leaders = max_leaders
        
        self.traders: Dict[str, BybitTrader] = {}
        self.positions: List[BybitPosition] = []
        self.last_update: Optional[datetime] = None
        
        # Symbol mapping
        self.symbol_map = {
            "BTC-USD": "BTCUSDT",
            "ETH-USD": "ETHUSDT",
            "SOL-USD": "SOLUSDT",
            "DOGE-USD": "DOGEUSDT",
            "XRP-USD": "XRPUSDT",
            "SUI-USD": "SUIUSDT",
        }
        self.reverse_symbol_map = {v: k for k, v in self.symbol_map.items()}
        
        logger.info(f"BybitCopyTrading initialized: min_roi={min_roi}%, max_leaders={max_leaders}")
    
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
                    if resp.status == 403:
                        logger.warning(f"Bybit API returned 403 (blocked) for {url}")
                        return None
                    if resp.status != 200:
                        logger.warning(f"Bybit API returned {resp.status} for {url}")
                        return None
                    
                    data = await resp.json()
                    return data
                    
        except asyncio.TimeoutError:
            logger.warning(f"Bybit API timeout for {url}")
            return None
        except Exception as e:
            logger.warning(f"Bybit API error: {e}")
            return None
    
    async def fetch_leaderboard(self) -> List[BybitTrader]:
        """Fetch top traders from Bybit leaderboard - exact scraper logic"""
        url = f"{self.BASE}{self.LEADERBOARD_URL}"
        params = {
            "rankingType": "RANKING_TYPE_BEST_FOLLOW_PROFIT",
            "period": "LEADERBOARD_PERIOD_WEEK",
        }
        
        data = await self._fetch_json(url, params)
        if not data:
            return []
        
        traders = []
        
        # Navigate to the list - try multiple possible paths
        container = data.get("result") or data.get("data") or {}
        raw_list = container.get("list") or container.get("leaderboardList") or []
        
        for item in raw_list:
            try:
                # Try multiple field names for leader_mark
                leader_mark = (
                    item.get("leaderMark") or 
                    item.get("leader_mark") or 
                    item.get("mark")
                )
                if not leader_mark:
                    continue
                
                # Parse ROI
                roi_raw = item.get("yield") or item.get("roi") or item.get("pnlRatio") or 0
                try:
                    roi = float(roi_raw)
                    if abs(roi) < 2:  # Likely decimal, convert to percentage
                        roi = roi * 100
                except:
                    roi = 0
                
                nickname = item.get("nickName") or item.get("nickname") or item.get("name") or "Unknown"
                
                trader = BybitTrader(
                    leader_mark=leader_mark,
                    nickname=nickname,
                    roi=roi,
                    pnl=float(item.get("pnl", 0) or 0),
                    follower_count=int(item.get("followerNum", 0) or 0)
                )
                
                # Filter by minimum ROI
                if trader.roi >= self.min_roi:
                    traders.append(trader)
                    
                if len(traders) >= self.max_leaders:
                    break
                    
            except Exception as e:
                logger.debug(f"Error parsing trader: {e}")
                continue
        
        if traders:
            logger.info(f"Fetched {len(traders)} qualified traders from Bybit")
        return traders
    
    async def fetch_trader_positions(self, trader: BybitTrader) -> List[BybitPosition]:
        """Fetch open positions for a specific trader - exact scraper logic"""
        url = f"{self.BASE}{self.POSITION_URL}"
        params = {"leaderMark": trader.leader_mark}
        
        data = await self._fetch_json(url, params)
        if not data:
            return []
        
        positions = []
        
        # Navigate to positions list
        container = data.get("result") or data.get("data") or {}
        raw_list = container.get("list") or container.get("positions") or []
        
        for p in raw_list:
            try:
                symbol = p.get("symbol") or p.get("symbolName") or p.get("s")
                if not symbol:
                    continue
                
                side = p.get("side") or p.get("positionSide") or p.get("posSide") or "Buy"
                
                # Parse numeric fields safely
                size_raw = p.get("size") or p.get("qty") or p.get("volume") or 0
                entry_raw = p.get("entryPrice") or p.get("avgPrice") or p.get("openPrice") or 0
                lev_raw = p.get("leverage") or p.get("lever") or 1
                
                try:
                    size = float(size_raw) if size_raw else 0.0
                except:
                    size = 0.0
                    
                try:
                    entry_price = float(entry_raw) if entry_raw else 0.0
                except:
                    entry_price = 0.0
                    
                try:
                    leverage = float(lev_raw) if lev_raw else 1.0
                except:
                    leverage = 1.0
                
                position = BybitPosition(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    leverage=leverage,
                    leader_mark=trader.leader_mark
                )
                
                positions.append(position)
                
            except Exception as e:
                logger.debug(f"Error parsing position: {e}")
                continue
        
        return positions
    
    async def update(self) -> Dict[str, Any]:
        """Update all trader data and positions"""
        all_traders = {}
        all_positions = []
        
        # Fetch leaderboard
        traders = await self.fetch_leaderboard()
        
        if not traders:
            logger.debug("No traders fetched from Bybit leaderboard")
            return {
                "traders": 0,
                "positions": 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Fetch positions for each trader
        for trader in traders:
            all_traders[trader.leader_mark] = trader
            
            positions = await self.fetch_trader_positions(trader)
            all_positions.extend(positions)
            
            # Rate limiting - be gentle
            await asyncio.sleep(0.3)
        
        self.traders = all_traders
        self.positions = all_positions
        self.last_update = datetime.now(timezone.utc)
        
        logger.info(f"Bybit copy trade update: {len(all_traders)} traders, {len(all_positions)} positions")
        
        return {
            "traders": len(all_traders),
            "positions": len(all_positions),
            "timestamp": self.last_update.isoformat()
        }
    
    def get_consensus(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading consensus for a symbol based on top trader positions
        
        Returns:
            Dict with direction, confidence, and position counts
        """
        our_symbol = symbol  # e.g., "BTC-USD"
        bybit_symbol = self.symbol_map.get(our_symbol, our_symbol.replace("-USD", "USDT"))
        
        long_count = 0
        short_count = 0
        long_weight = 0.0
        short_weight = 0.0
        
        for pos in self.positions:
            # Check if position matches our symbol
            pos_symbol = pos.symbol.upper() if pos.symbol else ""
            if pos_symbol == bybit_symbol.upper() or pos_symbol == bybit_symbol.replace("USDT", "USD"):
                trader = self.traders.get(pos.leader_mark)
                weight = 1.0
                
                # Weight by trader ROI
                if trader and trader.roi > 0:
                    weight = 1.0 + (trader.roi / 100)
                
                if pos.normalized_side == "LONG":
                    long_count += 1
                    long_weight += weight
                else:
                    short_count += 1
                    short_weight += weight
        
        total = long_count + short_count
        
        if total == 0:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "long_count": 0,
                "short_count": 0,
                "consensus_ratio": 0.0
            }
        
        # Determine direction based on weighted counts
        if long_weight > short_weight:
            direction = "LONG"
            consensus_ratio = long_count / total
            confidence = min(0.9, 0.5 + (consensus_ratio - 0.5) * 0.8)
        elif short_weight > long_weight:
            direction = "SHORT"
            consensus_ratio = short_count / total
            confidence = min(0.9, 0.5 + (consensus_ratio - 0.5) * 0.8)
        else:
            direction = "NEUTRAL"
            consensus_ratio = 0.5
            confidence = 0.0
        
        # Boost confidence if strong consensus (>70%)
        if consensus_ratio >= 0.7:
            confidence = min(0.95, confidence * 1.1)
        
        return {
            "direction": direction,
            "confidence": confidence,
            "long_count": long_count,
            "short_count": short_count,
            "consensus_ratio": consensus_ratio
        }


class BybitCopyTradeAggregator:
    """
    Wrapper class that integrates Bybit copy trading with the trading system
    """
    
    def __init__(
        self,
        symbols: List[str],
        min_roi: float = 10.0,
        max_leaders: int = 30
    ):
        self.source = BybitCopyTrading(
            symbols=symbols,
            min_roi=min_roi,
            max_leaders=max_leaders
        )
        self.symbols = symbols
        logger.info(f"BybitCopyTradeAggregator initialized for {symbols}")
    
    async def update(self) -> Dict[str, Any]:
        """Update copy trade data"""
        return await self.source.update()
    
    def get_all_consensus(self) -> Dict[str, Dict]:
        """Get consensus for all tracked symbols"""
        consensus = {}
        for symbol in self.symbols:
            consensus[symbol] = self.source.get_consensus(symbol)
        return consensus
