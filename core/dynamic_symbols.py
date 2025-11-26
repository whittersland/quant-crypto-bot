"""
Dynamic Symbol Scanner

Automatically finds and adds top-moving cryptocurrencies to the trading universe.
Scans Coinbase for volume spikes, price momentum, and volatility.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ScanMetric(Enum):
    """Metrics for scanning top movers"""
    VOLUME_SPIKE = "volume_spike"
    PRICE_CHANGE_24H = "price_change_24h"
    PRICE_CHANGE_1H = "price_change_1h"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    COMPOSITE = "composite"


@dataclass
class SymbolScore:
    """Score for a tradeable symbol"""
    symbol: str
    volume_score: float = 0.0
    price_change_24h: float = 0.0
    price_change_1h: float = 0.0
    volatility_score: float = 0.0
    momentum_score: float = 0.0
    composite_score: float = 0.0
    
    current_price: float = 0.0
    volume_24h: float = 0.0
    market_cap_rank: int = 999
    
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_composite(self, weights: Dict[str, float] = None):
        """Calculate composite score from individual metrics"""
        weights = weights or {
            "volume": 0.25,
            "price_24h": 0.20,
            "price_1h": 0.15,
            "volatility": 0.20,
            "momentum": 0.20
        }
        
        self.composite_score = (
            abs(self.volume_score) * weights["volume"] +
            abs(self.price_change_24h) * weights["price_24h"] +
            abs(self.price_change_1h) * weights["price_1h"] +
            self.volatility_score * weights["volatility"] +
            abs(self.momentum_score) * weights["momentum"]
        )


@dataclass
class ScanConfig:
    """Configuration for symbol scanning"""
    # Core settings
    base_symbols: List[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD"])
    max_dynamic_symbols: int = 5
    min_volume_24h: float = 100_000  # $100K minimum (VVE/CVD matters more than raw volume)
    min_price: float = 0.01  # Minimum price $0.01
    
    # Scan frequency
    scan_interval_minutes: int = 30
    
    # Score thresholds - lowered since VVE+CVD is more selective
    min_composite_score: float = 0.15
    
    # Filters
    exclude_stablecoins: bool = True
    exclude_wrapped: bool = True
    only_usd_pairs: bool = True
    
    # Blacklist
    blacklist: List[str] = field(default_factory=lambda: [
        "USDT-USD", "USDC-USD", "DAI-USD", "BUSD-USD",  # Stablecoins
        "WBTC-USD", "WETH-USD",  # Wrapped
    ])


class DynamicSymbolScanner:
    """
    Scans for top-moving symbols to add to trading universe
    
    Features:
    - Volume spike detection
    - Price momentum analysis
    - Volatility scoring
    - Automatic symbol addition/removal
    - Configurable filters
    """
    
    def __init__(
        self,
        client,  # CoinbaseClient instance
        config: ScanConfig = None
    ):
        self.client = client
        self.config = config or ScanConfig()
        
        # Active symbols (base + dynamic)
        self.base_symbols = set(self.config.base_symbols)
        self.dynamic_symbols: Dict[str, SymbolScore] = {}
        
        # All scored symbols
        self.all_scores: Dict[str, SymbolScore] = {}
        
        # Scan history
        self.last_scan: Optional[datetime] = None
        self.scan_count: int = 0
        
        logger.info(f"DynamicSymbolScanner initialized with base: {self.base_symbols}")
    
    @property
    def active_symbols(self) -> List[str]:
        """Get all active trading symbols"""
        return list(self.base_symbols | set(self.dynamic_symbols.keys()))
    
    async def scan(self) -> Dict:
        """
        Perform full market scan
        
        Returns:
            Dict with scan results including new_symbols
        """
        logger.info("Starting market scan for top movers...")
        
        result = {"new_symbols": [], "scanned": 0, "top_movers": []}
        
        try:
            # Check if enough time has passed since last scan
            if self.last_scan:
                elapsed = (datetime.now(timezone.utc) - self.last_scan).total_seconds() / 60
                if elapsed < self.config.scan_interval_minutes:
                    return result  # Not time yet
            
            # Get all available products
            products = await self.client.get_products()
            
            # Filter to tradeable USD pairs
            tradeable = self._filter_products(products)
            result["scanned"] = len(tradeable)
            logger.info(f"Found {len(tradeable)} tradeable symbols")
            
            # Score each symbol
            scores = await self._score_symbols(tradeable)
            
            # Update top movers
            new_symbols = self._update_dynamic_symbols(scores)
            result["new_symbols"] = new_symbols
            result["top_movers"] = list(self.dynamic_symbols.keys())
            
            self.last_scan = datetime.now(timezone.utc)
            self.scan_count += 1
            
            logger.info(f"Scan complete. Active symbols: {self.active_symbols}")
            
            return result
            
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return result
    
    def _filter_products(self, products: List[Dict]) -> List[Dict]:
        """Filter products based on config"""
        filtered = []
        
        for product in products:
            product_id = product.get("product_id", "")
            
            # Skip if in blacklist
            if product_id in self.config.blacklist:
                continue
            
            # Only USD pairs
            if self.config.only_usd_pairs and not product_id.endswith("-USD"):
                continue
            
            # Skip stablecoins
            if self.config.exclude_stablecoins:
                base = product_id.split("-")[0]
                if base in ["USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "GUSD"]:
                    continue
            
            # Skip wrapped tokens
            if self.config.exclude_wrapped:
                base = product_id.split("-")[0]
                if base.startswith("W") and len(base) <= 5:
                    continue
            
            # Check if tradeable
            if product.get("status") != "online":
                continue
            
            # Check minimum price
            price = float(product.get("price", 0) or 0)
            if price < self.config.min_price:
                continue
            
            filtered.append(product)
        
        return filtered
    
    async def _score_symbols(self, products: List[Dict]) -> List[SymbolScore]:
        """Score all symbols"""
        scores = []
        
        # First, try to sort by volume_24h if available
        # Coinbase returns this differently depending on endpoint
        def get_volume(p):
            # Try different possible keys
            vol = p.get("volume_24h") or p.get("volume_24hr") or p.get("base_volume") or 0
            try:
                return float(vol)
            except (ValueError, TypeError):
                return 0
        
        sorted_products = sorted(
            products,
            key=get_volume,
            reverse=True
        )[:50]  # Only score top 50 by volume
        
        logger.info(f"Scoring top {len(sorted_products)} symbols...")
        
        scored_count = 0
        failed_count = 0
        
        for i, product in enumerate(sorted_products):
            symbol = product.get("product_id")
            
            try:
                score = await self._score_single_symbol(symbol, product)
                if score:
                    scores.append(score)
                    self.all_scores[symbol] = score
                    scored_count += 1
            except Exception as e:
                logger.debug(f"Could not score {symbol}: {e}")
                failed_count += 1
                continue
            
            # Progress update every 10 symbols
            if (i + 1) % 10 == 0:
                logger.info(f"  Scored {i+1}/{len(sorted_products)} symbols ({scored_count} successful)...")
        
        logger.info(f"Scoring complete: {scored_count} scored, {failed_count} failed")
        
        return scores
    
    async def _score_single_symbol(
        self,
        symbol: str,
        product: Dict
    ) -> Optional[SymbolScore]:
        """
        Calculate score for a single symbol using Volume-Volatility Expansion (VVE) + CVD
        
        VVE detects when both volume AND volatility expand together - a sign of
        institutional activity or breakout potential.
        
        CVD (Cumulative Volume Delta) estimates buying vs selling pressure
        by analyzing price movement relative to volume.
        """
        try:
            # Get candles for analysis - need more data for CVD
            candles = await self.client.get_candles(
                product_id=symbol,
                granularity="ONE_HOUR",
                limit=48  # 48 hours for better baseline
            )
            
            if not candles or len(candles) < 20:
                logger.debug(f"{symbol}: Not enough candles ({len(candles) if candles else 0})")
                return None
            
            # Extract OHLCV data - Candle objects have attributes, not dict keys
            opens = [float(c.open) for c in candles]
            closes = [float(c.close) for c in candles]
            volumes = [float(c.volume) for c in candles]
            highs = [float(c.high) for c in candles]
            lows = [float(c.low) for c in candles]
            
            if not closes or closes[-1] == 0:
                logger.debug(f"{symbol}: No valid close prices")
                return None
            
            current_price = closes[-1]
            
            # ===== VOLUME-VOLATILITY EXPANSION (VVE) =====
            # VVE = (Volume_Ratio) * (Volatility_Ratio)
            # Both must expand together for high score
            
            # Volume expansion: recent 4h vs baseline 24h
            if len(volumes) >= 24:
                recent_vol = np.mean(volumes[-4:]) if len(volumes) >= 4 else volumes[-1]
                baseline_vol = np.mean(volumes[-24:-4]) if len(volumes) > 4 else np.mean(volumes)
                volume_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            # Volatility expansion: recent ATR vs baseline ATR
            if len(highs) >= 24:
                recent_ranges = [h - l for h, l in zip(highs[-4:], lows[-4:])]
                baseline_ranges = [h - l for h, l in zip(highs[-24:-4], lows[-24:-4])]
                recent_atr = np.mean(recent_ranges) if recent_ranges else 0
                baseline_atr = np.mean(baseline_ranges) if baseline_ranges else 1
                volatility_ratio = recent_atr / baseline_atr if baseline_atr > 0 else 1.0
            else:
                volatility_ratio = 1.0
            
            # VVE Score: geometric mean of ratios, minus 1 to center at 0
            # High VVE = both volume AND volatility expanding
            vve_score = np.sqrt(volume_ratio * volatility_ratio) - 1.0
            vve_score = max(-1.0, min(2.0, vve_score))  # Cap at -100% to +200%
            
            # ===== CUMULATIVE VOLUME DELTA (CVD) =====
            # Approximates buying vs selling pressure
            # Positive CVD = more buying, Negative CVD = more selling
            
            cvd_values = []
            for i in range(len(candles)):
                o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]
                
                if h == l:
                    delta = 0
                else:
                    # Volume delta = volume * (close - open) / (high - low)
                    # Normalized by candle range
                    delta = v * (c - o) / (h - l)
                
                cvd_values.append(delta)
            
            # Recent CVD trend (last 6 hours)
            if len(cvd_values) >= 6:
                recent_cvd = sum(cvd_values[-6:])
                baseline_cvd = sum(cvd_values[-24:-6]) if len(cvd_values) >= 24 else sum(cvd_values[:-6])
                
                # Normalize CVD by total volume
                total_recent_vol = sum(volumes[-6:]) if volumes[-6:] else 1
                total_baseline_vol = sum(volumes[-24:-6]) if len(volumes) >= 24 else sum(volumes[:-6])
                
                recent_cvd_norm = recent_cvd / total_recent_vol if total_recent_vol > 0 else 0
                baseline_cvd_norm = baseline_cvd / total_baseline_vol if total_baseline_vol > 0 else 0
                
                # CVD score: recent normalized CVD (range roughly -1 to +1)
                cvd_score = recent_cvd_norm
                cvd_score = max(-1.0, min(1.0, cvd_score))
            else:
                cvd_score = 0
            
            # ===== TREND STRENGTH =====
            # EMA crossover style: short EMA vs long EMA
            if len(closes) >= 20:
                ema_short = self._calculate_ema(closes, 8)
                ema_long = self._calculate_ema(closes, 21)
                trend_score = (ema_short - ema_long) / ema_long if ema_long > 0 else 0
                trend_score = max(-0.5, min(0.5, trend_score))
            else:
                trend_score = 0
            
            # ===== PRICE MOMENTUM =====
            # Rate of change over different periods
            price_change_24h = (closes[-1] - closes[-24]) / closes[-24] if len(closes) >= 24 and closes[-24] > 0 else 0
            price_change_4h = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 and closes[-4] > 0 else 0
            
            momentum_score = (price_change_24h * 0.4 + price_change_4h * 0.6)
            
            # ===== CALCULATE 24H VOLUME (for minimum filter) =====
            volume_24h = sum(v * p for v, p in zip(volumes[-24:], closes[-24:])) if len(volumes) >= 24 else sum(v * p for v, p in zip(volumes, closes))
            
            # Relaxed volume filter - $100K minimum instead of $1M
            # VVE and CVD matter more than absolute volume
            min_volume = 100_000  # $100K
            if volume_24h < min_volume:
                # Only log first few to avoid spam
                if symbol in ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"]:
                    logger.info(f"{symbol}: Volume ${volume_24h/1e3:.0f}K < min ${min_volume/1e3:.0f}K")
                return None
            
            # Log successful score calculation for visibility
            logger.info(f"✓ {symbol}: VVE={vve_score:+.2f}, CVD={cvd_score:+.2f}, vol=${volume_24h/1e6:.1f}M")
            
            # ===== COMPOSITE SCORE =====
            # Weight VVE and CVD heavily as they detect institutional activity
            score = SymbolScore(
                symbol=symbol,
                volume_score=vve_score,  # VVE instead of raw volume spike
                price_change_24h=price_change_24h,
                price_change_1h=price_change_4h,  # Using 4h instead
                volatility_score=volatility_ratio - 1,  # Volatility expansion
                momentum_score=cvd_score,  # CVD as momentum proxy
                current_price=current_price,
                volume_24h=volume_24h
            )
            
            # Custom composite calculation for VVE + CVD
            # VVE: 35%, CVD: 25%, Trend: 20%, Momentum: 20%
            score.composite_score = (
                vve_score * 0.35 +           # Volume-Volatility Expansion
                abs(cvd_score) * 0.25 +      # CVD magnitude (direction matters less for scanning)
                abs(trend_score) * 0.20 +    # Trend strength
                abs(momentum_score) * 0.20   # Price momentum
            )
            
            # Bonus for aligned signals (CVD direction matches price direction)
            if (cvd_score > 0 and price_change_4h > 0) or (cvd_score < 0 and price_change_4h < 0):
                score.composite_score *= 1.2  # 20% bonus for alignment
            
            return score
            
        except Exception as e:
            logger.warning(f"Error scoring {symbol}: {e}")
            return None
    
    def _calculate_ema(self, data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return data[-1] if data else 0
        
        multiplier = 2 / (period + 1)
        ema = data[0]
        
        for price in data[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _update_dynamic_symbols(self, scores: List[SymbolScore]) -> List[str]:
        """Update dynamic symbols based on scores"""
        logger.info(f"Scoring complete: {len(scores)} symbols scored successfully")
        
        # Sort by composite score
        sorted_scores = sorted(
            scores,
            key=lambda s: s.composite_score,
            reverse=True
        )
        
        # Log top 10 by score with VVE and CVD info
        if sorted_scores:
            logger.info("Top 10 movers by VVE+CVD score:")
            for i, s in enumerate(sorted_scores[:10]):
                logger.info(
                    f"  {i+1}. {s.symbol}: score={s.composite_score:.3f}, "
                    f"VVE={s.volume_score:+.2f}, "  # Volume-Volatility Expansion
                    f"CVD={s.momentum_score:+.2f}, "  # Cumulative Volume Delta
                    f"24h={s.price_change_24h*100:+.1f}%, "
                    f"vol=${s.volume_24h/1e3:.0f}K"
                )
        
        # Filter out base symbols and low scores
        candidates = [
            s for s in sorted_scores
            if s.symbol not in self.base_symbols
            and s.composite_score >= self.config.min_composite_score
        ]
        
        logger.info(f"Candidates after filtering (score >= {self.config.min_composite_score}, not in base): {len(candidates)}")
        
        # Select top N
        top_movers = candidates[:self.config.max_dynamic_symbols]
        
        if top_movers:
            logger.info(f"Selected top {len(top_movers)} movers to add:")
            for s in top_movers:
                logger.info(f"  → {s.symbol}: score={s.composite_score:.3f}")
        else:
            logger.info("No new top movers met the criteria")
        
        # Track new additions
        new_symbols = []
        old_dynamic = set(self.dynamic_symbols.keys())
        
        # Update dynamic symbols
        self.dynamic_symbols = {s.symbol: s for s in top_movers}
        
        # Log changes
        new_set = set(self.dynamic_symbols.keys())
        added = new_set - old_dynamic
        removed = old_dynamic - new_set
        
        for symbol in added:
            score = self.dynamic_symbols[symbol]
            logger.info(
                f"✓ ADDED dynamic symbol: {symbol} "
                f"(score={score.composite_score:.3f}, "
                f"price_24h={score.price_change_24h*100:.1f}%, "
                f"volume=${score.volume_24h/1e6:.1f}M)"
            )
            new_symbols.append(symbol)
        
        for symbol in removed:
            logger.info(f"✗ Removed dynamic symbol: {symbol}")
        
        return new_symbols
    
    def add_base_symbol(self, symbol: str) -> None:
        """Add a symbol to base (permanent) list"""
        self.base_symbols.add(symbol)
        # Remove from dynamic if present
        self.dynamic_symbols.pop(symbol, None)
        logger.info(f"Added {symbol} to base symbols")
    
    def remove_base_symbol(self, symbol: str) -> None:
        """Remove a symbol from base list"""
        self.base_symbols.discard(symbol)
        logger.info(f"Removed {symbol} from base symbols")
    
    def get_top_movers(self, n: int = 10) -> List[SymbolScore]:
        """Get top N movers by composite score"""
        return sorted(
            self.all_scores.values(),
            key=lambda s: s.composite_score,
            reverse=True
        )[:n]
    
    def get_symbol_score(self, symbol: str) -> Optional[SymbolScore]:
        """Get score for a specific symbol"""
        return self.all_scores.get(symbol)
    
    def get_scan_report(self) -> Dict:
        """Get comprehensive scan report"""
        return {
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "scan_count": self.scan_count,
            "base_symbols": list(self.base_symbols),
            "dynamic_symbols": list(self.dynamic_symbols.keys()),
            "active_symbols": self.active_symbols,
            "total_scored": len(self.all_scores),
            "top_10_movers": [
                {
                    "symbol": s.symbol,
                    "composite_score": round(s.composite_score, 3),
                    "price_change_24h": f"{s.price_change_24h*100:.1f}%",
                    "price_change_1h": f"{s.price_change_1h*100:.1f}%",
                    "volume_24h": f"${s.volume_24h/1e6:.1f}M",
                    "volatility": f"{s.volatility_score:.2f}%"
                }
                for s in self.get_top_movers(10)
            ],
            "dynamic_details": {
                symbol: {
                    "composite_score": round(score.composite_score, 3),
                    "price_change_24h": f"{score.price_change_24h*100:.1f}%",
                    "volume_24h": f"${score.volume_24h/1e6:.1f}M"
                }
                for symbol, score in self.dynamic_symbols.items()
            }
        }
    
    async def should_scan(self) -> bool:
        """Check if it's time to scan"""
        if not self.last_scan:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self.last_scan).total_seconds() / 60
        return elapsed >= self.config.scan_interval_minutes


class SymbolManager:
    """
    Manages the overall symbol universe combining static and dynamic symbols
    """
    
    def __init__(
        self,
        scanner: DynamicSymbolScanner,
        max_total_symbols: int = 8
    ):
        self.scanner = scanner
        self.max_total_symbols = max_total_symbols
        
        # Symbol state
        self.symbol_performance: Dict[str, Dict] = {}
        
    @property
    def symbols(self) -> List[str]:
        """Get current trading symbols"""
        all_symbols = self.scanner.active_symbols
        
        # Limit total if needed
        if len(all_symbols) > self.max_total_symbols:
            # Prioritize base symbols
            base = list(self.scanner.base_symbols)
            dynamic = [s for s in all_symbols if s not in base]
            
            # Take as many dynamic as we can fit
            remaining_slots = self.max_total_symbols - len(base)
            
            # Sort dynamic by score
            dynamic_sorted = sorted(
                dynamic,
                key=lambda s: self.scanner.all_scores.get(s, SymbolScore(symbol=s)).composite_score,
                reverse=True
            )
            
            return base + dynamic_sorted[:remaining_slots]
        
        return all_symbols
    
    async def update(self) -> Dict:
        """Update symbol universe"""
        if await self.scanner.should_scan():
            new_symbols = await self.scanner.scan()
            
            return {
                "scanned": True,
                "new_symbols": new_symbols,
                "active_symbols": self.symbols
            }
        
        return {
            "scanned": False,
            "active_symbols": self.symbols
        }
    
    def record_symbol_performance(
        self,
        symbol: str,
        pnl: float,
        win: bool
    ) -> None:
        """Record performance for a symbol"""
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {
                "trades": 0,
                "wins": 0,
                "total_pnl": 0.0
            }
        
        self.symbol_performance[symbol]["trades"] += 1
        if win:
            self.symbol_performance[symbol]["wins"] += 1
        self.symbol_performance[symbol]["total_pnl"] += pnl
    
    def get_symbol_stats(self, symbol: str) -> Dict:
        """Get performance stats for a symbol"""
        stats = self.symbol_performance.get(symbol, {})
        if not stats:
            return {"trades": 0, "win_rate": 0, "total_pnl": 0}
        
        return {
            "trades": stats["trades"],
            "win_rate": stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0,
            "total_pnl": stats["total_pnl"]
        }
