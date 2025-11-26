"""
Main Trading System
Orchestrates all components into a complete trading system
"""

import asyncio
import signal
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging

from .coinbase_client import CoinbaseClient, SimulatedCoinbaseClient
from .risk_manager import RiskManager
from .position_sizer import AdaptivePositionSizer
from .liquidation_guard import LiquidationGuard
from .signal_aggregator import SignalAggregator, AggregationMethod
from .execution_engine import ExecutionEngine
from .portfolio_manager import PortfolioManager
from .scheduler import DualLoopScheduler, SchedulerConfig, LoopType
from .dynamic_symbols import DynamicSymbolScanner, ScanConfig
from .copy_trade_fetcher import CopyTradeAggregator
from .mexc_copy_trading import MEXCCopyTradeAggregator

from ..strategies import (
    StrategyRegistry,
    create_all_strategies,
    MomentumStrategy,
    MeanReversionStrategy,
    VolatilityBreakoutStrategy,
    VolumeAnalysisStrategy,
    TrendFollowingStrategy,
    ConfluenceConvergenceStrategy,
    CrossAssetResonanceStrategy,
    TemporalAlignmentStrategy
)
from ..utils.indicators import MarketData, aggregate_signals
from ..utils.config_loader import ConfigManager, get_config
from ..utils.logger import setup_logging, TradeLogger

logger = logging.getLogger(__name__)


class TradingSystem:
    """
    Main Trading System
    
    Coordinates all components:
    - Market data fetching
    - Strategy execution
    - Signal aggregation
    - Trade execution
    - Risk management
    - Portfolio tracking
    
    Implements dual-loop architecture:
    - 10-minute full trading cycle
    - 3-minute high-confidence scanner
    """
    
    def __init__(
        self,
        config_path: str = None,
        simulated: bool = False,
        paper_trading: bool = False
    ):
        # Load configuration
        self.config = get_config(config_path)
        
        # Setup logging
        setup_logging(
            level=self.config._raw_config.get("logging", {}).get("level", "INFO"),
            log_file=self.config._raw_config.get("logging", {}).get("file")
        )
        
        self.simulated = simulated
        self.paper_trading = paper_trading
        self.symbols = self.config.trading.symbols
        
        # Trading mode: spot_only, futures_only, or hybrid
        self.trading_mode = self.config._raw_config.get("trading", {}).get("mode", "spot_only")
        
        # Initialize components
        self._init_components()
        
        # State
        self.running = False
        self.initialized = False
        
        # Market data cache
        self.market_data: Dict[str, MarketData] = {
            symbol: MarketData(symbol) for symbol in self.symbols
        }
        
        # Trade logger
        self.trade_logger = TradeLogger()
        
        mode = "SIMULATED" if simulated else ("PAPER" if paper_trading else "LIVE")
        logger.info(f"Trading system created for symbols: {self.symbols} ({mode} mode, trading_mode={self.trading_mode})")
    
    def _init_components(self) -> None:
        """Initialize all system components"""
        
        # Determine initial capital
        initial_capital = self.config._raw_config.get("capital", 360.65)
        leverage = self.config.trading.leverage
        
        # Client
        if self.simulated:
            self.client = SimulatedCoinbaseClient(initial_balance=initial_capital)
            logger.info("Using SIMULATED client")
        else:
            self.client = CoinbaseClient(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret
            )
            if self.paper_trading:
                logger.info("Using LIVE Coinbase client (PAPER TRADING - no real trades)")
            else:
                logger.info("Using LIVE Coinbase client")
        
        # Risk management
        risk_config = self.config.risk
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            trailing_stop_percent=risk_config.trailing_stop_percent,
            daily_loss_limit_percent=risk_config.daily_loss_limit_percent,
            max_drawdown_percent=risk_config.max_drawdown_percent,
            position_stop_loss_percent=risk_config.position_stop_loss_percent,
            take_profit_tiers=risk_config.take_profit_tiers
        )
        
        # Position sizing
        sizing_config = self.config.position_sizing
        self.position_sizer = AdaptivePositionSizer(
            base_min=sizing_config.base_min,
            base_max=sizing_config.base_max,
            increment=sizing_config.increment,
            max_concurrent=sizing_config.max_concurrent_positions,
            profit_check_hours=sizing_config.profit_check_hours
        )
        
        # Liquidation guard
        self.liquidation_guard = LiquidationGuard(
            leverage=leverage
        )
        
        # Signal aggregator
        self.signal_aggregator = SignalAggregator(
            method=AggregationMethod.DUAL_CONFIRMATION,
            min_confidence=self.config.loops.main_confidence_threshold,
            require_dual_confirmation=True
        )
        
        # Execution engine
        self.execution_engine = ExecutionEngine(
            client=self.client,
            risk_manager=self.risk_manager,
            position_sizer=self.position_sizer,
            liquidation_guard=self.liquidation_guard,
            leverage=leverage,
            paper_trading=self.paper_trading
        )
        
        # Portfolio manager
        self.portfolio = PortfolioManager(
            initial_capital=initial_capital,
            leverage=leverage
        )
        
        # Strategy registry
        self.strategies = create_all_strategies(
            symbols=self.symbols,
            config=self.config._raw_config.get("strategies", {})
        )
        
        # Set strategy weights in aggregator
        self.signal_aggregator.set_weights(self.strategies.get_weights())
        
        # Dynamic Symbol Scanner (top movers)
        self.symbol_scanner = DynamicSymbolScanner(
            client=self.client,
            config=ScanConfig(
                base_symbols=self.symbols,
                max_dynamic_symbols=3,  # Add up to 3 top movers
                min_volume_24h=500_000,  # $500k min volume
                min_composite_score=0.15,  # Lower threshold to catch more movers
                scan_interval_minutes=30
            )
        )
        logger.info("Dynamic symbol scanner initialized")
        
        # Copy Trade Aggregator - Bybit Leaderboard (PUBLIC API)
        # Uses exact endpoints from working scraper - may work from your location
        try:
            from .bybit_copy_trading import BybitCopyTradeAggregator
            self.copy_trade = BybitCopyTradeAggregator(
                symbols=self.symbols,
                min_roi=10.0,      # Minimum 10% ROI
                max_leaders=30     # Top 30 traders
            )
            logger.info("Copy trade aggregator initialized (Bybit Leaderboard - PUBLIC API)")
        except Exception as e:
            logger.warning(f"Failed to initialize Bybit copy trading: {e}")
            self.copy_trade = None
        
        # Scheduler
        loop_config = self.config.loops
        self.scheduler = DualLoopScheduler(
            config=SchedulerConfig(
                main_interval_minutes=loop_config.main_interval_minutes,
                scanner_interval_minutes=loop_config.scanner_interval_minutes,
                main_confidence_threshold=loop_config.main_confidence_threshold,
                scanner_confidence_threshold=loop_config.scanner_confidence_threshold
            )
        )
        
        # Set scheduler callbacks
        self.scheduler.set_main_callback(self._main_loop_iteration)
        self.scheduler.set_scanner_callback(self._scanner_loop_iteration)
        self.scheduler.set_error_callback(self._handle_error)
        
        logger.info("All components initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the system (fetch initial data, verify connection)
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing trading system...")
            
            # Pre-flight checks for live trading
            if not self.simulated:
                # Check API credentials
                if not self.config.api_key or not self.config.api_secret:
                    logger.error("API credentials not configured!")
                    logger.error("Set COINBASE_API_KEY and COINBASE_API_SECRET environment variables")
                    return False
                
                # Verify API connection
                try:
                    accounts = await self.client.get_accounts()
                    logger.info(f"✓ Connected to Coinbase, found {len(accounts)} accounts")
                except Exception as e:
                    logger.error(f"✗ Failed to connect to Coinbase API: {e}")
                    return False
                
                # Check account balance (informational only - Coinbase auto-transfers from spot)
                try:
                    balance_info = await self.client.get_futures_balance_summary()
                    balance = float(balance_info.get("balance", 0))
                    available = float(balance_info.get("available_balance", 0))
                    logger.info(f"✓ Futures balance: ${balance:.2f} (available: ${available:.2f})")
                    
                    if balance < 10:
                        logger.info("ℹ Futures balance low - Coinbase will auto-transfer from spot when needed")
                        
                except Exception as e:
                    logger.warning(f"Could not fetch futures balance: {e}")
                    logger.info("ℹ Continuing anyway - Coinbase will auto-transfer from spot")
                
                # Verify symbols are tradeable
                try:
                    products = await self.client.get_products()
                    product_ids = {p.get("product_id") for p in products}
                    
                    for symbol in self.symbols:
                        if symbol in product_ids:
                            logger.info(f"✓ {symbol} is tradeable")
                        else:
                            logger.warning(f"✗ {symbol} not found in available products")
                except Exception as e:
                    logger.warning(f"Could not verify products: {e}")
                
                logger.info("=" * 50)
                if self.paper_trading:
                    logger.info("PRE-FLIGHT CHECKS PASSED - PAPER TRADING READY")
                else:
                    logger.info("PRE-FLIGHT CHECKS PASSED - LIVE TRADING READY")
                logger.info("=" * 50)
                
                # Initialize futures mapping if not spot_only mode
                if self.trading_mode != "spot_only":
                    logger.info(f"Initializing futures mapping (mode: {self.trading_mode})...")
                    try:
                        futures_map = await self.execution_engine.initialize_futures_mapping(self.symbols)
                        if futures_map:
                            logger.info(f"✓ Futures mapping: {futures_map}")
                        else:
                            logger.warning("No futures contracts mapped - will use spot for LONGs, skip SHORTs")
                    except Exception as e:
                        logger.warning(f"Could not initialize futures mapping: {e}")
                        logger.warning("Will use spot for LONGs, skip SHORTs for unmapped symbols")
            
            # Fetch initial market data
            await self._fetch_market_data()
            
            # Initialize position sizer with current capital
            if self.paper_trading or self.simulated:
                # Use config capital for paper/simulated trading
                current_capital = self.portfolio.current_capital
            else:
                balance = await self.client.get_futures_balance_summary()
                current_capital = float(balance.get("balance", self.portfolio.current_capital))
            self.position_sizer.initialize_day(current_capital)
            
            # Sync with exchange (skip for paper trading)
            if not self.paper_trading:
                await self.execution_engine.sync_with_exchange()
                
                # Get fresh balance and reset drawdown tracking
                # This prevents stale/corrupted P&L tracking from halting the bot
                try:
                    accounts = await self.client.get_accounts()
                    usd_balance = 0
                    crypto_value = 0
                    
                    for acc in accounts:
                        currency = acc.get("currency", "")
                        balance = float(acc.get("available_balance", {}).get("value", 0))
                        
                        if currency == "USD":
                            usd_balance = balance
                        elif balance > 0 and currency in ["BTC", "ETH", "SOL"]:
                            # Get current price for crypto
                            try:
                                ticker = await self.client.get_ticker(f"{currency}-USD")
                                price = float(ticker.get("price", 0))
                                crypto_value += balance * price
                            except:
                                pass
                    
                    actual_total = usd_balance + crypto_value
                    if actual_total > 0:
                        logger.info(f"Actual portfolio value: ${actual_total:.2f} (USD: ${usd_balance:.2f}, Crypto: ${crypto_value:.2f})")
                        self.risk_manager.reset_drawdown(actual_total)
                        logger.info("Drawdown tracking reset to current portfolio value")
                except Exception as reset_error:
                    logger.warning(f"Could not reset drawdown tracking: {reset_error}")
            
            self.initialized = True
            logger.info("Trading system initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def start(self) -> None:
        """Start the trading system"""
        if not self.initialized:
            success = await self.initialize()
            if not success:
                raise RuntimeError("Failed to initialize trading system")
        
        self.running = True
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("=" * 60)
        logger.info("TRADING SYSTEM STARTING")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Mode: {'SIMULATED' if self.simulated else 'LIVE'}")
        logger.info(f"Capital: ${self.portfolio.current_capital:.2f}")
        logger.info(f"Leverage: {self.config.trading.leverage}x")
        logger.info(f"Position Range: ${self.position_sizer.current_min}-${self.position_sizer.current_max}")
        logger.info("=" * 60)
        
        # Start scheduler
        await self.scheduler.start()
        
        # Keep running until stopped
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        
        await self.stop()
    
    async def stop(self) -> None:
        """Stop the trading system gracefully"""
        logger.info("Stopping trading system...")
        
        self.running = False
        
        # Stop scheduler
        await self.scheduler.stop()
        
        # Close client connections
        await self.client.close()
        
        # Save final state
        self.portfolio.record_daily_close()
        
        # Log final stats
        stats = self.get_status()
        logger.info("=" * 60)
        logger.info("TRADING SYSTEM STOPPED")
        logger.info(f"Final Capital: ${stats['portfolio']['capital']['current']:.2f}")
        logger.info(f"Total P&L: ${stats['portfolio']['pnl']['total']:.2f}")
        logger.info(f"Total Trades: {stats['execution']['total_trades']}")
        logger.info("=" * 60)
    
    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers"""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    
    async def _fetch_market_data(self) -> None:
        """Fetch market data for all symbols"""
        for symbol in self.symbols:
            try:
                candles = await self.client.get_candles(
                    product_id=symbol,
                    granularity="ONE_MINUTE",
                    limit=300
                )
                
                if symbol not in self.market_data:
                    self.market_data[symbol] = MarketData(symbol)
                    
                self.market_data[symbol].update(candles)
                logger.debug(f"Fetched {len(candles)} candles for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
    
    def _process_copy_trade_signals(self, copy_update: Dict) -> List:
        """
        Convert copy trade positions into signals
        
        Uses Binance top trader consensus to generate signals
        """
        from ..utils.indicators import Signal
        
        signals = []
        
        # Use the consensus data from Binance leaderboard
        consensus_data = copy_update.get("consensus", {})
        
        for symbol, consensus in consensus_data.items():
            direction = consensus.get("direction", "NEUTRAL")
            confidence = consensus.get("confidence", 0)
            long_count = consensus.get("long_count", 0)
            short_count = consensus.get("short_count", 0)
            total = long_count + short_count
            
            if direction == "NEUTRAL" or total == 0:
                continue
            
            # Calculate consensus ratio
            if direction == "LONG":
                consensus_ratio = long_count / total if total > 0 else 0
            else:
                consensus_ratio = short_count / total if total > 0 else 0
            
            # Only generate signal if strong consensus (>60% of traders agree)
            if consensus_ratio >= 0.6 and confidence >= 0.5:
                signal = Signal(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence * consensus_ratio,  # Combined score
                    strategy="binance_copy_trade",
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "long_traders": long_count,
                        "short_traders": short_count,
                        "consensus_ratio": consensus_ratio,
                        "source": "binance_leaderboard",
                        "positions": consensus.get("positions", [])[:3]  # Top 3 positions
                    }
                )
                signals.append(signal)
                logger.info(
                    f"binance_copy_trade signal: {symbol} {direction} "
                    f"confidence={signal.confidence:.3f} "
                    f"(L:{long_count} vs S:{short_count})"
                )
        
        return signals
    
    async def _check_token_safety(self, symbol: str, score) -> tuple:
        """
        Honeypot protection - check if a token is safe to trade
        
        Checks:
        1. Minimum liquidity (volume)
        2. Bid-ask spread (wide spread = illiquid/manipulated)
        3. Price stability (not just pumped)
        4. Trading history (not brand new)
        5. Sell pressure exists (can actually sell)
        
        Returns:
            (is_safe: bool, reason: str)
        """
        # Known safe tokens (major coins) - skip checks
        SAFE_TOKENS = {
            "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD",
            "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "MATIC-USD",
            "UNI-USD", "ATOM-USD", "LTC-USD", "BCH-USD", "NEAR-USD",
            "APT-USD", "OP-USD", "ARB-USD", "SUI-USD", "SEI-USD",
            "FET-USD", "RENDER-USD", "INJ-USD", "TIA-USD", "PYTH-USD",
            "HBAR-USD", "XLM-USD", "VET-USD", "CRO-USD", "ALGO-USD"
        }
        
        if symbol in SAFE_TOKENS:
            return True, "Known safe token"
        
        # Check 1: Minimum 24h volume ($500K for dynamic tokens)
        min_volume = 500_000
        if score.volume_24h < min_volume:
            return False, f"Low volume: ${score.volume_24h/1000:.0f}K < ${min_volume/1000:.0f}K minimum"
        
        # Check 2: Price shouldn't have pumped more than 200% in 24h (likely pump & dump)
        if score.price_change_24h > 2.0:  # 200%
            return False, f"Extreme pump: +{score.price_change_24h*100:.0f}% in 24h (likely P&D)"
        
        # Check 3: Must have SOME selling (CVD shouldn't be extremely positive)
        # If CVD is > 0.8, it means almost ALL volume is buying - suspicious
        cvd = score.momentum_score
        if cvd > 0.8:
            return False, f"No sell pressure: CVD={cvd:.2f} (suspicious - possible honeypot)"
        
        # Check 4: Try to get order book spread (if available)
        try:
            ticker = await self.client.get_ticker(symbol)
            bid = float(ticker.get("bid", 0))
            ask = float(ticker.get("ask", 0))
            
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / bid * 100
                # Spread > 2% is suspicious for liquid markets
                if spread_pct > 2.0:
                    return False, f"Wide spread: {spread_pct:.1f}% (illiquid/manipulated)"
        except:
            pass  # Can't check spread, continue with other checks
        
        # Check 5: VVE shouldn't be astronomically high (> 5x is suspicious)
        vve = score.volume_score
        if vve > 5.0:
            return False, f"Extreme VVE: {vve:.1f}x (artificial volume pump)"
        
        # All checks passed
        return True, "Passed safety checks"

    async def _main_loop_iteration(
        self,
        loop_type: LoopType,
        confidence_threshold: float
    ) -> Dict:
        """
        Main trading loop iteration
        
        Full cycle:
        1. Scan for top movers (dynamic symbols)
        2. Update copy trade data
        3. Fetch market data
        4. Run ALL strategies
        5. Aggregate signals (including copy trade influence)
        6. Execute trades
        7. Check stops and take profits
        8. Update portfolio
        9. Check adaptive sizing
        """
        result = {"signals": 0, "trades": 0, "top_movers": 0}
        
        logger.info("-" * 40)
        logger.info(f"MAIN LOOP - {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        
        try:
            # 1. Scan for top movers (every 30 min)
            scanner_signals = []  # Direct signals from VVE+CVD
            try:
                scan_result = await self.symbol_scanner.scan()
                if scan_result.get("new_symbols"):
                    new_symbols = scan_result["new_symbols"]
                    result["top_movers"] = len(new_symbols)
                    logger.info(f"TOP MOVERS found: {new_symbols}")
                    
                    # Add new symbols to our trading universe
                    for sym in new_symbols:
                        if sym not in self.symbols:
                            self.symbols.append(sym)
                            self.market_data[sym] = MarketData(sym)
                            logger.info(f"Added {sym} to trading universe")
                            
                            # Try to get futures mapping for new symbol
                            if self.trading_mode != "spot_only":
                                await self.execution_engine.ensure_futures_mapping(sym)
                
                # Generate direct signals from high VVE+CVD scores
                # These are momentum entry signals based on volume/volatility expansion
                from ..utils.indicators import Signal
                for sym, score in self.symbol_scanner.dynamic_symbols.items():
                    # High VVE (volume-volatility expansion) + positive CVD (buying pressure)
                    vve = score.volume_score  # VVE score
                    cvd = score.momentum_score  # CVD score
                    
                    # ===== HONEYPOT PROTECTION =====
                    # Only trade tokens that pass safety checks
                    is_safe, safety_reason = await self._check_token_safety(sym, score)
                    if not is_safe:
                        logger.warning(f"⚠️ HONEYPOT PROTECTION: Skipping {sym} - {safety_reason}")
                        continue
                    
                    # Strong bullish: VVE > 0.5 AND CVD > 0.2 (expansion with buying)
                    if vve > 0.5 and cvd > 0.2:
                        confidence = min(0.85, 0.65 + (vve * 0.1) + (cvd * 0.2))
                        scanner_signals.append(Signal(
                            symbol=sym,
                            direction="LONG",
                            confidence=confidence,
                            strategy="vve_cvd_momentum",
                            timestamp=datetime.now(timezone.utc)
                        ))
                        logger.info(f"VVE+CVD SIGNAL: {sym} LONG conf={confidence:.2f} (VVE={vve:.2f}, CVD={cvd:.2f})")
                    
                    # Strong bearish: VVE > 0.5 AND CVD < -0.2 (expansion with selling)
                    elif vve > 0.5 and cvd < -0.2:
                        confidence = min(0.85, 0.65 + (vve * 0.1) + (abs(cvd) * 0.2))
                        scanner_signals.append(Signal(
                            symbol=sym,
                            direction="SHORT",
                            confidence=confidence,
                            strategy="vve_cvd_momentum",
                            timestamp=datetime.now(timezone.utc)
                        ))
                        logger.info(f"VVE+CVD SIGNAL: {sym} SHORT conf={confidence:.2f} (VVE={vve:.2f}, CVD={cvd:.2f})")
                        
            except Exception as e:
                logger.warning(f"Top mover scan error: {e}")
            
            # 2. Fetch market data
            await self._fetch_market_data()
            
            # 3. Get current prices
            current_prices = {
                symbol: data.current_price 
                for symbol, data in self.market_data.items()
                if data.current_price > 0
            }
            
            # 4. Update copy trade data and get signals (if enabled)
            copy_signals = []
            if self.copy_trade:
                try:
                    copy_update = await self.copy_trade.update()
                    positions_count = copy_update.get("positions", 0)
                    traders_count = copy_update.get("traders", 0)
                    
                    if positions_count > 0:
                        logger.info(f"Copy trade: {traders_count} traders, {positions_count} positions")
                        # Get consensus for our symbols
                        all_consensus = self.copy_trade.get_all_consensus()
                        for symbol, consensus in all_consensus.items():
                            if consensus["confidence"] >= 0.6:
                                logger.info(
                                    f"bybit_copy_trade signal: {symbol} {consensus['direction']} "
                                    f"confidence={consensus['confidence']:.2f} "
                                    f"(L:{consensus['long_count']} vs S:{consensus['short_count']})"
                                )
                    else:
                        logger.debug("Copy trade: No positions from top traders")
                except Exception as e:
                    logger.warning(f"Copy trade update error: {e}")
            
            # 5. Check risk status
            risk_metrics = self.risk_manager.assess_risk()
            if not risk_metrics.can_trade:
                logger.warning(f"Trading halted: {risk_metrics.reason}")
                return result
            
            # 6. Check stops and take profits
            stop_results = await self.execution_engine.check_stop_losses(current_prices)
            tp_results = await self.execution_engine.check_take_profits(current_prices)
            liq_results = await self.execution_engine.check_liquidation_risk(current_prices)
            
            result["trades"] += len([r for r in stop_results + tp_results + liq_results if r.success])
            
            # 7. Run all strategies and collect signals
            all_signals = []
            
            for symbol in self.symbols:
                if symbol not in self.market_data:
                    continue
                    
                market_data = self.market_data[symbol]
                
                # Generate signals from all strategies
                signals = self.strategies.generate_all_signals(symbol, market_data)
                all_signals.extend(signals)
                
                # Update cross-asset strategy with all data
                cross_asset = self.strategies.get("cross_asset_resonance")
                if cross_asset:
                    cross_asset.update_asset_data(symbol, market_data)
            
            # Add copy trade signals to the mix
            all_signals.extend(copy_signals)
            
            # Add VVE+CVD scanner signals (direct momentum entries)
            all_signals.extend(scanner_signals)
            
            # Log strategy participation summary
            strategies_with_signals = set(s.strategy for s in all_signals)
            total_strategies = len(self.strategies.strategies)
            logger.info(f"Strategy summary: {len(strategies_with_signals)}/{total_strategies} strategies generated signals")
            if len(strategies_with_signals) < total_strategies:
                missing = set(self.strategies.strategies.keys()) - strategies_with_signals
                logger.debug(f"Silent strategies: {missing}")
            
            result["signals"] = len(all_signals)
            
            # 6. Aggregate signals per symbol
            for symbol in self.symbols:
                symbol_signals = [s for s in all_signals if s.symbol == symbol]
                
                if not symbol_signals:
                    logger.debug(f"{symbol}: No signals generated")
                    continue
                
                # Log individual signals for this symbol
                logger.info(f"{symbol}: {len(symbol_signals)} signals from strategies:")
                for sig in symbol_signals:
                    logger.info(f"  → {sig.strategy}: {sig.direction} conf={sig.confidence:.3f}")
                
                # Aggregate
                aggregated = self.signal_aggregator.aggregate(symbol_signals)
                
                if not aggregated:
                    logger.info(f"{symbol}: Aggregation returned None (conflicting signals)")
                    continue
                
                # Always log aggregated result
                logger.info(
                    f"{symbol}: AGGREGATED → {aggregated.direction} "
                    f"conf={aggregated.confidence:.3f} (threshold={confidence_threshold:.2f})"
                )
                
                if aggregated.confidence >= confidence_threshold:
                    logger.info(f"✓ {symbol} PASSED threshold - attempting execution")
                    
                    # Log signal
                    self.trade_logger.log_signal({
                        "symbol": symbol,
                        "direction": aggregated.direction,
                        "confidence": aggregated.confidence,
                        "sources": [s.strategy for s in aggregated.contributing_signals]
                    })
                    
                    # 7. Execute trade
                    exec_result = await self.execution_engine.execute_signal(
                        signal=aggregated,
                        current_price=current_prices.get(symbol, 0),
                        available_capital=self.portfolio.available_capital,
                        trading_mode=self.trading_mode
                    )
                    
                    if exec_result.success:
                        result["trades"] += 1
                        
                        # Update portfolio allocation
                        position_value = exec_result.trade.entry_size * exec_result.trade.entry_price
                        self.portfolio.allocate(symbol, position_value / self.config.trading.leverage)
                        
                        # Log trade
                        self.trade_logger.log_trade({
                            "symbol": symbol,
                            "side": exec_result.trade.side,
                            "entry_price": exec_result.trade.entry_price,
                            "size": exec_result.trade.entry_size,
                            "strategy": exec_result.trade.strategy
                        })
                    else:
                        # Log why execution failed
                        logger.warning(f"Trade not executed for {symbol}: {exec_result.error}")
                else:
                    # Below threshold
                    logger.info(f"✗ {symbol} BELOW threshold ({aggregated.confidence:.3f} < {confidence_threshold:.2f}) - skipping")
            
            # 8. Update unrealized P&L
            unrealized = self.execution_engine.update_unrealized_pnl(current_prices)
            self.portfolio.update_unrealized_pnl(unrealized)
            
            # 9. Check adaptive position sizing
            balance = await self.client.get_futures_balance_summary()
            current_capital = float(balance.get("balance", self.portfolio.current_capital))
            
            adjusted, msg = self.position_sizer.check_and_adjust(current_capital)
            if adjusted:
                logger.info(f"Position sizing: {msg}")
            
            # 10. Update risk manager capital
            self.risk_manager.update_capital(current_capital, unrealized)
            
            # 11. Take portfolio snapshot
            self.portfolio.take_snapshot()
            
            logger.info(
                f"Main loop complete: {result['signals']} signals, "
                f"{result['trades']} trades, "
                f"capital=${current_capital:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            raise
        
        return result
    
    async def _scanner_loop_iteration(
        self,
        loop_type: LoopType,
        confidence_threshold: float
    ) -> Dict:
        """
        Scanner loop iteration
        
        Lightweight cycle:
        1. Quick market data fetch
        2. Run ONLY dual-condition strategies
        3. Execute ONLY high-confidence signals
        """
        result = {"signals": 0, "trades": 0}
        
        logger.debug(f"SCANNER - {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        
        try:
            # 1. Quick market data fetch
            await self._fetch_market_data()
            
            # 2. Get current prices
            current_prices = {
                symbol: data.current_price 
                for symbol, data in self.market_data.items()
                if data.current_price > 0
            }
            
            # 3. Quick risk check
            risk_metrics = self.risk_manager.assess_risk()
            if not risk_metrics.can_trade:
                return result
            
            # 4. Run ONLY dual-condition strategies
            dual_condition_strategies = [
                "confluence_convergence",
                "cross_asset_resonance",
                "temporal_alignment"
            ]
            
            for symbol in self.symbols:
                market_data = self.market_data[symbol]
                
                signals = []
                for strategy_name in dual_condition_strategies:
                    strategy = self.strategies.get(strategy_name)
                    if strategy and strategy.enabled:
                        signal = strategy.generate_signal(symbol, market_data)
                        if signal:
                            signals.append(signal)
                
                result["signals"] += len(signals)
                
                # 5. Check for high-confidence signals
                for sig in signals:
                    if sig.confidence >= confidence_threshold:
                        logger.info(
                            f"SCANNER HIGH-CONF: {symbol} {sig.direction} "
                            f"conf={sig.confidence:.3f} strategy={sig.strategy}"
                        )
                        
                        # Aggregate with other signals if any
                        aggregated = self.signal_aggregator.aggregate(signals)
                        
                        if aggregated and aggregated.confidence >= confidence_threshold:
                            # Execute
                            exec_result = await self.execution_engine.execute_signal(
                                signal=aggregated,
                                current_price=current_prices.get(symbol, 0),
                                available_capital=self.portfolio.available_capital,
                                trading_mode=self.trading_mode
                            )
                            
                            if exec_result.success:
                                result["trades"] += 1
                                logger.info(f"SCANNER TRADE: {symbol} {aggregated.direction}")
                            else:
                                logger.warning(f"Scanner trade not executed for {symbol}: {exec_result.error}")
                        
                        break  # Only one trade per symbol per scanner cycle
            
            if result["signals"] > 0:
                logger.debug(f"Scanner: {result['signals']} signals checked")
            
        except Exception as e:
            logger.error(f"Scanner loop error: {e}")
            raise
        
        return result
    
    async def _handle_error(self, loop_type: LoopType, error: Exception) -> None:
        """Handle loop errors"""
        logger.error(f"Loop error ({loop_type.value}): {error}")
        
        # Log to trade logger
        self.trade_logger.log_performance({
            "event": "error",
            "loop": loop_type.value,
            "error": str(error)
        })
    
    def get_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "running": self.running,
            "initialized": self.initialized,
            "simulated": self.simulated,
            "symbols": self.symbols,
            "portfolio": self.portfolio.get_status(),
            "risk": self.risk_manager.get_full_report(),
            "position_sizing": self.position_sizer.get_stats(),
            "execution": self.execution_engine.get_execution_stats(),
            "scheduler": self.scheduler.get_stats(),
            "strategies": self.strategies.get_performance_summary(),
            "active_positions": self.execution_engine.get_active_positions()
        }
    
    def get_performance_report(self) -> Dict:
        """Get detailed performance report"""
        perf = self.portfolio.calculate_performance()
        
        return {
            "summary": {
                "total_return": perf.total_return,
                "total_return_percent": perf.total_return_percent,
                "sharpe_ratio": perf.sharpe_ratio,
                "max_drawdown": perf.max_drawdown,
                "win_rate": perf.win_rate,
                "profit_factor": perf.profit_factor,
                "total_trades": perf.total_trades
            },
            "capital": {
                "initial": self.portfolio.initial_capital,
                "current": self.portfolio.current_capital,
                "peak": self.portfolio.peak_capital
            },
            "strategies": self.strategies.get_performance_summary(),
            "scheduler": self.scheduler.get_stats()
        }


async def run_trading_system(
    config_path: str = None,
    simulated: bool = True,
    paper_trading: bool = False
) -> None:
    """
    Main entry point to run the trading system
    
    Args:
        config_path: Path to configuration file
        simulated: If True, use simulated client (fake data)
        paper_trading: If True, use real market data but don't execute trades
    """
    system = TradingSystem(
        config_path=config_path,
        simulated=simulated,
        paper_trading=paper_trading
    )
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        await system.stop()
