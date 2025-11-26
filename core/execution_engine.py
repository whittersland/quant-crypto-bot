"""
Trade Execution Engine
Handles order creation, execution, and management
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

from .coinbase_client import CoinbaseClient, OrderResponse, Position
from .signal_aggregator import AggregatedSignal
from .risk_manager import RiskManager
from .position_sizer import AdaptivePositionSizer
from .liquidation_guard import LiquidationGuard

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status tracking"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class TradeType(Enum):
    """Trade type classification"""
    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


@dataclass
class Trade:
    """Represents a complete trade with entry and exit"""
    trade_id: str
    symbol: str  # Original symbol (e.g., "BTC-USD")
    side: str  # "LONG" or "SHORT"
    entry_price: float
    entry_size: float
    entry_time: datetime
    entry_order_id: str
    
    # The actual product used for execution (may differ from symbol for futures)
    # e.g., symbol="BTC-USD" but execution_product="BIT-26DEC25-CDE"
    execution_product: Optional[str] = None
    
    # Exit details (filled when closed)
    exit_price: Optional[float] = None
    exit_size: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_order_id: Optional[str] = None
    exit_reason: Optional[str] = None
    
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Tracking
    strategy: str = ""
    signal_confidence: float = 0.0
    status: str = "open"
    
    # Stop/TP levels
    stop_loss: Optional[float] = None
    take_profit_levels: List[float] = field(default_factory=list)
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current unrealized P&L"""
        if self.side == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.entry_size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.entry_size
        return self.unrealized_pnl
    
    def close(self, exit_price: float, exit_size: float, reason: str) -> float:
        """Close the trade and calculate final P&L"""
        self.exit_price = exit_price
        self.exit_size = exit_size
        self.exit_time = datetime.now(timezone.utc)
        self.exit_reason = reason
        self.status = "closed"
        
        if self.side == "LONG":
            self.realized_pnl = (exit_price - self.entry_price) * exit_size
        else:
            self.realized_pnl = (self.entry_price - exit_price) * exit_size
        
        return self.realized_pnl


@dataclass
class ExecutionResult:
    """Result of a trade execution attempt"""
    success: bool
    trade: Optional[Trade] = None
    order: Optional[OrderResponse] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """
    Trade Execution Engine
    
    Coordinates:
    - Order creation and submission
    - Position tracking
    - Risk management integration
    - Stop loss and take profit management
    """
    
    def __init__(
        self,
        client: CoinbaseClient,
        risk_manager: RiskManager,
        position_sizer: AdaptivePositionSizer,
        liquidation_guard: LiquidationGuard,
        leverage: float = 2.0,
        max_slippage_percent: float = 0.5,
        paper_trading: bool = False
    ):
        self.client = client
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.liquidation_guard = liquidation_guard
        self.leverage = leverage
        self.max_slippage_percent = max_slippage_percent
        self.paper_trading = paper_trading
        
        # Active trades
        self.active_trades: Dict[str, Trade] = {}
        
        # Trade history
        self.trade_history: List[Trade] = []
        
        # Order tracking
        self.pending_orders: Dict[str, Dict] = {}
        
        # Futures product mapping (spot symbol -> futures contract)
        # Will be populated by initialize_futures_mapping()
        self.futures_mapping: Dict[str, str] = {}
        
        # Execution statistics
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "total_volume": 0.0
        }
        
        # Size precision per asset (Coinbase requirements)
        # Futures contracts use different precision (number of contracts)
        self.size_precision = {
            "BTC-USD": 8,
            "ETH-USD": 8,
            "SOL-USD": 2,
            "DOGE-USD": 0,
            "XRP-USD": 0,
            "ADA-USD": 0,
            "AVAX-USD": 2,
            "LINK-USD": 2,
            "DOT-USD": 2,
            "MATIC-USD": 0,
            "UNI-USD": 2,
            "ATOM-USD": 2,
            "LTC-USD": 4,
            "BCH-USD": 4,
            "NEAR-USD": 2,
            "APT-USD": 2,
            "OP-USD": 2,
            "ARB-USD": 2,
            # Additional tokens (dynamic scanner may find these)
            "SUI-USD": 2,
            "ZK-USD": 0,
            "DNT-USD": 0,
            "PEPE-USD": 0,
            "SHIB-USD": 0,
            "FET-USD": 2,
            "HBAR-USD": 0,
            "XLM-USD": 0,
            "VET-USD": 0,
            "CRO-USD": 0,
            "SEI-USD": 2,
            "FLR-USD": 0,
            "PENGU-USD": 0,
            "DIMO-USD": 0,
            "SPK-USD": 0,
            "STRK-USD": 2,
            "MON-USD": 2,
            "MOODENG-USD": 0,
            "ZORA-USD": 0,
            "LINEA-USD": 0,
            "HOME-USD": 0,
            "SAPIEN-USD": 0,
            "MAMO-USD": 0,
            "TRUST-USD": 0,
            "FORT-USD": 0,
            "BOBBOB-USD": 0,
            "FARTCOIN-USD": 0,
            "USELESS-USD": 0,
            # Futures contracts - typically whole numbers (contracts)
            "BIT": 0,  # nano Bitcoin futures
            "ET": 0,   # nano Ether futures
        }
    
    async def initialize_futures_mapping(self, symbols: List[str]) -> Dict[str, str]:
        """
        Initialize futures product mapping for given spot symbols
        
        Args:
            symbols: List of spot symbols like ["BTC-USD", "ETH-USD"]
        
        Returns:
            Mapping of spot symbol to futures contract
        """
        self.futures_mapping = {}
        
        for symbol in symbols:
            base_asset = symbol.split("-")[0]  # "BTC-USD" -> "BTC"
            
            try:
                futures_contract = await self.client.get_best_futures_contract(base_asset)
                if futures_contract:
                    self.futures_mapping[symbol] = futures_contract
                    logger.info(f"Mapped {symbol} -> {futures_contract}")
                else:
                    logger.warning(f"No futures contract found for {symbol} - will use spot for LONG, skip SHORT")
            except Exception as e:
                logger.warning(f"Could not get futures contract for {symbol}: {e}")
        
        return self.futures_mapping
    
    async def ensure_futures_mapping(self, symbol: str) -> Optional[str]:
        """
        Ensure we have futures mapping for a symbol, fetching if needed
        
        Args:
            symbol: Spot symbol like "DOGE-USD"
        
        Returns:
            Futures contract ID if available, None otherwise
        """
        if symbol in self.futures_mapping:
            return self.futures_mapping[symbol]
        
        # Try to fetch futures contract for this symbol
        base_asset = symbol.split("-")[0]
        
        try:
            futures_contract = await self.client.get_best_futures_contract(base_asset)
            if futures_contract:
                self.futures_mapping[symbol] = futures_contract
                logger.info(f"Dynamically mapped {symbol} -> {futures_contract}")
                return futures_contract
        except Exception as e:
            logger.debug(f"Could not get futures for {symbol}: {e}")
        
        return None
    
    def get_execution_product(self, symbol: str, trading_mode: str, direction: str) -> str:
        """
        Get the actual product ID to use for execution
        
        Args:
            symbol: Original symbol like "BTC-USD"
            trading_mode: "spot_only", "futures_only", or "hybrid"
            direction: "LONG" or "SHORT"
        
        Returns:
            Product ID to use for the order
        """
        if trading_mode == "spot_only":
            return symbol
        
        if trading_mode == "futures_only":
            # Always use futures
            return self.futures_mapping.get(symbol, symbol)
        
        if trading_mode == "hybrid":
            # LONG via spot, SHORT via futures
            if direction == "SHORT":
                return self.futures_mapping.get(symbol, symbol)
            return symbol
        
        return symbol
    
    def _get_size_precision(self, symbol: str) -> int:
        """Get size precision for a symbol"""
        # Default to 6 decimals if not specified
        return self.size_precision.get(symbol, 6)
    
    async def execute_signal(
        self,
        signal: AggregatedSignal,
        current_price: float,
        available_capital: float,
        trading_mode: str = "spot_only"
    ) -> ExecutionResult:
        """
        Execute a trading signal
        
        Args:
            signal: Aggregated signal to execute
            current_price: Current market price
            available_capital: Available trading capital
            trading_mode: "spot_only", "futures_only", or "hybrid"
        
        Returns:
            ExecutionResult with trade details
        """
        symbol = signal.symbol
        direction = signal.direction
        
        # Check trading mode constraints
        if trading_mode == "spot_only" and direction == "SHORT":
            logger.info(f"Skipping SHORT signal for {symbol} - spot_only mode (can't short on spot)")
            return ExecutionResult(
                success=False,
                error="SHORT signals disabled in spot_only mode"
            )
        
        # For futures modes, try to get futures mapping if we don't have it
        if trading_mode in ["futures_only", "hybrid"] and symbol not in self.futures_mapping:
            # Try to dynamically fetch futures contract
            await self.ensure_futures_mapping(symbol)
        
        # Check if futures are available for this symbol when needed
        if trading_mode in ["futures_only", "hybrid"] and direction == "SHORT":
            if symbol not in self.futures_mapping:
                logger.info(f"Skipping SHORT signal for {symbol} - no futures contract available")
                return ExecutionResult(
                    success=False,
                    error=f"No futures contract available for {symbol} - cannot short"
                )
        
        # For futures_only mode, also check LONG signals have futures
        if trading_mode == "futures_only" and symbol not in self.futures_mapping:
            logger.info(f"Skipping {direction} signal for {symbol} - no futures contract (futures_only mode)")
            return ExecutionResult(
                success=False,
                error=f"No futures contract available for {symbol} in futures_only mode"
            )
        
        # Check if we can trade
        risk_metrics = self.risk_manager.assess_risk()
        if not risk_metrics.can_trade:
            return ExecutionResult(
                success=False,
                error=f"Trading halted: {risk_metrics.reason}"
            )
        
        # Check if we already have a position in this symbol
        if symbol in self.active_trades:
            existing = self.active_trades[symbol]
            
            # If same direction, might scale in (not implemented in v1)
            if existing.side == direction:
                return ExecutionResult(
                    success=False,
                    error=f"Already have {direction} position in {symbol}"
                )
            
            # If opposite direction, close existing first
            logger.info(f"Closing existing {existing.side} position before {direction}")
            await self.close_position(symbol, "signal_reversal")
        
        # Calculate position size
        risk_multiplier = self.risk_manager.get_position_risk_multiplier()
        current_positions = len(self.active_trades)
        
        position_size, sizing_details = self.position_sizer.calculate_size(
            confidence=signal.confidence,
            available_capital=available_capital,
            current_positions=current_positions,
            risk_multiplier=risk_multiplier
        )
        
        if position_size <= 0:
            return ExecutionResult(
                success=False,
                error=f"Position size too small: {sizing_details.get('reason', 'unknown')}"
            )
        
        # Check liquidation guard
        can_open, guard_reason = self.liquidation_guard.can_open_position(
            symbol=symbol,
            entry_price=current_price,
            size=position_size / current_price,  # Convert to base currency
            available_margin=available_capital,
            side=direction
        )
        
        if not can_open:
            return ExecutionResult(
                success=False,
                error=f"Liquidation guard: {guard_reason}"
            )
        
        # Determine which product to execute on
        execution_product = self.get_execution_product(symbol, trading_mode, direction)
        is_futures = execution_product != symbol
        
        if is_futures:
            logger.info(f"Using futures contract: {execution_product} for {symbol} {direction}")
        
        # Calculate order size
        if is_futures:
            # For futures, position_size is the MARGIN we're willing to use
            # With leverage, we can control more notional value
            # notional_value = margin * leverage
            notional_value = position_size * self.leverage
            
            # Futures: size is in number of contracts
            # nano BTC = 1/100 BTC, nano ETH = 1/10 ETH, nano SOL = 5 SOL
            if "BIT" in execution_product or "BIP" in execution_product:
                # nano Bitcoin: 1 contract = 1/100 BTC
                contract_size = current_price / 100  # Value per contract
                base_size = notional_value / contract_size
                logger.info(f"BTC futures: margin=${position_size:.2f}, leverage={self.leverage}x, notional=${notional_value:.2f}, contract_value=${contract_size:.2f}, contracts={base_size:.2f}")
            elif "ET" in execution_product or "ETP" in execution_product:
                # nano Ether: 1 contract = 1/10 ETH
                contract_size = current_price / 10
                base_size = notional_value / contract_size
                logger.info(f"ETH futures: margin=${position_size:.2f}, leverage={self.leverage}x, notional=${notional_value:.2f}, contract_value=${contract_size:.2f}, contracts={base_size:.2f}")
            elif "SOL" in execution_product or "SLP" in execution_product:
                # nano Solana: 5 SOL per contract
                contract_size = current_price * 5  # 5 SOL per contract
                base_size = notional_value / contract_size
                logger.info(f"SOL futures: margin=${position_size:.2f}, leverage={self.leverage}x, notional=${notional_value:.2f}, contract_value=${contract_size:.2f}, contracts={base_size:.2f}")
            elif "SUI" in execution_product:
                # SUI futures: 500 SUI per contract (based on Coinbase UI)
                # At $1.55/SUI, 1 contract = $775 notional
                # Max leverage is 4x, so min margin ~$194 per contract
                contract_size = current_price * 500  # 500 SUI per contract
                base_size = notional_value / contract_size
                logger.info(f"SUI futures: margin=${position_size:.2f}, leverage={self.leverage}x, notional=${notional_value:.2f}, contract_value=${contract_size:.2f}, contracts={base_size:.2f}")
                
                # SUI has 4x max leverage, warn if trying to use more
                if self.leverage > 4:
                    logger.warning(f"SUI futures max leverage is 4x, but configured for {self.leverage}x - position will be smaller")
            else:
                # Default: assume 1 unit per contract
                contract_size = current_price
                base_size = notional_value / contract_size
                logger.info(f"Other futures: margin=${position_size:.2f}, leverage={self.leverage}x, notional=${notional_value:.2f}, contract_value=${contract_size:.2f}, contracts={base_size:.2f}")
            
            # Futures contracts are whole numbers
            base_size = int(round(base_size))
            
            if base_size < 1:
                logger.warning(f"Margin ${position_size:.2f} x {self.leverage}x leverage = ${notional_value:.2f} too small for {execution_product} (min 1 contract = ${contract_size:.2f})")
                return ExecutionResult(
                    success=False,
                    error=f"Order size too small for futures (need at least 1 contract worth ${contract_size:.2f}, have ${notional_value:.2f} notional)"
                )
        else:
            # Spot: size is in base currency
            base_size = position_size / current_price
            
            # Round to appropriate precision
            precision = self._get_size_precision(symbol)
            base_size = round(base_size, precision)
            
            if base_size == 0:
                return ExecutionResult(
                    success=False,
                    error="Order size too small after rounding"
                )
        
        # Execute the order
        try:
            order_side = "BUY" if direction == "LONG" else "SELL"
            
            # Paper trading mode - simulate order without executing
            if self.paper_trading:
                import uuid
                order = OrderResponse(
                    order_id=f"paper-{uuid.uuid4().hex[:8]}",
                    product_id=execution_product,
                    side=order_side,
                    size=base_size,
                    price=current_price,
                    status="FILLED",
                    created_at=datetime.now(timezone.utc),
                    filled_size=base_size,
                    average_fill_price=current_price
                )
                contract_type = "contracts" if is_futures else symbol.split("-")[0]
                logger.info(f"[PAPER] Would {order_side} {base_size} {contract_type} on {execution_product} @ ${current_price:.2f}")
            else:
                contract_type = "contracts" if is_futures else symbol.split("-")[0]
                logger.info(f"[LIVE] Submitting {order_side} order: {base_size} {contract_type} on {execution_product} @ ~${current_price:.2f}")
                order = await self.client.create_order(
                    product_id=execution_product,
                    side=order_side,
                    size=base_size,
                    order_type="MARKET"
                )
                logger.info(f"[LIVE] Order response: id={order.order_id}, status={order.status}")
            
            if order.status in ["FILLED", "PENDING"]:
                # Create trade record
                trade = Trade(
                    trade_id=order.order_id,
                    symbol=symbol,
                    side=direction,
                    entry_price=order.average_fill_price or current_price,
                    entry_size=base_size,
                    entry_time=datetime.now(timezone.utc),
                    entry_order_id=order.order_id,
                    execution_product=execution_product,  # Track actual product used
                    strategy=signal.contributing_signals[0].strategy if signal.contributing_signals else "aggregated",
                    signal_confidence=signal.confidence
                )
                
                # Calculate stop loss
                stop_price = self._calculate_stop_loss(
                    entry_price=trade.entry_price,
                    direction=direction
                )
                trade.stop_loss = stop_price
                
                # Register with risk manager
                self.risk_manager.register_position(
                    symbol=symbol,
                    entry_price=trade.entry_price,
                    side=direction
                )
                
                # Register with liquidation guard
                self.liquidation_guard.add_position(
                    symbol=symbol,
                    entry_price=trade.entry_price,
                    size=base_size,
                    side=direction
                )
                
                # Store active trade
                self.active_trades[symbol] = trade
                
                # Update stats
                self.stats["total_trades"] += 1
                self.stats["total_volume"] += position_size
                
                logger.info(
                    f"TRADE OPENED: {direction} {symbol} @ {trade.entry_price:.2f}, "
                    f"size={base_size:.6f}, stop={stop_price:.2f}"
                )
                
                return ExecutionResult(
                    success=True,
                    trade=trade,
                    order=order,
                    metadata={
                        "position_size_usd": position_size,
                        "sizing_details": sizing_details,
                        "risk_multiplier": risk_multiplier
                    }
                )
            else:
                return ExecutionResult(
                    success=False,
                    order=order,
                    error=f"Order not filled: {order.status}"
                )
                
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e)
            )
    
    async def close_position(
        self,
        symbol: str,
        reason: str,
        size: Optional[float] = None
    ) -> ExecutionResult:
        """
        Close an open position
        
        Args:
            symbol: Symbol to close
            reason: Reason for closing
            size: Optional partial size (None = close all)
        
        Returns:
            ExecutionResult
        """
        if symbol not in self.active_trades:
            return ExecutionResult(
                success=False,
                error=f"No active trade for {symbol}"
            )
        
        trade = self.active_trades[symbol]
        close_size = size or trade.entry_size
        
        # Use the same product that was used to open the position
        # If it was opened on futures, close on futures
        # If it was opened on spot, close on spot
        close_product = trade.execution_product or symbol
        
        try:
            # Execute closing order
            order_side = "SELL" if trade.side == "LONG" else "BUY"
            
            # Paper trading mode - simulate order without executing
            if self.paper_trading:
                import uuid
                # Use current price (would need to be passed in for accurate sim)
                exit_price = trade.entry_price  # Simplified for paper trading
                order = OrderResponse(
                    order_id=f"paper-{uuid.uuid4().hex[:8]}",
                    product_id=close_product,
                    side=order_side,
                    size=close_size,
                    price=exit_price,
                    status="FILLED",
                    created_at=datetime.now(timezone.utc),
                    filled_size=close_size,
                    average_fill_price=exit_price
                )
                logger.info(f"[PAPER] Would {order_side} {close_size:.6f} {close_product} @ ${exit_price:.2f}")
            else:
                logger.info(f"[LIVE] Closing position: {order_side} {close_size} on {close_product}")
                order = await self.client.create_order(
                    product_id=close_product,
                    side=order_side,
                    size=close_size,
                    order_type="MARKET"
                )
            
            if order.status in ["FILLED", "PENDING"]:
                exit_price = order.average_fill_price or order.price
                
                # CRITICAL FIX: If we don't have a fill price, get market price
                # Market orders fill immediately but the API may not return fill price
                if not exit_price or exit_price == 0:
                    try:
                        # Get current market price as fallback
                        ticker = await self.client.get_ticker(symbol)
                        exit_price = ticker.get("price", 0)
                        logger.warning(f"Using market price for exit: ${exit_price:.2f} (fill price not returned)")
                        
                        # If still no price, try to get from fills
                        if not exit_price:
                            import asyncio
                            await asyncio.sleep(0.5)  # Brief delay for fill to register
                            fills = await self.client.get_fills(close_product, limit=1)
                            if fills:
                                exit_price = float(fills[0].get("price", 0))
                                logger.info(f"Got exit price from fills: ${exit_price:.2f}")
                    except Exception as price_error:
                        logger.error(f"Could not get exit price: {price_error}")
                        # Last resort: use entry price (prevents massive fake P&L)
                        exit_price = trade.entry_price
                        logger.warning(f"Using entry price as fallback: ${exit_price:.2f}")
                
                # Sanity check: exit_price should be reasonable (within 20% of entry)
                if exit_price > 0:
                    price_diff_pct = abs(exit_price - trade.entry_price) / trade.entry_price * 100
                    if price_diff_pct > 50:
                        logger.error(f"Exit price ${exit_price:.2f} seems wrong (entry was ${trade.entry_price:.2f}, diff={price_diff_pct:.1f}%)")
                        exit_price = trade.entry_price  # Use entry as fallback
                        logger.warning(f"Using entry price to prevent erroneous P&L")
                
                # Calculate P&L
                pnl = trade.close(exit_price, close_size, reason)
                
                # Update risk manager
                self.risk_manager.close_position(symbol, pnl)
                
                # Remove from liquidation guard
                self.liquidation_guard.remove_position(symbol)
                
                # Move to history
                self.trade_history.append(trade)
                del self.active_trades[symbol]
                
                # Update stats
                self.stats["total_pnl"] += pnl
                if pnl >= 0:
                    self.stats["winning_trades"] += 1
                else:
                    self.stats["losing_trades"] += 1
                
                logger.info(
                    f"TRADE CLOSED: {trade.side} {symbol} @ {exit_price:.2f}, "
                    f"P&L=${pnl:.2f}, reason={reason}" +
                    (f" (via {close_product})" if close_product != symbol else "")
                )
                
                return ExecutionResult(
                    success=True,
                    trade=trade,
                    order=order,
                    metadata={"pnl": pnl, "reason": reason}
                )
            else:
                return ExecutionResult(
                    success=False,
                    order=order,
                    error=f"Close order not filled: {order.status}"
                )
                
        except Exception as e:
            logger.error(f"Close position failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e)
            )
    
    async def check_stop_losses(
        self,
        current_prices: Dict[str, float]
    ) -> List[ExecutionResult]:
        """
        Check and execute stop losses for all positions
        
        Returns:
            List of execution results for any triggered stops
        """
        results = []
        
        # Log position status for each active trade
        if self.active_trades:
            logger.info("=" * 50)
            logger.info("POSITION STATUS CHECK")
            logger.info("=" * 50)
            
            for symbol, trade in self.active_trades.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    entry_price = trade.entry_price
                    size = trade.entry_size
                    side = trade.side
                    
                    # Calculate P&L
                    if side == "LONG":
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        pnl_usd = (current_price - entry_price) * size
                    else:
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                        pnl_usd = (entry_price - current_price) * size
                    
                    # Get stop price from risk manager
                    stop_price = self.risk_manager.get_stop_price(symbol)
                    
                    # Calculate distance to stop
                    if stop_price and side == "LONG":
                        stop_distance = ((current_price - stop_price) / current_price) * 100
                    elif stop_price and side == "SHORT":
                        stop_distance = ((stop_price - current_price) / current_price) * 100
                    else:
                        stop_distance = 0
                    
                    # Status emoji
                    if pnl_pct >= 10:
                        status = "🚀"
                    elif pnl_pct >= 5:
                        status = "🟢"
                    elif pnl_pct >= 0:
                        status = "🟡"
                    elif pnl_pct >= -3:
                        status = "🟠"
                    else:
                        status = "🔴"
                    
                    logger.info(
                        f"{status} {symbol} {side}: "
                        f"entry=${entry_price:.2f}, "
                        f"current=${current_price:.2f}, "
                        f"P&L={pnl_pct:+.2f}% (${pnl_usd:+.2f}), "
                        f"stop=${stop_price:.2f} ({stop_distance:.1f}% away)"
                    )
            
            logger.info("=" * 50)
        
        # Update prices in risk manager and check stops
        sides = {s: t.side for s, t in self.active_trades.items()}
        stopped_symbols = self.risk_manager.update_prices(current_prices, sides)
        
        # Close stopped positions
        for symbol in stopped_symbols:
            if symbol in self.active_trades:
                logger.warning(f"🛑 STOP LOSS TRIGGERED for {symbol}!")
                result = await self.close_position(symbol, "stop_loss")
                results.append(result)
        
        return results
    
    async def check_take_profits(
        self,
        current_prices: Dict[str, float]
    ) -> List[ExecutionResult]:
        """
        Check and execute take profits for all positions
        
        Returns:
            List of execution results for any triggered TPs
        """
        results = []
        
        # Build position data for risk manager
        positions = {}
        for symbol, trade in self.active_trades.items():
            if symbol in current_prices:
                positions[symbol] = {
                    "entry_price": trade.entry_price,
                    "current_price": current_prices[symbol],
                    "size": trade.entry_size,
                    "side": trade.side
                }
        
        # Check take profits
        tp_triggers = self.risk_manager.check_take_profits(positions)
        
        for symbol, size_to_close, threshold in tp_triggers:
            if symbol in self.active_trades:
                result = await self.close_position(
                    symbol, 
                    f"take_profit_{threshold}%",
                    size=size_to_close
                )
                results.append(result)
        
        return results
    
    async def check_liquidation_risk(
        self,
        current_prices: Dict[str, float]
    ) -> List[ExecutionResult]:
        """
        Check liquidation risk and close positions if needed
        
        Returns:
            List of execution results for emergency closes
        """
        results = []
        
        # Get positions that need emergency close
        to_close = self.liquidation_guard.get_positions_to_close(current_prices)
        
        for symbol in to_close:
            if symbol in self.active_trades:
                logger.warning(f"EMERGENCY CLOSE: {symbol} - liquidation risk")
                result = await self.close_position(symbol, "liquidation_risk")
                results.append(result)
        
        return results
    
    def _calculate_stop_loss(
        self,
        entry_price: float,
        direction: str
    ) -> float:
        """Calculate stop loss price"""
        stop_percent = self.risk_manager.trailing_stops.trailing_percent / 100
        
        if direction == "LONG":
            return entry_price * (1 - stop_percent)
        else:
            return entry_price * (1 + stop_percent)
    
    def update_unrealized_pnl(
        self,
        current_prices: Dict[str, float]
    ) -> float:
        """
        Update unrealized P&L for all positions
        
        Returns:
            Total unrealized P&L
        """
        total_unrealized = 0.0
        
        for symbol, trade in self.active_trades.items():
            if symbol in current_prices:
                unrealized = trade.calculate_pnl(current_prices[symbol])
                total_unrealized += unrealized
        
        return total_unrealized
    
    def get_active_positions(self) -> List[Dict]:
        """Get summary of all active positions"""
        return [
            {
                "symbol": trade.symbol,
                "side": trade.side,
                "entry_price": trade.entry_price,
                "entry_size": trade.entry_size,
                "unrealized_pnl": trade.unrealized_pnl,
                "entry_time": trade.entry_time.isoformat(),
                "strategy": trade.strategy,
                "stop_loss": trade.stop_loss
            }
            for trade in self.active_trades.values()
        ]
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        total = self.stats["total_trades"]
        winners = self.stats["winning_trades"]
        
        return {
            "total_trades": total,
            "winning_trades": winners,
            "losing_trades": self.stats["losing_trades"],
            "win_rate": (winners / total * 100) if total > 0 else 0,
            "total_pnl": self.stats["total_pnl"],
            "total_volume": self.stats["total_volume"],
            "active_positions": len(self.active_trades),
            "avg_pnl_per_trade": self.stats["total_pnl"] / total if total > 0 else 0
        }
    
    async def sync_with_exchange(self) -> None:
        """
        Synchronize local state with exchange
        
        Fetches actual positions (both spot and futures) and reconciles with local tracking
        """
        try:
            # Get futures positions
            exchange_positions = await self.client.get_futures_positions()
            
            # Get spot holdings
            accounts = await self.client.get_accounts()
            spot_holdings = {}
            
            for acc in accounts:
                currency = acc.get("currency", "")
                balance = float(acc.get("available_balance", {}).get("value", 0))
                
                # Check for meaningful spot holdings (not just dust)
                if balance > 0 and currency in ["BTC", "ETH", "SOL", "SUI", "ZK", "DNT"]:
                    symbol = f"{currency}-USD"
                    try:
                        ticker = await self.client.get_ticker(symbol)
                        price = float(ticker.get("price", 0))
                        value = balance * price
                        
                        # Only track if worth more than $10
                        if value > 10:
                            spot_holdings[symbol] = {
                                "size": balance,
                                "price": price,
                                "value": value
                            }
                    except:
                        pass
            
            # Check for spot positions not being tracked
            for symbol, holding in spot_holdings.items():
                if symbol not in self.active_trades:
                    logger.info(f"Found untracked spot position: {symbol} - {holding['size']:.6f} @ ${holding['price']:.2f} (${holding['value']:.2f})")
                    
                    # Create a trade entry to track it
                    trade = Trade(
                        symbol=symbol,
                        side="LONG",
                        entry_price=holding["price"],  # Use current price as estimate
                        entry_size=holding["size"],
                        stop_loss=holding["price"] * 0.94,  # 6% stop
                        execution_product=symbol
                    )
                    trade.status = "open"
                    self.active_trades[symbol] = trade
                    
                    # Register with risk manager
                    self.risk_manager.register_position(symbol, holding["price"], "LONG")
                    self.liquidation_guard.add_position(symbol, holding["price"], holding["size"], "LONG")
                    
                    logger.info(f"Now tracking {symbol}: size={holding['size']:.6f}, stop=${trade.stop_loss:.2f}")
            
            # Handle futures positions (existing logic)
            exchange_symbols = {p.product_id for p in exchange_positions}
            local_symbols = set(self.active_trades.keys())
            
            # Find discrepancies
            futures_missing_local = exchange_symbols - local_symbols
            
            if futures_missing_local:
                logger.warning(f"Futures positions on exchange not tracked locally: {futures_missing_local}")
            
            # Update unrealized P&L from exchange
            for pos in exchange_positions:
                if pos.product_id in self.active_trades:
                    self.active_trades[pos.product_id].unrealized_pnl = pos.unrealized_pnl
            
            logger.info(f"Exchange sync complete: {len(spot_holdings)} spot, {len(exchange_positions)} futures")
            
        except Exception as e:
            logger.error(f"Exchange sync failed: {e}")
