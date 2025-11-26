"""
Coinbase Advanced Trade API Client
Handles authentication, requests, and WebSocket connections
"""

import os
import time
import hmac
import hashlib
import base64
import json
import asyncio
import aiohttp
import secrets
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


# Try to import JWT library
try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False
    logger.warning("PyJWT not installed. Install with: pip install PyJWT cryptography")


@dataclass
class OrderResponse:
    order_id: str
    product_id: str
    side: str
    size: float
    price: float
    status: str
    created_at: datetime
    filled_size: float = 0.0
    average_fill_price: float = 0.0


@dataclass
class Position:
    product_id: str
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float
    liquidation_price: Optional[float] = None


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class CoinbaseClient:
    """Coinbase Advanced Trade API Client with full trading capabilities"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.getenv("COINBASE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET", "")
        self.base_url = "https://api.coinbase.com"
        self.ws_url = "wss://advanced-trade-ws.coinbase.com"
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Determine auth method based on key format
        self._use_jwt = "-----BEGIN" in self.api_secret or self.api_key.startswith("organizations/")
        
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _build_jwt(self, method: str, path: str) -> str:
        """Build JWT token for Coinbase Advanced Trade API"""
        if not HAS_JWT:
            raise ImportError("PyJWT library required. Install with: pip install PyJWT cryptography")
        
        # Parse the key name from api_key (format: organizations/{org_id}/apiKeys/{key_id})
        key_name = self.api_key
        
        # Remove query params from path for JWT URI (JWT should only have the base path)
        base_path = path.split("?")[0]
        
        # JWT payload
        uri = f"{method} api.coinbase.com{base_path}"
        
        payload = {
            "sub": key_name,
            "iss": "coinbase-cloud",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,  # 2 minute expiry
            "uri": uri,
        }
        
        # JWT headers
        headers = {
            "kid": key_name,
            "nonce": secrets.token_hex(16),
            "typ": "JWT",
            "alg": "ES256"
        }
        
        # Sign with EC private key
        token = jwt.encode(
            payload,
            self.api_secret,
            algorithm="ES256",
            headers=headers
        )
        
        return token
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC signature for legacy authentication"""
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode("utf-8"),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode("utf-8")
    
    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate authenticated headers"""
        if self._use_jwt:
            # JWT authentication (new method)
            token = self._build_jwt(method, path)
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        else:
            # Legacy HMAC authentication
            timestamp = str(int(time.time()))
            signature = self._generate_signature(timestamp, method, path, body)
            return {
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "Content-Type": "application/json"
            }
    
    async def _request(self, method: str, path: str, data: Dict = None) -> Dict:
        """Make authenticated API request"""
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        body = json.dumps(data) if data else ""
        headers = self._get_headers(method, path, body)
        
        try:
            async with session.request(method, url, headers=headers, data=body or None) as response:
                response_data = await response.json()
                
                if response.status >= 400:
                    logger.error(f"API Error {response.status}: {response_data}")
                    raise Exception(f"API Error: {response_data}")
                
                return response_data
                
        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise
    
    # Market Data Methods
    async def get_products(self) -> List[Dict]:
        """Get all available trading products"""
        response = await self._request("GET", "/api/v3/brokerage/products")
        return response.get("products", [])
    
    async def get_product(self, product_id: str) -> Dict:
        """Get specific product details"""
        response = await self._request("GET", f"/api/v3/brokerage/products/{product_id}")
        return response
    
    async def _public_request(self, url: str) -> Dict:
        """Make unauthenticated request to public endpoint"""
        session = await self._get_session()
        try:
            async with session.get(url) as response:
                if response.status >= 400:
                    text = await response.text()
                    logger.error(f"Public API Error {response.status}: {text}")
                    raise Exception(f"Public API Error: {response.status}")
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Public request failed: {e}")
            raise
    
    async def get_candles(
        self, 
        product_id: str, 
        granularity: str = "ONE_MINUTE",
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 300
    ) -> List[Candle]:
        """
        Get historical candles
        
        Granularity options:
        - ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE
        - THIRTY_MINUTE, ONE_HOUR, TWO_HOUR
        - SIX_HOUR, ONE_DAY
        """
        # Try authenticated endpoint first
        path = f"/api/v3/brokerage/products/{product_id}/candles"
        params = [f"granularity={granularity}"]
        
        if start:
            params.append(f"start={start}")
        if end:
            params.append(f"end={end}")
        if limit:
            params.append(f"limit={limit}")
            
        if params:
            path += "?" + "&".join(params)
        
        try:
            response = await self._request("GET", path)
        except Exception as e:
            # Fallback to public Exchange API
            logger.info(f"Falling back to public API for {product_id} candles")
            granularity_map = {
                "ONE_MINUTE": 60,
                "FIVE_MINUTE": 300,
                "FIFTEEN_MINUTE": 900,
                "THIRTY_MINUTE": 1800,
                "ONE_HOUR": 3600,
                "TWO_HOUR": 7200,
                "SIX_HOUR": 21600,
                "ONE_DAY": 86400
            }
            gran_seconds = granularity_map.get(granularity, 60)
            
            # Use Coinbase Exchange public API
            public_url = f"https://api.exchange.coinbase.com/products/{product_id}/candles?granularity={gran_seconds}"
            response = await self._public_request(public_url)
            
            # Public API returns array of arrays: [time, low, high, open, close, volume]
            candles = []
            for c in response[:limit] if isinstance(response, list) else []:
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(int(c[0]), tz=timezone.utc),
                    open=float(c[3]),
                    high=float(c[2]),
                    low=float(c[1]),
                    close=float(c[4]),
                    volume=float(c[5])
                ))
            return sorted(candles, key=lambda x: x.timestamp)
        
        candles = []
        
        for c in response.get("candles", []):
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(int(c["start"]), tz=timezone.utc),
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                volume=float(c["volume"])
            ))
        
        return sorted(candles, key=lambda x: x.timestamp)
    
    async def get_ticker(self, product_id: str) -> Dict:
        """Get current ticker for a product"""
        response = await self._request("GET", f"/api/v3/brokerage/products/{product_id}/ticker")
        return response
    
    # Account Methods
    async def get_accounts(self) -> List[Dict]:
        """Get all accounts"""
        response = await self._request("GET", "/api/v3/brokerage/accounts")
        return response.get("accounts", [])
    
    async def get_account(self, account_id: str) -> Dict:
        """Get specific account"""
        response = await self._request("GET", f"/api/v3/brokerage/accounts/{account_id}")
        return response.get("account", {})
    
    # Futures-specific Methods
    async def get_futures_balance_summary(self) -> Dict:
        """Get futures account balance summary"""
        response = await self._request("GET", "/api/v3/brokerage/cfm/balance_summary")
        return response
    
    async def get_futures_products(self) -> List[Dict]:
        """
        Get available futures products
        
        Returns list of futures contracts like BIT-26DEC25-CDE (nano Bitcoin)
        """
        try:
            response = await self._request(
                "GET", 
                "/api/v3/brokerage/products?product_type=FUTURE"
            )
            products = response.get("products", [])
            
            # Filter to only non-disabled futures
            active_futures = [
                p for p in products 
                if not p.get("trading_disabled", False)
            ]
            
            logger.info(f"Found {len(active_futures)} active futures products (of {len(products)} total)")
            
            if active_futures:
                sample = [p.get("product_id") for p in active_futures[:5]]
                logger.info(f"Sample futures: {sample}")
            
            return active_futures
            
        except Exception as e:
            logger.error(f"Failed to fetch futures products: {e}")
            return []
    
    async def get_best_futures_contract(self, base_asset: str) -> Optional[str]:
        """
        Get the best (nearest expiry, most liquid) futures contract for a base asset
        
        Args:
            base_asset: "BTC", "ETH", etc.
        
        Returns:
            Product ID like "BIT-26DEC25-CDE" or None if not found
        """
        futures = await self.get_futures_products()
        
        # Map base assets to futures product codes
        # Based on Coinbase Derivatives available contracts (as of late 2025):
        # - BIT = nano Bitcoin (1/100 BTC)
        # - ET = nano Ether (1/10 ETH)
        # - SOL = nano Solana (5 SOL per contract)
        # - SLC = Solana (100 SOL per contract)
        # - DOG = Dogecoin futures
        # - XRP = XRP futures
        # - ADA = Cardano futures
        # - AVAX = Avalanche futures
        # - LINK = Chainlink futures
        # - DOT = Polkadot futures
        # - XLM = Stellar futures
        # - SHIB = Shiba Inu futures
        # - LTC = Litecoin futures
        # - BCH = Bitcoin Cash futures
        # - SUI = SUI futures
        # - HBAR/HED = Hedera futures
        code_map = {
            "BTC": ["BIT", "BTC"],      # nano Bitcoin preferred
            "ETH": ["ET", "ETH"],        # nano Ether preferred
            "SOL": ["SOL", "SLC"],       # nano SOL (5 SOL) preferred over SLC (100 SOL)
            "DOGE": ["DOG", "DOGE"],
            "XRP": ["XRP"],
            "ADA": ["ADA"],
            "AVAX": ["AVAX", "AVA"],
            "LINK": ["LINK"],
            "DOT": ["DOT"],
            "XLM": ["XLM"],
            "SHIB": ["SHIB"],
            "LTC": ["LTC"],
            "BCH": ["BCH"],
            "SUI": ["SUI"],
            "HBAR": ["HED", "HBAR"],
            "MATIC": ["MATIC", "POL"],
            "UNI": ["UNI"],
            "ATOM": ["ATOM"],
            "NEAR": ["NEAR"],
            "APT": ["APT"],
            "OP": ["OP"],
            "ARB": ["ARB"],
        }
        
        target_codes = code_map.get(base_asset.upper(), [base_asset.upper()])
        
        # Find matching contracts for any of the possible codes
        for target_code in target_codes:
            matching = [
                p for p in futures
                if p.get("product_id", "").startswith(target_code + "-")
                or p.get("product_id", "").startswith(target_code + ".")  # Some use dot notation
            ]
            
            if matching:
                # Sort by expiry (nearest first) - product_id format: BIT-26DEC25-CDE
                # For now, just take the first one (usually nearest expiry)
                best = matching[0]
                product_id = best.get("product_id")
                
                logger.info(f"Best futures contract for {base_asset}: {product_id}")
                return product_id
        
        logger.warning(f"No futures contracts found for {base_asset} (tried codes: {target_codes})")
        return None
    
    async def get_futures_positions(self) -> List[Position]:
        """Get all open futures positions"""
        response = await self._request("GET", "/api/v3/brokerage/cfm/positions")
        positions = []
        
        for p in response.get("positions", []):
            positions.append(Position(
                product_id=p.get("product_id", ""),
                side=p.get("side", ""),
                size=float(p.get("number_of_contracts", 0)),
                entry_price=float(p.get("avg_entry_price", 0)),
                unrealized_pnl=float(p.get("unrealized_pnl", 0)),
                liquidation_price=float(p.get("liquidation_price", 0)) if p.get("liquidation_price") else None
            ))
        
        return positions
    
    # Order Methods
    async def create_order(
        self,
        product_id: str,
        side: str,
        size: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """
        Create a new order
        
        Args:
            product_id: Trading pair (e.g., "BTC-USD")
            side: "BUY" or "SELL"
            size: Order size in base currency
            order_type: "MARKET", "LIMIT", "STOP_LIMIT"
            limit_price: Required for LIMIT orders
            stop_price: Required for STOP orders
            client_order_id: Optional client-specified ID
        """
        order_config = {}
        
        if order_type == "MARKET":
            order_config["market_market_ioc"] = {
                "base_size": str(size)
            }
        elif order_type == "LIMIT":
            order_config["limit_limit_gtc"] = {
                "base_size": str(size),
                "limit_price": str(limit_price)
            }
        elif order_type == "STOP_LIMIT":
            order_config["stop_limit_stop_limit_gtc"] = {
                "base_size": str(size),
                "limit_price": str(limit_price),
                "stop_price": str(stop_price)
            }
        
        data = {
            "client_order_id": client_order_id or f"order_{int(time.time() * 1000)}",
            "product_id": product_id,
            "side": side,
            "order_configuration": order_config
        }
        
        logger.info(f"Creating order: {side} {size} {product_id}")
        response = await self._request("POST", "/api/v3/brokerage/orders", data)
        
        # Check for error response
        if "error_response" in response:
            error = response.get("error_response", {})
            error_msg = error.get("message", "Unknown error")
            preview_reason = error.get("preview_failure_reason", "")
            
            # Provide actionable guidance for common errors
            if "INSUFFICIENT_FUNDS" in preview_reason or "INSUFFICIENT_FUNDS" in error_msg:
                if "FUTURES" in preview_reason:
                    logger.error(f"Order failed: Insufficient futures balance")
                    logger.error(f"→ Go to Coinbase → Futures → Transfer funds from spot wallet")
                else:
                    logger.error(f"Order failed: Insufficient funds")
            else:
                logger.error(f"Order failed: {error_msg}")
            
            logger.error(f"Full response: {response}")
            raise Exception(f"Order failed: {error_msg}")
        
        order = response.get("success_response", {})
        if not order:
            logger.error(f"No success_response in order response: {response}")
            raise Exception(f"Unexpected order response: {response}")
            
        logger.info(f"Order created: {order.get('order_id')} - {order.get('status')}")
        
        return OrderResponse(
            order_id=order.get("order_id", ""),
            product_id=order.get("product_id", product_id),
            side=order.get("side", side),
            size=size,
            price=limit_price or 0,
            status=order.get("status", "PENDING"),
            created_at=datetime.now(timezone.utc),
            filled_size=0,
            average_fill_price=0
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        data = {"order_ids": [order_id]}
        response = await self._request("POST", "/api/v3/brokerage/orders/batch_cancel", data)
        results = response.get("results", [])
        return len(results) > 0 and results[0].get("success", False)
    
    async def get_order(self, order_id: str) -> Dict:
        """Get order details"""
        response = await self._request("GET", f"/api/v3/brokerage/orders/historical/{order_id}")
        return response.get("order", {})
    
    async def get_open_orders(self, product_id: Optional[str] = None) -> List[Dict]:
        """Get all open orders"""
        path = "/api/v3/brokerage/orders/historical/batch?order_status=OPEN"
        if product_id:
            path += f"&product_id={product_id}"
        response = await self._request("GET", path)
        return response.get("orders", [])
    
    async def get_fills(self, product_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get order fills"""
        path = f"/api/v3/brokerage/orders/historical/fills?limit={limit}"
        if product_id:
            path += f"&product_id={product_id}"
        response = await self._request("GET", path)
        return response.get("fills", [])


class CoinbaseWebSocket:
    """WebSocket client for real-time market data"""
    
    def __init__(self, client: CoinbaseClient):
        self.client = client
        self.ws_url = "wss://advanced-trade-ws.coinbase.com"
        self._ws = None
        self._subscriptions = set()
        self._callbacks = {}
        self._running = False
    
    def _generate_ws_signature(self, timestamp: str, channel: str, products: List[str]) -> str:
        """Generate signature for WebSocket authentication"""
        message = f"{timestamp}{channel}{','.join(products)}"
        signature = hmac.new(
            base64.b64decode(self.client.api_secret),
            message.encode("utf-8"),
            hashlib.sha256
        )
        return signature.hexdigest()
    
    async def connect(self):
        """Establish WebSocket connection"""
        session = await self.client._get_session()
        self._ws = await session.ws_connect(self.ws_url)
        self._running = True
        logger.info("WebSocket connected")
    
    async def subscribe(self, channel: str, products: List[str], callback: callable):
        """Subscribe to a channel"""
        if not self._ws:
            await self.connect()
        
        timestamp = str(int(time.time()))
        signature = self._generate_ws_signature(timestamp, channel, products)
        
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": products,
            "channel": channel,
            "api_key": self.client.api_key,
            "timestamp": timestamp,
            "signature": signature
        }
        
        await self._ws.send_json(subscribe_msg)
        
        for product in products:
            key = f"{channel}:{product}"
            self._subscriptions.add(key)
            self._callbacks[key] = callback
        
        logger.info(f"Subscribed to {channel} for {products}")
    
    async def listen(self):
        """Listen for WebSocket messages"""
        if not self._ws:
            raise Exception("WebSocket not connected")
        
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                channel = data.get("channel", "")
                
                for event in data.get("events", []):
                    for product_id in self._subscriptions:
                        if product_id.startswith(channel):
                            callback = self._callbacks.get(product_id)
                            if callback:
                                await callback(event)
            
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {msg.data}")
                break
    
    async def close(self):
        """Close WebSocket connection"""
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("WebSocket closed")


# Simulated client for testing without API keys
class SimulatedCoinbaseClient(CoinbaseClient):
    """Simulated client for backtesting and paper trading"""
    
    def __init__(self, initial_balance: float = 360.65):
        super().__init__("simulated", "simulated")
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, OrderResponse] = {}
        self._simulated_prices = {
            "BTC-USD": 67500.0,
            "ETH-USD": 3450.0,
            "SOL-USD": 145.0
        }
    
    async def get_futures_balance_summary(self) -> Dict:
        return {
            "balance": str(self.balance),
            "available_balance": str(self.balance - self._get_margin_used()),
            "unrealized_pnl": str(sum(p.unrealized_pnl for p in self.positions.values()))
        }
    
    def _get_margin_used(self) -> float:
        return sum(p.size * p.entry_price / 2 for p in self.positions.values())
    
    async def get_futures_positions(self) -> List[Position]:
        return list(self.positions.values())
    
    async def get_candles(self, product_id: str, **kwargs) -> List[Candle]:
        """Generate simulated candle data"""
        import random
        candles = []
        base_price = self._simulated_prices.get(product_id, 100.0)
        current_time = datetime.now(timezone.utc)
        
        for i in range(300):
            noise = random.gauss(0, 0.002)
            trend = 0.0001 * (150 - i)
            
            open_price = base_price * (1 + noise + trend)
            close_price = open_price * (1 + random.gauss(0, 0.001))
            high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, 0.0005)))
            low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, 0.0005)))
            volume = random.uniform(100, 1000)
            
            candles.append(Candle(
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            ))
            current_time = datetime.fromtimestamp(
                current_time.timestamp() - 60, 
                tz=timezone.utc
            )
            base_price = close_price
        
        return sorted(candles, key=lambda x: x.timestamp)
    
    async def create_order(self, product_id: str, side: str, size: float, **kwargs) -> OrderResponse:
        """Simulate order execution"""
        price = self._simulated_prices.get(product_id, 100.0)
        order_id = f"sim_{int(time.time() * 1000)}"
        
        order = OrderResponse(
            order_id=order_id,
            product_id=product_id,
            side=side,
            size=size,
            price=price,
            status="FILLED",
            created_at=datetime.now(timezone.utc),
            filled_size=size,
            average_fill_price=price
        )
        
        self.orders[order_id] = order
        
        # Update positions
        cost = size * price
        if side == "BUY":
            if product_id in self.positions:
                pos = self.positions[product_id]
                new_size = pos.size + size
                new_entry = (pos.entry_price * pos.size + price * size) / new_size
                pos.size = new_size
                pos.entry_price = new_entry
            else:
                self.positions[product_id] = Position(
                    product_id=product_id,
                    side="LONG",
                    size=size,
                    entry_price=price,
                    unrealized_pnl=0
                )
            self.balance -= cost / 2  # 2x leverage
        else:
            if product_id in self.positions:
                pos = self.positions[product_id]
                pos.size -= size
                if pos.size <= 0:
                    del self.positions[product_id]
                self.balance += cost / 2
        
        logger.info(f"Simulated {side} order: {size} {product_id} @ {price}")
        return order
