"""
Liquidation Guard Module
Prevents positions from approaching liquidation on leveraged futures
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class LiquidationRisk:
    """Liquidation risk assessment for a position"""
    symbol: str
    entry_price: float
    current_price: float
    liquidation_price: float
    distance_to_liquidation: float  # Percentage
    risk_level: str  # "safe", "warning", "danger", "critical"
    recommended_action: str


class LiquidationGuard:
    """
    Monitors and prevents liquidation on leveraged positions
    
    With 10x leverage:
    - Liquidation occurs at ~10% adverse move (minus fees/funding)
    - We want to exit well before that
    """
    
    # Risk thresholds (distance to liquidation as % of entry)
    SAFE_THRESHOLD = 6.0       # >6% away = safe
    WARNING_THRESHOLD = 4.0    # 4-6% away = warning  
    DANGER_THRESHOLD = 2.5     # 2.5-4% away = danger
    CRITICAL_THRESHOLD = 1.5   # <1.5% away = critical (auto-close)
    
    def __init__(
        self,
        leverage: float = 10.0,
        maintenance_margin: float = 0.03,  # 3% typical for crypto futures
        safety_buffer: float = 0.02        # Extra 2% safety (not percentage of leverage!)
    ):
        self.leverage = leverage
        self.maintenance_margin = maintenance_margin
        self.safety_buffer = safety_buffer
        
        # Calculate liquidation distance as percentage of position
        # With 10x leverage: 100%/10 = 10% move causes liquidation
        # Minus maintenance margin buffer
        self.base_liquidation_distance = (1 / leverage) - maintenance_margin
        
        logger.info(f"LiquidationGuard initialized: {leverage}x leverage, liquidation at ~{self.base_liquidation_distance*100:.1f}% adverse move")
        
        # Tracked positions
        self.positions: Dict[str, Dict] = {}
    
    def calculate_liquidation_price(
        self, 
        entry_price: float, 
        side: str = "LONG"
    ) -> float:
        """
        Calculate estimated liquidation price
        
        For 10x leverage:
        - LONG: liquidation at ~7% below entry (10% - 3% maintenance)
        - SHORT: liquidation at ~7% above entry
        """
        # Don't subtract safety buffer from liquidation distance
        # Safety buffer is for margin calculations, not liquidation price
        liquidation_distance = self.base_liquidation_distance
        
        if side == "LONG":
            return entry_price * (1 - liquidation_distance)
        else:
            return entry_price * (1 + liquidation_distance)
    
    def add_position(
        self, 
        symbol: str, 
        entry_price: float, 
        size: float,
        side: str = "LONG",
        actual_liquidation_price: Optional[float] = None
    ) -> None:
        """Add a position to monitor"""
        
        # Use actual liquidation price if provided, otherwise estimate
        if actual_liquidation_price:
            liq_price = actual_liquidation_price
        else:
            liq_price = self.calculate_liquidation_price(entry_price, side)
        
        self.positions[symbol] = {
            "entry_price": entry_price,
            "size": size,
            "side": side,
            "liquidation_price": liq_price,
            "added_at": datetime.now(timezone.utc)
        }
        
        logger.info(
            f"Liquidation guard tracking {symbol}: entry={entry_price:.2f}, "
            f"liquidation={liq_price:.2f}, side={side}"
        )
    
    def update_position(
        self, 
        symbol: str, 
        current_price: float
    ) -> Optional[LiquidationRisk]:
        """
        Update position with current price and assess risk
        
        Returns:
            LiquidationRisk if position exists, None otherwise
        """
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        entry_price = pos["entry_price"]
        liq_price = pos["liquidation_price"]
        side = pos["side"]
        
        # Calculate distance to liquidation
        if side == "LONG":
            distance = ((current_price - liq_price) / current_price) * 100
        else:
            distance = ((liq_price - current_price) / current_price) * 100
        
        # Determine risk level and action
        if distance > self.SAFE_THRESHOLD:
            risk_level = "safe"
            action = "Hold position"
        elif distance > self.WARNING_THRESHOLD:
            risk_level = "warning"
            action = "Monitor closely"
        elif distance > self.DANGER_THRESHOLD:
            risk_level = "danger"
            action = "Consider reducing position"
        elif distance > self.CRITICAL_THRESHOLD:
            risk_level = "critical"
            action = "REDUCE POSITION IMMEDIATELY"
        else:
            risk_level = "liquidation_imminent"
            action = "EMERGENCY CLOSE - LIQUIDATION IMMINENT"
        
        risk = LiquidationRisk(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            liquidation_price=liq_price,
            distance_to_liquidation=distance,
            risk_level=risk_level,
            recommended_action=action
        )
        
        # Log warnings for dangerous positions
        if risk_level in ["danger", "critical", "liquidation_imminent"]:
            logger.warning(
                f"LIQUIDATION RISK - {symbol}: {distance:.1f}% to liquidation, "
                f"level={risk_level}, action={action}"
            )
        
        return risk
    
    def remove_position(self, symbol: str) -> None:
        """Remove position from monitoring"""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Liquidation guard removed {symbol}")
    
    def check_all_positions(
        self, 
        current_prices: Dict[str, float]
    ) -> List[LiquidationRisk]:
        """
        Check all positions for liquidation risk
        
        Returns:
            List of positions with elevated risk
        """
        risks = []
        
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                risk = self.update_position(symbol, current_prices[symbol])
                if risk and risk.risk_level != "safe":
                    risks.append(risk)
        
        return risks
    
    def get_positions_to_close(
        self, 
        current_prices: Dict[str, float]
    ) -> List[str]:
        """
        Get list of positions that should be closed immediately
        
        Returns:
            List of symbols to close
        """
        to_close = []
        
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                risk = self.update_position(symbol, current_prices[symbol])
                if risk and risk.risk_level in ["critical", "liquidation_imminent"]:
                    to_close.append(symbol)
        
        return to_close
    
    def can_open_position(
        self,
        symbol: str,
        entry_price: float,
        size: float,
        available_margin: float,
        side: str = "LONG"
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be safely opened
        
        Returns:
            Tuple of (can_open, reason)
        """
        # Calculate required margin
        position_value = size * entry_price
        required_margin = position_value / self.leverage
        
        # Check if we have enough margin with buffer
        margin_with_buffer = required_margin * (1 + self.safety_buffer)
        
        if margin_with_buffer > available_margin:
            return False, f"Insufficient margin: need ${margin_with_buffer:.2f}, have ${available_margin:.2f}"
        
        # Calculate liquidation price
        liq_price = self.calculate_liquidation_price(entry_price, side)
        
        # Check if liquidation price is reasonable
        if side == "LONG":
            distance = ((entry_price - liq_price) / entry_price) * 100
        else:
            distance = ((liq_price - entry_price) / entry_price) * 100
        
        # For high leverage, use lower threshold
        min_distance = self.DANGER_THRESHOLD  # 2.5% minimum distance
        
        if distance < min_distance:
            return False, f"Position would start too close to liquidation ({distance:.1f}% < {min_distance}%)"
        
        logger.info(f"Liquidation check OK: {symbol} {side} distance={distance:.1f}% (min={min_distance}%)")
        return True, "OK"
    
    def get_safe_position_size(
        self,
        entry_price: float,
        available_margin: float,
        min_liquidation_distance: float = None
    ) -> float:
        """
        Calculate maximum safe position size given available margin
        
        Args:
            entry_price: Expected entry price
            available_margin: Available margin capital
            min_liquidation_distance: Minimum distance to liquidation (default: WARNING_THRESHOLD)
        
        Returns:
            Maximum safe position size
        """
        if min_liquidation_distance is None:
            min_liquidation_distance = self.WARNING_THRESHOLD
        
        # Available margin after safety buffer
        usable_margin = available_margin * (1 - self.safety_buffer)
        
        # Maximum position value
        max_position_value = usable_margin * self.leverage
        
        # Maximum size
        max_size = max_position_value / entry_price
        
        return max_size
    
    def get_report(self) -> Dict:
        """Get liquidation guard status report"""
        return {
            "leverage": self.leverage,
            "maintenance_margin": self.maintenance_margin,
            "safety_buffer": self.safety_buffer,
            "thresholds": {
                "safe": f">{self.SAFE_THRESHOLD}%",
                "warning": f"{self.WARNING_THRESHOLD}-{self.SAFE_THRESHOLD}%",
                "danger": f"{self.DANGER_THRESHOLD}-{self.WARNING_THRESHOLD}%",
                "critical": f"<{self.DANGER_THRESHOLD}%"
            },
            "tracked_positions": len(self.positions),
            "positions": {
                symbol: {
                    "entry": pos["entry_price"],
                    "liquidation": pos["liquidation_price"],
                    "side": pos["side"]
                }
                for symbol, pos in self.positions.items()
            }
        }
