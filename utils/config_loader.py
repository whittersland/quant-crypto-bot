"""
Configuration loader and manager
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Trading configuration container"""
    symbols: list = field(default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD"])
    leverage: int = 2


@dataclass
class PositionSizingConfig:
    """Position sizing configuration"""
    base_min: float = 15.0
    base_max: float = 60.0
    increment: float = 5.0
    max_concurrent_positions: int = 4
    profit_check_hours: int = 24


@dataclass
class RiskConfig:
    """Risk management configuration"""
    trailing_stop_percent: float = 12.0
    daily_loss_limit_percent: float = 15.0
    max_drawdown_percent: float = 25.0
    position_stop_loss_percent: float = 8.0
    take_profit_tiers: list = field(default_factory=list)


@dataclass
class LoopConfig:
    """Trading loop configuration"""
    main_interval_minutes: int = 10
    scanner_interval_minutes: int = 3
    scanner_confidence_threshold: float = 0.25
    main_confidence_threshold: float = 0.15


@dataclass
class StrategyConfig:
    """Strategy-specific configuration"""
    enabled: bool = True
    weight: float = 0.1
    params: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manages system configuration"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._find_config()
        self._raw_config: Dict = {}
        self._load_config()
    
    def _find_config(self) -> str:
        """Find configuration file"""
        possible_paths = [
            "config/settings.yaml",
            "../config/settings.yaml",
            "settings.yaml",
            os.path.expanduser("~/.trading_system/settings.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return "config/settings.yaml"
    
    def _load_config(self) -> None:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self._raw_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                self._raw_config = {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._raw_config = {}
    
    def _resolve_env_vars(self, value: Any) -> Any:
        """Resolve environment variables in config values"""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, "")
        return value
    
    @property
    def api_key(self) -> str:
        """Get API key"""
        creds = self._raw_config.get("credentials", {})
        return self._resolve_env_vars(creds.get("api_key", ""))
    
    @property
    def api_secret(self) -> str:
        """Get API secret"""
        creds = self._raw_config.get("credentials", {})
        return self._resolve_env_vars(creds.get("api_secret", ""))
    
    @property
    def trading(self) -> TradingConfig:
        """Get trading configuration"""
        cfg = self._raw_config.get("trading", {})
        return TradingConfig(
            symbols=cfg.get("symbols", ["BTC-USD", "ETH-USD", "SOL-USD"]),
            leverage=cfg.get("leverage", 2)
        )
    
    @property
    def position_sizing(self) -> PositionSizingConfig:
        """Get position sizing configuration"""
        cfg = self._raw_config.get("position_sizing", {})
        return PositionSizingConfig(
            base_min=cfg.get("base_min", 15.0),
            base_max=cfg.get("base_max", 60.0),
            increment=cfg.get("increment", 5.0),
            max_concurrent_positions=cfg.get("max_concurrent_positions", 4),
            profit_check_hours=cfg.get("profit_check_hours", 24)
        )
    
    @property
    def risk(self) -> RiskConfig:
        """Get risk management configuration"""
        cfg = self._raw_config.get("risk_management", {})
        return RiskConfig(
            trailing_stop_percent=cfg.get("trailing_stop_percent", 12.0),
            daily_loss_limit_percent=cfg.get("daily_loss_limit_percent", 15.0),
            max_drawdown_percent=cfg.get("max_drawdown_percent", 25.0),
            position_stop_loss_percent=cfg.get("position_stop_loss_percent", 8.0),
            take_profit_tiers=cfg.get("take_profit_tiers", [])
        )
    
    @property
    def loops(self) -> LoopConfig:
        """Get loop configuration"""
        cfg = self._raw_config.get("loops", {})
        return LoopConfig(
            main_interval_minutes=cfg.get("main_interval_minutes", 10),
            scanner_interval_minutes=cfg.get("scanner_interval_minutes", 3),
            scanner_confidence_threshold=cfg.get("scanner_confidence_threshold", 0.25),
            main_confidence_threshold=cfg.get("main_confidence_threshold", 0.15)
        )
    
    def get_strategy_config(self, strategy_name: str) -> StrategyConfig:
        """Get configuration for a specific strategy"""
        strategies = self._raw_config.get("strategies", {})
        cfg = strategies.get(strategy_name, {})
        
        return StrategyConfig(
            enabled=cfg.get("enabled", True),
            weight=cfg.get("weight", 0.1),
            params={k: v for k, v in cfg.items() if k not in ["enabled", "weight"]}
        )
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get all strategy weights"""
        strategies = self._raw_config.get("strategies", {})
        weights = {}
        
        for name, cfg in strategies.items():
            if cfg.get("enabled", True):
                weights[name] = cfg.get("weight", 0.1)
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def save(self) -> None:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self._raw_config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def update(self, section: str, key: str, value: Any) -> None:
        """Update a configuration value"""
        if section not in self._raw_config:
            self._raw_config[section] = {}
        self._raw_config[section][key] = value


# Global config instance
_config: Optional[ConfigManager] = None


def get_config(config_path: str = None) -> ConfigManager:
    """Get or create global configuration manager"""
    global _config
    if _config is None or config_path is not None:
        _config = ConfigManager(config_path)
    return _config
