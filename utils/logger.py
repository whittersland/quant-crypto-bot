"""
Logging configuration and utilities
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored console output formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Build message
        formatted = f"{color}[{timestamp}] {record.levelname:8}{reset} | {record.getMessage()}"
        
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


class TradeLogger:
    """Specialized logger for trade events"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.trade_file = self.log_dir / "trades.log"
        self.signal_file = self.log_dir / "signals.log"
        self.performance_file = self.log_dir / "performance.log"
    
    def log_trade(self, trade_data: dict) -> None:
        """Log a trade execution"""
        timestamp = datetime.now().isoformat()
        with open(self.trade_file, 'a') as f:
            f.write(f"{timestamp} | TRADE | {trade_data}\n")
    
    def log_signal(self, signal_data: dict) -> None:
        """Log a generated signal"""
        timestamp = datetime.now().isoformat()
        with open(self.signal_file, 'a') as f:
            f.write(f"{timestamp} | SIGNAL | {signal_data}\n")
    
    def log_performance(self, metrics: dict) -> None:
        """Log performance metrics"""
        timestamp = datetime.now().isoformat()
        with open(self.performance_file, 'a') as f:
            f.write(f"{timestamp} | PERF | {metrics}\n")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size_mb: int = 100,
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        max_size_mb: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Root logger instance
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        ))
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger"""
    return logging.getLogger(name)
