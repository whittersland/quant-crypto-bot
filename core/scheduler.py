"""
Dual-Loop Scheduler
Manages the 10-minute main trading loop and 3-minute scanner loop
"""

import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LoopType(Enum):
    """Loop type identifiers"""
    MAIN = "main"           # Full 10-minute trading cycle
    SCANNER = "scanner"     # Fast 3-minute scanner


@dataclass
class LoopStats:
    """Statistics for a loop"""
    loop_type: LoopType
    iterations: int = 0
    signals_generated: int = 0
    trades_executed: int = 0
    errors: int = 0
    last_run: Optional[datetime] = None
    avg_duration_ms: float = 0.0
    total_duration_ms: float = 0.0


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""
    main_interval_minutes: int = 10
    scanner_interval_minutes: int = 3
    main_confidence_threshold: float = 0.15
    scanner_confidence_threshold: float = 0.25  # Higher threshold for scanner
    max_consecutive_errors: int = 5
    error_cooldown_seconds: int = 60


class DualLoopScheduler:
    """
    Dual-Loop Scheduler
    
    Manages two concurrent loops:
    
    1. MAIN LOOP (every 10 minutes):
       - Full market data fetch
       - Run ALL strategies
       - Signal aggregation
       - Trade execution
       - Portfolio sync
       - Risk checks
       - Adaptive position sizing update
    
    2. SCANNER LOOP (every 3 minutes):
       - Quick market data snapshot
       - Run ONLY dual-condition strategies
       - Execute ONLY if high confidence (>0.25)
       - No full portfolio sync
       - Lightweight risk check
    """
    
    def __init__(self, config: SchedulerConfig = None):
        self.config = config or SchedulerConfig()
        
        # Loop state
        self.running = False
        self.paused = False
        
        # Tasks
        self._main_task: Optional[asyncio.Task] = None
        self._scanner_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._main_callback: Optional[Callable] = None
        self._scanner_callback: Optional[Callable] = None
        self._error_callback: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            LoopType.MAIN: LoopStats(loop_type=LoopType.MAIN),
            LoopType.SCANNER: LoopStats(loop_type=LoopType.SCANNER)
        }
        
        # Error tracking
        self._consecutive_errors = {
            LoopType.MAIN: 0,
            LoopType.SCANNER: 0
        }
        self._error_cooldown_until: Dict[LoopType, Optional[datetime]] = {
            LoopType.MAIN: None,
            LoopType.SCANNER: None
        }
        
        # Execution lock to prevent overlapping trades
        self._execution_lock = asyncio.Lock()
    
    def set_main_callback(self, callback: Callable) -> None:
        """Set callback for main loop iteration"""
        self._main_callback = callback
    
    def set_scanner_callback(self, callback: Callable) -> None:
        """Set callback for scanner loop iteration"""
        self._scanner_callback = callback
    
    def set_error_callback(self, callback: Callable) -> None:
        """Set callback for error handling"""
        self._error_callback = callback
    
    async def start(self) -> None:
        """Start both loops"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.paused = False
        
        logger.info(
            f"Starting dual-loop scheduler: "
            f"main={self.config.main_interval_minutes}min, "
            f"scanner={self.config.scanner_interval_minutes}min"
        )
        
        # Create tasks
        self._main_task = asyncio.create_task(self._main_loop())
        self._scanner_task = asyncio.create_task(self._scanner_loop())
        
        logger.info("Dual-loop scheduler started")
    
    async def stop(self) -> None:
        """Stop both loops gracefully"""
        logger.info("Stopping dual-loop scheduler...")
        self.running = False
        
        # Cancel tasks
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        
        if self._scanner_task:
            self._scanner_task.cancel()
            try:
                await self._scanner_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Dual-loop scheduler stopped")
    
    def pause(self) -> None:
        """Pause execution (loops continue but don't execute callbacks)"""
        self.paused = True
        logger.info("Scheduler paused")
    
    def resume(self) -> None:
        """Resume execution"""
        self.paused = False
        logger.info("Scheduler resumed")
    
    async def _main_loop(self) -> None:
        """Main trading loop - runs every 10 minutes"""
        loop_type = LoopType.MAIN
        interval = self.config.main_interval_minutes * 60
        
        while self.running:
            try:
                # Check cooldown
                if self._in_cooldown(loop_type):
                    await asyncio.sleep(10)
                    continue
                
                # Check pause
                if self.paused:
                    await asyncio.sleep(5)
                    continue
                
                start_time = datetime.now(timezone.utc)
                
                # Execute callback with lock
                if self._main_callback:
                    async with self._execution_lock:
                        result = await self._main_callback(
                            loop_type=loop_type,
                            confidence_threshold=self.config.main_confidence_threshold
                        )
                        self._update_stats(loop_type, start_time, result)
                
                # Reset error count on success
                self._consecutive_errors[loop_type] = 0
                
                # Calculate sleep time
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                sleep_time = max(0, interval - elapsed)
                
                logger.info(f"Main loop done. Next run in {sleep_time/60:.1f} minutes...")
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(loop_type, e)
                await asyncio.sleep(10)  # Brief pause before retry
    
    async def _scanner_loop(self) -> None:
        """Scanner loop - runs every 3 minutes for high-confidence signals"""
        loop_type = LoopType.SCANNER
        interval = self.config.scanner_interval_minutes * 60
        
        while self.running:
            try:
                # Check cooldown
                if self._in_cooldown(loop_type):
                    await asyncio.sleep(10)
                    continue
                
                # Check pause
                if self.paused:
                    await asyncio.sleep(5)
                    continue
                
                start_time = datetime.now(timezone.utc)
                
                # Execute callback with lock
                if self._scanner_callback:
                    async with self._execution_lock:
                        result = await self._scanner_callback(
                            loop_type=loop_type,
                            confidence_threshold=self.config.scanner_confidence_threshold
                        )
                        self._update_stats(loop_type, start_time, result)
                
                # Reset error count on success
                self._consecutive_errors[loop_type] = 0
                
                # Calculate sleep time
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                sleep_time = max(0, interval - elapsed)
                
                logger.info(f"Scanner loop done. Next scan in {sleep_time/60:.1f} minutes...")
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error(loop_type, e)
                await asyncio.sleep(10)
    
    def _in_cooldown(self, loop_type: LoopType) -> bool:
        """Check if loop is in error cooldown"""
        cooldown_until = self._error_cooldown_until.get(loop_type)
        if cooldown_until and datetime.now(timezone.utc) < cooldown_until:
            return True
        return False
    
    def _update_stats(
        self, 
        loop_type: LoopType, 
        start_time: datetime,
        result: Dict = None
    ) -> None:
        """Update loop statistics"""
        stats = self.stats[loop_type]
        
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        stats.iterations += 1
        stats.last_run = datetime.now(timezone.utc)
        stats.total_duration_ms += duration_ms
        stats.avg_duration_ms = stats.total_duration_ms / stats.iterations
        
        if result:
            stats.signals_generated += result.get("signals", 0)
            stats.trades_executed += result.get("trades", 0)
    
    async def _handle_error(self, loop_type: LoopType, error: Exception) -> None:
        """Handle loop error"""
        self._consecutive_errors[loop_type] += 1
        self.stats[loop_type].errors += 1
        
        logger.error(f"{loop_type.value} loop error ({self._consecutive_errors[loop_type]}): {error}")
        
        # Check if we need cooldown
        if self._consecutive_errors[loop_type] >= self.config.max_consecutive_errors:
            cooldown = timedelta(seconds=self.config.error_cooldown_seconds)
            self._error_cooldown_until[loop_type] = datetime.now(timezone.utc) + cooldown
            logger.warning(f"{loop_type.value} loop entering cooldown for {self.config.error_cooldown_seconds}s")
        
        # Call error callback
        if self._error_callback:
            try:
                await self._error_callback(loop_type, error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        return {
            "running": self.running,
            "paused": self.paused,
            "main_loop": {
                "iterations": self.stats[LoopType.MAIN].iterations,
                "signals": self.stats[LoopType.MAIN].signals_generated,
                "trades": self.stats[LoopType.MAIN].trades_executed,
                "errors": self.stats[LoopType.MAIN].errors,
                "avg_duration_ms": self.stats[LoopType.MAIN].avg_duration_ms,
                "last_run": self.stats[LoopType.MAIN].last_run.isoformat() if self.stats[LoopType.MAIN].last_run else None,
                "in_cooldown": self._in_cooldown(LoopType.MAIN)
            },
            "scanner_loop": {
                "iterations": self.stats[LoopType.SCANNER].iterations,
                "signals": self.stats[LoopType.SCANNER].signals_generated,
                "trades": self.stats[LoopType.SCANNER].trades_executed,
                "errors": self.stats[LoopType.SCANNER].errors,
                "avg_duration_ms": self.stats[LoopType.SCANNER].avg_duration_ms,
                "last_run": self.stats[LoopType.SCANNER].last_run.isoformat() if self.stats[LoopType.SCANNER].last_run else None,
                "in_cooldown": self._in_cooldown(LoopType.SCANNER)
            },
            "config": {
                "main_interval_minutes": self.config.main_interval_minutes,
                "scanner_interval_minutes": self.config.scanner_interval_minutes,
                "main_confidence_threshold": self.config.main_confidence_threshold,
                "scanner_confidence_threshold": self.config.scanner_confidence_threshold
            }
        }
    
    async def run_once(self, loop_type: LoopType) -> Dict:
        """Run a single iteration of specified loop (for testing)"""
        if loop_type == LoopType.MAIN and self._main_callback:
            async with self._execution_lock:
                return await self._main_callback(
                    loop_type=loop_type,
                    confidence_threshold=self.config.main_confidence_threshold
                )
        elif loop_type == LoopType.SCANNER and self._scanner_callback:
            async with self._execution_lock:
                return await self._scanner_callback(
                    loop_type=loop_type,
                    confidence_threshold=self.config.scanner_confidence_threshold
                )
        return {}
