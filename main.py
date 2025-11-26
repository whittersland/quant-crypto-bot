#!/usr/bin/env python3
"""
Dual-Condition Algorithmic Trading System
Main Entry Point

Usage:
    python main.py                    # Run in simulated mode (paper trading)
    python main.py --live             # Run in live mode (real trading)
    python main.py --paper            # Paper trading with REAL market data (no real trades)
    python main.py --config path.yaml # Use custom config file
    python main.py --status           # Show current status and exit
"""

import asyncio
import argparse
import sys
import os

# Add parent directory to path so we can import as package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_system.core.trading_system import TradingSystem, run_trading_system
from trading_system.utils.logger import setup_logging, get_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Dual-Condition Algorithmic Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run in simulated (paper) mode
  python main.py --live             Run with real Coinbase API
  python main.py --paper            Paper trade with REAL market data
  python main.py --config my.yaml   Use custom configuration
  python main.py --status           Show system status
  python main.py --test             Run single iteration test
        """
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in LIVE mode (real trading). Default is simulated."
    )
    
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper trading with REAL market data but NO real trades."
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current system status and exit"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a single iteration test and exit"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()


async def show_status(config_path: str = None):
    """Show system status and exit"""
    system = TradingSystem(config_path=config_path, simulated=True)
    await system.initialize()
    
    status = system.get_status()
    
    print("\n" + "=" * 60)
    print("TRADING SYSTEM STATUS")
    print("=" * 60)
    
    print(f"\nSymbols: {status['symbols']}")
    print(f"Running: {status['running']}")
    print(f"Simulated: {status['simulated']}")
    
    print("\n--- Portfolio ---")
    portfolio = status['portfolio']
    print(f"Capital: ${portfolio['capital']['current']:.2f}")
    print(f"Available: ${portfolio['capital']['available']:.2f}")
    print(f"Allocated: ${portfolio['capital']['allocated']:.2f}")
    print(f"Realized P&L: ${portfolio['pnl']['realized']:.2f}")
    print(f"Unrealized P&L: ${portfolio['pnl']['unrealized']:.2f}")
    
    print("\n--- Position Sizing ---")
    sizing = status['position_sizing']
    print(f"Current Range: {sizing['current_range']}")
    print(f"Consecutive Profit Periods: {sizing['consecutive_profit_periods']}")
    
    print("\n--- Risk ---")
    risk = status['risk']['risk_assessment']
    print(f"Level: {risk['level']}")
    print(f"Can Trade: {risk['can_trade']}")
    print(f"Reason: {risk['reason']}")
    
    print("\n--- Active Positions ---")
    positions = status['active_positions']
    if positions:
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['side']} @ ${pos['entry_price']:.2f}")
    else:
        print("  No active positions")
    
    print("\n--- Strategies ---")
    for name, strat in status['strategies'].items():
        print(f"  {name}: {strat['total_signals']} signals, {strat['win_rate']:.1f}% win rate")
    
    print("\n" + "=" * 60)
    
    await system.client.close()


async def run_test(config_path: str = None):
    """Run single iteration test"""
    from trading_system.core.scheduler import LoopType
    
    print("\n" + "=" * 60)
    print("RUNNING SINGLE ITERATION TEST")
    print("=" * 60)
    
    system = TradingSystem(config_path=config_path, simulated=True)
    await system.initialize()
    
    print("\n--- Running Main Loop Iteration ---")
    result = await system.scheduler.run_once(LoopType.MAIN)
    print(f"Signals generated: {result.get('signals', 0)}")
    print(f"Trades executed: {result.get('trades', 0)}")
    
    print("\n--- Running Scanner Loop Iteration ---")
    result = await system.scheduler.run_once(LoopType.SCANNER)
    print(f"Signals generated: {result.get('signals', 0)}")
    print(f"Trades executed: {result.get('trades', 0)}")
    
    print("\n--- Final Status ---")
    status = system.get_status()
    print(f"Capital: ${status['portfolio']['capital']['current']:.2f}")
    print(f"Active Positions: {len(status['active_positions'])}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    await system.client.close()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = get_logger(__name__)
    
    # Determine mode
    simulated = not args.live and not args.paper
    paper_mode = args.paper
    
    if args.live:
        # Safety confirmation for live mode
        print("\n" + "!" * 60)
        print("WARNING: LIVE TRADING MODE")
        print("This will execute REAL trades with REAL money!")
        print("!" * 60)
        
        confirm = input("\nType 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            sys.exit(0)
    
    try:
        if args.status:
            asyncio.run(show_status(args.config))
        elif args.test:
            asyncio.run(run_test(args.config))
        elif paper_mode:
            # Paper trading with real market data
            print("\n" + "=" * 60)
            print("PAPER TRADING MODE")
            print("Using REAL market data, but NO real trades")
            print("=" * 60)
            print("\nPress Ctrl+C to stop\n")
            
            asyncio.run(run_trading_system(
                config_path=args.config,
                simulated=False,  # Use real API for market data
                paper_trading=True  # But don't execute real trades
            ))
        else:
            # Run the trading system
            print("\n" + "=" * 60)
            print(f"STARTING {'SIMULATED' if simulated else 'LIVE'} TRADING")
            print("=" * 60)
            print("\nPress Ctrl+C to stop\n")
            
            asyncio.run(run_trading_system(
                config_path=args.config,
                simulated=simulated
            ))
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
