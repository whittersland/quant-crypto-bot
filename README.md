# Algorithmic Trading System

A sophisticated cryptocurrency trading bot with 9 strategies, dynamic symbol scanning, and risk management.

## Features

### 9 Trading Strategies
1. **Momentum** - Trend continuation signals
2. **Mean Reversion** - Oversold/overbought detection
3. **Volatility Breakout** - Breakout from consolidation
4. **Volume Analysis** - Volume-price divergence
5. **Trend Following** - EMA crossovers with ADX filter
6. **Confluence Convergence** - Multi-indicator agreement
7. **Cross Asset Resonance** - Correlation-based signals
8. **Temporal Alignment** - Multi-timeframe confirmation
9. **Quantum Trader Scoring** - Adaptive ensemble strategy

### VVE+CVD Dynamic Scanner
- Scans 296 Coinbase symbols every 10 minutes
- **VVE (Volume-Volatility Expansion)** - Detects unusual activity
- **CVD (Cumulative Volume Delta)** - Measures buy/sell pressure
- Automatically adds top movers to trading universe

### Risk Management
- 10x leverage with liquidation protection
- 6% trailing stop losses
- 10% daily loss limit
- Position sizing based on signal confidence ($50-$150)
- Honeypot protection for low-volume tokens

### Hybrid Execution
- LONG positions → Spot market
- SHORT positions → Futures contracts
- Supports BTC, ETH, SOL futures + dynamic spot tokens

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-system.git
cd trading-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy example env and configure
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

### API Keys Setup

1. Go to [Coinbase Advanced](https://www.coinbase.com/settings/api)
2. Create new API key with trading permissions
3. Download the private key (.pem file)
4. Set environment variables:

```bash
export COINBASE_API_KEY="organizations/YOUR_ORG_ID/apiKeys/YOUR_KEY_ID"
export COINBASE_API_SECRET="$(cat ~/path/to/coinbase_key.pem)"
```

### Settings (config/settings.yaml)

```yaml
trading:
  leverage: 10
  min_position_size: 50
  max_position_size: 150
  confidence_threshold: 0.65

risk:
  max_daily_loss_pct: 0.10
  stop_loss_pct: 0.06
  max_positions: 5
```

## Usage

### Paper Trading (Simulation)
```bash
python main.py
```

### Live Trading
```bash
python main.py --live
```

### Keep Running (Prevent Sleep)
```bash
caffeinate -i python main.py --live
```

## Architecture

```
trading_system/
├── main.py                 # Entry point
├── config/
│   └── settings.yaml       # Configuration
├── core/
│   ├── trading_system.py   # Main orchestrator
│   ├── execution_engine.py # Order execution
│   ├── risk_manager.py     # Risk controls
│   ├── liquidation_guard.py# Liquidation protection
│   ├── dynamic_symbols.py  # VVE+CVD scanner
│   ├── signal_aggregator.py# Signal combination
│   └── coinbase_client.py  # Exchange API
├── strategies/
│   ├── base_strategy.py    # Strategy interface
│   ├── momentum.py
│   ├── mean_reversion.py
│   ├── volatility_breakout.py
│   ├── volume_analysis.py
│   ├── trend_following.py
│   ├── confluence_convergence.py
│   ├── cross_asset_resonance.py
│   ├── temporal_alignment.py
│   └── quantum_trader_scoring.py
└── utils/
    ├── indicators.py       # Technical indicators
    └── logger.py           # Logging config
```

## Signal Flow

1. **Scanner** identifies top movers (VVE+CVD scoring)
2. **Strategies** analyze each symbol independently
3. **Aggregator** combines signals with confidence weighting
4. **Risk Manager** validates position sizing and limits
5. **Execution Engine** places orders on Coinbase

## Dual-Loop System

- **Main Loop** (10 min): Full analysis, 0.65 confidence threshold
- **Scanner Loop** (3 min): Quick scans, 0.70 confidence threshold

## Safety Features

- Daily loss limit halts trading at 10% drawdown
- Honeypot protection blocks low-volume tokens (<$500K)
- Liquidation guard triggers at 7% adverse move
- Position size limited to $150 max per trade

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## License

MIT License
