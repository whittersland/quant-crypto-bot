# trading_system2 - Live Trading Deployment Guide

## 🚀 Quick Start for Live Trading

### Step 1: Get Coinbase API Credentials

1. Go to [Coinbase Advanced Trade](https://www.coinbase.com/advanced-trade)
2. Click on your profile → Settings → API
3. Create a new API key with these permissions:
   - ✅ View
   - ✅ Trade
   - ✅ Transfer (optional, for deposits/withdrawals)
4. Save your **API Key** and **API Secret** securely

### Step 2: Set Up Credentials

**Option A: Environment Variables (Recommended)**
```bash
export COINBASE_API_KEY="your-api-key-here"
export COINBASE_API_SECRET="your-api-secret-here"
```

**Option B: Edit config/settings.yaml**
```yaml
credentials:
  api_key: "your-api-key-here"
  api_secret: "your-api-secret-here"
```

### Step 3: Test the System

```bash
# Run a single test iteration (simulated)
python main.py --test

# Check system status
python main.py --status
```

### Step 4: Start Live Trading

**Using the launcher script:**
```bash
./start_live.sh
```

**Or directly:**
```bash
python main.py --live
```

---

## ⚙️ Configuration Reference

### config/settings.yaml

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capital` | 360.65 | Starting capital in USD |
| `trading.leverage` | 2 | Leverage multiplier |
| `trading.symbols` | BTC, ETH, SOL | Trading pairs |
| `position_sizing.base_min` | 15 | Minimum position size ($) |
| `position_sizing.base_max` | 60 | Maximum position size ($) |
| `position_sizing.increment` | 5 | Increase per profitable day |
| `position_sizing.max_concurrent` | 4 | Max open positions |
| `risk_management.trailing_stop` | 12% | Trailing stop loss |
| `risk_management.daily_loss_limit` | 15% | Max daily loss |
| `risk_management.max_drawdown` | 25% | Max total drawdown |
| `loops.main_interval` | 10 min | Main loop frequency |
| `loops.scanner_interval` | 3 min | Scanner loop frequency |
| `loops.main_confidence` | 0.15 | Min confidence for main |
| `loops.scanner_confidence` | 0.25 | Min confidence for scanner |

---

## 🖥️ VPS Deployment (Recommended)

### Using Screen (Simple)

```bash
# Start a screen session
screen -S trading

# Run the trading system
./start_live.sh

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r trading
```

### Using Systemd (Production)

Create `/etc/systemd/system/trading_system2.service`:

```ini
[Unit]
Description=trading_system2 Algorithmic Trading
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/trading_system
Environment="COINBASE_API_KEY=your-key"
Environment="COINBASE_API_SECRET=your-secret"
ExecStart=/usr/bin/python3 main.py --live
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable trading_system2
sudo systemctl start trading_system2
sudo systemctl status trading_system2

# View logs
journalctl -u trading_system2 -f
```

---

## 📊 Monitoring

### Log Files

- `trading_system.log` - Main system log
- `logs/trades.log` - Trade executions
- `logs/signals.log` - Generated signals
- `logs/performance.log` - Performance metrics

### Check Status While Running

```bash
# View recent logs
tail -f trading_system.log

# View trades
tail -f logs/trades.log
```

---

## 🛑 Emergency Stop

1. **Keyboard:** Press `Ctrl+C` (graceful shutdown)

2. **Kill process:**
   ```bash
   pkill -f "python.*main.py"
   ```

3. **Systemd:**
   ```bash
   sudo systemctl stop trading_system2
   ```

---

## ⚠️ Risk Warnings

1. **Start Small:** The default $360 capital is intentionally small
2. **Test First:** Always run `--test` before going live
3. **Monitor:** Check the system regularly, especially initially
4. **Understand:** Know what each strategy does before trading
5. **Accept Risk:** Only trade money you can afford to lose

---

## 🔧 Troubleshooting

### "API credentials not configured"
- Set environment variables or edit settings.yaml

### "Failed to connect to Coinbase API"
- Check your API key and secret are correct
- Verify API permissions include Trade access
- Check your internet connection

### "Insufficient balance"
- Deposit funds to your Coinbase account
- Ensure funds are in your Futures wallet

### System stops unexpectedly
- Check logs for errors
- Verify network connectivity
- Restart with `./start_live.sh`

---

## 📈 Performance Tips

1. **Stable Connection:** Use a VPS with good uptime
2. **Low Latency:** Choose a VPS close to Coinbase servers (US)
3. **Monitor Daily:** Check performance at least once per day
4. **Adjust Sizing:** Increase position sizes as you gain confidence
5. **Review Trades:** Analyze winning/losing patterns weekly
