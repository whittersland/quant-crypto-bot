#!/bin/bash
# ============================================================
# trading_system2 - Live Trading Launcher
# ============================================================
#
# This script starts the trading system in LIVE mode
# 
# Before running:
# 1. Set your API credentials (choose one method):
#    
#    Method A - Environment variables (recommended):
#      export COINBASE_API_KEY="your-api-key"
#      export COINBASE_API_SECRET="your-api-secret"
#    
#    Method B - Edit config/settings.yaml directly
#
# 2. Test first:
#      python main.py --test
#      python main.py --status
#
# ============================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo -e "${YELLOW}⚠️  LIVE TRADING MODE${NC}"
echo "============================================================"
echo ""

# Check for API credentials
if [ -z "$COINBASE_API_KEY" ]; then
    echo -e "${RED}ERROR: COINBASE_API_KEY not set${NC}"
    echo ""
    echo "Set your credentials:"
    echo "  export COINBASE_API_KEY=\"your-key\""
    echo "  export COINBASE_API_SECRET=\"your-secret\""
    echo ""
    exit 1
fi

if [ -z "$COINBASE_API_SECRET" ]; then
    echo -e "${RED}ERROR: COINBASE_API_SECRET not set${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ API credentials found${NC}"
echo ""

# Show configuration
echo "Configuration:"
echo "  Capital: \$360.65"
echo "  Leverage: 2x"
echo "  Symbols: BTC-USD, ETH-USD, SOL-USD"
echo "  Position Size: \$15-\$60"
echo "  Trailing Stop: 12%"
echo "  Daily Loss Limit: 15%"
echo "  Max Drawdown: 25%"
echo ""

# Confirmation
echo -e "${RED}This will execute REAL trades with REAL money!${NC}"
echo ""
read -p "Type 'CONFIRM' to start live trading: " confirm

if [ "$confirm" != "CONFIRM" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${GREEN}Starting live trading...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Run live trading
python3 main.py --live
