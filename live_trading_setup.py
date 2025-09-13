#!/usr/bin/env python3
"""
Live Trading Setup - Configure and start live trading system
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List

# Import our live trading components
try:
    from agents.live_data_manager import setup_live_data
    from agents.live_trading_engine import LiveTradingEngine
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False

def create_config_template():
    """Create configuration template file"""
    config_template = {
        "trading": {
            "mode": "paper",  # "paper" or "live"
            "account_value": 100000.0,
            "symbols": ["SPY", "QQQ", "AAPL"]
        },
        "risk_management": {
            "max_portfolio_risk": 0.02,
            "max_position_size": 0.05,
            "max_daily_trades": 50,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.06
        },
        "data_sources": {
            "yahoo_finance": {
                "enabled": True,
                "note": "Free but 15-minute delayed data"
            },
            "finnhub": {
                "enabled": False,
                "api_key": "your_finnhub_api_key_here",
                "note": "Free tier: 60 calls/minute. Get key at https://finnhub.io"
            },
            "twelvedata": {
                "enabled": False,
                "api_key": "your_twelvedata_api_key_here", 
                "note": "Free tier: 800 calls/day. Get key at https://twelvedata.com"
            },
            "polygon": {
                "enabled": False,
                "api_key": "your_polygon_api_key_here",
                "note": "Paid service. Get key at https://polygon.io"
            }
        },
        "brokers": {
            "alpaca": {
                "enabled": False,
                "api_key": "your_alpaca_api_key_here",
                "secret_key": "your_alpaca_secret_key_here",
                "base_url": "https://paper-api.alpaca.markets",
                "note": "Free paper trading. Get keys at https://alpaca.markets"
            }
        }
    }
    
    with open('live_trading_config.json', 'w') as f:
        json.dump(config_template, f, indent=4)
    
    print("+ Configuration template created: live_trading_config.json")
    return config_template

def load_config() -> Dict:
    """Load trading configuration"""
    if not os.path.exists('live_trading_config.json'):
        print("Configuration file not found. Creating template...")
        return create_config_template()
    
    try:
        with open('live_trading_config.json', 'r') as f:
            config = json.load(f)
        print("+ Configuration loaded from live_trading_config.json")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return create_config_template()

def validate_config(config: Dict) -> bool:
    """Validate configuration"""
    print("\nCONFIGURATION VALIDATION")
    print("=" * 30)
    
    valid = True
    
    # Check data sources
    data_sources_enabled = 0
    for source, settings in config.get('data_sources', {}).items():
        if settings.get('enabled', False):
            if source == 'yahoo_finance':
                data_sources_enabled += 1
                print(f"+ {source}: enabled (free)")
            elif settings.get('api_key') and 'your_' not in settings['api_key']:
                data_sources_enabled += 1
                print(f"+ {source}: enabled with API key")
            else:
                print(f"! {source}: enabled but no API key configured")
    
    if data_sources_enabled == 0:
        print("X No data sources configured!")
        valid = False
    else:
        print(f"+ {data_sources_enabled} data source(s) configured")
    
    # Check broker
    broker_configured = False
    for broker, settings in config.get('brokers', {}).items():
        if settings.get('enabled', False):
            if (settings.get('api_key') and 'your_' not in settings['api_key'] and
                settings.get('secret_key') and 'your_' not in settings['secret_key']):
                broker_configured = True
                print(f"+ {broker}: configured for live trading")
            else:
                print(f"! {broker}: enabled but API keys not configured (will use simulation)")
    
    if not broker_configured:
        print("! No broker configured - will run in simulation mode")
    
    # Check strategies
    if os.path.exists('deployed_strategies'):
        strategy_files = [f for f in os.listdir('deployed_strategies') if f.endswith('.py')]
        if strategy_files:
            print(f"+ {len(strategy_files)} strategies available")
        else:
            print("X No strategies found in deployed_strategies/")
            valid = False
    else:
        print("X deployed_strategies/ directory not found!")
        valid = False
    
    return valid

def print_setup_instructions():
    """Print setup instructions"""
    print("\n" + "="*60)
    print("LIVE TRADING SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. GET FREE API KEYS (Optional but recommended):")
    print("   [DATA] Finnhub (Free real-time data):")
    print("      - Visit: https://finnhub.io")
    print("      - Sign up for free account")
    print("      - Copy API key to config file")
    
    print("\n   [DATA] TwelveData (Free real-time data):")
    print("      - Visit: https://twelvedata.com")
    print("      - Sign up for free account") 
    print("      - Copy API key to config file")
    
    print("\n2. PAPER TRADING (Recommended first step):")
    print("   [BROKER] Alpaca Paper Trading (Free):")
    print("      - Visit: https://alpaca.markets")
    print("      - Sign up for paper trading account")
    print("      - Get API keys from dashboard")
    print("      - Copy keys to config file")
    
    print("\n3. CONFIGURATION:")
    print("   [CONFIG] Edit live_trading_config.json:")
    print("      - Add your API keys")
    print("      - Set trading mode ('paper' or 'live')")
    print("      - Adjust risk limits")
    print("      - Choose symbols to trade")
    
    print("\n4. START TRADING:")
    print("   [START] Run: python live_trading_setup.py --start")
    print("      - Starts with your deployed strategies")
    print("      - Real-time data feeds")
    print("      - Automated trade execution")
    print("      - Risk management")

async def start_live_trading_system(config: Dict):
    """Start the complete live trading system"""
    if not DATA_AVAILABLE:
        print("X Live trading modules not available!")
        return False
    
    print("\nSTARTING LIVE TRADING SYSTEM")
    print("=" * 40)
    
    # Setup data manager
    data_config = {}
    
    # Configure data sources
    for source, settings in config.get('data_sources', {}).items():
        if settings.get('enabled', False) and settings.get('api_key'):
            if source == 'finnhub':
                data_config['finnhub_key'] = settings['api_key']
            elif source == 'twelvedata':
                data_config['twelvedata_key'] = settings['api_key']
            elif source == 'polygon':
                data_config['polygon_key'] = settings['api_key']
    
    # Configure broker
    for broker, settings in config.get('brokers', {}).items():
        if settings.get('enabled', False):
            if broker == 'alpaca':
                data_config['alpaca_key'] = settings.get('api_key')
                data_config['alpaca_secret'] = settings.get('secret_key')
                data_config['alpaca_base_url'] = settings.get('base_url')
    
    # Setup live data
    data_manager = setup_live_data(data_config)
    
    # Setup trading engine
    trading_config = {**data_config}
    trading_config['trading_mode'] = config.get('trading', {}).get('mode', 'paper')
    
    trading_engine = LiveTradingEngine(trading_config)
    
    # Override risk limits if specified
    if 'risk_management' in config:
        trading_engine.risk_limits.update(config['risk_management'])
    
    # Load strategies
    strategies_loaded = await trading_engine.load_strategies()
    if not strategies_loaded:
        print("X No strategies loaded!")
        return False
    
    # Get symbols
    symbols = config.get('trading', {}).get('symbols', [])
    if not symbols:
        # Auto-detect from strategies
        symbols = []
        for strategy_info in trading_engine.strategies.values():
            if strategy_info['instance']:
                symbol = strategy_info['instance'].symbol
                if symbol not in symbols:
                    symbols.append(symbol)
    
    print(f"Trading symbols: {', '.join(symbols)}")
    print(f"Mode: {trading_config['trading_mode'].upper()}")
    print("\n[START] Starting live trading...")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    
    try:
        # Start trading
        await trading_engine.start_trading(symbols)
    except KeyboardInterrupt:
        print("\n\n[STOP] Stopping live trading...")
        await trading_engine.stop_trading()
    
    return True

def main():
    """Main setup function"""
    import sys
    
    print("LIVE TRADING SYSTEM SETUP")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Check if user wants to start trading
    if '--start' in sys.argv:
        if validate_config(config):
            print("\n[OK] Configuration valid! Starting live trading...")
            asyncio.run(start_live_trading_system(config))
        else:
            print("\n[ERROR] Configuration issues found. Please fix and try again.")
    
    elif '--validate' in sys.argv:
        validate_config(config)
    
    else:
        # Show setup instructions
        validate_config(config)
        print_setup_instructions()
        
        print(f"\n{'='*60}")
        print("QUICK START:")
        print("1. Edit live_trading_config.json with your API keys")
        print("2. Run: python live_trading_setup.py --validate")
        print("3. Run: python live_trading_setup.py --start")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()