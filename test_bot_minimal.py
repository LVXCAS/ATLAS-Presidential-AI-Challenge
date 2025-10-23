#!/usr/bin/env python3
"""
Minimal test to find the exact import issue
"""

import sys
import traceback

print("Testing basic imports...")

try:
    print("1. Testing core imports...")
    import os
    import time
    import json
    import random
    import requests
    from datetime import datetime, timedelta, time as dt_time
    from typing import Dict, List, Optional, Tuple
    print("[OK] Core imports OK")

    print("2. Testing data libraries...")
    import yfinance as yf
    import numpy as np
    import pandas as pd
    import pytz
    print("[OK] Data libraries OK")

    print("3. Testing dotenv...")
    from dotenv import load_dotenv
    load_dotenv('.env')
    print("[OK] Environment loading OK")

    print("4. Testing agent imports...")
    from agents.broker_integration import AlpacaBrokerIntegration
    print("[OK] Broker integration OK")

    from agents.options_trading_agent import OptionsTrader, OptionsStrategy
    print("[OK] Options trading agent OK")

    from agents.options_broker import OptionsBroker
    print("[OK] Options broker OK")

    from agents.risk_management import RiskManager, RiskLevel
    print("[OK] Risk management OK")

    print("5. Testing OPTIONS_BOT...")
    import OPTIONS_BOT
    print("[OK] OPTIONS_BOT import OK")

    print("6. Testing bot initialization...")
    bot = OPTIONS_BOT.TomorrowReadyOptionsBot()
    print("[OK] Bot initialization OK")

    print("\n[SUCCESS] All tests passed!")

except Exception as e:
    print(f"\n[ERROR] at step: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)