#!/usr/bin/env python3
"""
WORKING FOREX MONITOR
Single scan, no loops, guaranteed to work
"""

import sys
import json
import requests
from datetime import datetime

# Config
API_KEY = "0bff5dc7375409bb8747deebab8988a1-d8b26324102c95d6f2b6f641bc330a7c"
BASE_URL = "https://api-fxpractice.oanda.com/v3"
PAIRS = ['EUR_USD', 'USD_JPY', 'GBP_USD']

def fetch_price(pair):
    """Fetch current price"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    url = f"{BASE_URL}/instruments/{pair}/candles"
    params = {'count': 1, 'granularity': 'H1', 'price': 'M'}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=3)
        if response.status_code == 200:
            data = response.json()
            candles = data.get('candles', [])
            if candles:
                return float(candles[0]['mid']['c'])
    except:
        pass
    return None

# Run single scan
print(f"\nFOREX MONITOR - {datetime.now().strftime('%H:%M:%S')}")
print("-" * 40)

results = []
for pair in PAIRS:
    price = fetch_price(pair)
    if price:
        print(f"{pair}: {price:.5f}")
        results.append({'pair': pair, 'price': price, 'time': datetime.now().isoformat()})
    else:
        print(f"{pair}: FAILED")

# Save results
if results:
    with open('forex_prices_latest.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} prices to forex_prices_latest.json")

print("\nDone.")
sys.exit(0)