#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scanners.futures_scanner import AIEnhancedFuturesScanner
from datetime import datetime
import time

print("\n" + "="*70)
print("FUTURES PAPER TRADER - ACTIVE MODE")
print("="*70)

scanner = AIEnhancedFuturesScanner(paper_trading=True)

print("\nStarting continuous futures scanning...")
print("MES: Micro E-mini S&P 500 ($5/point)")
print("MNQ: Micro E-mini NASDAQ ($2/point)")
print("\nThis will EXECUTE paper trades on Alpaca\n")

scan_count = 0

while True:
    try:
        scan_count += 1
        print(f"\n[SCAN #{scan_count}] {datetime.now().strftime('%H:%M:%S')}")

        # Scan for opportunities
        opportunities = scanner.scan_all_futures()

        if opportunities:
            print(f"  Found {len(opportunities)} opportunities")

            # Execute top opportunities (scanner will paper trade them)
            for opp in opportunities[:3]:  # Top 3
                print(f"  -> {opp['symbol']} {opp['direction']}: {opp['confidence']:.0%} confidence")
                # Scanner's internal execution will handle the paper trade
        else:
            print("  No opportunities meeting criteria")

        # Wait 15 minutes
        time.sleep(900)

    except KeyboardInterrupt:
        print("\n[STOP] Futures scanner stopped by user")
        break
    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(60)
