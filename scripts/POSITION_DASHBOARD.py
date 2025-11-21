"""
48-Hour Position Monitoring Dashboard
Tracks GBP_USD and EUR_USD positions with defined exit rules

Check this 2x per day: Morning (9am) and Evening (6pm)
"""

import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
from datetime import datetime, timedelta

client = API(access_token=os.getenv('OANDA_API_KEY'), environment='practice')
r = accounts.AccountSummary(accountID=os.getenv('OANDA_ACCOUNT_ID'))
resp = client.request(r)

balance = float(resp['account']['balance'])
unrealized_pl = float(resp['account']['unrealizedPL'])
pl_percent = (unrealized_pl / balance) * 100

print()
print("=" * 80)
print(" " * 20 + "48-HOUR POSITION MONITOR")
print("=" * 80)
print()
print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
print(f"Account Balance: ${balance:,.2f}")
print(f"Unrealized P/L: ${unrealized_pl:,.2f} ({pl_percent:+.2f}%)")
print()

# Exit rules
hard_stop_loss = -2808  # -1.5% of account
hard_stop_pct = -1.5

print("=" * 80)
print("EXIT RULES")
print("=" * 80)
print()
print(f"1. Hard Stop: ${hard_stop_loss:,.2f} ({hard_stop_pct}% of account)")
print(f"2. Time Deadline: Wednesday 11/6 9:00 AM")
print(f"3. If EUR_USD goes positive while GBP_USD negative: Close GBP_USD only")
print()

# Check distance to hard stop
distance_to_stop = unrealized_pl - hard_stop_loss
stop_buffer_pct = ((hard_stop_loss - unrealized_pl) / balance) * 100

print("=" * 80)
print("RISK STATUS")
print("=" * 80)
print()

if unrealized_pl <= hard_stop_loss:
    print("*** ALERT: HARD STOP HIT - CLOSE BOTH POSITIONS NOW ***")
    print(f"Current P/L (${unrealized_pl:,.2f}) is below hard stop (${hard_stop_loss:,.2f})")
    print()
    print("ACTION REQUIRED: Run close_both_positions.py")
elif unrealized_pl <= (hard_stop_loss * 0.8):
    print("WARNING: Approaching hard stop")
    print(f"Current P/L: ${unrealized_pl:,.2f}")
    print(f"Hard Stop: ${hard_stop_loss:,.2f}")
    print(f"Buffer Remaining: ${distance_to_stop:,.2f} ({abs(stop_buffer_pct):.2f}%)")
    print()
    print("Monitor closely - near exit threshold")
else:
    print("Status: Within acceptable risk range")
    print(f"Current P/L: ${unrealized_pl:,.2f}")
    print(f"Hard Stop: ${hard_stop_loss:,.2f}")
    print(f"Buffer Remaining: ${distance_to_stop:,.2f} ({abs(stop_buffer_pct):.2f}%)")

print()

# Deadline tracker (48 hours from now)
now = datetime.now()
deadline = now + timedelta(hours=48)  # 48 hours from now
time_remaining = deadline - now

hours_remaining = int(time_remaining.total_seconds() / 3600)

print("=" * 80)
print("TIME REMAINING")
print("=" * 80)
print()
print(f"Deadline: {deadline.strftime('%A %m/%d %I:%M %p')}")
print(f"Hours Remaining: ~{hours_remaining} hours")
print()

if hours_remaining <= 0:
    print("*** DEADLINE REACHED ***")
    print("If positions still negative, close both now")
    print()
elif hours_remaining <= 12:
    print("WARNING: Less than 12 hours to deadline")
    print("Prepare for potential exit tomorrow morning")
else:
    print("Status: Monitoring period active")

print()

# Historical context
print("=" * 80)
print("BACKTEST CONTEXT (For Reference)")
print("=" * 80)
print()
print("GBP_USD Performance:")
print("  Historical Win Rate: 28.6% (71.4% chance of loss)")
print("  Total P/L (6 months): -3.03%")
print("  Verdict: Worst performer")
print()
print("EUR_USD Performance:")
print("  Historical Win Rate: 35.7% (64.3% chance of loss)")
print("  Total P/L (6 months): +1.99%")
print("  Verdict: Marginal performer")
print()
print("Overall Strategy:")
print("  Win Rate: 38.9%")
print("  Profit Factor: 1.22")
print("  Sharpe Ratio: 7.43 (excellent)")
print()

# Perplexity validation
print("=" * 80)
print("PERPLEXITY VALIDATION")
print("=" * 80)
print()
print("Trade Quality Assessment:")
print("  Risk/Reward: Solid (2:1 ratio)")
print("  Position Sizing: Conservative (6.6x leverage)")
print("  Stop Placement: Reasonable ($1,872 per trade)")
print("  Market Confirmation: Not yet confirmed (waiting for breakdown)")
print()
print("Recommendation: Monitor closely, watch for economic news catalysts")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Check this dashboard 2x per day (morning & evening)")
print("2. If hard stop hit: Close both positions immediately")
print("3. If Wednesday arrives and still negative: Close both positions")
print("4. If EUR_USD goes positive: Consider closing GBP_USD only")
print()
print("For detailed position view: python VIEW_FOREX_POSITIONS.py")
print()
print("=" * 80)
