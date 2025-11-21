"""
ACCOUNT-LEVEL RISK MANAGER
Monitors total account drawdown and closes ALL positions if threshold breached
This is what E8 prop firms actually care about - not per-trade stops
"""
import os
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account_id = os.getenv('OANDA_ACCOUNT_ID', '101-001-37330890-001')

client = API(access_token=oanda_token, environment='practice')

# ACCOUNT-LEVEL RISK THRESHOLDS
STARTING_BALANCE = 187302.37  # Reset after -$1,800 loss (was 189102.18)
MAX_DRAWDOWN_PCT = 0.04       # 4% max drawdown (conservative for E8's 6%)
WARNING_DRAWDOWN_PCT = 0.03   # 3% warning level

# Calculate critical levels
MAX_DRAWDOWN_AMOUNT = STARTING_BALANCE * MAX_DRAWDOWN_PCT
STOP_LOSS_BALANCE = STARTING_BALANCE - MAX_DRAWDOWN_AMOUNT
WARNING_BALANCE = STARTING_BALANCE - (STARTING_BALANCE * WARNING_DRAWDOWN_PCT)

def get_account_status():
    """Get current account balance and unrealized P/L"""
    try:
        r = accounts.AccountSummary(accountID=oanda_account_id)
        response = client.request(r)

        balance = float(response['account']['balance'])
        unrealized_pl = float(response['account'].get('unrealizedPL', 0))

        # Total equity = balance + unrealized P/L
        total_equity = balance + unrealized_pl

        return {
            'balance': balance,
            'unrealized_pl': unrealized_pl,
            'total_equity': total_equity
        }
    except Exception as e:
        print(f"[ERROR] Getting account status: {e}")
        return None

def close_all_positions():
    """Emergency close ALL positions"""
    try:
        print("\n" + "="*70)
        print("EMERGENCY ACCOUNT PROTECTION - CLOSING ALL POSITIONS")
        print("="*70)

        r = trades.TradesList(accountID=oanda_account_id)
        response = client.request(r)
        trades_list = response.get('trades', [])

        if not trades_list:
            print("No positions to close")
            return True

        closed_count = 0
        total_loss = 0

        for trade in trades_list:
            trade_id = trade['id']
            instrument = trade['instrument']
            unrealized = float(trade['unrealizedPL'])

            print(f"\nClosing {instrument}:")
            print(f"  Trade ID: {trade_id}")
            print(f"  P/L: ${unrealized:,.2f}")

            try:
                r = trades.TradeClose(accountID=oanda_account_id, tradeID=trade_id)
                client.request(r)
                print(f"  [CLOSED] Position closed")
                closed_count += 1
                total_loss += unrealized
            except Exception as e:
                print(f"  [ERROR] Failed to close: {e}")

        print(f"\n{'='*70}")
        print(f"Closed {closed_count} positions")
        print(f"Total Realized P/L: ${total_loss:,.2f}")
        print(f"{'='*70}\n")

        return closed_count > 0

    except Exception as e:
        print(f"[ERROR] Emergency close failed: {e}")
        return False

def monitor_account_risk():
    """Main loop - monitor account-level drawdown"""
    print("="*70)
    print("ACCOUNT-LEVEL RISK MANAGER - ACTIVE")
    print("="*70)
    print(f"Starting Balance: ${STARTING_BALANCE:,.2f}")
    print(f"Warning Level (-3%): ${WARNING_BALANCE:,.2f}")
    print(f"STOP LOSS Level (-4%): ${STOP_LOSS_BALANCE:,.2f}")
    print(f"E8 Max Drawdown (-6%): ${STARTING_BALANCE * 0.94:,.2f}")
    print("="*70)

    peak_equity = STARTING_BALANCE
    warning_triggered = False

    while True:
        try:
            status = get_account_status()
            if not status:
                time.sleep(30)
                continue

            balance = status['balance']
            unrealized_pl = status['unrealized_pl']
            total_equity = status['total_equity']

            # Calculate drawdown from starting balance
            drawdown_dollars = STARTING_BALANCE - total_equity
            drawdown_pct = (drawdown_dollars / STARTING_BALANCE) * 100

            # Track peak equity
            if total_equity > peak_equity:
                peak_equity = total_equity
                warning_triggered = False  # Reset warning if we recover

            # Calculate drawdown from peak (alternative metric)
            peak_drawdown_pct = ((peak_equity - total_equity) / peak_equity) * 100

            timestamp = datetime.now().strftime('%H:%M:%S')

            # Check critical levels
            if total_equity <= STOP_LOSS_BALANCE:
                # CRITICAL: Hit max drawdown - close everything
                print(f"\n[{timestamp}] {'='*70}")
                print(f"CRITICAL: ACCOUNT STOP LOSS TRIGGERED")
                print(f"{'='*70}")
                print(f"Starting Balance: ${STARTING_BALANCE:,.2f}")
                print(f"Current Equity:   ${total_equity:,.2f}")
                print(f"Drawdown:         ${drawdown_dollars:,.2f} ({drawdown_pct:.2f}%)")
                print(f"MAX THRESHOLD:    {MAX_DRAWDOWN_PCT*100:.1f}% BREACHED")
                print(f"{'='*70}")

                # Close all positions immediately
                if close_all_positions():
                    print("\n[SUCCESS] All positions closed - Account protected")
                    print(f"[INFO] Final equity: ${balance:,.2f}")
                    print(f"[INFO] Session loss: ${drawdown_dollars:,.2f}")

                    # Could restart bot here or just monitor
                    print("\n[INFO] Risk manager will continue monitoring...")
                    print("[INFO] No new trades will be placed until restart")

                # After closing, just monitor (don't exit loop)
                time.sleep(60)
                continue

            elif total_equity <= WARNING_BALANCE and not warning_triggered:
                # WARNING: Approaching danger zone
                print(f"\n[{timestamp}] {'='*70}")
                print(f"WARNING: APPROACHING DRAWDOWN LIMIT")
                print(f"{'='*70}")
                print(f"Current Equity:   ${total_equity:,.2f}")
                print(f"Drawdown:         {drawdown_pct:.2f}%")
                print(f"Stop Loss Level:  {MAX_DRAWDOWN_PCT*100:.1f}% (${STOP_LOSS_BALANCE:,.2f})")
                print(f"Buffer Remaining: ${total_equity - STOP_LOSS_BALANCE:,.2f}")
                print(f"{'='*70}\n")
                warning_triggered = True

            else:
                # Normal monitoring - print status every 5 minutes
                if int(time.time()) % 300 < 30:  # Every ~5 minutes
                    status_indicator = "✓" if drawdown_pct < 1 else "⚠"
                    print(f"[{timestamp}] Account: ${total_equity:,.2f} | "
                          f"Unrealized: ${unrealized_pl:,.2f} | "
                          f"Drawdown: {drawdown_pct:.2f}%")

            # Check every 30 seconds
            time.sleep(30)

        except KeyboardInterrupt:
            print("\n[STOPPED] Account risk manager shutting down...")
            break
        except Exception as e:
            print(f"[ERROR] Monitor loop error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_account_risk()
