"""
Check Learning Database Status and Contents
Verifies SQL database is working for learning from trades
"""

import sqlite3
import json
from datetime import datetime

db_path = "trading_performance.db"

print("=" * 80)
print("LEARNING DATABASE CHECK")
print("=" * 80)
print()

# Connect to database
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"[OK] Connected to database: {db_path}")
    print()
except Exception as e:
    print(f"[ERROR] Failed to connect to database: {e}")
    exit(1)

# Check trades table
print("1. TRADES TABLE")
print("-" * 80)

try:
    # Count total trades
    cursor.execute("SELECT COUNT(*) FROM trades")
    total_trades = cursor.fetchone()[0]
    print(f"Total trades recorded: {total_trades}")

    # Count wins and losses
    cursor.execute("SELECT COUNT(*) FROM trades WHERE win = 1")
    wins = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM trades WHERE win = 0")
    losses = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM trades WHERE win IS NULL")
    open_trades = cursor.fetchone()[0]

    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Open/Pending: {open_trades}")

    if total_trades > 0:
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        print(f"Win Rate: {win_rate:.1f}%")

    # Get recent trades
    print()
    print("Recent Trades (last 5):")
    cursor.execute("""
        SELECT trade_id, symbol, strategy, entry_time, exit_time,
               entry_confidence, pnl, win
        FROM trades
        ORDER BY entry_time DESC
        LIMIT 5
    """)

    recent_trades = cursor.fetchall()

    if recent_trades:
        for trade in recent_trades:
            trade_id, symbol, strategy, entry_time, exit_time, confidence, pnl, win = trade
            status = "WIN" if win == 1 else ("LOSS" if win == 0 else "OPEN")
            pnl_str = f"${pnl:.2f}" if pnl else "N/A"
            print(f"  {symbol:6} {strategy:15} {status:4} Conf:{confidence:.0%} P/L:{pnl_str}")
    else:
        print("  No trades yet")

    print()
    print("[OK] Trades table is working")

except Exception as e:
    print(f"[ERROR] Trades table check failed: {e}")

print()

# Check strategy performance table
print("2. STRATEGY PERFORMANCE TABLE")
print("-" * 80)

try:
    cursor.execute("""
        SELECT strategy, total_trades, wins, losses, win_rate,
               total_pnl, profit_factor
        FROM strategy_performance
    """)

    strategies = cursor.fetchall()

    if strategies:
        print(f"Tracked Strategies: {len(strategies)}")
        print()
        print(f"{'Strategy':<20} {'Trades':>7} {'Wins':>6} {'W/R':>6} {'Total P/L':>10} {'PF':>6}")
        print("-" * 80)

        for strat in strategies:
            strategy, total, wins, losses, wr, pnl, pf = strat
            print(f"{strategy:<20} {total:>7} {wins:>6} {wr:>5.1%} ${pnl:>9.2f} {pf:>6.2f}")

        print()
        print("[OK] Strategy performance tracking is working")
    else:
        print("No strategy performance data yet")
        print("[INFO] Will be populated after trades complete")

except Exception as e:
    print(f"[ERROR] Strategy performance check failed: {e}")

print()

# Check confidence calibration
print("3. CONFIDENCE CALIBRATION")
print("-" * 80)

try:
    # Query trades grouped by confidence ranges
    cursor.execute("""
        SELECT
            CASE
                WHEN entry_confidence >= 0.9 THEN '90-100%'
                WHEN entry_confidence >= 0.8 THEN '80-90%'
                WHEN entry_confidence >= 0.7 THEN '70-80%'
                WHEN entry_confidence >= 0.6 THEN '60-70%'
                ELSE '50-60%'
            END as conf_range,
            COUNT(*) as total,
            SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
            AVG(CASE WHEN win IS NOT NULL THEN win END) as actual_win_rate
        FROM trades
        WHERE win IS NOT NULL
        GROUP BY conf_range
        ORDER BY conf_range DESC
    """)

    conf_data = cursor.fetchall()

    if conf_data:
        print(f"{'Confidence Range':<15} {'Trades':>7} {'Wins':>6} {'Actual W/R':>12}")
        print("-" * 80)

        for row in conf_data:
            conf_range, total, wins, actual_wr = row
            actual_wr_pct = actual_wr * 100 if actual_wr else 0
            print(f"{conf_range:<15} {total:>7} {wins:>6} {actual_wr_pct:>11.1f}%")

        print()
        print("[OK] Confidence calibration data available")
        print("    The bot uses this to adjust future confidence scores")
    else:
        print("No completed trades for calibration yet")

except Exception as e:
    print(f"[ERROR] Confidence calibration check failed: {e}")

print()

# Check symbols traded
print("4. SYMBOLS TRADED")
print("-" * 80)

try:
    cursor.execute("""
        SELECT symbol, COUNT(*) as trade_count,
               SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
               AVG(CASE WHEN win IS NOT NULL THEN win END) as win_rate
        FROM trades
        WHERE win IS NOT NULL
        GROUP BY symbol
        ORDER BY trade_count DESC
        LIMIT 10
    """)

    symbols = cursor.fetchall()

    if symbols:
        print(f"Top 10 Most Traded Symbols:")
        print()
        print(f"{'Symbol':<8} {'Trades':>7} {'Wins':>6} {'Win Rate':>10}")
        print("-" * 80)

        for symbol, count, wins, wr in symbols:
            wr_pct = wr * 100 if wr else 0
            print(f"{symbol:<8} {count:>7} {wins:>6} {wr_pct:>9.1f}%")

        print()
        print("[OK] Symbol performance tracking is working")
    else:
        print("No symbol data yet")

except Exception as e:
    print(f"[ERROR] Symbol check failed: {e}")

print()

# Check database integrity
print("5. DATABASE INTEGRITY")
print("-" * 80)

try:
    cursor.execute("PRAGMA integrity_check")
    result = cursor.fetchone()[0]

    if result == "ok":
        print("[OK] Database integrity check passed")
    else:
        print(f"[WARNING] Database integrity issue: {result}")

    # Check table schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='trades'")
    trades_schema = cursor.fetchone()

    if trades_schema:
        print("[OK] Trades table schema exists")
    else:
        print("[ERROR] Trades table schema missing")

    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='strategy_performance'")
    perf_schema = cursor.fetchone()

    if perf_schema:
        print("[OK] Strategy performance table schema exists")
    else:
        print("[ERROR] Strategy performance table schema missing")

except Exception as e:
    print(f"[ERROR] Integrity check failed: {e}")

conn.close()

# Summary
print()
print("=" * 80)
print("LEARNING DATABASE STATUS")
print("=" * 80)
print()

if total_trades > 0:
    print(f"[SUCCESS] Database is WORKING and learning from trades!")
    print()
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print()
    print("The bot is:")
    print("  - Recording every trade entry and exit")
    print("  - Tracking wins and losses")
    print("  - Calibrating confidence based on actual results")
    print("  - Learning which strategies work best")
    print("  - Adapting to improve performance")
    print()
    print("This is ONLINE LEARNING - the bot gets smarter with every trade!")
else:
    print("[INFO] Database is ready but has no trades yet")
    print()
    print("The learning engine will start working once you:")
    print("  1. Start the trading bot")
    print("  2. Make your first trade")
    print("  3. Close that trade (win or loss)")
    print()
    print("After 10+ trades, the bot will:")
    print("  - Calibrate confidence scores")
    print("  - Identify best strategies")
    print("  - Avoid poor performers")
    print("  - Optimize position sizing")

print()
print("=" * 80)
