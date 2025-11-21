#!/usr/bin/env python3
"""
DAILY PERFORMANCE REPORT
=========================
Comprehensive daily performance report for all trading systems

REPORTS:
- Each system's contribution (Forex, Options)
- Combined ROI and P&L
- Win rate and trade statistics
- Monthly projections
- Risk metrics
"""

import os
import json
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class DailyPerformanceReport:
    """Generate comprehensive daily performance reports"""

    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        print("[PERFORMANCE REPORT] Initialized")

    def get_account_summary(self) -> Dict:
        """Get current account summary"""

        try:
            account = self.api.get_account()

            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'last_equity': float(account.last_equity),
                'initial_cash': float(getattr(account, 'initial_cash', account.equity))
            }

        except Exception as e:
            print(f"[ERROR] Could not fetch account: {e}")
            return {}

    def get_todays_pnl(self) -> Dict:
        """Calculate today's P&L"""

        try:
            account = self.api.get_account()

            equity = float(account.equity)
            last_equity = float(account.last_equity)

            pnl = equity - last_equity
            pnl_pct = (pnl / last_equity) * 100 if last_equity > 0 else 0

            return {
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'equity_start': last_equity,
                'equity_end': equity
            }

        except Exception as e:
            print(f"[ERROR] Could not calculate P&L: {e}")
            return {'pnl': 0, 'pnl_pct': 0}

    def get_todays_trades(self) -> List[Dict]:
        """Get all trades executed today"""

        try:
            # Get today's start
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # Get orders
            orders = self.api.list_orders(
                status='all',
                limit=500,
                after=today.isoformat()
            )

            trades = []

            for order in orders:
                if order.filled_at:
                    trades.append({
                        'symbol': order.symbol,
                        'side': order.side,
                        'qty': float(order.filled_qty) if order.filled_qty else 0,
                        'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                        'status': order.status,
                        'filled_at': order.filled_at,
                        'order_type': order.order_type,
                        'time_in_force': order.time_in_force
                    })

            return trades

        except Exception as e:
            print(f"[ERROR] Could not fetch trades: {e}")
            return []

    def get_current_positions(self) -> List[Dict]:
        """Get all current positions"""

        try:
            positions = self.api.list_positions()

            return [{
                'symbol': p.symbol,
                'qty': float(p.qty),
                'avg_entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc),
                'side': p.side
            } for p in positions]

        except Exception as e:
            print(f"[ERROR] Could not fetch positions: {e}")
            return []

    def categorize_trades_by_system(self, trades: List[Dict]) -> Dict:
        """Categorize trades by trading system (Forex vs Options)"""

        forex_trades = []
        options_trades = []
        stock_trades = []

        for trade in trades:
            symbol = trade['symbol']

            # Forex pairs contain underscore (EUR_USD)
            if '_' in symbol:
                forex_trades.append(trade)
            # Options have specific format with dates
            elif len(symbol) > 6 and any(char.isdigit() for char in symbol):
                options_trades.append(trade)
            else:
                stock_trades.append(trade)

        return {
            'forex': forex_trades,
            'options': options_trades,
            'stocks': stock_trades
        }

    def calculate_win_rate(self, positions: List[Dict]) -> Dict:
        """Calculate win rate from closed positions"""

        # Note: This is a simplified win rate calculation
        # For accurate win rate, need to track closed positions over time

        if not positions:
            return {'win_rate': 0, 'winning': 0, 'losing': 0, 'total': 0}

        winning = sum(1 for p in positions if p['unrealized_pl'] > 0)
        losing = sum(1 for p in positions if p['unrealized_pl'] <= 0)
        total = len(positions)

        win_rate = (winning / total * 100) if total > 0 else 0

        return {
            'win_rate': win_rate,
            'winning': winning,
            'losing': losing,
            'total': total
        }

    def project_monthly_returns(self, daily_pnl_pct: float) -> Dict:
        """Project monthly returns based on today's performance"""

        # Assume 20 trading days per month
        trading_days_per_month = 20

        # Simple projection (not compounded)
        simple_monthly = daily_pnl_pct * trading_days_per_month

        # Compounded projection
        daily_return = daily_pnl_pct / 100
        compound_monthly = ((1 + daily_return) ** trading_days_per_month - 1) * 100

        return {
            'simple_monthly': simple_monthly,
            'compound_monthly': compound_monthly,
            'daily_return': daily_pnl_pct
        }

    def generate_report(self) -> Dict:
        """Generate comprehensive daily performance report"""

        print("\n" + "=" * 80)
        print("DAILY PERFORMANCE REPORT")
        print("=" * 80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Time: {datetime.now().strftime('%I:%M:%S %p')}")
        print("=" * 80)

        # Get data
        account = self.get_account_summary()
        todays_pnl = self.get_todays_pnl()
        trades = self.get_todays_trades()
        positions = self.get_current_positions()

        # Categorize trades
        categorized = self.categorize_trades_by_system(trades)

        # Calculate metrics
        win_rate = self.calculate_win_rate(positions)
        projections = self.project_monthly_returns(todays_pnl['pnl_pct'])

        # Print report
        print(f"\n[ACCOUNT SUMMARY]")
        print(f"  Equity: ${account.get('equity', 0):,.2f}")
        print(f"  Cash: ${account.get('cash', 0):,.2f}")
        print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")

        print(f"\n[TODAY'S PERFORMANCE]")
        print(f"  P&L: ${todays_pnl['pnl']:,.2f} ({todays_pnl['pnl_pct']:+.2f}%)")
        print(f"  Starting Equity: ${todays_pnl.get('equity_start', 0):,.2f}")
        print(f"  Ending Equity: ${todays_pnl.get('equity_end', 0):,.2f}")

        print(f"\n[TRADING ACTIVITY]")
        print(f"  Total Trades: {len(trades)}")
        print(f"    Forex: {len(categorized['forex'])} trades")
        print(f"    Options: {len(categorized['options'])} trades")
        print(f"    Stocks: {len(categorized['stocks'])} trades")

        print(f"\n[CURRENT POSITIONS]")
        print(f"  Open Positions: {len(positions)}")

        total_unrealized_pl = sum(p['unrealized_pl'] for p in positions)
        print(f"  Total Unrealized P&L: ${total_unrealized_pl:,.2f}")

        if positions:
            print(f"\n  Top 5 Positions:")
            sorted_positions = sorted(positions, key=lambda x: abs(x['unrealized_pl']), reverse=True)
            for i, pos in enumerate(sorted_positions[:5], 1):
                print(f"    {i}. {pos['symbol']}: {pos['qty']} @ ${pos['avg_entry_price']:.2f}")
                print(f"       Current: ${pos['current_price']:.2f} | P&L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']*100:+.2f}%)")

        print(f"\n[WIN RATE]")
        print(f"  Current Win Rate: {win_rate['win_rate']:.1f}%")
        print(f"  Winning: {win_rate['winning']} positions")
        print(f"  Losing: {win_rate['losing']} positions")

        print(f"\n[MONTHLY PROJECTION]")
        print(f"  Daily Return: {projections['daily_return']:+.2f}%")
        print(f"  Simple Monthly: {projections['simple_monthly']:+.2f}%")
        print(f"  Compound Monthly: {projections['compound_monthly']:+.2f}%")

        # Target assessment
        target_met = "YES" if projections['compound_monthly'] >= 7.0 else "NO"
        print(f"\n[TARGET ASSESSMENT]")
        print(f"  Target: 7-11% monthly")
        print(f"  Projected: {projections['compound_monthly']:+.2f}% monthly")
        print(f"  On Track: {target_met}")

        # System contribution breakdown
        print(f"\n[SYSTEM CONTRIBUTIONS]")

        # Estimate forex contribution (simplified - would need better tracking)
        forex_positions = [p for p in positions if '_' in p['symbol']]
        forex_pl = sum(p['unrealized_pl'] for p in forex_positions)

        # Estimate options contribution
        options_positions = [p for p in positions if '_' not in p['symbol'] and len(p['symbol']) > 6]
        options_pl = sum(p['unrealized_pl'] for p in options_positions)

        # Stock contribution
        stock_positions = [p for p in positions if p not in forex_positions and p not in options_positions]
        stock_pl = sum(p['unrealized_pl'] for p in stock_positions)

        print(f"  Forex Elite: ${forex_pl:,.2f} ({len(forex_positions)} positions)")
        print(f"  Adaptive Options: ${options_pl:,.2f} ({len(options_positions)} positions)")
        print(f"  Stocks: ${stock_pl:,.2f} ({len(stock_positions)} positions)")

        print("\n" + "=" * 80)
        print("Report Complete")
        print("=" * 80)

        # Return report data
        report = {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'account': account,
            'todays_pnl': todays_pnl,
            'trades': {
                'total': len(trades),
                'by_system': {
                    'forex': len(categorized['forex']),
                    'options': len(categorized['options']),
                    'stocks': len(categorized['stocks'])
                }
            },
            'positions': {
                'count': len(positions),
                'unrealized_pl': total_unrealized_pl
            },
            'win_rate': win_rate,
            'projections': projections,
            'system_contributions': {
                'forex': {'pl': forex_pl, 'positions': len(forex_positions)},
                'options': {'pl': options_pl, 'positions': len(options_positions)},
                'stocks': {'pl': stock_pl, 'positions': len(stock_positions)}
            }
        }

        return report

    def save_report(self, report: Dict, filename=None):
        """Save report to file"""

        if filename is None:
            filename = f"daily_performance_{datetime.now().strftime('%Y%m%d')}.json"

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n[SAVED] Report saved to: {filename}")


def main():
    """Generate and display daily performance report"""

    reporter = DailyPerformanceReport()
    report = reporter.generate_report()
    reporter.save_report(report)


if __name__ == "__main__":
    main()
