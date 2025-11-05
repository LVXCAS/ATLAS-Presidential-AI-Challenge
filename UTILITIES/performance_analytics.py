"""
PERFORMANCE ANALYTICS DASHBOARD
Real-time tracking of trading system edge and E8 compliance
"""
import json
import os
from datetime import datetime
from pathlib import Path
import statistics
import math

class PerformanceAnalytics:
    """Track and analyze trading performance metrics"""

    def __init__(self, trades_file='trade_history.json'):
        self.trades_file = trades_file
        self.trades = self.load_trades()

    def load_trades(self):
        """Load trade history from JSON file"""
        if os.path.exists(self.trades_file):
            with open(self.trades_file, 'r') as f:
                return json.load(f)
        return []

    def save_trades(self):
        """Save trade history to JSON file"""
        with open(self.trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2)

    def add_trade(self, trade_data):
        """
        Add a new trade to history

        Args:
            trade_data: dict with keys:
                - timestamp: ISO format datetime
                - pair: str (e.g., 'EUR_USD')
                - direction: 'long' or 'short'
                - entry: float (entry price)
                - exit: float (exit price)
                - size: int (units)
                - pnl: float (profit/loss in dollars)
                - duration: str (time held)
                - outcome: 'win' or 'loss'
                - technical_score: float (0-10)
                - fundamental_score: int (-6 to +6)
                - kelly_fraction: float
                - signals: list of signal names
        """
        self.trades.append(trade_data)
        self.save_trades()

    def get_win_rate(self):
        """Calculate win rate percentage"""
        if not self.trades:
            return 0.0

        wins = sum(1 for t in self.trades if t.get('outcome') == 'win')
        total = len(self.trades)

        return (wins / total) * 100

    def get_profit_factor(self):
        """
        Calculate profit factor (gross wins / gross losses)
        >1.0 = profitable, >1.5 = good, >2.0 = excellent
        """
        wins = [t['pnl'] for t in self.trades if t.get('outcome') == 'win']
        losses = [abs(t['pnl']) for t in self.trades if t.get('outcome') == 'loss']

        if not wins or not losses:
            return 0.0

        gross_wins = sum(wins)
        gross_losses = sum(losses)

        if gross_losses == 0:
            return float('inf')

        return gross_wins / gross_losses

    def get_expectancy(self):
        """
        Calculate expectancy (average profit per trade)
        Positive = edge, Negative = no edge
        """
        if not self.trades:
            return 0.0

        total_pnl = sum(t['pnl'] for t in self.trades)
        return total_pnl / len(self.trades)

    def get_sharpe_ratio(self, risk_free_rate=0.05):
        """
        Calculate Sharpe Ratio (risk-adjusted returns)
        >1.0 = good, >1.5 = very good, >2.0 = excellent

        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        if len(self.trades) < 2:
            return 0.0

        # Calculate returns as percentage of account
        # Assuming starting balance (will be passed from main bot)
        returns = [t['pnl'] for t in self.trades]

        if not returns:
            return 0.0

        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0

        if std_return == 0:
            return 0.0

        # Annualized Sharpe (assuming ~250 trading days)
        # Daily risk-free rate
        daily_rf = risk_free_rate / 252

        sharpe = (avg_return - daily_rf) / std_return

        # Annualize it
        sharpe_annualized = sharpe * math.sqrt(252)

        return sharpe_annualized

    def get_max_drawdown(self, starting_balance):
        """
        Calculate maximum drawdown from peak equity
        E8 limit: -6%
        """
        if not self.trades:
            return 0.0

        # Calculate equity curve
        equity = starting_balance
        peak = starting_balance
        max_dd = 0.0

        for trade in self.trades:
            equity += trade['pnl']

            if equity > peak:
                peak = equity

            drawdown = ((peak - equity) / peak) * 100

            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def get_average_win_loss(self):
        """Calculate average win and average loss"""
        wins = [t['pnl'] for t in self.trades if t.get('outcome') == 'win']
        losses = [abs(t['pnl']) for t in self.trades if t.get('outcome') == 'loss']

        avg_win = statistics.mean(wins) if wins else 0.0
        avg_loss = statistics.mean(losses) if losses else 0.0

        return avg_win, avg_loss

    def get_risk_reward_ratio(self):
        """Calculate average risk:reward ratio"""
        avg_win, avg_loss = self.get_average_win_loss()

        if avg_loss == 0:
            return 0.0

        return avg_win / avg_loss

    def get_consecutive_stats(self):
        """Track consecutive wins/losses (psychological metric)"""
        if not self.trades:
            return {'max_win_streak': 0, 'max_loss_streak': 0, 'current_streak': 0}

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in self.trades:
            if trade.get('outcome') == 'win':
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        # Current streak
        if self.trades[-1].get('outcome') == 'win':
            current_streak = current_wins
        else:
            current_streak = -current_losses

        return {
            'max_win_streak': max_wins,
            'max_loss_streak': max_losses,
            'current_streak': current_streak
        }

    def print_dashboard(self, starting_balance, current_balance):
        """Print comprehensive performance dashboard"""
        print("\n" + "="*70)
        print("PERFORMANCE ANALYTICS DASHBOARD")
        print("="*70)

        if not self.trades:
            print("\nNo trades recorded yet. Waiting for first trade...")
            print("="*70 + "\n")
            return

        # Account Overview
        total_pnl = sum(t['pnl'] for t in self.trades)
        roi = (total_pnl / starting_balance) * 100

        print(f"\nACCOUNT OVERVIEW")
        print(f"  Starting Balance: ${starting_balance:,.2f}")
        print(f"  Current Balance:  ${current_balance:,.2f}")
        print(f"  Total P/L:        ${total_pnl:,.2f} ({roi:+.2f}%)")
        print(f"  Total Trades:     {len(self.trades)}")

        # Win/Loss Stats
        wins = sum(1 for t in self.trades if t.get('outcome') == 'win')
        losses = sum(1 for t in self.trades if t.get('outcome') == 'loss')
        win_rate = self.get_win_rate()

        print(f"\nWIN/LOSS STATISTICS")
        print(f"  Win Rate:         {win_rate:.1f}% ({wins}W / {losses}L)")

        avg_win, avg_loss = self.get_average_win_loss()
        print(f"  Average Win:      ${avg_win:,.2f}")
        print(f"  Average Loss:     ${avg_loss:,.2f}")

        rr_ratio = self.get_risk_reward_ratio()
        print(f"  Risk:Reward:      1:{rr_ratio:.2f}")

        # Performance Metrics
        profit_factor = self.get_profit_factor()
        expectancy = self.get_expectancy()
        sharpe = self.get_sharpe_ratio()
        max_dd = self.get_max_drawdown(starting_balance)

        print(f"\nPERFORMANCE METRICS")
        print(f"  Profit Factor:    {profit_factor:.2f} {'[Good]' if profit_factor >= 1.5 else '[Needs Improvement]'}")
        print(f"  Expectancy:       ${expectancy:,.2f} per trade {'[Positive Edge]' if expectancy > 0 else '[No Edge]'}")
        print(f"  Sharpe Ratio:     {sharpe:.2f} {'[Excellent]' if sharpe >= 2.0 else '[Good]' if sharpe >= 1.5 else '[Risky]'}")
        print(f"  Max Drawdown:     {max_dd:.2f}% {'[Safe]' if max_dd < 4.0 else '[Caution]' if max_dd < 6.0 else '[DANGER]'}")

        # Consecutive Stats
        streaks = self.get_consecutive_stats()
        print(f"\nSTREAK ANALYSIS")
        print(f"  Max Win Streak:   {streaks['max_win_streak']} trades")
        print(f"  Max Loss Streak:  {streaks['max_loss_streak']} trades")

        current = streaks['current_streak']
        if current > 0:
            print(f"  Current Streak:   {current} wins [HOT]")
        elif current < 0:
            print(f"  Current Streak:   {abs(current)} losses [WARNING]")
        else:
            print(f"  Current Streak:   None")

        # E8 Compliance
        print(f"\nE8 COMPLIANCE CHECK")
        print(f"  Max Drawdown:     {max_dd:.2f}% / 6.00% limit")

        if max_dd < 4.0:
            status = "[SAFE - 2% buffer]"
        elif max_dd < 6.0:
            status = "[CAUTION - approaching limit]"
        else:
            status = "[VIOLATION - would fail E8]"

        print(f"  Status:           {status}")

        # Recent Trades (last 5)
        print(f"\nRECENT TRADES (Last 5)")
        print(f"  {'Pair':<10} {'Dir':<6} {'P/L':<12} {'Outcome':<8} {'Duration':<10}")
        print(f"  {'-'*55}")

        for trade in self.trades[-5:]:
            pair = trade.get('pair', 'N/A')
            direction = trade.get('direction', 'N/A').upper()
            pnl = trade.get('pnl', 0)
            outcome = trade.get('outcome', 'N/A').upper()
            duration = trade.get('duration', 'N/A')

            outcome_icon = "[W]" if outcome == "WIN" else "[L]"
            pnl_str = f"${pnl:+,.2f}"

            print(f"  {pair:<10} {direction:<6} {pnl_str:<12} {outcome_icon} {outcome:<6} {duration:<10}")

        print("\n" + "="*70 + "\n")

    def export_report(self, filename='performance_report.txt'):
        """Export performance report to text file"""
        with open(filename, 'w') as f:
            # Redirect print to file
            import sys
            old_stdout = sys.stdout
            sys.stdout = f

            self.print_dashboard(198899.52, 190307.70)  # Will be parameterized

            sys.stdout = old_stdout

        print(f"[EXPORTED] Performance report saved to {filename}")


def test_analytics():
    """Test the analytics with sample data"""
    analytics = PerformanceAnalytics('test_trades.json')

    # Clear previous test data
    analytics.trades = []

    # Add sample trades
    sample_trades = [
        {
            'timestamp': '2025-10-30T10:00:00',
            'pair': 'EUR_USD',
            'direction': 'long',
            'entry': 1.0850,
            'exit': 1.0920,
            'size': 1000000,
            'pnl': 1750.00,
            'duration': '4h 23m',
            'outcome': 'win',
            'technical_score': 6.5,
            'fundamental_score': 4,
            'kelly_fraction': 0.12,
            'signals': ['RSI_OVERSOLD', 'MACD_BULLISH']
        },
        {
            'timestamp': '2025-10-30T16:00:00',
            'pair': 'GBP_USD',
            'direction': 'short',
            'entry': 1.2650,
            'exit': 1.2580,
            'size': 1200000,
            'pnl': 2100.00,
            'duration': '2h 15m',
            'outcome': 'win',
            'technical_score': 7.0,
            'fundamental_score': 5,
            'kelly_fraction': 0.14,
            'signals': ['RSI_OVERBOUGHT', 'TREND_DOWN']
        },
        {
            'timestamp': '2025-10-30T20:00:00',
            'pair': 'USD_JPY',
            'direction': 'short',
            'entry': 153.069,
            'exit': 154.600,
            'size': 1256249,
            'pnl': -8317.15,
            'duration': '8h 45m',
            'outcome': 'loss',
            'technical_score': 4.5,
            'fundamental_score': -5,
            'kelly_fraction': 0.13,
            'signals': ['SHORT_SETUP', 'JPY_STRENGTH']
        }
    ]

    for trade in sample_trades:
        analytics.add_trade(trade)

    # Print dashboard
    analytics.print_dashboard(starting_balance=198899.52, current_balance=190307.70)


if __name__ == "__main__":
    test_analytics()
