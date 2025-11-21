"""
TRADE LOGGER & MONTE CARLO SIMULATOR
Logs all trades from FOREX + STOCKS systems
Enables Monte Carlo analysis after 50+ trades collected
"""
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np

class TradeLogger:
    """Logs every trade for later Monte Carlo analysis"""

    def __init__(self):
        self.log_dir = Path("trade_logs")
        self.log_dir.mkdir(exist_ok=True)

        self.log_file = self.log_dir / "all_trades.json"

        # Load existing trades
        self.trades = self._load_trades()

        print("=" * 70)
        print("TRADE LOGGER - MONTE CARLO READY")
        print("=" * 70)
        print(f"Total trades logged: {len(self.trades)}")
        print(f"Log file: {self.log_file}")
        print("=" * 70)

    def _load_trades(self):
        """Load existing trade log"""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []

    def _save_trades(self):
        """Save trades to disk"""
        with open(self.log_file, 'w') as f:
            json.dump(self.trades, f, indent=2)

    def log_trade(self, trade_data):
        """
        Log a single trade

        trade_data = {
            'timestamp': '2025-10-21 20:48:45',
            'market': 'FOREX' or 'STOCKS',
            'symbol': 'EUR_USD' or 'TSLA',
            'direction': 'LONG' or 'SHORT',
            'entry_price': 1.16056,
            'exit_price': 1.18377,  # Fill when closed
            'quantity': 1000,
            'pnl': 23.21,  # Fill when closed
            'pnl_pct': 0.02,  # 2%
            'score': 3.5,
            'signals': ['RSI_OVERSOLD', 'STRONG_TREND'],
            'status': 'OPEN' or 'CLOSED',
            'win': True/False  # Fill when closed
        }
        """
        trade_data['logged_at'] = datetime.now().isoformat()
        self.trades.append(trade_data)
        self._save_trades()

        print(f"[TRADE LOGGED] {trade_data['market']} {trade_data['symbol']} - Total: {len(self.trades)}")

    def get_stats(self):
        """Get basic trade statistics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }

        closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED']

        if not closed_trades:
            return {
                'total_trades': len(self.trades),
                'open_trades': len(self.trades),
                'closed_trades': 0
            }

        wins = [t for t in closed_trades if t.get('win', False)]
        losses = [t for t in closed_trades if not t.get('win', False)]

        total_win = sum(t.get('pnl', 0) for t in wins)
        total_loss = abs(sum(t.get('pnl', 0) for t in losses))

        return {
            'total_trades': len(self.trades),
            'open_trades': len(self.trades) - len(closed_trades),
            'closed_trades': len(closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(closed_trades) if closed_trades else 0,
            'avg_win': total_win / len(wins) if wins else 0,
            'avg_loss': total_loss / len(losses) if losses else 0,
            'profit_factor': total_win / total_loss if total_loss > 0 else 0,
            'total_pnl': sum(t.get('pnl', 0) for t in closed_trades)
        }

    def monte_carlo_simulation(self, num_simulations=1000, trades_to_simulate=100):
        """
        Run Monte Carlo simulation on trade data
        Requires at least 50 closed trades
        """
        closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED']

        if len(closed_trades) < 50:
            print(f"[WARNING] Need 50+ closed trades for Monte Carlo. Have: {len(closed_trades)}")
            return None

        print(f"\n[MONTE CARLO] Running {num_simulations:,} simulations...")
        print(f"[MONTE CARLO] Each simulation: {trades_to_simulate} trades")

        # Extract trade returns
        returns = [t.get('pnl_pct', 0) for t in closed_trades]

        # Run simulations
        simulation_results = []

        for sim in range(num_simulations):
            # Randomly sample trades with replacement
            sampled_returns = np.random.choice(returns, size=trades_to_simulate, replace=True)

            # Calculate cumulative return
            cumulative_return = np.prod([1 + r for r in sampled_returns]) - 1

            # Calculate max drawdown
            cumulative = np.cumprod([1 + r for r in sampled_returns])
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)

            simulation_results.append({
                'final_return': cumulative_return,
                'max_drawdown': max_drawdown
            })

        # Analyze results
        final_returns = [s['final_return'] for s in simulation_results]
        max_drawdowns = [s['max_drawdown'] for s in simulation_results]

        results = {
            'simulations': num_simulations,
            'trades_per_sim': trades_to_simulate,
            'expected_return': np.mean(final_returns),
            'median_return': np.median(final_returns),
            'best_case': np.percentile(final_returns, 95),
            'worst_case': np.percentile(final_returns, 5),
            'probability_profit': sum(1 for r in final_returns if r > 0) / num_simulations,
            'expected_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown_95th': np.percentile(max_drawdowns, 5)
        }

        print("\n" + "=" * 70)
        print("MONTE CARLO RESULTS")
        print("=" * 70)
        print(f"Expected Return: {results['expected_return']*100:.2f}%")
        print(f"Median Return: {results['median_return']*100:.2f}%")
        print(f"Best Case (95th): {results['best_case']*100:.2f}%")
        print(f"Worst Case (5th): {results['worst_case']*100:.2f}%")
        print(f"Probability of Profit: {results['probability_profit']*100:.1f}%")
        print(f"Expected Max Drawdown: {results['expected_max_drawdown']*100:.2f}%")
        print(f"Worst Drawdown (95th): {results['worst_drawdown_95th']*100:.2f}%")
        print("=" * 70)

        return results

    def can_hit_50_percent_monthly(self):
        """
        Analyze if 50% monthly ROI is realistic based on trade data
        Requires at least 100 closed trades
        """
        closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED']

        if len(closed_trades) < 100:
            print(f"[INFO] Need 100+ trades to analyze 50% monthly goal. Have: {len(closed_trades)}")
            return None

        # Run Monte Carlo for 1 month (assume 20 trading days, 2-3 trades/day = 50 trades)
        results = self.monte_carlo_simulation(num_simulations=10000, trades_to_simulate=50)

        if not results:
            return None

        print("\n" + "=" * 70)
        print("50% MONTHLY ROI ANALYSIS")
        print("=" * 70)

        expected_monthly = results['expected_return']
        probability_50_percent = sum(1 for s in range(10000) if results > 0.5) / 10000

        print(f"Expected Monthly Return: {expected_monthly*100:.2f}%")
        print(f"Probability of 50%+ month: {probability_50_percent*100:.1f}%")

        if expected_monthly >= 0.50:
            print("✅ 50% monthly ROI is ACHIEVABLE with current strategy!")
        elif expected_monthly >= 0.30:
            print("⚠️  50% is possible but aggressive. Expected: {expected_monthly*100:.1f}%")
        else:
            print(f"❌ 50% monthly unlikely. Expected: {expected_monthly*100:.1f}%")

        print("=" * 70)

        return {
            'expected_monthly': expected_monthly,
            'target_monthly': 0.50,
            'achievable': expected_monthly >= 0.50
        }

def print_current_stats():
    """Print current trade statistics"""
    logger = TradeLogger()
    stats = logger.get_stats()

    print("\n" + "=" * 70)
    print("CURRENT TRADING STATISTICS")
    print("=" * 70)
    print(f"Total Trades: {stats['total_trades']}")

    if stats.get('closed_trades', 0) > 0:
        print(f"Open Trades: {stats['open_trades']}")
        print(f"Closed Trades: {stats['closed_trades']}")
        print(f"Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"Avg Win: ${stats['avg_win']:.2f}")
        print(f"Avg Loss: ${stats['avg_loss']:.2f}")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Total P&L: ${stats['total_pnl']:.2f}")

        if stats['closed_trades'] >= 50:
            print("\n✅ Enough data for Monte Carlo simulation!")
            print("Run: logger.monte_carlo_simulation()")
        else:
            print(f"\n⏳ Need {50 - stats['closed_trades']} more closed trades for Monte Carlo")
    else:
        print("No closed trades yet. Keep trading!")

    print("=" * 70)

if __name__ == "__main__":
    print_current_stats()
