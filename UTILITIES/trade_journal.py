"""
TRADE JOURNAL / LOGGING SYSTEM
Captures every trade decision with full context for analysis
"""
import json
import os
from datetime import datetime
from pathlib import Path

class TradeJournal:
    """Log all trading decisions with complete context"""

    def __init__(self, journal_file='trade_journal.json'):
        self.journal_file = journal_file
        self.ensure_journal_exists()

    def ensure_journal_exists(self):
        """Create journal file if it doesn't exist"""
        if not os.path.exists(self.journal_file):
            with open(self.journal_file, 'w') as f:
                json.dump([], f)

    def log_trade_entry(self, trade_data):
        """
        Log when a trade is entered

        Args:
            trade_data: dict containing:
                - timestamp: ISO datetime string
                - pair: str (e.g., 'EUR_USD')
                - direction: 'long' or 'short'
                - entry_price: float
                - position_size: int (units)
                - stop_loss: float
                - take_profit: float
                - technical_score: float (0-10)
                - fundamental_score: int (-6 to +6)
                - combined_confidence: float (0-1)
                - kelly_fraction: float
                - kelly_multiplier: float
                - win_probability: float (from Kelly calculation)
                - signals: list of signal names that triggered
                - fundamental_reasons: list of fundamental factors
                - rsi: float
                - macd: str ('bullish' or 'bearish')
                - adx: float (trend strength)
                - volatility: float
                - session: str ('london', 'ny', 'asian', 'overlap')
                - account_balance: float (before trade)
                - trade_id: str (OANDA trade ID when available)
        """
        # Load existing trades
        trades = self._load_journal()

        # Add entry log
        entry = {
            'type': 'ENTRY',
            **trade_data,
            'status': 'open'
        }

        trades.append(entry)
        self._save_journal(trades)

        print(f"[JOURNAL] Trade entry logged: {trade_data['pair']} {trade_data['direction'].upper()}")

    def log_trade_exit(self, exit_data):
        """
        Log when a trade is exited

        Args:
            exit_data: dict containing:
                - trade_id: str (OANDA trade ID)
                - exit_timestamp: ISO datetime
                - exit_price: float
                - exit_reason: str ('take_profit', 'stop_loss', 'trailing_stop', 'manual', 'account_risk_manager')
                - pnl: float (profit/loss in dollars)
                - pnl_pct: float (% of account)
                - duration: str (time held, e.g., '4h 23m')
                - peak_profit: float (max unrealized profit during trade)
                - peak_loss: float (max unrealized loss during trade)
                - outcome: str ('win' or 'loss')
        """
        trades = self._load_journal()

        # Find the matching entry
        for trade in reversed(trades):
            if trade.get('trade_id') == exit_data.get('trade_id') and trade.get('status') == 'open':
                # Update with exit data
                trade.update({
                    'type': 'COMPLETE',
                    'status': 'closed',
                    **exit_data
                })

                self._save_journal(trades)
                print(f"[JOURNAL] Trade exit logged: {exit_data.get('trade_id')} - {exit_data.get('outcome').upper()}")
                return

        # If no matching entry found, log as standalone exit
        exit_entry = {
            'type': 'EXIT',
            **exit_data,
            'status': 'closed',
            'note': 'No matching entry found (possibly from previous session)'
        }
        trades.append(exit_entry)
        self._save_journal(trades)

    def log_signal_scan(self, scan_data):
        """
        Log scanning results even when no trade taken
        Useful for analyzing why bot didn't trade

        Args:
            scan_data: dict containing:
                - timestamp: ISO datetime
                - pairs_scanned: list of pairs
                - opportunities_found: int
                - opportunities: list of dicts with pair, score, reason
                - trades_executed: int
                - rejected_reason: str (if opportunities rejected)
        """
        trades = self._load_journal()

        scan_entry = {
            'type': 'SCAN',
            **scan_data
        }

        trades.append(scan_entry)
        self._save_journal(trades)

    def get_all_trades(self):
        """Get all trades from journal"""
        return [t for t in self._load_journal() if t.get('type') in ['COMPLETE', 'EXIT']]

    def get_open_trades(self):
        """Get currently open trades"""
        return [t for t in self._load_journal() if t.get('status') == 'open']

    def get_trade_by_id(self, trade_id):
        """Get specific trade by OANDA trade ID"""
        trades = self._load_journal()
        for trade in reversed(trades):
            if trade.get('trade_id') == trade_id:
                return trade
        return None

    def get_trades_by_pair(self, pair):
        """Get all completed trades for a specific pair"""
        all_trades = self.get_all_trades()
        return [t for t in all_trades if t.get('pair') == pair]

    def get_trades_by_outcome(self, outcome):
        """Get trades by outcome ('win' or 'loss')"""
        all_trades = self.get_all_trades()
        return [t for t in all_trades if t.get('outcome') == outcome]

    def get_statistics_by_setup(self):
        """
        Analyze performance by setup type
        Returns win rates for different signal combinations
        """
        all_trades = self.get_all_trades()

        if not all_trades:
            return {}

        # Analyze by signal combinations
        stats = {}

        for trade in all_trades:
            signals = tuple(sorted(trade.get('signals', [])))

            if signals not in stats:
                stats[signals] = {
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0
                }

            stats[signals]['count'] += 1

            if trade.get('outcome') == 'win':
                stats[signals]['wins'] += 1
            else:
                stats[signals]['losses'] += 1

            stats[signals]['total_pnl'] += trade.get('pnl', 0)

        # Calculate averages
        for signals, data in stats.items():
            data['win_rate'] = (data['wins'] / data['count']) * 100 if data['count'] > 0 else 0
            data['avg_pnl'] = data['total_pnl'] / data['count'] if data['count'] > 0 else 0

        return stats

    def print_recent_trades(self, limit=10):
        """Print recent trades with full details"""
        all_trades = self.get_all_trades()

        if not all_trades:
            print("[JOURNAL] No trades recorded yet")
            return

        print(f"\n{'='*70}")
        print(f"RECENT TRADES (Last {limit})")
        print(f"{'='*70}\n")

        for trade in all_trades[-limit:]:
            print(f"[{trade.get('timestamp', 'N/A')}]")
            print(f"  Pair: {trade.get('pair', 'N/A')} {trade.get('direction', '').upper()}")
            print(f"  Entry: {trade.get('entry_price', 0):.5f} -> Exit: {trade.get('exit_price', 0):.5f}")
            print(f"  Position: {trade.get('position_size', 0):,} units")
            print(f"  Scores: Tech {trade.get('technical_score', 0):.1f}/10 | Fund {trade.get('fundamental_score', 0)}/6")
            print(f"  Signals: {', '.join(trade.get('signals', []))}")
            print(f"  Kelly: {trade.get('kelly_fraction', 0):.3f} (Win Prob: {trade.get('win_probability', 0)*100:.1f}%)")
            print(f"  Duration: {trade.get('duration', 'N/A')}")
            print(f"  P/L: ${trade.get('pnl', 0):+,.2f} ({trade.get('outcome', 'N/A').upper()})")
            print(f"  Exit Reason: {trade.get('exit_reason', 'N/A')}")
            print()

    def _load_journal(self):
        """Load journal from JSON file"""
        try:
            with open(self.journal_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_journal(self, trades):
        """Save journal to JSON file"""
        with open(self.journal_file, 'w') as f:
            json.dump(trades, f, indent=2)


def test_journal():
    """Test the trade journal"""
    journal = TradeJournal('test_journal.json')

    # Test entry logging
    entry_data = {
        'timestamp': datetime.now().isoformat(),
        'pair': 'EUR_USD',
        'direction': 'long',
        'entry_price': 1.0850,
        'position_size': 1200000,
        'stop_loss': 1.0742,
        'take_profit': 1.1067,
        'technical_score': 6.5,
        'fundamental_score': 4,
        'combined_confidence': 0.66,
        'kelly_fraction': 0.131,
        'kelly_multiplier': 1.31,
        'win_probability': 0.682,
        'signals': ['RSI_OVERSOLD', 'MACD_BULLISH', 'STRONG_TREND'],
        'fundamental_reasons': ['FED_HAWKISH', 'ECB_NEUTRAL'],
        'rsi': 38.5,
        'macd': 'bullish',
        'adx': 28.3,
        'volatility': 0.0045,
        'session': 'london',
        'account_balance': 190307.70,
        'trade_id': '12345'
    }

    journal.log_trade_entry(entry_data)

    # Test exit logging
    exit_data = {
        'trade_id': '12345',
        'exit_timestamp': datetime.now().isoformat(),
        'exit_price': 1.0920,
        'exit_reason': 'take_profit',
        'pnl': 1750.00,
        'pnl_pct': 0.92,
        'duration': '4h 23m',
        'peak_profit': 2100.00,
        'peak_loss': -150.00,
        'outcome': 'win'
    }

    journal.log_trade_exit(exit_data)

    # Print trades
    journal.print_recent_trades(5)

    # Get statistics
    stats = journal.get_statistics_by_setup()
    print(f"\n{'='*70}")
    print("SETUP STATISTICS")
    print(f"{'='*70}\n")

    for signals, data in stats.items():
        print(f"Signals: {', '.join(signals)}")
        print(f"  Trades: {data['count']} ({data['wins']}W / {data['losses']}L)")
        print(f"  Win Rate: {data['win_rate']:.1f}%")
        print(f"  Avg P/L: ${data['avg_pnl']:,.2f}")
        print()


if __name__ == "__main__":
    test_journal()
