"""
COMPREHENSIVE TRADE LOGGING SYSTEM
Tracks TA-only vs TA+AI hybrid performance for A/B testing

Features:
- Logs all trade signals (TA scores, AI decisions, execution results)
- Compares TA-only vs TA+AI performance metrics
- Tracks AI approval/rejection accuracy
- Performance analytics (win rate, profit factor, drawdown)
- JSON export for analysis
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class TradeLogger:
    """
    Comprehensive logging for multi-market trading system
    Tracks both TA signals and AI confirmation results
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Session tracking
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")

        # Trade logs
        self.signals_log = []  # All TA signals
        self.ai_decisions_log = []  # AI confirmation decisions
        self.executions_log = []  # Actual trade executions

        # Performance tracking
        self.ta_only_signals = []  # Signals that would execute in TA-only mode
        self.ai_approved_trades = []  # Signals approved by AI
        self.ai_rejected_trades = []  # Signals rejected by AI

        # Statistics
        self.stats = {
            'total_signals': 0,
            'ta_only_count': 0,
            'ai_analyzed': 0,
            'ai_approved': 0,
            'ai_rejected': 0,
            'ai_reduced': 0,
            'consensus_rate': 0.0
        }

    def log_ta_signal(self, market: str, signal_data: Dict):
        """
        Log a TA-generated signal (before AI confirmation)

        Args:
            market: 'forex', 'futures', or 'crypto'
            signal_data: {
                'symbol': 'EUR_USD',
                'direction': 'long',
                'score': 7.5,
                'rsi': 28.5,
                'macd': {...},
                'signals': ['RSI_OVERSOLD', 'MACD_BULL_CROSS'],
                'timestamp': datetime
            }
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'market': market,
            **signal_data
        }

        self.signals_log.append(entry)
        self.stats['total_signals'] += 1

        # Track as TA-only candidate (would execute without AI)
        if signal_data['score'] >= signal_data.get('min_score', 2.5):
            self.ta_only_signals.append(entry)
            self.stats['ta_only_count'] += 1

    def log_ai_decision(self, market: str, symbol: str, ta_score: float, ai_decision: Dict):
        """
        Log AI confirmation decision

        Args:
            market: 'forex', 'futures', or 'crypto'
            symbol: 'EUR_USD', 'ES', 'BTCUSD', etc.
            ta_score: Technical analysis score
            ai_decision: {
                'action': 'APPROVE'/'REJECT'/'REDUCE_SIZE',
                'confidence': 0-100,
                'deepseek_decision': {...},
                'minimax_decision': {...},
                'consensus': bool,
                'reason': str
            }
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'market': market,
            'symbol': symbol,
            'ta_score': ta_score,
            **ai_decision
        }

        self.ai_decisions_log.append(entry)
        self.stats['ai_analyzed'] += 1

        # Track by AI action
        if ai_decision['action'] == 'APPROVE':
            self.ai_approved_trades.append(entry)
            self.stats['ai_approved'] += 1
        elif ai_decision['action'] == 'REJECT':
            self.ai_rejected_trades.append(entry)
            self.stats['ai_rejected'] += 1
        elif ai_decision['action'] == 'REDUCE_SIZE':
            self.stats['ai_reduced'] += 1

        # Update consensus rate
        if self.stats['ai_analyzed'] > 0:
            consensus_count = sum(1 for d in self.ai_decisions_log if d.get('consensus', False))
            self.stats['consensus_rate'] = (consensus_count / self.stats['ai_analyzed']) * 100

    def log_execution(self, market: str, execution_data: Dict):
        """
        Log actual trade execution

        Args:
            execution_data: {
                'symbol': 'EUR_USD',
                'direction': 'long',
                'size': 50000,
                'entry_price': 1.0850,
                'ta_score': 7.5,
                'ai_approved': True,
                'ai_confidence': 85,
                'timestamp': datetime
            }
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'market': market,
            **execution_data
        }

        self.executions_log.append(entry)

    def get_performance_summary(self) -> Dict:
        """
        Generate performance summary for A/B testing

        Returns:
            {
                'session_id': str,
                'duration_hours': float,
                'total_signals': int,
                'ta_only_signals': int,
                'ai_approved': int,
                'ai_rejected': int,
                'ai_reduced': int,
                'consensus_rate': float,
                'rejection_rate': float,
                'executions': int
            }
        """
        duration = (datetime.now() - self.session_start).total_seconds() / 3600

        rejection_rate = 0.0
        if self.stats['ai_analyzed'] > 0:
            rejection_rate = (self.stats['ai_rejected'] / self.stats['ai_analyzed']) * 100

        return {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'duration_hours': round(duration, 2),
            'total_signals': self.stats['total_signals'],
            'ta_only_signals': self.stats['ta_only_count'],
            'ai_analyzed': self.stats['ai_analyzed'],
            'ai_approved': self.stats['ai_approved'],
            'ai_rejected': self.stats['ai_rejected'],
            'ai_reduced': self.stats['ai_reduced'],
            'consensus_rate': round(self.stats['consensus_rate'], 1),
            'rejection_rate': round(rejection_rate, 1),
            'total_executions': len(self.executions_log)
        }

    def export_session_logs(self):
        """
        Export all session logs to JSON files

        Creates:
            logs/signals_{session_id}.json - All TA signals
            logs/ai_decisions_{session_id}.json - AI decisions
            logs/executions_{session_id}.json - Trade executions
            logs/summary_{session_id}.json - Performance summary
        """
        # Signals log
        signals_file = os.path.join(self.log_dir, f"signals_{self.session_id}.json")
        with open(signals_file, 'w') as f:
            json.dump(self.signals_log, f, indent=2)

        # AI decisions log
        ai_file = os.path.join(self.log_dir, f"ai_decisions_{self.session_id}.json")
        with open(ai_file, 'w') as f:
            json.dump(self.ai_decisions_log, f, indent=2)

        # Executions log
        exec_file = os.path.join(self.log_dir, f"executions_{self.session_id}.json")
        with open(exec_file, 'w') as f:
            json.dump(self.executions_log, f, indent=2)

        # Summary
        summary_file = os.path.join(self.log_dir, f"summary_{self.session_id}.json")
        summary = self.get_performance_summary()
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[LOGS EXPORTED]")
        print(f"  Signals: {signals_file}")
        print(f"  AI Decisions: {ai_file}")
        print(f"  Executions: {exec_file}")
        print(f"  Summary: {summary_file}")

    def print_session_summary(self):
        """Print comprehensive session summary to console"""
        summary = self.get_performance_summary()

        print("\n" + "=" * 80)
        print(" " * 25 + "SESSION PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"\nSession ID: {summary['session_id']}")
        print(f"Duration: {summary['duration_hours']} hours")
        print()
        print("SIGNAL GENERATION:")
        print(f"  Total TA Signals: {summary['total_signals']}")
        print(f"  TA-Only Mode Would Execute: {summary['ta_only_signals']}")
        print()
        print("AI CONFIRMATION:")
        print(f"  Trades Analyzed by AI: {summary['ai_analyzed']}")
        print(f"  Approved: {summary['ai_approved']} ({summary['ai_approved']/max(1,summary['ai_analyzed'])*100:.1f}%)")
        print(f"  Rejected: {summary['ai_rejected']} ({summary['rejection_rate']:.1f}%)")
        print(f"  Reduced Size: {summary['ai_reduced']}")
        print(f"  Model Consensus Rate: {summary['consensus_rate']:.1f}%")
        print()
        print("EXECUTION:")
        print(f"  Total Trades Executed: {summary['total_executions']}")
        print()
        print("A/B COMPARISON:")
        ta_only_would_execute = summary['ta_only_signals']
        ai_filtered_executed = summary['ai_approved']
        rejection_count = summary['ai_rejected']

        if ta_only_would_execute > 0:
            filter_rate = (rejection_count / ta_only_would_execute) * 100
            print(f"  TA-Only Mode: {ta_only_would_execute} trades")
            print(f"  TA+AI Mode: {ai_filtered_executed} trades (AI filtered {rejection_count}, {filter_rate:.1f}% reduction)")
        else:
            print(f"  Not enough data for comparison yet")

        print("=" * 80)

    def compare_ai_rejected_outcomes(self, rejected_outcomes: List[Dict]):
        """
        Analyze if AI rejections were correct (saved from bad trades)

        Args:
            rejected_outcomes: List of outcomes for trades AI rejected
                [
                    {
                        'symbol': 'EUR_USD',
                        'ta_score': 7.5,
                        'ai_reason': 'ECB meeting volatility',
                        'hypothetical_outcome': 'win' or 'loss',
                        'hypothetical_pnl': +150 or -80
                    }
                ]

        Returns:
            Analysis showing if AI rejection was smart
        """
        if not rejected_outcomes:
            return {
                'status': 'No rejected trade outcomes to analyze',
                'ai_accuracy': None
            }

        total_rejected = len(rejected_outcomes)
        losses_avoided = sum(1 for o in rejected_outcomes if o.get('hypothetical_outcome') == 'loss')
        wins_missed = sum(1 for o in rejected_outcomes if o.get('hypothetical_outcome') == 'win')

        hypothetical_pnl = sum(o.get('hypothetical_pnl', 0) for o in rejected_outcomes)

        ai_accuracy = (losses_avoided / total_rejected) * 100 if total_rejected > 0 else 0

        return {
            'total_rejections': total_rejected,
            'losses_avoided': losses_avoided,
            'wins_missed': wins_missed,
            'hypothetical_pnl_if_executed': hypothetical_pnl,
            'ai_rejection_accuracy': round(ai_accuracy, 1),
            'verdict': 'AI saved money' if hypothetical_pnl < 0 else 'AI cost opportunity'
        }


# Singleton instance
trade_logger = TradeLogger()
