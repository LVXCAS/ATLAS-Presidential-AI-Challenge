"""
Trade Logger - Comprehensive trade tracking and performance analysis

Logs every trade with:
- Entry/exit details (price, time, size)
- Agent votes and reasoning
- Kelly Criterion calculations
- P/L and performance metrics
- Exit reason and outcomes
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TradeLog:
    """Complete trade record"""
    # IDs
    trade_id: str
    oanda_trade_ids: List[str]

    # Timestamps
    timestamp_decision: str
    timestamp_entry: Optional[str]
    timestamp_exit: Optional[str]
    duration_minutes: Optional[float]

    # Trade details
    pair: str
    direction: str
    units: int
    lots: float

    # Prices
    entry_price: Optional[float]
    exit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]

    # Performance
    pnl: Optional[float]
    pnl_pct: Optional[float]
    pips: Optional[float]
    r_multiple: Optional[float]

    # ATLAS decision data
    atlas_score: float
    atlas_threshold: float
    agent_votes: Dict

    # Account state
    account_balance_before: float
    account_balance_after: Optional[float]

    # Kelly Criterion
    kelly_calculation: Dict

    # Exit info
    exit_reason: Optional[str]
    status: str  # "open", "closed", "failed"
    notes: str


class TradeLogger:
    """
    Comprehensive trade logging system for ATLAS.

    Features:
    - Logs every trade decision and execution
    - Tracks performance metrics (P/L, win rate, R-multiple)
    - Stores agent votes for analysis
    - Records Kelly Criterion calculations
    - Generates daily trade log files
    - Provides performance analytics
    """

    def __init__(self, log_dir: Path = None):
        """
        Initialize trade logger.

        Args:
            log_dir: Directory for trade logs (default: ./logs/trades/)
        """
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs" / "trades"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.current_session_file = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.daily_log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y-%m-%d')}.json"

        self.open_trades = {}  # trade_id -> TradeLog
        self.trade_counter = 0

        logger.info(f"[TradeLogger] Initialized - logging to {self.log_dir}")

    def log_trade_decision(self, decision_data: Dict, market_data: Dict,
                          kelly_calc: Dict, account_balance: float) -> str:
        """
        Log a trade decision (BUY/SELL signal).

        Args:
            decision_data: Coordinator decision output
            market_data: Market data at decision time
            kelly_calc: Kelly Criterion calculation details
            account_balance: Current account balance

        Returns:
            trade_id: Unique trade identifier
        """
        self.trade_counter += 1
        trade_id = f"ATLAS_{datetime.now().strftime('%Y%m%d')}_{self.trade_counter:04d}"

        # Create trade log entry
        trade_log = TradeLog(
            trade_id=trade_id,
            oanda_trade_ids=[],

            timestamp_decision=datetime.now().isoformat(),
            timestamp_entry=None,
            timestamp_exit=None,
            duration_minutes=None,

            pair=market_data.get('pair', 'UNKNOWN'),
            direction=decision_data.get('decision'),
            units=kelly_calc.get('units', 0),
            lots=kelly_calc.get('units', 0) / 100000,

            entry_price=None,
            exit_price=None,
            stop_loss=kelly_calc.get('stop_loss_price'),
            take_profit=kelly_calc.get('take_profit_price'),

            pnl=None,
            pnl_pct=None,
            pips=None,
            r_multiple=None,

            atlas_score=decision_data.get('score', 0),
            atlas_threshold=decision_data.get('reasoning', {}).get('threshold', 0),
            agent_votes=decision_data.get('agent_votes', {}),

            account_balance_before=account_balance,
            account_balance_after=None,

            kelly_calculation=kelly_calc,

            exit_reason=None,
            status="pending",
            notes=""
        )

        self.open_trades[trade_id] = trade_log

        logger.info(f"[TradeLogger] Logged decision: {trade_id} - {trade_log.pair} {trade_log.direction}")

        return trade_id

    def log_trade_entry(self, trade_id: str, oanda_result: Dict):
        """
        Log successful trade entry.

        Args:
            trade_id: Trade identifier
            oanda_result: OANDA execution result
        """
        if trade_id not in self.open_trades:
            logger.warning(f"[TradeLogger] Trade {trade_id} not found for entry logging")
            return

        trade = self.open_trades[trade_id]
        trade.timestamp_entry = datetime.now().isoformat()
        trade.entry_price = oanda_result.get('price', 0)
        trade.oanda_trade_ids.append(str(oanda_result.get('id', '')))
        trade.status = "open"

        # Save to disk immediately
        self._save_trade(trade)

        logger.info(f"[TradeLogger] Entry logged: {trade_id} @ {trade.entry_price:.5f}")

    def log_trade_failure(self, trade_id: str, error_msg: str):
        """
        Log failed trade execution.

        Args:
            trade_id: Trade identifier
            error_msg: Error message
        """
        if trade_id not in self.open_trades:
            return

        trade = self.open_trades[trade_id]
        trade.status = "failed"
        trade.exit_reason = "execution_failed"
        trade.notes = error_msg

        # Save and remove from open trades
        self._save_trade(trade)
        del self.open_trades[trade_id]

        logger.warning(f"[TradeLogger] Trade failed: {trade_id} - {error_msg}")

    def log_trade_exit(self, trade_id: str, exit_price: float, pnl: float,
                       exit_reason: str, account_balance: float):
        """
        Log trade exit/close.

        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            pnl: Profit/Loss in dollars
            exit_reason: Reason for exit (stop_loss, take_profit, manual, etc.)
            account_balance: Account balance after exit
        """
        if trade_id not in self.open_trades:
            logger.warning(f"[TradeLogger] Trade {trade_id} not found for exit logging")
            return

        trade = self.open_trades[trade_id]
        trade.timestamp_exit = datetime.now().isoformat()
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.exit_reason = exit_reason
        trade.account_balance_after = account_balance
        trade.status = "closed"

        # Calculate duration
        if trade.timestamp_entry:
            entry_time = datetime.fromisoformat(trade.timestamp_entry)
            exit_time = datetime.fromisoformat(trade.timestamp_exit)
            trade.duration_minutes = (exit_time - entry_time).total_seconds() / 60

        # Calculate metrics
        if trade.entry_price and exit_price:
            # Calculate pips
            pip_value = 0.0001 if 'JPY' not in trade.pair else 0.01
            if trade.direction == 'BUY':
                trade.pips = (exit_price - trade.entry_price) / pip_value
            else:
                trade.pips = (trade.entry_price - exit_price) / pip_value

            # Calculate R-multiple
            if trade.stop_loss:
                risk_pips = abs((trade.entry_price - trade.stop_loss) / pip_value)
                if risk_pips > 0:
                    trade.r_multiple = trade.pips / risk_pips

            # Calculate percentage return
            trade.pnl_pct = (pnl / trade.account_balance_before) * 100

        # Save and remove from open trades
        self._save_trade(trade)
        del self.open_trades[trade_id]

        logger.info(f"[TradeLogger] Exit logged: {trade_id} - P/L: ${pnl:+,.2f} ({trade.pips:+.1f} pips)")

    def _save_trade(self, trade: TradeLog):
        """Save trade to both session and daily log files"""
        trade_dict = asdict(trade)

        # Append to daily log
        self._append_to_log(self.daily_log_file, trade_dict)

        # Append to session log
        self._append_to_log(self.current_session_file, trade_dict)

    def _append_to_log(self, file_path: Path, trade_dict: Dict):
        """Append trade to JSON log file"""
        try:
            # Read existing logs
            if file_path.exists():
                with open(file_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []

            # Append new trade
            logs.append(trade_dict)

            # Write back
            with open(file_path, 'w') as f:
                json.dump(logs, f, indent=2)

        except Exception as e:
            logger.error(f"[TradeLogger] Failed to save trade: {e}")

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary from daily log.

        Returns:
            Dictionary with performance metrics
        """
        if not self.daily_log_file.exists():
            return {"error": "No trades logged today"}

        try:
            with open(self.daily_log_file, 'r') as f:
                trades = json.load(f)

            closed_trades = [t for t in trades if t.get('status') == 'closed']

            if not closed_trades:
                return {"message": "No closed trades yet"}

            wins = [t for t in closed_trades if t.get('pnl', 0) > 0]
            losses = [t for t in closed_trades if t.get('pnl', 0) < 0]

            total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
            total_wins = sum(t.get('pnl', 0) for t in wins)
            total_losses = sum(t.get('pnl', 0) for t in losses)

            avg_win = total_wins / len(wins) if wins else 0
            avg_loss = total_losses / len(losses) if losses else 0

            return {
                "total_trades": len(closed_trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(closed_trades) * 100 if closed_trades else 0,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": abs(total_wins / total_losses) if total_losses != 0 else float('inf'),
                "expectancy": total_pnl / len(closed_trades) if closed_trades else 0,
                "largest_win": max((t.get('pnl', 0) for t in closed_trades), default=0),
                "largest_loss": min((t.get('pnl', 0) for t in closed_trades), default=0),
                "avg_duration_minutes": sum(t.get('duration_minutes', 0) for t in closed_trades) / len(closed_trades) if closed_trades else 0
            }

        except Exception as e:
            logger.error(f"[TradeLogger] Failed to calculate performance: {e}")
            return {"error": str(e)}

    def close_session(self):
        """Close logging session and save summary"""
        summary = self.get_performance_summary()

        summary_file = self.log_dir / f"summary_{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"[TradeLogger] Session closed - Summary saved to {summary_file}")
        logger.info(f"[TradeLogger] Performance: {summary.get('total_trades', 0)} trades, "
                   f"${summary.get('total_pnl', 0):+,.2f} P/L")
