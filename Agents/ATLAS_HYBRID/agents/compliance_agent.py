"""
Compliance Agent

Monitors challenge rules and enforces compliance.

VETO power: Blocks trades that would violate E8 rules.

Responsibilities:
1. Track daily drawdown (kill switch at -$2,500 to prevent -$3k violation)
2. Track trailing drawdown (6% max)
3. Monitor profit target ($20,000)
4. Circuit breaker for losing streaks
"""

from typing import Dict, Tuple
from datetime import datetime, date
from .base_agent import BaseAgent


class E8ComplianceAgent(BaseAgent):
    """
     prop firm compliance monitoring.

    Rules:
    - Starting Capital: $200,000
    - Profit Target: $20,000 (10%)
    - Trailing DD: 6% ($12,000)
    - Daily DD: ~$3,000-4,000 (not officially stated but observed)
    - One violation = Account terminated

    Circuit Breakers:
    - Daily loss > $2,500 → STOP trading for the day
    - Trailing DD > 5.5% → REDUCE position sizes
    - Losing streak > 5 trades → PAUSE and review
    """

    def __init__(self, starting_balance: float = 200000, initial_weight: float = 2.0):
        """
        Initialize with VETO power.

        Args:
            starting_balance: E8 starting capital ($200,000)
            initial_weight: VETO power (2.0)
        """
        super().__init__(name="E8ComplianceAgent", initial_weight=initial_weight)

        # E8 Parameters
        self.starting_balance = starting_balance
        self.profit_target = 20000  # $20k profit
        self.max_trailing_dd_pct = 0.06  # 6%
        self.daily_dd_limit = 3000  # $3k daily loss limit
        self.daily_dd_circuit_breaker = 2500  # Stop at $2,500 to leave buffer

        # Account tracking
        self.current_balance = starting_balance
        self.peak_balance = starting_balance
        self.daily_start_balance = starting_balance

        # Daily tracking
        self.current_date = date.today()
        self.daily_pnl = 0
        self.daily_trades = 0

        # Streak tracking
        self.current_streak = 0  # Positive = wins, Negative = losses
        self.max_losing_streak = 5

        # Violation tracking
        self.daily_dd_violations = 0
        self.trailing_dd_violations = 0

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Check  compliance before allowing trade.

        Returns:
            - "ALLOW" if trade is compliant
            - "BLOCK" if trade would violate E8 rules (VETO)
        """
        # Update account state
        current_balance = market_data.get("account_balance", self.current_balance)
        today = market_data.get("date", date.today())

        self._update_daily_tracking(current_balance, today)

        # Check all compliance rules
        violations = []

        # 1. Daily DD check (CRITICAL)
        daily_loss = self.daily_start_balance - current_balance
        daily_dd_remaining = self.daily_dd_limit - abs(daily_loss)

        if daily_loss >= self.daily_dd_circuit_breaker:
            violations.append(f"Circuit breaker: Already down ${daily_loss:.2f} today")
            return ("BLOCK", 1.0, {
                "reason": "Daily DD circuit breaker triggered",
                "daily_loss": daily_loss,
                "limit": self.daily_dd_circuit_breaker,
                "veto": True,
                "violations": violations
            })

        if daily_dd_remaining < 500:
            # Less than $500 cushion remaining
            violations.append(f"Only ${daily_dd_remaining:.2f} daily DD cushion left")
            return ("BLOCK", 1.0, {
                "reason": "Insufficient daily DD cushion",
                "daily_dd_remaining": daily_dd_remaining,
                "veto": True,
                "violations": violations
            })

        # 2. Trailing DD check
        current_dd_pct = (self.peak_balance - current_balance) / self.peak_balance

        if current_dd_pct >= self.max_trailing_dd_pct:
            violations.append(f"Trailing DD at {current_dd_pct*100:.2f}% (limit: {self.max_trailing_dd_pct*100}%)")
            return ("BLOCK", 1.0, {
                "reason": "Trailing DD limit reached",
                "current_dd": current_dd_pct * 100,
                "limit": self.max_trailing_dd_pct * 100,
                "veto": True,
                "violations": violations
            })

        # 3. Losing streak circuit breaker
        if self.current_streak <= -self.max_losing_streak:
            violations.append(f"Losing streak: {abs(self.current_streak)} trades")
            return ("BLOCK", 1.0, {
                "reason": "Excessive losing streak - need review",
                "losing_streak": abs(self.current_streak),
                "veto": True,
                "violations": violations
            })

        # 4. Check if already hit profit target (can stop trading)
        profit = current_balance - self.starting_balance
        if profit >= self.profit_target:
            return ("ALLOW", 1.0, {
                "reason": "Profit target reached - can trade conservatively or stop",
                "profit": profit,
                "target": self.profit_target,
                "status": "TARGET_REACHED"
            })

        # All checks passed
        return ("ALLOW", 1.0, {
            "reason": "All E8 compliance checks passed",
            "daily_dd_remaining": daily_dd_remaining,
            "trailing_dd_pct": round(current_dd_pct * 100, 2),
            "profit_to_target": self.profit_target - profit,
            "safe": True
        })

    def _update_daily_tracking(self, current_balance: float, today: date):
        """
        Update daily tracking metrics.

        Resets daily counters on new day.
        """
        self.current_balance = current_balance

        # Update peak balance (for trailing DD)
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # Check if new day
        if today != self.current_date:
            # Reset daily tracking
            self.current_date = today
            self.daily_start_balance = current_balance
            self.daily_pnl = 0
            self.daily_trades = 0

    def record_trade_result(self, trade: Dict):
        """
        Record trade result for streak tracking.

        Args:
            trade: Trade result with 'outcome' ("WIN" or "LOSS")
        """
        outcome = trade.get("outcome")

        if outcome == "WIN":
            # Winning trade
            if self.current_streak >= 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
        elif outcome == "LOSS":
            # Losing trade
            if self.current_streak <= 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1

        # Update daily trades
        self.daily_trades += 1

    def check_daily_dd_violation(self) -> bool:
        """
        Check if we violated daily DD limit.

        Called at end of day to log violations.

        Returns:
            True if violation occurred
        """
        daily_loss = self.daily_start_balance - self.current_balance

        if daily_loss >= self.daily_dd_limit:
            self.daily_dd_violations += 1
            print(f"[E8_COMPLIANCE] DAILY DD VIOLATION: Lost ${daily_loss:.2f} on {self.current_date}")
            return True

        return False

    def get_compliance_status(self) -> Dict:
        """
        Get current E8 compliance status.

        Returns:
            Dictionary with all compliance metrics
        """
        profit = self.current_balance - self.starting_balance
        trailing_dd_pct = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        daily_loss = self.daily_start_balance - self.current_balance

        return {
            "current_balance": self.current_balance,
            "starting_balance": self.starting_balance,
            "profit": profit,
            "profit_target": self.profit_target,
            "profit_progress_pct": (profit / self.profit_target * 100) if profit > 0 else 0,
            "peak_balance": self.peak_balance,
            "trailing_dd_pct": round(trailing_dd_pct, 2),
            "trailing_dd_limit": self.max_trailing_dd_pct * 100,
            "trailing_dd_remaining": round((self.max_trailing_dd_pct * 100) - trailing_dd_pct, 2),
            "daily_loss": daily_loss,
            "daily_dd_limit": self.daily_dd_limit,
            "daily_dd_remaining": self.daily_dd_limit - abs(daily_loss),
            "daily_trades_today": self.daily_trades,
            "current_streak": self.current_streak,
            "daily_dd_violations": self.daily_dd_violations,
            "trailing_dd_violations": self.trailing_dd_violations,
        }

    def adjust_weight(self, learning_rate: float = 0.0):
        """
        Override - E8ComplianceAgent maintains VETO weight.

        This is a compliance agent, not a profit optimizer.
        Weight should never change.
        """
        self.weight = 2.0  # Lock VETO power

    def __repr__(self):
        status = self.get_compliance_status()
        return (f"<E8ComplianceAgent "
                f"profit=${status['profit']:+,.2f} "
                f"dd={status['trailing_dd_pct']:.1f}% "
                f"violations={self.daily_dd_violations}>")
