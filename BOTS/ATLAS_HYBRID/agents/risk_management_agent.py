"""Risk Management Agent - Position sizing and exposure limits with VETO power."""
from typing import Dict, Tuple
from .base_agent import BaseAgent
import json
from pathlib import Path
from datetime import datetime, timedelta

class RiskManagementAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.5):
        super().__init__(name="RiskManagementAgent", initial_weight=initial_weight)
        self.state_file = Path(__file__).parent.parent / "learning" / "state" / "risk_manager_state.json"
        self.load_state()

    def load_state(self, state_path: str = None):
        """Load daily tracking state"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.daily_pnl = data.get('daily_pnl', 0)
                    self.consecutive_losses = data.get('consecutive_losses', 0)
                    self.last_reset = datetime.fromisoformat(data.get('last_reset', datetime.now().isoformat()))
                    self.trades_today = data.get('trades_today', 0)
            else:
                self.reset_daily()
        except:
            self.reset_daily()

    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.last_reset = datetime.now()

    def save_state(self, filepath=None):
        """Save state to disk"""
        save_path = Path(filepath) if filepath else self.state_file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({
                'daily_pnl': self.daily_pnl,
                'consecutive_losses': self.consecutive_losses,
                'trades_today': self.trades_today,
                'last_reset': self.last_reset.isoformat()
            }, f, indent=2)

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """Enforce professional risk management rules"""
        persist_state = market_data.get("persist_state", True)

        balance = market_data.get("account_balance", 200000)
        atr = market_data.get("indicators", {}).get("atr", 0.001)

        # Reset daily if new day
        if datetime.now().date() > self.last_reset.date():
            self.reset_daily()

        # RULE 1: Max 3% Daily Drawdown - HARD VETO
        max_daily_loss = balance * 0.03
        if abs(self.daily_pnl) > max_daily_loss and self.daily_pnl < 0:
            if persist_state:
                self.save_state()
            return ("HOLD", 1.0, {
                "veto_reason": f"DAILY_LOSS_LIMIT ({abs(self.daily_pnl):.0f}/{max_daily_loss:.0f})",
                "daily_pnl": self.daily_pnl,
                "limit": max_daily_loss
            })

        # RULE 2: Max 3 Consecutive Losses - Cooldown
        if self.consecutive_losses >= 3:
            if persist_state:
                self.save_state()
            return ("HOLD", 0.9, {
                "veto_reason": f"CONSECUTIVE_LOSSES ({self.consecutive_losses})",
                "consecutive_losses": self.consecutive_losses
            })

        # RULE 3: Max 10 trades per day - Prevent overtrading
        if self.trades_today >= 10:
            if persist_state:
                self.save_state()
            return ("HOLD", 1.0, {
                "veto_reason": f"MAX_DAILY_TRADES ({self.trades_today}/10)",
                "trades_today": self.trades_today
            })

        # RULE 4: High volatility - reduce confidence
        # Higher ATR = higher risk, reduce confidence
        risk_score = min(1 / (atr * 10000 + 1), 0.85)

        # Normal operation - allow trade but with risk-adjusted confidence
        if persist_state:
            self.save_state()
        return ("NEUTRAL", 0.5, {
            "risk_score": round(risk_score, 2),
            "atr": atr,
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "trades_today": self.trades_today
        })
