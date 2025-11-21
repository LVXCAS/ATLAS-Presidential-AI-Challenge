"""
RISK KILL-SWITCH
Auto-stop trading and close positions on excessive drawdown
"""
import os
import json
import requests
from datetime import datetime
from typing import Dict, Optional
from unified_pnl_tracker import UnifiedPnLTracker

class RiskKillSwitch:
    def __init__(self):
        self.pnl_tracker = UnifiedPnLTracker()
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Risk limits
        self.daily_loss_limit = 0.02  # 2% daily loss triggers stop
        self.drawdown_limit = 0.05    # 5% total drawdown triggers emergency close
        self.position_limit = 0.10    # Max 10% of account per position

        # Override flag (can be set via Telegram)
        self.override_enabled = False

        # Load risk state
        self.risk_state = self._load_risk_state()

    def _load_risk_state(self) -> Dict:
        """Load risk state from file"""
        if os.path.exists('data/risk_state.json'):
            with open('data/risk_state.json') as f:
                return json.load(f)
        return {
            'kill_switch_active': False,
            'last_triggered': None,
            'override_until': None,
            'total_stops_triggered': 0
        }

    def _save_risk_state(self):
        """Save risk state to file"""
        os.makedirs('data', exist_ok=True)
        with open('data/risk_state.json', 'w') as f:
            json.dump(self.risk_state, f, indent=2)

    def send_telegram_alert(self, message: str):
        """Send urgent Telegram alert"""
        try:
            url = f'https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage'
            data = {
                'chat_id': self.telegram_chat_id,
                'text': f'RISK ALERT!\n\n{message}',
                'parse_mode': 'Markdown'
            }
            requests.post(url, data=data, timeout=5)
            print(f"[RISK ALERT] Telegram notification sent")
        except Exception as e:
            print(f"[RISK ALERT] Failed to send Telegram: {e}")

    def check_daily_loss_limit(self, unified_pnl) -> bool:
        """Check if daily loss exceeds limit"""
        if unified_pnl.total_starting_balance == 0:
            print("[RISK] No starting balance - skipping daily loss check")
            return False

        daily_loss_percent = abs(unified_pnl.daily_pnl / unified_pnl.total_starting_balance)

        if unified_pnl.daily_pnl < 0 and daily_loss_percent > self.daily_loss_limit:
            print(f"[RISK] Daily loss limit breached: {daily_loss_percent*100:.2f}% > {self.daily_loss_limit*100:.2f}%")
            return True

        return False

    def check_drawdown_limit(self, unified_pnl) -> bool:
        """Check if total drawdown exceeds limit"""
        if unified_pnl.total_starting_balance == 0:
            print("[RISK] No starting balance - skipping drawdown check")
            return False

        drawdown_percent = abs(unified_pnl.total_pnl / unified_pnl.total_starting_balance)

        if unified_pnl.total_pnl < 0 and drawdown_percent > self.drawdown_limit:
            print(f"[RISK] Drawdown limit breached: {drawdown_percent*100:.2f}% > {self.drawdown_limit*100:.2f}%")
            return True

        return False

    def close_all_oanda_positions(self):
        """Close all OANDA Forex positions"""
        try:
            headers = {
                'Authorization': f'Bearer {os.getenv("OANDA_API_KEY")}',
                'Content-Type': 'application/json'
            }

            account_id = os.getenv('OANDA_ACCOUNT_ID')

            # Get all open trades
            trades_url = f'https://api-fxpractice.oanda.com/v3/accounts/{account_id}/openTrades'
            response = requests.get(trades_url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"[OANDA] Error fetching trades: {response.status_code}")
                return False

            trades = response.json().get('trades', [])
            print(f"[OANDA] Closing {len(trades)} open positions...")

            for trade in trades:
                trade_id = trade['id']
                close_url = f'https://api-fxpractice.oanda.com/v3/accounts/{account_id}/trades/{trade_id}/close'
                close_response = requests.put(close_url, headers=headers, timeout=10)

                if close_response.status_code == 200:
                    print(f"[OANDA] Closed trade {trade_id}")
                else:
                    print(f"[OANDA] Failed to close trade {trade_id}: {close_response.status_code}")

            return True
        except Exception as e:
            print(f"[OANDA] Error closing positions: {e}")
            return False

    def close_all_alpaca_positions(self):
        """Close all Alpaca positions"""
        try:
            headers = {
                'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
                'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY')
            }

            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

            # Close all positions at once
            url = f'{base_url}/v2/positions'
            response = requests.delete(url, headers=headers, timeout=10)

            if response.status_code in [200, 207]:
                print(f"[ALPACA] All positions closed")
                return True
            else:
                print(f"[ALPACA] Error closing positions: {response.status_code}")
                return False
        except Exception as e:
            print(f"[ALPACA] Error closing positions: {e}")
            return False

    def stop_all_trading_systems(self):
        """Stop all running trading systems"""
        import psutil

        trading_scripts = [
            'START_FOREX_ELITE.py',
            'futures_live_validation.py',
            'START_ACTIVE_FOREX_PAPER_TRADING.py',
            'START_ACTIVE_FUTURES_PAPER_TRADING.py'
        ]

        stopped_count = 0

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any(script in ' '.join(cmdline) for script in trading_scripts):
                    print(f"[KILL] Stopping PID {proc.info['pid']}: {proc.info['name']}")
                    proc.terminate()
                    stopped_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        print(f"[KILL] Stopped {stopped_count} trading processes")
        return stopped_count > 0

    def trigger_kill_switch(self, reason: str):
        """Trigger the kill switch - stop everything"""
        print(f"\n{'='*70}")
        print(f"KILL SWITCH TRIGGERED: {reason}")
        print(f"{'='*70}\n")

        # Update state
        self.risk_state['kill_switch_active'] = True
        self.risk_state['last_triggered'] = datetime.now().isoformat()
        self.risk_state['total_stops_triggered'] += 1
        self._save_risk_state()

        # Send urgent Telegram alert
        alert_message = f"""
KILL SWITCH TRIGGERED!

Reason: {reason}

Actions taken:
1. Closing all open positions
2. Stopping all trading systems
3. Waiting for manual override

Use /risk override to resume trading
"""
        self.send_telegram_alert(alert_message)

        # Close all positions
        print("[KILL] Closing all positions...")
        oanda_closed = self.close_all_oanda_positions()
        alpaca_closed = self.close_all_alpaca_positions()

        # Stop trading systems
        print("[KILL] Stopping trading systems...")
        systems_stopped = self.stop_all_trading_systems()

        # Log results
        print(f"\n[KILL] Results:")
        print(f"  OANDA positions closed: {oanda_closed}")
        print(f"  Alpaca positions closed: {alpaca_closed}")
        print(f"  Trading systems stopped: {systems_stopped}")
        print(f"\n{'='*70}\n")

        return True

    def check_and_enforce(self):
        """Main monitoring loop - check risk limits and enforce"""
        # Skip if override is active
        if self.override_enabled:
            print("[RISK] Override active - skipping risk checks")
            return False

        # Skip if kill switch already active
        if self.risk_state['kill_switch_active']:
            print("[RISK] Kill switch already active")
            return False

        # Get current P&L
        print("[RISK] Checking risk limits...")
        unified_pnl = self.pnl_tracker.get_unified_pnl()

        # Check daily loss limit
        if self.check_daily_loss_limit(unified_pnl):
            self.trigger_kill_switch(f"Daily loss exceeded {self.daily_loss_limit*100:.1f}% limit")
            return True

        # Check total drawdown limit
        if self.check_drawdown_limit(unified_pnl):
            self.trigger_kill_switch(f"Total drawdown exceeded {self.drawdown_limit*100:.1f}% limit")
            return True

        print("[RISK] All risk checks passed")
        return False

    def reset_kill_switch(self):
        """Reset kill switch (manual override)"""
        self.risk_state['kill_switch_active'] = False
        self.risk_state['override_until'] = datetime.now().isoformat()
        self._save_risk_state()

        print("[RISK] Kill switch reset - trading can resume")

        # Send Telegram notification
        self.send_telegram_alert("Kill switch RESET. Trading systems can be restarted manually.")

        return True

    def get_position_size(self, account_balance: float, risk_percent: float = 0.01) -> float:
        """Calculate safe position size based on account equity"""
        # Risk 1% per trade by default
        max_position_value = account_balance * self.position_limit
        risk_amount = account_balance * risk_percent

        print(f"[RISK] Position sizing:")
        print(f"  Account balance: ${account_balance:,.2f}")
        print(f"  Max position value: ${max_position_value:,.2f} ({self.position_limit*100:.0f}% of account)")
        print(f"  Risk per trade: ${risk_amount:,.2f} ({risk_percent*100:.1f}% of account)")

        return max_position_value

def main():
    """Test the risk kill-switch"""
    kill_switch = RiskKillSwitch()

    print("="*70)
    print("RISK KILL-SWITCH - Monitoring Mode")
    print("="*70)

    # Check risk limits
    triggered = kill_switch.check_and_enforce()

    if triggered:
        print("\nKILL SWITCH ACTIVATED!")
    else:
        print("\nAll systems operational - risk within limits")

    # Show position sizing example
    print("\n" + "="*70)
    print("POSITION SIZING EXAMPLE")
    print("="*70)
    kill_switch.get_position_size(account_balance=100000, risk_percent=0.01)

if __name__ == '__main__':
    main()
