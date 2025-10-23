#!/usr/bin/env python3
"""
TELEGRAM NOTIFIER
Send real-time alerts to your phone for all trading activity
"""
import os
import requests
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class TelegramNotifier:
    """Send trading alerts via Telegram"""

    def __init__(self):
        """Initialize Telegram bot"""
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Check if configured
        if not self.bot_token or self.bot_token == 'your_telegram_bot_token_here':
            print("[WARNING] Telegram not configured - alerts disabled")
            print("To enable: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
            self.enabled = False
        else:
            self.enabled = True
            print("[OK] Telegram notifier enabled")

    def send(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """
        Send message to Telegram

        Args:
            message: Text to send (supports Markdown)
            parse_mode: 'Markdown' or 'HTML'

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }

            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200

        except Exception as e:
            print(f"[ERROR] Telegram send failed: {e}")
            return False

    def trade_opened(self, symbol: str, side: str, price: float,
                    strategy: str, score: float, risk: float):
        """Alert on new trade opened"""
        message = f"""
📈 *TRADE OPENED*

*Symbol:* {symbol}
*Side:* {side}
*Price:* ${price:.2f}
*Strategy:* {strategy}
*Score:* {score:.1f}
*Risk:* ${risk:.2f}

_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        self.send(message)

    def trade_closed(self, symbol: str, entry: float, exit: float,
                    pnl: float, pnl_pct: float, duration: str):
        """Alert on trade closed"""
        emoji = "✅" if pnl > 0 else "❌"
        message = f"""
{emoji} *TRADE CLOSED*

*Symbol:* {symbol}
*Entry:* ${entry:.2f}
*Exit:* ${exit:.2f}
*P&L:* ${pnl:.2f} ({pnl_pct:+.1f}%)
*Duration:* {duration}

_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        self.send(message)

    def stop_loss_hit(self, symbol: str, loss: float, reason: str):
        """Alert on stop loss triggered"""
        message = f"""
🛑 *STOP LOSS HIT*

*Symbol:* {symbol}
*Loss:* -${abs(loss):.2f}
*Reason:* {reason}

_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        self.send(message)

    def large_loss_warning(self, symbol: str, current_loss: float, threshold: float):
        """Alert on large unrealized loss"""
        message = f"""
⚠️ *LARGE LOSS WARNING*

*Symbol:* {symbol}
*Current Loss:* -${abs(current_loss):.2f}
*Threshold:* ${threshold:.2f}

Consider closing position manually.

_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        self.send(message)

    def system_error(self, component: str, error: str):
        """Alert on system error"""
        message = f"""
🚨 *SYSTEM ERROR*

*Component:* {component}
*Error:* {error}

Check logs immediately!

_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        self.send(message)

    def system_restarted(self, component: str, reason: str):
        """Alert on system restart"""
        message = f"""
🔄 *SYSTEM RESTARTED*

*Component:* {component}
*Reason:* {reason}

System is back online.

_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        self.send(message)

    def daily_summary(self, total_trades: int, winners: int, losers: int,
                     total_pnl: float, win_rate: float):
        """Send daily performance summary"""
        emoji = "📈" if total_pnl > 0 else "📉"
        message = f"""
{emoji} *DAILY SUMMARY*

*Trades:* {total_trades} ({winners}W, {losers}L)
*Win Rate:* {win_rate:.1f}%
*Total P&L:* ${total_pnl:+.2f}

_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        self.send(message)

    def weekly_summary(self, total_trades: int, win_rate: float,
                      total_pnl: float, sharpe: float, max_drawdown: float):
        """Send weekly performance summary"""
        emoji = "🎯" if win_rate >= 60 else "⚠️"
        message = f"""
{emoji} *WEEKLY SUMMARY*

*Total Trades:* {total_trades}
*Win Rate:* {win_rate:.1f}%
*Total P&L:* ${total_pnl:+.2f}
*Sharpe Ratio:* {sharpe:.2f}
*Max Drawdown:* {max_drawdown:.1f}%

_{datetime.now().strftime('%Y-%m-%d')}_
"""
        self.send(message)

    def test_connection(self):
        """Test Telegram connection"""
        if not self.enabled:
            print("[X] Telegram not configured")
            return False

        message = """
✅ *TELEGRAM TEST*

Connection successful!
You will now receive trading alerts.

_{}_
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        success = self.send(message)
        if success:
            print("[OK] Telegram test message sent!")
        else:
            print("[X] Failed to send test message")

        return success


# Singleton instance
_notifier = None

def get_notifier() -> TelegramNotifier:
    """Get or create Telegram notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


if __name__ == "__main__":
    # Test the notifier
    print("\n" + "="*70)
    print("TELEGRAM NOTIFIER TEST")
    print("="*70)

    notifier = get_notifier()

    if notifier.enabled:
        print("\n[TESTING] Sending test message...")
        notifier.test_connection()
        print("\nCheck your Telegram app for the test message!")
    else:
        print("\n[SETUP NEEDED] To enable Telegram notifications:")
        print("1. Message @BotFather on Telegram")
        print("2. Send: /newbot")
        print("3. Follow prompts to create bot")
        print("4. Copy the bot token")
        print("5. Message your bot to get your chat_id")
        print("6. Add to .env:")
        print("   TELEGRAM_BOT_TOKEN=your_token_here")
        print("   TELEGRAM_CHAT_ID=your_chat_id_here")
        print("\nDetailed guide: https://core.telegram.org/bots#creating-a-new-bot")

    print("="*70 + "\n")
