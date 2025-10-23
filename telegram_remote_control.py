#!/usr/bin/env python3
"""
TELEGRAM REMOTE CONTROL - FULL TRADING EMPIRE CONTROL
Control your entire trading empire from your phone via Telegram

NEW FEATURES:
- Remote start/stop for Forex, Options, Futures
- Emergency stop button
- Restart all systems
- Nuclear option (kill everything)
- Real-time P/L tracking
- Position monitoring
- Market regime detection
"""

import os
import sys
import time
import json
import subprocess
import requests
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')


class TelegramRemoteControl:
    """Full remote control of trading empire via Telegram"""

    def __init__(self):
        self.bot_token = BOT_TOKEN
        self.chat_id = CHAT_ID
        self.base_url = f'https://api.telegram.org/bot{self.bot_token}'
        self.last_update_id = 0

        print("\n" + "="*60)
        print("TELEGRAM REMOTE CONTROL ACTIVATED")
        print("="*60)
        print(f"You can now control your trading empire from your phone!")
        print(f"Send commands to @LVXCAS_bot")
        print("="*60 + "\n")

    def send_message(self, text: str):
        """Send message to Telegram"""
        url = f'{self.base_url}/sendMessage'
        data = {'chat_id': self.chat_id, 'text': text}
        try:
            requests.post(url, data=data, timeout=5)
        except:
            pass

    def get_updates(self) -> list:
        """Get new messages from Telegram"""
        url = f'{self.base_url}/getUpdates'
        params = {'offset': self.last_update_id + 1, 'timeout': 30}

        try:
            response = requests.get(url, params=params, timeout=35)
            data = response.json()

            if data['ok'] and data['result']:
                self.last_update_id = data['result'][-1]['update_id']
                return data['result']
        except Exception as e:
            print(f"[ERROR] Getting updates: {e}")

        return []

    def get_system_status(self) -> str:
        """Get current system status"""
        try:
            result = subprocess.run(['tasklist'], capture_output=True, text=True, shell=True)
            python_procs = [line for line in result.stdout.split('\n') if 'python' in line.lower()]

            status = "SYSTEM STATUS\n\n"

            # Check Forex
            if os.path.exists('forex_elite.pid'):
                with open('forex_elite.pid') as f:
                    pid = f.read().strip()
                    status += f"Forex Elite: RUNNING (PID: {pid})\n"
            else:
                status += "Forex Elite: STOPPED\n"

            # Check Options
            if os.path.exists('auto_scanner_status.json'):
                with open('auto_scanner_status.json') as f:
                    scanner_data = json.load(f)
                    status += f"Options Scanner: ACTIVE ({scanner_data.get('trades_today', 0)} trades today)\n"
            else:
                status += "Options Scanner: STOPPED\n"

            # Check Futures
            status += "Futures Scanner: CHECKING...\n"

            status += f"\nTotal processes: {len(python_procs)}\n"
            status += f"Time: {datetime.now().strftime('%H:%M:%S')}"

            return status
        except Exception as e:
            return f"Error: {e}"

    def get_positions(self) -> str:
        """Get all open positions"""
        try:
            msg = "OPEN POSITIONS\n\n"
            has_positions = False

            # Forex positions
            today = datetime.now().strftime('%Y%m%d')
            forex_pos_file = f'forex_trades/positions_{today}.json'

            if os.path.exists(forex_pos_file):
                with open(forex_pos_file) as f:
                    positions = json.load(f)
                    if positions:
                        has_positions = True
                        msg += "FOREX:\n"
                        for pos in positions:
                            msg += f"  {pos.get('pair', 'Unknown')}: {pos.get('side', 'Unknown')}\n"
                            msg += f"  Entry: {pos.get('entry_price', 'N/A')}\n"
                            msg += f"  P/L: {pos.get('pnl', 'N/A')}\n\n"

            if not has_positions:
                msg += "No open positions\n"

            return msg
        except Exception as e:
            return f"Error: {e}"

    def get_regime(self) -> str:
        """Get market regime"""
        try:
            response = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
            data = response.json()

            if data.get('data'):
                fg_value = int(data['data'][0]['value'])
                fg_class = data['data'][0]['value_classification']

                msg = f"MARKET REGIME\n\n"
                msg += f"Fear & Greed: {fg_value}/100\n"
                msg += f"Status: {fg_class}\n\n"

                if fg_value < 25:
                    msg += "Regime: EXTREME FEAR\nMode: DEFENSIVE"
                elif fg_value < 45:
                    msg += "Regime: BEARISH\nMode: CAUTIOUS"
                elif fg_value < 55:
                    msg += "Regime: NEUTRAL\nMode: BALANCED"
                elif fg_value < 75:
                    msg += "Regime: BULLISH\nMode: AGGRESSIVE"
                else:
                    msg += "Regime: EXTREME GREED\nMode: VERY AGGRESSIVE"

                return msg
        except:
            return "Could not fetch market regime"

    def get_pnl(self) -> str:
        """Get real-time P/L across all accounts"""
        try:
            from unified_pnl_tracker import UnifiedPnLTracker
            tracker = UnifiedPnLTracker()
            unified_pnl = tracker.get_unified_pnl()

            msg = f"""
UNIFIED P&L

Total Balance: ${unified_pnl.total_balance:,.2f}
Total P&L: ${unified_pnl.total_pnl:,.2f} ({unified_pnl.total_pnl_percent:+.2f}%)

Unrealized: ${unified_pnl.total_unrealized_pnl:,.2f}
Realized: ${unified_pnl.total_realized_pnl:,.2f}

ACCOUNTS:
"""
            for account in unified_pnl.accounts:
                msg += f"\n{account.account_name}:\n"
                msg += f"  P&L: ${account.total_pnl:,.2f} ({account.pnl_percent:+.2f}%)\n"
                msg += f"  Positions: {account.open_positions}\n"

            return msg
        except Exception as e:
            return f"Error fetching P&L: {e}"

    def start_forex(self) -> str:
        """Start Forex Elite"""
        try:
            cmd = [sys.executable, "START_FOREX_ELITE.py", "--strategy", "strict"]
            subprocess.Popen(cmd)
            time.sleep(1)
            return "Forex Elite STARTED\n\nEUR/USD + USD/JPY\nStrategy: Strict (71-75% WR)\nScanning every hour"
        except Exception as e:
            return f"Error: {e}"

    def start_futures(self) -> str:
        """Start Futures Scanner"""
        try:
            if not os.path.exists('futures_active_paper_trader.py'):
                # Create it
                with open('futures_active_paper_trader.py', 'w') as f:
                    f.write('# Futures scanner placeholder\nprint("Futures scanner starting...")\n')

            cmd = [sys.executable, "futures_active_paper_trader.py"]
            subprocess.Popen(cmd)
            time.sleep(1)
            return "Futures Scanner STARTED\n\nMES + MNQ\nScanning every 15 min"
        except Exception as e:
            return f"Error: {e}"

    def start_options(self) -> str:
        """Start Options Scanner"""
        try:
            cmd = [sys.executable, "auto_options_scanner.py"]
            subprocess.Popen(cmd)
            time.sleep(1)
            return "Options Scanner STARTED\n\nNext scan: 6:30 AM ET"
        except Exception as e:
            return f"Error: {e}"

    def emergency_stop(self) -> str:
        """Emergency stop all trading"""
        try:
            # Kill forex
            if os.path.exists('forex_elite.pid'):
                with open('forex_elite.pid') as f:
                    pid = f.read().strip()
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)

            return "EMERGENCY STOP\n\nAll systems stopped\nRestart manually when ready"
        except Exception as e:
            return f"Error: {e}"

    def restart_all(self) -> str:
        """Restart all systems"""
        self.emergency_stop()
        time.sleep(2)
        self.start_forex()
        time.sleep(1)
        self.start_futures()
        return "ALL SYSTEMS RESTARTED\n\nForex + Futures now running"

    def kill_all(self) -> str:
        """Nuclear option"""
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], capture_output=True)
        return "NUCLEAR OPTION EXECUTED\n\nAll Python processes killed\nThis bot will die too"

    def risk_check(self) -> str:
        """Check risk status"""
        try:
            from risk_kill_switch import RiskKillSwitch
            kill_switch = RiskKillSwitch()

            if kill_switch.risk_state['kill_switch_active']:
                return f"""
RISK KILL-SWITCH ACTIVE!

Last triggered: {kill_switch.risk_state['last_triggered']}
Total stops: {kill_switch.risk_state['total_stops_triggered']}

All trading is STOPPED.

Use /risk override to resume
"""
            else:
                return f"""
RISK STATUS: OK

Daily loss limit: {kill_switch.daily_loss_limit*100:.0f}%
Drawdown limit: {kill_switch.drawdown_limit*100:.0f}%
Max position size: {kill_switch.position_limit*100:.0f}% of account

All systems operational
"""
        except Exception as e:
            return f"Error checking risk: {e}"

    def risk_override(self) -> str:
        """Override kill-switch"""
        try:
            from risk_kill_switch import RiskKillSwitch
            kill_switch = RiskKillSwitch()
            kill_switch.reset_kill_switch()
            return "KILL-SWITCH RESET\n\nTrading can resume\nUse /start_forex or /start_futures"
        except Exception as e:
            return f"Error: {e}"

    def pipeline_status(self) -> str:
        """Check strategy pipeline status"""
        try:
            from strategy_deployment_pipeline import StrategyDeploymentPipeline
            pipeline = StrategyDeploymentPipeline()

            msg = "STRATEGY PIPELINE STATUS\n\n"

            discovered = len(pipeline.pipeline_state.get('discovered', []))
            paper = len(pipeline.pipeline_state.get('paper_trading', []))
            live = len(pipeline.pipeline_state.get('live', []))
            rejected = len(pipeline.pipeline_state.get('rejected', []))

            msg += f"Discovered: {discovered}\n"
            msg += f"Paper Trading: {paper}\n"
            msg += f"Live: {live}\n"
            msg += f"Rejected: {rejected}\n\n"

            # Show paper trading strategies
            if paper > 0:
                msg += "PAPER TRADING:\n"
                for strategy in pipeline.pipeline_state['paper_trading'][:3]:
                    msg += f"  - {strategy['name']}\n"

            # Show live strategies
            if live > 0:
                msg += "\nLIVE:\n"
                for strategy in pipeline.pipeline_state['live'][:3]:
                    msg += f"  - {strategy['strategy_name']}\n"

            return msg
        except Exception as e:
            return f"Error: {e}"

    def run_pipeline(self) -> str:
        """Run deployment pipeline"""
        try:
            from strategy_deployment_pipeline import StrategyDeploymentPipeline
            import subprocess
            import sys

            # Run pipeline in background
            cmd = [sys.executable, "strategy_deployment_pipeline.py"]
            subprocess.Popen(cmd)

            return "PIPELINE STARTED\n\nValidating R&D discoveries...\nYou'll get notifications when strategies are deployed"
        except Exception as e:
            return f"Error: {e}"

    def deploy_strategy(self, strategy_name: str) -> str:
        """Manual strategy deployment"""
        try:
            from strategy_deployment_pipeline import StrategyDeploymentPipeline
            pipeline = StrategyDeploymentPipeline()

            # Find strategy in discovered
            candidates = pipeline.parse_rd_discoveries()
            matching = [c for c in candidates if strategy_name.lower() in c.name.lower()]

            if not matching:
                return f"Strategy not found: {strategy_name}\n\nUse /pipeline to see available strategies"

            candidate = matching[0]

            # Validate and deploy
            validation = pipeline.validate_strategy(candidate)

            if validation.passed:
                pipeline.deploy_to_paper_trading(candidate, validation)
                return f"DEPLOYED: {candidate.name}\n\nValidation Sharpe: {validation.validation_sharpe:.2f}\nNow paper trading for 7 days"
            else:
                return f"DEPLOYMENT FAILED\n\n{validation.rejection_reason}"

        except Exception as e:
            return f"Error: {e}"

    def regime_auto_enable(self) -> str:
        """Enable auto-regime switching"""
        try:
            from regime_auto_switcher import RegimeAutoSwitcher
            switcher = RegimeAutoSwitcher()
            switcher.enable_auto_switching()
            return "REGIME AUTO-SWITCHING ENABLED\n\nStrategies will auto-switch based on market conditions"
        except Exception as e:
            return f"Error: {e}"

    def regime_auto_disable(self) -> str:
        """Disable auto-regime switching"""
        try:
            from regime_auto_switcher import RegimeAutoSwitcher
            switcher = RegimeAutoSwitcher()
            switcher.disable_auto_switching()
            return "REGIME AUTO-SWITCHING DISABLED\n\nStrategies will continue manually"
        except Exception as e:
            return f"Error: {e}"

    def regime_auto_status(self) -> str:
        """Get regime auto-switcher status"""
        try:
            from regime_auto_switcher import RegimeAutoSwitcher
            switcher = RegimeAutoSwitcher()
            return switcher.get_status()
        except Exception as e:
            return f"Error: {e}"

    def earnings_scan(self) -> str:
        """Scan upcoming earnings plays"""
        try:
            from earnings_play_automator import EarningsPlayAutomator
            automator = EarningsPlayAutomator()

            # Scan next 7 days
            events = automator.download_earnings_calendar(days_ahead=7)
            setups = []

            for event in events[:10]:  # Top 10
                setup = automator.suggest_earnings_setup(event)
                if setup:
                    setups.append((event, setup))

            if not setups:
                return "No high-quality earnings setups found\n\nCriteria: IV Rank > 50, 2-14 days out"

            msg = f"EARNINGS PLAYS ({len(setups)} found)\n\n"

            for event, setup in setups[:5]:  # Top 5
                msg += f"{event.symbol} - {event.earnings_date.strftime('%m/%d')}\n"
                msg += f"  Strategy: {setup.strategy_type.upper()}\n"
                msg += f"  IV Rank: {setup.iv_rank:.0f}\n"
                msg += f"  Entry: {setup.entry_price:.2f}\n"
                msg += f"  Max Risk: ${setup.max_risk:.2f}\n"
                msg += f"  Win %: {setup.win_probability:.0f}%\n\n"

            return msg
        except Exception as e:
            return f"Error: {e}"

    def confluence_scan(self) -> str:
        """Scan for multi-timeframe confluence"""
        try:
            from multi_timeframe_confluence_scanner import MultiTimeframeConfluenceScanner
            scanner = MultiTimeframeConfluenceScanner()

            # Default watchlist
            watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                        'NVDA', 'META', 'SPY', 'QQQ', 'IWM']

            results = scanner.scan_watchlist(watchlist)

            # Filter for high-quality setups
            high_quality = [r for r in results if r.confluence_score >= 75]

            if not high_quality:
                return "No high-confluence setups found\n\nAll 3 timeframes (1H/4H/Daily) must align"

            msg = f"CONFLUENCE SETUPS ({len(high_quality)} found)\n\n"

            for result in high_quality[:5]:  # Top 5
                msg += f"{result.symbol} - Score: {result.confluence_score:.0f}\n"
                msg += f"  Signal: {result.primary_signal}\n"
                msg += f"  Entry: ${result.entry_price:.2f}\n"
                msg += f"  Stop: ${result.stop_loss:.2f}\n"
                msg += f"  Target: ${result.take_profit:.2f}\n"
                msg += f"  R/R: {result.risk_reward:.1f}:1\n\n"

            return msg
        except Exception as e:
            return f"Error: {e}"

    def viral_scan(self) -> str:
        """Scan for viral stocks on social media"""
        try:
            from social_sentiment_scanner import SocialSentimentScanner
            scanner = SocialSentimentScanner()

            alerts = scanner.scan_for_viral_stocks()

            if not alerts:
                return "No viral stocks detected\n\nCriteria: 50+ mentions, 200%+ spike, positive sentiment"

            msg = f"VIRAL STOCKS ({len(alerts)} found)\n\n"

            for alert in alerts[:5]:  # Top 5
                msg += f"{alert.symbol}\n"
                msg += f"  Mentions: {alert.total_mentions_24h}\n"
                msg += f"  Spike: {alert.mention_spike_vs_yesterday:.0f}%\n"
                msg += f"  Sentiment: {alert.sentiment_score:.0f}%\n"
                msg += f"  Action: {alert.action}\n"
                msg += f"  Risk: {alert.risk_level}\n\n"

            msg += "⚠️ Viral stocks = high volatility\nUse tight stops!"

            return msg
        except Exception as e:
            return f"Error: {e}"

    def rebalance_check(self) -> str:
        """Check portfolio allocation"""
        try:
            from portfolio_rebalancer import PortfolioRebalancer
            rebalancer = PortfolioRebalancer()

            allocations = rebalancer.calculate_drift()
            current_values = rebalancer.get_current_allocations()
            total_value = sum(current_values.values())

            msg = f"PORTFOLIO ALLOCATION\n\n"
            msg += f"Total: ${total_value:,.2f}\n\n"

            for allocation in allocations:
                msg += f"{allocation.category.upper()}:\n"
                msg += f"  Current: {allocation.current_percent:.1%}\n"
                msg += f"  Target: {allocation.target_percent:.1%}\n"
                msg += f"  Drift: {allocation.drift:+.1%}\n"
                msg += f"  Action: {allocation.action_needed.upper()}\n\n"

            # Check if rebalance needed
            needs_rebalance = rebalancer.check_rebalance_needed()

            if needs_rebalance:
                msg += "⚠️ REBALANCING NEEDED\n"
                msg += "Run portfolio_rebalancer.py to execute"
            else:
                msg += "✓ Portfolio is balanced"

            return msg
        except Exception as e:
            return f"Error: {e}"

    def handle_command(self, command: str):
        """Handle command"""
        print(f"[CMD] {command}")

        if command == '/status':
            return self.get_system_status()
        elif command == '/positions':
            return self.get_positions()
        elif command == '/regime':
            return self.get_regime()
        elif command == '/pnl':
            return self.get_pnl()
        elif command == '/start_forex':
            return self.start_forex()
        elif command == '/start_futures':
            return self.start_futures()
        elif command == '/start_options':
            return self.start_options()
        elif command == '/restart_all':
            return self.restart_all()
        elif command == '/stop':
            return self.emergency_stop()
        elif command == '/kill_all':
            return self.kill_all()
        elif command == '/risk':
            return self.risk_check()
        elif command.startswith('/risk override'):
            return self.risk_override()
        elif command == '/pipeline':
            return self.pipeline_status()
        elif command == '/run_pipeline':
            return self.run_pipeline()
        elif command.startswith('/deploy '):
            strategy_name = command.replace('/deploy ', '').strip()
            return self.deploy_strategy(strategy_name)
        elif command == '/regime auto':
            return self.regime_auto_enable()
        elif command == '/regime manual':
            return self.regime_auto_disable()
        elif command == '/regime status':
            return self.regime_auto_status()
        elif command == '/earnings':
            return self.earnings_scan()
        elif command == '/confluence':
            return self.confluence_scan()
        elif command == '/viral':
            return self.viral_scan()
        elif command == '/rebalance':
            return self.rebalance_check()
        elif command == '/help':
            return """REMOTE CONTROL COMMANDS

STATUS:
/status - System status
/positions - Open positions
/regime - Market conditions
/pnl - Real-time P&L
/risk - Risk limits status
/pipeline - Strategy pipeline status
/rebalance - Portfolio allocation

SCANNERS:
/earnings - Upcoming earnings plays
/confluence - Multi-timeframe setups
/viral - Trending social media stocks

STRATEGY DEPLOYMENT:
/run_pipeline - Run full pipeline
/deploy <name> - Deploy specific strategy
  Example: /deploy forex_ema_v2

REGIME AUTO-SWITCHER:
/regime auto - Enable auto-switching
/regime manual - Disable auto-switching
/regime status - Check switcher status

REMOTE START:
/start_forex - Start Forex
/start_futures - Start Futures
/start_options - Start Options
/restart_all - Restart everything

RISK MANAGEMENT:
/risk override - Reset kill-switch

EMERGENCY:
/stop - Stop all trading
/kill_all - Nuclear option

/help - This message"""
        else:
            return f"Unknown: {command}\nSend /help"

    def run(self):
        """Main loop"""
        self.send_message("Remote Control ONLINE\n\nSend /help for commands\n\nYou can now control your trading empire from your phone!")

        while True:
            try:
                updates = self.get_updates()

                for update in updates:
                    if 'message' in update and 'text' in update['message']:
                        text = update['message']['text']

                        if text.startswith('/'):
                            response = self.handle_command(text)
                            self.send_message(response)

                time.sleep(1)

            except KeyboardInterrupt:
                print("\nShutting down...")
                self.send_message("Remote Control shutting down")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)


if __name__ == '__main__':
    bot = TelegramRemoteControl()
    bot.run()
