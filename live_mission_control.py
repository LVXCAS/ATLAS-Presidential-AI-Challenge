#!/usr/bin/env python3
"""
LIVE MISSION CONTROL MONITOR
=============================
Real-time dashboard showing all system activity
"""

import asyncio
import os
from datetime import datetime
from mission_control_logger import MissionControlLogger
import json

class LiveMissionControl:
    """Live monitoring dashboard"""

    def __init__(self):
        self.logger = MissionControlLogger()
        self.week2_report_file = None

        # Find latest Week 2 report
        try:
            import glob
            reports = glob.glob('week2_sp500_report_*.json')
            if reports:
                self.week2_report_file = max(reports)
        except:
            pass

    def get_week2_status(self):
        """Get Week 2 scanner status"""
        if not self.week2_report_file or not os.path.exists(self.week2_report_file):
            return {
                'scans': 0,
                'trades': 0,
                'last_update': 'No data'
            }

        try:
            with open(self.week2_report_file, 'r') as f:
                data = json.load(f)

            return {
                'scans': data.get('scans_completed', 0),
                'trades': data.get('trades_executed', 0),
                'last_update': data.get('date', 'Unknown'),
                'trade_list': data.get('trades', [])
            }
        except:
            return {
                'scans': 0,
                'trades': 0,
                'last_update': 'Error reading'
            }

    async def run_live_monitor(self):
        """Run live monitoring dashboard"""

        print("="*100)
        print("LIVE MISSION CONTROL - Press CTRL+C to exit")
        print("="*100)
        print()

        scan_num = 1

        while True:
            try:
                # Get Week 2 status
                week2_status = self.get_week2_status()

                # Display full dashboard
                os.system('cls' if os.name == 'nt' else 'clear')

                # Custom header with Week 2 info
                self.logger.display_header()
                self.logger.display_pnl_section()

                # Week 2 Scanner Status
                print(f"\033[95m\033[1m[WEEK 2 S&P 500 SCANNER STATUS]\033[0m")
                print(f"\033[97m{'-' * 100}")
                print(f"\033[97mScans Completed:  \033[96m{week2_status['scans']}")
                print(f"\033[97mTrades Executed:  \033[96m{week2_status['trades']}/3 (max)")
                print(f"\033[97mLast Update:      \033[96m{week2_status['last_update']}")
                print(f"\033[97mMode:             \033[92mPAPER TRADING (Alpaca)")
                print(f"\033[97mUniverse:         \033[96m503 S&P 500 stocks")
                print(f"\033[97mThreshold:        \033[96m2.8+ confidence")

                if week2_status.get('trade_list'):
                    print(f"\n\033[97mWeek 2 Trades Today:")
                    for i, trade in enumerate(week2_status['trade_list'][:5], 1):
                        symbol = trade.get('symbol', 'Unknown')
                        strategy = trade.get('strategy', 'Unknown')
                        score = trade.get('score', 0)
                        momentum = trade.get('momentum', 0)
                        print(f"  {i}. \033[96m{symbol}\033[97m - {strategy}")
                        print(f"     Score: \033[92m{score:.2f}\033[97m | Momentum: \033[92m{momentum:+.1%}\033[97m")

                print(f"\033[97m{'-' * 100}\n")

                self.logger.display_ml_systems()
                self.logger.display_agents()
                self.logger.display_active_positions()
                self.logger.display_system_metrics()

                # Footer with live update time
                print(f"\033[97m{'=' * 100}")
                print(f"\033[96m[LIVE] Updated: {datetime.now().strftime('%I:%M:%S %p')} | Refreshing every 15 seconds...")
                print(f"\033[96m[WEEK 1] 6 positions open, -$440 P&L | [WEEK 2] Scanner active, paper trading enabled")
                print(f"\033[97m{'=' * 100}\n")

                # Wait 15 seconds
                await asyncio.sleep(15)
                scan_num += 1

            except KeyboardInterrupt:
                print("\n\n[STOPPED] Live monitor stopped")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                await asyncio.sleep(15)


async def main():
    monitor = LiveMissionControl()
    await monitor.run_live_monitor()


if __name__ == "__main__":
    asyncio.run(main())
