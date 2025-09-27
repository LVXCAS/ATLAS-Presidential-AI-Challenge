#!/usr/bin/env python3
"""
MASTER AUTONOMOUS TRADING ENGINE
Complete autonomous cycle: Portfolio cleanup -> Intel-puts-style execution
The ultimate trading brain that scales your 68.3% ROI success
"""

import subprocess
import time
from datetime import datetime
import os
import sys
import threading
import queue
import logging
import json

class MasterAutonomousTradingEngine:
    """Master engine coordinating all autonomous trading systems"""

    def __init__(self):
        self.systems_running = {}
        self.execution_queue = queue.Queue()

        # Core autonomous systems
        self.core_systems = [
            'autonomous_portfolio_cleanup.py',
            'live_execution_monitor.py',
            'hybrid_conviction_genetic_trader.py',
            'level4_ai_trading_agent.py',
            'options_pricing_integration.py'
        ]

        # Background intelligence systems
        self.intelligence_systems = [
            'explosive_roi_hunter.py',
            'overnight_gap_scanner.py',
            'catalyst_news_monitor.py',
            'continuous_explosive_monitor.py'
        ]

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - MASTER - %(message)s')
        self.logger = logging.getLogger(__name__)

    def check_account_readiness(self):
        """Check if account is ready for Intel-puts-style trades"""
        try:
            import alpaca_trade_api as tradeapi
            from dotenv import load_dotenv

            load_dotenv()
            api = tradeapi.REST(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                os.getenv('ALPACA_BASE_URL')
            )

            account = api.get_account()
            buying_power = float(account.buying_power)

            return {
                'ready': buying_power >= 500000,  # Need $500K+ for Intel-puts-style trades
                'buying_power': buying_power,
                'cash': float(account.cash)
            }
        except Exception as e:
            self.logger.error(f"Error checking account readiness: {e}")
            return {'ready': False, 'buying_power': 0, 'cash': 0}

    def run_portfolio_cleanup_cycle(self):
        """Run one cycle of autonomous portfolio cleanup"""
        print(f"\n{'='*60}")
        print(f"AUTONOMOUS CLEANUP CYCLE - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        try:
            # Import and run cleanup directly for immediate execution
            from autonomous_portfolio_cleanup import AutonomousPortfolioCleanup

            cleanup_engine = AutonomousPortfolioCleanup()
            liquidated_positions = cleanup_engine.check_and_cleanup_portfolio()

            if liquidated_positions:
                print(f"CLEANUP SUCCESSFUL: {len(liquidated_positions)} positions liquidated")
                return True
            else:
                print("CLEANUP COMPLETE: No liquidation needed")
                return True

        except Exception as e:
            self.logger.error(f"Cleanup cycle failed: {e}")
            return False

    def launch_background_system(self, system_script):
        """Launch a background system"""
        try:
            process = subprocess.Popen(
                [sys.executable, system_script],
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self.systems_running[system_script] = {
                'process': process,
                'started': datetime.now(),
                'status': 'running'
            }

            print(f"[OK] Launched: {system_script}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to launch {system_script}: {e}")
            return False

    def check_system_health(self):
        """Check health of all running systems"""
        healthy_systems = 0

        for system_name, system_info in self.systems_running.items():
            process = system_info['process']

            if process.poll() is None:  # Still running
                healthy_systems += 1
                system_info['status'] = 'running'
            else:
                system_info['status'] = 'stopped'
                self.logger.warning(f"System stopped: {system_name}")

        return healthy_systems

    def run_execution_cycle(self):
        """Run the execution cycle for Intel-puts-style trades"""
        print(f"\n{'='*60}")
        print(f"INTEL-PUTS-STYLE EXECUTION CYCLE - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        # Check account readiness
        account_status = self.check_account_readiness()

        if not account_status['ready']:
            print(f"ACCOUNT NOT READY: Buying Power ${account_status['buying_power']:,.0f}")
            print("Running portfolio cleanup...")
            cleanup_success = self.run_portfolio_cleanup_cycle()

            if cleanup_success:
                # Wait for cleanup to settle
                time.sleep(30)

                # Re-check account status
                account_status = self.check_account_readiness()
                if account_status['ready']:
                    print(f"ACCOUNT NOW READY: Buying Power ${account_status['buying_power']:,.0f}")
                else:
                    print(f"STILL NOT READY: Buying Power ${account_status['buying_power']:,.0f}")
                    return False

        # Execute dual cash-secured put + long call strategy
        try:
            from hybrid_conviction_genetic_trader import HybridConvictionGeneticTrader
            from adaptive_dual_options_engine import AdaptiveDualOptionsEngine
            import asyncio

            trader = HybridConvictionGeneticTrader()
            dual_engine = AdaptiveDualOptionsEngine()

            # Run async method properly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            signals = loop.run_until_complete(trader.run_hybrid_analysis())
            loop.close()

            if signals:
                print(f"EXECUTION READY: {len(signals)} dual strategy opportunities identified")

                # Get current buying power for dual strategy execution
                account_status = self.check_account_readiness()
                buying_power = account_status['buying_power']

                # EXECUTE DUAL STRATEGY (replicating your 68.3% ROI method)
                executed_trades = dual_engine.execute_dual_strategy(signals, buying_power)

                if executed_trades:
                    print(f"DUAL STRATEGY EXECUTED: {len(executed_trades)} trades completed")
                    return True
                else:
                    print("DUAL STRATEGY EXECUTION FAILED")
                    return False
            else:
                print("NO EXECUTION OPPORTUNITIES FOUND")
                return False

        except Exception as e:
            self.logger.error(f"Execution cycle failed: {e}")
            return False

    def execute_options_trades(self, opportunities):
        """Actually execute the identified options trades"""
        try:
            import alpaca_trade_api as tradeapi
            from dotenv import load_dotenv

            load_dotenv()

            api = tradeapi.REST(
                key_id=os.getenv('ALPACA_API_KEY'),
                secret_key=os.getenv('ALPACA_SECRET_KEY'),
                base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
                api_version='v2'
            )

            account = api.get_account()
            buying_power = float(account.buying_power)

            print(f"EXECUTING {len(opportunities)} OPTIONS TRADES")
            print(f"Available buying power: ${buying_power:,.0f}")

            executed_count = 0

            for i, opp in enumerate(opportunities):
                try:
                    symbol = opp['symbol']
                    # OPTIMIZED ALLOCATION STRATEGY based on conviction and performance
                    conviction_multiplier = 1.0
                    if 'conviction_level' in opp:
                        conviction_multiplier = {'HIGH': 1.2, 'MEDIUM': 1.0, 'LOW': 0.8}.get(opp['conviction_level'], 1.0)

                    # Base allocation with Intel-puts-style concentration (your proven pattern)
                    base_allocations = [0.45, 0.25, 0.20, 0.10]  # More aggressive top pick like your Intel puts
                    base_allocation = base_allocations[i] if i < 4 else 0.05

                    # Apply conviction weighting
                    allocation = min(0.50, base_allocation * conviction_multiplier)  # Cap at 50% per position

                    print(f"ALLOCATION: {symbol} - {allocation:.1%} (conviction: {opp.get('conviction_level', 'MEDIUM')})")

                    # Try options first, fall back to stocks
                    options_executed = False

                    if 'CALL' in str(opp).upper() or 'PUT' in str(opp).upper():
                        try:
                            # USE ENHANCED OPTIONS CHECKER FOR RELIABLE EXECUTION
                            from enhanced_options_checker import EnhancedOptionsChecker

                            options_checker = EnhancedOptionsChecker()

                            # Determine target strike and option type
                            if 'target_strike' in opp and opp['target_strike']:
                                target_strike = opp['target_strike']
                                print(f"USING SCANNER STRIKE: {symbol} ${target_strike}")
                            else:
                                # SMART STRIKE CALCULATION: Use realistic strikes
                                try:
                                    bars = api.get_latest_bar(symbol)
                                    current_price = float(bars.c)

                                    if 'CALL' in str(opp).upper():
                                        target_strike = round(current_price * 1.03)  # 3% OTM calls
                                    else:
                                        target_strike = round(current_price * 0.97)  # 3% OTM puts

                                    print(f"SMART STRIKE: {symbol} @ ${current_price:.2f} -> ${target_strike}")

                                except Exception:
                                    # Realistic fallback strikes
                                    if symbol == 'SPY':
                                        target_strike = 680 if 'CALL' in str(opp).upper() else 640
                                    elif symbol == 'AAPL':
                                        target_strike = 265 if 'CALL' in str(opp).upper() else 245
                                    elif symbol == 'GOOGL':
                                        target_strike = 255 if 'CALL' in str(opp).upper() else 235
                                    elif symbol == 'META':
                                        target_strike = 750 if 'CALL' in str(opp).upper() else 700
                                    else:
                                        target_strike = 105 if 'CALL' in str(opp).upper() else 95

                            option_type = 'C' if 'CALL' in str(opp).upper() else 'P'

                            # USE ENHANCED CHECKER TO FIND BEST AVAILABLE OPTION
                            best_option = options_checker.get_best_option(symbol, target_strike, option_type)

                            if best_option['available']:
                                options_symbol = best_option['symbol']
                                actual_strike = best_option['strike']

                                # Calculate contracts with realistic pricing
                                trade_amount = buying_power * allocation
                                estimated_premium = (best_option['bid'] + best_option['ask']) / 2
                                contracts = min(1000, max(1, int(trade_amount / (estimated_premium * 100))))

                                print(f"ENHANCED OPTIONS: {symbol} {option_type} ${actual_strike} - {contracts} contracts")
                                print(f"  Using: {options_symbol} @ ${estimated_premium:.2f}")

                                # Submit options order
                                order = api.submit_order(
                                    symbol=options_symbol,
                                    qty=contracts,
                                    side='buy',
                                    type='market',
                                    time_in_force='day'
                                )

                                print(f"OPTIONS ORDER SUBMITTED: {order.id}")
                                executed_count += 1
                                options_executed = True

                            else:
                                print(f"NO SUITABLE OPTIONS: {best_option['reason']}")
                                print(f"Falling back to stock position for {symbol}")

                        except Exception as e:
                            print(f"Enhanced options failed for {symbol}: {e}")
                            print(f"Falling back to stock position for {symbol}")

                    if not options_executed:
                        # Stock fallback
                        quote = api.get_latest_quote(symbol)
                        price = float(quote.ask_price)
                        trade_amount = buying_power * allocation
                        shares = max(1, int(trade_amount / price))

                        print(f"EXECUTING STOCK: {symbol} - {shares} shares @ ${price:.2f}")

                        order = api.submit_order(
                            symbol=symbol,
                            qty=shares,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )

                        print(f"STOCK ORDER SUBMITTED: {order.id}")
                        executed_count += 1

                except Exception as e:
                    print(f"Failed to execute {symbol}: {e}")
                    continue

            print(f"EXECUTION COMPLETE: {executed_count}/{len(opportunities)} trades submitted")

            # TRACK EXECUTION METRICS FOR AI LEARNING
            self.save_execution_report({
                'timestamp': datetime.now().isoformat(),
                'total_opportunities': len(opportunities),
                'successful_executions': executed_count,
                'failed_executions': len(opportunities) - executed_count,
                'overall_metrics': {
                    'successful_executions': executed_count,
                    'total_executions': len(opportunities),
                    'success_rate': executed_count / len(opportunities) if opportunities else 0,
                    'average_slippage': 0.01,  # Estimate for market orders
                    'average_fill_time': 2.0   # Estimate 2 seconds
                }
            })

            return executed_count > 0

        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return False

    def save_execution_report(self, execution_data):
        """Save execution report for AI learning"""
        try:
            filename = f'execution_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(filename, 'w') as f:
                json.dump(execution_data, f, indent=2)
            print(f"Execution report saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save execution report: {e}")

    def get_next_friday(self):
        """Get next Friday for options expiration"""
        from datetime import datetime, timedelta
        today = datetime.now()
        days_until_friday = 4 - today.weekday()
        if days_until_friday <= 0:
            days_until_friday += 7
        next_friday = today + timedelta(days=days_until_friday)
        return next_friday.strftime('%y%m%d')

    def run_master_autonomous_cycle(self):
        """Run the complete autonomous trading cycle"""
        print("MASTER AUTONOMOUS TRADING ENGINE")
        print("=" * 80)
        print("Complete autonomous cycle: Cleanup -> Analysis -> Execution")
        print("Scaling your 68.3% ROI Intel-puts-style success")
        print("=" * 80)

        cycle_count = 0

        while True:
            try:
                cycle_count += 1
                print(f"\n[CYCLE] MASTER CYCLE #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Step 1: Health check of background systems
                healthy_systems = self.check_system_health()
                print(f"Background systems running: {healthy_systems}/{len(self.systems_running)}")

                # Step 2: Run execution cycle
                execution_success = self.run_execution_cycle()

                if execution_success:
                    print("[SUCCESS] CYCLE COMPLETE: Ready for Intel-puts-style execution")
                else:
                    print("[WARNING] CYCLE INCOMPLETE: Issues detected")

                # Step 3: Wait before next cycle
                wait_time = 300  # 5 minutes
                print(f"Next master cycle in {wait_time//60} minutes...")
                time.sleep(wait_time)

            except KeyboardInterrupt:
                print("\nMaster autonomous engine stopped by user")
                self.shutdown_all_systems()
                break
            except Exception as e:
                self.logger.error(f"Master cycle error: {e}")
                time.sleep(60)  # Wait 1 minute on error

    def launch_intelligence_systems(self):
        """Launch all background intelligence systems"""
        print("Launching background intelligence systems...")

        for system in self.intelligence_systems:
            if os.path.exists(system):
                self.launch_background_system(system)
            else:
                print(f"[WARNING] System not found: {system}")

        time.sleep(5)  # Let systems initialize

        healthy_systems = self.check_system_health()
        print(f"Background systems launched: {healthy_systems}/{len(self.intelligence_systems)}")

    def shutdown_all_systems(self):
        """Shutdown all running systems"""
        print("Shutting down all autonomous systems...")

        for system_name, system_info in self.systems_running.items():
            try:
                process = system_info['process']
                if process.poll() is None:  # Still running
                    process.terminate()
                    print(f"Stopped: {system_name}")
            except Exception as e:
                self.logger.error(f"Error stopping {system_name}: {e}")

    def run_master_engine(self):
        """Run the complete master autonomous trading engine"""
        try:
            # Launch background intelligence systems
            self.launch_intelligence_systems()

            # Run master autonomous cycle
            self.run_master_autonomous_cycle()

        except Exception as e:
            self.logger.error(f"Master engine error: {e}")
        finally:
            self.shutdown_all_systems()

def main():
    """Launch the master autonomous trading engine"""
    master_engine = MasterAutonomousTradingEngine()
    master_engine.run_master_engine()

if __name__ == "__main__":
    main()