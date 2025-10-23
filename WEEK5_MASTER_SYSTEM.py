#!/usr/bin/env python3
"""
WEEK 5+ MASTER TRADING SYSTEM
==============================
Complete institutional-grade trading system for $8M across 80 accounts

Architecture:
1. Multi-account orchestration (80 accounts × $100k)
2. Advanced strategies (Iron Condor, Butterfly, Dual Options)
3. Risk management (Kelly Criterion, Correlation, Greeks)
4. Analytics (IV Surface, Options Flow, Portfolio Analysis)
5. Real-time monitoring dashboard

Usage:
    python WEEK5_MASTER_SYSTEM.py

Requirements:
    - accounts_config.json with 80 account credentials
    - All Week 5+ modules installed
    - QuantLib, Greeks, ML/DL/RL systems active
"""

import asyncio
from datetime import datetime
import os
import sys

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

# Import all Week 5+ modules
from orchestration.multi_account_orchestrator import MultiAccountOrchestrator
from strategies.iron_condor_engine import IronCondorEngine
from strategies.butterfly_spread_engine import ButterflySpreadEngine
from core.adaptive_dual_options_engine import AdaptiveDualOptionsEngine
from analytics.portfolio_correlation_analyzer import PortfolioCorrelationAnalyzer
from analytics.kelly_criterion_sizer import KellyCriterionSizer
from analytics.volatility_surface_analyzer import VolatilitySurfaceAnalyzer
from analytics.options_flow_detector import OptionsFlowDetector
from week2_sp500_scanner import Week2SP500Scanner


class Week5MasterSystem:
    """Master trading system coordinating all Week 5+ components"""

    def __init__(self):
        print("="*80)
        print("INITIALIZING WEEK 5+ MASTER TRADING SYSTEM")
        print("="*80)

        # Core systems
        self.orchestrator = MultiAccountOrchestrator()
        self.scanner = None  # Will be initialized async

        # Strategy engines
        self.iron_condor = IronCondorEngine()
        self.butterfly = ButterflySpreadEngine()
        self.dual_options = AdaptiveDualOptionsEngine()

        # Analytics
        self.correlation_analyzer = PortfolioCorrelationAnalyzer()
        self.kelly_sizer = KellyCriterionSizer(kelly_fraction=0.25)
        self.vol_analyzer = VolatilitySurfaceAnalyzer()
        self.flow_detector = OptionsFlowDetector()

        # State
        self.daily_trades = []
        self.max_trades_per_day = 100  # With 80 accounts, can do many more trades

        print(f"\n[OK] All systems initialized")
        print(f"Accounts: {len(self.orchestrator.accounts)}")
        print(f"Total Capital: ${sum([a['max_allocation'] for a in self.orchestrator.accounts]):,}")

    async def run_daily_cycle(self):
        """Run full trading cycle for the day"""

        print(f"\n{'='*80}")
        print(f"STARTING DAILY TRADING CYCLE")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*80}")

        # 1. Pre-market analysis (6:00 AM)
        await self.pre_market_analysis()

        # 2. Market hours trading (6:30 AM - 1:00 PM PDT)
        await self.market_hours_trading()

        # 3. Post-market analysis (after 1:00 PM)
        await self.post_market_analysis()

    async def pre_market_analysis(self):
        """Pre-market preparation and analysis"""

        print(f"\n{'='*80}")
        print("PRE-MARKET ANALYSIS")
        print(f"{'='*80}")

        # 1. Get aggregate portfolio status
        print("\n[1/5] Checking aggregate portfolio...")
        aggregate = self.orchestrator.get_aggregate_portfolio()

        # 2. Analyze correlations in current positions
        print("\n[2/5] Analyzing portfolio correlations...")
        current_symbols = list(aggregate['positions'].keys())
        if current_symbols:
            diversification_score = self.correlation_analyzer.get_portfolio_diversification_score(current_symbols)
            clusters = self.correlation_analyzer.identify_clusters(current_symbols)

        # 3. Scan for high IV opportunities
        print("\n[3/5] Scanning for high IV opportunities...")
        test_symbols = ['AAPL', 'NVDA', 'TSLA', 'AMD', 'INTC', 'MSFT', 'GOOGL', 'META']
        high_iv_stocks = self.vol_analyzer.find_high_iv_opportunities(test_symbols)

        # 4. Detect unusual options flow
        print("\n[4/5] Detecting unusual options flow...")
        unusual_flows = self.flow_detector.scan_multiple_symbols(test_symbols, volume_multiplier=2.5)

        # 5. Generate trading plan
        print("\n[5/5] Generating trading plan...")
        self.trading_plan = {
            'high_iv_opportunities': high_iv_stocks,
            'unusual_flows': unusual_flows,
            'portfolio_diversification': diversification_score if current_symbols else 100,
            'max_trades_today': self.max_trades_per_day - len(self.daily_trades)
        }

        print(f"\n[OK] Pre-market analysis complete")
        print(f"High IV Opportunities: {len(high_iv_stocks)}")
        print(f"Unusual Flow Events: {len(unusual_flows)}")
        print(f"Trades Remaining Today: {self.trading_plan['max_trades_today']}")

    async def market_hours_trading(self):
        """Execute trades during market hours"""

        print(f"\n{'='*80}")
        print("MARKET HOURS TRADING")
        print(f"{'='*80}")

        # Initialize Week 2 scanner
        print("\n[SCANNER] Starting S&P 500 scanner...")
        self.scanner = Week2SP500Scanner()

        # Scan for opportunities
        opportunities = await self.scanner.scan_sp500_opportunities()

        if not opportunities:
            print(f"[WARNING] No opportunities found")
            return

        # Execute top opportunities
        top_opportunities = opportunities[:10]  # Top 10

        for i, opp in enumerate(top_opportunities, 1):
            if len(self.daily_trades) >= self.max_trades_per_day:
                print(f"\n[LIMIT] Daily trade limit reached ({self.max_trades_per_day})")
                break

            print(f"\n{'='*80}")
            print(f"OPPORTUNITY {i}/{len(top_opportunities)}: {opp['symbol']}")
            print(f"{'='*80}")

            # 1. Check correlation risk
            existing_symbols = [t['symbol'] for t in self.daily_trades]
            if not self.correlation_analyzer.check_new_position_correlation(
                opp['symbol'], existing_symbols, max_correlation=0.7
            ):
                print(f"[SKIP] Correlation risk too high")
                continue

            # 2. Analyze IV for strategy selection
            iv_analysis = self.vol_analyzer.get_implied_volatility_rank(opp['symbol'])

            # 3. Select strategy based on conditions
            strategy = self.select_optimal_strategy(opp, iv_analysis)

            # 4. Calculate position size using Kelly Criterion
            aggregate = self.orchestrator.get_aggregate_portfolio()
            total_capital = aggregate['total_equity']

            # Estimate win probability from momentum and Greeks
            win_prob = 0.70 if opp['momentum'] > 0.10 else 0.60
            profit_loss_ratio = 0.40

            position_size = self.kelly_sizer.calculate_kelly_size(
                win_prob, profit_loss_ratio, total_capital
            )

            # Convert to number of contracts
            estimated_cost_per_contract = 300  # Rough estimate
            total_contracts = max(1, int(position_size / estimated_cost_per_contract))

            # Distribute across 80 accounts
            total_contracts = min(total_contracts, len(self.orchestrator.accounts) * 2)  # Max 2 per account

            # 5. Execute distributed trade
            print(f"\n[EXECUTE] Distributing {total_contracts} contracts across {len(self.orchestrator.accounts)} accounts")

            trade_signal = {
                'symbol': opp['symbol'],
                'strategy': strategy,
                'strikes': {},
                'opportunity': opp
            }

            results = self.orchestrator.execute_distributed_trade(trade_signal, total_contracts)

            # Track trade
            self.daily_trades.append({
                'symbol': opp['symbol'],
                'strategy': strategy,
                'contracts': total_contracts,
                'results': results,
                'timestamp': datetime.now().isoformat()
            })

            # Wait 30 seconds between trades
            await asyncio.sleep(30)

        print(f"\n[COMPLETE] Market hours trading complete")
        print(f"Trades Executed: {len(self.daily_trades)}")

    def select_optimal_strategy(self, opportunity, iv_analysis):
        """Select best strategy based on conditions"""

        # High IV + neutral momentum = Iron Condor
        if iv_analysis and iv_analysis.get('iv_rank', 50) > 70 and abs(opportunity['momentum']) < 0.05:
            return 'IRON_CONDOR'

        # Moderate IV + low momentum = Butterfly
        elif abs(opportunity['momentum']) < 0.03:
            return 'BUTTERFLY'

        # Strong momentum + favorable regime = Dual Options
        else:
            return 'DUAL_OPTIONS'

    async def post_market_analysis(self):
        """Post-market reporting and analysis"""

        print(f"\n{'='*80}")
        print("POST-MARKET ANALYSIS")
        print(f"{'='*80}")

        # 1. Get final aggregate portfolio
        aggregate = self.orchestrator.get_aggregate_portfolio()

        # 2. Generate performance report
        print(f"\n{'='*80}")
        print("DAILY PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Trades Executed: {len(self.daily_trades)}")
        print(f"Total Equity: ${aggregate['total_equity']:,.2f}")
        print(f"Total P&L: ${aggregate['total_pnl']:+,.2f} ({(aggregate['total_pnl']/aggregate['total_equity'])*100:+.2f}%)")

        # 3. Save daily report
        report_filename = f"week5_report_{datetime.now().strftime('%Y%m%d')}.json"
        import json
        with open(report_filename, 'w') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'trades': self.daily_trades,
                'aggregate': {
                    'total_equity': aggregate['total_equity'],
                    'total_pnl': aggregate['total_pnl'],
                    'accounts': len(self.orchestrator.accounts)
                }
            }, f, indent=2)

        print(f"\n[SAVED] Report: {report_filename}")


async def main():
    """Main entry point"""

    print("\n" + "="*80)
    print("WEEK 5+ MASTER TRADING SYSTEM")
    print("Autonomous Trading Across 80 Accounts ($8M Capital)")
    print("="*80 + "\n")

    # Check if accounts config exists
    if not os.path.exists('accounts_config.json'):
        print("[WARNING] accounts_config.json not found")
        print("Creating sample config...")
        from orchestration.multi_account_orchestrator import create_sample_config
        create_sample_config()
        print("\n[ACTION REQUIRED] Edit accounts_config.json with your 80 account credentials")
        print("Then run this script again")
        return

    # Initialize master system
    master_system = Week5MasterSystem()

    # Run daily cycle
    await master_system.run_daily_cycle()

    print("\n" + "="*80)
    print("WEEK 5+ MASTER SYSTEM - DAY COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
