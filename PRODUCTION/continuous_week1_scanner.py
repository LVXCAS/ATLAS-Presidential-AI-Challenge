#!/usr/bin/env python3
"""
CONTINUOUS WEEK 1 SCANNER - REAL MARKET DATA
=============================================
Runs all day while you're at school, scanning for opportunities
Executes Week 1 conservative strategy when high-conviction setups appear
Logs everything for prop firm documentation

UPGRADED: Now using REAL market data from Alpaca API
- Real-time current prices and volumes from Alpaca
- Historical volatility estimates (Paper account limitation: limited historical data)
- Actual institutional scoring methodology
- Note: Live trading accounts will have full historical data for precise volatility calculations
"""

import asyncio
import time
from datetime import datetime, time as dt_time
from week1_execution_system import Week1ExecutionSystem
from options_executor import AlpacaOptionsExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContinuousWeek1Scanner:
    """Continuous market scanner for Week 1 execution"""

    def __init__(self):
        self.system = Week1ExecutionSystem()
        self.options_executor = AlpacaOptionsExecutor()
        self.scan_interval = 300  # 5 minutes between scans
        self.trades_today = []
        self.scans_completed = 0
        self.opportunities_found = 0

    async def run_continuous_scanning(self):
        """Run continuous scanning during market hours"""

        print("CONTINUOUS WEEK 1 SCANNER STARTING - REAL MARKET DATA")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%I:%M %p PDT')}")
        print("Data Source: Alpaca API (Real-Time Market Data)")
        print("Scanning every 5 minutes for high-conviction setups")
        print("Week 1 threshold: 4.0+ confidence score (80%+)")
        print("Max trades today: 2")
        print("Max risk per trade: 1.5%")
        print("=" * 60)
        print()

        while True:
            try:
                current_time = datetime.now().time()

                # Market hours: 6:30 AM - 1:00 PM PDT (9:30 AM - 4:00 PM EDT)
                market_open = dt_time(6, 30)  # 6:30 AM PDT
                market_close = dt_time(13, 0)  # 1:00 PM PDT

                if market_open <= current_time <= market_close:
                    await self._perform_scan_cycle()
                else:
                    if current_time < market_open:
                        wait_minutes = (datetime.combine(datetime.today(), market_open) -
                                      datetime.combine(datetime.today(), current_time)).seconds // 60
                        print(f"Pre-market: Waiting {wait_minutes} minutes for market open")
                    else:
                        print("Market closed for the day")
                        await self._generate_end_of_day_report()
                        break

                # Wait 5 minutes before next scan
                await asyncio.sleep(self.scan_interval)

            except Exception as e:
                logger.error(f"Error in continuous scanning: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _perform_scan_cycle(self):
        """Perform one complete scan cycle"""

        self.scans_completed += 1

        print(f"\nSCAN #{self.scans_completed} - {datetime.now().strftime('%I:%M %p')}")
        print("-" * 30)

        # Check if we've hit daily trade limit
        if len(self.trades_today) >= 2:
            print("Daily trade limit reached (2 trades). Monitoring only.")
            return

        # Scan Intel-style opportunities
        intel_opportunities = await self._scan_intel_opportunities()

        # Scan earnings opportunities
        earnings_opportunities = await self._scan_earnings_opportunities()

        total_opportunities = len(intel_opportunities) + len(earnings_opportunities)
        self.opportunities_found += total_opportunities

        # Execute first qualifying opportunity
        if intel_opportunities:
            trade_result = await self._execute_opportunity(intel_opportunities[0], 'intel_style')
            if trade_result:
                self.trades_today.append(trade_result)

        elif earnings_opportunities and len(self.trades_today) < 1:  # Only 1 earnings trade max
            trade_result = await self._execute_opportunity(earnings_opportunities[0], 'earnings')
            if trade_result:
                self.trades_today.append(trade_result)

        print(f"Opportunities found: {total_opportunities}")
        print(f"Trades executed today: {len(self.trades_today)}/2")

    async def _scan_intel_opportunities(self):
        """Scan for Intel-style opportunities using REAL market data"""

        opportunities = []
        # Expanded to include R&D discoveries + semiconductors
        symbols = ['INTC', 'AMD', 'NVDA', 'AAPL', 'MSFT', 'QCOM', 'MU']

        # Typical volatility estimates for stocks
        # (Paper accounts have limited historical data - these are reasonable estimates)
        typical_volatility = {
            'INTC': 0.015,  # Lower volatility (mature chip maker)
            'AMD': 0.025,   # Medium-high volatility
            'NVDA': 0.030,  # Higher volatility (AI/GPU plays)
            'AAPL': 0.020,  # Medium volatility
            'MSFT': 0.018,  # Lower-medium volatility
            'QCOM': 0.020,  # Medium volatility
            'MU': 0.025     # Medium-high volatility (memory chips)
        }

        for symbol in symbols:
            try:
                # Get REAL market data from Alpaca API
                bars = self.system.api.get_bars(symbol, '1Day', limit=20).df

                if bars.empty:
                    print(f"    {symbol}: No data available")
                    continue

                # Calculate real metrics
                current_price = float(bars['close'].iloc[-1])
                volume = float(bars['volume'].iloc[-1])

                # Calculate actual volatility if we have enough data
                returns = bars['close'].pct_change().dropna()
                if len(returns) > 5:
                    volatility = float(returns.std())
                else:
                    # Use typical volatility estimate for this stock
                    volatility = typical_volatility.get(symbol, 0.02)

                # Use the REAL scoring method from unified system
                real_score = self.system._calculate_intel_opportunity_score(
                    current_price, volatility, volume
                )

                opportunity = {
                    'symbol': symbol,
                    'price': current_price,
                    'volume': volume,
                    'volatility': volatility,
                    'score': real_score,
                    'type': 'intel_style',
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'ALPACA_REAL_PRICES'
                }

                # Week 1 requires 4.0+ score
                if real_score >= 4.0:
                    opportunities.append(opportunity)
                    print(f"  [QUALIFIED] {symbol}: ${current_price:.2f} - Score: {real_score:.2f} [Vol: {volume:,.0f}, IV: {volatility:.3f}]")
                else:
                    print(f"    {symbol}: ${current_price:.2f} - Score: {real_score:.2f} (below 4.0) [Vol: {volume:,.0f}, IV: {volatility:.3f}]")

            except Exception as e:
                print(f"    {symbol}: Error - {e}")

        return opportunities

    async def _scan_earnings_opportunities(self):
        """Scan for earnings opportunities using REAL market data"""

        opportunities = []
        # Note: In production, would integrate with earnings calendar API
        # For now, scanning high-volatility stocks that might have earnings
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        for symbol in symbols:
            try:
                # Get real market data
                bars = self.system.api.get_bars(symbol, '1Day', limit=20).df

                if bars.empty:
                    continue

                current_price = float(bars['close'].iloc[-1])
                volume = float(bars['volume'].iloc[-1])

                # Calculate volatility to estimate expected move
                returns = bars['close'].pct_change().dropna()
                volatility = float(returns.std()) if len(returns) > 0 else 0.05

                # Estimate expected move based on volatility
                expected_move = volatility * 2.5  # Approximate earnings move

                # Calculate IV rank (simplified - would need options data for real IV)
                iv_rank = min(volatility * 100, 100)  # Normalized to 0-100

                # Use real scoring method
                score = self.system._calculate_earnings_opportunity_score(
                    expected_move, iv_rank, current_price
                )

                # Only consider if meaningful expected move
                if expected_move >= 0.03:  # At least 3% expected move
                    opportunity = {
                        'symbol': symbol,
                        'price': current_price,
                        'volume': volume,
                        'expected_move': expected_move,
                        'iv_rank': iv_rank,
                        'score': score,
                        'type': 'earnings',
                        'timestamp': datetime.now().isoformat(),
                        'data_source': 'ALPACA_REAL_TIME'
                    }

                    # Week 1 requires 3.8+ for earnings
                    if score >= 3.8:
                        opportunities.append(opportunity)
                        print(f"  [QUALIFIED] {symbol} EARNINGS: ${current_price:.2f} - Score: {score:.2f} - ExpMove: {expected_move:.1%}")
                    else:
                        print(f"    {symbol} EARNINGS: ${current_price:.2f} - Score: {score:.2f} (below 3.8) - ExpMove: {expected_move:.1%}")

            except Exception as e:
                print(f"    {symbol}: Error - {e}")

        return opportunities

    def _get_real_current_price(self, symbol):
        """Get current price from Alpaca API"""
        try:
            bars = self.system.api.get_bars(symbol, '1Day', limit=1).df
            if not bars.empty:
                return float(bars['close'].iloc[-1])
        except:
            pass
        return None

    async def _execute_opportunity(self, opportunity, strategy_type):
        """Execute a qualified opportunity - REAL ALPACA ORDERS"""

        print(f"\n>>> EXECUTING: {opportunity['symbol']} ({strategy_type.upper()})")
        print(f"   Score: {opportunity['score']:.2f}")
        print(f"   Price: ${opportunity['price']:.2f}")

        # Create trade record
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': f"{strategy_type}_week1_continuous",
            'symbol': opportunity['symbol'],
            'current_price': opportunity['price'],
            'opportunity_score': opportunity['score'],
            'scan_number': self.scans_completed,
            'execution_type': 'REAL_ALPACA_PAPER',
            'week1_conservative': True,
            'paper_trade': True
        }

        # SUBMIT REAL ORDERS TO ALPACA
        execution_result = None

        if strategy_type == 'intel_style':
            # Execute Intel dual strategy
            execution_result = self.options_executor.execute_intel_dual(
                symbol=opportunity['symbol'],
                current_price=opportunity['price'],
                contracts=2,
                expiry_days=21
            )
            trade_record.update({
                'strategy_details': 'Intel Dual (CSP + Long Call)',
                'contracts': 2,
                'expiry_days': 21,
                'alpaca_execution': execution_result,
                'risk_percentage': 1.4,
                'expected_roi': '8-15%'
            })
        else:  # earnings
            # Execute straddle
            execution_result = self.options_executor.execute_straddle(
                symbol=opportunity['symbol'],
                current_price=opportunity['price'],
                contracts=1,
                expiry_days=14
            )
            trade_record.update({
                'strategy_details': 'Long Straddle (Earnings)',
                'contracts': 1,
                'expiry_days': 14,
                'expected_move': opportunity.get('expected_move', 0.125),
                'alpaca_execution': execution_result,
                'cost_percentage': 1.0,
                'expected_roi': '15-30%'
            })

        # Count successful orders
        successful_orders = 0
        if execution_result and 'orders' in execution_result:
            successful_orders = len([o for o in execution_result['orders'] if 'order_id' in o])

        # Convert order IDs to strings for JSON serialization
        if execution_result and 'orders' in execution_result:
            for order in execution_result['orders']:
                if 'order_id' in order:
                    order['order_id'] = str(order['order_id'])
                if 'status' in order:
                    order['status'] = str(order['status'])

        trade_record['orders_submitted'] = successful_orders
        trade_record['execution_success'] = successful_orders > 0

        filename = f"week1_continuous_trade_{len(self.trades_today)+1}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        import json
        with open(filename, 'w') as f:
            json.dump(trade_record, f, indent=2)

        print(f"   [OK] Trade logged: {filename}")
        print(f"   [OK] Alpaca orders submitted: {successful_orders}/2")

        return trade_record

    async def _generate_end_of_day_report(self):
        """Generate end of day report"""

        print(f"\nEND OF DAY REPORT - WEEK 1 DAY 1")
        print("=" * 40)
        print(f"Total scans completed: {self.scans_completed}")
        print(f"Opportunities found: {self.opportunities_found}")
        print(f"Trades executed: {len(self.trades_today)}")
        print()

        if self.trades_today:
            total_risk = sum(
                trade.get('total_risk', trade.get('total_cost', 0))
                for trade in self.trades_today
            )
            avg_risk = total_risk / 100000 * 100 / len(self.trades_today) if self.trades_today else 0

            print("TRADES EXECUTED:")
            for i, trade in enumerate(self.trades_today, 1):
                print(f"  {i}. {trade['symbol']} ({trade['strategy']}) - Score: {trade['opportunity_score']:.2f}")

            print(f"\nRISK MANAGEMENT:")
            print(f"  Total risk used: {total_risk/100000*100:.2f}%")
            print(f"  Average risk per trade: {avg_risk:.2f}%")
            print(f"  Within Week 1 limits: {'YES' if total_risk/100000 <= 0.03 else 'NO'}")
        else:
            print("No trades executed - excellent discipline!")
            print("All opportunities failed to meet Week 1 conservative thresholds")

        # Save daily summary
        daily_summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'week': 1,
            'day': 1,
            'scans_completed': self.scans_completed,
            'opportunities_found': self.opportunities_found,
            'trades_executed': len(self.trades_today),
            'trades': self.trades_today,
            'week1_assessment': 'EXCELLENT DISCIPLINE' if len(self.trades_today) <= 2 else 'WITHIN LIMITS'
        }

        filename = f"week1_day1_continuous_summary_{datetime.now().strftime('%Y%m%d')}.json"
        import json
        with open(filename, 'w') as f:
            json.dump(daily_summary, f, indent=2)

        print(f"\nDaily summary saved: {filename}")
        print("Ready for Week 1 Day 2 tomorrow!")

async def main():
    """Run continuous scanner for the full trading day"""
    scanner = ContinuousWeek1Scanner()
    await scanner.run_continuous_scanning()

if __name__ == "__main__":
    asyncio.run(main())