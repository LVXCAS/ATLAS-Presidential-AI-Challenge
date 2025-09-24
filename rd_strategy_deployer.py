"""
ENHANCED R&D STRATEGY DEPLOYMENT SYSTEM
Deploys high-Sharpe R&D strategies with real capital allocation
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('rd_deployment.log'),
        logging.StreamHandler()
    ]
)

class RDStrategyDeployer:
    """Deploy R&D strategies with real capital allocation"""

    def __init__(self):
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        self.deployment_capital = 200000  # $200K for R&D strategies
        self.min_sharpe_threshold = 1.0  # Only deploy strategies with Sharpe > 1.0

    async def load_rd_analysis(self) -> Dict:
        """Load latest R&D analysis results"""

        try:
            with open('rd_analysis_20250919_094742.json', 'r') as f:
                rd_data = json.load(f)

            logging.info("R&D analysis loaded successfully")
            return rd_data

        except Exception as e:
            logging.error(f"Failed to load R&D analysis: {e}")
            return {}

    async def filter_high_quality_strategies(self, rd_data: Dict) -> List[Dict]:
        """Filter and rank strategies by quality metrics"""

        logging.info("FILTERING HIGH-QUALITY R&D STRATEGIES")
        logging.info("=" * 45)

        high_quality_strategies = []

        # Process momentum strategies
        momentum_strategies = rd_data.get('momentum_strategies', {})
        for symbol, data in momentum_strategies.items():
            if data['sharpe'] >= self.min_sharpe_threshold:
                strategy = {
                    'type': 'MOMENTUM',
                    'symbol': symbol,
                    'sharpe_ratio': data['sharpe'],
                    'total_return': data['total_return'],
                    'lookback': data['lookback'],
                    'threshold': data['threshold'],
                    'quality_score': data['sharpe'] * data['total_return']
                }
                high_quality_strategies.append(strategy)
                logging.info(f"✓ MOMENTUM {symbol}: Sharpe {data['sharpe']:.2f}, Return {data['total_return']:.1%}")

        # Process mean reversion strategies
        mean_reversion_strategies = rd_data.get('mean_reversion_strategies', {})
        for symbol, data in mean_reversion_strategies.items():
            if data['sharpe'] >= self.min_sharpe_threshold:
                strategy = {
                    'type': 'MEAN_REVERSION',
                    'symbol': symbol,
                    'sharpe_ratio': data['sharpe'],
                    'total_return': data['total_return'],
                    'lookback': data['lookback'],
                    'std_threshold': data['std_threshold'],
                    'quality_score': data['sharpe'] * data['total_return']
                }
                high_quality_strategies.append(strategy)
                logging.info(f"✓ MEAN_REV {symbol}: Sharpe {data['sharpe']:.2f}, Return {data['total_return']:.1%}")

        # Sort by quality score (Sharpe * Return)
        high_quality_strategies.sort(key=lambda x: x['quality_score'], reverse=True)

        logging.info(f"\nHigh-Quality Strategies Found: {len(high_quality_strategies)}")
        return high_quality_strategies

    async def calculate_strategy_allocations(self, strategies: List[Dict]) -> Dict:
        """Calculate optimal capital allocation for each strategy"""

        logging.info("CALCULATING STRATEGY ALLOCATIONS")
        logging.info("=" * 35)

        if not strategies:
            return {}

        # Weight allocation by quality score
        total_quality_score = sum(s['quality_score'] for s in strategies)

        allocations = {}
        for strategy in strategies[:5]:  # Top 5 strategies only
            weight = strategy['quality_score'] / total_quality_score
            allocation = self.deployment_capital * weight * 0.8  # 80% deployment rate

            allocations[f"{strategy['type']}_{strategy['symbol']}"] = {
                'strategy': strategy,
                'allocation': allocation,
                'weight': weight,
                'target_contracts': max(1, int(allocation / 10000))  # Rough estimation
            }

            logging.info(f"{strategy['type']} {strategy['symbol']}: ${allocation:,.0f} ({weight:.1%})")

        total_allocated = sum(a['allocation'] for a in allocations.values())
        logging.info(f"Total Allocation: ${total_allocated:,.0f}")

        return allocations

    async def generate_trading_signals(self, strategies: List[Dict]) -> List[Dict]:
        """Generate actual trading signals from R&D strategies"""

        logging.info("GENERATING TRADING SIGNALS")
        logging.info("=" * 30)

        signals = []

        for strategy in strategies[:3]:  # Top 3 strategies
            try:
                # Get current market data
                symbol = strategy['symbol']
                bars = self.api.get_bars(symbol, tradeapi.TimeFrame.Day, limit=50).df
                current_price = bars['close'].iloc[-1]

                if strategy['type'] == 'MOMENTUM':
                    # Momentum signal generation
                    lookback = strategy['lookback']
                    threshold = strategy['threshold']

                    returns = bars['close'].pct_change(lookback).iloc[-1]

                    if returns > threshold:
                        signal = {
                            'symbol': symbol,
                            'strategy_type': 'MOMENTUM_LONG',
                            'action': 'BUY',
                            'current_price': current_price,
                            'signal_strength': min(returns / threshold, 3.0),
                            'expected_return': strategy['total_return'],
                            'sharpe_ratio': strategy['sharpe_ratio']
                        }
                        signals.append(signal)
                        logging.info(f"🔥 MOMENTUM SIGNAL: {symbol} BUY at ${current_price:.2f}")

                elif strategy['type'] == 'MEAN_REVERSION':
                    # Mean reversion signal generation
                    lookback = strategy['lookback']
                    std_threshold = strategy['std_threshold']

                    rolling_mean = bars['close'].rolling(lookback).mean().iloc[-1]
                    rolling_std = bars['close'].rolling(lookback).std().iloc[-1]
                    z_score = (current_price - rolling_mean) / rolling_std

                    if z_score < -std_threshold:  # Oversold
                        signal = {
                            'symbol': symbol,
                            'strategy_type': 'MEAN_REVERSION_LONG',
                            'action': 'BUY',
                            'current_price': current_price,
                            'signal_strength': abs(z_score / std_threshold),
                            'expected_return': strategy['total_return'],
                            'sharpe_ratio': strategy['sharpe_ratio']
                        }
                        signals.append(signal)
                        logging.info(f"🔄 MEAN_REV SIGNAL: {symbol} BUY at ${current_price:.2f}")
                    elif z_score > std_threshold:  # Overbought
                        signal = {
                            'symbol': symbol,
                            'strategy_type': 'MEAN_REVERSION_SHORT',
                            'action': 'SELL',
                            'current_price': current_price,
                            'signal_strength': z_score / std_threshold,
                            'expected_return': strategy['total_return'],
                            'sharpe_ratio': strategy['sharpe_ratio']
                        }
                        signals.append(signal)
                        logging.info(f"🔄 MEAN_REV SIGNAL: {symbol} SELL at ${current_price:.2f}")

            except Exception as e:
                logging.error(f"Signal generation error for {strategy['symbol']}: {e}")

        logging.info(f"Generated {len(signals)} trading signals")
        return signals

    async def execute_rd_deployment(self, signals: List[Dict], allocations: Dict) -> Dict:
        """Execute R&D strategy deployment with real orders"""

        logging.info("EXECUTING R&D STRATEGY DEPLOYMENT")
        logging.info("=" * 40)

        execution_results = {
            'timestamp': datetime.now().isoformat(),
            'signals_processed': len(signals),
            'orders_placed': 0,
            'total_deployed': 0,
            'successful_deployments': [],
            'failed_deployments': []
        }

        for signal in signals:
            try:
                symbol = signal['symbol']
                action = signal['action']
                allocation_key = None

                # Find matching allocation
                for key, alloc in allocations.items():
                    if alloc['strategy']['symbol'] == symbol:
                        allocation_key = key
                        break

                if not allocation_key:
                    continue

                allocation = allocations[allocation_key]['allocation']
                shares = max(1, int(allocation / signal['current_price']))

                # Place market order
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side=action.lower(),
                    type='market',
                    time_in_force='day'
                )

                deployment = {
                    'symbol': symbol,
                    'strategy_type': signal['strategy_type'],
                    'action': action,
                    'shares': shares,
                    'allocation': allocation,
                    'order_id': order.id,
                    'expected_return': signal['expected_return'],
                    'sharpe_ratio': signal['sharpe_ratio']
                }

                execution_results['successful_deployments'].append(deployment)
                execution_results['orders_placed'] += 1
                execution_results['total_deployed'] += allocation

                logging.info(f"✅ DEPLOYED: {action} {shares} {symbol} (${allocation:,.0f})")

            except Exception as e:
                failed_deployment = {
                    'symbol': signal['symbol'],
                    'error': str(e),
                    'signal': signal
                }
                execution_results['failed_deployments'].append(failed_deployment)
                logging.error(f"❌ DEPLOYMENT FAILED: {signal['symbol']} - {e}")

        # Save execution results
        with open('rd_deployment_results.json', 'w') as f:
            json.dump(execution_results, f, indent=2)

        logging.info("=" * 40)
        logging.info(f"R&D DEPLOYMENT COMPLETE")
        logging.info(f"Orders Placed: {execution_results['orders_placed']}")
        logging.info(f"Capital Deployed: ${execution_results['total_deployed']:,.0f}")
        logging.info(f"Success Rate: {len(execution_results['successful_deployments'])}/{len(signals)}")

        return execution_results

    async def monitor_rd_performance(self) -> Dict:
        """Monitor performance of deployed R&D strategies"""

        logging.info("MONITORING R&D STRATEGY PERFORMANCE")
        logging.info("=" * 38)

        try:
            with open('rd_deployment_results.json', 'r') as f:
                deployment_data = json.load(f)
        except:
            logging.error("No deployment data found")
            return {}

        portfolio_performance = {
            'total_deployed': deployment_data['total_deployed'],
            'active_positions': 0,
            'total_unrealized_pl': 0,
            'individual_performance': []
        }

        # Check performance of each deployment
        for deployment in deployment_data.get('successful_deployments', []):
            try:
                symbol = deployment['symbol']
                position = self.api.get_position(symbol)

                unrealized_pl = float(position.unrealized_pl)
                unrealized_plpc = float(position.unrealized_plpc)

                performance = {
                    'symbol': symbol,
                    'strategy_type': deployment['strategy_type'],
                    'shares': deployment['shares'],
                    'unrealized_pl': unrealized_pl,
                    'unrealized_plpc': unrealized_plpc,
                    'expected_return': deployment['expected_return'],
                    'sharpe_ratio': deployment['sharpe_ratio']
                }

                portfolio_performance['individual_performance'].append(performance)
                portfolio_performance['active_positions'] += 1
                portfolio_performance['total_unrealized_pl'] += unrealized_pl

                logging.info(f"{symbol}: ${unrealized_pl:,.0f} ({unrealized_plpc:.1%})")

            except Exception as e:
                logging.warning(f"Could not get performance for {deployment['symbol']}: {e}")

        total_return_pct = portfolio_performance['total_unrealized_pl'] / portfolio_performance['total_deployed'] * 100
        logging.info(f"Total R&D Portfolio P&L: ${portfolio_performance['total_unrealized_pl']:,.0f} ({total_return_pct:.1%})")

        return portfolio_performance

    async def run_full_rd_deployment_cycle(self):
        """Run complete R&D deployment cycle"""

        logging.info("R&D STRATEGY DEPLOYMENT SYSTEM")
        logging.info("Advanced Strategy Deployment with Real Capital")
        logging.info("=" * 55)

        # Load R&D analysis
        rd_data = await self.load_rd_analysis()
        if not rd_data:
            logging.error("No R&D data available")
            return

        # Filter high-quality strategies
        high_quality_strategies = await self.filter_high_quality_strategies(rd_data)
        if not high_quality_strategies:
            logging.error("No high-quality strategies found")
            return

        # Calculate allocations
        allocations = await self.calculate_strategy_allocations(high_quality_strategies)

        # Generate trading signals
        signals = await self.generate_trading_signals(high_quality_strategies)

        # Execute deployment
        if signals:
            execution_results = await self.execute_rd_deployment(signals, allocations)

            # Monitor performance
            await asyncio.sleep(30)  # Wait for orders to fill
            performance = await self.monitor_rd_performance()

            logging.info("R&D DEPLOYMENT CYCLE COMPLETE")
            return {
                'execution_results': execution_results,
                'performance': performance
            }
        else:
            logging.info("No trading signals generated - market conditions not favorable")
            return {}

async def main():
    """Run R&D strategy deployment"""

    deployer = RDStrategyDeployer()
    results = await deployer.run_full_rd_deployment_cycle()

    if results:
        print(f"\nR&D DEPLOYMENT SUMMARY:")
        print(f"Capital Deployed: ${results['execution_results']['total_deployed']:,.0f}")
        print(f"Active Strategies: {results['performance']['active_positions']}")
        print(f"Current P&L: ${results['performance']['total_unrealized_pl']:,.0f}")

if __name__ == "__main__":
    asyncio.run(main())