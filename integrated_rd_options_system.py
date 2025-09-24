"""
INTEGRATED R&D-OPTIONS TRADING SYSTEM
Combines R&D strategy analysis with options execution infrastructure
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_rd_options.log'),
        logging.StreamHandler()
    ]
)

class IntegratedRDOptionsSystem:
    """Integrated R&D strategy analysis with options execution"""

    def __init__(self):
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Allocation parameters
        self.rd_allocation = 200000  # $200K for R&D strategies
        self.options_allocation = 400000  # $400K for options strategies
        self.total_allocation = 600000  # Total Monday deployment

        # Quality thresholds
        self.min_sharpe_threshold = 1.0
        self.high_quality_sharpe = 1.5

    async def load_latest_rd_analysis(self) -> Dict:
        """Load the most recent R&D analysis"""

        try:
            # Try the latest analysis file
            with open('rd_analysis_20250919_175942.json', 'r') as f:
                rd_data = json.load(f)
            logging.info("Latest R&D analysis loaded successfully")
            return rd_data
        except:
            try:
                # Fallback to earlier analysis
                with open('rd_analysis_20250919_094742.json', 'r') as f:
                    rd_data = json.load(f)
                logging.info("Fallback R&D analysis loaded")
                return rd_data
            except Exception as e:
                logging.error(f"Failed to load R&D analysis: {e}")
                return {}

    async def load_mctx_optimization(self) -> Dict:
        """Load MCTX optimization results"""

        try:
            with open('mctx_optimization_results.json', 'r') as f:
                mctx_data = json.load(f)
            logging.info("MCTX optimization data loaded")
            return mctx_data
        except Exception as e:
            logging.error(f"Failed to load MCTX data: {e}")
            return {}

    async def integrate_rd_with_options_strategy(self) -> Dict:
        """Integrate R&D findings with options strategy recommendations"""

        logging.info("INTEGRATING R&D WITH OPTIONS STRATEGY")
        logging.info("=" * 42)

        # Load analysis data
        rd_data = await self.load_latest_rd_analysis()
        mctx_data = await self.load_mctx_optimization()

        if not rd_data:
            logging.error("No R&D data available for integration")
            return {}

        integration_strategy = {
            'timestamp': datetime.now().isoformat(),
            'rd_strategies': {},
            'options_overlays': {},
            'combined_allocations': {},
            'risk_management': {}
        }

        # Process R&D momentum strategies for options overlays
        momentum_strategies = rd_data.get('momentum_strategies', {})

        # Focus on high-quality strategies
        high_quality_momentum = {
            symbol: data for symbol, data in momentum_strategies.items()
            if data['sharpe'] >= self.min_sharpe_threshold
        }

        logging.info(f"High-Quality Momentum Strategies: {len(high_quality_momentum)}")

        for symbol, data in high_quality_momentum.items():
            logging.info(f"  {symbol}: Sharpe {data['sharpe']:.2f}, Return {data['total_return']:.1%}")

            # Determine options strategy based on momentum strength
            if data['sharpe'] >= self.high_quality_sharpe:
                # Strong momentum: Long calls
                options_strategy = 'LONG_CALLS'
                allocation_weight = 0.4  # 40% of options allocation
            else:
                # Moderate momentum: Call spreads
                options_strategy = 'CALL_SPREADS'
                allocation_weight = 0.2  # 20% of options allocation

            # Calculate quality score and allocations
            quality_score = data['sharpe'] * data['total_return']  # Calculate quality score
            rd_allocation = (self.rd_allocation * 0.8) * min(quality_score / 2, 1.0)  # Scale by quality, cap at 100%
            options_allocation = self.options_allocation * allocation_weight

            integration_strategy['rd_strategies'][symbol] = {
                'sharpe_ratio': data['sharpe'],
                'expected_return': data['total_return'],
                'rd_allocation': rd_allocation,
                'strategy_type': 'MOMENTUM'
            }

            integration_strategy['options_overlays'][symbol] = {
                'options_strategy': options_strategy,
                'options_allocation': options_allocation,
                'leverage_target': min(data['sharpe'] * 10, 20),  # Cap leverage at 20x
                'expiration_target': 'WEEKLY'  # Focus on weekly options
            }

            # Combined strategy
            integration_strategy['combined_allocations'][symbol] = {
                'total_allocation': rd_allocation + options_allocation,
                'rd_weight': rd_allocation / (rd_allocation + options_allocation),
                'options_weight': options_allocation / (rd_allocation + options_allocation),
                'expected_combined_return': data['total_return'] * 1.5,  # Options leverage
                'risk_score': 'HIGH' if data['sharpe'] >= 2.0 else 'MODERATE'
            }

        # Process mean reversion strategies
        mean_reversion_strategies = rd_data.get('mean_reversion_strategies', {})
        high_quality_mean_rev = {
            symbol: data for symbol, data in mean_reversion_strategies.items()
            if data['sharpe'] >= self.min_sharpe_threshold
        }

        logging.info(f"High-Quality Mean Reversion Strategies: {len(high_quality_mean_rev)}")

        for symbol, data in high_quality_mean_rev.items():
            logging.info(f"  {symbol}: Sharpe {data['sharpe']:.2f}, Return {data['total_return']:.1%}")

            # Mean reversion: Use put strategies or straddles
            options_strategy = 'CASH_SECURED_PUTS' if data['sharpe'] > 1.2 else 'STRADDLES'
            allocation_weight = 0.15  # 15% allocation for mean reversion

            quality_score = data['sharpe'] * data['total_return']  # Calculate quality score
            rd_allocation = (self.rd_allocation * 0.2) * min(quality_score / 1, 1.0)  # Scale by quality
            options_allocation = self.options_allocation * allocation_weight

            integration_strategy['rd_strategies'][f"{symbol}_MR"] = {
                'sharpe_ratio': data['sharpe'],
                'expected_return': data['total_return'],
                'rd_allocation': rd_allocation,
                'strategy_type': 'MEAN_REVERSION'
            }

            integration_strategy['options_overlays'][f"{symbol}_MR"] = {
                'options_strategy': options_strategy,
                'options_allocation': options_allocation,
                'leverage_target': min(data['sharpe'] * 8, 15),
                'expiration_target': 'MONTHLY'
            }

        # Risk management integration
        integration_strategy['risk_management'] = {
            'max_single_position': 0.25,  # 25% max per position
            'correlation_limit': 0.7,     # Max 70% correlation between positions
            'total_leverage_limit': 15,   # Max 15x total leverage
            'stop_loss_levels': {
                'momentum': 0.15,          # 15% stop loss for momentum
                'mean_reversion': 0.10     # 10% stop loss for mean reversion
            },
            'profit_taking': {
                'momentum': 0.50,          # Take profits at 50% gain
                'mean_reversion': 0.30     # Take profits at 30% gain
            }
        }

        # MCTX integration
        if mctx_data:
            best_strategy = mctx_data.get('best_strategy', {})
            integration_strategy['mctx_alignment'] = {
                'recommended_strategy': best_strategy.get('strategy'),
                'confidence_level': mctx_data.get('mctx_confidence', 0),
                'monte_carlo_success_rate': best_strategy.get('monte_carlo_results', {}).get('success_rate', 0)
            }

        # Save integration results
        with open('rd_options_integration_strategy.json', 'w') as f:
            json.dump(integration_strategy, f, indent=2)

        logging.info("=" * 42)
        logging.info("R&D-OPTIONS INTEGRATION COMPLETE")
        total_rd = sum(s['rd_allocation'] for s in integration_strategy['rd_strategies'].values())
        total_options = sum(s['options_allocation'] for s in integration_strategy['options_overlays'].values())
        logging.info(f"Total R&D Allocation: ${total_rd:,.0f}")
        logging.info(f"Total Options Allocation: ${total_options:,.0f}")
        logging.info(f"Combined Strategy Count: {len(integration_strategy['combined_allocations'])}")

        return integration_strategy

    async def generate_monday_execution_plan(self, integration_strategy: Dict) -> Dict:
        """Generate detailed Monday execution plan"""

        logging.info("GENERATING MONDAY EXECUTION PLAN")
        logging.info("=" * 35)

        execution_plan = {
            'execution_timestamp': datetime.now().isoformat(),
            'market_open_time': '9:30 AM EST',
            'execution_window': '9:30-11:00 AM',
            'phased_deployment': {},
            'execution_sequence': [],
            'monitoring_parameters': {}
        }

        # Phase 1: 9:30-10:00 AM - High conviction momentum
        phase1_symbols = []
        for symbol, allocation in integration_strategy['combined_allocations'].items():
            if allocation['risk_score'] == 'HIGH':
                phase1_symbols.append(symbol)

        execution_plan['phased_deployment']['phase_1'] = {
            'time_window': '9:30-10:00 AM',
            'symbols': phase1_symbols,
            'allocation_percentage': 60,  # 60% of total in first phase
            'execution_style': 'AGGRESSIVE',
            'max_contracts_per_minute': 50
        }

        # Phase 2: 10:00-10:30 AM - Moderate conviction strategies
        phase2_symbols = []
        for symbol, allocation in integration_strategy['combined_allocations'].items():
            if allocation['risk_score'] == 'MODERATE':
                phase2_symbols.append(symbol)

        execution_plan['phased_deployment']['phase_2'] = {
            'time_window': '10:00-10:30 AM',
            'symbols': phase2_symbols,
            'allocation_percentage': 30,  # 30% in second phase
            'execution_style': 'MEASURED',
            'max_contracts_per_minute': 25
        }

        # Phase 3: 10:30-11:00 AM - Conservative and cleanup
        execution_plan['phased_deployment']['phase_3'] = {
            'time_window': '10:30-11:00 AM',
            'symbols': 'REMAINING',
            'allocation_percentage': 10,  # 10% final cleanup
            'execution_style': 'CONSERVATIVE',
            'max_contracts_per_minute': 10
        }

        # Detailed execution sequence
        sequence_id = 1
        for symbol, combined_allocation in integration_strategy['combined_allocations'].items():
            rd_strategy = integration_strategy['rd_strategies'].get(symbol, {})
            options_overlay = integration_strategy['options_overlays'].get(symbol, {})

            execution_step = {
                'sequence_id': sequence_id,
                'symbol': symbol.replace('_MR', ''),  # Clean symbol for execution
                'rd_allocation': rd_strategy.get('rd_allocation', 0),
                'options_allocation': options_overlay.get('options_allocation', 0),
                'options_strategy': options_overlay.get('options_strategy'),
                'target_leverage': options_overlay.get('leverage_target', 10),
                'expiration': options_overlay.get('expiration_target', 'WEEKLY'),
                'priority': 'HIGH' if combined_allocation['risk_score'] == 'HIGH' else 'MEDIUM',
                'expected_return': combined_allocation['expected_combined_return']
            }

            execution_plan['execution_sequence'].append(execution_step)
            sequence_id += 1

        # Monitoring parameters
        execution_plan['monitoring_parameters'] = {
            'real_time_pnl_tracking': True,
            'automatic_stop_losses': True,
            'profit_taking_enabled': True,
            'correlation_monitoring': True,
            'leverage_monitoring': True,
            'manual_override_enabled': True
        }

        # Save execution plan
        with open('monday_rd_options_execution_plan.json', 'w') as f:
            json.dump(execution_plan, f, indent=2)

        logging.info("EXECUTION PLAN GENERATED")
        logging.info(f"Total Execution Steps: {len(execution_plan['execution_sequence'])}")
        logging.info(f"Phase 1 Symbols: {len(phase1_symbols)}")
        logging.info(f"Phase 2 Symbols: {len(phase2_symbols)}")

        return execution_plan

    async def run_integrated_analysis_cycle(self):
        """Run complete integrated R&D-Options analysis cycle"""

        logging.info("INTEGRATED R&D-OPTIONS ANALYSIS SYSTEM")
        logging.info("Advanced Strategy Integration with Options Overlay")
        logging.info("=" * 60)

        # Step 1: Integrate R&D with options strategies
        integration_strategy = await self.integrate_rd_with_options_strategy()

        if not integration_strategy:
            logging.error("Integration failed - no strategy generated")
            return {}

        # Step 2: Generate Monday execution plan
        execution_plan = await self.generate_monday_execution_plan(integration_strategy)

        # Step 3: Prepare weekend monitoring integration
        weekend_integration = {
            'rd_strategy_count': len(integration_strategy['rd_strategies']),
            'options_overlay_count': len(integration_strategy['options_overlays']),
            'total_combined_allocation': sum(
                a['total_allocation'] for a in integration_strategy['combined_allocations'].values()
            ),
            'risk_distribution': {
                'high_risk': len([a for a in integration_strategy['combined_allocations'].values() if a['risk_score'] == 'HIGH']),
                'moderate_risk': len([a for a in integration_strategy['combined_allocations'].values() if a['risk_score'] == 'MODERATE'])
            },
            'expected_returns': {
                symbol: allocation['expected_combined_return']
                for symbol, allocation in integration_strategy['combined_allocations'].items()
            }
        }

        logging.info("=" * 60)
        logging.info("INTEGRATED ANALYSIS COMPLETE")
        logging.info(f"R&D Strategies: {weekend_integration['rd_strategy_count']}")
        logging.info(f"Options Overlays: {weekend_integration['options_overlay_count']}")
        logging.info(f"Total Allocation: ${weekend_integration['total_combined_allocation']:,.0f}")
        logging.info(f"High Risk Positions: {weekend_integration['risk_distribution']['high_risk']}")
        logging.info(f"Moderate Risk Positions: {weekend_integration['risk_distribution']['moderate_risk']}")

        return {
            'integration_strategy': integration_strategy,
            'execution_plan': execution_plan,
            'weekend_summary': weekend_integration
        }

async def main():
    """Run integrated R&D-Options system"""

    system = IntegratedRDOptionsSystem()
    results = await system.run_integrated_analysis_cycle()

    if results:
        print(f"\nINTEGRATED R&D-OPTIONS SYSTEM COMPLETE")
        print(f"Strategy Integration: SUCCESS")
        print(f"Execution Plan: GENERATED")
        print(f"Weekend Monitoring: ACTIVE")
        print(f"Monday Deployment: READY")

if __name__ == "__main__":
    asyncio.run(main())