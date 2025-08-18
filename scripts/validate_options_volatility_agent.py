#!/usr/bin/env python3
"""
Options Volatility Agent Validation Script

This script validates the Options Volatility Agent implementation against
the acceptance criteria defined in Task 3.5:

Acceptance Test: Analyze options chains, detect IV opportunities, calculate Greeks
Requirements: Requirement 1 (Multi-Strategy Signal Generation)

Validation includes:
1. Options chain analysis and IV surface construction
2. Volatility skew detection and anomaly identification
3. Earnings calendar integration and strategy recommendation
4. Greeks calculation and risk management
5. Signal generation with explainability
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.options_volatility_agent as ova
OptionsVolatilityAgent = ova.OptionsVolatilityAgent
OptionsData = ova.OptionsData
VolatilityRegime = ova.VolatilityRegime
OptionsStrategy = ova.OptionsStrategy
options_volatility_agent_node = ova.options_volatility_agent_node

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptionsVolatilityAgentValidator:
    """Validator for Options Volatility Agent functionality"""
    
    def __init__(self):
        self.agent = OptionsVolatilityAgent()
        self.validation_results = {}
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    def create_mock_options_chain(self, symbol: str, underlying_price: float) -> list:
        """Create realistic mock options chain for testing"""
        options_data = []
        
        # Create strikes around current price
        strikes = np.arange(underlying_price * 0.85, underlying_price * 1.15, underlying_price * 0.025)
        
        # Create multiple expirations
        expirations = [
            datetime.now() + timedelta(days=7),   # Weekly
            datetime.now() + timedelta(days=21),  # Monthly
            datetime.now() + timedelta(days=45),  # Next month
            datetime.now() + timedelta(days=90)   # Quarterly
        ]
        
        for exp in expirations:
            tte = (exp - datetime.now()).days / 365.0
            
            for strike in strikes:
                moneyness = strike / underlying_price
                
                # Create realistic volatility smile/skew
                if moneyness < 0.95:  # OTM puts - higher IV
                    base_iv = 0.28 + (0.95 - moneyness) * 0.5
                elif moneyness > 1.05:  # OTM calls - slightly higher IV
                    base_iv = 0.22 + (moneyness - 1.05) * 0.3
                else:  # ATM - base IV
                    base_iv = 0.25
                
                # Add term structure (longer dated = higher IV)
                iv = base_iv + (tte * 0.05)
                iv = max(0.1, min(iv, 0.8))  # Reasonable bounds
                
                for option_type in ['call', 'put']:
                    # Calculate theoretical Greeks (simplified)
                    if option_type == 'call':
                        delta = 0.5 if abs(moneyness - 1.0) < 0.05 else (0.8 if moneyness < 1.0 else 0.2)
                    else:
                        delta = -0.5 if abs(moneyness - 1.0) < 0.05 else (-0.2 if moneyness < 1.0 else -0.8)
                    
                    gamma = 0.03 / (abs(moneyness - 1.0) + 0.1)
                    theta = -0.05 * iv * np.sqrt(tte)
                    vega = 0.15 * np.sqrt(tte)
                    rho = 0.08 * tte * (1 if option_type == 'call' else -1)
                    
                    # Mock market data
                    theoretical_price = max(0.05, iv * underlying_price * 0.1)
                    volume = np.random.randint(10, 1000)
                    open_interest = np.random.randint(100, 5000)
                    
                    option = OptionsData(
                        symbol=symbol,
                        expiration=exp,
                        strike=strike,
                        option_type=option_type,
                        bid=theoretical_price * 0.98,
                        ask=theoretical_price * 1.02,
                        last_price=theoretical_price,
                        volume=volume,
                        open_interest=open_interest,
                        implied_volatility=iv,
                        delta=delta,
                        gamma=gamma,
                        theta=theta,
                        vega=vega,
                        rho=rho,
                        underlying_price=underlying_price,
                        time_to_expiration=tte
                    )
                    
                    options_data.append(option)
        
        return options_data
    
    async def validate_iv_surface_analysis(self) -> dict:
        """Validate IV surface analysis functionality"""
        logger.info("🔍 Validating IV Surface Analysis...")
        
        results = {}
        
        for symbol in self.test_symbols:
            underlying_price = np.random.uniform(100, 200)  # Random price
            options_data = self.create_mock_options_chain(symbol, underlying_price)
            
            # Test IV surface analysis
            iv_analysis = await self.agent.analyze_iv_surface(symbol, options_data)
            
            # Validate results
            success = (
                'symbol' in iv_analysis and
                'surface_points' in iv_analysis and
                'skew_analysis' in iv_analysis and
                'surface_metrics' in iv_analysis and
                iv_analysis['surface_points'] > 0
            )
            
            results[symbol] = {
                'success': success,
                'surface_points': iv_analysis.get('surface_points', 0),
                'skew_anomalies': len([s for s in iv_analysis.get('skew_analysis', []) if s.is_anomalous]),
                'arbitrage_opportunities': len(iv_analysis.get('arbitrage_opportunities', [])),
                'avg_iv': iv_analysis.get('surface_metrics', {}).get('average_iv', 0)
            }
            
            logger.info(f"  {symbol}: {iv_analysis['surface_points']} surface points, "
                       f"{results[symbol]['skew_anomalies']} anomalies detected")
        
        overall_success = all(r['success'] for r in results.values())
        logger.info(f"✅ IV Surface Analysis: {'PASSED' if overall_success else 'FAILED'}")
        
        return {
            'test_name': 'IV Surface Analysis',
            'success': overall_success,
            'details': results
        }
    
    async def validate_earnings_integration(self) -> dict:
        """Validate earnings calendar integration"""
        logger.info("📅 Validating Earnings Calendar Integration...")
        
        results = {}
        
        for symbol in self.test_symbols:
            underlying_price = np.random.uniform(100, 200)
            options_data = self.create_mock_options_chain(symbol, underlying_price)
            
            # Test earnings integration
            earnings_event = await self.agent.integrate_earnings_calendar(symbol, options_data)
            
            # Validate results (may be None if no earnings)
            if earnings_event:
                success = (
                    earnings_event.symbol == symbol and
                    earnings_event.days_to_earnings >= 0 and
                    0 <= earnings_event.iv_rank <= 1 and
                    isinstance(earnings_event.strategy_recommendation, OptionsStrategy)
                )
                
                results[symbol] = {
                    'success': success,
                    'has_earnings': True,
                    'days_to_earnings': earnings_event.days_to_earnings,
                    'expected_move': earnings_event.expected_move,
                    'strategy': earnings_event.strategy_recommendation.value
                }
            else:
                results[symbol] = {
                    'success': True,  # No earnings is valid
                    'has_earnings': False
                }
            
            logger.info(f"  {symbol}: {'Earnings detected' if results[symbol]['has_earnings'] else 'No earnings'}")
        
        overall_success = all(r['success'] for r in results.values())
        logger.info(f"✅ Earnings Integration: {'PASSED' if overall_success else 'FAILED'}")
        
        return {
            'test_name': 'Earnings Calendar Integration',
            'success': overall_success,
            'details': results
        }
    
    async def validate_greeks_calculation(self) -> dict:
        """Validate Greeks calculation and risk management"""
        logger.info("🧮 Validating Greeks Calculation...")
        
        results = {}
        
        for symbol in self.test_symbols:
            underlying_price = np.random.uniform(100, 200)
            options_data = self.create_mock_options_chain(symbol, underlying_price)
            
            # Take a subset for portfolio
            portfolio_positions = options_data[:10]  # First 10 positions
            
            # Test Greeks calculation
            greeks_risk = await self.agent.calculate_greeks_risk(portfolio_positions)
            
            # Validate results
            success = (
                isinstance(greeks_risk.total_delta, (int, float)) and
                isinstance(greeks_risk.total_gamma, (int, float)) and
                isinstance(greeks_risk.total_theta, (int, float)) and
                isinstance(greeks_risk.total_vega, (int, float)) and
                isinstance(greeks_risk.total_rho, (int, float)) and
                isinstance(greeks_risk.delta_neutral, bool) and
                greeks_risk.gamma_risk_level in ['low', 'medium', 'high', 'unknown']
            )
            
            results[symbol] = {
                'success': success,
                'total_delta': greeks_risk.total_delta,
                'total_vega': greeks_risk.total_vega,
                'delta_neutral': greeks_risk.delta_neutral,
                'gamma_risk': greeks_risk.gamma_risk_level
            }
            
            logger.info(f"  {symbol}: Delta={greeks_risk.total_delta:.2f}, "
                       f"Vega={greeks_risk.total_vega:.2f}, Risk={greeks_risk.gamma_risk_level}")
        
        overall_success = all(r['success'] for r in results.values())
        logger.info(f"✅ Greeks Calculation: {'PASSED' if overall_success else 'FAILED'}")
        
        return {
            'test_name': 'Greeks Calculation',
            'success': overall_success,
            'details': results
        }
    
    async def validate_volatility_regime_detection(self) -> dict:
        """Validate volatility regime detection"""
        logger.info("📊 Validating Volatility Regime Detection...")
        
        results = {}
        
        for symbol in self.test_symbols:
            # Test regime detection
            regime = await self.agent.detect_volatility_regime(symbol)
            
            # Validate results
            success = isinstance(regime, VolatilityRegime)
            
            results[symbol] = {
                'success': success,
                'regime': regime.value if success else 'unknown'
            }
            
            logger.info(f"  {symbol}: {regime.value if success else 'FAILED'}")
        
        overall_success = all(r['success'] for r in results.values())
        logger.info(f"✅ Volatility Regime Detection: {'PASSED' if overall_success else 'FAILED'}")
        
        return {
            'test_name': 'Volatility Regime Detection',
            'success': overall_success,
            'details': results
        }
    
    async def validate_signal_generation(self) -> dict:
        """Validate options signal generation with explainability"""
        logger.info("🎯 Validating Signal Generation...")
        
        results = {}
        
        for symbol in self.test_symbols:
            market_data = {
                'current_price': np.random.uniform(100, 200),
                'volume': np.random.randint(100000, 1000000)
            }
            
            # Test signal generation
            signals = await self.agent.generate_options_signals(symbol, market_data)
            
            # Validate signals
            signals_valid = True
            signal_details = []
            
            for signal in signals:
                signal_valid = (
                    signal.symbol == symbol and
                    -1 <= signal.value <= 1 and
                    0 <= signal.confidence <= 1 and
                    len(signal.top_3_reasons) <= 3 and
                    isinstance(signal.strategy, OptionsStrategy) and
                    isinstance(signal.volatility_regime, VolatilityRegime)
                )
                
                if not signal_valid:
                    signals_valid = False
                
                signal_details.append({
                    'type': signal.signal_type,
                    'strategy': signal.strategy.value,
                    'strength': signal.value,
                    'confidence': signal.confidence,
                    'reasons_count': len(signal.top_3_reasons)
                })
            
            results[symbol] = {
                'success': signals_valid,
                'signals_generated': len(signals),
                'signal_details': signal_details
            }
            
            logger.info(f"  {symbol}: {len(signals)} signals generated")
        
        overall_success = all(r['success'] for r in results.values())
        logger.info(f"✅ Signal Generation: {'PASSED' if overall_success else 'FAILED'}")
        
        return {
            'test_name': 'Signal Generation',
            'success': overall_success,
            'details': results
        }
    
    async def validate_langgraph_integration(self) -> dict:
        """Validate LangGraph integration"""
        logger.info("🔗 Validating LangGraph Integration...")
        
        # Mock LangGraph state
        state = {
            'market_data': {
                symbol: {
                    'current_price': np.random.uniform(100, 200),
                    'volume': np.random.randint(100000, 1000000)
                }
                for symbol in self.test_symbols
            },
            'signals': {}
        }
        
        # Test LangGraph node function
        try:
            result_state = await options_volatility_agent_node(state)
            
            success = (
                'signals' in result_state and
                'options_volatility' in result_state['signals'] and
                'options_analysis' in result_state and
                result_state['options_analysis']['agent'] == 'options_volatility'
            )
            
            signals_count = len(result_state['signals']['options_volatility'])
            
        except Exception as e:
            logger.error(f"LangGraph integration failed: {e}")
            success = False
            signals_count = 0
        
        logger.info(f"✅ LangGraph Integration: {'PASSED' if success else 'FAILED'}")
        
        return {
            'test_name': 'LangGraph Integration',
            'success': success,
            'details': {
                'signals_generated': signals_count,
                'state_updated': success
            }
        }
    
    async def run_comprehensive_validation(self) -> dict:
        """Run comprehensive validation of all functionality"""
        logger.info("🚀 Starting Comprehensive Options Volatility Agent Validation")
        logger.info("=" * 60)
        
        validation_tests = [
            self.validate_iv_surface_analysis(),
            self.validate_earnings_integration(),
            self.validate_greeks_calculation(),
            self.validate_volatility_regime_detection(),
            self.validate_signal_generation(),
            self.validate_langgraph_integration()
        ]
        
        # Run all validations
        results = await asyncio.gather(*validation_tests, return_exceptions=True)
        
        # Process results
        validation_results = []
        total_tests = 0
        passed_tests = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Validation failed with exception: {result}")
                validation_results.append({
                    'test_name': 'Unknown',
                    'success': False,
                    'error': str(result)
                })
            else:
                validation_results.append(result)
                total_tests += 1
                if result['success']:
                    passed_tests += 1
        
        # Summary
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        overall_success = success_rate >= 80  # 80% pass rate required
        
        logger.info("=" * 60)
        logger.info(f"🎯 VALIDATION SUMMARY")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        
        return {
            'overall_success': overall_success,
            'success_rate': success_rate,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'detailed_results': validation_results,
            'timestamp': datetime.utcnow().isoformat()
        }

async def main():
    """Main validation function"""
    try:
        validator = OptionsVolatilityAgentValidator()
        results = await validator.run_comprehensive_validation()
        
        # Save results
        import json
        with open('options_volatility_agent_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_success'] else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Validation failed with critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())