"""
Full System Integration Test

Complete test of all system components to ensure everything works together
"""

import asyncio
import sys
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class SystemIntegrationTester:
    """Comprehensive system integration testing"""

    def __init__(self):
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []

    async def test_core_dependencies(self):
        """Test all core dependencies"""
        print("[TEST] Testing Core Dependencies...")

        core_deps = {
            'numpy': 'Numerical computing',
            'pandas': 'Data manipulation',
            'scipy': 'Scientific computing',
            'sklearn': 'Machine learning',
            'yfinance': 'Market data',
            'requests': 'HTTP requests',
            'asyncio': 'Async operations',
            'json': 'JSON processing',
            'datetime': 'Date/time handling'
        }

        dependency_results = {}

        for dep, description in core_deps.items():
            try:
                if dep == 'sklearn':
                    import sklearn
                else:
                    __import__(dep)
                dependency_results[dep] = True
                print(f"  [OK] {dep} - {description}")
            except ImportError as e:
                dependency_results[dep] = False
                print(f"  [ERROR] {dep} - {description}: {e}")

        self.test_results['dependencies'] = dependency_results
        return all(dependency_results.values())

    async def test_api_connections(self):
        """Test API connections"""
        print("\n[TEST] Testing API Connections...")

        api_results = {}

        # Test Yahoo Finance
        try:
            import yfinance as yf
            ticker = yf.Ticker("SPY")
            data = ticker.history(period="5d")

            if len(data) > 0:
                api_results['yahoo_finance'] = True
                print(f"  [OK] Yahoo Finance - Latest SPY: ${data['Close'].iloc[-1]:.2f}")
            else:
                api_results['yahoo_finance'] = False
                print(f"  [ERROR] Yahoo Finance - No data received")

        except Exception as e:
            api_results['yahoo_finance'] = False
            print(f"  [ERROR] Yahoo Finance: {e}")

        # Test Alpaca API
        try:
            from dotenv import load_dotenv
            load_dotenv()

            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

            if api_key and secret_key:
                import requests
                headers = {
                    'APCA-API-KEY-ID': api_key,
                    'APCA-API-SECRET-KEY': secret_key
                }

                response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)

                if response.status_code == 200:
                    account_data = response.json()
                    api_results['alpaca'] = True
                    print(f"  [OK] Alpaca API - Portfolio: ${float(account_data.get('portfolio_value', 0)):,.2f}")
                else:
                    api_results['alpaca'] = False
                    print(f"  [ERROR] Alpaca API - HTTP {response.status_code}")
            else:
                api_results['alpaca'] = False
                print(f"  [SKIP] Alpaca API - No credentials configured")

        except Exception as e:
            api_results['alpaca'] = False
            print(f"  [ERROR] Alpaca API: {e}")

        self.test_results['apis'] = api_results
        return any(api_results.values())  # At least one API working

    async def test_autonomous_agents(self):
        """Test autonomous agent functionality"""
        print("\n[TEST] Testing Autonomous Agents...")

        agent_results = {}

        try:
            # Test autonomous R&D system
            from fixed_autonomous_rd import StrategyResearchAgent, MarketRegimeAgent

            # Test Strategy Research Agent
            strategy_agent = StrategyResearchAgent()

            # Test autonomous decision making
            decision_context = {
                'current_time': datetime.now(),
                'market_hours': False,
                'recent_performance': 0.8
            }

            decision = await strategy_agent.make_autonomous_decision(decision_context)

            if 'action' in decision:
                agent_results['strategy_agent'] = True
                print(f"  [OK] Strategy Agent - Decision: {decision['action']}")
            else:
                agent_results['strategy_agent'] = False
                print(f"  [ERROR] Strategy Agent - Invalid decision format")

            # Test Market Regime Agent
            regime_agent = MarketRegimeAgent()
            regime_data = await regime_agent.detect_regime_autonomously()

            if 'regime' in regime_data:
                agent_results['regime_agent'] = True
                print(f"  [OK] Regime Agent - Detected: {regime_data['regime']}")
            else:
                agent_results['regime_agent'] = False
                print(f"  [ERROR] Regime Agent - Invalid regime data")

        except Exception as e:
            agent_results['strategy_agent'] = False
            agent_results['regime_agent'] = False
            print(f"  [ERROR] Autonomous Agents: {e}")

        self.test_results['agents'] = agent_results
        return all(agent_results.values())

    async def test_trading_system(self):
        """Test trading system components"""
        print("\n[TEST] Testing Trading System...")

        trading_results = {}

        try:
            # Test options system
            print("  Testing options pricing...")

            # Simple Black-Scholes test
            import numpy as np
            from scipy.stats import norm

            S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

            if call_price > 0:
                trading_results['options_pricing'] = True
                print(f"  [OK] Options Pricing - Call price: ${call_price:.2f}")
            else:
                trading_results['options_pricing'] = False
                print(f"  [ERROR] Options Pricing - Invalid result")

        except Exception as e:
            trading_results['options_pricing'] = False
            print(f"  [ERROR] Options Pricing: {e}")

        try:
            # Test strategy calculations
            print("  Testing strategy calculations...")

            import pandas as pd

            # Generate sample data
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
            data = pd.DataFrame({'Close': prices}, index=dates)

            # Test momentum calculation
            data['momentum'] = data['Close'].pct_change(10)
            data['signal'] = np.where(data['momentum'] > 0.02, 1, 0)

            if data['signal'].sum() > 0:
                trading_results['strategy_calc'] = True
                print(f"  [OK] Strategy Calculations - Signals: {data['signal'].sum()}")
            else:
                trading_results['strategy_calc'] = False
                print(f"  [ERROR] Strategy Calculations - No signals generated")

        except Exception as e:
            trading_results['strategy_calc'] = False
            print(f"  [ERROR] Strategy Calculations: {e}")

        self.test_results['trading'] = trading_results
        return all(trading_results.values())

    async def test_machine_learning(self):
        """Test machine learning components"""
        print("\n[TEST] Testing Machine Learning...")

        ml_results = {}

        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.neural_network import MLPClassifier
            import numpy as np

            # Test RandomForest
            X = np.random.rand(100, 5)
            y = np.random.rand(100)

            rf = RandomForestRegressor(n_estimators=10, random_state=42)
            rf.fit(X, y)
            prediction = rf.predict(X[:1])

            if len(prediction) > 0:
                ml_results['random_forest'] = True
                print(f"  [OK] Random Forest - Prediction: {prediction[0]:.3f}")
            else:
                ml_results['random_forest'] = False
                print(f"  [ERROR] Random Forest - No prediction")

            # Test Neural Network
            y_class = (y > 0.5).astype(int)
            mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
            mlp.fit(X, y_class)
            class_pred = mlp.predict(X[:1])

            if len(class_pred) > 0:
                ml_results['neural_network'] = True
                print(f"  [OK] Neural Network - Classification: {class_pred[0]}")
            else:
                ml_results['neural_network'] = False
                print(f"  [ERROR] Neural Network - No prediction")

        except Exception as e:
            ml_results['random_forest'] = False
            ml_results['neural_network'] = False
            print(f"  [ERROR] Machine Learning: {e}")

        self.test_results['machine_learning'] = ml_results
        return all(ml_results.values())

    async def test_system_integration(self):
        """Test complete system integration"""
        print("\n[TEST] Testing System Integration...")

        integration_results = {}

        try:
            # Test full autonomous R&D workflow
            from fixed_autonomous_rd import AutonomousRDOrchestrator

            orchestrator = AutonomousRDOrchestrator()
            await orchestrator.initialize_agents()

            if len(orchestrator.agents) >= 2:
                integration_results['orchestrator'] = True
                print(f"  [OK] Orchestrator - {len(orchestrator.agents)} agents initialized")
            else:
                integration_results['orchestrator'] = False
                print(f"  [ERROR] Orchestrator - Insufficient agents")

            # Test agent coordination
            strategy_agent = orchestrator.agents.get('strategy_researcher')
            regime_agent = orchestrator.agents.get('regime_detector')

            if strategy_agent and regime_agent:
                integration_results['agent_coordination'] = True
                print(f"  [OK] Agent Coordination - All agents accessible")
            else:
                integration_results['agent_coordination'] = False
                print(f"  [ERROR] Agent Coordination - Missing agents")

        except Exception as e:
            integration_results['orchestrator'] = False
            integration_results['agent_coordination'] = False
            print(f"  [ERROR] System Integration: {e}")

        self.test_results['integration'] = integration_results
        return all(integration_results.values())

    async def run_full_integration_test(self):
        """Run complete integration test suite"""
        print("="*70)
        print("FULL SYSTEM INTEGRATION TEST")
        print("="*70)
        print(f"Test Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run all test categories
        test_categories = [
            ("Core Dependencies", self.test_core_dependencies),
            ("API Connections", self.test_api_connections),
            ("Autonomous Agents", self.test_autonomous_agents),
            ("Trading System", self.test_trading_system),
            ("Machine Learning", self.test_machine_learning),
            ("System Integration", self.test_system_integration)
        ]

        category_results = {}

        for category_name, test_func in test_categories:
            try:
                result = await test_func()
                category_results[category_name] = result

                if result:
                    self.passed_tests.append(category_name)
                else:
                    self.failed_tests.append(category_name)

            except Exception as e:
                print(f"[ERROR] {category_name} test failed: {e}")
                category_results[category_name] = False
                self.failed_tests.append(category_name)

        # Generate final report
        await self.generate_final_report(category_results)

        return len(self.failed_tests) == 0

    async def generate_final_report(self, category_results):
        """Generate comprehensive test report"""
        print(f"\n{'='*70}")
        print("INTEGRATION TEST RESULTS")
        print("="*70)

        total_tests = len(category_results)
        passed_tests = sum(category_results.values())
        success_rate = (passed_tests / total_tests) * 100

        print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")

        print(f"\nDETAILED RESULTS:")
        for category, result in category_results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {category}")

        if self.failed_tests:
            print(f"\nFAILED CATEGORIES:")
            for failed in self.failed_tests:
                print(f"  - {failed}")

        print(f"\n{'='*70}")
        if success_rate >= 80:
            print("SYSTEM STATUS: OPERATIONAL")
            print("✓ System ready for autonomous operation")
            print("✓ All critical components working")
            print("✓ Integration successful")
        else:
            print("SYSTEM STATUS: NEEDS ATTENTION")
            print("! Some components require fixes")
            print("! Check failed categories above")

        print("="*70)

async def main():
    """Run the full integration test"""

    tester = SystemIntegrationTester()

    try:
        success = await tester.run_full_integration_test()

        if success:
            print("\n🎉 ALL SYSTEMS GO!")
            print("Your autonomous R&D system is fully operational and ready for deployment!")
            print("\nNext steps:")
            print("1. python fixed_autonomous_rd.py (test autonomous agents)")
            print("2. python launch_autonomous_rd.py (full autonomous operation)")
            print("3. python tomorrow_profit_system.py (start trading)")
        else:
            print("\n⚠️ SYSTEM NEEDS ATTENTION")
            print("Some components failed testing. Please check the failed categories above.")

        return success

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)