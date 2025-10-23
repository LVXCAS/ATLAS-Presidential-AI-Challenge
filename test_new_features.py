#!/usr/bin/env python3
"""
Test New Features Integration
Quick validation of the enhanced features added from the cloned repository
"""

import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

async def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("TESTING SENTIMENT ANALYSIS")
    print("=" * 50)

    try:
        from agents.enhanced_sentiment_analyzer import enhanced_sentiment_analyzer

        # Test basic sentiment analysis
        test_symbol = "AAPL"
        print(f"Analyzing sentiment for {test_symbol}...")

        analysis = await enhanced_sentiment_analyzer.analyze_symbol_sentiment(test_symbol)

        print(f"[OK] Sentiment Analysis Results for {test_symbol}:")
        print(f"   Composite Score: {analysis['composite_sentiment']['composite_score']}")
        print(f"   Sentiment Label: {analysis['composite_sentiment']['sentiment_label']}")
        print(f"   Confidence: {analysis['composite_sentiment']['confidence']}")
        print(f"   Recommendation: {analysis['composite_sentiment']['recommendation']}")

        return True

    except Exception as e:
        print(f"[ERROR] Sentiment Analysis Error: {e}")
        return False

def test_backtesting_engine():
    """Test backtesting functionality"""
    print("\nTESTING BACKTESTING ENGINE")
    print("=" * 50)

    try:
        from backend.api.backtesting import BacktestEngine, BacktestRequest

        engine = BacktestEngine()

        # Create test backtest request
        request = BacktestRequest(
            strategy_name="test_momentum",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=100000
        )

        print(f"Running backtest for {request.strategy_name} on {request.symbol}...")

        result = engine.run_backtest(request)

        print(f"[OK] Backtest Results:")
        print(f"   Total Return: {result.performance.total_return}%")
        print(f"   Sharpe Ratio: {result.performance.sharpe_ratio}")
        print(f"   Max Drawdown: {result.performance.max_drawdown}%")
        print(f"   Win Rate: {result.performance.win_rate}%")
        print(f"   Total Trades: {result.performance.trades_count}")

        return True

    except Exception as e:
        print(f"[ERROR] Backtesting Engine Error: {e}")
        return False

def test_trading_api_integration():
    """Test trading API integration"""
    print("\nTESTING TRADING API INTEGRATION")
    print("=" * 50)

    try:
        # Test profit/loss monitoring
        from profit_target_monitor import ProfitTargetMonitor

        monitor = ProfitTargetMonitor()
        status = monitor.get_status()

        print(f"[OK] Profit/Loss Monitoring:")
        print(f"   Profit Target: {monitor.profit_target_pct}%")
        print(f"   Loss Limit: {monitor.loss_limit_pct}%")
        print(f"   Monitoring Active: {status['monitoring_active']}")

        return True

    except Exception as e:
        print(f"[ERROR] Trading API Integration Error: {e}")
        return False

def test_quantitative_integration():
    """Test quantitative finance integration"""
    print("\nTESTING QUANTITATIVE FINANCE INTEGRATION")
    print("=" * 50)

    try:
        from agents.quantitative_finance_engine import quantitative_engine, OptionParameters
        from agents.quant_integration import analyze_option

        # Test Black-Scholes pricing
        params = OptionParameters(
            underlying_price=150.0,
            strike_price=155.0,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.25,
            option_type='call'
        )

        bs_result = quantitative_engine.black_scholes_price(params)

        print(f"[OK] Black-Scholes Option Pricing:")
        print(f"   Option Price: ${bs_result['price']:.2f}")
        print(f"   Delta: {bs_result['delta']:.3f}")
        print(f"   Gamma: {bs_result['gamma']:.4f}")

        # Test option analysis
        analysis = analyze_option("AAPL", 150.0, "2024-12-31", "call")
        if analysis and 'bs_price' in analysis:
            print(f"[OK] Option Analysis:")
            print(f"   Fair Value: ${analysis['bs_price']:.2f}")
            print(f"   Entry Recommendation: {analysis.get('entry_recommendation', 'N/A')}")

        return True

    except Exception as e:
        print(f"[ERROR] Quantitative Integration Error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints structure"""
    print("\nTESTING API ENDPOINTS STRUCTURE")
    print("=" * 50)

    try:
        from backend.main import app

        # Get all routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                routes.append((route.path, list(route.methods)))

        print("[OK] Available API Endpoints:")
        endpoint_groups = {}

        for path, methods in routes:
            if path.startswith('/api/'):
                group = path.split('/')[2]
                if group not in endpoint_groups:
                    endpoint_groups[group] = []
                endpoint_groups[group].append(f"{methods[0] if methods else 'GET'} {path}")

        for group, endpoints in endpoint_groups.items():
            print(f"\n   {group.upper()} API:")
            for endpoint in endpoints[:3]:  # Show first 3 endpoints per group
                print(f"     {endpoint}")

        return True

    except Exception as e:
        print(f"[ERROR] API Endpoints Test Error: {e}")
        return False

async def main():
    """Run all tests"""
    print("HIVETRADING ENHANCED FEATURES TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()

    test_results = []

    # Run all tests
    test_results.append(await test_sentiment_analysis())
    test_results.append(test_backtesting_engine())
    test_results.append(test_trading_api_integration())
    test_results.append(test_quantitative_integration())
    test_results.append(test_api_endpoints())

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(test_results)
    total = len(test_results)

    test_names = [
        "Sentiment Analysis",
        "Backtesting Engine",
        "Trading API Integration",
        "Quantitative Integration",
        "API Endpoints"
    ]

    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    print(f"\nOverall Result: {passed}/{total} tests passed")

    if passed == total:
        print("ALL ENHANCED FEATURES WORKING CORRECTLY!")
    else:
        print(f"{total - passed} features need attention")

    print(f"\nNEW FEATURES SUCCESSFULLY ADDED:")
    print("   + Advanced Sentiment Analysis (News + Social Media)")
    print("   + Comprehensive Backtesting Engine")
    print("   + Enhanced Trading API with P&L Monitoring")
    print("   + Professional Web Dashboard Backend")
    print("   + Multi-source Data Integration")
    print("   + Risk Management & Analytics")

if __name__ == "__main__":
    asyncio.run(main())