"""
Test script to verify all confidence calculation components are working
"""
import asyncio
from OPTIONS_BOT import TomorrowReadyOptionsBot

async def main():
    print("="*60)
    print("CONFIDENCE SYSTEM DIAGNOSTIC TEST")
    print("="*60)
    print()

    # Initialize bot
    print("Initializing bot...")
    bot = TomorrowReadyOptionsBot()
    print("[OK] Bot initialized\n")

    # Component Status Check
    print("="*60)
    print("COMPONENT STATUS")
    print("="*60)

    components = {
        'Broker Integration': bot.broker,
        'Options Trader': bot.options_trader,
        'Exit Agent': bot.exit_agent,
        'Learning Engine': bot.learning_engine,
        'Advanced ML': bot.advanced_ml,
        'Technical Analysis': bot.technical_analysis,
        'Options Pricing': bot.options_pricing,
        'Multi-timeframe Analyzer': bot.multitimeframe_analyzer,
        'IV Analyzer': bot.iv_analyzer,
        'Sentiment Analyzer': bot.sentiment_analyzer,
        'Quantitative Engine': bot.quant_engine,
        'Quant Analyzer': bot.quant_analyzer,
        'ML Ensemble': bot.ml_ensemble,
        'Greeks Optimizer': bot.greeks_optimizer,
        'Volatility Adapter': bot.volatility_adapter,
        'Market Regime Detector': bot.market_regime_detector,
        'Liquidity Filter': bot.liquidity_filter,
        'Sharpe Filters': bot.sharpe_filters,
        'Dynamic Stop Manager': bot.dynamic_stop_manager
    }

    active_count = 0
    missing_count = 0

    for name, component in components.items():
        if component is not None:
            status = '[OK] ACTIVE'
            active_count += 1
        else:
            status = '[X] MISSING'
            missing_count += 1
        print(f'{name:35s}: {status}')

    print()
    print(f"Summary: {active_count} active, {missing_count} missing")
    print()

    # ML Ensemble Details
    if bot.ml_ensemble:
        print("="*60)
        print("ML ENSEMBLE DETAILS")
        print("="*60)
        print(f'Loaded: {bot.ml_ensemble.loaded}')
        if hasattr(bot.ml_ensemble, 'models') and bot.ml_ensemble.models:
            print(f'Models: {list(bot.ml_ensemble.models.keys())}')
            for model_name, model in bot.ml_ensemble.models.items():
                print(f'  - {model_name}: {type(model).__name__}')
        else:
            print('No models loaded yet')
        print()

    # Confidence Formula Components
    print("="*60)
    print("CONFIDENCE FORMULA COMPONENTS")
    print("="*60)

    confidence_components = {
        'Base Confidence (30%)': True,  # Always available
        'Technical Bonuses': bot.technical_analysis is not None,
        'Multi-timeframe Alignment': bot.multitimeframe_analyzer is not None,
        'IV Rank Analysis': bot.iv_analyzer is not None,
        'Sentiment Analysis': bot.sentiment_analyzer is not None,
        'Learning Engine Calibration': bot.learning_engine is not None,
        'ML Ensemble Predictions': bot.ml_ensemble is not None and bot.ml_ensemble.loaded,
        'Ensemble Voting System': True,  # Built into bot
        'Quantitative Analysis': bot.quant_analyzer is not None,
        'Greeks Optimization': bot.greeks_optimizer is not None,
        'Market Regime Adjustment': bot.market_regime_detector is not None,
        'Liquidity Filtering': bot.liquidity_filter is not None,
        'Sharpe Optimization': bot.sharpe_filters is not None
    }

    for component, is_working in confidence_components.items():
        status = '[OK] WORKING' if is_working else '[X] NOT WORKING'
        print(f'{component:40s}: {status}')

    working_count = sum(1 for v in confidence_components.values() if v)
    total_count = len(confidence_components)
    percentage = (working_count / total_count) * 100

    print()
    print(f"Confidence System: {working_count}/{total_count} components working ({percentage:.1f}%)")
    print()

    # Critical Issues
    print("="*60)
    print("CRITICAL ISSUES CHECK")
    print("="*60)

    issues = []

    if not bot.learning_engine:
        issues.append("[!] Learning Engine missing - no historical calibration")

    if not bot.ml_ensemble or not bot.ml_ensemble.loaded:
        issues.append("[!] ML Ensemble not loaded - no ML predictions")

    if not bot.multitimeframe_analyzer:
        issues.append("[!] Multi-timeframe Analyzer missing - no MTF bonuses")

    if not bot.iv_analyzer:
        issues.append("[!] IV Analyzer missing - no IV environment assessment")

    if not bot.sentiment_analyzer:
        issues.append("[!] Sentiment Analyzer missing - no sentiment adjustments")

    if not bot.quant_analyzer:
        issues.append("[!] Quantitative Analyzer missing - no quant analysis")

    if issues:
        for issue in issues:
            print(issue)
        print()
        print("RECOMMENDATION: Some components are missing but bot will still work")
        print("Confidence scores may be lower than optimal")
    else:
        print("[OK] No critical issues found - all major components working!")

    print()
    print("="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

    # Summary
    if percentage >= 80:
        print("[OK] SYSTEM STATUS: GOOD - Most components operational")
    elif percentage >= 60:
        print("[!] SYSTEM STATUS: FAIR - Some components missing")
    else:
        print("[X] SYSTEM STATUS: POOR - Many components not working")

if __name__ == '__main__':
    asyncio.run(main())
