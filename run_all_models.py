#!/usr/bin/env python3
"""
Run All ML Models
This script runs all the machine learning models in the Hive Trading system.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration validation
def check_configuration() -> Dict[str, Any]:
    """Check system configuration and return status."""
    config_status = {
        "master_password": False,
        "alpaca_api_key": False,
        "database_config": False,
        "warnings": [],
        "errors": []
    }
    
    try:
        # Check environment variables
        master_password = os.getenv("TRADING_SYSTEM_MASTER_PASSWORD")
        alpaca_key_id = os.getenv("ALPACA_API_KEY_ID")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
        
        config_status["master_password"] = bool(master_password)
        config_status["alpaca_api_key"] = bool(alpaca_key_id and alpaca_secret)
        
        if not master_password:
            config_status["warnings"].append("Master password not set - some models may fail")
        
        if not (alpaca_key_id and alpaca_secret):
            config_status["warnings"].append("Alpaca API keys not set - market data models may fail")
        
        # Check database configuration
        try:
            from config.database import get_database_config
            db_config = get_database_config()
            config_status["database_config"] = True
        except Exception as e:
            config_status["database_config"] = False
            config_status["errors"].append(f"Database configuration error: {e}")
        
    except Exception as e:
        config_status["errors"].append(f"Configuration check failed: {e}")
    
    return config_status

# Import demo modules
from examples.execution_engine_demo import ExecutionEngineDemo
from examples.mean_reversion_demo import MeanReversionDemo
from examples.momentum_trading_demo import main as momentum_main
from examples.market_data_ingestor_demo import run_comprehensive_demo as market_data_demo
from examples.risk_manager_demo import RiskManagerDemo
from examples.news_sentiment_demo import NewsSentimentDemo
from examples.technical_indicators_demo import main as technical_indicators_main
from examples.broker_integration_demo import BrokerIntegrationDemo
from examples.visualization_demo import main as visualization_main
from examples.portfolio_allocator_demo import main as portfolio_allocator_main
from examples.options_volatility_demo import main as options_volatility_main
from examples.enhanced_communication_demo import main as enhanced_communication_main
from examples.backtesting_demo import run_comprehensive_backtest_demo


async def run_all_models():
    """Run all ML models in the system"""
    print("\n" + "=" * 80)
    print("RUNNING ALL ML MODELS IN HIVE TRADING SYSTEM")
    print("=" * 80)
    
    # Check configuration first
    print("\n🔍 Checking system configuration...")
    config_status = check_configuration()
    
    if config_status["warnings"]:
        print("\n⚠️  Configuration Warnings:")
        for warning in config_status["warnings"]:
            print(f"   • {warning}")
    
    if config_status["errors"]:
        print("\n❌ Configuration Errors:")
        for error in config_status["errors"]:
            print(f"   • {error}")
    
    print(f"\n📊 Configuration Status:")
    print(f"   • Master Password: {'✅' if config_status['master_password'] else '❌'}")
    print(f"   • Alpaca API Keys: {'✅' if config_status['alpaca_api_key'] else '❌'}")
    print(f"   • Database Config: {'✅' if config_status['database_config'] else '❌'}")
    
    models = [
        ("Execution Engine", run_execution_engine, ["master_password"]),
        ("Mean Reversion", run_mean_reversion, ["master_password"]),
        ("Momentum Trading", run_momentum_trading, []),
        ("Market Data Ingestor", run_market_data_ingestor, ["alpaca_api_key"]),
        ("Risk Manager", run_risk_manager, ["database_config"]),
        ("News Sentiment Analysis", run_news_sentiment, ["master_password"]),
        ("Technical Indicators", run_technical_indicators, []),
        ("Broker Integration", run_broker_integration, []),
        ("Visualization", run_visualization, []),
        ("Portfolio Allocator", run_portfolio_allocator, []),
        ("Options Volatility", run_options_volatility, []),
        ("Enhanced Communication", run_enhanced_communication, []),
        ("Backtesting Engine", run_backtesting, [])
    ]
    
    results = {}
    
    for model_name, model_func, dependencies in models:
        print(f"\n\n{'=' * 80}")
        print(f"RUNNING {model_name.upper()} MODEL")
        print(f"{'=' * 80}")
        
        # Check dependencies
        missing_deps = [dep for dep in dependencies if not config_status.get(dep, False)]
        if missing_deps:
            print(f"⚠️  Warning: Missing dependencies: {', '.join(missing_deps)}")
            print("   This model may fail or have limited functionality.")
        
        try:
            await model_func()
            results[model_name] = "SUCCESS"
            print(f"\n✅ {model_name} completed successfully!")
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages for common issues
            if "master password" in error_msg.lower():
                error_msg += " (Set TRADING_SYSTEM_MASTER_PASSWORD environment variable)"
            elif "alpaca" in error_msg.lower() and "api" in error_msg.lower():
                error_msg += " (Set ALPACA_API_KEY_ID and ALPACA_SECRET_KEY environment variables)"
            elif "database" in error_msg.lower():
                error_msg += " (Check database configuration and connectivity)"
            
            results[model_name] = f"FAILED: {error_msg}"
            print(f"\n❌ {model_name} failed: {error_msg}")
            
            # Only print full traceback for unexpected errors
            if not any(keyword in error_msg.lower() for keyword in ["master password", "alpaca", "database", "api key"]):
                import traceback
                traceback.print_exc()
        
        # Small delay between models
        await asyncio.sleep(1)
    
    # Print summary
    print("\n\n" + "=" * 80)
    print("ML MODELS EXECUTION SUMMARY")
    print("=" * 80)
    
    successful = 0
    for model_name, result in results.items():
        status = "✅" if result == "SUCCESS" else "❌"
        print(f"{status} {model_name}: {result}")
        if result == "SUCCESS":
            successful += 1
    
    print(f"\nSuccessfully ran {successful}/{len(models)} models")
    
    # Provide setup instructions if many models failed
    if successful < len(models) / 2:
        print("\n💡 Setup Instructions:")
        print("   To resolve configuration issues:")
        print("   1. Copy .env.template to .env")
        print("   2. Set TRADING_SYSTEM_MASTER_PASSWORD in .env")
        print("   3. Set ALPACA_API_KEY_ID and ALPACA_SECRET_KEY in .env")
        print("   4. Ensure PostgreSQL database is running")
        print("   5. Run: python -m pip install -r requirements.txt")


async def run_execution_engine():
    """Run Execution Engine model"""
    demo = ExecutionEngineDemo()
    await demo.run_all_demos()


async def run_mean_reversion():
    """Run Mean Reversion model"""
    demo = MeanReversionDemo()
    demo.run_all_demos()


async def run_momentum_trading():
    """Run Momentum Trading model"""
    await momentum_main()


async def run_market_data_ingestor():
    """Run Market Data Ingestor model"""
    await market_data_demo()


async def run_risk_manager():
    """Run Risk Manager model"""
    demo = RiskManagerDemo()
    await demo.run_complete_demo()


async def run_news_sentiment():
    """Run News Sentiment Analysis model"""
    demo = NewsSentimentDemo()
    await demo.run_all_demos()


async def run_technical_indicators():
    """Run Technical Indicators model"""
    technical_indicators_main()
    return True


async def run_broker_integration():
    """Run Broker Integration model"""
    demo = BrokerIntegrationDemo()
    await demo.run_all_demos()


async def run_visualization():
    """Run Visualization model"""
    await visualization_main()


async def run_portfolio_allocator():
    """Run Portfolio Allocator model"""
    await portfolio_allocator_main()


async def run_options_volatility():
    """Run Options Volatility model"""
    await options_volatility_main()


async def run_enhanced_communication():
    """Run Enhanced Communication model"""
    await enhanced_communication_main()


async def run_backtesting():
    """Run Backtesting Engine model"""
    run_comprehensive_backtest_demo()
    return True


if __name__ == "__main__":
    try:
        asyncio.run(run_all_models())
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)