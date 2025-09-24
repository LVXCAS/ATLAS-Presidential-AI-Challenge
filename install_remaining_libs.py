"""
INSTALL REMAINING HIGH-IMPACT LIBRARIES
=======================================
Script to install the most important missing libraries
for maximum quantitative finance potential.
"""

import subprocess
import sys

# High-priority libraries to install
HIGH_PRIORITY_LIBS = [
    # Advanced Portfolio & Risk Management
    'pypfopt',
    'cvxpy',
    'pyfolio-reloaded',
    
    # More Data Sources  
    'pandas-datareader',
    'fredapi',
    'polygon-api-client',
    
    # Backtesting Frameworks
    'backtrader',
    'vectorbt',
    'fastquant',
    
    # Time Series & Forecasting
    'prophet',
    'pmdarima',
    
    # Advanced Visualization
    'dash',
    'streamlit',
    'cufflinks',
    
    # More Broker APIs
    'ib-insync',
    'python-binance',
    
    # Monte Carlo & Simulation
    'pymc',
    
    # Specialized Tools
    'ta',  # Simple technical analysis
]

MEDIUM_PRIORITY_LIBS = [
    # More ML Libraries
    'tensorflow',
    'optuna',
    
    # More Technical Analysis
    'finta',
    'tulip',
    
    # More Optimization
    'pulp',
    'deap',
    
    # Alternative Data
    'scrapy',
    'selenium',
    'newspaper3k',
    
    # More Trading Platforms
    'freqtrade',
]

def install_libraries(lib_list, priority="HIGH"):
    """Install a list of libraries"""
    
    print(f"\n🚀 INSTALLING {priority} PRIORITY LIBRARIES")
    print("=" * 50)
    
    successful = []
    failed = []
    
    for lib in lib_list:
        print(f"\n📦 Installing {lib}...")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', lib],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per library
            )
            
            if result.returncode == 0:
                print(f"   ✅ SUCCESS: {lib}")
                successful.append(lib)
            else:
                print(f"   ❌ FAILED: {lib}")
                print(f"      Error: {result.stderr[:200]}...")
                failed.append(lib)
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT: {lib} (taking too long)")
            failed.append(lib)
        except Exception as e:
            print(f"   ❌ ERROR: {lib} - {e}")
            failed.append(lib)
    
    print(f"\n📊 {priority} PRIORITY RESULTS:")
    print(f"   ✅ Successful: {len(successful)}")
    print(f"   ❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n✅ SUCCESSFULLY INSTALLED:")
        for lib in successful:
            print(f"   • {lib}")
    
    if failed:
        print(f"\n❌ FAILED TO INSTALL:")
        for lib in failed:
            print(f"   • {lib}")
    
    return successful, failed

def show_installation_summary():
    """Show what we would achieve with full installation"""
    
    print("""
🎯 WHAT FULL INSTALLATION ACHIEVES:
==================================

With ALL high-priority libraries installed, you would have:

📊 DATA SOURCES (15+ APIs):
   • Yahoo Finance, Alpha Vantage, Polygon, IEX Cloud
   • FRED economic data, SEC filings
   • 300+ crypto exchanges via CCXT
   • Chinese market data (akshare, tushare)

🧠 MACHINE LEARNING (20+ algorithms):
   • Traditional: sklearn, XGBoost, LightGBM
   • Deep Learning: TensorFlow, PyTorch
   • Reinforcement Learning: FinRL
   • Hyperparameter tuning: Optuna

📈 BACKTESTING (10+ frameworks):
   • Professional: Zipline, Backtrader
   • Fast: vectorbt, fastquant
   • Event-driven: QSTrader, basana

🛡️ RISK MANAGEMENT (15+ tools):
   • Modern Portfolio Theory: PyPortfolioOpt
   • Advanced optimization: CVXPY
   • Risk analytics: PyFolio, empyrical
   • Factor analysis: Alphalens

⚡ EXECUTION (10+ brokers):
   • Stock: Alpaca, Interactive Brokers
   • Crypto: Binance, Coinbase (300+ exchanges)
   • Forex: MetaTrader5, XTB

📊 VISUALIZATION (8+ libraries):
   • Interactive: Plotly, Dash, Streamlit
   • Financial: mplfinance, cufflinks
   • Professional: Matplotlib, Seaborn

🎯 RESULT: COMPLETE HEDGE FUND CAPABILITIES!
""")

def main():
    """Main installation workflow"""
    
    print("""
🌌 QUANTUM FINANCE LIBRARY INSTALLER
===================================

This script will attempt to install the remaining high-impact
quantitative finance libraries to achieve MAXIMUM POTENTIAL.

Current Status: ~30/200+ libraries installed
Target: 80+ core libraries for institutional capabilities
""")
    
    show_installation_summary()
    
    response = input("\n🚀 Install HIGH PRIORITY libraries? (y/n): ")
    
    if response.lower() == 'y':
        successful_high, failed_high = install_libraries(HIGH_PRIORITY_LIBS, "HIGH")
        
        if len(successful_high) > len(failed_high):
            response2 = input("\n🚀 Install MEDIUM PRIORITY libraries? (y/n): ")
            if response2.lower() == 'y':
                successful_med, failed_med = install_libraries(MEDIUM_PRIORITY_LIBS, "MEDIUM")
                
                total_successful = len(successful_high) + len(successful_med)
                total_attempted = len(HIGH_PRIORITY_LIBS) + len(MEDIUM_PRIORITY_LIBS)
            else:
                total_successful = len(successful_high)
                total_attempted = len(HIGH_PRIORITY_LIBS)
        else:
            total_successful = len(successful_high)
            total_attempted = len(HIGH_PRIORITY_LIBS)
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   📦 Attempted: {total_attempted} libraries")
        print(f"   ✅ Successful: {total_successful} libraries")
        print(f"   📊 Success Rate: {(total_successful/total_attempted)*100:.1f}%")
        
        if total_successful >= len(HIGH_PRIORITY_LIBS) * 0.7:
            print(f"\n🎯 STATUS: MAXIMUM POTENTIAL APPROACHING!")
            print(f"   You now have institutional-grade capabilities!")
        else:
            print(f"\n⚠️ STATUS: PARTIAL INSTALLATION")
            print(f"   Some advanced features may not be available.")
            
    else:
        print("\n📊 Installation cancelled. Current capabilities maintained.")
        print("   Run 'python install_remaining_libs.py' anytime to upgrade!")

if __name__ == "__main__":
    main()