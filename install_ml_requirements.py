#!/usr/bin/env python3
"""
Install required libraries for advanced ML features
"""
import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("🔧 INSTALLING ADVANCED ML REQUIREMENTS")
    print("=" * 50)
    
    # Required packages for advanced ML
    packages = [
        "pandas",
        "numpy", 
        "scipy",
        "scikit-learn",
        "requests",
        "plotly",  # For volatility surface visualization
    ]
    
    # Optional but recommended packages
    optional_packages = [
        "ta-lib",  # Technical Analysis Library
        "pandas-ta",  # Alternative technical indicators
        "alpha-vantage",  # Alpha Vantage API wrapper
    ]
    
    print("Installing required packages...")
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstalled {success_count}/{len(packages)} required packages")
    
    print("\nInstalling optional packages...")
    optional_success = 0
    for package in optional_packages:
        if install_package(package):
            optional_success += 1
    
    print(f"Installed {optional_success}/{len(optional_packages)} optional packages")
    
    print("\n" + "=" * 50)
    print("📋 SETUP API KEYS (Optional but recommended)")
    print("=" * 50)
    
    print("""
To enable full ML features, set these environment variables:

1. Alpha Vantage (Free tier available):
   - Sign up: https://www.alphavantage.co/support/#api-key
   - Set: ALPHA_VANTAGE_API_KEY=your_key_here
   
2. News API (Free tier available):
   - Sign up: https://newsapi.org/
   - Set: NEWS_API_KEY=your_key_here

3. Example Windows setup:
   set ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   set NEWS_API_KEY=your_news_api_key

4. Example Linux/Mac setup:
   export ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   export NEWS_API_KEY=your_news_api_key
""")
    
    print("=" * 50)
    print("🚀 WHAT'S ENABLED NOW")
    print("=" * 50)
    
    features_enabled = []
    
    # Check what's available
    try:
        import talib
        features_enabled.append("✅ Technical Analysis (RSI, MACD, Bollinger Bands)")
    except ImportError:
        print("⚠️  TA-Lib not available - technical indicators will use simplified versions")
    
    try:
        import sklearn
        features_enabled.append("✅ Machine Learning (Random Forest, Gradient Boosting)")
    except ImportError:
        print("❌ scikit-learn not available - ML predictions disabled")
    
    try:
        import requests
        features_enabled.append("✅ API Data Fetching")
    except ImportError:
        print("❌ requests not available - external data disabled")
    
    if features_enabled:
        print("\nEnabled features:")
        for feature in features_enabled:
            print(f"  {feature}")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Set API keys as environment variables (optional)")
    print("2. Run: python test_advanced_ml.py")
    print("3. The bot will now use advanced ML features!")

if __name__ == "__main__":
    main()