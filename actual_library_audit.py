"""
Actual Library Audit

Real audit of ALL libraries actually installed and used in the system
"""

import sys
import importlib
import subprocess
from datetime import datetime

def get_installed_packages():
    """Get all installed packages with versions"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'],
                              capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error getting packages: {e}"

def audit_core_imports():
    """Audit all imports actually used in the codebase"""

    print("ACTUAL LIBRARY AUDIT - HIVE TRADING SYSTEM")
    print("="*70)
    print(f"Audit Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print()

    # Libraries actually imported in the codebase
    core_libraries = [
        ('numpy', 'Numerical computing foundation'),
        ('pandas', 'Data manipulation and analysis'),
        ('scipy', 'Scientific computing'),
        ('sklearn', 'Machine learning algorithms'),
        ('yfinance', 'Yahoo Finance market data'),
        ('requests', 'HTTP requests'),
        ('alpaca_trade_api', 'Alpaca broker API'),
        ('alpha_vantage', 'Alpha Vantage market data'),
        ('asyncio', 'Asynchronous programming'),
        ('json', 'JSON data handling'),
        ('datetime', 'Date and time operations'),
        ('warnings', 'Warning control'),
        ('os', 'Operating system interface'),
        ('sys', 'System-specific parameters'),
        ('pathlib', 'Object-oriented filesystem paths'),
        ('typing', 'Type hints'),
        ('dataclasses', 'Data classes'),
        ('enum', 'Enumerations'),
        ('concurrent.futures', 'Concurrent execution'),
        ('threading', 'Thread-based parallelism'),
        ('multiprocessing', 'Process-based parallelism'),
        ('random', 'Random number generation'),
        ('time', 'Time-related functions'),
        ('abc', 'Abstract base classes'),
        ('collections', 'Container datatypes'),
        ('itertools', 'Iterator functions'),
        ('functools', 'Higher-order functions'),
        ('pickle', 'Object serialization'),
        ('dotenv', 'Environment variables'),
    ]

    # Optional/Advanced libraries
    advanced_libraries = [
        ('tensorflow', 'Deep learning framework'),
        ('torch', 'PyTorch deep learning'),
        ('keras', 'High-level neural networks'),
        ('xgboost', 'Gradient boosting'),
        ('lightgbm', 'LightGBM gradient boosting'),
        ('optuna', 'Hyperparameter optimization'),
        ('ib_insync', 'Interactive Brokers API'),
        ('ccxt', 'Cryptocurrency exchange APIs'),
        ('arch', 'GARCH volatility models'),
        ('ta', 'Technical analysis indicators'),
        ('streamlit', 'Web dashboard framework'),
        ('fastapi', 'Web API framework'),
        ('uvicorn', 'ASGI server'),
        ('redis', 'Redis client'),
        ('sqlalchemy', 'SQL toolkit and ORM'),
        ('docker', 'Docker API client'),
        ('kubernetes', 'Kubernetes client'),
    ]

    print("CORE LIBRARIES (Required for autonomous R&D):")
    print("-" * 50)

    core_working = 0
    core_total = len(core_libraries)

    for lib, description in core_libraries:
        try:
            if lib == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif lib == 'alpaca_trade_api':
                import alpaca_trade_api as alpaca
                version = getattr(alpaca, '__version__', 'unknown')
            elif lib == 'alpha_vantage':
                import alpha_vantage
                version = getattr(alpha_vantage, '__version__', 'unknown')
            elif lib == 'dotenv':
                import dotenv
                version = getattr(dotenv, '__version__', 'unknown')
            else:
                module = importlib.import_module(lib)
                version = getattr(module, '__version__', 'built-in')

            print(f"✓ {lib:<20} {version:<15} - {description}")
            core_working += 1

        except ImportError:
            print(f"✗ {lib:<20} {'NOT FOUND':<15} - {description}")
        except Exception as e:
            print(f"? {lib:<20} {'ERROR':<15} - {description} ({e})")

    print(f"\nCORE LIBRARIES STATUS: {core_working}/{core_total} working ({(core_working/core_total)*100:.1f}%)")

    print(f"\nADVANCED LIBRARIES (Optional features):")
    print("-" * 50)

    advanced_working = 0
    advanced_total = len(advanced_libraries)

    for lib, description in advanced_libraries:
        try:
            module = importlib.import_module(lib)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {lib:<20} {version:<15} - {description}")
            advanced_working += 1

        except ImportError:
            print(f"- {lib:<20} {'NOT INSTALLED':<15} - {description}")
        except Exception as e:
            print(f"? {lib:<20} {'ERROR':<15} - {description}")

    print(f"\nADVANCED LIBRARIES STATUS: {advanced_working}/{advanced_total} available ({(advanced_working/advanced_total)*100:.1f}%)")

    # Test critical functionality
    print(f"\nCRITICAL FUNCTIONALITY TEST:")
    print("-" * 50)

    functionality_tests = [
        ('Market Data Access', test_market_data),
        ('Machine Learning', test_ml_functionality),
        ('Async Operations', test_async_functionality),
        ('API Connections', test_api_functionality),
        ('Data Processing', test_data_processing),
    ]

    tests_passed = 0
    for test_name, test_func in functionality_tests:
        try:
            result = test_func()
            status = "✓ WORKING" if result else "✗ FAILED"
            print(f"{status:<12} {test_name}")
            if result:
                tests_passed += 1
        except Exception as e:
            print(f"✗ ERROR     {test_name} - {e}")

    print(f"\nFUNCTIONALITY STATUS: {tests_passed}/{len(functionality_tests)} working ({(tests_passed/len(functionality_tests))*100:.1f}%)")

    # Overall system status
    overall_score = (core_working/core_total * 0.7 + tests_passed/len(functionality_tests) * 0.3) * 100

    print(f"\n{'='*70}")
    print("OVERALL SYSTEM STATUS")
    print("="*70)
    print(f"Core Libraries:      {core_working}/{core_total} ({(core_working/core_total)*100:.1f}%)")
    print(f"Advanced Libraries:  {advanced_working}/{advanced_total} ({(advanced_working/advanced_total)*100:.1f}%)")
    print(f"Critical Functions:  {tests_passed}/{len(functionality_tests)} ({(tests_passed/len(functionality_tests))*100:.1f}%)")
    print(f"OVERALL SCORE:       {overall_score:.1f}%")

    if overall_score >= 90:
        print("STATUS: EXCELLENT - System fully operational")
    elif overall_score >= 75:
        print("STATUS: GOOD - System mostly operational")
    elif overall_score >= 60:
        print("STATUS: FAIR - Some features may be limited")
    else:
        print("STATUS: POOR - System needs attention")

def test_market_data():
    """Test market data functionality"""
    try:
        import yfinance as yf
        ticker = yf.Ticker("SPY")
        data = ticker.history(period="1d")
        return len(data) > 0
    except:
        return False

def test_ml_functionality():
    """Test machine learning functionality"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np

        X = np.random.rand(50, 3)
        y = np.random.rand(50)

        rf = RandomForestRegressor(n_estimators=5, random_state=42)
        rf.fit(X, y)
        prediction = rf.predict(X[:1])

        return len(prediction) > 0
    except:
        return False

def test_async_functionality():
    """Test async functionality"""
    try:
        import asyncio

        async def test_async():
            await asyncio.sleep(0.001)
            return True

        result = asyncio.run(test_async())
        return result
    except:
        return False

def test_api_functionality():
    """Test API functionality"""
    try:
        import requests
        response = requests.get('https://httpbin.org/status/200', timeout=5)
        return response.status_code == 200
    except:
        return False

def test_data_processing():
    """Test data processing functionality"""
    try:
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            'price': np.random.rand(100) * 100,
            'volume': np.random.randint(1000, 10000, 100)
        })

        df['returns'] = df['price'].pct_change()
        result = df['returns'].std()

        return not np.isnan(result)
    except:
        return False

def main():
    """Run the complete library audit"""
    audit_core_imports()

    print(f"\n{'='*70}")
    print("COMPLETE PACKAGE LIST")
    print("="*70)
    print("All installed packages:")
    print()
    packages = get_installed_packages()
    print(packages)

if __name__ == "__main__":
    main()