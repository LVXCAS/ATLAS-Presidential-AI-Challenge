"""
Complete Library and Dependencies Audit

Comprehensive analysis of all libraries and their usage in the HiveTrading system
"""

import sys
import pkg_resources
import importlib
import subprocess
import warnings
from pathlib import Path
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class LibraryAuditor:
    """Audit all libraries and dependencies"""

    def __init__(self):
        self.core_libraries = {}
        self.trading_libraries = {}
        self.ml_libraries = {}
        self.data_libraries = {}
        self.infrastructure_libraries = {}
        self.missing_libraries = []
        self.optional_libraries = []

    def audit_core_python_libraries(self):
        """Audit core Python libraries"""
        print("1. CORE PYTHON LIBRARIES")
        print("-" * 50)

        core_libs = [
            'os', 'sys', 'asyncio', 'datetime', 'pathlib', 'json', 'logging', 'warnings',
            'typing', 'dataclasses', 'enum', 'collections', 'itertools', 'functools',
            'concurrent.futures', 'threading', 'multiprocessing', 'queue', 'time'
        ]

        for lib in core_libs:
            try:
                importlib.import_module(lib)
                self.core_libraries[lib] = "✓ Built-in"
                print(f"[OK] {lib:<20} - Built-in Python library")
            except ImportError:
                self.core_libraries[lib] = "✗ Missing"
                print(f"[ERROR] {lib:<20} - Missing (should be built-in)")

    def audit_trading_libraries(self):
        """Audit trading-specific libraries"""
        print(f"\n2. TRADING & FINANCE LIBRARIES")
        print("-" * 50)

        trading_libs = {
            'alpaca_trade_api': 'Alpaca broker integration',
            'ib_insync': 'Interactive Brokers integration',
            'ccxt': 'Cryptocurrency exchange integration',
            'quantlib': 'Quantitative finance library',
            'bt': 'Backtesting framework',
            'zipline': 'Algorithmic trading library',
            'pyfolio': 'Portfolio performance analysis',
            'empyrical': 'Performance statistics',
            'ta': 'Technical analysis indicators',
            'ta_lib': 'Technical Analysis Library',
            'arch': 'GARCH models for volatility',
            'fredapi': 'Federal Reserve Economic Data'
        }

        for lib, description in trading_libs.items():
            try:
                importlib.import_module(lib)
                version = self.get_package_version(lib)
                self.trading_libraries[lib] = f"✓ {version}"
                print(f"[OK] {lib:<25} {version:<10} - {description}")
            except ImportError:
                self.trading_libraries[lib] = "✗ Missing"
                self.missing_libraries.append(lib)
                print(f"[MISSING] {lib:<25} {'N/A':<10} - {description}")

    def audit_ml_libraries(self):
        """Audit machine learning libraries"""
        print(f"\n3. MACHINE LEARNING LIBRARIES")
        print("-" * 50)

        ml_libs = {
            'numpy': 'Numerical computing',
            'pandas': 'Data manipulation and analysis',
            'scikit-learn': 'Machine learning toolkit',
            'scipy': 'Scientific computing',
            'statsmodels': 'Statistical modeling',
            'tensorflow': 'Deep learning framework',
            'torch': 'PyTorch deep learning',
            'keras': 'High-level neural networks',
            'xgboost': 'Gradient boosting',
            'lightgbm': 'Gradient boosting machine',
            'catboost': 'Gradient boosting',
            'optuna': 'Hyperparameter optimization',
            'mlflow': 'ML lifecycle management',
            'joblib': 'Parallel computing',
            'dask': 'Parallel computing'
        }

        for lib, description in ml_libs.items():
            try:
                importlib.import_module(lib)
                version = self.get_package_version(lib)
                self.ml_libraries[lib] = f"✓ {version}"
                print(f"[OK] {lib:<20} {version:<15} - {description}")
            except ImportError:
                self.ml_libraries[lib] = "✗ Missing"
                if lib in ['numpy', 'pandas', 'scikit-learn', 'scipy']:
                    self.missing_libraries.append(lib)
                    print(f"[CRITICAL] {lib:<20} {'N/A':<15} - {description}")
                else:
                    self.optional_libraries.append(lib)
                    print(f"[OPTIONAL] {lib:<20} {'N/A':<15} - {description}")

    def audit_data_libraries(self):
        """Audit data acquisition libraries"""
        print(f"\n4. DATA ACQUISITION LIBRARIES")
        print("-" * 50)

        data_libs = {
            'yfinance': 'Yahoo Finance data',
            'alpha_vantage': 'Alpha Vantage API',
            'polygon': 'Polygon.io market data',
            'quandl': 'Financial and economic data',
            'bloomberg': 'Bloomberg API',
            'eikon': 'Refinitiv Eikon API',
            'requests': 'HTTP requests',
            'websockets': 'WebSocket client/server',
            'aiohttp': 'Async HTTP client/server',
            'urllib3': 'HTTP client'
        }

        for lib, description in data_libs.items():
            try:
                importlib.import_module(lib)
                version = self.get_package_version(lib)
                self.data_libraries[lib] = f"✓ {version}"
                print(f"[OK] {lib:<20} {version:<15} - {description}")
            except ImportError:
                self.data_libraries[lib] = "✗ Missing"
                if lib in ['yfinance', 'requests']:
                    self.missing_libraries.append(lib)
                    print(f"[IMPORTANT] {lib:<20} {'N/A':<15} - {description}")
                else:
                    self.optional_libraries.append(lib)
                    print(f"[OPTIONAL] {lib:<20} {'N/A':<15} - {description}")

    def audit_infrastructure_libraries(self):
        """Audit infrastructure and deployment libraries"""
        print(f"\n5. INFRASTRUCTURE LIBRARIES")
        print("-" * 50)

        infra_libs = {
            'fastapi': 'Web API framework',
            'uvicorn': 'ASGI server',
            'streamlit': 'Dashboard framework',
            'dash': 'Interactive web applications',
            'flask': 'Web framework',
            'celery': 'Distributed task queue',
            'redis': 'In-memory data store',
            'psycopg2': 'PostgreSQL adapter',
            'sqlalchemy': 'SQL toolkit and ORM',
            'alembic': 'Database migrations',
            'prometheus_client': 'Metrics collection',
            'docker': 'Docker integration',
            'kubernetes': 'Kubernetes integration',
            'boto3': 'AWS SDK',
            'azure': 'Azure SDK',
            'google-cloud': 'Google Cloud SDK'
        }

        for lib, description in infra_libs.items():
            try:
                importlib.import_module(lib)
                version = self.get_package_version(lib)
                self.infrastructure_libraries[lib] = f"✓ {version}"
                print(f"[OK] {lib:<20} {version:<15} - {description}")
            except ImportError:
                self.infrastructure_libraries[lib] = "✗ Missing"
                self.optional_libraries.append(lib)
                print(f"[OPTIONAL] {lib:<20} {'N/A':<15} - {description}")

    def get_package_version(self, package_name):
        """Get version of installed package"""
        try:
            return pkg_resources.get_distribution(package_name).version
        except:
            return "Unknown"

    def generate_install_commands(self):
        """Generate installation commands for missing libraries"""
        print(f"\n{'='*70}")
        print("INSTALLATION COMMANDS FOR MISSING LIBRARIES")
        print("=" * 70)

        if self.missing_libraries:
            print(f"\nCRITICAL MISSING LIBRARIES:")
            print("pip install " + " ".join(self.missing_libraries))

        if self.optional_libraries:
            print(f"\nOPTIONAL LIBRARIES (install as needed):")
            for lib in self.optional_libraries[:10]:  # Show first 10
                print(f"pip install {lib}")

        # Create a comprehensive requirements file
        print(f"\nCREATING requirements.txt FILE...")
        self.create_requirements_file()

    def create_requirements_file(self):
        """Create comprehensive requirements.txt file"""

        requirements = [
            "# HiveTrading System Requirements",
            "# Generated on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "",
            "# Core Dependencies",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "requests>=2.25.0",
            "",
            "# Trading Libraries",
            "yfinance>=0.1.70",
            "alpaca-trade-api>=2.0.0",
            "quantlib>=1.26",
            "",
            "# Machine Learning",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "tensorflow>=2.8.0",
            "torch>=1.11.0",
            "",
            "# Infrastructure",
            "fastapi>=0.75.0",
            "uvicorn>=0.17.0",
            "streamlit>=1.8.0",
            "redis>=4.1.0",
            "psycopg2-binary>=2.9.0",
            "sqlalchemy>=1.4.0",
            "",
            "# Optional Advanced Libraries",
            "# ib_insync>=0.9.70",
            "# ccxt>=1.70.0",
            "# ta>=0.10.0",
            "# arch>=5.3.0",
            "",
            "# Development Tools",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.931"
        ]

        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(requirements))

        print(f"[OK] Created requirements.txt")

    def analyze_system_compatibility(self):
        """Analyze system compatibility and performance"""
        print(f"\n{'='*70}")
        print("SYSTEM COMPATIBILITY ANALYSIS")
        print("=" * 70)

        print(f"Python Version: {sys.version}")
        print(f"Platform: {sys.platform}")

        # Check memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"Available Memory: {memory.total // (1024**3):.1f} GB")
            print(f"Memory Usage: {memory.percent:.1f}%")
        except ImportError:
            print("Memory info unavailable (psutil not installed)")

        # Check CPU
        try:
            import multiprocessing
            print(f"CPU Cores: {multiprocessing.cpu_count()}")
        except:
            print("CPU info unavailable")

    def generate_library_summary(self):
        """Generate comprehensive library summary"""
        print(f"\n{'='*70}")
        print("LIBRARY AUDIT SUMMARY")
        print("=" * 70)

        total_core = len(self.core_libraries)
        available_core = sum(1 for v in self.core_libraries.values() if "✓" in v)

        total_trading = len(self.trading_libraries)
        available_trading = sum(1 for v in self.trading_libraries.values() if "✓" in v)

        total_ml = len(self.ml_libraries)
        available_ml = sum(1 for v in self.ml_libraries.values() if "✓" in v)

        total_data = len(self.data_libraries)
        available_data = sum(1 for v in self.data_libraries.values() if "✓" in v)

        total_infra = len(self.infrastructure_libraries)
        available_infra = sum(1 for v in self.infrastructure_libraries.values() if "✓" in v)

        print(f"Core Libraries:           {available_core}/{total_core} available")
        print(f"Trading Libraries:        {available_trading}/{total_trading} available")
        print(f"ML Libraries:             {available_ml}/{total_ml} available")
        print(f"Data Libraries:           {available_data}/{total_data} available")
        print(f"Infrastructure Libraries: {available_infra}/{total_infra} available")

        print(f"\nCRITICAL MISSING: {len(self.missing_libraries)}")
        print(f"OPTIONAL MISSING: {len(self.optional_libraries)}")

        # Overall readiness score
        critical_score = (available_core + available_trading + available_ml + available_data) / (total_core + total_trading + total_ml + total_data) * 100
        print(f"\nSYSTEM READINESS: {critical_score:.1f}%")

        return critical_score >= 80

def main():
    """Run comprehensive library audit"""

    print("HIVE TRADING - COMPREHENSIVE LIBRARY AUDIT")
    print("=" * 70)
    print(f"Audit Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    auditor = LibraryAuditor()

    # Run all audits
    auditor.audit_core_python_libraries()
    auditor.audit_trading_libraries()
    auditor.audit_ml_libraries()
    auditor.audit_data_libraries()
    auditor.audit_infrastructure_libraries()

    # Generate analysis
    auditor.analyze_system_compatibility()

    # Generate summary and installation commands
    is_ready = auditor.generate_library_summary()
    auditor.generate_install_commands()

    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR TOMORROW'S TRADING")
    print("=" * 70)

    if is_ready:
        print("✓ Library ecosystem is ready for trading")
        print("✓ Core functionality available")
        print("✓ Advanced features accessible")
    else:
        print("! Install missing critical libraries first")
        print("! Run: pip install -r requirements.txt")
        print("! Re-run audit after installation")

    return is_ready

if __name__ == "__main__":
    main()