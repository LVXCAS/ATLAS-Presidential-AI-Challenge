"""
SIMPLE MULTI-SOURCE DATA TEST
==============================
Simpler approach to test multi-source data integration
without timezone complications.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

class SimpleMultiDataTest:
    """Simple multi-source data test"""
    
    def __init__(self):
        print("DAY 2 - SIMPLE MULTI-SOURCE DATA TEST")
        print("=" * 50)
        print("Testing impact of additional data on JPM prediction:")
        print("  • Price data (baseline)")
        print("  • Fundamental ratios")
        print("  • Economic indicators")
        print("  • Cross-market features")
        print("=" * 50)
    
    def get_baseline_data(self):
        """Get baseline JPM price data"""
        print("\nGETTING BASELINE DATA...")
        
        ticker = yf.Ticker('JPM')
        data = ticker.history(period='2y')
        
        # Remove timezone info to avoid complications
        data.index = data.index.tz_localize(None)
        
        print(f"   JPM data: {len(data)} days")
        return data
    
    def create_baseline_features(self, data):
        """Create baseline price-only features"""
        print("   Creating baseline price features...")
        
        features = pd.DataFrame(index=data.index)
        
        # Simple price features
        features['RETURN_1D'] = data['Close'].pct_change()
        features['RETURN_5D'] = data['Close'].pct_change(5)
        features['VOLATILITY_20D'] = features['RETURN_1D'].rolling(20).std()
        features['RSI_14'] = self.calculate_rsi(data['Close'], 14)
        features['SMA_20'] = data['Close'].rolling(20).mean()
        features['PRICE_VS_SMA'] = (data['Close'] - features['SMA_20']) / features['SMA_20']
        
        # Volume feature
        features['VOLUME_RATIO'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        features = features.fillna(method='ffill').fillna(0)
        return features
    
    def calculate_rsi(self, prices, period=14):
        """Simple RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def add_fundamental_features(self, features):
        """Add simulated fundamental features"""
        print("   Adding fundamental features...")
        
        # Simulate fundamental data for JPM (these would come from API)
        fundamentals = {
            'PE_RATIO': 15.1,
            'BOOK_VALUE': 123.0,
            'PRICE_TO_BOOK': 2.0,
            'DIVIDEND_YIELD': 0.019,  # 1.9%
            'ROE': 0.15,  # 15%
            'DEBT_TO_EQUITY': 1.8
        }
        
        for name, value in fundamentals.items():
            features[f'FUND_{name}'] = value
        
        return features
    
    def add_economic_features(self, features):
        """Add simulated economic indicators"""
        print("   Adding economic indicators...")
        
        # Simulate time-varying economic data
        n_days = len(features)
        
        # Generate realistic economic time series
        fed_rate_base = 4.5
        features['ECON_FED_RATE'] = fed_rate_base + np.random.normal(0, 0.1, n_days).cumsum() * 0.01
        features['ECON_FED_RATE'] = features['ECON_FED_RATE'].clip(0, 8)
        
        unemployment_base = 3.8
        features['ECON_UNEMPLOYMENT'] = unemployment_base + np.random.normal(0, 0.05, n_days).cumsum() * 0.01
        features['ECON_UNEMPLOYMENT'] = features['ECON_UNEMPLOYMENT'].clip(3, 6)
        
        treasury_10y_base = 4.2
        features['ECON_TREASURY_10Y'] = treasury_10y_base + np.random.normal(0, 0.1, n_days).cumsum() * 0.01
        features['ECON_TREASURY_10Y'] = features['ECON_TREASURY_10Y'].clip(2, 7)
        
        # Derived features
        features['ECON_REAL_RATE'] = features['ECON_FED_RATE'] - 2.5  # Assuming 2.5% inflation
        features['ECON_YIELD_SPREAD'] = features['ECON_TREASURY_10Y'] - features['ECON_FED_RATE']
        
        return features
    
    def add_cross_market_features(self, features, price_data):
        """Add cross-market features"""
        print("   Adding cross-market features...")
        
        try:
            # Get SPY data for market comparison
            spy_data = yf.download('SPY', start=price_data.index[0], end=price_data.index[-1], progress=False)
            spy_data.index = spy_data.index.tz_localize(None)
            
            # Align SPY with JPM data
            spy_aligned = spy_data['Close'].reindex(features.index, method='ffill')
            
            # Market relative performance
            features['MARKET_RETURN'] = spy_aligned.pct_change()
            features['RELATIVE_RETURN'] = features['RETURN_1D'] - features['MARKET_RETURN']
            features['BETA_20D'] = features['RETURN_1D'].rolling(20).corr(features['MARKET_RETURN'])
            
            print(f"     Added SPY cross-market features")
            
        except Exception as e:
            print(f"     Warning: Could not get SPY data: {e}")
            # Add dummy market features
            features['MARKET_RETURN'] = features['RETURN_1D'] * 0.8  # Simulated market
            features['RELATIVE_RETURN'] = features['RETURN_1D'] - features['MARKET_RETURN']
            features['BETA_20D'] = 1.2  # JPM typically has beta > 1
        
        return features
    
    def test_feature_sets(self, price_data):
        """Test different feature combinations"""
        print("\nTESTING FEATURE COMBINATIONS...")
        
        results = {}
        
        # Create target
        target = (price_data['Close'].pct_change().shift(-1) > 0).astype(int)
        
        # 1. Baseline: Price-only features
        print("\n1. TESTING BASELINE (Price-only)...")
        baseline_features = self.create_baseline_features(price_data)
        baseline_acc = self.test_features(baseline_features, target, "Baseline")
        results['Baseline'] = baseline_acc
        
        # 2. Add fundamental features
        print("\n2. TESTING + Fundamentals...")
        fund_features = baseline_features.copy()
        fund_features = self.add_fundamental_features(fund_features)
        fund_acc = self.test_features(fund_features, target, "With_Fundamentals")
        results['With_Fundamentals'] = fund_acc
        
        # 3. Add economic features
        print("\n3. TESTING + Economics...")
        econ_features = fund_features.copy()
        econ_features = self.add_economic_features(econ_features)
        econ_acc = self.test_features(econ_features, target, "With_Economics")
        results['With_Economics'] = econ_acc
        
        # 4. Add cross-market features
        print("\n4. TESTING + Cross-Market...")
        full_features = econ_features.copy()
        full_features = self.add_cross_market_features(full_features, price_data)
        full_acc = self.test_features(full_features, target, "Full_Multi_Source")
        results['Full_Multi_Source'] = full_acc
        
        return results
    
    def test_features(self, features, target, name):
        """Test a feature set"""
        
        # Align data
        X = features.iloc[:-1]  # Remove last row (no target)
        y = target.iloc[:-1].dropna()
        
        # Only use overlapping indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        print(f"     Samples: {len(X)}, Features: {X.shape[1]}")
        
        if len(X) < 100:
            print(f"     ERROR: Insufficient data")
            return 0.0
        
        # Scale and test
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(0))
        
        model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        
        scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
        
        print(f"     {name} CV Accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")
        return scores.mean()
    
    def analyze_results(self, results):
        """Analyze which data sources help most"""
        print(f"\nMULTI-SOURCE DATA ANALYSIS:")
        print("=" * 40)
        
        baseline = results['Baseline']
        
        print(f"Baseline (Price-only): {baseline:.1%}")
        
        for feature_set, accuracy in results.items():
            if feature_set != 'Baseline':
                improvement = accuracy - baseline
                print(f"{feature_set}: {accuracy:.1%} ({improvement:+.1%})")
        
        # Find best approach
        best_approach = max(results, key=results.get)
        best_accuracy = results[best_approach]
        
        print(f"\nBEST APPROACH: {best_approach}")
        print(f"BEST ACCURACY: {best_accuracy:.1%}")
        
        total_improvement = best_accuracy - baseline
        print(f"TOTAL IMPROVEMENT: {total_improvement:+.1%}")
        
        # Compare to Day 1 result
        day1_comparison = best_accuracy - 0.507  # Day 1 was 50.7%
        print(f"vs Day 1 Result: {day1_comparison:+.1%}")
        
        return best_approach, best_accuracy
    
    def run_test(self):
        """Run complete multi-source test"""
        
        # Get data
        price_data = self.get_baseline_data()
        
        # Test feature combinations
        results = self.test_feature_sets(price_data)
        
        # Analyze results
        best_approach, best_accuracy = self.analyze_results(results)
        
        print(f"\nDAY 2 MULTI-SOURCE TEST COMPLETE!")
        print("=" * 40)
        
        if best_accuracy >= 0.55:
            print("EXCELLENT! Multi-source data shows strong improvement!")
        elif best_accuracy > 0.507:
            print("GOOD! Multi-source data helps!")
        else:
            print("MIXED: Need different approach to data integration")
        
        return results

if __name__ == "__main__":
    tester = SimpleMultiDataTest()
    results = tester.run_test()