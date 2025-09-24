"""
PREDICTION TIMEFRAME OPTIMIZATION
==================================
Test different prediction horizons to find the optimal timeframe
for bank trading with our 5 best features.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

class TimeframeOptimizer:
    """Optimize prediction timeframe for bank trading"""
    
    def __init__(self):
        print("DAY 2 - PREDICTION TIMEFRAME OPTIMIZATION")
        print("=" * 50)
        print("Testing optimal prediction horizon:")
        print("  • Next 1 day (current)")
        print("  • Next 2 days")
        print("  • Next 3 days")
        print("  • Next 5 days")
        print("  • Next 10 days")
        print("Using our optimal 5-feature set")
        print("=" * 50)
    
    def get_optimal_features(self, data):
        """Create our optimal 5-feature set"""
        print("\nCREATING OPTIMAL 5-FEATURE SET...")
        
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        
        # 1. ECON_UNEMPLOYMENT (simulated)
        np.random.seed(42)
        unemployment = 3.8 + np.random.normal(0, 0.05, len(data)).cumsum() * 0.01
        features['ECON_UNEMPLOYMENT'] = np.clip(unemployment, 3, 6)
        
        # 2. RETURN_10D
        features['RETURN_10D'] = close.pct_change(10)
        
        # 3. VOLATILITY_10D  
        daily_returns = close.pct_change()
        features['VOLATILITY_10D'] = daily_returns.rolling(10).std()
        
        # 4. PRICE_VS_SMA_50
        sma_50 = close.rolling(50).mean()
        features['PRICE_VS_SMA_50'] = (close - sma_50) / sma_50
        
        # 5. RELATIVE_RETURN (vs market)
        market_return = daily_returns * 0.8 + np.random.normal(0, 0.005, len(data))
        features['RELATIVE_RETURN'] = daily_returns - market_return
        
        # Clean features
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"   Created {features.shape[1]} optimal features")
        return features
    
    def create_targets_multiple_horizons(self, data):
        """Create targets for different prediction horizons"""
        print("   Creating targets for multiple horizons...")
        
        close = data['Close']
        targets = {}
        
        # Different prediction horizons
        horizons = [1, 2, 3, 5, 10]
        
        for horizon in horizons:
            # Predict if price will be higher in N days
            future_return = close.pct_change(horizon).shift(-horizon)
            targets[f'{horizon}D'] = (future_return > 0).astype(int)
        
        return targets
    
    def test_prediction_horizons(self, features, targets):
        """Test different prediction horizons"""
        print("\nTESTING PREDICTION HORIZONS...")
        
        results = {}
        
        for horizon_name, target in targets.items():
            print(f"\n   Testing {horizon_name} prediction...")
            
            # Align data - remove NaN values from target
            X = features
            y = target
            
            # Find valid indices (no NaN in target)
            valid_idx = y.dropna().index
            X_clean = X.loc[valid_idx].fillna(0)
            y_clean = y.loc[valid_idx]
            
            if len(X_clean) < 100:
                print(f"     Insufficient data: {len(X_clean)} samples")
                results[horizon_name] = {'accuracy': 0.0, 'samples': len(X_clean)}
                continue
            
            print(f"     Samples: {len(X_clean)}")
            print(f"     Target distribution: {y_clean.value_counts().to_dict()}")
            
            # Test performance
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            tscv = TimeSeriesSplit(n_splits=3)
            
            scores = cross_val_score(model, X_scaled, y_clean, cv=tscv, scoring='accuracy')
            
            results[horizon_name] = {
                'accuracy': scores.mean(),
                'std': scores.std(),
                'samples': len(X_clean),
                'target_balance': y_clean.mean()  # Proportion of positive cases
            }
            
            print(f"     {horizon_name} Accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")
        
        return results
    
    def analyze_timeframe_results(self, results):
        """Analyze which timeframe works best"""
        print(f"\nTIMEFRAME ANALYSIS:")
        print("=" * 40)
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print("RESULTS BY ACCURACY:")
        for horizon, metrics in sorted_results:
            if metrics['accuracy'] > 0:
                acc = metrics['accuracy']
                std = metrics['std']
                balance = metrics['target_balance']
                samples = metrics['samples']
                print(f"{horizon:4s}: {acc:.1%} (+/- {std:.1%}) | Balance: {balance:.1%} | Samples: {samples}")
        
        # Find best timeframe
        best_horizon = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_horizon]['accuracy']
        
        print(f"\nBEST PREDICTION HORIZON: {best_horizon}")
        print(f"BEST ACCURACY: {best_accuracy:.1%}")
        
        # Compare to our 1-day baseline
        baseline_1d = results['1D']['accuracy']
        improvement = best_accuracy - baseline_1d
        
        print(f"1-Day Baseline: {baseline_1d:.1%}")
        print(f"Improvement: {improvement:+.1%}")
        
        # Practical implications
        print(f"\nPRACTICAL IMPLICATIONS:")
        if best_horizon == '1D':
            print("• Daily predictions are optimal")
            print("• Good for day trading and short-term positions")
        elif best_horizon in ['2D', '3D']:
            print("• Short-term predictions work best")
            print("• Good for swing trading (2-3 day holds)")
        elif best_horizon == '5D':
            print("• Weekly predictions are optimal")
            print("• Good for weekly rebalancing strategies")
        elif best_horizon == '10D':
            print("• Longer-term predictions work better")
            print("• Good for bi-weekly or monthly strategies")
        
        return best_horizon, best_accuracy
    
    def test_intraday_concepts(self):
        """Test concept of intraday vs daily predictions"""
        print(f"\nINTRADAY PREDICTION CONCEPTS:")
        print("-" * 30)
        print("Higher frequency predictions (1min, 5min, 1hour) would require:")
        print("• Intraday data (not just daily closes)")
        print("• Different feature engineering (microstructure)")
        print("• Much larger datasets")
        print("• Different market dynamics (noise vs signal)")
        print("")
        print("For institutional-grade results:")
        print("• Daily predictions: Good starting point")
        print("• Intraday predictions: Advanced technique")
        print("• Our current approach is solid foundation")
    
    def run_timeframe_optimization(self):
        """Run complete timeframe optimization"""
        
        # Get JPM data
        ticker = yf.Ticker('JPM')
        data = ticker.history(period='2y')
        data.index = data.index.tz_localize(None)
        
        print(f"Data: {len(data)} days of JPM")
        
        # Create optimal features
        features = self.get_optimal_features(data)
        
        # Create targets for different horizons
        targets = self.create_targets_multiple_horizons(data)
        
        # Test prediction horizons
        results = self.test_prediction_horizons(features, targets)
        
        # Analyze results
        best_horizon, best_accuracy = self.analyze_timeframe_results(results)
        
        # Intraday concepts
        self.test_intraday_concepts()
        
        print(f"\nTIMEFRAME OPTIMIZATION COMPLETE!")
        print("=" * 50)
        
        if best_accuracy > 0.554:  # Our current 55.4% benchmark
            print("EXCELLENT! Found better prediction horizon!")
        else:
            print("Daily predictions remain optimal for our approach")
        
        return {
            'best_horizon': best_horizon,
            'best_accuracy': best_accuracy,
            'all_results': results
        }

if __name__ == "__main__":
    optimizer = TimeframeOptimizer()
    results = optimizer.run_timeframe_optimization()