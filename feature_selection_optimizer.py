"""
FEATURE SELECTION OPTIMIZATION
===============================
Day 2 - Identify the most predictive features for bank trading.
Use statistical tests to find the signal within the noise.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureSelectionOptimizer:
    """Optimize feature selection for bank trading"""
    
    def __init__(self):
        print("DAY 2 - FEATURE SELECTION OPTIMIZATION")
        print("=" * 50)
        print("Goal: Find the most predictive features")
        print("Methods: Statistical tests, importance ranking, RFE")
        print("Target: Achieve 55%+ accuracy with fewer features")
        print("=" * 50)
    
    def get_data_and_features(self):
        """Get JPM data and create comprehensive feature set"""
        print("\nCREATING COMPREHENSIVE FEATURE SET...")
        
        # Get JPM data
        ticker = yf.Ticker('JPM')
        data = ticker.history(period='2y')
        data.index = data.index.tz_localize(None)
        
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        volume = data['Volume']
        
        print("   Adding price features...")
        # Price features
        features['RETURN_1D'] = close.pct_change()
        features['RETURN_5D'] = close.pct_change(5)
        features['RETURN_10D'] = close.pct_change(10)
        features['RETURN_20D'] = close.pct_change(20)
        
        # Volatility features
        features['VOLATILITY_5D'] = features['RETURN_1D'].rolling(5).std()
        features['VOLATILITY_10D'] = features['RETURN_1D'].rolling(10).std()
        features['VOLATILITY_20D'] = features['RETURN_1D'].rolling(20).std()
        
        # Technical indicators
        features['RSI_14'] = self.calculate_rsi(close, 14)
        features['RSI_21'] = self.calculate_rsi(close, 21)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            ma = close.rolling(period).mean()
            features[f'SMA_{period}'] = ma
            features[f'PRICE_VS_SMA_{period}'] = (close - ma) / ma
        
        # Volume features
        features['VOLUME_RATIO_10'] = volume / volume.rolling(10).mean()
        features['VOLUME_RATIO_20'] = volume / volume.rolling(20).mean()
        
        print("   Adding fundamental features...")
        # Simulated fundamental features (would come from API)
        fundamentals = {
            'PE_RATIO': 15.1, 'BOOK_VALUE': 123.0, 'PRICE_TO_BOOK': 2.0,
            'DIVIDEND_YIELD': 0.019, 'ROE': 0.15, 'DEBT_TO_EQUITY': 1.8
        }
        for name, value in fundamentals.items():
            features[f'FUND_{name}'] = value
        
        print("   Adding economic features...")
        # Economic indicators (simulated time series)
        n_days = len(features)
        np.random.seed(42)  # Consistent results
        
        fed_rate = 4.5 + np.random.normal(0, 0.1, n_days).cumsum() * 0.01
        features['ECON_FED_RATE'] = np.clip(fed_rate, 0, 8)
        
        unemployment = 3.8 + np.random.normal(0, 0.05, n_days).cumsum() * 0.01
        features['ECON_UNEMPLOYMENT'] = np.clip(unemployment, 3, 6)
        
        treasury_10y = 4.2 + np.random.normal(0, 0.1, n_days).cumsum() * 0.01
        features['ECON_TREASURY_10Y'] = np.clip(treasury_10y, 2, 7)
        
        # Derived economic features
        features['ECON_REAL_RATE'] = features['ECON_FED_RATE'] - 2.5
        features['ECON_YIELD_SPREAD'] = features['ECON_TREASURY_10Y'] - features['ECON_FED_RATE']
        
        print("   Adding cross-market features...")
        # Market features (simulated)
        market_return = features['RETURN_1D'] * 0.8 + np.random.normal(0, 0.005, len(features))
        features['MARKET_RETURN'] = market_return
        features['RELATIVE_RETURN'] = features['RETURN_1D'] - features['MARKET_RETURN']
        features['BETA_20D'] = features['RETURN_1D'].rolling(20).corr(features['MARKET_RETURN']).fillna(1.2)
        
        # Clean features
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        print(f"   TOTAL FEATURES CREATED: {features.shape[1]}")
        
        # Create target
        target = (close.pct_change().shift(-1) > 0).astype(int)
        
        return features, target, data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def test_feature_importance_methods(self, features, target):
        """Test different feature importance methods"""
        print("\nTESTING FEATURE IMPORTANCE METHODS...")
        
        # Align data
        X = features.iloc[:-1]
        y = target.iloc[:-1].dropna()
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx].fillna(0)
        y = y.loc[common_idx]
        
        print(f"   Data: {len(X)} samples, {X.shape[1]} features")
        
        importance_results = {}
        
        # 1. Statistical F-test
        print("   1. Statistical F-test...")
        f_selector = SelectKBest(score_func=f_classif, k='all')
        f_selector.fit(X, y)
        f_scores = f_selector.scores_
        f_importance = pd.Series(f_scores, index=X.columns).sort_values(ascending=False)
        importance_results['F_Test'] = f_importance
        
        # 2. Mutual Information
        print("   2. Mutual Information...")
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_importance = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        importance_results['Mutual_Info'] = mi_importance
        
        # 3. Random Forest Feature Importance
        print("   3. Random Forest Importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        importance_results['Random_Forest'] = rf_importance
        
        # 4. Recursive Feature Elimination
        print("   4. Recursive Feature Elimination...")
        rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=15)
        rfe.fit(X, y)
        rfe_ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()
        importance_results['RFE_Ranking'] = rfe_ranking
        
        return importance_results, X, y
    
    def analyze_feature_importance(self, importance_results):
        """Analyze and combine feature importance results"""
        print("\nANALYZING FEATURE IMPORTANCE...")
        
        # Show top features from each method
        for method, importance in importance_results.items():
            print(f"\n{method} - Top 10 Features:")
            if method == 'RFE_Ranking':
                top_features = importance.head(10)
                for feature, rank in top_features.items():
                    print(f"     {feature}: Rank {rank}")
            else:
                top_features = importance.head(10)
                for feature, score in top_features.items():
                    print(f"     {feature}: {score:.4f}")
        
        # Create consensus ranking
        print("\nCREATING CONSENSUS RANKING...")
        
        all_features = importance_results['F_Test'].index
        consensus_scores = pd.Series(0, index=all_features)
        
        # Normalize and combine scores
        for method, importance in importance_results.items():
            if method == 'RFE_Ranking':
                # Convert ranking to score (lower rank = higher score)
                max_rank = importance.max()
                normalized = (max_rank - importance + 1) / max_rank
            else:
                # Normalize to 0-1
                normalized = (importance - importance.min()) / (importance.max() - importance.min())
            
            consensus_scores += normalized
        
        # Final consensus ranking
        consensus_ranking = consensus_scores.sort_values(ascending=False)
        
        print("\nCONSENSUS TOP 15 FEATURES:")
        for i, (feature, score) in enumerate(consensus_ranking.head(15).items(), 1):
            print(f"     {i:2d}. {feature}: {score:.3f}")
        
        return consensus_ranking
    
    def test_feature_subsets(self, X, y, consensus_ranking):
        """Test different numbers of top features"""
        print("\nTESTING FEATURE SUBSET PERFORMANCE...")
        
        results = {}
        feature_counts = [5, 10, 15, 20, 25, 30]
        
        for n_features in feature_counts:
            if n_features > len(consensus_ranking):
                continue
                
            print(f"\n   Testing top {n_features} features...")
            
            # Select top features
            top_features = consensus_ranking.head(n_features).index
            X_subset = X[top_features]
            
            # Test performance
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            tscv = TimeSeriesSplit(n_splits=5)
            
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
            
            results[n_features] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'features': list(top_features)
            }
            
            print(f"     {n_features} features: {scores.mean():.1%} (+/- {scores.std():.1%})")
        
        return results
    
    def find_optimal_feature_set(self, results):
        """Find the optimal number of features"""
        print("\nFINDING OPTIMAL FEATURE SET...")
        
        # Find best performance
        best_n = max(results.keys(), key=lambda k: results[k]['mean'])
        best_accuracy = results[best_n]['mean']
        best_features = results[best_n]['features']
        
        print(f"\nOPTIMAL FEATURE SELECTION RESULTS:")
        print("=" * 40)
        print(f"Best number of features: {best_n}")
        print(f"Best CV accuracy: {best_accuracy:.1%}")
        print(f"Improvement vs Day 1: {best_accuracy - 0.507:+.1%}")
        print(f"Improvement vs baseline: {best_accuracy - 0.523:+.1%}")  # vs 52.3% from multi-source
        
        print(f"\nOPTIMAL FEATURE SET:")
        for i, feature in enumerate(best_features, 1):
            print(f"     {i:2d}. {feature}")
        
        return best_n, best_accuracy, best_features
    
    def run_feature_optimization(self):
        """Run complete feature selection optimization"""
        
        # Get data and features
        features, target, data = self.get_data_and_features()
        
        # Test importance methods
        importance_results, X, y = self.test_feature_importance_methods(features, target)
        
        # Analyze importance
        consensus_ranking = self.analyze_feature_importance(importance_results)
        
        # Test feature subsets
        subset_results = self.test_feature_subsets(X, y, consensus_ranking)
        
        # Find optimal set
        optimal_n, optimal_accuracy, optimal_features = self.find_optimal_feature_set(subset_results)
        
        print(f"\nFEATURE SELECTION OPTIMIZATION COMPLETE!")
        print("=" * 50)
        
        if optimal_accuracy >= 0.55:
            print("EXCELLENT! Achieved 55%+ accuracy target!")
        elif optimal_accuracy > 0.523:
            print("GOOD! Improved on multi-source baseline!")
        else:
            print("MIXED: Need different approach")
        
        return {
            'optimal_n_features': optimal_n,
            'optimal_accuracy': optimal_accuracy,
            'optimal_features': optimal_features,
            'all_results': subset_results
        }

if __name__ == "__main__":
    optimizer = FeatureSelectionOptimizer()
    results = optimizer.run_feature_optimization()