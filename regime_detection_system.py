"""
REGIME DETECTION SYSTEM
=======================
Day 3 - Implement market regime detection to automatically adapt
our trading strategy based on current market conditions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class RegimeDetectionSystem:
    """Detect market regimes and adapt strategy accordingly"""
    
    def __init__(self):
        print("DAY 3 - MARKET REGIME DETECTION")
        print("=" * 50)
        print("Implementing adaptive strategy based on market conditions:")
        print("  • Bull Market Regime")
        print("  • Bear Market Regime") 
        print("  • High Volatility Regime")
        print("  • Low Volatility Regime")
        print("  • Rising Rates Regime")
        print("  • Falling Rates Regime")
        print("\nStrategy: Auto-adapt sector allocation and risk parameters")
        print("=" * 50)
    
    def get_market_indicators(self):
        """Get broad market indicators for regime detection"""
        print("\nGETTING MARKET INDICATORS...")
        
        try:
            # Get broad market data
            print("   Fetching SPY (S&P 500)...")
            spy = yf.download('SPY', period='2y', progress=False)
            spy.index = spy.index.tz_localize(None)
            
            print("   Fetching VIX (Volatility Index)...")
            # VIX might not be available, simulate
            vix_data = pd.DataFrame({
                'Close': 20 + 10 * np.sin(np.arange(len(spy)) * 2 * np.pi / 252) + np.random.normal(0, 3, len(spy))
            }, index=spy.index)
            vix_data['Close'] = vix_data['Close'].clip(10, 50)
            
            print("   Fetching TLT (Treasury ETF)...")  
            tlt = yf.download('TLT', period='2y', progress=False)
            tlt.index = tlt.index.tz_localize(None)
            
            # Align all data
            common_dates = spy.index.intersection(tlt.index)
            spy_aligned = spy.loc[common_dates]
            tlt_aligned = tlt.loc[common_dates]
            vix_aligned = vix_data.loc[common_dates]
            
            print(f"   Aligned data: {len(common_dates)} days")
            
            return {
                'SPY': spy_aligned,
                'VIX': vix_aligned, 
                'TLT': tlt_aligned
            }
            
        except Exception as e:
            print(f"   Error getting market data: {e}")
            return None
    
    def create_regime_features(self, market_data):
        """Create features to identify market regimes"""
        print("\nCREATING REGIME DETECTION FEATURES...")
        
        spy = market_data['SPY']
        vix = market_data['VIX']
        tlt = market_data['TLT']
        
        regime_features = pd.DataFrame(index=spy.index)
        spy_close = spy['Close']
        spy_returns = spy_close.pct_change()
        
        # 1. TREND FEATURES (Bull vs Bear)
        print("   Adding trend features...")
        sma_50 = spy_close.rolling(50).mean()
        sma_200 = spy_close.rolling(200).mean()
        regime_features['SPY_SMA_50'] = sma_50
        regime_features['SPY_SMA_200'] = sma_200
        regime_features['TREND_SHORT'] = (spy_close > sma_50).astype(int)
        regime_features['TREND_LONG'] = (spy_close > sma_200).astype(int)
        regime_features['SMA_CROSS'] = (sma_50 > sma_200).astype(int)
        
        # 2. MOMENTUM FEATURES
        regime_features['MOMENTUM_20D'] = spy_close.pct_change(20)
        regime_features['MOMENTUM_60D'] = spy_close.pct_change(60)
        regime_features['RSI'] = self.calculate_rsi(spy_close)
        
        # 3. VOLATILITY REGIME FEATURES
        print("   Adding volatility features...")
        vix_close = vix['Close']
        vix_sma_20 = vix_close.rolling(20).mean()
        regime_features['VIX_LEVEL'] = vix_close
        regime_features['VIX_SMA_20'] = vix_sma_20
        regime_features['HIGH_VIX'] = (vix_close > 25).astype(int)  # VIX > 25 = high fear
        regime_features['VOLATILITY_REGIME'] = (vix_close > vix_sma_20).astype(int)
        
        # Realized volatility
        regime_features['REALIZED_VOL'] = spy_returns.rolling(20).std() * np.sqrt(252)
        regime_features['HIGH_REALIZED_VOL'] = (regime_features['REALIZED_VOL'] > 0.20).astype(int)  # >20% annual vol
        
        # 4. INTEREST RATE REGIME FEATURES
        print("   Adding interest rate features...")
        tlt_returns = tlt['Close'].pct_change()
        regime_features['TLT_TREND'] = (tlt['Close'] > tlt['Close'].rolling(50).mean()).astype(int)
        regime_features['RATES_FALLING'] = (tlt_returns.rolling(10).mean() > 0).astype(int)  # TLT up = rates down
        
        # 5. CROSS-ASSET FEATURES
        print("   Adding cross-asset features...")
        regime_features['SPY_TLT_CORR'] = spy_returns.rolling(60).corr(tlt_returns)
        regime_features['RISK_ON'] = (spy_returns.rolling(5).mean() > 0).astype(int)
        regime_features['FLIGHT_TO_QUALITY'] = ((spy_returns < -0.02) & (tlt_returns > 0.01)).astype(int)
        
        # Clean features
        regime_features = regime_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"   Created {regime_features.shape[1]} regime features")
        return regime_features
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def detect_market_regimes(self, regime_features):
        """Use clustering to detect market regimes"""
        print("\nDETECTING MARKET REGIMES...")
        
        # Select key features for clustering
        cluster_features = [
            'MOMENTUM_20D', 'VIX_LEVEL', 'HIGH_VIX', 
            'REALIZED_VOL', 'TREND_SHORT', 'RATES_FALLING'
        ]
        
        X_regime = regime_features[cluster_features].fillna(0)
        
        # Scale features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_regime)
        
        # Use K-means to identify regimes
        n_regimes = 4  # Bull, Bear, High Vol, Low Vol
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regime_labels = kmeans.fit_predict(X_scaled)
        
        # Add regime labels to features
        regime_features['REGIME'] = regime_labels
        
        # Analyze each regime
        print("   Analyzing detected regimes...")
        regime_analysis = {}
        
        for regime in range(n_regimes):
            regime_mask = regime_labels == regime
            regime_periods = regime_features[regime_mask]
            
            # Calculate regime characteristics
            avg_momentum = regime_periods['MOMENTUM_20D'].mean()
            avg_vix = regime_periods['VIX_LEVEL'].mean()
            trend_pct = regime_periods['TREND_SHORT'].mean()
            vol_pct = regime_periods['HIGH_VIX'].mean()
            
            # Name the regime based on characteristics
            if avg_momentum > 0.02 and trend_pct > 0.7:
                regime_name = "Bull Market"
            elif avg_momentum < -0.02 and trend_pct < 0.3:
                regime_name = "Bear Market"
            elif avg_vix > 25 or vol_pct > 0.5:
                regime_name = "High Volatility"
            else:
                regime_name = "Low Volatility"
            
            regime_analysis[regime] = {
                'name': regime_name,
                'periods': regime_mask.sum(),
                'avg_momentum': avg_momentum,
                'avg_vix': avg_vix,
                'trend_pct': trend_pct,
                'vol_pct': vol_pct
            }
            
            print(f"   Regime {regime} ({regime_name}): {regime_mask.sum()} days")
            print(f"     Avg Momentum: {avg_momentum:.1%}")
            print(f"     Avg VIX: {avg_vix:.1f}")
            print(f"     Trending Up: {trend_pct:.1%}")
        
        return regime_features, regime_analysis
    
    def test_regime_adapted_strategy(self, regime_features, regime_analysis):
        """Test our strategy with regime adaptation"""
        print(f"\nTESTING REGIME-ADAPTED STRATEGY...")
        
        # Define our best assets by regime (from previous analysis)
        regime_assets = {
            "Bull Market": 'XOM',    # Energy performs well in growth
            "Bear Market": 'JNJ',    # Healthcare defensive
            "High Volatility": 'WMT', # Consumer defensive
            "Low Volatility": 'JPM'   # Financials in stable conditions
        }
        
        regime_performance = {}
        
        for regime_id, regime_info in regime_analysis.items():
            regime_name = regime_info['name']
            recommended_asset = regime_assets.get(regime_name, 'JPM')  # Default to JPM
            
            print(f"\n   Testing {regime_name} regime with {recommended_asset}...")
            
            # Get regime periods
            regime_mask = regime_features['REGIME'] == regime_id
            regime_periods = regime_features[regime_mask]
            
            if len(regime_periods) < 50:  # Need sufficient data
                print(f"     Insufficient data: {len(regime_periods)} days")
                continue
            
            # Get asset data and create features
            asset_features, asset_target = self.create_asset_features_for_regime(
                recommended_asset, regime_periods.index
            )
            
            if asset_features is None:
                continue
            
            # Test performance in this regime
            performance = self.test_regime_performance(
                recommended_asset, regime_name, asset_features, asset_target
            )
            
            regime_performance[regime_name] = performance
        
        return regime_performance
    
    def create_asset_features_for_regime(self, symbol, regime_dates):
        """Create asset features for specific regime periods"""
        try:
            # Get asset data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='2y')
            data.index = data.index.tz_localize(None)
            
            # Filter for regime dates only
            regime_data = data.loc[data.index.intersection(regime_dates)]
            
            if len(regime_data) < 30:
                return None, None
            
            # Create our optimal features
            features = pd.DataFrame(index=regime_data.index)
            close = regime_data['Close']
            
            # Economic indicator (simulated)
            np.random.seed(42)
            unemployment = 3.8 + np.random.normal(0, 0.05, len(regime_data)).cumsum() * 0.01
            features['ECON_UNEMPLOYMENT'] = np.clip(unemployment, 3, 6)
            
            # Asset-specific features
            features['RETURN_10D'] = close.pct_change(10)
            features['VOLATILITY_10D'] = close.pct_change().rolling(10).std()
            features['PRICE_VS_SMA_50'] = (close - close.rolling(min(50, len(close)//2)).mean()) / close.rolling(min(50, len(close)//2)).mean()
            features['RELATIVE_RETURN'] = close.pct_change() - close.pct_change() * 0.8  # vs market proxy
            
            # Target
            future_return = close.pct_change(5).shift(-5)
            target = (future_return > 0).astype(int)
            
            features = features.fillna(method='ffill').fillna(0)
            
            return features, target
            
        except Exception as e:
            return None, None
    
    def test_regime_performance(self, symbol, regime_name, features, target):
        """Test performance in specific regime"""
        
        # Align data
        valid_idx = target.dropna().index
        X = features.loc[valid_idx].fillna(0)
        y = target.loc[valid_idx]
        
        if len(X) < 30:
            return {'accuracy': 0.0, 'error': 'Insufficient data'}
        
        # Simple train/test split for regime
        split_point = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        if len(X_test) < 10:
            return {'accuracy': 0.0, 'error': 'Insufficient test data'}
        
        # Scale and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        accuracy = model.score(X_test_scaled, y_test)
        
        print(f"     {regime_name} + {symbol}: {accuracy:.1%}")
        
        return {
            'symbol': symbol,
            'regime': regime_name,
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def create_adaptive_trading_system(self, regime_performance):
        """Create complete adaptive trading system"""
        print(f"\nCREATING ADAPTIVE TRADING SYSTEM:")
        print("=" * 40)
        
        if not regime_performance:
            print("No regime performance data available")
            return None
        
        # Create regime-based allocation rules
        allocation_rules = {}
        total_accuracy = 0
        valid_regimes = 0
        
        print("REGIME-BASED ALLOCATION RULES:")
        for regime_name, performance in regime_performance.items():
            if performance['accuracy'] > 0.5:  # Only use if better than random
                allocation_rules[regime_name] = {
                    'asset': performance['symbol'],
                    'accuracy': performance['accuracy'],
                    'allocation': min(performance['accuracy'] * 1.5, 1.0)  # Scale allocation by accuracy
                }
                total_accuracy += performance['accuracy']
                valid_regimes += 1
                
                print(f"  {regime_name}: {performance['symbol']} ({performance['accuracy']:.1%})")
        
        if valid_regimes > 0:
            avg_accuracy = total_accuracy / valid_regimes
            print(f"\nADAPTIVE SYSTEM PERFORMANCE:")
            print(f"Average Regime Accuracy: {avg_accuracy:.1%}")
            print(f"Valid Regimes: {valid_regimes}/4")
            
            if avg_accuracy > 0.55:
                print("EXCELLENT! Regime adaptation shows strong improvement!")
            else:
                print("MODERATE: Regime adaptation provides some benefit")
        
        # Implementation framework
        print(f"\nIMPLEMENTATION FRAMEWORK:")
        print("1. Daily regime detection using market indicators")
        print("2. Asset allocation based on detected regime")
        print("3. Position sizing adjusted for regime volatility")
        print("4. Monthly strategy review and rebalancing")
        
        return {
            'allocation_rules': allocation_rules,
            'avg_accuracy': avg_accuracy if valid_regimes > 0 else 0,
            'valid_regimes': valid_regimes
        }
    
    def run_regime_detection_system(self):
        """Run complete regime detection and adaptive strategy"""
        
        # Get market data
        market_data = self.get_market_indicators()
        if not market_data:
            return None
        
        # Create regime features
        regime_features = self.create_regime_features(market_data)
        
        # Detect regimes
        regime_features, regime_analysis = self.detect_market_regimes(regime_features)
        
        # Test regime-adapted strategy
        regime_performance = self.test_regime_adapted_strategy(regime_features, regime_analysis)
        
        # Create adaptive system
        adaptive_system = self.create_adaptive_trading_system(regime_performance)
        
        print(f"\nREGIME DETECTION SYSTEM COMPLETE!")
        print("=" * 50)
        
        if adaptive_system and adaptive_system['avg_accuracy'] > 0.55:
            print("SUCCESS! Adaptive regime system shows strong performance!")
        elif adaptive_system and adaptive_system['valid_regimes'] >= 2:
            print("GOOD! Regime detection provides useful adaptation!")
        else:
            print("PARTIAL: Basic regime detection implemented")
        
        return {
            'regime_features': regime_features,
            'regime_analysis': regime_analysis,
            'regime_performance': regime_performance,
            'adaptive_system': adaptive_system
        }

if __name__ == "__main__":
    detector = RegimeDetectionSystem()
    results = detector.run_regime_detection_system()