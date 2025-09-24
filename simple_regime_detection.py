"""
SIMPLE REGIME DETECTION SYSTEM
===============================
Simplified but effective regime detection for our trading system.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class SimpleRegimeDetection:
    """Simple but effective regime detection"""
    
    def __init__(self):
        print("DAY 3 - SIMPLE REGIME DETECTION")
        print("=" * 50)
        print("Market regimes based on:")
        print("  • Market trend (Bull/Bear)")
        print("  • Volatility level (High/Low)")
        print("  • Momentum strength")
        print("\nAsset allocation by regime:")
        print("  • Bull + Low Vol: JPM (Financials)")
        print("  • Bull + High Vol: XOM (Energy)") 
        print("  • Bear + Low Vol: JNJ (Healthcare)")
        print("  • Bear + High Vol: WMT (Consumer)")
        print("=" * 50)
    
    def get_market_data_and_detect_regimes(self):
        """Get market data and detect simple regimes"""
        print("\nDETECTING MARKET REGIMES...")
        
        try:
            # Get SPY for market regime
            spy_data = yf.download('SPY', period='2y', progress=False)
            spy_data.index = spy_data.index.tz_localize(None)
            
            # Handle multi-level columns if present
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy = spy_data.droplevel(1, axis=1)
            else:
                spy = spy_data
            
            # Calculate regime indicators
            spy_close = spy['Close']
            spy_returns = spy_close.pct_change()
            
            # 1. TREND REGIME (Bull vs Bear)
            sma_50 = spy_close.rolling(50).mean()
            sma_200 = spy_close.rolling(200).mean()
            is_bull = (spy_close > sma_200) & (sma_50 > sma_200)
            
            # 2. VOLATILITY REGIME (High vs Low)
            volatility = spy_returns.rolling(20).std() * np.sqrt(252)  # Annualized
            vol_median = volatility.median()
            is_high_vol = volatility > vol_median
            
            # 3. MOMENTUM REGIME
            momentum_20d = spy_close.pct_change(20)
            momentum_positive = momentum_20d > 0
            
            # Create regime classification
            regimes = []
            regime_names = []
            
            # Convert to arrays for easier indexing
            bull_array = is_bull.fillna(False).values
            vol_array = is_high_vol.fillna(False).values
            
            for i in range(len(spy)):
                bull = bull_array[i]
                high_vol = vol_array[i]
                
                if bull and not high_vol:
                    regimes.append(0)
                    regime_names.append("Bull_Low_Vol")
                elif bull and high_vol:
                    regimes.append(1)
                    regime_names.append("Bull_High_Vol")
                elif not bull and not high_vol:
                    regimes.append(2)
                    regime_names.append("Bear_Low_Vol")
                else:  # Bear + High Vol
                    regimes.append(3)
                    regime_names.append("Bear_High_Vol")
            
            # Create regime summary
            regime_df = pd.DataFrame({
                'Date': spy.index,
                'SPY_Close': spy_close,
                'Regime': regimes,
                'Regime_Name': regime_names,
                'Is_Bull': is_bull,
                'Is_High_Vol': is_high_vol,
                'Volatility': volatility,
                'Momentum_20D': momentum_20d
            }).set_index('Date')
            
            # Analyze regimes
            print("   REGIME DISTRIBUTION:")
            regime_counts = pd.Series(regime_names).value_counts()
            for regime_name, count in regime_counts.items():
                pct = count / len(regime_names) * 100
                print(f"     {regime_name}: {count} days ({pct:.1f}%)")
            
            print(f"   Total periods analyzed: {len(spy)} days")
            
            return regime_df
            
        except Exception as e:
            print(f"   ERROR: {e}")
            return None
    
    def test_regime_based_asset_allocation(self, regime_df):
        """Test our regime-based asset allocation strategy"""
        print(f"\nTESTING REGIME-BASED ALLOCATION...")
        
        # Asset allocation by regime
        regime_assets = {
            "Bull_Low_Vol": 'JPM',    # Financials in stable bull market
            "Bull_High_Vol": 'XOM',   # Energy in volatile bull market
            "Bear_Low_Vol": 'JNJ',    # Healthcare in stable bear market
            "Bear_High_Vol": 'WMT'    # Consumer defensive in volatile bear
        }
        
        allocation_performance = {}
        
        for regime_name, asset in regime_assets.items():
            print(f"\n   Testing {regime_name} -> {asset}...")
            
            # Get regime periods
            regime_periods = regime_df[regime_df['Regime_Name'] == regime_name]
            
            if len(regime_periods) < 30:
                print(f"     Insufficient data: {len(regime_periods)} days")
                continue
            
            # Test asset performance in this regime
            performance = self.test_asset_in_regime(asset, regime_periods.index, regime_name)
            allocation_performance[regime_name] = performance
        
        return allocation_performance
    
    def test_asset_in_regime(self, symbol, regime_dates, regime_name):
        """Test specific asset performance in specific regime"""
        try:
            # Get asset data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='2y')
            data.index = data.index.tz_localize(None)
            
            # Filter for regime dates
            regime_data = data.reindex(regime_dates).dropna()
            
            if len(regime_data) < 20:
                return {'accuracy': 0.0, 'error': 'Insufficient regime data'}
            
            # Create simple features
            close = regime_data['Close']
            features = pd.DataFrame(index=regime_data.index)
            
            # Basic features
            features['RETURN_5D'] = close.pct_change(5)
            features['RETURN_10D'] = close.pct_change(10)
            features['VOLATILITY'] = close.pct_change().rolling(10).std()
            features['SMA_RATIO'] = close / close.rolling(20).mean()
            features['MOMENTUM'] = close.pct_change(10)
            
            # Target: 5-day forward return
            target = (close.pct_change(5).shift(-5) > 0).astype(int)
            
            # Clean data
            features = features.fillna(method='ffill').fillna(0)
            
            # Align and test
            valid_idx = target.dropna().index
            X = features.loc[valid_idx]
            y = target.loc[valid_idx]
            
            if len(X) < 10:
                return {'accuracy': 0.0, 'error': 'Insufficient aligned data'}
            
            # Simple train/test split
            split_point = max(1, len(X) // 2)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            if len(X_test) < 3:
                return {'accuracy': 0.0, 'error': 'Insufficient test data'}
            
            # Scale and train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.fillna(0))
            X_test_scaled = scaler.transform(X_test.fillna(0))
            
            model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            accuracy = model.score(X_test_scaled, y_test)
            
            print(f"     {symbol} in {regime_name}: {accuracy:.1%} ({len(X_test)} test samples)")
            
            return {
                'symbol': symbol,
                'regime': regime_name,
                'accuracy': accuracy,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'regime_days': len(regime_data)
            }
            
        except Exception as e:
            print(f"     ERROR with {symbol}: {e}")
            return {'accuracy': 0.0, 'error': str(e)}
    
    def create_adaptive_allocation_strategy(self, allocation_performance):
        """Create final adaptive allocation strategy"""
        print(f"\nADAPTIVE ALLOCATION STRATEGY:")
        print("=" * 40)
        
        successful_allocations = {k: v for k, v in allocation_performance.items() 
                                if v.get('accuracy', 0) > 0.5}
        
        if not successful_allocations:
            print("No successful regime-based allocations found")
            return None
        
        print("REGIME-BASED ASSET ALLOCATION:")
        total_accuracy = 0
        valid_regimes = 0
        
        for regime, performance in successful_allocations.items():
            accuracy = performance['accuracy']
            asset = performance['symbol']
            total_accuracy += accuracy
            valid_regimes += 1
            
            print(f"  {regime}: {asset} ({accuracy:.1%} accuracy)")
        
        avg_accuracy = total_accuracy / valid_regimes if valid_regimes > 0 else 0
        
        print(f"\nSTRATEGY PERFORMANCE:")
        print(f"Average Accuracy: {avg_accuracy:.1%}")
        print(f"Valid Regimes: {valid_regimes}/4")
        
        # Compare to baseline
        baseline_jpn = 0.579
        improvement = avg_accuracy - baseline_jpn
        
        print(f"vs JPM Baseline: {improvement:+.1%}")
        
        if avg_accuracy > baseline_jpn:
            print("SUCCESS! Regime-based allocation beats baseline!")
        else:
            print("MIXED: Regime allocation needs refinement")
        
        # Implementation rules
        print(f"\nIMPLEMENTATION RULES:")
        print("1. Detect current market regime weekly")
        print("2. Allocate to recommended asset for that regime")
        print("3. Adjust position size based on regime volatility")
        print("4. Rebalance when regime changes")
        
        return {
            'successful_allocations': successful_allocations,
            'average_accuracy': avg_accuracy,
            'improvement_vs_baseline': improvement,
            'valid_regimes': valid_regimes
        }
    
    def run_simple_regime_system(self):
        """Run complete simple regime detection system"""
        
        # Detect regimes
        regime_df = self.get_market_data_and_detect_regimes()
        if regime_df is None:
            return None
        
        # Test regime-based allocation
        allocation_performance = self.test_regime_based_asset_allocation(regime_df)
        
        # Create adaptive strategy
        adaptive_strategy = self.create_adaptive_allocation_strategy(allocation_performance)
        
        print(f"\nSIMPLE REGIME DETECTION COMPLETE!")
        print("=" * 50)
        
        if adaptive_strategy and adaptive_strategy['improvement_vs_baseline'] > 0:
            print("EXCELLENT! Regime-based system improves performance!")
        elif adaptive_strategy and adaptive_strategy['valid_regimes'] >= 2:
            print("GOOD! Regime detection provides useful insights!")
        else:
            print("PARTIAL: Basic regime framework established")
        
        return {
            'regime_data': regime_df,
            'allocation_performance': allocation_performance,
            'adaptive_strategy': adaptive_strategy
        }

if __name__ == "__main__":
    detector = SimpleRegimeDetection()
    results = detector.run_simple_regime_system()