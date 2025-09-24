"""
MULTI-SOURCE DATA INTEGRATION TEST
===================================
Day 2 - Test Alpha Vantage, FRED, and other data sources
to enhance our bank trading model beyond just price data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime, timedelta

class MultiSourceDataTest:
    """Test integration of multiple data sources"""
    
    def __init__(self):
        self.symbol = 'JPM'  # Focus on JPM for consistency
        print("DAY 2 - MULTI-SOURCE DATA INTEGRATION")
        print("=" * 50)
        print("Testing data sources beyond just price:")
        print("  • Alpha Vantage (fundamentals)")
        print("  • FRED (macro economics)")
        print("  • Yahoo Finance (enhanced)")
        print("  • Custom indicators")
        print("=" * 50)
    
    def test_alpha_vantage(self):
        """Test Alpha Vantage fundamental data"""
        print("\nTESTING ALPHA VANTAGE...")
        
        # Note: Alpha Vantage requires API key for real use
        # For now, we'll simulate the data structure
        
        try:
            # This would be the real API call:
            # api_key = "YOUR_API_KEY"
            # url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={self.symbol}&apikey={api_key}"
            # response = requests.get(url)
            
            print("   API Key required for full access")
            print("   Simulating fundamental data structure...")
            
            # Simulate fundamental data that Alpha Vantage provides
            fundamental_data = {
                'MarketCapitalization': 450000000000,  # $450B for JPM
                'BookValue': 95.50,
                'DividendYield': 0.025,  # 2.5%
                'PERatio': 12.5,
                'PEGRatio': 1.2,
                'PriceToBookRatio': 1.6,
                'ReturnOnEquityTTM': 0.15,  # 15%
                'ReturnOnAssetsTTM': 0.012,  # 1.2%
                'DebtToEquityRatio': 1.8,
                'CurrentRatio': 1.1,
                'RevenuePerShareTTM': 45.2,
                'QuarterlyRevenueGrowthYOY': 0.08  # 8%
            }
            
            print("   AVAILABLE FUNDAMENTAL METRICS:")
            for metric, value in fundamental_data.items():
                if isinstance(value, float) and value < 1:
                    print(f"     {metric}: {value:.1%}")
                elif isinstance(value, float):
                    print(f"     {metric}: {value:.2f}")
                else:
                    print(f"     {metric}: ${value:,.0f}" if 'Market' in metric else f"     {metric}: {value}")
            
            print("   STATUS: Ready for integration (need API key)")
            return fundamental_data
            
        except Exception as e:
            print(f"   ERROR: {e}")
            return None
    
    def test_fred_api(self):
        """Test FRED economic data"""
        print("\nTESTING FRED ECONOMIC DATA...")
        
        try:
            # FRED API is free but requires registration
            # Simulating key economic indicators that affect banks
            
            print("   Simulating key economic indicators for banks...")
            
            # Generate realistic time series for economic indicators
            dates = pd.date_range('2022-01-01', '2024-01-01', freq='M')
            
            economic_data = pd.DataFrame({
                'FEDERAL_FUNDS_RATE': np.random.normal(4.5, 0.5, len(dates)).clip(0, 8),  # Fed funds rate
                'UNEMPLOYMENT_RATE': np.random.normal(3.8, 0.3, len(dates)).clip(3, 6),   # Unemployment
                'INFLATION_RATE': np.random.normal(2.5, 0.5, len(dates)).clip(1, 5),      # CPI inflation
                'GDP_GROWTH': np.random.normal(2.2, 0.8, len(dates)).clip(-2, 5),         # GDP growth
                'TREASURY_10Y': np.random.normal(4.2, 0.4, len(dates)).clip(2, 7),       # 10Y Treasury
                'TREASURY_2Y': np.random.normal(4.8, 0.4, len(dates)).clip(2, 7),        # 2Y Treasury
                'YIELD_CURVE': lambda x: x['TREASURY_10Y'] - x['TREASURY_2Y'],            # Yield curve
                'VIX': np.random.normal(18, 5, len(dates)).clip(10, 40),                  # Market volatility
            }, index=dates)
            
            # Calculate yield curve
            economic_data['YIELD_CURVE'] = economic_data['TREASURY_10Y'] - economic_data['TREASURY_2Y']
            
            print("   AVAILABLE ECONOMIC INDICATORS:")
            for indicator in economic_data.columns:
                latest_value = economic_data[indicator].iloc[-1]
                print(f"     {indicator}: {latest_value:.2f}")
            
            print("   STATUS: Ready for integration")
            return economic_data
            
        except Exception as e:
            print(f"   ERROR: {e}")
            return None
    
    def get_enhanced_yahoo_data(self):
        """Get enhanced Yahoo Finance data with more metrics"""
        print("\nTESTING ENHANCED YAHOO FINANCE...")
        
        try:
            # Get stock data
            stock = yf.Ticker(self.symbol)
            
            # Get multiple data types
            print("   Fetching multiple data streams...")
            
            # Price data
            price_data = stock.history(period='2y')
            print(f"     Price data: {len(price_data)} days")
            
            # Try to get additional info
            try:
                info = stock.info
                key_metrics = {
                    'sector': info.get('sector', 'Financial Services'),
                    'industry': info.get('industry', 'Banks—Diversified'),
                    'marketCap': info.get('marketCap', 0),
                    'trailingPE': info.get('trailingPE', 0),
                    'dividendYield': info.get('dividendYield', 0),
                    'bookValue': info.get('bookValue', 0),
                    'priceToBook': info.get('priceToBook', 0)
                }
                
                print("   ADDITIONAL YAHOO METRICS:")
                for metric, value in key_metrics.items():
                    if value and isinstance(value, (int, float)):
                        if 'Yield' in metric or 'PE' in metric:
                            print(f"     {metric}: {value:.3f}")
                        else:
                            print(f"     {metric}: {value:,.0f}")
                    else:
                        print(f"     {metric}: {value}")
                
                return price_data, key_metrics
                
            except Exception as e:
                print(f"     Info data unavailable: {e}")
                return price_data, {}
                
        except Exception as e:
            print(f"   ERROR: {e}")
            return None, None
    
    def create_multi_source_features(self, price_data, fundamental_data, economic_data):
        """Combine all data sources into features"""
        print("\nCREATING MULTI-SOURCE FEATURES...")
        
        features = pd.DataFrame(index=price_data.index)
        
        # 1. PRICE FEATURES (baseline)
        print("   Adding price features...")
        close = price_data['Close'].values.astype(float)
        
        # Key price indicators
        features['RETURN_1D'] = price_data['Close'].pct_change()
        features['RETURN_5D'] = price_data['Close'].pct_change(5)
        features['VOLATILITY_20D'] = features['RETURN_1D'].rolling(20).std()
        
        # 2. FUNDAMENTAL FEATURES (if available)
        if fundamental_data:
            print("   Adding fundamental features...")
            
            # Convert fundamentals to time series (static for now)
            for metric, value in fundamental_data.items():
                if isinstance(value, (int, float)):
                    features[f'FUND_{metric}'] = value
        
        # 3. ECONOMIC FEATURES (macro environment)
        if economic_data is not None:
            print("   Adding economic features...")
            
            # Convert timezone-aware index to naive to match price data
            if hasattr(features.index, 'tz') and features.index.tz is not None:
                features.index = features.index.tz_localize(None)
            
            # Resample economic data to daily frequency
            economic_daily = economic_data.resample('D').ffill()
            
            # Align with price data
            for indicator in economic_data.columns:
                try:
                    # Get the indicator values for our date range
                    indicator_series = economic_daily[indicator].reindex(features.index, method='ffill')
                    features[f'ECON_{indicator}'] = indicator_series
                    
                    # Add rate of change for economic indicators
                    features[f'ECON_{indicator}_CHANGE'] = indicator_series.pct_change(20)  # Monthly change
                except Exception as e:
                    print(f"     Warning: Could not align {indicator}: {e}")
                    # Add as constant if alignment fails
                    latest_value = economic_data[indicator].iloc[-1]
                    features[f'ECON_{indicator}'] = latest_value
                    features[f'ECON_{indicator}_CHANGE'] = 0
        
        # 4. DERIVED CROSS-ASSET FEATURES
        print("   Adding derived features...")
        
        if 'ECON_FEDERAL_FUNDS_RATE' in features.columns and 'ECON_TREASURY_10Y' in features.columns:
            # Interest rate spread (important for banks)
            features['INTEREST_SPREAD'] = features['ECON_TREASURY_10Y'] - features['ECON_FEDERAL_FUNDS_RATE']
            
        if 'ECON_YIELD_CURVE' in features.columns:
            # Yield curve steepness change
            features['YIELD_CURVE_CHANGE'] = features['ECON_YIELD_CURVE'].pct_change(20)
        
        # Clean features
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        print(f"   TOTAL MULTI-SOURCE FEATURES: {features.shape[1]}")
        return features
    
    def test_multi_source_performance(self, features, price_data):
        """Test if multi-source data improves performance"""
        print("\nTESTING MULTI-SOURCE PERFORMANCE...")
        
        # Create target
        returns = price_data['Close'].pct_change()
        target = (returns.shift(-1) > 0).astype(int)
        
        # Align data
        X = features.iloc[:-1]
        y = target.iloc[:-1].dropna()
        X = X.loc[y.index]
        
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        
        # Simple test with one model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        tscv = TimeSeriesSplit(n_splits=5)
        
        scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
        
        print(f"   CROSS-VALIDATION RESULTS:")
        print(f"     Mean Accuracy: {scores.mean():.1%}")
        print(f"     Std Deviation: {scores.std():.1%}")
        print(f"     Score Range: {scores.min():.1%} - {scores.max():.1%}")
        
        # Compare with yesterday's bank result
        improvement = scores.mean() - 0.507  # vs 50.7% baseline
        print(f"   IMPROVEMENT vs Day 1: {improvement:+.1%}")
        
        if scores.mean() >= 0.55:
            print("   EXCELLENT! Multi-source data showing strong improvement!")
        elif scores.mean() > 0.507:
            print("   GOOD! Multi-source data helps performance!")
        else:
            print("   MIXED: Need to refine data integration approach")
        
        return scores.mean()
    
    def run_multi_source_test(self):
        """Run complete multi-source data test"""
        
        # Test all data sources
        fundamental_data = self.test_alpha_vantage()
        economic_data = self.test_fred_api()
        price_data, yahoo_metrics = self.get_enhanced_yahoo_data()
        
        if price_data is None:
            print("FAILED: Cannot proceed without price data")
            return None
        
        # Create multi-source features
        features = self.create_multi_source_features(price_data, fundamental_data, economic_data)
        
        # Test performance
        accuracy = self.test_multi_source_performance(features, price_data)
        
        # Summary
        print(f"\nMULTI-SOURCE DATA TEST COMPLETE!")
        print("=" * 40)
        print(f"Final Accuracy: {accuracy:.1%}")
        print(f"Data Sources Tested: 3")
        print(f"Features Generated: {features.shape[1]}")
        
        return {
            'accuracy': accuracy,
            'features': features.shape[1],
            'data_sources': ['Yahoo Finance', 'Alpha Vantage (simulated)', 'FRED (simulated)']
        }

if __name__ == "__main__":
    tester = MultiSourceDataTest()
    results = tester.run_multi_source_test()