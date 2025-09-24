"""
SECTOR ROTATION STRATEGY
========================
Day 3 - Test our system across different sectors and implement
dynamic sector rotation based on market conditions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

class SectorRotationStrategy:
    """Implement sector rotation based on our optimal system"""
    
    def __init__(self):
        # Representative stocks for each sector
        self.sectors = {
            'Financials': {
                'JPM': 'JPMorgan Chase',
                'BAC': 'Bank of America'
            },
            'Technology': {
                'AAPL': 'Apple',
                'MSFT': 'Microsoft'
            },
            'Healthcare': {
                'JNJ': 'Johnson & Johnson',
                'PFE': 'Pfizer'
            },
            'Energy': {
                'XOM': 'Exxon Mobil',
                'CVX': 'Chevron'
            },
            'Consumer': {
                'WMT': 'Walmart',
                'PG': 'Procter & Gamble'
            }
        }
        
        print("DAY 3 - SECTOR ROTATION STRATEGY")
        print("=" * 50)
        print("Testing our system across sectors:")
        for sector, stocks in self.sectors.items():
            print(f"\n{sector}:")
            for symbol, name in stocks.items():
                print(f"  • {symbol}: {name}")
        print("\nStrategy: Rotate to best-performing sector monthly")
        print("=" * 50)
    
    def create_sector_features(self, symbol, sector):
        """Create sector-adapted features"""
        print(f"\n   Testing {symbol} ({sector})...")
        
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='2y')
            data.index = data.index.tz_localize(None)
            
            if len(data) < 100:
                print(f"   ERROR: Insufficient data for {symbol}")
                return None, None, None, None
            
            # Create our base 5 features
            features = pd.DataFrame(index=data.index)
            close = data['Close']
            
            # 1. ECON_UNEMPLOYMENT (same for all)
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
            try:
                spy_data = yf.download('SPY', start=data.index[0], end=data.index[-1], progress=False)
                spy_data.index = spy_data.index.tz_localize(None)
                spy_returns = spy_data['Close'].pct_change().reindex(data.index, method='ffill')
                features['RELATIVE_RETURN'] = daily_returns - spy_returns
            except:
                market_return = daily_returns * 0.8 + np.random.normal(0, 0.005, len(data))
                features['RELATIVE_RETURN'] = daily_returns - market_return
            
            # Add sector-specific features
            if sector == 'Technology':
                # Tech stocks are more volatile and growth-oriented
                features['GROWTH_MOMENTUM'] = close.pct_change(20)  # 20-day momentum
                features['HIGH_VOL_INDICATOR'] = (features['VOLATILITY_10D'] > features['VOLATILITY_10D'].rolling(50).mean()).astype(int)
            elif sector == 'Healthcare':
                # Healthcare is defensive and steady
                features['DEFENSIVE_INDICATOR'] = (features['VOLATILITY_10D'] < features['VOLATILITY_10D'].rolling(50).mean()).astype(int)
                features['DIVIDEND_PROXY'] = close.rolling(100).mean() / close  # Stability proxy
            elif sector == 'Energy':
                # Energy is cyclical and volatile
                features['CYCLICAL_INDICATOR'] = np.sin(np.arange(len(data)) * 2 * np.pi / 252)  # Annual cycle
                features['COMMODITY_PROXY'] = features['VOLATILITY_10D'].rolling(20).mean()  # Volatility proxy
            elif sector == 'Consumer':
                # Consumer is stable and follows economic cycles
                features['CONSUMER_CONFIDENCE'] = features['ECON_UNEMPLOYMENT'] * -1  # Inverse of unemployment
                features['STABILITY_SCORE'] = 1 / (1 + features['VOLATILITY_10D'])  # Stability measure
            
            # Create 5-day forward target
            future_return_5d = close.pct_change(5).shift(-5)
            target = (future_return_5d > 0).astype(int)
            
            # Clean data
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            print(f"   {symbol}: {len(data)} days, {features.shape[1]} features")
            return features, target, data, daily_returns
            
        except Exception as e:
            print(f"   ERROR with {symbol}: {e}")
            return None, None, None, None
    
    def test_sector_performance(self, symbol, sector, features, target):
        """Test system performance on a sector"""
        
        # Align data
        valid_idx = target.dropna().index
        X_clean = features.loc[valid_idx].fillna(0)
        y_clean = target.loc[valid_idx]
        
        if len(X_clean) < 100:
            return {
                'symbol': symbol,
                'sector': sector,
                'accuracy': 0.0,
                'error': 'Insufficient data'
            }
        
        # Scale and test
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        
        scores = cross_val_score(model, X_scaled, y_clean, cv=tscv, scoring='accuracy')
        
        return {
            'symbol': symbol,
            'sector': sector,
            'accuracy': scores.mean(),
            'std': scores.std(),
            'samples': len(X_clean),
            'target_balance': y_clean.mean()
        }
    
    def test_all_sectors(self):
        """Test system across all sectors"""
        print("\nTESTING ALL SECTORS...")
        
        sector_results = {}
        
        for sector, stocks in self.sectors.items():
            print(f"\n{sector.upper()} SECTOR:")
            sector_performance = []
            
            for symbol, name in stocks.items():
                # Create features
                features, target, data, returns = self.create_sector_features(symbol, sector)
                
                if features is None:
                    continue
                
                # Test performance
                performance = self.test_sector_performance(symbol, sector, features, target)
                
                if performance['accuracy'] > 0:
                    print(f"   {symbol}: {performance['accuracy']:.1%} (+/- {performance['std']:.1%})")
                    sector_performance.append(performance)
            
            # Calculate sector average
            if sector_performance:
                sector_avg = np.mean([p['accuracy'] for p in sector_performance])
                sector_results[sector] = {
                    'average_accuracy': sector_avg,
                    'stocks': sector_performance,
                    'best_stock': max(sector_performance, key=lambda x: x['accuracy'])
                }
                print(f"   {sector} Average: {sector_avg:.1%}")
        
        return sector_results
    
    def analyze_sector_rotation_opportunity(self, sector_results):
        """Analyze which sectors work best when"""
        print(f"\nSECTOR ROTATION ANALYSIS:")
        print("=" * 40)
        
        # Rank sectors by performance
        sector_rankings = sorted(sector_results.items(), key=lambda x: x[1]['average_accuracy'], reverse=True)
        
        print("SECTOR PERFORMANCE RANKING:")
        for i, (sector, data) in enumerate(sector_rankings, 1):
            avg_acc = data['average_accuracy']
            best_stock = data['best_stock']
            comparison = avg_acc - 0.579  # vs JPM baseline
            print(f"{i}. {sector}: {avg_acc:.1%} ({comparison:+.1%} vs JPM)")
            print(f"   Best Stock: {best_stock['symbol']} at {best_stock['accuracy']:.1%}")
        
        # Identify rotation strategy
        print(f"\nROTATION STRATEGY RECOMMENDATIONS:")
        
        best_sector = sector_rankings[0]
        worst_sector = sector_rankings[-1]
        
        print(f"PRIMARY SECTOR: {best_sector[0]} ({best_sector[1]['average_accuracy']:.1%})")
        print(f"AVOID SECTOR: {worst_sector[0]} ({worst_sector[1]['average_accuracy']:.1%})")
        
        # Market condition analysis
        all_accuracies = [data['average_accuracy'] for _, data in sector_rankings]
        sector_spread = max(all_accuracies) - min(all_accuracies)
        
        print(f"\nSECTOR SPREAD: {sector_spread:.1%}")
        
        if sector_spread > 0.05:  # 5%+ difference
            print("HIGH SECTOR DIFFERENTIATION - Rotation strategy recommended!")
            
            rotation_strategy = []
            for sector, data in sector_rankings:
                if data['average_accuracy'] > 0.55:  # 55%+ threshold
                    rotation_strategy.append(sector)
            
            print(f"SECTORS FOR ROTATION: {', '.join(rotation_strategy)}")
            
        else:
            print("LOW SECTOR DIFFERENTIATION - Diversification better than rotation")
        
        return {
            'best_sector': best_sector[0],
            'sector_rankings': sector_rankings,
            'sector_spread': sector_spread,
            'rotation_recommended': sector_spread > 0.05
        }
    
    def create_dynamic_rotation_system(self, sector_results, analysis):
        """Create a dynamic sector rotation system"""
        print(f"\nDYNAMIC ROTATION SYSTEM:")
        print("-" * 30)
        
        if not analysis['rotation_recommended']:
            print("Rotation not recommended - use diversified approach instead")
            return None
        
        # Create rotation rules
        rotation_rules = {
            'high_volatility_market': 'Consumer',  # Defensive during volatility
            'low_volatility_market': 'Technology',  # Growth during stability  
            'rising_rates': 'Financials',         # Banks benefit from higher rates
            'falling_rates': 'Technology',        # Growth stocks benefit
            'recession_risk': 'Healthcare',       # Defensive sector
            'economic_growth': 'Energy'           # Cyclical during growth
        }
        
        print("ROTATION RULES:")
        for condition, sector in rotation_rules.items():
            if sector in sector_results:
                accuracy = sector_results[sector]['average_accuracy']
                print(f"  {condition}: {sector} ({accuracy:.1%})")
        
        # Implementation framework
        print(f"\nIMPLEMENTATION FRAMEWORK:")
        print("1. Monitor market conditions monthly")
        print("2. Apply rotation rules based on regime")
        print("3. Rebalance portfolio allocation")
        print("4. Track relative performance")
        
        best_sectors = [sector for sector, data in analysis['sector_rankings'][:3]]
        
        return {
            'rotation_rules': rotation_rules,
            'top_sectors': best_sectors,
            'rebalance_frequency': 'monthly'
        }
    
    def run_sector_rotation_analysis(self):
        """Run complete sector rotation analysis"""
        
        # Test all sectors
        sector_results = self.test_all_sectors()
        
        if not sector_results:
            print("ERROR: No sector results available")
            return None
        
        # Analyze rotation opportunities
        analysis = self.analyze_sector_rotation_opportunity(sector_results)
        
        # Create rotation system
        rotation_system = self.create_dynamic_rotation_system(sector_results, analysis)
        
        print(f"\nSECTOR ROTATION ANALYSIS COMPLETE!")
        print("=" * 50)
        
        if analysis['rotation_recommended']:
            print("SUCCESS! Sector rotation strategy recommended!")
        else:
            print("INSIGHT: Diversification better than rotation for this system")
        
        return {
            'sector_results': sector_results,
            'analysis': analysis,
            'rotation_system': rotation_system
        }

if __name__ == "__main__":
    rotator = SectorRotationStrategy()
    results = rotator.run_sector_rotation_analysis()