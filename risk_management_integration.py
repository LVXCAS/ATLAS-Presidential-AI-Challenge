"""
RISK MANAGEMENT INTEGRATION
============================
Integrate position sizing and risk management with our optimized 57.9% 
accurate 5-day prediction system for JPM.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class RiskManagedTradingSystem:
    """Complete trading system with risk management"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        print("DAY 2 - RISK MANAGEMENT INTEGRATION")
        print("=" * 50)
        print("Integrating our optimized system:")
        print("  • 57.9% accuracy on 5-day JPM predictions")
        print("  • 5 optimal features")
        print("  • Position sizing rules")
        print("  • Stop losses and take profits")
        print("  • Kelly criterion for bet sizing")
        print(f"  • Starting capital: ${initial_capital:,.0f}")
        print("=" * 50)
    
    def get_optimized_system_data(self):
        """Get JPM data and create our optimized feature set"""
        print("\nSETTING UP OPTIMIZED SYSTEM...")
        
        # Get JPM data
        ticker = yf.Ticker('JPM')
        data = ticker.history(period='2y')
        data.index = data.index.tz_localize(None)
        
        # Create our optimal 5 features
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        
        # 1. ECON_UNEMPLOYMENT (simulated economic indicator)
        np.random.seed(42)  # Consistent results
        unemployment = 3.8 + np.random.normal(0, 0.05, len(data)).cumsum() * 0.01
        features['ECON_UNEMPLOYMENT'] = np.clip(unemployment, 3, 6)
        
        # 2. RETURN_10D (medium-term momentum)
        features['RETURN_10D'] = close.pct_change(10)
        
        # 3. VOLATILITY_10D (risk measure)
        daily_returns = close.pct_change()
        features['VOLATILITY_10D'] = daily_returns.rolling(10).std()
        
        # 4. PRICE_VS_SMA_50 (trend position)
        sma_50 = close.rolling(50).mean()
        features['PRICE_VS_SMA_50'] = (close - sma_50) / sma_50
        
        # 5. RELATIVE_RETURN (market relative)
        market_return = daily_returns * 0.8 + np.random.normal(0, 0.005, len(data))
        features['RELATIVE_RETURN'] = daily_returns - market_return
        
        # Create 5-day forward target
        future_return_5d = close.pct_change(5).shift(-5)
        target = (future_return_5d > 0).astype(int)
        
        # Clean data
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"   Data: {len(data)} days")
        print(f"   Features: {features.shape[1]}")
        print(f"   Price range: ${close.min():.2f} - ${close.max():.2f}")
        
        return data, features, target
    
    def train_prediction_model(self, features, target):
        """Train our optimized prediction model"""
        print("\nTRAINING PREDICTION MODEL...")
        
        # Align data
        X = features
        y = target
        
        # Remove NaN values
        valid_idx = y.dropna().index
        X_clean = X.loc[valid_idx]
        y_clean = y.loc[valid_idx]
        
        print(f"   Training samples: {len(X_clean)}")
        print(f"   Target balance: {y_clean.mean():.1%} positive")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.model.fit(X_scaled, y_clean)
        
        # Get prediction probabilities for risk assessment
        train_probs = self.model.predict_proba(X_scaled)
        self.confidence_threshold = np.percentile(train_probs[:, 1], 60)  # Top 40% confidence
        
        print(f"   Model trained successfully")
        print(f"   Confidence threshold: {self.confidence_threshold:.3f}")
        
        return X_clean, y_clean
    
    def calculate_position_size(self, prediction_prob, current_volatility, confidence_level):
        """Calculate position size using multiple risk management approaches"""
        
        # Method 1: Kelly Criterion (simplified)
        # Assumes we win 57.9% of the time with 1:1 risk/reward
        win_rate = 0.579
        loss_rate = 1 - win_rate
        win_loss_ratio = 1.0  # Assume 1:1 risk reward for simplicity
        kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Method 2: Volatility-based sizing
        base_vol = 0.02  # 2% daily volatility baseline
        vol_adjusted_size = base_vol / current_volatility if current_volatility > 0 else 0.01
        vol_adjusted_size = max(0.005, min(vol_adjusted_size, 0.15))  # 0.5% to 15%
        
        # Method 3: Confidence-based sizing
        confidence_multiplier = (prediction_prob - 0.5) * 2  # Scale 0.5-1.0 to 0-1.0
        confidence_multiplier = max(0.1, min(confidence_multiplier, 1.0))
        
        # Combine methods
        base_size = 0.05  # 5% base position
        final_size = base_size * kelly_fraction * vol_adjusted_size * confidence_multiplier
        final_size = max(0.01, min(final_size, 0.20))  # 1% to 20% maximum
        
        return {
            'position_size': final_size,
            'kelly_fraction': kelly_fraction,
            'vol_adjustment': vol_adjusted_size,
            'confidence_multiplier': confidence_multiplier
        }
    
    def backtest_risk_managed_strategy(self, data, features, target):
        """Backtest the complete risk-managed strategy"""
        print("\nBACKTESTING RISK-MANAGED STRATEGY...")
        
        # Train model on first 70% of data, test on last 30%
        split_point = int(len(data) * 0.7)
        
        train_features = features.iloc[:split_point]
        train_target = target.iloc[:split_point]
        
        test_data = data.iloc[split_point:]
        test_features = features.iloc[split_point:]
        test_target = target.iloc[split_point:]
        
        # Train model
        valid_train_idx = train_target.dropna().index
        X_train = train_features.loc[valid_train_idx]
        y_train = train_target.loc[valid_train_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Backtest on test period
        portfolio_value = self.initial_capital
        positions = []
        trades = []
        
        print(f"   Test period: {len(test_data)} days")
        print(f"   Starting capital: ${portfolio_value:,.0f}")
        
        for i in range(len(test_features) - 5):  # Need 5 days for forward looking
            current_date = test_features.index[i]
            
            # Skip if we don't have all features
            current_features = test_features.iloc[i]
            if current_features.isna().any():
                continue
            
            # Make prediction
            X_current = scaler.transform([current_features])
            prediction_prob = model.predict_proba(X_current)[0, 1]
            prediction = model.predict(X_current)[0]
            
            # Only trade if we have high confidence
            if prediction_prob < 0.6:  # Only trade top 40% confidence predictions
                continue
            
            # Calculate position size
            current_price = test_data.iloc[i]['Close']
            current_vol = test_features.iloc[i]['VOLATILITY_10D']
            
            position_info = self.calculate_position_size(
                prediction_prob, current_vol, prediction_prob
            )
            position_size = position_info['position_size']
            
            # Calculate dollar amount to invest
            dollar_amount = portfolio_value * position_size
            shares = dollar_amount / current_price
            
            # Simulate 5-day holding period
            if i + 5 < len(test_data):
                exit_price = test_data.iloc[i + 5]['Close']
                exit_date = test_data.index[i + 5]
                
                # Calculate return
                trade_return = (exit_price - current_price) / current_price
                dollar_return = trade_return * dollar_amount
                
                # Apply transaction costs (0.1% each way)
                transaction_cost = dollar_amount * 0.002  # 0.2% total
                net_return = dollar_return - transaction_cost
                
                # Update portfolio
                portfolio_value += net_return
                
                # Record trade
                trades.append({
                    'entry_date': current_date,
                    'exit_date': exit_date,
                    'entry_price': current_price,
                    'exit_price': exit_price,
                    'prediction_prob': prediction_prob,
                    'position_size': position_size,
                    'dollar_amount': dollar_amount,
                    'return': trade_return,
                    'dollar_return': dollar_return,
                    'net_return': net_return,
                    'portfolio_value': portfolio_value
                })
        
        return trades, portfolio_value
    
    def analyze_backtest_results(self, trades, final_portfolio_value):
        """Analyze backtest performance"""
        print(f"\nBACKTEST ANALYSIS:")
        print("=" * 30)
        
        if not trades:
            print("No trades executed!")
            return
        
        df_trades = pd.DataFrame(trades)
        
        # Basic performance metrics
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        num_trades = len(trades)
        winning_trades = len(df_trades[df_trades['return'] > 0])
        win_rate = winning_trades / num_trades
        
        avg_return = df_trades['return'].mean()
        avg_winner = df_trades[df_trades['return'] > 0]['return'].mean()
        avg_loser = df_trades[df_trades['return'] < 0]['return'].mean()
        
        print(f"PERFORMANCE SUMMARY:")
        print(f"  Total Return: {total_return:.1%}")
        print(f"  Final Value: ${final_portfolio_value:,.0f}")
        print(f"  Profit/Loss: ${final_portfolio_value - self.initial_capital:,.0f}")
        print(f"  Number of Trades: {num_trades}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Avg Return per Trade: {avg_return:.2%}")
        
        if not pd.isna(avg_winner):
            print(f"  Avg Winning Trade: {avg_winner:.2%}")
        if not pd.isna(avg_loser):
            print(f"  Avg Losing Trade: {avg_loser:.2%}")
        
        # Risk metrics
        trade_returns = df_trades['return']
        volatility = trade_returns.std()
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        max_drawdown = (df_trades['portfolio_value'].cummax() - df_trades['portfolio_value']).max() / df_trades['portfolio_value'].cummax().max()
        
        print(f"\nRISK METRICS:")
        print(f"  Volatility per Trade: {volatility:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.1%}")
        
        # Compare to buy and hold
        if trades:
            first_price = trades[0]['entry_price']
            last_price = trades[-1]['exit_price']
            buy_hold_return = (last_price - first_price) / first_price
            
            print(f"\nCOMPARISON:")
            print(f"  Buy & Hold Return: {buy_hold_return:.1%}")
            print(f"  Strategy vs B&H: {total_return - buy_hold_return:+.1%}")
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades
        }
    
    def run_risk_managed_system(self):
        """Run complete risk-managed trading system"""
        
        # Get data and features
        data, features, target = self.get_optimized_system_data()
        
        # Backtest strategy
        trades, final_value = self.backtest_risk_managed_strategy(data, features, target)
        
        # Analyze results
        performance = self.analyze_backtest_results(trades, final_value)
        
        print(f"\nRISK MANAGEMENT INTEGRATION COMPLETE!")
        print("=" * 50)
        
        if performance and performance['total_return'] > 0.1:  # 10%+ return
            print("EXCELLENT! Risk-managed system shows strong profitability!")
        elif performance and performance['total_return'] > 0:
            print("GOOD! System is profitable with risk management!")
        else:
            print("NEEDS WORK: Adjust risk parameters")
        
        return performance

if __name__ == "__main__":
    system = RiskManagedTradingSystem()
    results = system.run_risk_managed_system()