"""
Hive Trade Strategy Ensemble Methods
Advanced ensemble techniques to combine multiple trading strategies
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class StrategyEnsemble:
    """Advanced strategy ensemble system"""
    
    def __init__(self):
        self.strategies = {}
        self.ensemble_models = {}
        self.scalers = {}
        self.weights = {}
        self.performance_history = {}
        
    def load_strategy_signals(self, symbol: str, timeframe: str = '1d', period: str = '2y') -> pd.DataFrame:
        """Load and generate signals from individual strategies"""
        
        # Fetch market data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=timeframe)
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Calculate technical indicators
        df = self.calculate_indicators(df)
        
        # Generate signals from different strategies
        signals_df = pd.DataFrame(index=df.index)
        
        # Strategy 1: Momentum
        signals_df['momentum_signal'] = self.momentum_strategy_signals(df)
        
        # Strategy 2: Mean Reversion
        signals_df['mean_reversion_signal'] = self.mean_reversion_strategy_signals(df)
        
        # Strategy 3: Breakout
        signals_df['breakout_signal'] = self.breakout_strategy_signals(df)
        
        # Strategy 4: RSI-based
        signals_df['rsi_signal'] = self.rsi_strategy_signals(df)
        
        # Strategy 5: MACD-based
        signals_df['macd_signal'] = self.macd_strategy_signals(df)
        
        # Strategy 6: Bollinger Bands
        signals_df['bb_signal'] = self.bollinger_bands_strategy_signals(df)
        
        # Add price and volume data for ensemble learning
        signals_df['price'] = df['Close']
        signals_df['volume'] = df['Volume']
        signals_df['returns'] = df['Close'].pct_change()
        signals_df['volatility'] = df['Close'].pct_change().rolling(20).std()
        
        return signals_df.dropna()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        result = df.copy()
        
        # Moving averages
        result['sma_20'] = df['Close'].rolling(20).mean()
        result['sma_50'] = df['Close'].rolling(50).mean()
        result['ema_12'] = df['Close'].ewm(span=12).mean()
        result['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        result['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        result['bb_upper'] = result['bb_middle'] + (bb_std * 2)
        result['bb_lower'] = result['bb_middle'] - (bb_std * 2)
        
        # Stochastic
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        result['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        result['stoch_d'] = result['stoch_k'].rolling(3).mean()
        
        # Price momentum
        result['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        result['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        return result
    
    def momentum_strategy_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate momentum strategy signals"""
        signals = pd.Series(index=df.index, dtype=int)
        signals[:] = 0  # Hold
        
        # Buy conditions
        buy_condition = (
            (df['Close'] > df['sma_20']) &
            (df['momentum_10'] > 0.02) &
            (df['macd'] > df['macd_signal']) &
            (df['rsi'] < 70)
        )
        
        # Sell conditions  
        sell_condition = (
            (df['Close'] < df['sma_20']) &
            (df['momentum_10'] < -0.02) &
            (df['macd'] < df['macd_signal']) &
            (df['rsi'] > 30)
        )
        
        signals[buy_condition] = 1   # Buy
        signals[sell_condition] = -1 # Sell
        
        return signals
    
    def mean_reversion_strategy_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate mean reversion strategy signals"""
        signals = pd.Series(index=df.index, dtype=int)
        signals[:] = 0
        
        # Buy oversold
        buy_condition = (
            (df['Close'] < df['bb_lower']) &
            (df['rsi'] < 30) &
            (df['stoch_k'] < 20)
        )
        
        # Sell overbought
        sell_condition = (
            (df['Close'] > df['bb_upper']) &
            (df['rsi'] > 70) &
            (df['stoch_k'] > 80)
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def breakout_strategy_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate breakout strategy signals"""
        signals = pd.Series(index=df.index, dtype=int)
        signals[:] = 0
        
        # Calculate support/resistance
        resistance = df['High'].rolling(20).max()
        support = df['Low'].rolling(20).min()
        
        # Upward breakout
        buy_condition = (
            (df['Close'] > resistance.shift(1)) &
            (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5)
        )
        
        # Downward breakout
        sell_condition = (
            (df['Close'] < support.shift(1)) &
            (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5)
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def rsi_strategy_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate RSI-based strategy signals"""
        signals = pd.Series(index=df.index, dtype=int)
        signals[:] = 0
        
        signals[df['rsi'] < 30] = 1   # Oversold - Buy
        signals[df['rsi'] > 70] = -1  # Overbought - Sell
        
        return signals
    
    def macd_strategy_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate MACD-based strategy signals"""
        signals = pd.Series(index=df.index, dtype=int)
        signals[:] = 0
        
        # MACD bullish crossover
        macd_bullish = (
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        # MACD bearish crossover
        macd_bearish = (
            (df['macd'] < df['macd_signal']) &
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        signals[macd_bullish] = 1
        signals[macd_bearish] = -1
        
        return signals
    
    def bollinger_bands_strategy_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate Bollinger Bands strategy signals"""
        signals = pd.Series(index=df.index, dtype=int)
        signals[:] = 0
        
        signals[df['Close'] < df['bb_lower']] = 1   # Buy at lower band
        signals[df['Close'] > df['bb_upper']] = -1  # Sell at upper band
        
        return signals
    
    def create_ensemble_features(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ensemble learning"""
        features = pd.DataFrame(index=signals_df.index)
        
        # Strategy signals as features
        strategy_cols = [col for col in signals_df.columns if 'signal' in col]
        for col in strategy_cols:
            features[col] = signals_df[col]
        
        # Market condition features
        features['volatility'] = signals_df['volatility']
        features['volume_ratio'] = signals_df['volume'] / signals_df['volume'].rolling(20).mean()
        features['price_momentum'] = signals_df['returns'].rolling(5).mean()
        features['price_acceleration'] = signals_df['returns'].diff()
        
        # Consensus features
        features['bullish_consensus'] = (signals_df[strategy_cols] == 1).sum(axis=1)
        features['bearish_consensus'] = (signals_df[strategy_cols] == -1).sum(axis=1)
        features['signal_diversity'] = signals_df[strategy_cols].std(axis=1)
        
        return features.dropna()
    
    def create_ensemble_labels(self, signals_df: pd.DataFrame, forward_days: int = 3) -> pd.Series:
        """Create labels for ensemble learning based on future returns"""
        
        # Calculate forward returns
        forward_returns = signals_df['price'].shift(-forward_days) / signals_df['price'] - 1
        
        # Create labels: 0=sell, 1=hold, 2=buy
        labels = pd.Series(index=signals_df.index, dtype=int)
        labels[forward_returns > 0.01] = 2     # Buy (>1% expected return)
        labels[forward_returns < -0.01] = 0    # Sell (<-1% expected return)  
        labels[(forward_returns >= -0.01) & (forward_returns <= 0.01)] = 1  # Hold
        
        return labels
    
    def train_ensemble_models(self, symbols: List[str]) -> Dict[str, Any]:
        """Train ensemble models on multiple symbols"""
        
        print("\nTraining strategy ensemble models...")
        print("="*40)
        
        all_features = []
        all_labels = []
        
        # Collect training data from multiple symbols
        for symbol in symbols:
            try:
                print(f"Processing {symbol}...")
                
                # Load strategy signals
                signals_df = self.load_strategy_signals(symbol)
                
                # Create features and labels
                features = self.create_ensemble_features(signals_df)
                labels = self.create_ensemble_labels(signals_df)
                
                # Align features and labels
                common_index = features.index.intersection(labels.index)
                features_aligned = features.loc[common_index]
                labels_aligned = labels.loc[common_index]
                
                all_features.append(features_aligned)
                all_labels.append(labels_aligned)
                
                print(f"  {len(features_aligned)} samples collected")
                
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid data collected for training")
        
        # Combine all data
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_labels, axis=0)
        
        print(f"\nTotal training samples: {len(X)}")
        print(f"Feature columns: {list(X.columns)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble models
        models = {
            'voting_soft': VotingClassifier([
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('svm', SVC(probability=True, random_state=42))
            ], voting='soft'),
            
            'voting_hard': VotingClassifier([
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('svm', SVC(random_state=42))
            ], voting='hard'),
            
            'random_forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"  Accuracy: {accuracy:.3f}")
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred.tolist(),
                    'actual': y_test.tolist()
                }
                
                # Store model and scaler
                self.ensemble_models[model_name] = model
                
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                continue
        
        # Store scaler
        self.scalers['ensemble'] = scaler
        
        return results
    
    def adaptive_weight_ensemble(self, symbols: List[str], lookback_days: int = 30) -> Dict[str, Any]:
        """Create adaptive weight ensemble based on recent performance"""
        
        print(f"\nCreating adaptive weight ensemble (lookback: {lookback_days} days)...")
        print("="*50)
        
        strategy_performance = {}
        
        for symbol in symbols:
            try:
                print(f"Analyzing {symbol}...")
                
                # Get recent data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=f"{lookback_days*2}d")
                
                if len(df) < lookback_days:
                    continue
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Get recent period for evaluation
                recent_df = df.tail(lookback_days).copy()
                
                # Generate signals for each strategy
                strategies = {
                    'momentum': self.momentum_strategy_signals(recent_df),
                    'mean_reversion': self.mean_reversion_strategy_signals(recent_df),
                    'breakout': self.breakout_strategy_signals(recent_df),
                    'rsi': self.rsi_strategy_signals(recent_df),
                    'macd': self.macd_strategy_signals(recent_df),
                    'bollinger_bands': self.bollinger_bands_strategy_signals(recent_df)
                }
                
                # Calculate performance for each strategy
                returns = recent_df['Close'].pct_change()
                
                for strategy_name, signals in strategies.items():
                    strategy_returns = returns * signals.shift(1)
                    total_return = (1 + strategy_returns.dropna()).prod() - 1
                    
                    if strategy_name not in strategy_performance:
                        strategy_performance[strategy_name] = []
                    strategy_performance[strategy_name].append(total_return)
                
                print(f"  Analyzed {len(strategies)} strategies")
                
            except Exception as e:
                print(f"  Error analyzing {symbol}: {e}")
                continue
        
        # Calculate average performance and weights
        weights = {}
        performance_summary = {}
        
        for strategy, performances in strategy_performance.items():
            avg_performance = np.mean(performances)
            performance_summary[strategy] = {
                'avg_return': avg_performance,
                'std_return': np.std(performances),
                'num_symbols': len(performances)
            }
        
        # Convert performance to weights (softmax-like transformation)
        if performance_summary:
            performances = [perf['avg_return'] for perf in performance_summary.values()]
            # Shift to positive domain
            min_perf = min(performances)
            shifted_perfs = [p - min_perf + 0.01 for p in performances]
            
            # Normalize to weights
            total = sum(shifted_perfs)
            strategy_names = list(performance_summary.keys())
            
            for i, strategy in enumerate(strategy_names):
                weights[strategy] = shifted_perfs[i] / total
        
        self.weights = weights
        
        print("\nAdaptive Weights:")
        for strategy, weight in weights.items():
            perf = performance_summary[strategy]['avg_return']
            print(f"  {strategy}: {weight:.3f} (performance: {perf:.2%})")
        
        return {
            'weights': weights,
            'performance_summary': performance_summary,
            'lookback_days': lookback_days
        }
    
    def generate_ensemble_signal(self, symbol: str, method: str = 'adaptive_weight') -> int:
        """Generate ensemble trading signal for a symbol"""
        
        try:
            # Load current market data
            signals_df = self.load_strategy_signals(symbol, period='3mo')
            latest_signals = signals_df.iloc[-1]
            
            if method == 'adaptive_weight':
                # Weighted combination
                if not self.weights:
                    raise ValueError("Adaptive weights not calculated. Run adaptive_weight_ensemble first.")
                
                weighted_signal = 0
                strategy_signals = {
                    'momentum': latest_signals['momentum_signal'],
                    'mean_reversion': latest_signals['mean_reversion_signal'],
                    'breakout': latest_signals['breakout_signal'],
                    'rsi': latest_signals['rsi_signal'],
                    'macd': latest_signals['macd_signal'],
                    'bollinger_bands': latest_signals['bb_signal']
                }
                
                for strategy, signal in strategy_signals.items():
                    if strategy in self.weights:
                        weighted_signal += signal * self.weights[strategy]
                
                # Convert to discrete signal
                if weighted_signal > 0.3:
                    return 1   # Buy
                elif weighted_signal < -0.3:
                    return -1  # Sell
                else:
                    return 0   # Hold
            
            elif method == 'majority_vote':
                # Simple majority voting
                signals = [
                    latest_signals['momentum_signal'],
                    latest_signals['mean_reversion_signal'],
                    latest_signals['breakout_signal'],
                    latest_signals['rsi_signal'],
                    latest_signals['macd_signal'],
                    latest_signals['bb_signal']
                ]
                
                buy_votes = sum(1 for s in signals if s == 1)
                sell_votes = sum(1 for s in signals if s == -1)
                
                if buy_votes > sell_votes:
                    return 1
                elif sell_votes > buy_votes:
                    return -1
                else:
                    return 0
            
            elif method == 'ml_ensemble':
                # Machine learning ensemble
                if 'voting_soft' not in self.ensemble_models:
                    raise ValueError("ML ensemble not trained. Run train_ensemble_models first.")
                
                # Prepare features
                features = self.create_ensemble_features(signals_df)
                latest_features = features.iloc[-1].values.reshape(1, -1)
                
                # Scale features
                if 'ensemble' in self.scalers:
                    latest_features_scaled = self.scalers['ensemble'].transform(latest_features)
                    
                    # Get prediction
                    prediction = self.ensemble_models['voting_soft'].predict(latest_features_scaled)[0]
                    
                    # Convert to signal: 0=sell, 1=hold, 2=buy
                    if prediction == 2:
                        return 1   # Buy
                    elif prediction == 0:
                        return -1  # Sell
                    else:
                        return 0   # Hold
                else:
                    raise ValueError("Scaler not available for ML ensemble")
            
        except Exception as e:
            print(f"Error generating ensemble signal for {symbol}: {e}")
            return 0  # Default to hold
    
    def backtest_ensemble(self, symbols: List[str], method: str = 'adaptive_weight', 
                         period: str = '1y') -> Dict[str, Any]:
        """Backtest ensemble strategy"""
        
        print(f"\nBacktesting ensemble strategy ({method})...")
        print("="*45)
        
        results = {}
        
        for symbol in symbols:
            try:
                print(f"Backtesting {symbol}...")
                
                # Get historical data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                
                if len(df) < 50:
                    continue
                
                # Initialize results
                signals = []
                returns = []
                
                # Rolling backtest (using 3-month windows)
                window_size = 60  # ~3 months
                
                for i in range(window_size, len(df)):
                    # Get window data
                    window_df = df.iloc[max(0, i-window_size):i].copy()
                    
                    # Calculate indicators
                    window_df = self.calculate_indicators(window_df)
                    
                    if len(window_df) < 20:  # Minimum data for indicators
                        signals.append(0)
                        continue
                    
                    # Generate individual strategy signals
                    latest_data = window_df.iloc[-1]
                    
                    strategy_signals = {
                        'momentum': self.momentum_strategy_signals(window_df).iloc[-1],
                        'mean_reversion': self.mean_reversion_strategy_signals(window_df).iloc[-1],
                        'breakout': self.breakout_strategy_signals(window_df).iloc[-1],
                        'rsi': self.rsi_strategy_signals(window_df).iloc[-1],
                        'macd': self.macd_strategy_signals(window_df).iloc[-1],
                        'bollinger_bands': self.bollinger_bands_strategy_signals(window_df).iloc[-1]
                    }
                    
                    # Generate ensemble signal
                    if method == 'majority_vote':
                        signal_values = list(strategy_signals.values())
                        buy_votes = sum(1 for s in signal_values if s == 1)
                        sell_votes = sum(1 for s in signal_values if s == -1)
                        
                        if buy_votes > sell_votes:
                            ensemble_signal = 1
                        elif sell_votes > buy_votes:
                            ensemble_signal = -1
                        else:
                            ensemble_signal = 0
                    
                    elif method == 'adaptive_weight':
                        if self.weights:
                            weighted_signal = sum(
                                strategy_signals.get(strategy, 0) * weight 
                                for strategy, weight in self.weights.items()
                            )
                            
                            if weighted_signal > 0.3:
                                ensemble_signal = 1
                            elif weighted_signal < -0.3:
                                ensemble_signal = -1
                            else:
                                ensemble_signal = 0
                        else:
                            ensemble_signal = 0
                    
                    signals.append(ensemble_signal)
                
                # Calculate returns
                df_signals = df.iloc[window_size:].copy()
                df_signals['signal'] = signals[:len(df_signals)]
                df_signals['returns'] = df_signals['Close'].pct_change()
                df_signals['strategy_returns'] = df_signals['returns'] * df_signals['signal'].shift(1)
                
                # Calculate performance metrics
                strategy_returns = df_signals['strategy_returns'].dropna()
                
                if len(strategy_returns) > 0:
                    total_return = (1 + strategy_returns).prod() - 1
                    annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
                    volatility = strategy_returns.std() * np.sqrt(252)
                    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                    
                    # Drawdown
                    cumulative_returns = (1 + strategy_returns).cumprod()
                    running_max = cumulative_returns.cummax()
                    drawdown = (cumulative_returns - running_max) / running_max
                    max_drawdown = drawdown.min()
                    
                    # Trade stats
                    num_trades = (df_signals['signal'].diff().abs() > 0).sum()
                    winning_trades = (strategy_returns > 0).sum()
                    win_rate = winning_trades / len(strategy_returns) if len(strategy_returns) > 0 else 0
                    
                    results[symbol] = {
                        'total_return': float(total_return),
                        'annual_return': float(annual_return),
                        'volatility': float(volatility),
                        'sharpe_ratio': float(sharpe_ratio),
                        'max_drawdown': float(max_drawdown),
                        'num_trades': int(num_trades),
                        'win_rate': float(win_rate),
                        'periods': len(strategy_returns)
                    }
                    
                    print(f"  Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, Trades: {num_trades}")
                
            except Exception as e:
                print(f"  Error backtesting {symbol}: {e}")
                continue
        
        return results

def main():
    """Run strategy ensemble analysis"""
    
    print("HIVE TRADE STRATEGY ENSEMBLE METHODS")
    print("="*40)
    
    ensemble = StrategyEnsemble()
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
    
    # 1. Train ML ensemble models
    print("\n1. TRAINING ML ENSEMBLE MODELS")
    print("-" * 30)
    try:
        ml_results = ensemble.train_ensemble_models(symbols)
        print("\nML Ensemble Training Results:")
        for model_name, results in ml_results.items():
            if 'accuracy' in results:
                print(f"  {model_name}: {results['accuracy']:.3f} accuracy")
    except Exception as e:
        print(f"ML ensemble training failed: {e}")
    
    # 2. Create adaptive weight ensemble
    print("\n2. ADAPTIVE WEIGHT ENSEMBLE")
    print("-" * 30)
    try:
        adaptive_results = ensemble.adaptive_weight_ensemble(symbols)
        print(f"Adaptive ensemble created with {len(adaptive_results['weights'])} strategies")
    except Exception as e:
        print(f"Adaptive ensemble failed: {e}")
    
    # 3. Generate current signals
    print("\n3. CURRENT ENSEMBLE SIGNALS")
    print("-" * 30)
    current_signals = {}
    
    for symbol in symbols:
        try:
            signal = ensemble.generate_ensemble_signal(symbol, method='adaptive_weight')
            current_signals[symbol] = signal
            
            signal_text = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}.get(signal, 'UNKNOWN')
            print(f"  {symbol}: {signal_text}")
            
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
    
    # 4. Backtest ensemble strategies
    print("\n4. ENSEMBLE BACKTESTING")
    print("-" * 30)
    
    backtest_methods = ['majority_vote', 'adaptive_weight']
    backtest_results = {}
    
    for method in backtest_methods:
        try:
            print(f"\nBacktesting {method}...")
            results = ensemble.backtest_ensemble(symbols, method=method)
            backtest_results[method] = results
            
            # Calculate average performance
            if results:
                avg_return = np.mean([r['total_return'] for r in results.values()])
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in results.values()])
                print(f"Average Return: {avg_return:.2%}")
                print(f"Average Sharpe: {avg_sharpe:.2f}")
            
        except Exception as e:
            print(f"Backtesting {method} failed: {e}")
    
    # 5. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    ensemble_results = {
        'ml_ensemble': ml_results if 'ml_results' in locals() else {},
        'adaptive_weights': adaptive_results if 'adaptive_results' in locals() else {},
        'current_signals': current_signals,
        'backtest_results': backtest_results,
        'timestamp': timestamp
    }
    
    # Save to file
    results_file = f"ensemble_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        # Clean results for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    if k == 'model':  # Skip model objects
                        continue
                    elif isinstance(v, (list, dict)):
                        cleaned[k] = clean_for_json(v)
                    elif isinstance(v, (np.integer, np.floating)):
                        cleaned[k] = float(v) if isinstance(v, np.floating) else int(v)
                    else:
                        cleaned[k] = v
                return cleaned
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            else:
                return obj
        
        clean_results = clean_for_json(ensemble_results)
        json.dump(clean_results, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    print("\nSTRATEGY ENSEMBLE ANALYSIS COMPLETE!")
    print("="*40)
    
    return ensemble_results

if __name__ == "__main__":
    main()