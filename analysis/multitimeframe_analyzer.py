"""
Hive Trade Multi-Timeframe Strategy Analysis
Advanced analysis across multiple timeframes for optimal strategy selection
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeAnalyzer:
    """Advanced multi-timeframe strategy analysis system"""
    
    def __init__(self):
        self.timeframes = {
            '1d': '1 day',
            '5d': '5 days', 
            '1wk': '1 week',
            '1mo': '1 month',
            '3mo': '3 months'
        }
        self.strategies = {}
        self.results = {}
        
    def fetch_multitimeframe_data(self, symbol: str, period: str = '2y') -> Dict[str, pd.DataFrame]:
        """Fetch data for all timeframes"""
        print(f"Fetching multi-timeframe data for {symbol}...")
        
        data = {}
        ticker = yf.Ticker(symbol)
        
        for tf_key, tf_name in self.timeframes.items():
            try:
                df = ticker.history(period=period, interval=tf_key)
                if not df.empty:
                    data[tf_key] = df
                    print(f"  {tf_key}: {len(df)} periods")
            except Exception as e:
                print(f"  Failed to fetch {tf_key}: {e}")
                
        return data
    
    def calculate_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calculate technical indicators for specific timeframe"""
        result = df.copy()
        
        # Trend indicators
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
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        # Volatility
        result['volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Volume indicators (if available)
        if 'Volume' in df.columns:
            result['vol_sma'] = df['Volume'].rolling(20).mean()
            result['vol_ratio'] = df['Volume'] / result['vol_sma']
        
        # Price momentum
        result['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        result['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        result['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        return result
    
    def momentum_strategy(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Enhanced momentum strategy for specific timeframe"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['position'] = 0
        
        # Multi-factor momentum conditions
        conditions = {
            'price_trend': df['Close'] > df['sma_20'],
            'momentum_positive': df['momentum_10'] > 0.02,  # 2% threshold
            'macd_bullish': df['macd'] > df['macd_signal'],
            'rsi_oversold': df['rsi'] < 70,  # Not overbought
            'volume_confirmation': df.get('vol_ratio', 1) > 1.2  # Above average volume
        }
        
        # Long signals
        long_condition = (
            conditions['price_trend'] & 
            conditions['momentum_positive'] & 
            conditions['macd_bullish'] & 
            conditions['rsi_oversold'] &
            conditions['volume_confirmation']
        )
        
        signals.loc[long_condition, 'signal'] = 1
        
        # Exit conditions
        exit_condition = (
            (df['Close'] < df['sma_20']) |
            (df['rsi'] > 80) |
            (df['macd'] < df['macd_signal'])
        )
        
        signals.loc[exit_condition, 'signal'] = -1
        
        # Calculate positions
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return signals
    
    def mean_reversion_strategy(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Enhanced mean reversion strategy for specific timeframe"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['position'] = 0
        
        # Mean reversion conditions
        conditions = {
            'price_below_bb': df['Close'] < df['bb_lower'],
            'price_above_bb': df['Close'] > df['bb_upper'],
            'rsi_oversold': df['rsi'] < 30,
            'rsi_overbought': df['rsi'] > 70,
            'high_volatility': df['bb_width'] > df['bb_width'].rolling(50).quantile(0.8)
        }
        
        # Long signals (oversold)
        long_condition = (
            conditions['price_below_bb'] & 
            conditions['rsi_oversold'] &
            conditions['high_volatility']
        )
        
        # Short signals (overbought)  
        short_condition = (
            conditions['price_above_bb'] & 
            conditions['rsi_overbought'] &
            conditions['high_volatility']
        )
        
        signals.loc[long_condition, 'signal'] = 1
        signals.loc[short_condition, 'signal'] = -1
        
        # Exit at mean
        exit_condition = (
            (df['Close'] > df['bb_middle']) & (signals['signal'].shift(1) == 1) |  # Exit long
            (df['Close'] < df['bb_middle']) & (signals['signal'].shift(1) == -1)    # Exit short
        )
        
        signals.loc[exit_condition, 'signal'] = 0
        
        # Calculate positions
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return signals
    
    def breakout_strategy(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Enhanced breakout strategy for specific timeframe"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['position'] = 0
        
        # Calculate breakout levels
        lookback = {'1d': 20, '5d': 10, '1wk': 8, '1mo': 6, '3mo': 4}.get(timeframe, 20)
        
        df['resistance'] = df['High'].rolling(lookback).max()
        df['support'] = df['Low'].rolling(lookback).min()
        df['range_width'] = (df['resistance'] - df['support']) / df['Close']
        
        # Breakout conditions
        conditions = {
            'upward_breakout': df['Close'] > df['resistance'].shift(1),
            'downward_breakout': df['Close'] < df['support'].shift(1),
            'volume_surge': df.get('vol_ratio', 1) > 1.5,
            'narrow_range': df['range_width'] < df['range_width'].rolling(50).quantile(0.3),
            'momentum_confirmation': abs(df['momentum_5']) > 0.01
        }
        
        # Long breakout
        long_condition = (
            conditions['upward_breakout'] &
            conditions['volume_surge'] &
            conditions['momentum_confirmation']
        )
        
        # Short breakout
        short_condition = (
            conditions['downward_breakout'] &
            conditions['volume_surge'] &
            conditions['momentum_confirmation']
        )
        
        signals.loc[long_condition, 'signal'] = 1
        signals.loc[short_condition, 'signal'] = -1
        
        # Stop loss
        stop_loss_pct = {'1d': 0.02, '5d': 0.03, '1wk': 0.05, '1mo': 0.08, '3mo': 0.12}.get(timeframe, 0.05)
        
        # Calculate positions with stop loss
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return signals
    
    def calculate_performance_metrics(self, df: pd.DataFrame, signals: pd.DataFrame, 
                                    strategy_name: str, timeframe: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        # Calculate returns
        returns = df['Close'].pct_change()
        strategy_returns = returns * signals['position'].shift(1)
        
        # Remove NaN and infinite values
        strategy_returns = strategy_returns.dropna().replace([np.inf, -np.inf], 0)
        
        if len(strategy_returns) == 0:
            return {'error': 'No valid returns'}
        
        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        position_changes = signals['position'].diff().abs()
        num_trades = (position_changes > 0).sum()
        winning_trades = (strategy_returns > 0).sum()
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        return {
            'strategy': strategy_name,
            'timeframe': timeframe,
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'num_trades': int(num_trades),
            'win_rate': float(win_rate),
            'periods_analyzed': len(strategy_returns)
        }
    
    def run_multitimeframe_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Run comprehensive multi-timeframe analysis"""
        
        print("\n" + "="*60)
        print("HIVE TRADE MULTI-TIMEFRAME ANALYSIS")
        print("="*60)
        
        all_results = {}
        
        strategies = {
            'momentum': self.momentum_strategy,
            'mean_reversion': self.mean_reversion_strategy,
            'breakout': self.breakout_strategy
        }
        
        for symbol in symbols:
            print(f"\nAnalyzing {symbol}...")
            symbol_results = {}
            
            # Fetch multi-timeframe data
            timeframe_data = self.fetch_multitimeframe_data(symbol)
            
            if not timeframe_data:
                print(f"No data available for {symbol}")
                continue
            
            for timeframe, df in timeframe_data.items():
                print(f"\n  Timeframe: {timeframe}")
                
                # Calculate technical indicators
                df_with_indicators = self.calculate_technical_indicators(df, timeframe)
                
                timeframe_results = {}
                
                for strategy_name, strategy_func in strategies.items():
                    try:
                        # Generate signals
                        signals = strategy_func(df_with_indicators, timeframe)
                        
                        # Calculate performance
                        metrics = self.calculate_performance_metrics(
                            df_with_indicators, signals, strategy_name, timeframe
                        )
                        
                        timeframe_results[strategy_name] = metrics
                        
                        if 'error' not in metrics:
                            print(f"    {strategy_name}: Return={metrics['total_return']:.2%}, "
                                  f"Sharpe={metrics['sharpe_ratio']:.2f}, Trades={metrics['num_trades']}")
                        
                    except Exception as e:
                        print(f"    {strategy_name}: Error - {e}")
                        timeframe_results[strategy_name] = {'error': str(e)}
                
                symbol_results[timeframe] = timeframe_results
            
            all_results[symbol] = symbol_results
        
        return all_results
    
    def generate_analysis_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        
        report = []
        report.append("MULTI-TIMEFRAME STRATEGY ANALYSIS REPORT")
        report.append("="*50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        all_metrics = []
        for symbol, symbol_data in results.items():
            for timeframe, tf_data in symbol_data.items():
                for strategy, metrics in tf_data.items():
                    if 'error' not in metrics:
                        metrics['symbol'] = symbol
                        all_metrics.append(metrics)
        
        if all_metrics:
            df_metrics = pd.DataFrame(all_metrics)
            
            report.append("OVERALL PERFORMANCE SUMMARY:")
            report.append("-" * 30)
            
            # Best performing strategies
            best_sharpe = df_metrics.loc[df_metrics['sharpe_ratio'].idxmax()]
            best_return = df_metrics.loc[df_metrics['total_return'].idxmax()]
            
            report.append(f"Best Sharpe Ratio: {best_sharpe['strategy']} ({best_sharpe['symbol']}, {best_sharpe['timeframe']}) = {best_sharpe['sharpe_ratio']:.2f}")
            report.append(f"Best Total Return: {best_return['strategy']} ({best_return['symbol']}, {best_return['timeframe']}) = {best_return['total_return']:.2%}")
            report.append("")
            
            # Strategy performance by timeframe
            report.append("PERFORMANCE BY TIMEFRAME:")
            report.append("-" * 30)
            
            for tf in self.timeframes.keys():
                tf_metrics = df_metrics[df_metrics['timeframe'] == tf]
                if not tf_metrics.empty:
                    avg_return = tf_metrics['total_return'].mean()
                    avg_sharpe = tf_metrics['sharpe_ratio'].mean()
                    report.append(f"{tf}: Avg Return={avg_return:.2%}, Avg Sharpe={avg_sharpe:.2f}")
            
            report.append("")
            
            # Strategy performance across all timeframes
            report.append("STRATEGY PERFORMANCE SUMMARY:")
            report.append("-" * 30)
            
            for strategy in ['momentum', 'mean_reversion', 'breakout']:
                strategy_metrics = df_metrics[df_metrics['strategy'] == strategy]
                if not strategy_metrics.empty:
                    avg_return = strategy_metrics['total_return'].mean()
                    avg_sharpe = strategy_metrics['sharpe_ratio'].mean()
                    win_rate = strategy_metrics['win_rate'].mean()
                    report.append(f"{strategy.title()}: Return={avg_return:.2%}, Sharpe={avg_sharpe:.2f}, Win Rate={win_rate:.1%}")
        
        report.append("")
        report.append("DETAILED RESULTS BY SYMBOL:")
        report.append("="*40)
        
        # Detailed results
        for symbol, symbol_data in results.items():
            report.append(f"\n{symbol}:")
            report.append("-" * 20)
            
            for timeframe, tf_data in symbol_data.items():
                report.append(f"  {timeframe}:")
                
                for strategy, metrics in tf_data.items():
                    if 'error' in metrics:
                        report.append(f"    {strategy}: ERROR - {metrics['error']}")
                    else:
                        report.append(f"    {strategy}:")
                        report.append(f"      Total Return: {metrics['total_return']:.2%}")
                        report.append(f"      Annual Return: {metrics['annual_return']:.2%}")
                        report.append(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                        report.append(f"      Max Drawdown: {metrics['max_drawdown']:.2%}")
                        report.append(f"      Win Rate: {metrics['win_rate']:.1%}")
                        report.append(f"      Trades: {metrics['num_trades']}")
        
        return "\n".join(report)

def main():
    """Run multi-timeframe analysis"""
    analyzer = MultiTimeframeAnalyzer()
    
    # Analyze key symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
    
    # Run analysis
    results = analyzer.run_multitimeframe_analysis(symbols)
    
    # Generate report
    report = analyzer.generate_analysis_report(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    results_file = f"multitimeframe_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for symbol, symbol_data in results.items():
            json_results[symbol] = {}
            for tf, tf_data in symbol_data.items():
                json_results[symbol][tf] = {}
                for strategy, metrics in tf_data.items():
                    if isinstance(metrics, dict):
                        clean_metrics = {}
                        for k, v in metrics.items():
                            if isinstance(v, (np.integer, np.floating)):
                                clean_metrics[k] = float(v) if isinstance(v, np.floating) else int(v)
                            else:
                                clean_metrics[k] = v
                        json_results[symbol][tf][strategy] = clean_metrics
                    else:
                        json_results[symbol][tf][strategy] = metrics
        
        json.dump(json_results, f, indent=2)
    
    # Save report
    report_file = f"multitimeframe_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nResults saved:")
    print(f"- JSON: {results_file}")
    print(f"- Report: {report_file}")
    
    print("\nMULTI-TIMEFRAME ANALYSIS COMPLETE!")
    print("="*50)
    
    return results

if __name__ == "__main__":
    main()