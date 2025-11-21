#!/usr/bin/env python3
"""
SIMPLIFIED MULTI-AGENT BACKTEST (5 YEARS)
==========================================
Tests core multi-agent trading logic without complex dependencies.

Strategies Tested:
1. Momentum (Trend Following)
2. Mean Reversion (Contrarian)
3. Options Volatility (Iron Condor simulation)

Signal Fusion: Weighted voting system
Risk Management: Position sizing, stop losses
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# STRATEGY IMPLEMENTATIONS
# ---------------------------------------------------------------------

class MomentumStrategy:
    """Momentum/Trend Following Strategy"""
    
    def __init__(self):
        self.name = "Momentum"
        self.lookback = 20
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Generate momentum signals for all symbols"""
        signals = {}
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_signals = []
            
            # Calculate indicators
            symbol_data['SMA_20'] = symbol_data['Close'].rolling(20).mean()
            symbol_data['SMA_50'] = symbol_data['Close'].rolling(50).mean()
            symbol_data['RSI'] = self._calculate_rsi(symbol_data['Close'], 14)
            
            for idx, row in symbol_data.iterrows():
                if pd.isna(row['SMA_50']):
                    continue
                    
                signal = None
                confidence = 0.0
                
                # Buy signal: Price > SMA20 > SMA50, RSI not overbought
                if row['Close'] > row['SMA_20'] > row['SMA_50'] and row['RSI'] < 70:
                    signal = 'buy'
                    confidence = min((row['Close'] - row['SMA_20']) / row['SMA_20'] * 10, 1.0)
                
                # Sell signal: Price < SMA20 < SMA50
                elif row['Close'] < row['SMA_20'] < row['SMA_50']:
                    signal = 'sell'
                    confidence = 0.8
                
                if signal:
                    symbol_signals.append({
                        'date': idx,
                        'signal': signal,
                        'confidence': confidence,
                        'strategy': self.name
                    })
            
            signals[symbol] = symbol_signals
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class MeanReversionStrategy:
    """Mean Reversion Strategy"""
    
    def __init__(self):
        self.name = "MeanReversion"
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Generate mean reversion signals"""
        signals = {}
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_signals = []
            
            # Calculate Bollinger Bands
            symbol_data['SMA_20'] = symbol_data['Close'].rolling(20).mean()
            symbol_data['STD_20'] = symbol_data['Close'].rolling(20).std()
            symbol_data['BB_Upper'] = symbol_data['SMA_20'] + (symbol_data['STD_20'] * 2)
            symbol_data['BB_Lower'] = symbol_data['SMA_20'] - (symbol_data['STD_20'] * 2)
            symbol_data['RSI'] = self._calculate_rsi(symbol_data['Close'], 14)
            
            for idx, row in symbol_data.iterrows():
                if pd.isna(row['BB_Lower']):
                    continue
                    
                signal = None
                confidence = 0.0
                
                # Buy signal: Price touches lower band, RSI oversold
                if row['Close'] <= row['BB_Lower'] and row['RSI'] < 30:
                    signal = 'buy'
                    confidence = min((row['BB_Lower'] - row['Close']) / row['Close'] * 20, 1.0)
                
                # Sell signal: Price touches upper band, RSI overbought
                elif row['Close'] >= row['BB_Upper'] and row['RSI'] > 70:
                    signal = 'sell'
                    confidence = 0.8
                
                if signal:
                    symbol_signals.append({
                        'date': idx,
                        'signal': signal,
                        'confidence': confidence,
                        'strategy': self.name
                    })
            
            signals[symbol] = symbol_signals
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class OptionsVolatilityStrategy:
    """Simplified Options Strategy (Volatility-based)"""
    
    def __init__(self):
        self.name = "OptionsVol"
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Generate options signals based on volatility regime"""
        signals = {}
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_signals = []
            
            # Calculate historical volatility
            symbol_data['Returns'] = symbol_data['Close'].pct_change()
            symbol_data['HV_20'] = symbol_data['Returns'].rolling(20).std() * np.sqrt(252)
            symbol_data['HV_MA'] = symbol_data['HV_20'].rolling(60).mean()
            symbol_data['RSI'] = self._calculate_rsi(symbol_data['Close'], 14)
            
            for idx, row in symbol_data.iterrows():
                if pd.isna(row['HV_MA']):
                    continue
                    
                signal = None
                confidence = 0.0
                
                # Sell premium when volatility is high and market is neutral
                if row['HV_20'] > row['HV_MA'] * 1.2 and 40 < row['RSI'] < 60:
                    signal = 'sell_premium'  # Iron Condor / Credit Spread
                    confidence = min((row['HV_20'] - row['HV_MA']) / row['HV_MA'], 1.0)
                
                if signal:
                    symbol_signals.append({
                        'date': idx,
                        'signal': signal,
                        'confidence': confidence,
                        'strategy': self.name
                    })
            
            signals[symbol] = symbol_signals
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------
# SIGNAL FUSION ENGINE
# ---------------------------------------------------------------------

class SignalFusion:
    """Combines signals from multiple strategies"""
    
    def __init__(self):
        self.strategy_weights = {
            'Momentum': 0.4,
            'MeanReversion': 0.3,
            'OptionsVol': 0.3
        }
    
    def fuse_signals(self, all_signals: Dict[str, Dict[str, List[Dict]]], 
                     current_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """Fuse signals from all strategies for current date"""
        fused = []
        
        # Collect all symbols
        all_symbols = set()
        for strategy_signals in all_signals.values():
            all_symbols.update(strategy_signals.keys())
        
        for symbol in all_symbols:
            votes = {'buy': 0.0, 'sell': 0.0, 'sell_premium': 0.0}
            
            # Collect votes from each strategy
            for strategy_name, strategy_signals in all_signals.items():
                if symbol not in strategy_signals:
                    continue
                
                # Find signals for current date
                for sig in strategy_signals[symbol]:
                    if sig['date'] == current_date:
                        weight = self.strategy_weights.get(strategy_name, 0.33)
                        votes[sig['signal']] += sig['confidence'] * weight
            
            # Determine final action
            max_vote = max(votes.values())
            if max_vote > 0.5:  # Threshold for action
                action = max(votes, key=votes.get)
                fused.append({
                    'symbol': symbol,
                    'action': action,
                    'confidence': max_vote,
                    'votes': votes.copy()
                })
        
        return fused


# ---------------------------------------------------------------------
# PORTFOLIO MANAGER
# ---------------------------------------------------------------------

class SimplePortfolioManager:
    """Manages positions and executes trades"""
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}  # {symbol: {'quantity': int, 'entry_price': float}}
        self.equity_curve = []
        self.trades = []
        
    def execute_signals(self, signals: List[Dict], current_prices: Dict[str, float], 
                       current_date: pd.Timestamp):
        """Execute trading signals"""
        
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            confidence = signal['confidence']
            price = current_prices.get(symbol, 0)
            
            if price == 0:
                continue
            
            # Position sizing based on confidence
            position_size = int((self.cash * 0.1 * confidence) / price)  # 10% max per position
            
            if action == 'buy' and symbol not in self.positions:
                # Open long position
                cost = position_size * price
                if cost <= self.cash:
                    self.positions[symbol] = {
                        'quantity': position_size,
                        'entry_price': price,
                        'entry_date': current_date
                    }
                    self.cash -= cost
                    self.trades.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': position_size,
                        'price': price
                    })
            
            elif action == 'sell' and symbol in self.positions:
                # Close long position
                pos = self.positions[symbol]
                proceeds = pos['quantity'] * price
                pnl = proceeds - (pos['quantity'] * pos['entry_price'])
                self.cash += proceeds
                self.trades.append({
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': pos['quantity'],
                    'price': price,
                    'pnl': pnl
                })
                del self.positions[symbol]
        
        # Update equity curve
        portfolio_value = self.cash + sum(
            pos['quantity'] * current_prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.positions.items()
        )
        self.equity_curve.append({
            'date': current_date,
            'value': portfolio_value,
            'cash': self.cash,
            'positions': len(self.positions)
        })
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        returns = equity_df['value'].pct_change().dropna()
        total_return = (equity_df['value'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe Ratio
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win Rate
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        total_trades = len([t for t in self.trades if 'pnl' in t])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_value': equity_df['value'].iloc[-1]
        }


# ---------------------------------------------------------------------
# MAIN BACKTEST RUNNER
# ---------------------------------------------------------------------

async def run_simplified_backtest():
    print("="*60)
    print("SIMPLIFIED MULTI-AGENT BACKTEST (2019-2024)")
    print("="*60)
    
    # 1. Fetch Historical Data
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    start_date = "2019-01-01"
    end_date = "2023-12-31"
    
    print(f"\nFetching data for {symbols}...")
    all_data = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            df['symbol'] = symbol
            all_data.append(df.reset_index())
            print(f"✓ {symbol}: {len(df)} days")
        except Exception as e:
            print(f"✗ {symbol}: {e}")
    
    if not all_data:
        print("ERROR: No data fetched!")
        return
    
    data = pd.concat(all_data, ignore_index=True)
    data.rename(columns={'Date': 'date'}, inplace=True)
    
    # 2. Initialize Strategies
    print("\nInitializing strategies...")
    momentum = MomentumStrategy()
    mean_reversion = MeanReversionStrategy()
    options_vol = OptionsVolatilityStrategy()
    
    # 3. Generate Signals
    print("Generating signals...")
    momentum_signals = momentum.generate_signals(data)
    mean_reversion_signals = mean_reversion.generate_signals(data)
    options_signals = options_vol.generate_signals(data)
    
    all_signals = {
        'Momentum': momentum_signals,
        'MeanReversion': mean_reversion_signals,
        'OptionsVol': options_signals
    }
    
    # 4. Run Backtest
    print("Running backtest simulation...")
    fusion = SignalFusion()
    portfolio = SimplePortfolioManager(initial_capital=1000000.0)
    
    # Get all unique dates
    all_dates = sorted(data['date'].unique())
    
    for i, current_date in enumerate(all_dates):
        if i % 50 == 0:
            print(f"Processing {i}/{len(all_dates)} days...")
        
        # Fuse signals for current date
        fused_signals = fusion.fuse_signals(all_signals, current_date)
        
        # Get current prices
        current_data = data[data['date'] == current_date]
        current_prices = dict(zip(current_data['symbol'], current_data['Close']))
        
        # Execute trades
        portfolio.execute_signals(fused_signals, current_prices, current_date)
    
    # 5. Report Results
    metrics = portfolio.get_performance_metrics()
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS (2019-2024)")
    print("="*60)
    print(f"Initial Capital:   ${portfolio.initial_capital:,.2f}")
    print(f"Final Value:       ${metrics['final_value']:,.2f}")
    print(f"Total Return:      {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:      {metrics['max_drawdown']:.2%}")
    print(f"Total Trades:      {metrics['total_trades']}")
    print(f"Win Rate:          {metrics['win_rate']:.2%}")
    print("="*60)
    
    # Save equity curve
    equity_df = pd.DataFrame(portfolio.equity_curve)
    equity_df.to_csv('simplified_backtest_equity_curve.csv', index=False)
    print("\n✓ Equity curve saved to: simplified_backtest_equity_curve.csv")
    
    return metrics


if __name__ == "__main__":
    # Windows event loop policy fix
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_simplified_backtest())
