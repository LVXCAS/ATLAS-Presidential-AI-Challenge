#!/usr/bin/env python3
"""
Backtrader Engine
Professional backtesting with custom strategies and advanced analytics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Try to import backtrader
try:
    import backtrader as bt
    import backtrader.indicators as btind
    import backtrader.feeds as btfeeds
    BACKTRADER_AVAILABLE = True
    print("+ Backtrader available for professional backtesting")
    
    class HiveStrategy(bt.Strategy):
        """Custom Hive Trading Strategy for Backtrader"""
        
        params = (
            ('rsi_period', 14),
            ('rsi_upper', 70),
            ('rsi_lower', 30),
            ('ma_fast', 10),
            ('ma_slow', 30),
            ('atr_period', 14),
            ('stop_loss', 0.02),
            ('take_profit', 0.04)
        )
        
        def __init__(self):
            # Technical indicators
            self.rsi = btind.RSI_Safe(self.data.close, period=self.params.rsi_period)
            self.ma_fast = btind.SMA(self.data.close, period=self.params.ma_fast)
            self.ma_slow = btind.SMA(self.data.close, period=self.params.ma_slow)
            self.atr = btind.ATR(self.data, period=self.params.atr_period)
            
            # Order variables
            self.order = None
            self.entry_price = None
            
        def next(self):
            if self.order:
                return  # Skip if order pending
            
            if not self.position:
                # Entry conditions
                if (self.rsi < self.params.rsi_lower and 
                    self.ma_fast > self.ma_slow and
                    self.data.close[0] > self.ma_fast[0]):
                    
                    # Calculate position size based on ATR
                    risk_amount = self.broker.getcash() * 0.02  # 2% risk
                    atr_stop = self.atr[0] * 2
                    if atr_stop > 0:
                        size = risk_amount / atr_stop
                        self.order = self.buy(size=int(size))
                        self.entry_price = self.data.close[0]
            else:
                # Exit conditions
                current_price = self.data.close[0]
                if self.entry_price:
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                    
                    # Take profit or stop loss
                    if (pnl_pct >= self.params.take_profit or 
                        pnl_pct <= -self.params.stop_loss or
                        self.rsi > self.params.rsi_upper):
                        self.order = self.close()
        
        def notify_order(self, order):
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.entry_price = order.executed.price
                else:
                    self.entry_price = None
            self.order = None
    
    class MomentumStrategy(bt.Strategy):
        """Momentum-based trading strategy"""
        
        params = (
            ('momentum_period', 10),
            ('ma_period', 20),
            ('momentum_threshold', 0.02)
        )
        
        def __init__(self):
            self.momentum = btind.Momentum(self.data.close, period=self.params.momentum_period)
            self.ma = btind.SMA(self.data.close, period=self.params.ma_period)
            self.order = None
        
        def next(self):
            if self.order:
                return
            
            momentum_pct = self.momentum[0] / self.data.close[-self.params.momentum_period]
            
            if not self.position:
                if (momentum_pct > self.params.momentum_threshold and 
                    self.data.close[0] > self.ma[0]):
                    self.order = self.buy()
            else:
                if momentum_pct < -self.params.momentum_threshold:
                    self.order = self.close()
        
        def notify_order(self, order):
            self.order = None if order.status in [order.Completed] else self.order
    
    class MeanReversionStrategy(bt.Strategy):
        """Mean reversion trading strategy"""
        
        params = (
            ('bb_period', 20),
            ('bb_std', 2),
            ('rsi_period', 14),
            ('rsi_overbought', 80),
            ('rsi_oversold', 20)
        )
        
        def __init__(self):
            self.bb = btind.BollingerBands(self.data.close, period=self.params.bb_period, devfactor=self.params.bb_std)
            self.rsi = btind.RSI_Safe(self.data.close, period=self.params.rsi_period)
            self.order = None
        
        def next(self):
            if self.order:
                return
            
            if not self.position:
                # Buy when oversold
                if (self.data.close[0] < self.bb.lines.bot[0] and 
                    self.rsi[0] < self.params.rsi_oversold):
                    self.order = self.buy()
            else:
                # Sell when overbought
                if (self.data.close[0] > self.bb.lines.top[0] or 
                    self.rsi[0] > self.params.rsi_overbought):
                    self.order = self.close()
        
        def notify_order(self, order):
            self.order = None if order.status in [order.Completed] else self.order

except ImportError:
    BACKTRADER_AVAILABLE = False
    print("- Backtrader not available - using custom implementation")
    
    # Dummy classes when Backtrader is not available
    class HiveStrategy:
        def __init__(self):
            pass
    
    class MomentumStrategy:
        def __init__(self):
            pass
    
    class MeanReversionStrategy:
        def __init__(self):
            pass

class BacktraderEngine:
    """Professional backtesting engine using Backtrader"""
    
    def __init__(self):
        self.backtrader_available = BACKTRADER_AVAILABLE
        self.results_cache = {}
        
        if BACKTRADER_AVAILABLE:
            print("+ Backtrader Engine initialized with professional strategies")
        else:
            print("+ Backtrader Engine initialized with custom implementation")
    
    async def run_backtest(self, data: pd.DataFrame, strategy: str = 'hive_strategy', 
                          initial_cash: float = 100000) -> Dict:
        """Run backtest with specified strategy"""
        
        if self.backtrader_available:
            return await self._run_backtrader_backtest(data, strategy, initial_cash)
        else:
            return await self._run_custom_backtest(data, strategy, initial_cash)
    
    async def _run_backtrader_backtest(self, data: pd.DataFrame, strategy: str, initial_cash: float) -> Dict:
        """Run backtest using Backtrader"""
        try:
            # Create cerebro engine
            cerebro = bt.Cerebro()
            
            # Set initial cash
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
            
            # Add strategy
            strategy_map = {
                'hive_strategy': HiveStrategy,
                'momentum': MomentumStrategy,
                'mean_reversion': MeanReversionStrategy
            }
            
            selected_strategy = strategy_map.get(strategy, HiveStrategy)
            cerebro.addstrategy(selected_strategy)
            
            # Prepare data
            data_feed = bt.feeds.PandasData(
                dataname=data,
                datetime=None,
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=None
            )
            cerebro.adddata(data_feed)
            
            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # Run backtest
            results = cerebro.run()
            final_value = cerebro.broker.getvalue()
            
            # Extract results
            strategy_result = results[0]
            
            # Get analyzer results
            sharpe_ratio = strategy_result.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            drawdown = strategy_result.analyzers.drawdown.get_analysis()
            returns_analysis = strategy_result.analyzers.returns.get_analysis()
            trades_analysis = strategy_result.analyzers.trades.get_analysis()
            
            return {
                'strategy_name': strategy,
                'initial_value': initial_cash,
                'final_value': final_value,
                'total_return': (final_value - initial_cash) / initial_cash,
                'sharpe_ratio': sharpe_ratio if sharpe_ratio else 0,
                'max_drawdown': drawdown.get('max', {}).get('drawdown', 0) / 100,
                'total_trades': trades_analysis.get('total', {}).get('total', 0),
                'winning_trades': trades_analysis.get('won', {}).get('total', 0),
                'losing_trades': trades_analysis.get('lost', {}).get('total', 0),
                'win_rate': trades_analysis.get('won', {}).get('total', 0) / max(1, trades_analysis.get('total', {}).get('total', 1)),
                'engine': 'backtrader'
            }
            
        except Exception as e:
            print(f"- Backtrader backtest error: {e}")
            return await self._run_custom_backtest(data, strategy, initial_cash)
    
    async def _run_custom_backtest(self, data: pd.DataFrame, strategy: str, initial_cash: float) -> Dict:
        """Custom backtest implementation"""
        try:
            print(f"Running custom backtest for {strategy}")
            
            # Simple buy and hold strategy
            entry_price = data['close'].iloc[0]
            exit_price = data['close'].iloc[-1]
            
            total_return = (exit_price - entry_price) / entry_price
            final_value = initial_cash * (1 + total_return)
            
            # Calculate simple metrics
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (total_return * 252) / (volatility * np.sqrt(252)) if volatility > 0 else 0
            
            # Drawdown calculation
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'strategy_name': strategy,
                'initial_value': initial_cash,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': 2,  # Buy and sell
                'winning_trades': 1 if total_return > 0 else 0,
                'losing_trades': 1 if total_return < 0 else 0,
                'win_rate': 1.0 if total_return > 0 else 0.0,
                'engine': 'custom'
            }
            
        except Exception as e:
            print(f"- Custom backtest error: {e}")
            return {}
    
    async def compare_strategies(self, data: pd.DataFrame, strategies: List[str] = None) -> pd.DataFrame:
        """Compare multiple strategies"""
        
        if strategies is None:
            strategies = ['hive_strategy', 'momentum', 'mean_reversion']
        
        results = []
        
        for strategy in strategies:
            try:
                result = await self.run_backtest(data, strategy)
                if result:
                    results.append({
                        'Strategy': result['strategy_name'],
                        'Total Return': result['total_return'],
                        'Sharpe Ratio': result['sharpe_ratio'],
                        'Max Drawdown': result['max_drawdown'],
                        'Win Rate': result['win_rate'],
                        'Total Trades': result['total_trades'],
                        'Final Value': result['final_value'],
                        'Engine': result['engine']
                    })
            except Exception as e:
                print(f"- Error comparing strategy {strategy}: {e}")
        
        if results:
            df = pd.DataFrame(results)
            return df.sort_values('Sharpe Ratio', ascending=False)
        else:
            return pd.DataFrame()
    
    async def optimize_strategy_parameters(self, data: pd.DataFrame, strategy: str = 'hive_strategy') -> Dict:
        """Optimize strategy parameters"""
        
        if not self.backtrader_available:
            return {'error': 'Parameter optimization requires Backtrader'}
        
        try:
            best_result = None
            best_sharpe = -999
            
            # Simple parameter optimization (would be more sophisticated in practice)
            if strategy == 'hive_strategy':
                rsi_periods = [10, 14, 20]
                ma_fasts = [5, 10, 15]
                
                for rsi_period in rsi_periods:
                    for ma_fast in ma_fasts:
                        # Create cerebro with parameters
                        cerebro = bt.Cerebro()
                        cerebro.broker.setcash(100000)
                        
                        # Add strategy with parameters
                        cerebro.addstrategy(HiveStrategy, 
                                          rsi_period=rsi_period,
                                          ma_fast=ma_fast)
                        
                        # Add data
                        data_feed = bt.feeds.PandasData(
                            dataname=data,
                            datetime=None,
                            open='open',
                            high='high', 
                            low='low',
                            close='close',
                            volume='volume',
                            openinterest=None
                        )
                        cerebro.adddata(data_feed)
                        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                        
                        # Run optimization
                        results = cerebro.run()
                        sharpe = results[0].analyzers.sharpe.get_analysis().get('sharperatio', 0)
                        
                        if sharpe and sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_result = {
                                'rsi_period': rsi_period,
                                'ma_fast': ma_fast,
                                'sharpe_ratio': sharpe,
                                'final_value': cerebro.broker.getvalue()
                            }
            
            return best_result or {'error': 'No optimization results'}
            
        except Exception as e:
            print(f"- Strategy optimization error: {e}")
            return {'error': str(e)}

# Create global instance
backtrader_engine = BacktraderEngine()