# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# PC-HIVE-TRADING Integration with QuantConnect LEAN
# Enhanced with Live Trading Strategies from your existing system

from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os

class HiveTradingAlgorithm(QCAlgorithm):
    """
    PC-HIVE-TRADING Enhanced Algorithm for QuantConnect LEAN
    
    Integrates your existing trading strategies:
    - Adaptive Momentum (AAPL)
    - Breakout Confirmation (SPY) 
    - Mean Reversion (QQQ)
    
    Features:
    - Professional risk management
    - Multi-strategy portfolio
    - Real-time signal generation
    - Performance optimization
    """
    
    def Initialize(self):
        """Initialize the algorithm"""
        
        # Set date range and starting cash
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        
        # Set benchmark
        self.SetBenchmark("SPY")
        
        # Trading universe
        self.symbols = {
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust',
            'AAPL': 'Apple Inc.'
        }
        
        # Add equity data
        self.equity_symbols = {}
        for symbol, name in self.symbols.items():
            equity = self.AddEquity(symbol, Resolution.Minute)
            equity.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
            self.equity_symbols[symbol] = equity.Symbol
        
        # Strategy parameters (from your deployed strategies)
        self.strategy_params = {
            'adaptive_momentum_aapl': {
                'fast_period': 12,
                'slow_period': 50,
                'volatility_lookback': 20,
                'regime_threshold': 1.0,
                'symbol': 'AAPL',
                'allocation': 0.33
            },
            'breakout_momentum_spy': {
                'breakout_period': 20,
                'volume_threshold': 1.5,
                'confirmation_period': 3,
                'atr_multiplier': 2.0,
                'symbol': 'SPY',
                'allocation': 0.33
            },
            'mean_reversion_qqq': {
                'lookback_period': 30,
                'threshold_multiplier': 2.0,
                'volatility_window': 10,
                'ml_confidence_threshold': 0.7,
                'symbol': 'QQQ',
                'allocation': 0.34
            }
        }
        
        # Risk management (from your existing system)
        self.risk_limits = {
            'max_position_size': 0.15,      # 15% max position (enhanced from 5%)
            'stop_loss_pct': 0.02,          # 2% stop loss
            'take_profit_pct': 0.06,        # 6% take profit
            'max_daily_trades': 10,         # Max trades per day
            'max_drawdown': 0.20            # 20% max drawdown
        }
        
        # Trading state
        self.daily_trades = 0
        self.positions = {}
        self.entry_prices = {}
        self.trade_count = 0
        
        # Technical indicators
        self.indicators = {}
        self._setup_indicators()
        
        # Schedule functions
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.ResetDailyTrades
        )
        
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.Every(TimeSpan.FromMinutes(5)),
            self.CheckExitConditions
        )
        
        # Set universe selection
        self.SetUniverseSelection(ManualUniverseSelectionModel([
            Symbol.Create(symbol, SecurityType.Equity, Market.USA) 
            for symbol in self.symbols.keys()
        ]))
        
        # Set alpha model (signal generation)
        self.SetAlpha(HiveTradingAlphaModel(self))
        
        # Set portfolio construction
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        
        # Set execution model
        self.SetExecution(ImmediateExecutionModel())
        
        # Set risk management
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(self.risk_limits['max_drawdown']))
        
        self.Debug("HiveTrading Algorithm Initialized Successfully!")
        self.Debug(f"Trading Universe: {list(self.symbols.keys())}")
    
    def _setup_indicators(self):
        """Setup technical indicators for all symbols"""
        for symbol in self.symbols.keys():
            equity_symbol = self.equity_symbols[symbol]
            
            self.indicators[symbol] = {
                # Moving averages
                'sma_5': self.SMA(equity_symbol, 5, Resolution.Daily),
                'sma_20': self.SMA(equity_symbol, 20, Resolution.Daily),
                'sma_50': self.SMA(equity_symbol, 50, Resolution.Daily),
                
                # Volatility
                'atr': self.ATR(equity_symbol, 14, Resolution.Daily),
                'bb': self.BB(equity_symbol, 20, 2, Resolution.Daily),
                
                # Momentum
                'rsi': self.RSI(equity_symbol, 14, Resolution.Daily),
                'macd': self.MACD(equity_symbol, 12, 26, 9, Resolution.Daily),
                
                # Volume
                'volume_sma': self.SMA(equity_symbol, 20, Resolution.Daily, Field.Volume)
            }
            
            # Warm up indicators
            history = self.History(equity_symbol, 60, Resolution.Daily)
            for indicator in self.indicators[symbol].values():
                if hasattr(indicator, 'Update'):
                    for bar in history.itertuples():
                        if hasattr(bar, 'close'):
                            indicator.Update(bar.Index, bar.close)
                        elif hasattr(bar, 'volume'):
                            indicator.Update(bar.Index, bar.volume)
    
    def OnData(self, data):
        """Main trading logic"""
        # Update indicators
        for symbol in self.symbols.keys():
            if symbol not in data:
                continue
                
            equity_symbol = self.equity_symbols[symbol]
            if equity_symbol in data:
                bar = data[equity_symbol]
                
                # Update price-based indicators
                for name, indicator in self.indicators[symbol].items():
                    if name != 'volume_sma' and hasattr(indicator, 'Update'):
                        indicator.Update(bar.Time, bar.Close)
                
                # Update volume indicator
                if 'volume_sma' in self.indicators[symbol]:
                    self.indicators[symbol]['volume_sma'].Update(bar.Time, bar.Volume)
        
        # Generate and execute trading signals
        self._execute_strategy_signals(data)
    
    def _execute_strategy_signals(self, data):
        """Execute trading signals from all strategies"""
        
        for strategy_name, params in self.strategy_params.items():
            symbol = params['symbol']
            
            if symbol not in data or not self.indicators[symbol]['sma_20'].IsReady:
                continue
            
            # Get signal from appropriate strategy
            if 'adaptive_momentum' in strategy_name:
                signal = self._adaptive_momentum_signal(symbol, params)
            elif 'breakout_momentum' in strategy_name:
                signal = self._breakout_signal(symbol, params)
            elif 'mean_reversion' in strategy_name:
                signal = self._mean_reversion_signal(symbol, params)
            else:
                continue
            
            # Execute signal
            if signal != 0 and self._validate_trade(symbol):
                self._execute_trade(symbol, signal, params)
    
    def _adaptive_momentum_signal(self, symbol: str, params: Dict) -> int:
        """Adaptive momentum strategy signal (AAPL)"""
        try:
            indicators = self.indicators[symbol]
            
            if not all([indicators['sma_5'].IsReady, indicators['sma_50'].IsReady]):
                return 0
            
            fast_sma = indicators['sma_5'].Current.Value
            slow_sma = indicators['sma_50'].Current.Value
            volatility = indicators['atr'].Current.Value
            
            # Adaptive threshold based on volatility
            threshold = volatility * params['regime_threshold']
            momentum_score = (fast_sma - slow_sma) / slow_sma
            
            if momentum_score > threshold:
                return 1  # Buy signal
            elif momentum_score < -threshold:
                return -1  # Sell signal
            else:
                return 0
                
        except Exception as e:
            self.Debug(f"Adaptive momentum error for {symbol}: {e}")
            return 0
    
    def _breakout_signal(self, symbol: str, params: Dict) -> int:
        """Breakout confirmation strategy signal (SPY)"""
        try:
            indicators = self.indicators[symbol]
            current_price = self.Securities[symbol].Price
            
            if not indicators['sma_20'].IsReady:
                return 0
            
            # Simple breakout above/below 20-day SMA
            sma_20 = indicators['sma_20'].Current.Value
            volume_sma = indicators['volume_sma'].Current.Value
            current_volume = self.Securities[symbol].Volume
            
            # Volume confirmation
            volume_spike = current_volume > volume_sma * params['volume_threshold']
            
            if current_price > sma_20 * 1.02 and volume_spike:  # 2% breakout with volume
                return 1
            elif current_price < sma_20 * 0.98 and volume_spike:  # 2% breakdown with volume
                return -1
            else:
                return 0
                
        except Exception as e:
            self.Debug(f"Breakout signal error for {symbol}: {e}")
            return 0
    
    def _mean_reversion_signal(self, symbol: str, params: Dict) -> int:
        """Mean reversion strategy signal (QQQ)"""
        try:
            indicators = self.indicators[symbol]
            
            if not indicators['bb'].IsReady:
                return 0
            
            current_price = self.Securities[symbol].Price
            bb_upper = indicators['bb'].UpperBand.Current.Value
            bb_lower = indicators['bb'].LowerBand.Current.Value
            bb_middle = indicators['bb'].MiddleBand.Current.Value
            
            # Mean reversion using Bollinger Bands
            threshold = params['threshold_multiplier']
            
            if current_price < bb_lower:  # Oversold
                return 1
            elif current_price > bb_upper:  # Overbought
                return -1
            else:
                return 0
                
        except Exception as e:
            self.Debug(f"Mean reversion error for {symbol}: {e}")
            return 0
    
    def _validate_trade(self, symbol: str) -> bool:
        """Validate if trade should be executed"""
        
        # Check daily trade limit
        if self.daily_trades >= self.risk_limits['max_daily_trades']:
            return False
        
        # Check position size limits
        current_holdings = self.Portfolio[symbol].Quantity
        if abs(current_holdings) >= self.Portfolio.TotalPortfolioValue * self.risk_limits['max_position_size']:
            return False
        
        return True
    
    def _execute_trade(self, symbol: str, signal: int, params: Dict):
        """Execute trade with proper position sizing"""
        
        allocation = params['allocation']
        portfolio_value = self.Portfolio.TotalPortfolioValue
        
        # Calculate target position
        target_value = portfolio_value * allocation * signal
        current_value = self.Portfolio[symbol].HoldingsValue
        
        # Calculate order size
        price = self.Securities[symbol].Price
        if price <= 0:
            return
        
        target_quantity = int(target_value / price)
        current_quantity = self.Portfolio[symbol].Quantity
        
        order_quantity = target_quantity - current_quantity
        
        if abs(order_quantity) < 1:  # Less than 1 share
            return
        
        # Place order
        try:
            self.MarketOrder(symbol, order_quantity)
            self.daily_trades += 1
            self.trade_count += 1
            
            # Store entry price for exit logic
            if symbol not in self.entry_prices or current_quantity == 0:
                self.entry_prices[symbol] = price
            
            self.Debug(f"Trade executed: {signal} {symbol} - Quantity: {order_quantity}")
            
        except Exception as e:
            self.Debug(f"Trade execution error for {symbol}: {e}")
    
    def CheckExitConditions(self):
        """Check for exit conditions (stop loss, take profit)"""
        
        for symbol in self.symbols.keys():
            if symbol not in self.Portfolio or self.Portfolio[symbol].Quantity == 0:
                continue
            
            current_price = self.Securities[symbol].Price
            entry_price = self.entry_prices.get(symbol, current_price)
            quantity = self.Portfolio[symbol].Quantity
            
            if entry_price <= 0:
                continue
            
            # Calculate P&L percentage
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Adjust for short positions
            if quantity < 0:
                pnl_pct = -pnl_pct
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            if pnl_pct <= -self.risk_limits['stop_loss_pct']:
                should_exit = True
                exit_reason = "Stop Loss"
            elif pnl_pct >= self.risk_limits['take_profit_pct']:
                should_exit = True
                exit_reason = "Take Profit"
            
            if should_exit:
                try:
                    self.Liquidate(symbol)
                    self.Debug(f"{exit_reason} triggered for {symbol}: {pnl_pct:.1%}")
                    if symbol in self.entry_prices:
                        del self.entry_prices[symbol]
                except Exception as e:
                    self.Debug(f"Exit error for {symbol}: {e}")
    
    def ResetDailyTrades(self):
        """Reset daily trade counter"""
        self.daily_trades = 0
        self.Debug(f"Daily trades reset. Total trades: {self.trade_count}")
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order filled: {orderEvent.Symbol} - {orderEvent.FillQuantity} @ {orderEvent.FillPrice}")
    
    def OnEndOfAlgorithm(self):
        """Algorithm termination"""
        self.Debug(f"Algorithm completed. Total trades: {self.trade_count}")
        
        # Performance summary
        total_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
        self.Debug(f"Total Return: {total_return:.1%}")


class HiveTradingAlphaModel(AlphaModel):
    """Custom alpha model for signal generation"""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.last_signal_time = {}
    
    def Update(self, algorithm, data):
        """Generate alpha signals"""
        insights = []
        
        # This is handled in the main algorithm OnData method
        # Alpha model integration can be enhanced later
        
        return insights