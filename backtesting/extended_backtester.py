"""
Hive Trade Extended Backtesting Engine
Comprehensive multi-year, multi-timeframe backtesting system
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
import json
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtendedMetrics:
    """Extended backtesting metrics"""
    strategy_name: str
    period: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration_days: float
    win_rate: float
    profit_factor: float
    expectancy: float
    total_trades: int
    avg_trade_duration: float
    best_year: str
    worst_year: str
    consistency_score: float
    tail_ratio: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    information_ratio: float
    treynor_ratio: float

class ExtendedBacktester:
    """
    Advanced multi-period backtesting engine
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results_cache = {}
        
    def load_extended_data(self, symbols: List[str], start_date: str = '2019-01-01', 
                          end_date: str = '2024-12-31') -> Dict[str, pd.DataFrame]:
        """Load extended historical data with parallel processing"""
        logger.info(f"Loading extended data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        def fetch_symbol_data(symbol):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                if data.empty:
                    logger.warning(f"No data for {symbol}")
                    return symbol, None
                
                # Add comprehensive technical indicators
                data = self.add_technical_indicators(data)
                
                logger.info(f"Loaded {len(data)} days for {symbol}")
                return symbol, data
                
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                return symbol, None
        
        # Parallel data loading
        market_data = {}
        max_workers = min(len(symbols), mp.cpu_count())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(fetch_symbol_data, symbol): symbol 
                               for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                if data is not None:
                    market_data[symbol] = data
        
        logger.info(f"Successfully loaded data for {len(market_data)} symbols")
        return market_data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        df = data.copy()
        
        # Price indicators
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        df['RSI_30'] = self.calculate_rsi(df['Close'], 30)
        
        # Bollinger Bands
        df['BB_Upper_20'], df['BB_Lower_20'] = self.calculate_bollinger_bands(df['Close'], 20, 2)
        df['BB_Width'] = (df['BB_Upper_20'] - df['BB_Lower_20']) / df['Close']
        df['BB_Position'] = (df['Close'] - df['BB_Lower_20']) / (df['BB_Upper_20'] - df['BB_Lower_20'])
        
        # Volatility
        df['Volatility_10'] = df['Close'].rolling(10).std()
        df['Volatility_30'] = df['Close'].rolling(30).std()
        df['ATR'] = self.calculate_atr(df, 14)
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['OBV'] = self.calculate_obv(df)
        
        # Returns and momentum
        df['Returns_1d'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_20d'] = df['Close'].pct_change(20)
        df['Returns_60d'] = df['Close'].pct_change(60)
        
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Price position
        df['High_52w'] = df['High'].rolling(252).max()
        df['Low_52w'] = df['Low'].rolling(252).min()
        df['Price_Position_52w'] = (df['Close'] - df['Low_52w']) / (df['High_52w'] - df['Low_52w'])
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = data['Volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def enhanced_momentum_strategy(self, data: Dict[str, pd.DataFrame], 
                                  lookback: int = 20, momentum_threshold: float = 0.02,
                                  rsi_overbought: float = 70, rsi_oversold: float = 30) -> List[Dict]:
        """Enhanced momentum strategy with RSI filter"""
        signals = []
        
        for symbol, df in data.items():
            df = df.copy()
            
            for i in range(max(lookback, 200), len(df)):
                current_date = df.index[i]
                current_price = df['Close'].iloc[i]
                
                momentum = df['Returns_20d'].iloc[i]
                rsi = df['RSI_14'].iloc[i]
                sma_50 = df['SMA_50'].iloc[i]
                sma_200 = df['SMA_200'].iloc[i]
                volume_ratio = df['Volume_Ratio'].iloc[i]
                
                if pd.isna(momentum) or pd.isna(rsi):
                    continue
                
                # Enhanced buy conditions
                if (momentum > momentum_threshold and 
                    rsi < rsi_overbought and 
                    current_price > sma_50 and 
                    sma_50 > sma_200 and 
                    volume_ratio > 1.2):
                    
                    confidence = min(momentum * 2 + (1 - rsi/100), 1.0)
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'strategy': 'enhanced_momentum',
                        'confidence': confidence,
                        'momentum': momentum,
                        'rsi': rsi
                    })
                
                # Enhanced sell conditions
                elif (momentum < -momentum_threshold and 
                      rsi > rsi_oversold and 
                      current_price < sma_50 and 
                      volume_ratio > 1.2):
                    
                    confidence = min(abs(momentum) * 2 + (rsi/100), 1.0)
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'strategy': 'enhanced_momentum',
                        'confidence': confidence,
                        'momentum': momentum,
                        'rsi': rsi
                    })
        
        return signals
    
    def mean_reversion_strategy_v2(self, data: Dict[str, pd.DataFrame],
                                  bb_period: int = 20, rsi_oversold: float = 25, 
                                  rsi_overbought: float = 75) -> List[Dict]:
        """Enhanced mean reversion strategy"""
        signals = []
        
        for symbol, df in data.items():
            df = df.copy()
            
            for i in range(bb_period, len(df)):
                current_date = df.index[i]
                current_price = df['Close'].iloc[i]
                
                rsi = df['RSI_14'].iloc[i]
                bb_position = df['BB_Position'].iloc[i]
                bb_width = df['BB_Width'].iloc[i]
                volume_ratio = df['Volume_Ratio'].iloc[i]
                volatility = df['Volatility_30'].iloc[i]
                
                if pd.isna(rsi) or pd.isna(bb_position):
                    continue
                
                # Buy oversold conditions
                if (bb_position <= 0.1 and  # Near lower BB
                    rsi <= rsi_oversold and 
                    bb_width > 0.02 and  # Sufficient volatility
                    volume_ratio > 1.1):
                    
                    confidence = (rsi_oversold - rsi) / rsi_oversold + (0.1 - bb_position) / 0.1
                    confidence = min(confidence / 2, 1.0)
                    
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'strategy': 'enhanced_mean_reversion',
                        'confidence': confidence,
                        'rsi': rsi,
                        'bb_position': bb_position
                    })
                
                # Sell overbought conditions
                elif (bb_position >= 0.9 and  # Near upper BB
                      rsi >= rsi_overbought and 
                      bb_width > 0.02 and 
                      volume_ratio > 1.1):
                    
                    confidence = (rsi - rsi_overbought) / (100 - rsi_overbought) + (bb_position - 0.9) / 0.1
                    confidence = min(confidence / 2, 1.0)
                    
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'strategy': 'enhanced_mean_reversion',
                        'confidence': confidence,
                        'rsi': rsi,
                        'bb_position': bb_position
                    })
        
        return signals
    
    def breakout_strategy_v2(self, data: Dict[str, pd.DataFrame],
                            lookback: int = 20, volume_multiplier: float = 1.8,
                            atr_multiplier: float = 1.5) -> List[Dict]:
        """Enhanced breakout strategy with ATR filter"""
        signals = []
        
        for symbol, df in data.items():
            df = df.copy()
            df['High_Max'] = df['High'].rolling(lookback).max()
            df['Low_Min'] = df['Low'].rolling(lookback).min()
            df['Volume_Avg'] = df['Volume'].rolling(lookback).mean()
            
            for i in range(lookback, len(df)):
                current_date = df.index[i]
                current_price = df['Close'].iloc[i]
                current_high = df['High'].iloc[i]
                current_low = df['Low'].iloc[i]
                
                high_max = df['High_Max'].iloc[i-1]
                low_min = df['Low_Min'].iloc[i-1]
                volume_avg = df['Volume_Avg'].iloc[i-1]
                current_volume = df['Volume'].iloc[i]
                atr = df['ATR'].iloc[i]
                sma_50 = df['SMA_50'].iloc[i]
                
                if pd.isna(high_max) or pd.isna(atr):
                    continue
                
                # Upward breakout
                if (current_high > high_max and 
                    current_volume > volume_avg * volume_multiplier and
                    atr > df['ATR'].rolling(50).mean().iloc[i] * atr_multiplier and
                    current_price > sma_50):
                    
                    breakout_strength = (current_high - high_max) / high_max
                    volume_strength = current_volume / (volume_avg * volume_multiplier)
                    confidence = min(breakout_strength * 10 + volume_strength * 0.5, 1.0)
                    
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'strategy': 'enhanced_breakout',
                        'confidence': confidence,
                        'breakout_level': high_max,
                        'atr': atr
                    })
                
                # Downward breakout
                elif (current_low < low_min and 
                      current_volume > volume_avg * volume_multiplier and
                      atr > df['ATR'].rolling(50).mean().iloc[i] * atr_multiplier and
                      current_price < sma_50):
                    
                    breakout_strength = (low_min - current_low) / low_min
                    volume_strength = current_volume / (volume_avg * volume_multiplier)
                    confidence = min(breakout_strength * 10 + volume_strength * 0.5, 1.0)
                    
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'strategy': 'enhanced_breakout',
                        'confidence': confidence,
                        'breakout_level': low_min,
                        'atr': atr
                    })
        
        return signals
    
    def run_extended_backtest(self, signals: List[Dict], market_data: Dict[str, pd.DataFrame],
                             position_size_pct: float = 0.1, max_positions: int = 5) -> Dict[str, Any]:
        """Run extended backtest with detailed tracking"""
        
        # Initialize tracking
        cash = self.initial_capital
        positions = {}
        portfolio_history = []
        trades = []
        
        # Sort signals by date
        signals.sort(key=lambda x: x['date'])
        
        for signal in signals:
            signal_date = signal['date']
            symbol = signal['symbol']
            action = signal['action']
            price = signal['price']
            confidence = signal.get('confidence', 0.5)
            
            # Calculate portfolio value
            portfolio_value = cash
            for pos_symbol, position in positions.items():
                try:
                    # Get current price from market data
                    symbol_data = market_data[pos_symbol]
                    price_idx = symbol_data.index.get_indexer([signal_date], method='nearest')[0]
                    current_price = symbol_data['Close'].iloc[price_idx]
                    portfolio_value += position['quantity'] * current_price
                except:
                    portfolio_value += position['quantity'] * position['avg_cost']
            
            # Position sizing based on confidence
            max_position_value = portfolio_value * position_size_pct * confidence
            
            if action == 'BUY' and len(positions) < max_positions:
                if cash > max_position_value:
                    quantity = max_position_value / price
                    cost = quantity * price
                    cash -= cost
                    
                    positions[symbol] = {
                        'quantity': quantity,
                        'avg_cost': price,
                        'entry_date': signal_date,
                        'strategy': signal['strategy']
                    }
            
            elif action == 'SELL' and symbol in positions:
                position = positions[symbol]
                quantity = position['quantity']
                revenue = quantity * price
                
                # Calculate trade metrics
                pnl = revenue - (quantity * position['avg_cost'])
                pnl_pct = pnl / (quantity * position['avg_cost']) * 100
                duration = (signal_date - position['entry_date']).days
                
                trades.append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': signal_date,
                    'entry_price': position['avg_cost'],
                    'exit_price': price,
                    'quantity': quantity,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration_days': duration,
                    'strategy': position['strategy']
                })
                
                cash += revenue
                del positions[symbol]
            
            # Record portfolio state
            portfolio_history.append({
                'date': signal_date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'num_positions': len(positions)
            })
        
        return {
            'portfolio_history': portfolio_history,
            'trades': trades,
            'final_value': portfolio_history[-1]['portfolio_value'] if portfolio_history else self.initial_capital
        }
    
    def calculate_extended_metrics(self, backtest_results: Dict[str, Any], 
                                  strategy_name: str, period: str,
                                  benchmark_data: pd.DataFrame = None) -> ExtendedMetrics:
        """Calculate comprehensive performance metrics"""
        
        portfolio_history = backtest_results['portfolio_history']
        trades = backtest_results['trades']
        
        if not portfolio_history:
            return self._empty_metrics(strategy_name, period)
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['returns'] = df['portfolio_value'].pct_change().fillna(0)
        
        # Basic metrics
        total_return = (df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        trading_days = len(df)
        years = trading_days / 252
        
        if years > 0:
            annualized_return = ((df['portfolio_value'].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100
        else:
            annualized_return = 0
        
        # Risk metrics
        daily_returns = df['returns']
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        risk_free_rate = 0.02
        excess_returns = daily_returns - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return / 100) / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        rolling_max = df['portfolio_value'].expanding().max()
        drawdown = (df['portfolio_value'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Drawdown duration
        drawdown_periods = []
        in_drawdown = False
        start_dd = None
        
        for i, dd in enumerate(drawdown):
            if dd < -1 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                start_dd = i
            elif dd >= -1 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if start_dd is not None:
                    drawdown_periods.append(i - start_dd)
        
        avg_dd_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade-based metrics
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            
            total_wins = sum(t['pnl'] for t in winning_trades)
            total_losses = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            expectancy = np.mean([t['pnl'] for t in trades])
            avg_trade_duration = np.mean([t['duration_days'] for t in trades])
            
        else:
            win_rate = 0
            profit_factor = 0
            expectancy = 0
            avg_trade_duration = 0
        
        # Annual performance
        df['year'] = df.index.year
        annual_returns = df.groupby('year')['portfolio_value'].last().pct_change() * 100
        annual_returns = annual_returns.dropna()
        
        best_year = f"{annual_returns.idxmax()}: {annual_returns.max():.1f}%" if len(annual_returns) > 0 else "N/A"
        worst_year = f"{annual_returns.idxmin()}: {annual_returns.min():.1f}%" if len(annual_returns) > 0 else "N/A"
        
        # Consistency score (% of positive years)
        consistency_score = (annual_returns > 0).mean() * 100 if len(annual_returns) > 0 else 0
        
        # Higher moment statistics
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurtosis()
        
        # VaR and CVaR
        var_95 = np.percentile(daily_returns, 5) * df['portfolio_value'].iloc[-1]
        cvar_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * df['portfolio_value'].iloc[-1]
        
        # Tail ratio
        tail_ratio = abs(np.percentile(daily_returns, 95)) / abs(np.percentile(daily_returns, 5))
        
        # Beta and Alpha (if benchmark provided)
        beta = 0
        alpha = 0
        information_ratio = 0
        treynor_ratio = 0
        
        if benchmark_data is not None:
            try:
                # Align dates and calculate beta
                aligned_data = pd.merge(df['returns'], benchmark_data.pct_change(), 
                                      left_index=True, right_index=True, how='inner')
                
                if len(aligned_data) > 10:
                    portfolio_returns = aligned_data.iloc[:, 0]
                    benchmark_returns = aligned_data.iloc[:, 1]
                    
                    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                    benchmark_variance = np.var(benchmark_returns)
                    
                    if benchmark_variance > 0:
                        beta = covariance / benchmark_variance
                        alpha = (annualized_return / 100) - (risk_free_rate + beta * 
                                (benchmark_returns.mean() * 252 - risk_free_rate))
                        
                        # Information ratio
                        excess_return = portfolio_returns - benchmark_returns
                        information_ratio = excess_return.mean() / excess_return.std() * np.sqrt(252)
                        
                        # Treynor ratio
                        treynor_ratio = (annualized_return / 100 - risk_free_rate) / beta if beta != 0 else 0
            except:
                pass
        
        return ExtendedMetrics(
            strategy_name=strategy_name,
            period=period,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            drawdown_duration_days=avg_dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            total_trades=len(trades),
            avg_trade_duration=avg_trade_duration,
            best_year=best_year,
            worst_year=worst_year,
            consistency_score=consistency_score,
            tail_ratio=tail_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio
        )
    
    def _empty_metrics(self, strategy_name: str, period: str) -> ExtendedMetrics:
        """Return empty metrics for failed strategies"""
        return ExtendedMetrics(
            strategy_name=strategy_name, period=period, total_return=0, annualized_return=0,
            volatility=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
            avg_drawdown=0, drawdown_duration_days=0, win_rate=0, profit_factor=0, expectancy=0,
            total_trades=0, avg_trade_duration=0, best_year="N/A", worst_year="N/A",
            consistency_score=0, tail_ratio=0, skewness=0, kurtosis=0, var_95=0, cvar_95=0,
            beta=0, alpha=0, information_ratio=0, treynor_ratio=0
        )
    
    def run_multi_period_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Run comprehensive multi-period backtesting analysis"""
        
        logger.info("Starting comprehensive multi-period analysis...")
        
        # Load extended data
        market_data = self.load_extended_data(symbols, '2019-01-01', '2024-12-31')
        
        if not market_data:
            logger.error("No market data loaded")
            return {}
        
        # Load benchmark data (SPY)
        try:
            benchmark = yf.download('SPY', start='2019-01-01', end='2024-12-31')['Close']
        except:
            benchmark = None
        
        # Test periods
        test_periods = [
            ('2019-01-01', '2024-12-31', 'Full Period (2019-2024)'),
            ('2019-01-01', '2021-12-31', 'Pre-COVID + COVID (2019-2021)'),
            ('2022-01-01', '2024-12-31', 'Post-COVID (2022-2024)'),
            ('2020-03-01', '2020-06-30', 'COVID Crash (Mar-Jun 2020)'),
            ('2022-01-01', '2022-12-31', 'Bear Market 2022'),
            ('2023-01-01', '2024-12-31', 'Recovery Period (2023-2024)')
        ]
        
        # Enhanced strategies to test
        strategies = [
            {
                'name': 'Enhanced Momentum',
                'function': self.enhanced_momentum_strategy,
                'params': {'lookback': 20, 'momentum_threshold': 0.02}
            },
            {
                'name': 'Enhanced Mean Reversion',
                'function': self.mean_reversion_strategy_v2,
                'params': {'rsi_oversold': 25, 'rsi_overbought': 75}
            },
            {
                'name': 'Enhanced Breakout',
                'function': self.breakout_strategy_v2,
                'params': {'lookback': 20, 'volume_multiplier': 1.8}
            }
        ]
        
        results = {}
        
        for strategy in strategies:
            strategy_results = {}
            logger.info(f"Testing {strategy['name']} strategy...")
            
            # Generate signals for full period
            signals = strategy['function'](market_data, **strategy['params'])
            logger.info(f"Generated {len(signals)} signals for {strategy['name']}")
            
            for start_date, end_date, period_name in test_periods:
                logger.info(f"  Testing period: {period_name}")
                
                # Filter signals for this period
                start_dt = pd.to_datetime(start_date).tz_localize(None)
                end_dt = pd.to_datetime(end_date).tz_localize(None)
                period_signals = []
                for s in signals:
                    signal_date = s['date']
                    if hasattr(signal_date, 'tz') and signal_date.tz:
                        signal_date = signal_date.tz_localize(None)
                    if start_dt <= signal_date <= end_dt:
                        period_signals.append(s)
                
                if not period_signals:
                    logger.warning(f"No signals for {strategy['name']} in {period_name}")
                    continue
                
                # Filter market data for this period
                period_data = {}
                for symbol, data in market_data.items():
                    mask = (data.index >= start_date) & (data.index <= end_date)
                    period_data[symbol] = data.loc[mask]
                
                # Run backtest
                backtest_results = self.run_extended_backtest(period_signals, period_data)
                
                # Calculate metrics
                benchmark_period = benchmark.loc[start_date:end_date] if benchmark is not None else None
                metrics = self.calculate_extended_metrics(backtest_results, strategy['name'], 
                                                        period_name, benchmark_period)
                
                strategy_results[period_name] = metrics
                logger.info(f"    {strategy['name']} - {period_name}: {metrics.total_return:.1f}% return, "
                           f"{metrics.sharpe_ratio:.2f} Sharpe")
            
            results[strategy['name']] = strategy_results
        
        return results

def main():
    """Main extended backtesting workflow"""
    
    print("HIVE TRADE EXTENDED BACKTESTING ENGINE")
    print("="*50)
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX', 'SPY', 'QQQ']
    
    # Initialize backtester
    backtester = ExtendedBacktester(initial_capital=100000)
    
    print(f"Starting extended backtesting analysis...")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: 2019-2024 (6 years)")
    print(f"Test periods: 6 different market conditions")
    
    # Run comprehensive analysis
    results = backtester.run_multi_period_analysis(symbols)
    
    if not results:
        print("ERROR: No results generated")
        return
    
    # Generate comprehensive report
    report = generate_extended_report(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = f"extended_backtest_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        # Convert metrics to dict for JSON serialization
        json_results = {}
        for strategy, periods in results.items():
            json_results[strategy] = {}
            for period, metrics in periods.items():
                json_results[strategy][period] = asdict(metrics)
        json.dump(json_results, f, indent=2, default=str)
    
    # Save report
    report_file = f"extended_backtest_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nEXTENDED BACKTESTING COMPLETE!")
    print(f"- Results saved: {results_file}")
    print(f"- Report saved: {report_file}")
    
    # Show summary
    print(f"\nPERFORMANCE SUMMARY:")
    print("-" * 60)
    
    for strategy_name, periods in results.items():
        full_period = periods.get('Full Period (2019-2024)')
        if full_period:
            print(f"{strategy_name:25s}: {full_period.total_return:+7.1f}% | "
                  f"Sharpe: {full_period.sharpe_ratio:5.2f} | "
                  f"MaxDD: {full_period.max_drawdown:6.1f}% | "
                  f"Trades: {full_period.total_trades:4d}")

def generate_extended_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive extended backtesting report"""
    
    report = f"""
HIVE TRADE EXTENDED BACKTESTING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: 2019-2024 (6 Years)
{'='*80}

EXECUTIVE SUMMARY:
{'*'*40}

This comprehensive analysis tested 3 enhanced trading strategies across 6 different
market periods, including bull markets, bear markets, and crisis periods.

KEY FINDINGS:
"""
    
    # Find best performing strategy
    best_strategy = None
    best_sharpe = -999
    
    for strategy_name, periods in results.items():
        full_period = periods.get('Full Period (2019-2024)')
        if full_period and full_period.sharpe_ratio > best_sharpe:
            best_sharpe = full_period.sharpe_ratio
            best_strategy = (strategy_name, full_period)
    
    if best_strategy:
        name, metrics = best_strategy
        report += f"""
Best Overall Strategy: {name}
- Total Return: {metrics.total_return:+.1f}%
- Annualized Return: {metrics.annualized_return:+.1f}%
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Max Drawdown: {metrics.max_drawdown:.1f}%
- Win Rate: {metrics.win_rate:.1f}%
- Total Trades: {metrics.total_trades}
"""
    
    report += f"""

DETAILED STRATEGY ANALYSIS:
{'*'*40}
"""
    
    for strategy_name, periods in results.items():
        report += f"""
{strategy_name.upper()} STRATEGY:
{'-'*50}
"""
        
        for period_name, metrics in periods.items():
            report += f"""
{period_name}:
  Return: {metrics.total_return:+7.1f}% | Annualized: {metrics.annualized_return:+6.1f}%
  Risk:   Volatility: {metrics.volatility:5.1f}% | Max DD: {metrics.max_drawdown:6.1f}%
  Ratios: Sharpe: {metrics.sharpe_ratio:4.2f} | Sortino: {metrics.sortino_ratio:4.2f} | Calmar: {metrics.calmar_ratio:4.2f}
  Trades: Count: {metrics.total_trades:3d} | Win Rate: {metrics.win_rate:5.1f}% | Profit Factor: {metrics.profit_factor:4.2f}
  Years:  Best: {metrics.best_year} | Worst: {metrics.worst_year}
"""
    
    report += f"""

MARKET PERIOD ANALYSIS:
{'*'*40}

This analysis reveals how each strategy performed during different market conditions:

COVID Crash Period (Mar-Jun 2020):
- Tested strategies during extreme volatility and market decline
- Shows strategy robustness during crisis conditions

Bear Market 2022:
- Analysis during Fed tightening and market correction
- Reveals strategy performance in declining markets

Recovery Period (2023-2024):
- Performance during market recovery and new highs
- Shows strategy adaptation to changing conditions

RISK ANALYSIS:
{'*'*40}
"""
    
    # Risk comparison table
    report += f"{'Strategy':<25} {'Max DD':<8} {'Volatility':<10} {'Sharpe':<7} {'Sortino':<7}\n"
    report += f"{'-'*60}\n"
    
    for strategy_name, periods in results.items():
        full_period = periods.get('Full Period (2019-2024)')
        if full_period:
            report += f"{strategy_name:<25} {full_period.max_drawdown:6.1f}% "
            report += f"{full_period.volatility:8.1f}% {full_period.sharpe_ratio:6.2f} "
            report += f"{full_period.sortino_ratio:6.2f}\n"
    
    report += f"""

CONSISTENCY ANALYSIS:
{'*'*40}
"""
    
    for strategy_name, periods in results.items():
        full_period = periods.get('Full Period (2019-2024)')
        if full_period:
            report += f"""
{strategy_name}:
  Consistency Score: {full_period.consistency_score:.1f}% (% of profitable years)
  Average Drawdown: {full_period.avg_drawdown:.1f}%
  Drawdown Duration: {full_period.drawdown_duration_days:.1f} days average
  Tail Ratio: {full_period.tail_ratio:.2f} (upside/downside tail ratio)
"""
    
    report += f"""

RECOMMENDATIONS:
{'*'*40}

Strategy Selection:
1. For consistent returns: Choose strategy with highest consistency score
2. For maximum returns: Choose strategy with highest Sharpe ratio
3. For risk management: Choose strategy with lowest maximum drawdown

Implementation Notes:
1. All strategies show period-dependent performance
2. Consider ensemble approach combining multiple strategies
3. Implement dynamic position sizing based on market volatility
4. Add regime detection to switch between strategies

Market Regime Considerations:
1. Bull markets favor momentum strategies
2. Volatile markets favor mean reversion strategies
3. Trending markets favor breakout strategies
4. Consider macro-economic indicators for regime detection

Risk Management:
1. Implement maximum drawdown limits (suggest 15-20%)
2. Use position sizing based on volatility
3. Consider correlation limits across positions
4. Implement stop-loss and take-profit levels

NEXT STEPS:
{'*'*40}

1. Implement best-performing strategy in paper trading
2. Add regime detection for dynamic strategy selection
3. Optimize parameters for current market conditions
4. Implement ensemble methods for improved robustness
5. Add alternative assets (bonds, commodities, crypto)

{'='*80}
Extended Backtesting Analysis Complete
"""
    
    return report

if __name__ == "__main__":
    main()