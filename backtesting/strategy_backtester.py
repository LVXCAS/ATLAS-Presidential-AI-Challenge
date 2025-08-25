"""
Hive Trade Strategy Backtesting Engine
Comprehensive backtesting for new trading strategies
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import yfinance as yf
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestMetrics:
    """Comprehensive backtesting metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    var_99: float
    final_portfolio_value: float
    
@dataclass
class Trade:
    """Individual trade record"""
    entry_date: datetime
    exit_date: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_days: float
    strategy: str
    agent: str

class StrategyBacktester:
    """
    Advanced backtesting engine for trading strategies
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage: float = 0.0001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.trades = []
        self.portfolio_history = []
        self.cash = initial_capital
        self.positions = {}
        self.daily_returns = []
        
    def load_market_data(self, symbols: List[str], 
                        start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load historical market data for backtesting"""
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Add technical indicators
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['EMA_12'] = data['Close'].ewm(span=12).mean()
                data['EMA_26'] = data['Close'].ewm(span=26).mean()
                data['RSI'] = self.calculate_rsi(data['Close'])
                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
                data['BB_Upper'], data['BB_Lower'] = self.calculate_bollinger_bands(data['Close'])
                data['Volatility'] = data['Close'].rolling(window=20).std()
                data['Returns'] = data['Close'].pct_change()
                
                market_data[symbol] = data
                logger.info(f"Loaded {len(data)} days of data for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
        
        return market_data
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def momentum_strategy(self, data: Dict[str, pd.DataFrame], 
                         lookback: int = 20, momentum_threshold: float = 0.02) -> List[Dict]:
        """
        Momentum trading strategy
        """
        signals = []
        
        for symbol, df in data.items():
            df = df.copy()
            df['Momentum'] = df['Close'].pct_change(lookback)
            
            for i in range(lookback, len(df)):
                current_date = df.index[i]
                current_price = df['Close'].iloc[i]
                momentum = df['Momentum'].iloc[i]
                
                if pd.isna(momentum):
                    continue
                
                # Buy signal: Strong positive momentum
                if momentum > momentum_threshold:
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'strategy': 'momentum',
                        'agent': 'momentum_agent',
                        'confidence': min(momentum * 10, 1.0),
                        'reasoning': f'Strong momentum: {momentum:.3f}'
                    })
                
                # Sell signal: Strong negative momentum
                elif momentum < -momentum_threshold:
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'strategy': 'momentum',
                        'agent': 'momentum_agent',
                        'confidence': min(abs(momentum) * 10, 1.0),
                        'reasoning': f'Negative momentum: {momentum:.3f}'
                    })
        
        return signals
    
    def mean_reversion_strategy(self, data: Dict[str, pd.DataFrame],
                               bb_period: int = 20, rsi_oversold: float = 30, 
                               rsi_overbought: float = 70) -> List[Dict]:
        """
        Mean reversion strategy using Bollinger Bands and RSI
        """
        signals = []
        
        for symbol, df in data.items():
            df = df.copy()
            
            for i in range(bb_period, len(df)):
                current_date = df.index[i]
                current_price = df['Close'].iloc[i]
                rsi = df['RSI'].iloc[i]
                bb_upper = df['BB_Upper'].iloc[i]
                bb_lower = df['BB_Lower'].iloc[i]
                
                if pd.isna(rsi) or pd.isna(bb_upper) or pd.isna(bb_lower):
                    continue
                
                # Buy signal: Price near lower Bollinger Band and RSI oversold
                if current_price <= bb_lower and rsi <= rsi_oversold:
                    confidence = (rsi_oversold - rsi) / rsi_oversold + (bb_lower - current_price) / bb_lower
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'strategy': 'mean_reversion',
                        'agent': 'mean_reversion_agent',
                        'confidence': min(confidence, 1.0),
                        'reasoning': f'Oversold: RSI={rsi:.1f}, Price at lower BB'
                    })
                
                # Sell signal: Price near upper Bollinger Band and RSI overbought
                elif current_price >= bb_upper and rsi >= rsi_overbought:
                    confidence = (rsi - rsi_overbought) / (100 - rsi_overbought) + (current_price - bb_upper) / bb_upper
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'strategy': 'mean_reversion',
                        'agent': 'mean_reversion_agent',
                        'confidence': min(confidence, 1.0),
                        'reasoning': f'Overbought: RSI={rsi:.1f}, Price at upper BB'
                    })
        
        return signals
    
    def breakout_strategy(self, data: Dict[str, pd.DataFrame],
                         lookback: int = 20, volume_multiplier: float = 1.5) -> List[Dict]:
        """
        Breakout strategy based on price and volume
        """
        signals = []
        
        for symbol, df in data.items():
            df = df.copy()
            df['High_Max'] = df['High'].rolling(window=lookback).max()
            df['Low_Min'] = df['Low'].rolling(window=lookback).min()
            df['Volume_Avg'] = df['Volume'].rolling(window=lookback).mean()
            
            for i in range(lookback, len(df)):
                current_date = df.index[i]
                current_price = df['Close'].iloc[i]
                high_max = df['High_Max'].iloc[i-1]  # Previous period max
                low_min = df['Low_Min'].iloc[i-1]    # Previous period min
                volume_avg = df['Volume_Avg'].iloc[i-1]
                current_volume = df['Volume'].iloc[i]
                
                if pd.isna(high_max) or pd.isna(low_min):
                    continue
                
                # Upward breakout
                if (current_price > high_max and 
                    current_volume > volume_avg * volume_multiplier):
                    confidence = min((current_price - high_max) / high_max * 10, 1.0)
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'strategy': 'breakout',
                        'agent': 'breakout_agent',
                        'confidence': confidence,
                        'reasoning': f'Upward breakout above {high_max:.2f} with high volume'
                    })
                
                # Downward breakout
                elif (current_price < low_min and 
                      current_volume > volume_avg * volume_multiplier):
                    confidence = min((low_min - current_price) / low_min * 10, 1.0)
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'strategy': 'breakout',
                        'agent': 'breakout_agent',
                        'confidence': confidence,
                        'reasoning': f'Downward breakout below {low_min:.2f} with high volume'
                    })
        
        return signals
    
    def execute_backtest(self, signals: List[Dict], 
                        market_data: Dict[str, pd.DataFrame],
                        position_size_pct: float = 0.1,
                        stop_loss: float = 0.05,
                        take_profit: float = 0.10) -> None:
        """
        Execute backtest based on trading signals
        """
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        # Sort signals by date
        signals.sort(key=lambda x: x['date'])
        
        current_date = None
        
        for signal in signals:
            signal_date = signal['date']
            symbol = signal['symbol']
            action = signal['action']
            price = signal['price']
            strategy = signal['strategy']
            agent = signal['agent']
            
            # Apply slippage
            if action == 'BUY':
                execution_price = price * (1 + self.slippage)
            else:
                execution_price = price * (1 - self.slippage)
            
            # Calculate position size
            portfolio_value = self.calculate_portfolio_value(market_data, signal_date)
            max_position_value = portfolio_value * position_size_pct
            
            if action == 'BUY' and self.cash > 0:
                # Calculate quantity based on available cash and position size limit
                max_quantity_cash = self.cash / execution_price
                max_quantity_size = max_position_value / execution_price
                quantity = min(max_quantity_cash, max_quantity_size)
                
                if quantity > 0:
                    cost = quantity * execution_price
                    commission = cost * self.commission_rate
                    total_cost = cost + commission
                    
                    if total_cost <= self.cash:
                        # Execute buy
                        self.cash -= total_cost
                        
                        if symbol in self.positions:
                            self.positions[symbol]['quantity'] += quantity
                            self.positions[symbol]['avg_cost'] = (
                                (self.positions[symbol]['avg_cost'] * (self.positions[symbol]['quantity'] - quantity) + 
                                 execution_price * quantity) / self.positions[symbol]['quantity']
                            )
                        else:
                            self.positions[symbol] = {
                                'quantity': quantity,
                                'avg_cost': execution_price,
                                'entry_date': signal_date,
                                'strategy': strategy,
                                'agent': agent
                            }
            
            elif action == 'SELL' and symbol in self.positions and self.positions[symbol]['quantity'] > 0:
                # Sell position
                position = self.positions[symbol]
                quantity = position['quantity']
                avg_cost = position['avg_cost']
                
                revenue = quantity * execution_price
                commission = revenue * self.commission_rate
                net_revenue = revenue - commission
                
                # Calculate trade metrics
                total_cost = quantity * avg_cost
                pnl = net_revenue - total_cost
                pnl_pct = (pnl / total_cost) * 100
                duration = (signal_date - position['entry_date']).days
                
                # Record trade
                trade = Trade(
                    entry_date=position['entry_date'],
                    exit_date=signal_date,
                    symbol=symbol,
                    side='BUY',  # We bought, now selling
                    entry_price=avg_cost,
                    exit_price=execution_price,
                    quantity=quantity,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    duration_days=duration,
                    strategy=position['strategy'],
                    agent=position['agent']
                )
                self.trades.append(trade)
                
                # Update cash
                self.cash += net_revenue
                
                # Remove position
                del self.positions[symbol]
            
            # Record portfolio value
            if current_date != signal_date:
                portfolio_value = self.calculate_portfolio_value(market_data, signal_date)
                self.portfolio_history.append({
                    'date': signal_date,
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'positions_value': portfolio_value - self.cash
                })
                current_date = signal_date
    
    def calculate_portfolio_value(self, market_data: Dict[str, pd.DataFrame], date: datetime) -> float:
        """Calculate total portfolio value at a given date"""
        positions_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                try:
                    # Find the closest date
                    symbol_data = market_data[symbol]
                    closest_date_idx = symbol_data.index.get_indexer([date], method='nearest')[0]
                    current_price = symbol_data['Close'].iloc[closest_date_idx]
                    positions_value += position['quantity'] * current_price
                except:
                    # Use average cost if current price not available
                    positions_value += position['quantity'] * position['avg_cost']
        
        return self.cash + positions_value
    
    def calculate_metrics(self, market_data: Dict[str, pd.DataFrame]) -> BacktestMetrics:
        """Calculate comprehensive backtesting metrics"""
        if not self.portfolio_history:
            raise ValueError("No portfolio history available")
        
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        daily_returns = portfolio_df['returns'].dropna()
        
        # Basic metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Annualized return
        trading_days = len(portfolio_df)
        years = trading_days / 252
        annualized_return = ((portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100
        
        # Volatility
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = daily_returns - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return / 100) / downside_deviation if downside_deviation > 0 else 0
        
        # VaR calculations
        var_95 = np.percentile(daily_returns * portfolio_df['portfolio_value'].iloc[-1], 5)
        var_99 = np.percentile(daily_returns * portfolio_df['portfolio_value'].iloc[-1], 1)
        
        # Trade-based metrics
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            win_rate = (len(winning_trades) / len(self.trades)) * 100
            
            total_winning_pnl = sum(t.pnl for t in winning_trades)
            total_losing_pnl = sum(abs(t.pnl) for t in losing_trades)
            profit_factor = total_winning_pnl / total_losing_pnl if total_losing_pnl > 0 else float('inf')
            
            avg_trade_duration = np.mean([t.duration_days for t in self.trades])
            best_trade = max(t.pnl for t in self.trades)
            worst_trade = min(t.pnl for t in self.trades)
            
            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            current_wins = 0
            current_losses = 0
            
            for trade in self.trades:
                if trade.pnl > 0:
                    current_wins += 1
                    current_losses = 0
                    consecutive_wins = max(consecutive_wins, current_wins)
                else:
                    current_losses += 1
                    current_wins = 0
                    consecutive_losses = max(consecutive_losses, current_losses)
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_duration = 0
            best_trade = 0
            worst_trade = 0
            consecutive_wins = 0
            consecutive_losses = 0
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            avg_trade_duration=avg_trade_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            var_99=var_99,
            final_portfolio_value=portfolio_df['portfolio_value'].iloc[-1]
        )
    
    def generate_backtest_report(self, strategy_name: str, metrics: BacktestMetrics, 
                               market_data: Dict[str, pd.DataFrame]) -> str:
        """Generate comprehensive backtest report"""
        
        report = f"""
HIVE TRADE BACKTESTING REPORT - {strategy_name.upper()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

STRATEGY PERFORMANCE SUMMARY:
{'*'*40}

Initial Capital:        ${self.initial_capital:,.2f}
Final Portfolio Value:  ${metrics.final_portfolio_value:,.2f}
Total Return:           {metrics.total_return:+.2f}%
Annualized Return:      {metrics.annualized_return:+.2f}%

RISK METRICS:
{'*'*40}

Volatility:             {metrics.volatility:.2f}%
Maximum Drawdown:       {metrics.max_drawdown:.2f}%
Sharpe Ratio:           {metrics.sharpe_ratio:.2f}
Calmar Ratio:           {metrics.calmar_ratio:.2f}
Sortino Ratio:          {metrics.sortino_ratio:.2f}
VaR (95%):              ${metrics.var_95:,.2f}
VaR (99%):              ${metrics.var_99:,.2f}

TRADING STATISTICS:
{'*'*40}

Total Trades:           {metrics.total_trades}
Win Rate:               {metrics.win_rate:.1f}%
Profit Factor:          {metrics.profit_factor:.2f}
Average Trade Duration: {metrics.avg_trade_duration:.1f} days
Best Trade:             ${metrics.best_trade:,.2f}
Worst Trade:            ${metrics.worst_trade:,.2f}
Consecutive Wins:       {metrics.consecutive_wins}
Consecutive Losses:     {metrics.consecutive_losses}

TRADE DETAILS:
{'*'*40}
"""
        
        if self.trades:
            report += "\nTop 10 Best Trades:\n"
            report += "-" * 20 + "\n"
            best_trades = sorted(self.trades, key=lambda t: t.pnl, reverse=True)[:10]
            for i, trade in enumerate(best_trades, 1):
                report += f"{i:2d}. {trade.symbol} | {trade.entry_date.date()} to {trade.exit_date.date()} | "
                report += f"${trade.pnl:+,.2f} ({trade.pnl_pct:+.1f}%) | {trade.strategy}\n"
            
            report += "\nTop 10 Worst Trades:\n"
            report += "-" * 20 + "\n"
            worst_trades = sorted(self.trades, key=lambda t: t.pnl)[:10]
            for i, trade in enumerate(worst_trades, 1):
                report += f"{i:2d}. {trade.symbol} | {trade.entry_date.date()} to {trade.exit_date.date()} | "
                report += f"${trade.pnl:+,.2f} ({trade.pnl_pct:+.1f}%) | {trade.strategy}\n"
        
        # Strategy comparison
        report += f"""

STRATEGY BENCHMARKING:
{'*'*40}

vs Buy & Hold SPY:
- Strategy Return: {metrics.annualized_return:.2f}%
- Strategy Sharpe: {metrics.sharpe_ratio:.2f}
- Strategy MaxDD:  {metrics.max_drawdown:.2f}%

Risk-Adjusted Performance:
- Return/Risk Ratio: {metrics.annualized_return/metrics.volatility:.2f}
- Risk/Reward:       {abs(metrics.max_drawdown)/metrics.annualized_return:.2f}

CONFIGURATION:
{'*'*40}

Commission Rate:    {self.commission_rate*100:.3f}%
Slippage:          {self.slippage*100:.4f}%
Symbols Traded:    {list(market_data.keys())}
Backtest Period:   {len(self.portfolio_history)} days

RECOMMENDATIONS:
{'*'*40}

1. Risk Management:
   {'+++ Low risk' if metrics.max_drawdown > -10 else '+/- Moderate risk' if metrics.max_drawdown > -20 else '--- High risk'}
   
2. Strategy Performance:
   {'+++ Excellent' if metrics.sharpe_ratio > 2 else '++ Good' if metrics.sharpe_ratio > 1 else '+/- Below average' if metrics.sharpe_ratio > 0 else '-- Poor'}
   
3. Trading Frequency:
   {'+++ Conservative' if len(self.trades) < 50 else '++ Moderate' if len(self.trades) < 200 else '+/- Aggressive'}

4. Next Steps:
   - {'Reduce position sizes' if metrics.max_drawdown < -15 else 'Current position sizing acceptable'}
   - {'Add stop losses' if metrics.worst_trade < -1000 else 'Risk per trade controlled'}
   - {'Optimize entry/exit rules' if metrics.win_rate < 50 else 'Win rate acceptable'}

{'='*80}
Backtest Analysis Complete
"""
        
        return report

def main():
    """Main backtesting workflow"""
    print("HIVE TRADE STRATEGY BACKTESTING ENGINE")
    print("="*50)
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
    start_date = '2023-01-01'
    end_date = '2024-08-01'
    
    # Initialize backtester
    backtester = StrategyBacktester(
        initial_capital=100000,
        commission_rate=0.001,
        slippage=0.0001
    )
    
    print("Loading market data...")
    market_data = backtester.load_market_data(symbols, start_date, end_date)
    
    if not market_data:
        print("ERROR: No market data loaded")
        return
    
    print(f"Loaded data for {len(market_data)} symbols")
    
    # Test different strategies
    strategies = [
        {
            'name': 'momentum',
            'function': backtester.momentum_strategy,
            'params': {'lookback': 20, 'momentum_threshold': 0.02}
        },
        {
            'name': 'mean_reversion', 
            'function': backtester.mean_reversion_strategy,
            'params': {'bb_period': 20, 'rsi_oversold': 30, 'rsi_overbought': 70}
        },
        {
            'name': 'breakout',
            'function': backtester.breakout_strategy,
            'params': {'lookback': 20, 'volume_multiplier': 1.5}
        }
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy['name']} strategy...")
        
        # Generate signals
        signals = strategy['function'](market_data, **strategy['params'])
        print(f"Generated {len(signals)} signals")
        
        # Execute backtest
        backtester.execute_backtest(
            signals, 
            market_data,
            position_size_pct=0.1,
            stop_loss=0.05,
            take_profit=0.10
        )
        
        # Calculate metrics
        metrics = backtester.calculate_metrics(market_data)
        results[strategy['name']] = metrics
        
        # Generate report
        report = backtester.generate_backtest_report(strategy['name'], metrics, market_data)
        
        # Save report
        report_filename = f"backtest_{strategy['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"Strategy: {strategy['name']}")
        print(f"  Total Return: {metrics.total_return:+.2f}%")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")
        print(f"  Win Rate: {metrics.win_rate:.1f}%")
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Report saved: {report_filename}")
    
    # Strategy comparison
    print(f"\nSTRATEGY COMPARISON:")
    print("-" * 40)
    print(f"{'Strategy':<15} {'Return':<10} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8}")
    print("-" * 40)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics.total_return:+7.2f}% {metrics.sharpe_ratio:7.2f} {metrics.max_drawdown:7.2f}% {metrics.total_trades:7d}")
    
    # Best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"\nBest Strategy (by Sharpe Ratio): {best_strategy[0]} ({best_strategy[1].sharpe_ratio:.2f})")
    
    # Save results summary
    summary = {
        'backtest_date': datetime.now().isoformat(),
        'symbols': symbols,
        'period': f"{start_date} to {end_date}",
        'results': {name: asdict(metrics) for name, metrics in results.items()},
        'best_strategy': best_strategy[0]
    }
    
    with open(f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\nBacktesting completed successfully!")

if __name__ == "__main__":
    main()