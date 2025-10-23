#!/usr/bin/env python3
"""
Statistical Arbitrage Agent - HIVE TRADE
Pairs trading and statistical arbitrage using cointegration and mean reversion
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from collections import deque
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PairStatus(Enum):
    """Status of trading pairs"""
    SEARCHING = "SEARCHING"        # Looking for entry signal
    LONG_SPREAD = "LONG_SPREAD"    # Long the spread (short stronger, long weaker)
    SHORT_SPREAD = "SHORT_SPREAD"  # Short the spread (long stronger, short weaker)
    CLOSED = "CLOSED"              # Position closed
    ERROR = "ERROR"                # Error state

@dataclass
class TradingPair:
    """Statistical arbitrage trading pair"""
    symbol_a: str  # First symbol
    symbol_b: str  # Second symbol
    hedge_ratio: float  # Hedge ratio (beta coefficient)
    correlation: float  # Historical correlation
    cointegration_pvalue: float  # P-value of cointegration test
    half_life: float  # Mean reversion half-life in periods
    
    # Current state
    spread: float  # Current spread value
    zscore: float  # Z-score of current spread
    status: PairStatus = PairStatus.SEARCHING
    
    # Statistics
    spread_mean: float = 0.0
    spread_std: float = 1.0
    confidence: float = 0.0
    
    # Risk metrics
    max_spread_deviation: float = 0.0
    current_pnl: float = 0.0
    max_loss: float = 0.0
    
    def calculate_spread(self, price_a: float, price_b: float) -> float:
        """Calculate spread value"""
        self.spread = price_a - (self.hedge_ratio * price_b)
        return self.spread
    
    def calculate_zscore(self) -> float:
        """Calculate Z-score of current spread"""
        if self.spread_std <= 0:
            self.zscore = 0.0
        else:
            self.zscore = (self.spread - self.spread_mean) / self.spread_std
        return self.zscore

@dataclass
class ArbitrageSignal:
    """Arbitrage trading signal"""
    timestamp: datetime
    pair: TradingPair
    signal_type: str  # 'LONG_SPREAD', 'SHORT_SPREAD', 'CLOSE'
    confidence: float
    expected_return: float
    risk_score: float
    entry_threshold: float
    exit_threshold: float
    stop_loss: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol_a': self.pair.symbol_a,
            'symbol_b': self.pair.symbol_b,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'expected_return': self.expected_return,
            'risk_score': self.risk_score,
            'zscore': self.pair.zscore,
            'spread': self.pair.spread,
            'hedge_ratio': self.pair.hedge_ratio,
            'correlation': self.pair.correlation
        }

class StatisticalArbitrageAgent:
    """Advanced statistical arbitrage agent"""
    
    def __init__(self, symbols: List[str], lookback_period: int = 252):
        self.symbols = symbols
        self.lookback_period = lookback_period  # Number of periods for analysis
        
        # Trading parameters
        self.entry_zscore_threshold = 2.0    # Z-score threshold for entry
        self.exit_zscore_threshold = 0.5     # Z-score threshold for exit
        self.stop_loss_zscore = 3.5          # Stop loss Z-score
        self.min_correlation = 0.7           # Minimum correlation for pair trading
        self.max_cointegration_pvalue = 0.05 # Maximum p-value for cointegration
        self.min_half_life = 1               # Minimum half-life (periods)
        self.max_half_life = 50              # Maximum half-life (periods)
        
        # Data storage
        self.price_data: Dict[str, deque] = {}
        self.pairs: Dict[str, TradingPair] = {}
        self.active_signals: List[ArbitrageSignal] = []
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
        # Initialize price data storage
        for symbol in symbols:
            self.price_data[symbol] = deque(maxlen=lookback_period * 2)
        
        logger.info(f"Statistical Arbitrage Agent initialized")
        logger.info(f"Monitoring {len(symbols)} symbols for arbitrage opportunities")
        logger.info(f"Lookback period: {lookback_period} periods")

    def add_price_data(self, symbol: str, price: float, timestamp: datetime = None):
        """Add price data for a symbol"""
        if symbol not in self.price_data:
            self.price_data[symbol] = deque(maxlen=self.lookback_period * 2)
        
        timestamp = timestamp or datetime.now()
        self.price_data[symbol].append((timestamp, price))

    def calculate_cointegration(self, prices_a: np.array, prices_b: np.array) -> Tuple[float, float, float]:
        """Calculate cointegration statistics between two price series"""
        if len(prices_a) != len(prices_b) or len(prices_a) < 30:
            return 0.0, 1.0, float('inf')  # No cointegration
        
        # Fit linear regression to get hedge ratio
        X = prices_b.reshape(-1, 1)
        y = prices_a
        
        reg = LinearRegression()
        reg.fit(X, y)
        hedge_ratio = reg.coef_[0]
        
        # Calculate residuals (spread)
        residuals = y - hedge_ratio * prices_b
        
        # Augmented Dickey-Fuller test on residuals (simplified)
        # In practice, you'd use statsmodels.tsa.stattools.adfuller
        n = len(residuals)
        if n < 10:
            return hedge_ratio, 1.0, float('inf')
        
        # Simplified stationarity test
        # Calculate first differences
        diff_residuals = np.diff(residuals)
        lagged_residuals = residuals[:-1]
        
        # Regression: diff = alpha * lagged + error
        if np.std(lagged_residuals) > 1e-8:
            correlation = np.corrcoef(diff_residuals, lagged_residuals)[0, 1]
            # Convert correlation to p-value (simplified)
            t_stat = correlation * np.sqrt((n-2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        else:
            p_value = 1.0
        
        # Calculate half-life of mean reversion
        if correlation < 0:
            half_life = -np.log(2) / np.log(1 + correlation)
        else:
            half_life = float('inf')
        
        return hedge_ratio, p_value, half_life

    def find_cointegrated_pairs(self) -> List[TradingPair]:
        """Find all cointegrated pairs from available symbols"""
        pairs = []
        
        # Get symbols with sufficient data
        valid_symbols = [s for s in self.symbols 
                        if len(self.price_data.get(s, [])) >= self.lookback_period]
        
        if len(valid_symbols) < 2:
            return pairs
        
        logger.info(f"Analyzing {len(valid_symbols)} symbols for cointegration...")
        
        # Test all possible pairs
        for i, symbol_a in enumerate(valid_symbols):
            for j, symbol_b in enumerate(valid_symbols[i+1:], i+1):
                # Get price data
                data_a = list(self.price_data[symbol_a])[-self.lookback_period:]
                data_b = list(self.price_data[symbol_b])[-self.lookback_period:]
                
                if len(data_a) < self.lookback_period or len(data_b) < self.lookback_period:
                    continue
                
                # Extract prices
                prices_a = np.array([d[1] for d in data_a])
                prices_b = np.array([d[1] for d in data_b])
                
                # Calculate correlation
                correlation = np.corrcoef(prices_a, prices_b)[0, 1]
                
                if abs(correlation) < self.min_correlation:
                    continue
                
                # Test cointegration
                hedge_ratio, p_value, half_life = self.calculate_cointegration(prices_a, prices_b)
                
                if (p_value <= self.max_cointegration_pvalue and
                    self.min_half_life <= half_life <= self.max_half_life):
                    
                    # Calculate spread statistics
                    spread_values = prices_a - hedge_ratio * prices_b
                    spread_mean = np.mean(spread_values)
                    spread_std = np.std(spread_values)
                    
                    # Create trading pair
                    pair = TradingPair(
                        symbol_a=symbol_a,
                        symbol_b=symbol_b,
                        hedge_ratio=hedge_ratio,
                        correlation=correlation,
                        cointegration_pvalue=p_value,
                        half_life=half_life,
                        spread=spread_values[-1],  # Current spread
                        zscore=0.0,
                        spread_mean=spread_mean,
                        spread_std=spread_std,
                        confidence=1 - p_value  # Higher confidence for lower p-value
                    )
                    
                    # Calculate initial Z-score
                    pair.calculate_zscore()
                    
                    pairs.append(pair)
                    
                    logger.info(f"Found cointegrated pair: {symbol_a}/{symbol_b} "
                              f"(correlation: {correlation:.3f}, p-value: {p_value:.4f}, "
                              f"half-life: {half_life:.1f}, hedge-ratio: {hedge_ratio:.4f})")
        
        return pairs

    def update_pair_statistics(self, pair: TradingPair):
        """Update pair statistics with latest price data"""
        # Get recent price data
        if (pair.symbol_a not in self.price_data or 
            pair.symbol_b not in self.price_data):
            return
        
        data_a = list(self.price_data[pair.symbol_a])[-self.lookback_period:]
        data_b = list(self.price_data[pair.symbol_b])[-self.lookback_period:]
        
        if len(data_a) < 10 or len(data_b) < 10:
            return
        
        # Get latest prices
        latest_price_a = data_a[-1][1]
        latest_price_b = data_b[-1][1]
        
        # Update current spread and Z-score
        pair.calculate_spread(latest_price_a, latest_price_b)
        pair.calculate_zscore()
        
        # Update spread statistics using rolling window
        window_size = min(50, len(data_a))
        recent_data_a = data_a[-window_size:]
        recent_data_b = data_b[-window_size:]
        
        prices_a = np.array([d[1] for d in recent_data_a])
        prices_b = np.array([d[1] for d in recent_data_b])
        
        # Recalculate spread statistics
        spread_values = prices_a - pair.hedge_ratio * prices_b
        pair.spread_mean = np.mean(spread_values)
        pair.spread_std = np.std(spread_values)
        
        # Update max deviation
        pair.max_spread_deviation = max(pair.max_spread_deviation, abs(pair.zscore))

    def generate_arbitrage_signals(self) -> List[ArbitrageSignal]:
        """Generate arbitrage signals for all active pairs"""
        signals = []
        current_time = datetime.now()
        
        for pair_key, pair in self.pairs.items():
            # Update pair statistics
            self.update_pair_statistics(pair)
            
            signal = None
            
            # Entry signals
            if pair.status == PairStatus.SEARCHING:
                if pair.zscore > self.entry_zscore_threshold:
                    # Spread is too high - short the spread
                    # (sell stronger asset, buy weaker asset)
                    signal = ArbitrageSignal(
                        timestamp=current_time,
                        pair=pair,
                        signal_type='SHORT_SPREAD',
                        confidence=min(0.95, pair.confidence * (abs(pair.zscore) / self.entry_zscore_threshold)),
                        expected_return=self._calculate_expected_return(pair, 'SHORT_SPREAD'),
                        risk_score=self._calculate_risk_score(pair),
                        entry_threshold=self.entry_zscore_threshold,
                        exit_threshold=self.exit_zscore_threshold,
                        stop_loss=self.stop_loss_zscore
                    )
                    pair.status = PairStatus.SHORT_SPREAD
                    
                elif pair.zscore < -self.entry_zscore_threshold:
                    # Spread is too low - long the spread
                    # (buy stronger asset, sell weaker asset)
                    signal = ArbitrageSignal(
                        timestamp=current_time,
                        pair=pair,
                        signal_type='LONG_SPREAD',
                        confidence=min(0.95, pair.confidence * (abs(pair.zscore) / self.entry_zscore_threshold)),
                        expected_return=self._calculate_expected_return(pair, 'LONG_SPREAD'),
                        risk_score=self._calculate_risk_score(pair),
                        entry_threshold=-self.entry_zscore_threshold,
                        exit_threshold=-self.exit_zscore_threshold,
                        stop_loss=-self.stop_loss_zscore
                    )
                    pair.status = PairStatus.LONG_SPREAD
            
            # Exit signals
            elif pair.status == PairStatus.SHORT_SPREAD:
                if (pair.zscore <= self.exit_zscore_threshold or 
                    pair.zscore >= self.stop_loss_zscore):
                    # Close short spread position
                    signal = ArbitrageSignal(
                        timestamp=current_time,
                        pair=pair,
                        signal_type='CLOSE',
                        confidence=0.9,
                        expected_return=0.0,
                        risk_score=0.0,
                        entry_threshold=0.0,
                        exit_threshold=self.exit_zscore_threshold,
                        stop_loss=self.stop_loss_zscore
                    )
                    pair.status = PairStatus.SEARCHING
                    
            elif pair.status == PairStatus.LONG_SPREAD:
                if (pair.zscore >= -self.exit_zscore_threshold or 
                    pair.zscore <= -self.stop_loss_zscore):
                    # Close long spread position
                    signal = ArbitrageSignal(
                        timestamp=current_time,
                        pair=pair,
                        signal_type='CLOSE',
                        confidence=0.9,
                        expected_return=0.0,
                        risk_score=0.0,
                        entry_threshold=0.0,
                        exit_threshold=-self.exit_zscore_threshold,
                        stop_loss=-self.stop_loss_zscore
                    )
                    pair.status = PairStatus.SEARCHING
            
            if signal:
                signals.append(signal)
                logger.info(f"Generated signal: {signal.signal_type} for "
                          f"{pair.symbol_a}/{pair.symbol_b} (Z-score: {pair.zscore:.2f})")
        
        self.active_signals = signals
        return signals

    def _calculate_expected_return(self, pair: TradingPair, signal_type: str) -> float:
        """Calculate expected return for a signal"""
        # Simplified expected return calculation
        # Based on mean reversion assumption
        
        current_zscore = abs(pair.zscore)
        expected_reversion = current_zscore - self.exit_zscore_threshold
        
        # Convert Z-score to price movement
        expected_spread_change = expected_reversion * pair.spread_std
        
        # Estimate return as percentage of spread
        if pair.spread != 0:
            expected_return = abs(expected_spread_change / pair.spread)
        else:
            expected_return = 0.01  # 1% default
        
        return min(expected_return, 0.1)  # Cap at 10%

    def _calculate_risk_score(self, pair: TradingPair) -> float:
        """Calculate risk score for a pair"""
        base_risk = 0.3
        
        # Higher Z-score = higher risk
        zscore_risk = min(0.3, abs(pair.zscore) / 10)
        
        # Longer half-life = higher risk
        half_life_risk = min(0.2, pair.half_life / 100)
        
        # Lower correlation = higher risk
        correlation_risk = max(0, (1 - abs(pair.correlation)) * 0.2)
        
        # Higher p-value = higher risk
        p_value_risk = pair.cointegration_pvalue * 0.3
        
        total_risk = base_risk + zscore_risk + half_life_risk + correlation_risk + p_value_risk
        return min(total_risk, 1.0)

    def simulate_trade_execution(self, signal: ArbitrageSignal) -> Dict:
        """Simulate execution of arbitrage trade"""
        execution_time = datetime.now()
        
        # Get current prices
        if (signal.pair.symbol_a not in self.price_data or
            signal.pair.symbol_b not in self.price_data):
            return {}
        
        price_a = list(self.price_data[signal.pair.symbol_a])[-1][1]
        price_b = list(self.price_data[signal.pair.symbol_b])[-1][1]
        
        # Calculate position sizes (simplified)
        capital_allocation = 10000  # $10k per trade
        
        if signal.signal_type == 'LONG_SPREAD':
            # Buy symbol A, sell symbol B
            size_a = capital_allocation / (price_a + signal.pair.hedge_ratio * price_b)
            size_b = -size_a * signal.pair.hedge_ratio
            side_a, side_b = 'buy', 'sell'
            
        elif signal.signal_type == 'SHORT_SPREAD':
            # Sell symbol A, buy symbol B
            size_a = -capital_allocation / (price_a + signal.pair.hedge_ratio * price_b)
            size_b = -size_a * signal.pair.hedge_ratio
            side_a, side_b = 'sell', 'buy'
            
        else:  # CLOSE
            # Close existing position (simplified)
            size_a = 0
            size_b = 0
            side_a, side_b = 'close', 'close'
        
        # Record trade
        trade = {
            'timestamp': execution_time.isoformat(),
            'signal_type': signal.signal_type,
            'symbol_a': signal.pair.symbol_a,
            'symbol_b': signal.pair.symbol_b,
            'price_a': price_a,
            'price_b': price_b,
            'size_a': size_a,
            'size_b': size_b,
            'side_a': side_a,
            'side_b': side_b,
            'hedge_ratio': signal.pair.hedge_ratio,
            'entry_zscore': signal.pair.zscore,
            'confidence': signal.confidence,
            'expected_return': signal.expected_return,
            'risk_score': signal.risk_score
        }
        
        self.trade_history.append(trade)
        self.total_trades += 1
        
        logger.info(f"Executed trade: {signal.signal_type} "
                   f"{signal.pair.symbol_a}/{signal.pair.symbol_b} "
                   f"at Z-score {signal.pair.zscore:.2f}")
        
        return trade

    def get_arbitrage_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive arbitrage dashboard"""
        current_time = datetime.now()
        
        # Update all pairs
        for pair in self.pairs.values():
            self.update_pair_statistics(pair)
        
        # Pair summary
        pairs_summary = []
        for pair_key, pair in self.pairs.items():
            pairs_summary.append({
                'symbol_a': pair.symbol_a,
                'symbol_b': pair.symbol_b,
                'status': pair.status.value,
                'zscore': round(pair.zscore, 3),
                'spread': round(pair.spread, 4),
                'hedge_ratio': round(pair.hedge_ratio, 4),
                'correlation': round(pair.correlation, 3),
                'half_life': round(pair.half_life, 1),
                'p_value': round(pair.cointegration_pvalue, 4),
                'confidence': round(pair.confidence, 3),
                'max_deviation': round(pair.max_spread_deviation, 2)
            })
        
        # Active signals summary
        signals_summary = []
        for signal in self.active_signals:
            signals_summary.append(signal.to_dict())
        
        # Performance metrics
        recent_trades = self.trade_history[-20:] if self.trade_history else []
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        return {
            'timestamp': current_time.isoformat(),
            'strategy': 'Statistical Arbitrage',
            'pairs_monitored': len(self.pairs),
            'active_positions': len([p for p in self.pairs.values() 
                                   if p.status != PairStatus.SEARCHING]),
            'performance_metrics': {
                'total_trades': self.total_trades,
                'win_rate': round(win_rate, 1),
                'total_pnl': round(self.total_pnl, 2),
                'max_drawdown': round(self.max_drawdown, 2),
                'sharpe_ratio': round(self.sharpe_ratio, 3)
            },
            'pairs': pairs_summary,
            'active_signals': signals_summary,
            'recent_trades': recent_trades[-10:],  # Last 10 trades
            'parameters': {
                'entry_threshold': self.entry_zscore_threshold,
                'exit_threshold': self.exit_zscore_threshold,
                'stop_loss': self.stop_loss_zscore,
                'min_correlation': self.min_correlation,
                'max_half_life': self.max_half_life,
                'lookback_period': self.lookback_period
            }
        }

    def run_arbitrage_scan(self):
        """Run complete arbitrage opportunity scan"""
        logger.info("Running statistical arbitrage scan...")
        
        # Find cointegrated pairs
        new_pairs = self.find_cointegrated_pairs()
        
        # Update pairs dictionary
        for pair in new_pairs:
            pair_key = f"{pair.symbol_a}/{pair.symbol_b}"
            self.pairs[pair_key] = pair
        
        # Generate signals
        signals = self.generate_arbitrage_signals()
        
        # Execute signals (simulation)
        executed_trades = []
        for signal in signals:
            trade = self.simulate_trade_execution(signal)
            if trade:
                executed_trades.append(trade)
        
        logger.info(f"Scan completed: {len(new_pairs)} pairs found, "
                   f"{len(signals)} signals generated, {len(executed_trades)} trades executed")
        
        return {
            'pairs_found': len(new_pairs),
            'signals_generated': len(signals),
            'trades_executed': len(executed_trades),
            'pairs': new_pairs,
            'signals': signals,
            'trades': executed_trades
        }

def main():
    """Demonstrate statistical arbitrage agent"""
    print("HIVE TRADE - Statistical Arbitrage Agent Demo")
    print("=" * 55)
    
    # Initialize agent
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    agent = StatisticalArbitrageAgent(symbols, lookback_period=100)
    
    # Simulate price data
    print("\nSimulating historical price data...")
    np.random.seed(42)  # For reproducible results
    
    base_prices = {
        'AAPL': 180, 'MSFT': 340, 'GOOGL': 2800, 'AMZN': 3200,
        'TSLA': 220, 'NVDA': 450, 'META': 320, 'NFLX': 400
    }
    
    # Generate correlated price series
    n_periods = 150
    for i in range(n_periods):
        timestamp = datetime.now() - timedelta(days=n_periods-i)
        
        # Generate market factor
        market_return = np.random.normal(0, 0.02)
        
        for symbol in symbols:
            # Individual return with market correlation
            individual_return = np.random.normal(0, 0.03)
            beta = np.random.uniform(0.7, 1.3)
            
            total_return = beta * market_return + individual_return
            
            if i == 0:
                price = base_prices[symbol]
            else:
                previous_price = list(agent.price_data[symbol])[-1][1]
                price = previous_price * (1 + total_return)
            
            agent.add_price_data(symbol, price, timestamp)
    
    # Add some specific correlations for demo
    # Make AAPL and MSFT more correlated
    for i in range(50):
        timestamp = datetime.now() - timedelta(days=50-i)
        aapl_price = list(agent.price_data['AAPL'])[-1][1] if agent.price_data['AAPL'] else base_prices['AAPL']
        msft_price = list(agent.price_data['MSFT'])[-1][1] if agent.price_data['MSFT'] else base_prices['MSFT']
        
        # Correlated movement
        shock = np.random.normal(0, 0.025)
        aapl_new = aapl_price * (1 + shock + np.random.normal(0, 0.005))
        msft_new = msft_price * (1 + shock * 0.8 + np.random.normal(0, 0.005))
        
        agent.add_price_data('AAPL', aapl_new, timestamp)
        agent.add_price_data('MSFT', msft_new, timestamp)
    
    print(f"Generated {n_periods} periods of price data for {len(symbols)} symbols")
    
    # Run arbitrage scan
    print("\nRunning statistical arbitrage scan...")
    scan_results = agent.run_arbitrage_scan()
    
    # Get dashboard
    dashboard = agent.get_arbitrage_dashboard()
    
    # Display results
    print(f"\nSTATISTICAL ARBITRAGE RESULTS:")
    print(f"  Cointegrated pairs found: {scan_results['pairs_found']}")
    print(f"  Signals generated: {scan_results['signals_generated']}")
    print(f"  Trades executed: {scan_results['trades_executed']}")
    
    if dashboard['pairs']:
        print(f"\nCOINTEGRATED PAIRS:")
        for pair in dashboard['pairs'][:5]:  # Show top 5 pairs
            print(f"  {pair['symbol_a']}/{pair['symbol_b']}:")
            print(f"    Z-score: {pair['zscore']:+.2f}, Status: {pair['status']}")
            print(f"    Correlation: {pair['correlation']:.3f}, Half-life: {pair['half_life']:.1f}")
            print(f"    P-value: {pair['p_value']:.4f}, Confidence: {pair['confidence']:.3f}")
    
    if dashboard['active_signals']:
        print(f"\nACTIVE SIGNALS:")
        for signal in dashboard['active_signals']:
            print(f"  {signal['signal_type']}: {signal['symbol_a']}/{signal['symbol_b']}")
            print(f"    Z-score: {signal['zscore']:+.2f}, Confidence: {signal['confidence']:.2f}")
            print(f"    Expected Return: {signal['expected_return']:.2%}, Risk: {signal['risk_score']:.2f}")
    
    print(f"\nPERFORMANCE METRICS:")
    perf = dashboard['performance_metrics']
    print(f"  Total Trades: {perf['total_trades']}")
    print(f"  Win Rate: {perf['win_rate']:.1f}%")
    print(f"  Total P&L: ${perf['total_pnl']:,.2f}")
    print(f"  Max Drawdown: {perf['max_drawdown']:.2f}%")
    
    # Save results
    results_file = 'statistical_arbitrage_results.json'
    with open(results_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("Statistical Arbitrage Agent demonstration completed!")

if __name__ == "__main__":
    main()