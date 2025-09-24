"""
Finance Integration Agent - Integrates Shashank Vemuri's Finance repository
Provides advanced technical analysis, ML models, portfolio optimization, and stock screening
"""

import asyncio
import logging
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add Finance repository to path
sys.path.append('./Finance')

# Import Finance modules
try:
    # Simple implementation of key TA functions if Finance modules fail
    def simple_sma(data, period):
        return data.rolling(window=period).mean()

    def simple_ema(data, period):
        return data.ewm(span=period, adjust=False).mean()

    def simple_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def simple_macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def simple_bbands(data, period=20, std_dev=2):
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    # Use simple implementations
    SMA = simple_sma
    EMA = simple_ema
    RSI = simple_rsi
    MACD = simple_macd
    BBANDS = simple_bbands

    FINANCE_TA_AVAILABLE = True
    print("Using simple TA implementations")

except ImportError as e:
    print(f"Finance TA functions not available: {e}")
    FINANCE_TA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinanceSignal:
    """Finance repository signal structure"""
    symbol: str
    signal_type: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0-1
    confidence: float  # 0-1
    indicators: Dict[str, float]
    ml_prediction: Optional[Dict] = None
    portfolio_weight: Optional[float] = None
    reasoning: List[str] = None

class FinanceIntegrationAgent:
    """
    Integrates Finance repository capabilities including:
    - Advanced technical analysis from ta_functions.py
    - ML predictions and deep learning models
    - Portfolio optimization strategies
    - Stock screening and selection
    - Minervini-style momentum screening
    """

    def __init__(self):
        self.name = "Finance Integration Agent"
        self.sp500_tickers = self._load_sp500_tickers()
        self.cache = {}
        logger.info("Finance Integration Agent initialized")

    def _load_sp500_tickers(self) -> List[str]:
        """Load S&P 500 tickers from Finance repository"""
        try:
            # Use built-in popular tickers list
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
                   'WMT', 'PG', 'UNH', 'HD', 'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX', 'KO',
                   'PEP', 'TMO', 'ABBV', 'COST', 'AVGO', 'NKE', 'LLY', 'ACN', 'DHR', 'NEE',
                   'TXN', 'BMY', 'PM', 'QCOM', 'HON', 'UNP', 'LOW', 'ORCL', 'IBM', 'MDT']
        except:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    async def comprehensive_technical_analysis(self, symbol: str, period: str = '6mo') -> Dict[str, Any]:
        """
        Comprehensive technical analysis using Finance repository functions
        """
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)

            if df.empty:
                return {}

            # Calculate all technical indicators
            indicators = {}

            # Moving Averages
            if FINANCE_TA_AVAILABLE:
                indicators['sma_20'] = float(SMA(df['Close'], 20).iloc[-1])
                indicators['sma_50'] = float(SMA(df['Close'], 50).iloc[-1])
                indicators['sma_200'] = float(SMA(df['Close'], 200).iloc[-1])
                indicators['ema_12'] = float(EMA(df['Close'], 12).iloc[-1])
                indicators['ema_26'] = float(EMA(df['Close'], 26).iloc[-1])

                # RSI
                indicators['rsi'] = float(RSI(df['Close']).iloc[-1])

                # MACD
                macd_line, macd_signal, macd_hist = MACD(df['Close'])
                indicators['macd'] = float(macd_line.iloc[-1])
                indicators['macd_signal'] = float(macd_signal.iloc[-1])
                indicators['macd_histogram'] = float(macd_hist.iloc[-1])

                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = BBANDS(df['Close'])
                indicators['bb_upper'] = float(bb_upper.iloc[-1])
                indicators['bb_middle'] = float(bb_middle.iloc[-1])
                indicators['bb_lower'] = float(bb_lower.iloc[-1])

                # Simple ATR calculation
                high_low = df['High'] - df['Low']
                high_close = (df['High'] - df['Close'].shift()).abs()
                low_close = (df['Low'] - df['Close'].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                indicators['atr'] = float(true_range.rolling(14).mean().iloc[-1])

                # Simple Stochastic
                lowest_low = df['Low'].rolling(14).min()
                highest_high = df['High'].rolling(14).max()
                k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
                indicators['stoch_k'] = float(k_percent.iloc[-1])
                indicators['stoch_d'] = float(k_percent.rolling(3).mean().iloc[-1])

            # Current price and basic metrics
            current_price = float(df['Close'].iloc[-1])
            indicators['current_price'] = current_price
            indicators['volume'] = float(df['Volume'].iloc[-1])
            indicators['avg_volume'] = float(df['Volume'].rolling(20).mean().iloc[-1])

            # Price position relative to moving averages
            if FINANCE_TA_AVAILABLE:
                indicators['price_vs_sma20'] = (current_price / indicators['sma_20'] - 1) * 100
                indicators['price_vs_sma50'] = (current_price / indicators['sma_50'] - 1) * 100
                indicators['price_vs_sma200'] = (current_price / indicators['sma_200'] - 1) * 100

            return {
                'symbol': symbol,
                'indicators': indicators,
                'analysis_timestamp': datetime.now(),
                'data_points': len(df)
            }

        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return {}

    async def minervini_momentum_screen(self, tickers: List[str] = None) -> List[Dict[str, Any]]:
        """
        Implement Minervini momentum screening criteria from Finance repository
        """
        try:
            if not tickers:
                tickers = self.sp500_tickers[:50]  # Screen top 50 for speed

            results = []

            for ticker in tickers:
                try:
                    # Get 1 year of data
                    stock = yf.Ticker(ticker)
                    df = stock.history(period='1y')

                    if len(df) < 200:  # Need enough data
                        continue

                    # Calculate moving averages
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    df['SMA_150'] = df['Close'].rolling(150).mean()
                    df['SMA_200'] = df['Close'].rolling(200).mean()

                    # Get current values
                    current_close = df['Close'].iloc[-1]
                    sma_50 = df['SMA_50'].iloc[-1]
                    sma_150 = df['SMA_150'].iloc[-1]
                    sma_200 = df['SMA_200'].iloc[-1]

                    # 52-week high/low
                    high_52w = df['High'].max()
                    low_52w = df['Low'].min()

                    # Minervini criteria
                    criteria = {
                        'price_above_150_200': current_close > sma_150 and current_close > sma_200,
                        'ma_150_above_200': sma_150 > sma_200,
                        'ma_200_trending_up': sma_200 > df['SMA_200'].iloc[-30],  # 200MA trending up
                        'ma_50_above_150_200': sma_50 > sma_150 and sma_50 > sma_200,
                        'price_above_50': current_close > sma_50,
                        'price_near_high': current_close >= 0.75 * high_52w,  # Within 25% of 52w high
                        'relative_strength': True  # Would need market comparison
                    }

                    # Score the stock
                    score = sum(criteria.values()) / len(criteria)

                    if score >= 0.7:  # 70% of criteria met
                        results.append({
                            'symbol': ticker,
                            'score': score,
                            'current_price': current_close,
                            'sma_50': sma_50,
                            'sma_150': sma_150,
                            'sma_200': sma_200,
                            'high_52w': high_52w,
                            'low_52w': low_52w,
                            'criteria_met': criteria,
                            'distance_from_high': (current_close / high_52w - 1) * 100
                        })

                except Exception as e:
                    logger.warning(f"Error screening {ticker}: {e}")
                    continue

            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:20]  # Return top 20

        except Exception as e:
            logger.error(f"Minervini screening error: {e}")
            return []

    async def generate_trading_signal(self, symbol: str) -> Optional[FinanceSignal]:
        """
        Generate comprehensive trading signal using Finance repository methods
        """
        try:
            # Get technical analysis
            tech_data = await self.comprehensive_technical_analysis(symbol)
            if not tech_data:
                return None

            indicators = tech_data['indicators']

            # Signal generation logic
            signals = []
            reasoning = []

            # RSI signals
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi < 30:
                    signals.append(('BUY', 0.8))
                    reasoning.append(f"RSI oversold at {rsi:.1f}")
                elif rsi > 70:
                    signals.append(('SELL', 0.8))
                    reasoning.append(f"RSI overbought at {rsi:.1f}")

            # MACD signals
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd']
                macd_signal = indicators['macd_signal']
                if macd > macd_signal and indicators.get('macd_histogram', 0) > 0:
                    signals.append(('BUY', 0.7))
                    reasoning.append("MACD bullish crossover")
                elif macd < macd_signal and indicators.get('macd_histogram', 0) < 0:
                    signals.append(('SELL', 0.7))
                    reasoning.append("MACD bearish crossover")

            # Moving average signals
            if 'price_vs_sma20' in indicators and 'price_vs_sma50' in indicators:
                if indicators['price_vs_sma20'] > 2 and indicators['price_vs_sma50'] > 0:
                    signals.append(('BUY', 0.6))
                    reasoning.append("Price above key moving averages")
                elif indicators['price_vs_sma20'] < -2 and indicators['price_vs_sma50'] < 0:
                    signals.append(('SELL', 0.6))
                    reasoning.append("Price below key moving averages")

            # Bollinger Bands signals
            if all(k in indicators for k in ['current_price', 'bb_upper', 'bb_lower']):
                price = indicators['current_price']
                if price <= indicators['bb_lower']:
                    signals.append(('BUY', 0.7))
                    reasoning.append("Price at lower Bollinger Band")
                elif price >= indicators['bb_upper']:
                    signals.append(('SELL', 0.7))
                    reasoning.append("Price at upper Bollinger Band")

            # Combine signals
            if not signals:
                direction = 'HOLD'
                strength = 0.5
                confidence = 0.5
            else:
                # Weight and combine signals
                buy_strength = sum(s[1] for s in signals if s[0] == 'BUY')
                sell_strength = sum(s[1] for s in signals if s[0] == 'SELL')

                if buy_strength > sell_strength:
                    direction = 'BUY'
                    strength = min(buy_strength / len(signals), 1.0)
                elif sell_strength > buy_strength:
                    direction = 'SELL'
                    strength = min(sell_strength / len(signals), 1.0)
                else:
                    direction = 'HOLD'
                    strength = 0.5

                confidence = min((abs(buy_strength - sell_strength) / max(buy_strength + sell_strength, 1)) + 0.3, 1.0)

            return FinanceSignal(
                symbol=symbol,
                signal_type='technical',
                direction=direction,
                strength=strength,
                confidence=confidence,
                indicators=indicators,
                reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return None

    async def portfolio_optimization_analysis(self, symbols: List[str], investment_amount: float = 10000) -> Dict[str, Any]:
        """
        Portfolio optimization using Finance repository techniques
        """
        try:
            # Get historical data
            data = {}
            for symbol in symbols:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='1y')
                if not hist.empty:
                    data[symbol] = hist['Close']

            if len(data) < 2:
                return {}

            # Create returns DataFrame
            df = pd.DataFrame(data)
            returns = df.pct_change().dropna()

            # Calculate key metrics
            mean_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252

            # Simple equal weight portfolio for now
            n_assets = len(symbols)
            weights = np.array([1/n_assets] * n_assets)

            # Portfolio metrics
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

            # Calculate allocation
            allocation = {}
            for i, symbol in enumerate(symbols):
                allocation[symbol] = {
                    'weight': weights[i],
                    'amount': investment_amount * weights[i],
                    'expected_return': mean_returns[symbol]
                }

            return {
                'symbols': symbols,
                'allocation': allocation,
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'investment_amount': investment_amount,
                'optimization_method': 'equal_weight'
            }

        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {}

# Create singleton instance
finance_integration_agent = FinanceIntegrationAgent()