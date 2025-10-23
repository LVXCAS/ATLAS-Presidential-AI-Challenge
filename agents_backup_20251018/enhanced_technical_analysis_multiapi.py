"""
Enhanced Technical Analysis Agent (Multi-API Version)
Provides advanced indicators and signals using multiple data sources to avoid rate limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf

try:
    from finta import TA
    FINTA_AVAILABLE = True
except ImportError:
    FINTA_AVAILABLE = False

try:
    from scipy import stats
    from scipy.signal import argrelextrema
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from config.logging_config import get_logger

logger = get_logger(__name__)

# Import multi-API provider
try:
    from .multi_api_data_provider import multi_api_provider
    MULTI_API_AVAILABLE = True
except ImportError:
    MULTI_API_AVAILABLE = False

class EnhancedTechnicalAnalysisMultiAPI:
    """Advanced technical analysis with multiple API sources to avoid rate limits"""

    def __init__(self):
        self.indicators_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes

    async def get_comprehensive_analysis(self, symbol: str, period: str = "60d") -> Dict:
        """Get comprehensive technical analysis for a symbol using multiple API sources"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}"
            if self._is_cache_valid(cache_key):
                return self.indicators_cache[cache_key]

            # Try multi-API provider first (has built-in fallbacks)
            if MULTI_API_AVAILABLE:
                try:
                    logger.info(f"Getting data for {symbol} using multi-API provider")
                    analysis_data = await multi_api_provider.get_comprehensive_analysis(symbol, period)
                    if analysis_data and analysis_data.get('current_price', 0) > 0:
                        # Enhance the basic data with advanced analysis if possible
                        enhanced_data = await self._enhance_basic_data(symbol, analysis_data, period)

                        # Cache the result
                        self.indicators_cache[cache_key] = enhanced_data
                        self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                        return enhanced_data
                except Exception as e:
                    logger.warning(f"Multi-API provider failed for {symbol}: {e}")

            # Fallback to Yahoo Finance directly
            try:
                logger.info(f"Falling back to Yahoo Finance for {symbol}")
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)

                if not hist.empty and len(hist) >= 20:
                    analysis = await self._create_full_analysis(symbol, hist)
                    # Cache the result
                    self.indicators_cache[cache_key] = analysis
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                    return analysis
            except Exception as yf_error:
                logger.error(f"Yahoo Finance also failed for {symbol}: {yf_error}")

            # Last resort: return default analysis
            logger.warning(f"All data sources failed for {symbol}, returning default analysis")
            return self._get_default_analysis()

        except Exception as e:
            logger.error(f"Critical error in comprehensive analysis for {symbol}: {e}")
            return self._get_default_analysis()

    async def _enhance_basic_data(self, symbol: str, basic_data: Dict, period: str) -> Dict:
        """Enhance basic data from multi-API with additional analysis if possible"""
        try:
            # Try to get Yahoo Finance data for enhanced indicators
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")  # Shorter period to avoid issues

            if not hist.empty and len(hist) >= 10:
                # Calculate advanced indicators using Yahoo Finance data
                enhanced_indicators = self._calculate_core_indicators(hist)
                enhanced_signals = self._generate_trading_signals(hist, enhanced_indicators)
                enhanced_support_resistance = self._calculate_support_resistance(hist)

                # Merge with basic data
                basic_data['technical_indicators'].update(enhanced_indicators)
                basic_data['signals'].update(enhanced_signals)
                basic_data['support_resistance'].update(enhanced_support_resistance)

                logger.info(f"Enhanced basic data for {symbol} with Yahoo Finance indicators")

            return basic_data

        except Exception as e:
            logger.warning(f"Could not enhance data for {symbol}: {e}")
            return basic_data

    async def _create_full_analysis(self, symbol: str, hist: pd.DataFrame) -> Dict:
        """Create full analysis from historical data"""
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': float(hist['Close'].iloc[-1]),
            'technical_indicators': {},
            'signals': {},
            'support_resistance': {},
            'volatility_analysis': {},
            'momentum_analysis': {},
            'trend_analysis': {}
        }

        # Core Technical Indicators
        analysis['technical_indicators'] = self._calculate_core_indicators(hist)

        # Trading Signals
        analysis['signals'] = self._generate_trading_signals(hist, analysis['technical_indicators'])

        # Support and Resistance
        analysis['support_resistance'] = self._calculate_support_resistance(hist)

        # Volatility Analysis
        analysis['volatility_analysis'] = self._analyze_volatility(hist)

        # Momentum Analysis
        analysis['momentum_analysis'] = self._analyze_momentum(hist)

        # Trend Analysis
        analysis['trend_analysis'] = self._analyze_trend(hist)

        return analysis

    def _calculate_core_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate core technical indicators"""
        indicators = {}

        try:
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50

            # MACD
            ema_12 = hist['Close'].ewm(span=12).mean()
            ema_26 = hist['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1] if not macd.empty else 0
            indicators['macd_signal'] = macd_signal.iloc[-1] if not macd_signal.empty else 0

            # Moving Averages
            indicators['sma_20'] = hist['Close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else indicators['sma_20']

            # Bollinger Bands
            sma_20 = hist['Close'].rolling(20).mean()
            std_20 = hist['Close'].rolling(20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]

            # Volume indicators
            indicators['volume_sma'] = hist['Volume'].mean()
            indicators['volume_ratio'] = hist['Volume'].iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1.0

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Return defaults
            indicators = {
                'rsi': 50, 'macd': 0, 'macd_signal': 0,
                'sma_20': hist['Close'].iloc[-1], 'sma_50': hist['Close'].iloc[-1],
                'bb_upper': hist['Close'].iloc[-1] * 1.02, 'bb_lower': hist['Close'].iloc[-1] * 0.98,
                'volume_sma': hist['Volume'].mean(), 'volume_ratio': 1.0
            }

        return indicators

    def _generate_trading_signals(self, hist: pd.DataFrame, indicators: Dict) -> Dict:
        """Generate trading signals"""
        signals = {
            'overall_signal': 'NEUTRAL',
            'signal_strength': 0.0,
            'confidence': 0.5,
            'bullish_factors': [],
            'bearish_factors': []
        }

        try:
            current_price = hist['Close'].iloc[-1]
            bullish_count = 0
            bearish_count = 0

            # RSI signals
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                signals['bullish_factors'].append('oversold_rsi')
                bullish_count += 1
            elif rsi > 70:
                signals['bearish_factors'].append('overbought_rsi')
                bearish_count += 1

            # MACD signals
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                signals['bullish_factors'].append('macd_bullish')
                bullish_count += 1
            else:
                signals['bearish_factors'].append('macd_bearish')
                bearish_count += 1

            # Moving average signals
            sma_20 = indicators.get('sma_20', current_price)
            if current_price > sma_20:
                signals['bullish_factors'].append('above_sma20')
                bullish_count += 1
            else:
                signals['bearish_factors'].append('below_sma20')
                bearish_count += 1

            # Determine overall signal
            if bullish_count > bearish_count:
                signals['overall_signal'] = 'BULLISH'
                signals['signal_strength'] = min((bullish_count - bearish_count) / 3.0, 1.0)
            elif bearish_count > bullish_count:
                signals['overall_signal'] = 'BEARISH'
                signals['signal_strength'] = min((bearish_count - bullish_count) / 3.0, 1.0)

            signals['confidence'] = 0.3 + (signals['signal_strength'] * 0.4)

        except Exception as e:
            logger.error(f"Error generating signals: {e}")

        return signals

    def _calculate_support_resistance(self, hist: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        support_resistance = {
            'nearest_support': None,
            'nearest_resistance': None,
            'support_levels': [],
            'resistance_levels': []
        }

        try:
            if len(hist) < 20:
                return support_resistance

            # Simple support/resistance based on recent highs and lows
            recent_high = hist['High'].rolling(10).max().max()
            recent_low = hist['Low'].rolling(10).min().min()
            current_price = hist['Close'].iloc[-1]

            if recent_high > current_price:
                support_resistance['nearest_resistance'] = recent_high
            if recent_low < current_price:
                support_resistance['nearest_support'] = recent_low

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")

        return support_resistance

    def _analyze_volatility(self, hist: pd.DataFrame) -> Dict:
        """Analyze volatility"""
        try:
            returns = hist['Close'].pct_change().dropna()
            realized_vol = returns.std() * 100 * (252 ** 0.5)

            return {
                'realized_vol_20d': realized_vol,
                'vol_percentile': 0.5,
                'vol_trend': 'STABLE',
                'vol_regime': 'NORMAL' if 15 <= realized_vol <= 35 else 'HIGH' if realized_vol > 35 else 'LOW'
            }
        except:
            return {'realized_vol_20d': 20.0, 'vol_percentile': 0.5, 'vol_trend': 'STABLE', 'vol_regime': 'NORMAL'}

    def _analyze_momentum(self, hist: pd.DataFrame) -> Dict:
        """Analyze momentum"""
        try:
            current_price = hist['Close'].iloc[-1]
            price_5d_ago = hist['Close'].iloc[-6] if len(hist) >= 6 else current_price
            price_20d_ago = hist['Close'].iloc[-21] if len(hist) >= 21 else current_price

            momentum_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
            momentum_20d = ((current_price - price_20d_ago) / price_20d_ago) * 100

            return {
                'price_momentum_5d': momentum_5d,
                'price_momentum_20d': momentum_20d,
                'momentum_strength': 'STRONG' if abs(momentum_5d) > 3 else 'MODERATE' if abs(momentum_5d) > 1 else 'WEAK',
                'acceleration': momentum_5d - momentum_20d if len(hist) >= 21 else 0
            }
        except:
            return {'price_momentum_5d': 0, 'price_momentum_20d': 0, 'momentum_strength': 'WEAK', 'acceleration': 0}

    def _analyze_trend(self, hist: pd.DataFrame) -> Dict:
        """Analyze trend"""
        try:
            # Simple trend analysis based on moving averages
            sma_10 = hist['Close'].rolling(10).mean().iloc[-1]
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            current_price = hist['Close'].iloc[-1]

            if current_price > sma_20 > sma_10:
                trend = 'STRONG_UPTREND'
            elif current_price > sma_20:
                trend = 'UPTREND'
            elif current_price < sma_20 < sma_10:
                trend = 'STRONG_DOWNTREND'
            elif current_price < sma_20:
                trend = 'DOWNTREND'
            else:
                trend = 'SIDEWAYS'

            return {'primary_trend': trend, 'trend_strength': 0.5}
        except:
            return {'primary_trend': 'SIDEWAYS', 'trend_strength': 0.5}

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]

    def _get_default_analysis(self) -> Dict:
        """Return default analysis when all data sources fail"""
        return {
            'symbol': 'UNKNOWN',
            'timestamp': datetime.now(),
            'current_price': 100.0,
            'technical_indicators': {
                'rsi': 50, 'macd': 0, 'macd_signal': 0,
                'sma_20': 100.0, 'sma_50': 100.0,
                'bb_upper': 102.0, 'bb_lower': 98.0,
                'volume_sma': 1000000, 'volume_ratio': 1.0
            },
            'signals': {
                'overall_signal': 'NEUTRAL',
                'signal_strength': 0.0,
                'confidence': 0.3,
                'bullish_factors': [],
                'bearish_factors': []
            },
            'support_resistance': {
                'nearest_support': None,
                'nearest_resistance': None,
                'support_levels': [],
                'resistance_levels': []
            },
            'volatility_analysis': {
                'realized_vol_20d': 20.0,
                'vol_percentile': 0.5,
                'vol_trend': 'STABLE',
                'vol_regime': 'NORMAL'
            },
            'momentum_analysis': {
                'price_momentum_5d': 0,
                'price_momentum_20d': 0,
                'momentum_strength': 'WEAK',
                'acceleration': 0
            },
            'trend_analysis': {
                'primary_trend': 'SIDEWAYS',
                'trend_strength': 0.5
            }
        }

# Create global instance
enhanced_technical_analysis_multiapi = EnhancedTechnicalAnalysisMultiAPI()