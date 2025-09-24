"""
Enhanced Technical Analysis Agent
Provides advanced indicators and signals using professional libraries
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

class EnhancedTechnicalAnalysis:
    """Advanced technical analysis with professional indicators"""
    
    def __init__(self):
        self.indicators_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
        
    async def get_comprehensive_analysis(self, symbol: str, period: str = "60d") -> Dict:
        """Get comprehensive technical analysis for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}"
            if self._is_cache_valid(cache_key):
                return self.indicators_cache[cache_key]
            
            # Get market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty or len(hist) < 20:
                return self._get_default_analysis()
            
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
            
            # Advanced Signals
            analysis['signals'] = self._generate_trading_signals(hist, analysis['technical_indicators'])
            
            # Support & Resistance
            analysis['support_resistance'] = self._find_support_resistance(hist)
            
            # Volatility Analysis
            analysis['volatility_analysis'] = self._analyze_volatility(hist)
            
            # Momentum Analysis
            analysis['momentum_analysis'] = self._analyze_momentum(hist)
            
            # Trend Analysis
            analysis['trend_analysis'] = self._analyze_trend(hist)
            
            # Cache results
            self.indicators_cache[cache_key] = analysis
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return self._get_default_analysis()
    
    def _calculate_core_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate core technical indicators"""
        indicators = {}
        
        try:
            # Price-based indicators
            indicators['sma_20'] = hist['Close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
            indicators['ema_12'] = hist['Close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = hist['Close'].ewm(span=26).mean().iloc[-1]
            
            # Volatility indicators
            indicators['bb_upper'], indicators['bb_lower'] = self._bollinger_bands(hist['Close'])
            indicators['atr'] = self._average_true_range(hist)
            
            # Using finta if available
            if FINTA_AVAILABLE:
                try:
                    indicators['rsi'] = float(TA.RSI(hist, period=14).iloc[-1])
                    indicators['macd'] = float(TA.MACD(hist)['MACD'].iloc[-1])
                    indicators['macd_signal'] = float(TA.MACD(hist)['SIGNAL'].iloc[-1])
                    
                    # Fix STOCH indicator - handle different column naming
                    stoch_result = TA.STOCH(hist)
                    if isinstance(stoch_result, pd.DataFrame):
                        # Try different possible column names
                        if '%K' in stoch_result.columns:
                            indicators['stoch_k'] = float(stoch_result['%K'].iloc[-1])
                        elif 'K' in stoch_result.columns:
                            indicators['stoch_k'] = float(stoch_result['K'].iloc[-1])
                        elif len(stoch_result.columns) > 0:
                            indicators['stoch_k'] = float(stoch_result.iloc[-1, 0])
                    else:
                        indicators['stoch_k'] = float(stoch_result.iloc[-1])
                    
                    indicators['williams_r'] = float(TA.WILLIAMS(hist).iloc[-1])
                    indicators['cci'] = float(TA.CCI(hist).iloc[-1])
                except Exception as e:
                    logger.warning(f"Finta indicators error: {e}")
                    # Set fallback values for failed indicators
                    if 'stoch_k' not in indicators:
                        indicators['stoch_k'] = 50.0  # Neutral value
            else:
                # Fallback calculations
                indicators['rsi'] = self._calculate_rsi(hist['Close'])
                indicators['macd'], indicators['macd_signal'] = self._calculate_macd(hist['Close'])
            
            # Volume indicators if available
            if 'Volume' in hist.columns:
                indicators['volume_sma'] = hist['Volume'].rolling(20).mean().iloc[-1]
                indicators['volume_ratio'] = hist['Volume'].iloc[-1] / indicators['volume_sma']
                
                if FINTA_AVAILABLE:
                    try:
                        indicators['obv'] = float(TA.OBV(hist).iloc[-1])
                        indicators['vwap'] = float(TA.VWAP(hist).iloc[-1])
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"Core indicators calculation error: {e}")
            
        return indicators
    
    def _generate_trading_signals(self, hist: pd.DataFrame, indicators: Dict) -> Dict:
        """Generate trading signals based on indicators"""
        signals = {
            'overall_signal': 'NEUTRAL',
            'signal_strength': 0.0,
            'bullish_factors': [],
            'bearish_factors': [],
            'confidence': 0.5
        }
        
        try:
            bullish_count = 0
            bearish_count = 0
            total_signals = 0
            
            current_price = hist['Close'].iloc[-1]
            
            # Price vs Moving Averages
            if indicators.get('sma_20'):
                total_signals += 1
                if current_price > indicators['sma_20']:
                    bullish_count += 1
                    signals['bullish_factors'].append(f"Price above SMA20 (${indicators['sma_20']:.2f})")
                else:
                    bearish_count += 1
                    signals['bearish_factors'].append(f"Price below SMA20 (${indicators['sma_20']:.2f})")
            
            # RSI Analysis
            if indicators.get('rsi'):
                total_signals += 1
                rsi = indicators['rsi']
                if rsi < 30:
                    bullish_count += 1
                    signals['bullish_factors'].append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    bearish_count += 1
                    signals['bearish_factors'].append(f"RSI overbought ({rsi:.1f})")
                elif 40 <= rsi <= 60:
                    # Neutral zone
                    pass
                elif rsi > 50:
                    bullish_count += 0.5
                else:
                    bearish_count += 0.5
            
            # MACD Analysis
            if indicators.get('macd') and indicators.get('macd_signal'):
                total_signals += 1
                macd_diff = indicators['macd'] - indicators['macd_signal']
                if macd_diff > 0:
                    bullish_count += 1
                    signals['bullish_factors'].append("MACD above signal line")
                else:
                    bearish_count += 1
                    signals['bearish_factors'].append("MACD below signal line")
            
            # Bollinger Bands
            if indicators.get('bb_upper') and indicators.get('bb_lower'):
                total_signals += 1
                bb_position = (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
                if bb_position < 0.2:
                    bullish_count += 1
                    signals['bullish_factors'].append("Price near lower Bollinger Band")
                elif bb_position > 0.8:
                    bearish_count += 1
                    signals['bearish_factors'].append("Price near upper Bollinger Band")
            
            # Volume confirmation
            if indicators.get('volume_ratio'):
                if indicators['volume_ratio'] > 1.5:
                    # High volume - strengthen the dominant signal
                    if bullish_count > bearish_count:
                        signals['bullish_factors'].append(f"High volume confirmation ({indicators['volume_ratio']:.1f}x)")
                    else:
                        signals['bearish_factors'].append(f"High volume confirmation ({indicators['volume_ratio']:.1f}x)")
            
            # Calculate overall signal
            if total_signals > 0:
                bullish_ratio = bullish_count / total_signals
                bearish_ratio = bearish_count / total_signals
                
                signals['signal_strength'] = abs(bullish_ratio - bearish_ratio)
                signals['confidence'] = min(0.95, 0.5 + (signals['signal_strength'] * 0.5))
                
                if bullish_ratio > bearish_ratio + 0.2:
                    signals['overall_signal'] = 'BULLISH'
                elif bearish_ratio > bullish_ratio + 0.2:
                    signals['overall_signal'] = 'BEARISH'
                else:
                    signals['overall_signal'] = 'NEUTRAL'
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
        
        return signals
    
    def _find_support_resistance(self, hist: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        support_resistance = {
            'support_levels': [],
            'resistance_levels': [],
            'nearest_support': None,
            'nearest_resistance': None
        }
        
        try:
            if not SCIPY_AVAILABLE or len(hist) < 20:
                return support_resistance
            
            # Use local maxima and minima
            highs = hist['High'].values
            lows = hist['Low'].values
            current_price = hist['Close'].iloc[-1]
            
            # Find local maxima (resistance) and minima (support)
            resistance_indices = argrelextrema(highs, np.greater, order=5)[0]
            support_indices = argrelextrema(lows, np.less, order=5)[0]
            
            # Get resistance levels
            if len(resistance_indices) > 0:
                resistance_levels = [highs[i] for i in resistance_indices if highs[i] > current_price]
                resistance_levels.sort()
                support_resistance['resistance_levels'] = resistance_levels[:3]  # Top 3
                if resistance_levels:
                    support_resistance['nearest_resistance'] = resistance_levels[0]
            
            # Get support levels  
            if len(support_indices) > 0:
                support_levels = [lows[i] for i in support_indices if lows[i] < current_price]
                support_levels.sort(reverse=True)
                support_resistance['support_levels'] = support_levels[:3]  # Top 3
                if support_levels:
                    support_resistance['nearest_support'] = support_levels[0]
            
        except Exception as e:
            logger.error(f"Support/resistance calculation error: {e}")
        
        return support_resistance
    
    def _analyze_volatility(self, hist: pd.DataFrame) -> Dict:
        """Analyze volatility patterns"""
        volatility = {
            'realized_vol_20d': 0.0,
            'vol_percentile': 0.5,
            'vol_trend': 'STABLE',
            'vol_regime': 'NORMAL'
        }
        
        try:
            returns = hist['Close'].pct_change().dropna()
            
            if len(returns) >= 20:
                # 20-day realized volatility
                volatility['realized_vol_20d'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
                
                # Historical volatility for percentile
                if len(returns) >= 60:
                    historical_vols = returns.rolling(20).std() * np.sqrt(252) * 100
                    current_vol = volatility['realized_vol_20d']
                    volatility['vol_percentile'] = (historical_vols <= current_vol).mean()
                
                # Volatility trend
                recent_vol = returns.tail(10).std() * np.sqrt(252) * 100
                older_vol = returns.iloc[-20:-10].std() * np.sqrt(252) * 100
                
                if recent_vol > older_vol * 1.2:
                    volatility['vol_trend'] = 'INCREASING'
                elif recent_vol < older_vol * 0.8:
                    volatility['vol_trend'] = 'DECREASING'
                
                # Volatility regime
                if volatility['realized_vol_20d'] > 35:
                    volatility['vol_regime'] = 'HIGH'
                elif volatility['realized_vol_20d'] < 15:
                    volatility['vol_regime'] = 'LOW'
                else:
                    volatility['vol_regime'] = 'NORMAL'
        
        except Exception as e:
            logger.error(f"Volatility analysis error: {e}")
        
        return volatility
    
    def _analyze_momentum(self, hist: pd.DataFrame) -> Dict:
        """Analyze price momentum"""
        momentum = {
            'price_momentum_5d': 0.0,
            'price_momentum_20d': 0.0,
            'acceleration': 0.0,
            'momentum_strength': 'WEAK'
        }
        
        try:
            if len(hist) >= 20:
                current_price = hist['Close'].iloc[-1]
                price_5d_ago = hist['Close'].iloc[-6] if len(hist) >= 6 else current_price
                price_20d_ago = hist['Close'].iloc[-21] if len(hist) >= 21 else current_price
                
                momentum['price_momentum_5d'] = (current_price - price_5d_ago) / price_5d_ago * 100
                momentum['price_momentum_20d'] = (current_price - price_20d_ago) / price_20d_ago * 100
                
                # Momentum acceleration
                momentum['acceleration'] = momentum['price_momentum_5d'] - momentum['price_momentum_20d']
                
                # Momentum strength
                abs_momentum = abs(momentum['price_momentum_5d'])
                if abs_momentum > 5:
                    momentum['momentum_strength'] = 'STRONG'
                elif abs_momentum > 2:
                    momentum['momentum_strength'] = 'MODERATE'
                else:
                    momentum['momentum_strength'] = 'WEAK'
        
        except Exception as e:
            logger.error(f"Momentum analysis error: {e}")
        
        return momentum
    
    def _analyze_trend(self, hist: pd.DataFrame) -> Dict:
        """Analyze trend using multiple timeframes"""
        trend = {
            'short_term_trend': 'NEUTRAL',
            'medium_term_trend': 'NEUTRAL',
            'long_term_trend': 'NEUTRAL',
            'trend_strength': 0.0,
            'trend_consistency': 0.0
        }
        
        try:
            if len(hist) < 20:
                return trend
            
            current_price = hist['Close'].iloc[-1]
            
            # Short-term trend (5-day)
            if len(hist) >= 5:
                sma_5 = hist['Close'].rolling(5).mean().iloc[-1]
                if current_price > sma_5 * 1.01:
                    trend['short_term_trend'] = 'BULLISH'
                elif current_price < sma_5 * 0.99:
                    trend['short_term_trend'] = 'BEARISH'
            
            # Medium-term trend (20-day)
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            if current_price > sma_20 * 1.02:
                trend['medium_term_trend'] = 'BULLISH'
            elif current_price < sma_20 * 0.98:
                trend['medium_term_trend'] = 'BEARISH'
            
            # Long-term trend (50-day)
            if len(hist) >= 50:
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                if current_price > sma_50 * 1.03:
                    trend['long_term_trend'] = 'BULLISH'
                elif current_price < sma_50 * 0.97:
                    trend['long_term_trend'] = 'BEARISH'
            
            # Calculate trend strength and consistency
            trends = [trend['short_term_trend'], trend['medium_term_trend'], trend['long_term_trend']]
            bullish_count = trends.count('BULLISH')
            bearish_count = trends.count('BEARISH')
            
            trend['trend_strength'] = max(bullish_count, bearish_count) / len(trends)
            
            if bullish_count == len(trends) or bearish_count == len(trends):
                trend['trend_consistency'] = 1.0
            elif bullish_count > bearish_count or bearish_count > bullish_count:
                trend['trend_consistency'] = 0.67
            else:
                trend['trend_consistency'] = 0.33
        
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
        
        return trend
    
    # Helper methods for calculations
    def _bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return float(upper.iloc[-1]), float(lower.iloc[-1])
    
    def _average_true_range(self, hist: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = hist['High']
        low = hist['Low']
        close = hist['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return float(true_range.rolling(period).mean().iloc[-1])
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return float(100 - (100 / (1 + rs)).iloc[-1])
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD manually"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return float(macd.iloc[-1]), float(macd_signal.iloc[-1])
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.indicators_cache:
            return False
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis when calculation fails"""
        return {
            'symbol': '',
            'timestamp': datetime.now(),
            'current_price': 0.0,
            'technical_indicators': {},
            'signals': {'overall_signal': 'NEUTRAL', 'signal_strength': 0.0, 'confidence': 0.5},
            'support_resistance': {},
            'volatility_analysis': {'realized_vol_20d': 20.0, 'vol_regime': 'NORMAL'},
            'momentum_analysis': {'momentum_strength': 'WEAK'},
            'trend_analysis': {'short_term_trend': 'NEUTRAL'}
        }

# Singleton instance
enhanced_technical_analysis = EnhancedTechnicalAnalysis()