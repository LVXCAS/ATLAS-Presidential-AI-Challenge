"""
Advanced Technical Analysis Agent with Enhanced Libraries
Integrates ta, arch, and numba for professional-grade analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Core libraries
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

try:
    import arch
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

import yfinance as yf
from config.logging_config import get_logger

logger = get_logger(__name__)

class AdvancedTechnicalAnalysis:
    """Enhanced technical analysis with professional-grade indicators and ML"""
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
        
    async def get_comprehensive_analysis(self, symbol: str, period: str = "60d") -> Dict:
        """Get comprehensive technical analysis with advanced indicators"""
        
        # Check cache
        cache_key = f"{symbol}_{period}_advanced"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Get price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
            
            # Convert to polars for faster processing if available
            if POLARS_AVAILABLE:
                df = pl.from_pandas(hist.reset_index())
                df = df.with_columns([
                    pl.col("Date").dt.date().alias("date"),
                    pl.col("Close").alias("close"),
                    pl.col("High").alias("high"),
                    pl.col("Low").alias("low"),
                    pl.col("Open").alias("open"),
                    pl.col("Volume").alias("volume")
                ])
                hist_pd = df.to_pandas()
            else:
                hist_pd = hist.copy()
            
            current_price = float(hist['Close'].iloc[-1])
            
            analysis = {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'technical_indicators': {},
                'advanced_indicators': {},
                'volatility_analysis': {},
                'momentum_analysis': {},
                'signals': {},
                'support_resistance': {},
                'pattern_recognition': {},
                'regime_detection': {}
            }
            
            # Enhanced Technical Indicators
            analysis['technical_indicators'] = await self._calculate_enhanced_indicators(hist)
            
            # Advanced Volatility Analysis using ARCH
            analysis['volatility_analysis'] = await self._calculate_volatility_analysis(hist)
            
            # Enhanced Momentum Analysis
            analysis['momentum_analysis'] = await self._calculate_momentum_analysis(hist)
            
            # Advanced Signal Generation
            analysis['signals'] = await self._generate_advanced_signals(hist, analysis)
            
            # Support/Resistance with ML
            analysis['support_resistance'] = await self._calculate_advanced_support_resistance(hist)
            
            # Pattern Recognition
            analysis['pattern_recognition'] = await self._detect_patterns(hist)
            
            # Market Regime Detection
            analysis['regime_detection'] = await self._detect_market_regime(hist)
            
            # Cache results
            self.cache[cache_key] = analysis
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Advanced technical analysis error for {symbol}: {e}")
            return None
    
    async def _calculate_enhanced_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate enhanced technical indicators using ta library"""
        indicators = {}
        
        try:
            if not TA_AVAILABLE:
                return self._fallback_indicators(hist)
            
            # Price-based indicators
            indicators['sma_10'] = ta.trend.sma_indicator(hist['Close'], window=10).iloc[-1]
            indicators['sma_20'] = ta.trend.sma_indicator(hist['Close'], window=20).iloc[-1]
            indicators['sma_50'] = ta.trend.sma_indicator(hist['Close'], window=50).iloc[-1]
            indicators['ema_12'] = ta.trend.ema_indicator(hist['Close'], window=12).iloc[-1]
            indicators['ema_26'] = ta.trend.ema_indicator(hist['Close'], window=26).iloc[-1]
            
            # Momentum indicators
            indicators['rsi'] = ta.momentum.rsi(hist['Close'], window=14).iloc[-1]
            indicators['stoch_k'] = ta.momentum.stoch(hist['High'], hist['Low'], hist['Close']).iloc[-1]
            indicators['stoch_d'] = ta.momentum.stoch_signal(hist['High'], hist['Low'], hist['Close']).iloc[-1]
            indicators['williams_r'] = ta.momentum.williams_r(hist['High'], hist['Low'], hist['Close']).iloc[-1]
            indicators['cci'] = ta.trend.cci(hist['High'], hist['Low'], hist['Close']).iloc[-1]
            
            # MACD
            macd_line = ta.trend.macd(hist['Close'])
            macd_signal = ta.trend.macd_signal(hist['Close'])
            macd_histogram = ta.trend.macd_diff(hist['Close'])
            
            indicators['macd'] = macd_line.iloc[-1] if not macd_line.empty else 0
            indicators['macd_signal'] = macd_signal.iloc[-1] if not macd_signal.empty else 0
            indicators['macd_histogram'] = macd_histogram.iloc[-1] if not macd_histogram.empty else 0
            
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(hist['Close'])
            bb_low = ta.volatility.bollinger_lband(hist['Close'])
            bb_mid = ta.volatility.bollinger_mavg(hist['Close'])
            
            indicators['bb_upper'] = bb_high.iloc[-1] if not bb_high.empty else hist['Close'].iloc[-1] * 1.02
            indicators['bb_lower'] = bb_low.iloc[-1] if not bb_low.empty else hist['Close'].iloc[-1] * 0.98
            indicators['bb_middle'] = bb_mid.iloc[-1] if not bb_mid.empty else hist['Close'].iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = ta.volume.volume_sma(hist['Close'], hist['Volume']).iloc[-1]
            indicators['volume_ratio'] = hist['Volume'].iloc[-1] / hist['Volume'].rolling(20).mean().iloc[-1]
            
            # Volatility indicators
            indicators['atr'] = ta.volatility.average_true_range(hist['High'], hist['Low'], hist['Close']).iloc[-1]
            indicators['keltner_high'] = ta.volatility.keltner_channel_hband(hist['High'], hist['Low'], hist['Close']).iloc[-1]
            indicators['keltner_low'] = ta.volatility.keltner_channel_lband(hist['High'], hist['Low'], hist['Close']).iloc[-1]
            
            # Advanced momentum
            indicators['tsi'] = ta.momentum.tsi(hist['Close']).iloc[-1] if len(hist) > 25 else 0
            indicators['ultimate_oscillator'] = ta.momentum.ultimate_oscillator(hist['High'], hist['Low'], hist['Close']).iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Enhanced indicators error: {e}")
            return self._fallback_indicators(hist)
    
    async def _calculate_volatility_analysis(self, hist: pd.DataFrame) -> Dict:
        """Advanced volatility analysis using ARCH models"""
        volatility = {}
        
        try:
            # Basic volatility measures
            returns = hist['Close'].pct_change().dropna()
            volatility['realized_vol_10d'] = returns.tail(10).std() * np.sqrt(252) * 100
            volatility['realized_vol_20d'] = returns.tail(20).std() * np.sqrt(252) * 100
            volatility['realized_vol_60d'] = returns.std() * np.sqrt(252) * 100
            
            # Rolling volatility
            rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
            volatility['vol_percentile'] = (rolling_vol.iloc[-1] - rolling_vol.quantile(0.1)) / (rolling_vol.quantile(0.9) - rolling_vol.quantile(0.1))
            volatility['vol_percentile'] = max(0, min(1, volatility['vol_percentile']))
            
            # Volatility trend
            recent_vol = rolling_vol.tail(5).mean()
            older_vol = rolling_vol.tail(20).head(15).mean()
            vol_change = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0
            
            if vol_change > 0.2:
                volatility['vol_trend'] = 'RISING'
            elif vol_change < -0.2:
                volatility['vol_trend'] = 'FALLING'
            else:
                volatility['vol_trend'] = 'STABLE'
            
            # Volatility regime
            current_vol = volatility['realized_vol_20d']
            if current_vol > 40:
                volatility['vol_regime'] = 'EXTREME'
            elif current_vol > 25:
                volatility['vol_regime'] = 'HIGH'
            elif current_vol > 15:
                volatility['vol_regime'] = 'NORMAL'
            else:
                volatility['vol_regime'] = 'LOW'
            
            # ARCH/GARCH modeling if available
            if ARCH_AVAILABLE and len(returns) > 100:
                try:
                    # Fit GARCH(1,1) model
                    from arch import arch_model
                    
                    # Scale returns to percentage
                    scaled_returns = returns * 100
                    
                    # Fit GARCH model
                    garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1)
                    garch_fit = garch_model.fit(disp='off')
                    
                    # Get conditional volatility forecast
                    forecast = garch_fit.forecast(horizon=5)
                    volatility['garch_forecast_1d'] = float(np.sqrt(forecast.variance.iloc[-1, 0]))
                    volatility['garch_forecast_5d'] = float(np.sqrt(forecast.variance.iloc[-1, :].mean()))
                    
                    # Volatility clustering detection
                    conditional_vol = garch_fit.conditional_volatility
                    volatility['clustering_strength'] = float(conditional_vol.tail(20).std() / conditional_vol.mean())
                    
                except Exception as e:
                    logger.warning(f"GARCH modeling error: {e}")
                    volatility['garch_forecast_1d'] = current_vol / np.sqrt(252)
                    volatility['garch_forecast_5d'] = current_vol / np.sqrt(252)
                    volatility['clustering_strength'] = 0.5
            
            return volatility
            
        except Exception as e:
            logger.warning(f"Volatility analysis error: {e}")
            return {'realized_vol_20d': 20.0, 'vol_regime': 'NORMAL', 'vol_trend': 'STABLE'}
    
    async def _calculate_momentum_analysis(self, hist: pd.DataFrame) -> Dict:
        """Enhanced momentum analysis with multiple timeframes"""
        momentum = {}
        
        try:
            prices = hist['Close']
            
            # Multi-timeframe momentum
            momentum['momentum_1d'] = (prices.iloc[-1] / prices.iloc[-2] - 1) * 100 if len(prices) > 1 else 0
            momentum['momentum_5d'] = (prices.iloc[-1] / prices.iloc[-6] - 1) * 100 if len(prices) > 5 else 0
            momentum['momentum_10d'] = (prices.iloc[-1] / prices.iloc[-11] - 1) * 100 if len(prices) > 10 else 0
            momentum['momentum_20d'] = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100 if len(prices) > 20 else 0
            
            # Momentum acceleration
            if len(prices) > 10:
                recent_momentum = momentum['momentum_5d']
                past_momentum = (prices.iloc[-6] / prices.iloc[-11] - 1) * 100
                momentum['acceleration'] = recent_momentum - past_momentum
            else:
                momentum['acceleration'] = 0
            
            # Momentum strength classification
            abs_momentum = abs(momentum['momentum_20d'])
            if abs_momentum > 10:
                momentum['strength'] = 'VERY_STRONG'
            elif abs_momentum > 5:
                momentum['strength'] = 'STRONG'
            elif abs_momentum > 2:
                momentum['strength'] = 'MODERATE'
            else:
                momentum['strength'] = 'WEAK'
            
            # Momentum consistency
            momentum_series = [momentum['momentum_1d'], momentum['momentum_5d'], momentum['momentum_10d'], momentum['momentum_20d']]
            positive_count = sum(1 for m in momentum_series if m > 0)
            negative_count = sum(1 for m in momentum_series if m < 0)
            
            if positive_count >= 3:
                momentum['consistency'] = 'BULLISH'
            elif negative_count >= 3:
                momentum['consistency'] = 'BEARISH'
            else:
                momentum['consistency'] = 'MIXED'
            
            return momentum
            
        except Exception as e:
            logger.warning(f"Momentum analysis error: {e}")
            return {'momentum_20d': 0, 'strength': 'WEAK', 'consistency': 'MIXED'}
    
    async def _generate_advanced_signals(self, hist: pd.DataFrame, analysis: Dict) -> Dict:
        """Generate advanced trading signals using all available data"""
        signals = {}
        
        try:
            indicators = analysis['technical_indicators']
            volatility = analysis['volatility_analysis']
            momentum = analysis['momentum_analysis']
            
            # Signal components
            signal_components = []
            confidence_factors = []
            
            # RSI signals
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                signal_components.append(-0.3)  # Overbought
                confidence_factors.append(0.7)
            elif rsi < 30:
                signal_components.append(0.3)   # Oversold
                confidence_factors.append(0.7)
            else:
                signal_components.append(0)
                confidence_factors.append(0.3)
            
            # MACD signals
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                signal_components.append(0.2)
                confidence_factors.append(0.6)
            else:
                signal_components.append(-0.2)
                confidence_factors.append(0.6)
            
            # Momentum signals
            momentum_20d = momentum.get('momentum_20d', 0)
            if momentum_20d > 2:
                signal_components.append(0.25)
                confidence_factors.append(0.8)
            elif momentum_20d < -2:
                signal_components.append(-0.25)
                confidence_factors.append(0.8)
            else:
                signal_components.append(0)
                confidence_factors.append(0.4)
            
            # Bollinger Band signals
            current_price = hist['Close'].iloc[-1]
            bb_upper = indicators.get('bb_upper', current_price * 1.02)
            bb_lower = indicators.get('bb_lower', current_price * 0.98)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            
            if bb_position > 0.8:
                signal_components.append(-0.15)
                confidence_factors.append(0.5)
            elif bb_position < 0.2:
                signal_components.append(0.15)
                confidence_factors.append(0.5)
            else:
                signal_components.append(0)
                confidence_factors.append(0.3)
            
            # Volume confirmation
            volume_ratio = indicators.get('volume_ratio', 1.0)
            volume_confirmation = min(volume_ratio / 2, 1.0)  # Cap at 1.0
            
            # Calculate composite signal
            weighted_signal = sum(s * c for s, c in zip(signal_components, confidence_factors))
            total_weight = sum(confidence_factors)
            composite_signal = weighted_signal / total_weight if total_weight > 0 else 0
            
            # Apply volume confirmation
            composite_signal *= volume_confirmation
            
            # Signal classification
            if composite_signal > 0.1:
                signals['overall_signal'] = 'BULLISH'
            elif composite_signal < -0.1:
                signals['overall_signal'] = 'BEARISH'
            else:
                signals['overall_signal'] = 'NEUTRAL'
            
            signals['signal_strength'] = abs(composite_signal) * 100
            signals['confidence'] = np.mean(confidence_factors)
            signals['volume_confirmation'] = volume_confirmation
            
            # Bullish/Bearish factors
            bullish_factors = []
            bearish_factors = []
            
            if rsi < 30:
                bullish_factors.append("RSI oversold")
            elif rsi > 70:
                bearish_factors.append("RSI overbought")
            
            if macd > macd_signal:
                bullish_factors.append("MACD bullish crossover")
            else:
                bearish_factors.append("MACD bearish crossover")
            
            if momentum_20d > 2:
                bullish_factors.append("Strong positive momentum")
            elif momentum_20d < -2:
                bearish_factors.append("Strong negative momentum")
            
            if bb_position < 0.2:
                bullish_factors.append("Price near lower Bollinger Band")
            elif bb_position > 0.8:
                bearish_factors.append("Price near upper Bollinger Band")
            
            if volume_ratio > 1.5:
                bullish_factors.append("High volume confirmation")
            
            signals['bullish_factors'] = bullish_factors
            signals['bearish_factors'] = bearish_factors
            
            return signals
            
        except Exception as e:
            logger.warning(f"Signal generation error: {e}")
            return {'overall_signal': 'NEUTRAL', 'signal_strength': 0, 'confidence': 0.5}
    
    async def _calculate_advanced_support_resistance(self, hist: pd.DataFrame) -> Dict:
        """Calculate support/resistance using advanced methods"""
        support_resistance = {}
        
        try:
            prices = hist['Close'].values
            highs = hist['High'].values
            lows = hist['Low'].values
            
            # Use numba-optimized functions if available
            if NUMBA_AVAILABLE:
                support_levels, resistance_levels = self._find_levels_numba(prices, highs, lows)
            else:
                support_levels, resistance_levels = self._find_levels_python(prices, highs, lows)
            
            current_price = prices[-1]
            
            # Find nearest levels
            support_levels = [s for s in support_levels if s < current_price]
            resistance_levels = [r for r in resistance_levels if r > current_price]
            
            support_resistance['support_levels'] = sorted(support_levels, reverse=True)[:3]
            support_resistance['resistance_levels'] = sorted(resistance_levels)[:3]
            
            support_resistance['nearest_support'] = support_levels[0] if support_levels else None
            support_resistance['nearest_resistance'] = resistance_levels[0] if resistance_levels else None
            
            # Calculate strength of levels
            if support_resistance['nearest_support']:
                support_distance = (current_price - support_resistance['nearest_support']) / current_price
                support_resistance['support_strength'] = max(0, 1 - support_distance * 10)
            else:
                support_resistance['support_strength'] = 0
            
            if support_resistance['nearest_resistance']:
                resistance_distance = (support_resistance['nearest_resistance'] - current_price) / current_price
                support_resistance['resistance_strength'] = max(0, 1 - resistance_distance * 10)
            else:
                support_resistance['resistance_strength'] = 0
            
            return support_resistance
            
        except Exception as e:
            logger.warning(f"Support/resistance calculation error: {e}")
            return {'support_levels': [], 'resistance_levels': [], 'nearest_support': None, 'nearest_resistance': None}
    
    def _find_levels_python(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Tuple[List[float], List[float]]:
        """Find support/resistance levels using pure Python"""
        support_levels = []
        resistance_levels = []
        
        window = 10
        threshold = 0.02  # 2% threshold
        
        for i in range(window, len(prices) - window):
            # Check for local minima (support)
            if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                support_levels.append(lows[i])
            
            # Check for local maxima (resistance)
            if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                resistance_levels.append(highs[i])
        
        # Cluster nearby levels
        support_levels = self._cluster_levels(support_levels, threshold)
        resistance_levels = self._cluster_levels(resistance_levels, threshold)
        
        return support_levels, resistance_levels
    
    def _find_levels_numba(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Tuple[List[float], List[float]]:
        """Find support/resistance levels using numba optimization"""
        # Fallback to Python implementation for now
        return self._find_levels_python(prices, highs, lows)
    
    def _cluster_levels(self, levels: List[float], threshold: float) -> List[float]:
        """Cluster nearby levels together"""
        if not levels:
            return []
        
        clustered = []
        levels_sorted = sorted(levels)
        current_cluster = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            if abs(level - current_cluster[0]) / current_cluster[0] <= threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        return clustered
    
    async def _detect_patterns(self, hist: pd.DataFrame) -> Dict:
        """Detect chart patterns"""
        patterns = {}
        
        try:
            prices = hist['Close'].values
            
            # Simple pattern detection
            if len(prices) >= 20:
                # Double top/bottom detection
                patterns['double_top'] = self._detect_double_top(prices)
                patterns['double_bottom'] = self._detect_double_bottom(prices)
                
                # Trend channel detection
                patterns['trend_channel'] = self._detect_trend_channel(prices)
                
                # Head and shoulders (simplified)
                patterns['head_shoulders'] = self._detect_head_shoulders(prices)
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Pattern detection error: {e}")
            return {}
    
    async def _detect_market_regime(self, hist: pd.DataFrame) -> Dict:
        """Detect current market regime using multiple factors"""
        regime = {}
        
        try:
            returns = hist['Close'].pct_change().dropna()
            prices = hist['Close']
            
            # Trend detection
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean()
            
            if prices.iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1]:
                regime['trend'] = 'UPTREND'
            elif prices.iloc[-1] < sma_20.iloc[-1] < sma_50.iloc[-1]:
                regime['trend'] = 'DOWNTREND'
            else:
                regime['trend'] = 'SIDEWAYS'
            
            # Volatility regime
            vol_20d = returns.rolling(20).std() * np.sqrt(252) * 100
            vol_percentile = vol_20d.iloc[-1] / vol_20d.quantile(0.9) if len(vol_20d) > 60 else 0.5
            
            if vol_percentile > 0.8:
                regime['volatility'] = 'HIGH_VOL'
            elif vol_percentile < 0.3:
                regime['volatility'] = 'LOW_VOL'
            else:
                regime['volatility'] = 'NORMAL_VOL'
            
            # Market efficiency (using autocorrelation)
            if len(returns) > 20:
                autocorr = returns.autocorr(lag=1)
                if abs(autocorr) > 0.1:
                    regime['efficiency'] = 'TRENDING'
                else:
                    regime['efficiency'] = 'MEAN_REVERTING'
            
            return regime
            
        except Exception as e:
            logger.warning(f"Regime detection error: {e}")
            return {'trend': 'SIDEWAYS', 'volatility': 'NORMAL_VOL'}
    
    def _detect_double_top(self, prices: np.ndarray) -> bool:
        """Detect double top pattern"""
        # Simplified implementation
        if len(prices) < 20:
            return False
        
        window = 5
        peaks = []
        
        for i in range(window, len(prices) - window):
            if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                peaks.append((i, prices[i]))
        
        if len(peaks) >= 2:
            # Check if last two peaks are similar height
            last_two = peaks[-2:]
            height_diff = abs(last_two[0][1] - last_two[1][1]) / max(last_two[0][1], last_two[1][1])
            return height_diff < 0.05  # Within 5%
        
        return False
    
    def _detect_double_bottom(self, prices: np.ndarray) -> bool:
        """Detect double bottom pattern"""
        # Similar to double top but for troughs
        if len(prices) < 20:
            return False
        
        window = 5
        troughs = []
        
        for i in range(window, len(prices) - window):
            if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] <= prices[i+j] for j in range(1, window+1)):
                troughs.append((i, prices[i]))
        
        if len(troughs) >= 2:
            last_two = troughs[-2:]
            height_diff = abs(last_two[0][1] - last_two[1][1]) / max(last_two[0][1], last_two[1][1])
            return height_diff < 0.05
        
        return False
    
    def _detect_trend_channel(self, prices: np.ndarray) -> Dict:
        """Detect trend channel"""
        if len(prices) < 20:
            return {'trend_channel': False}
        
        # Simple linear regression for trend
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'trend_channel': r_squared > 0.7,
            'slope': slope,
            'r_squared': r_squared,
            'strength': 'STRONG' if r_squared > 0.8 else 'MODERATE' if r_squared > 0.6 else 'WEAK'
        }
    
    def _detect_head_shoulders(self, prices: np.ndarray) -> bool:
        """Detect head and shoulders pattern (simplified)"""
        if len(prices) < 30:
            return False
        
        # Look for three peaks with middle one being highest
        window = 5
        peaks = []
        
        for i in range(window, len(prices) - window):
            if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                peaks.append((i, prices[i]))
        
        if len(peaks) >= 3:
            last_three = peaks[-3:]
            # Check if middle peak is highest
            if (last_three[1][1] > last_three[0][1] and 
                last_three[1][1] > last_three[2][1] and
                abs(last_three[0][1] - last_three[2][1]) / last_three[1][1] < 0.05):
                return True
        
        return False
    
    def _fallback_indicators(self, hist: pd.DataFrame) -> Dict:
        """Fallback indicators when ta library is not available"""
        indicators = {}
        
        # Basic indicators using pandas
        indicators['sma_20'] = hist['Close'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = hist['Close'].rolling(50).mean().iloc[-1]
        
        # Simple RSI calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Basic Bollinger Bands
        sma_20 = hist['Close'].rolling(20).mean()
        std_20 = hist['Close'].rolling(20).std()
        indicators['bb_upper'] = (sma_20 + 2 * std_20).iloc[-1]
        indicators['bb_lower'] = (sma_20 - 2 * std_20).iloc[-1]
        indicators['bb_middle'] = sma_20.iloc[-1]
        
        # Volume indicators
        indicators['volume_sma'] = hist['Volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = hist['Volume'].iloc[-1] / indicators['volume_sma']
        
        return indicators
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]

# Singleton instance
advanced_technical_analysis = AdvancedTechnicalAnalysis()