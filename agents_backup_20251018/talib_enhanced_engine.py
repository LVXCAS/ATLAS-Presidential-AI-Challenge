#!/usr/bin/env python3
"""
Enhanced TA-Lib Technical Analysis Engine
Comprehensive technical indicators with momentum, volatility, and pattern recognition
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Try to import TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
    print("+ TA-Lib available for professional technical analysis")
except ImportError:
    TALIB_AVAILABLE = False
    print("- TA-Lib not available - using custom technical indicator implementations")

class TALibEnhancedEngine:
    """Enhanced technical analysis using TA-Lib with custom fallbacks"""
    
    def __init__(self):
        self.talib_available = TALIB_AVAILABLE
        self.indicator_cache = {}
        
        # Technical analysis configuration
        self.config = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'stoch_k': 14,
            'stoch_d': 3,
            'atr_period': 14,
            'adx_period': 14,
            'cci_period': 20,
            'williams_r_period': 14,
            'momentum_period': 10,
            'roc_period': 10
        }
        
        print(f"+ Enhanced TA-Lib Engine initialized (TA-Lib available: {TALIB_AVAILABLE})")
    
    async def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive set of technical indicators"""
        try:
            if df.empty or len(df) < 50:
                print("- Insufficient data for technical analysis")
                return {}
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    print(f"- Missing required column: {col}")
                    return {}
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'data_length': len(df),
                'momentum_indicators': await self._calculate_momentum_indicators(df),
                'volatility_indicators': await self._calculate_volatility_indicators(df),
                'trend_indicators': await self._calculate_trend_indicators(df),
                'volume_indicators': await self._calculate_volume_indicators(df),
                'oscillators': await self._calculate_oscillators(df),
                'pattern_recognition': await self._calculate_patterns(df),
                'composite_signals': await self._calculate_composite_signals(df)
            }
            
            return results
            
        except Exception as e:
            print(f"- Error calculating indicators: {e}")
            return {}
    
    async def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum-based indicators"""
        try:
            indicators = {}

            # Ensure data types are float64 (double) for TA-Lib
            close = df['close'].astype(np.float64).values
            high = df['high'].astype(np.float64).values
            low = df['low'].astype(np.float64).values
            
            if self.talib_available:
                # RSI
                indicators['rsi'] = talib.RSI(close, timeperiod=self.config['rsi_period'])
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(
                    close, 
                    fastperiod=self.config['macd_fast'],
                    slowperiod=self.config['macd_slow'],
                    signalperiod=self.config['macd_signal']
                )
                indicators['macd'] = macd
                indicators['macd_signal'] = macd_signal
                indicators['macd_histogram'] = macd_hist
                
                # Momentum
                indicators['momentum'] = talib.MOM(close, timeperiod=self.config['momentum_period'])
                
                # Rate of Change
                indicators['roc'] = talib.ROC(close, timeperiod=self.config['roc_period'])
                
                # Stochastic
                stoch_k, stoch_d = talib.STOCH(
                    high, low, close,
                    fastk_period=self.config['stoch_k'],
                    slowd_period=self.config['stoch_d']
                )
                indicators['stoch_k'] = stoch_k
                indicators['stoch_d'] = stoch_d
                
                # Williams %R
                indicators['williams_r'] = talib.WILLR(
                    high, low, close, timeperiod=self.config['williams_r_period']
                )
                
                # CCI
                indicators['cci'] = talib.CCI(
                    high, low, close, timeperiod=self.config['cci_period']
                )
                
            else:
                # Custom implementations
                indicators['rsi'] = self._custom_rsi(close, self.config['rsi_period'])
                
                # Custom MACD
                ema_fast = self._custom_ema(close, self.config['macd_fast'])
                ema_slow = self._custom_ema(close, self.config['macd_slow'])
                macd = ema_fast - ema_slow
                macd_signal = self._custom_ema(macd, self.config['macd_signal'])
                indicators['macd'] = macd
                indicators['macd_signal'] = macd_signal
                indicators['macd_histogram'] = macd - macd_signal
                
                # Custom momentum
                indicators['momentum'] = self._custom_momentum(close, self.config['momentum_period'])
                
                # Custom ROC
                indicators['roc'] = self._custom_roc(close, self.config['roc_period'])
                
                # Custom Stochastic
                stoch_k, stoch_d = self._custom_stochastic(
                    high, low, close, self.config['stoch_k'], self.config['stoch_d']
                )
                indicators['stoch_k'] = stoch_k
                indicators['stoch_d'] = stoch_d
                
                # Custom Williams %R
                indicators['williams_r'] = self._custom_williams_r(
                    high, low, close, self.config['williams_r_period']
                )
                
                # Custom CCI
                indicators['cci'] = self._custom_cci(
                    high, low, close, self.config['cci_period']
                )
            
            # Calculate current values (last non-NaN value)
            current_values = {}
            for key, values in indicators.items():
                if isinstance(values, np.ndarray) and len(values) > 0:
                    # Find last non-NaN value
                    valid_values = values[~np.isnan(values)]
                    current_values[f'{key}_current'] = float(valid_values[-1]) if len(valid_values) > 0 else 0
            
            indicators.update(current_values)
            return indicators
            
        except Exception as e:
            print(f"- Error calculating momentum indicators: {e}")
            return {}
    
    async def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility-based indicators"""
        try:
            indicators = {}

            # Ensure data types are float64 (double) for TA-Lib
            close = df['close'].astype(np.float64).values
            high = df['high'].astype(np.float64).values
            low = df['low'].astype(np.float64).values
            
            if self.talib_available:
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close, 
                    timeperiod=self.config['bb_period'],
                    nbdevup=self.config['bb_std'],
                    nbdevdn=self.config['bb_std']
                )
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
                
                # Calculate Bollinger Band width and position
                bb_width = (bb_upper - bb_lower) / bb_middle
                bb_position = (close - bb_lower) / (bb_upper - bb_lower)
                indicators['bb_width'] = bb_width
                indicators['bb_position'] = bb_position
                
                # Average True Range
                indicators['atr'] = talib.ATR(high, low, close, timeperiod=self.config['atr_period'])
                
                # True Range
                indicators['true_range'] = talib.TRANGE(high, low, close)
                
                # Normalized ATR (ATR as percentage of price)
                indicators['atr_percent'] = indicators['atr'] / close * 100
                
            else:
                # Custom Bollinger Bands
                bb_middle = self._custom_sma(close, self.config['bb_period'])
                bb_std = self._custom_rolling_std(close, self.config['bb_period'])
                bb_upper = bb_middle + (bb_std * self.config['bb_std'])
                bb_lower = bb_middle - (bb_std * self.config['bb_std'])
                
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
                
                bb_width = (bb_upper - bb_lower) / bb_middle
                bb_position = (close - bb_lower) / (bb_upper - bb_lower)
                indicators['bb_width'] = bb_width
                indicators['bb_position'] = bb_position
                
                # Custom ATR
                indicators['atr'] = self._custom_atr(high, low, close, self.config['atr_period'])
                indicators['atr_percent'] = indicators['atr'] / close * 100
            
            # Calculate current values
            current_values = {}
            for key, values in indicators.items():
                if isinstance(values, np.ndarray) and len(values) > 0:
                    valid_values = values[~np.isnan(values)]
                    current_values[f'{key}_current'] = float(valid_values[-1]) if len(valid_values) > 0 else 0
            
            indicators.update(current_values)
            return indicators
            
        except Exception as e:
            print(f"- Error calculating volatility indicators: {e}")
            return {}
    
    async def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate trend-following indicators"""
        try:
            indicators = {}

            # Ensure data types are float64 (double) for TA-Lib
            close = df['close'].astype(np.float64).values
            high = df['high'].astype(np.float64).values
            low = df['low'].astype(np.float64).values
            
            if self.talib_available:
                # Moving Averages
                indicators['sma_10'] = talib.SMA(close, timeperiod=10)
                indicators['sma_20'] = talib.SMA(close, timeperiod=20)
                indicators['sma_50'] = talib.SMA(close, timeperiod=50)
                indicators['sma_200'] = talib.SMA(close, timeperiod=200)
                
                indicators['ema_10'] = talib.EMA(close, timeperiod=10)
                indicators['ema_20'] = talib.EMA(close, timeperiod=20)
                indicators['ema_50'] = talib.EMA(close, timeperiod=50)
                indicators['ema_200'] = talib.EMA(close, timeperiod=200)
                
                # Weighted and Hull Moving Averages
                indicators['wma_20'] = talib.WMA(close, timeperiod=20)
                
                # ADX (trend strength)
                indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.config['adx_period'])
                indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=self.config['adx_period'])
                indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=self.config['adx_period'])
                
                # Parabolic SAR
                indicators['sar'] = talib.SAR(high, low)
                
                # Aroon
                aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
                indicators['aroon_up'] = aroon_up
                indicators['aroon_down'] = aroon_down
                indicators['aroon_oscillator'] = aroon_up - aroon_down
                
            else:
                # Custom implementations
                indicators['sma_10'] = self._custom_sma(close, 10)
                indicators['sma_20'] = self._custom_sma(close, 20)
                indicators['sma_50'] = self._custom_sma(close, 50)
                indicators['sma_200'] = self._custom_sma(close, 200)
                
                indicators['ema_10'] = self._custom_ema(close, 10)
                indicators['ema_20'] = self._custom_ema(close, 20)
                indicators['ema_50'] = self._custom_ema(close, 50)
                indicators['ema_200'] = self._custom_ema(close, 200)
                
                # Custom ADX (simplified)
                indicators['adx'] = self._custom_adx(high, low, close, self.config['adx_period'])
            
            # Calculate trend signals
            indicators['trend_signal'] = self._calculate_trend_signal(indicators, close)
            
            # Calculate current values
            current_values = {}
            for key, values in indicators.items():
                if isinstance(values, np.ndarray) and len(values) > 0:
                    valid_values = values[~np.isnan(values)]
                    current_values[f'{key}_current'] = float(valid_values[-1]) if len(valid_values) > 0 else 0
                elif isinstance(values, (int, float)):
                    current_values[f'{key}_current'] = float(values)
            
            indicators.update(current_values)
            return indicators
            
        except Exception as e:
            print(f"- Error calculating trend indicators: {e}")
            return {}
    
    async def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volume-based indicators"""
        try:
            indicators = {}

            # Ensure data types are float64 (double) for TA-Lib
            close = df['close'].astype(np.float64).values
            high = df['high'].astype(np.float64).values
            low = df['low'].astype(np.float64).values
            volume = df['volume'].astype(np.float64).values
            
            if self.talib_available:
                # On Balance Volume
                indicators['obv'] = talib.OBV(close, volume)
                
                # Accumulation/Distribution Line
                indicators['ad'] = talib.AD(high, low, close, volume)
                
                # Chaikin Money Flow
                indicators['adosc'] = talib.ADOSC(high, low, close, volume)
                
            else:
                # Custom OBV
                indicators['obv'] = self._custom_obv(close, volume)
                
                # Custom A/D Line
                indicators['ad'] = self._custom_ad_line(high, low, close, volume)
            
            # Volume moving averages
            indicators['volume_sma_20'] = self._custom_sma(volume, 20)
            indicators['volume_ratio'] = volume / indicators['volume_sma_20']
            
            # Calculate current values
            current_values = {}
            for key, values in indicators.items():
                if isinstance(values, np.ndarray) and len(values) > 0:
                    valid_values = values[~np.isnan(values)]
                    current_values[f'{key}_current'] = float(valid_values[-1]) if len(valid_values) > 0 else 0
            
            indicators.update(current_values)
            return indicators
            
        except Exception as e:
            print(f"- Error calculating volume indicators: {e}")
            return {}
    
    async def _calculate_oscillators(self, df: pd.DataFrame) -> Dict:
        """Calculate oscillator indicators"""
        try:
            indicators = {}

            # Ensure data types are float64 (double) for TA-Lib
            close = df['close'].astype(np.float64).values
            high = df['high'].astype(np.float64).values
            low = df['low'].astype(np.float64).values
            
            # Additional oscillators
            if self.talib_available:
                # Ultimate Oscillator
                indicators['ultosc'] = talib.ULTOSC(high, low, close)
                
                # Commodity Channel Index (already calculated in momentum)
                # Money Flow Index
                if 'volume' in df.columns:
                    volume = df['volume'].astype(np.float64).values
                    indicators['mfi'] = talib.MFI(high, low, close, volume)
                
            # Custom oscillator combinations
            indicators['combined_oscillator'] = self._calculate_combined_oscillator(df)
            
            # Overbought/Oversold levels
            indicators['overbought_oversold_signal'] = self._calculate_overbought_oversold_signals(indicators)
            
            return indicators
            
        except Exception as e:
            print(f"- Error calculating oscillators: {e}")
            return {}
    
    async def _calculate_patterns(self, df: pd.DataFrame) -> Dict:
        """Calculate pattern recognition indicators"""
        try:
            patterns = {}
            
            if self.talib_available and len(df) >= 20:
                # Ensure data types are float64 (double) for TA-Lib
                open_prices = df['open'].astype(np.float64).values
                high_prices = df['high'].astype(np.float64).values
                low_prices = df['low'].astype(np.float64).values
                close_prices = df['close'].astype(np.float64).values
                
                # Major candlestick patterns
                patterns['doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
                patterns['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
                patterns['hanging_man'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
                patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
                patterns['engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
                patterns['harami'] = talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices)
                patterns['morning_star'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
                patterns['evening_star'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
                
                # Count recent patterns (last 10 bars)
                pattern_summary = {}
                for pattern_name, pattern_values in patterns.items():
                    recent_patterns = pattern_values[-10:]
                    bullish_count = np.sum(recent_patterns > 0)
                    bearish_count = np.sum(recent_patterns < 0)
                    pattern_summary[f'{pattern_name}_bullish_recent'] = int(bullish_count)
                    pattern_summary[f'{pattern_name}_bearish_recent'] = int(bearish_count)
                
                patterns.update(pattern_summary)
            
            else:
                # Custom pattern recognition (simplified)
                patterns['custom_reversal_signal'] = self._custom_reversal_patterns(df)
            
            return patterns
            
        except Exception as e:
            print(f"- Error calculating patterns: {e}")
            return {}
    
    async def _calculate_composite_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate composite trading signals"""
        try:
            signals = {}
            
            close = df['close'].values
            
            # Get indicators
            momentum = await self._calculate_momentum_indicators(df)
            trend = await self._calculate_trend_indicators(df)
            volatility = await self._calculate_volatility_indicators(df)
            
            # Bull/Bear signal calculation
            bull_signals = 0
            bear_signals = 0
            
            # RSI signals
            rsi_current = momentum.get('rsi_current', 50)
            if rsi_current < 30:
                bull_signals += 1
            elif rsi_current > 70:
                bear_signals += 1
            
            # MACD signals
            macd_current = momentum.get('macd_current', 0)
            macd_signal_current = momentum.get('macd_signal_current', 0)
            if macd_current > macd_signal_current:
                bull_signals += 1
            else:
                bear_signals += 1
            
            # Moving average signals
            sma_20_current = trend.get('sma_20_current', 0)
            sma_50_current = trend.get('sma_50_current', 0)
            current_price = close[-1] if len(close) > 0 else 0
            
            if current_price > sma_20_current > sma_50_current:
                bull_signals += 2
            elif current_price < sma_20_current < sma_50_current:
                bear_signals += 2
            
            # Bollinger Bands signals
            bb_position_current = volatility.get('bb_position_current', 0.5)
            if bb_position_current < 0.2:
                bull_signals += 1
            elif bb_position_current > 0.8:
                bear_signals += 1
            
            # Calculate overall signal strength
            total_signals = bull_signals + bear_signals
            signal_strength = abs(bull_signals - bear_signals) / max(1, total_signals)
            
            if bull_signals > bear_signals:
                overall_signal = 'BULLISH'
                signal_score = signal_strength
            elif bear_signals > bull_signals:
                overall_signal = 'BEARISH'
                signal_score = -signal_strength
            else:
                overall_signal = 'NEUTRAL'
                signal_score = 0
            
            signals = {
                'overall_signal': overall_signal,
                'signal_score': signal_score,
                'bull_signals_count': bull_signals,
                'bear_signals_count': bear_signals,
                'signal_strength': signal_strength,
                'recommendation': self._generate_recommendation(overall_signal, signal_strength)
            }
            
            return signals
            
        except Exception as e:
            print(f"- Error calculating composite signals: {e}")
            return {}
    
    # Custom indicator implementations (fallbacks when TA-Lib not available)
    
    def _custom_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Custom RSI implementation"""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with NaN to match original length
        return np.concatenate([np.full(len(close) - len(rsi), np.nan), rsi])
    
    def _custom_sma(self, values: np.ndarray, period: int) -> np.ndarray:
        """Custom Simple Moving Average"""
        return np.convolve(values, np.ones(period)/period, mode='same')
    
    def _custom_ema(self, values: np.ndarray, period: int) -> np.ndarray:
        """Custom Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(values)
        ema[0] = values[0]
        
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _custom_momentum(self, close: np.ndarray, period: int) -> np.ndarray:
        """Custom Momentum implementation"""
        momentum = np.full_like(close, np.nan)
        momentum[period:] = close[period:] - close[:-period]
        return momentum
    
    def _custom_roc(self, close: np.ndarray, period: int) -> np.ndarray:
        """Custom Rate of Change implementation"""
        roc = np.full_like(close, np.nan)
        roc[period:] = ((close[period:] - close[:-period]) / close[:-period]) * 100
        return roc
    
    def _custom_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                          k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Custom Stochastic implementation"""
        stoch_k = np.full_like(close, np.nan)
        
        for i in range(k_period - 1, len(close)):
            highest_high = np.max(high[i - k_period + 1:i + 1])
            lowest_low = np.min(low[i - k_period + 1:i + 1])
            if highest_high != lowest_low:
                stoch_k[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
        
        stoch_d = self._custom_sma(stoch_k, d_period)
        return stoch_k, stoch_d
    
    def _custom_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Custom Williams %R implementation"""
        williams_r = np.full_like(close, np.nan)
        
        for i in range(period - 1, len(close)):
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])
            if highest_high != lowest_low:
                williams_r[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
    def _custom_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Custom Commodity Channel Index implementation"""
        typical_price = (high + low + close) / 3
        sma_tp = self._custom_sma(typical_price, period)
        
        cci = np.full_like(close, np.nan)
        for i in range(period - 1, len(close)):
            mean_deviation = np.mean(np.abs(typical_price[i - period + 1:i + 1] - sma_tp[i]))
            if mean_deviation != 0:
                cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mean_deviation)
        
        return cci
    
    def _custom_rolling_std(self, values: np.ndarray, period: int) -> np.ndarray:
        """Custom rolling standard deviation"""
        std = np.full_like(values, np.nan)
        for i in range(period - 1, len(values)):
            std[i] = np.std(values[i - period + 1:i + 1])
        return std
    
    def _custom_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Custom Average True Range implementation"""
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]  # First value
        return self._custom_sma(tr, period)
    
    def _custom_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Custom ADX implementation (simplified)"""
        # Simplified ADX calculation
        tr = self._custom_atr(high, low, close, 1)
        plus_dm = np.maximum(high - np.roll(high, 1), 0)
        minus_dm = np.maximum(np.roll(low, 1) - low, 0)
        
        plus_di = self._custom_sma(plus_dm, period) / self._custom_sma(tr, period) * 100
        minus_di = self._custom_sma(minus_dm, period) / self._custom_sma(tr, period) * 100
        
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        adx = self._custom_sma(dx, period)
        
        return adx
    
    def _custom_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Custom On Balance Volume implementation"""
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def _custom_ad_line(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Custom Accumulation/Distribution Line implementation"""
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        ad = np.cumsum(clv * volume)
        return ad
    
    def _calculate_trend_signal(self, indicators: Dict, close: np.ndarray) -> int:
        """Calculate overall trend signal"""
        try:
            signal = 0
            current_price = close[-1] if len(close) > 0 else 0
            
            # Moving average signals
            sma_20 = indicators.get('sma_20_current', 0)
            sma_50 = indicators.get('sma_50_current', 0)
            
            if current_price > sma_20 > sma_50:
                signal += 1
            elif current_price < sma_20 < sma_50:
                signal -= 1
            
            return signal
            
        except Exception:
            return 0
    
    def _calculate_combined_oscillator(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate combined oscillator signal"""
        try:
            close = df['close'].values
            # Simple combined oscillator (RSI + Stochastic average)
            rsi = self._custom_rsi(close, 14)
            stoch_k, _ = self._custom_stochastic(
                df['high'].values, df['low'].values, close, 14, 3
            )
            
            # Normalize and combine
            combined = (rsi + stoch_k) / 2
            return combined
            
        except Exception:
            return np.full_like(df['close'].values, 50)
    
    def _calculate_overbought_oversold_signals(self, indicators: Dict) -> Dict:
        """Calculate overbought/oversold signals"""
        signals = {}
        
        # RSI levels
        rsi_current = indicators.get('rsi_current', 50)
        if rsi_current > 70:
            signals['rsi_signal'] = 'OVERBOUGHT'
        elif rsi_current < 30:
            signals['rsi_signal'] = 'OVERSOLD'
        else:
            signals['rsi_signal'] = 'NEUTRAL'
        
        # Stochastic levels
        stoch_k_current = indicators.get('stoch_k_current', 50)
        if stoch_k_current > 80:
            signals['stoch_signal'] = 'OVERBOUGHT'
        elif stoch_k_current < 20:
            signals['stoch_signal'] = 'OVERSOLD'
        else:
            signals['stoch_signal'] = 'NEUTRAL'
        
        return signals
    
    def _custom_reversal_patterns(self, df: pd.DataFrame) -> int:
        """Custom reversal pattern detection"""
        try:
            if len(df) < 5:
                return 0
            
            close = df['close'].values[-5:]
            high = df['high'].values[-5:]
            low = df['low'].values[-5:]
            
            # Simple pattern: Check for potential hammer or shooting star
            last_candle = len(close) - 1
            body_size = abs(close[last_candle] - df['open'].iloc[-1])
            total_range = high[last_candle] - low[last_candle]
            
            if total_range > 0 and body_size / total_range < 0.3:
                # Small body relative to range
                if close[last_candle] > close[last_candle - 1]:
                    return 1  # Potential bullish reversal
                else:
                    return -1  # Potential bearish reversal
            
            return 0
            
        except Exception:
            return 0
    
    def _generate_recommendation(self, signal: str, strength: float) -> str:
        """Generate trading recommendation"""
        if signal == 'BULLISH':
            if strength > 0.7:
                return 'STRONG BUY'
            elif strength > 0.4:
                return 'BUY'
            else:
                return 'WEAK BUY'
        elif signal == 'BEARISH':
            if strength > 0.7:
                return 'STRONG SELL'
            elif strength > 0.4:
                return 'SELL'
            else:
                return 'WEAK SELL'
        else:
            return 'HOLD'

# Create global instance
talib_engine = TALibEnhancedEngine()