#!/usr/bin/env python3
"""
Enhanced Filters for Sharpe Ratio Optimization
All the technical indicators and filters to boost Sharpe from 1.38 to 2.0+
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import warnings
warnings.filterwarnings('ignore')

from config.logging_config import get_logger

logger = get_logger(__name__)

class SharpeEnhancedFilters:
    """All the filters to maximize Sharpe ratio"""

    def __init__(self):
        self.vix_cache = {}
        self.price_cache = {}
        self.cache_duration = 300  # 5 minutes

    async def get_vix_data(self) -> float:
        """Get current VIX level for volatility regime detection"""
        try:
            current_time = datetime.now().timestamp()

            # Check cache
            if ('VIX' in self.vix_cache and
                current_time - self.vix_cache['VIX']['timestamp'] < self.cache_duration):
                return self.vix_cache['VIX']['value']

            # Fetch VIX data
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")

            if not hist.empty:
                current_vix = hist['Close'].iloc[-1]
                self.vix_cache['VIX'] = {
                    'value': current_vix,
                    'timestamp': current_time
                }
                return current_vix
            else:
                return 20.0  # Default VIX level

        except Exception as e:
            logger.warning(f"Failed to get VIX data: {e}")
            return 20.0

    async def get_stock_data(self, symbol: str, period: str = "60d") -> pd.DataFrame:
        """Get stock price data with caching"""
        try:
            current_time = datetime.now().timestamp()
            cache_key = f"{symbol}_{period}"

            # Check cache
            if (cache_key in self.price_cache and
                current_time - self.price_cache[cache_key]['timestamp'] < self.cache_duration):
                return self.price_cache[cache_key]['data']

            # Fetch stock data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if not hist.empty:
                self.price_cache[cache_key] = {
                    'data': hist,
                    'timestamp': current_time
                }
                return hist
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {e}")
            return pd.DataFrame()

    async def calculate_rsi_filter(self, symbol: str, period: int = 14) -> Dict:
        """RSI filter to avoid extreme conditions"""
        try:
            data = await self.get_stock_data(symbol, "30d")
            if data.empty or len(data) < period + 5:
                return {'signal': 'NEUTRAL', 'rsi': 50, 'valid': False}

            # Calculate RSI
            rsi = ta.momentum.RSIIndicator(close=data['Close'], window=period).rsi()
            current_rsi = rsi.iloc[-1]

            # Determine signal
            if current_rsi < 30:
                signal = 'OVERSOLD'  # Avoid puts, favor calls
                trade_bias = 'BULLISH'
            elif current_rsi > 70:
                signal = 'OVERBOUGHT'  # Avoid calls, favor puts
                trade_bias = 'BEARISH'
            elif 40 <= current_rsi <= 60:
                signal = 'NEUTRAL'
                trade_bias = 'NEUTRAL'
            else:
                signal = 'TRENDING'
                trade_bias = 'BULLISH' if current_rsi > 50 else 'BEARISH'

            return {
                'signal': signal,
                'trade_bias': trade_bias,
                'rsi': current_rsi,
                'valid': True,
                'strength': abs(current_rsi - 50) / 50  # 0-1 strength
            }

        except Exception as e:
            logger.error(f"RSI calculation failed for {symbol}: {e}")
            return {'signal': 'NEUTRAL', 'rsi': 50, 'valid': False}

    async def calculate_ema_filter(self, symbol: str, fast: int = 12, slow: int = 26) -> Dict:
        """EMA crossover filter for trend direction"""
        try:
            data = await self.get_stock_data(symbol, "60d")
            if data.empty or len(data) < slow + 10:
                return {'signal': 'NEUTRAL', 'trend': 'SIDEWAYS', 'valid': False}

            # Calculate EMAs
            ema_fast = ta.trend.EMAIndicator(close=data['Close'], window=fast).ema_indicator()
            ema_slow = ta.trend.EMAIndicator(close=data['Close'], window=slow).ema_indicator()

            # Current and previous values
            fast_current = ema_fast.iloc[-1]
            slow_current = ema_slow.iloc[-1]
            fast_prev = ema_fast.iloc[-2]
            slow_prev = ema_slow.iloc[-2]

            # Determine trend and signals
            if fast_current > slow_current:
                if fast_prev <= slow_prev:
                    signal = 'BULLISH_CROSS'  # Fresh bullish crossover
                else:
                    signal = 'BULLISH_CONT'   # Continued bullish trend
                trend = 'UPTREND'
                trade_bias = 'CALLS_FAVORED'
            else:
                if fast_prev >= slow_prev:
                    signal = 'BEARISH_CROSS'  # Fresh bearish crossover
                else:
                    signal = 'BEARISH_CONT'   # Continued bearish trend
                trend = 'DOWNTREND'
                trade_bias = 'PUTS_FAVORED'

            # Calculate trend strength
            ema_spread = abs(fast_current - slow_current) / slow_current
            trend_strength = min(ema_spread * 100, 1.0)  # 0-1 scale

            return {
                'signal': signal,
                'trend': trend,
                'trade_bias': trade_bias,
                'ema_fast': fast_current,
                'ema_slow': slow_current,
                'trend_strength': trend_strength,
                'valid': True
            }

        except Exception as e:
            logger.error(f"EMA calculation failed for {symbol}: {e}")
            return {'signal': 'NEUTRAL', 'trend': 'SIDEWAYS', 'valid': False}

    async def calculate_momentum_filter(self, symbol: str, lookback: int = 5) -> Dict:
        """Momentum filter based on recent price action"""
        try:
            data = await self.get_stock_data(symbol, "30d")
            if data.empty or len(data) < lookback + 5:
                return {'signal': 'NEUTRAL', 'momentum': 0, 'valid': False}

            # Calculate momentum metrics
            closes = data['Close']
            current_price = closes.iloc[-1]

            # Short-term momentum (5 days)
            short_momentum = (current_price - closes.iloc[-lookback]) / closes.iloc[-lookback]

            # Medium-term momentum (10 days)
            if len(closes) >= 10:
                medium_momentum = (current_price - closes.iloc[-10]) / closes.iloc[-10]
            else:
                medium_momentum = short_momentum

            # Rate of change
            roc = ta.momentum.ROCIndicator(close=closes, window=lookback).roc().iloc[-1] / 100

            # Determine momentum signal
            if short_momentum > 0.02 and medium_momentum > 0:
                signal = 'STRONG_BULLISH'
                trade_bias = 'CALLS_PREFERRED'
            elif short_momentum < -0.02 and medium_momentum < 0:
                signal = 'STRONG_BEARISH'
                trade_bias = 'PUTS_PREFERRED'
            elif short_momentum > 0:
                signal = 'WEAK_BULLISH'
                trade_bias = 'CALLS_SLIGHT'
            elif short_momentum < 0:
                signal = 'WEAK_BEARISH'
                trade_bias = 'PUTS_SLIGHT'
            else:
                signal = 'NEUTRAL'
                trade_bias = 'NEUTRAL'

            return {
                'signal': signal,
                'trade_bias': trade_bias,
                'short_momentum': short_momentum,
                'medium_momentum': medium_momentum,
                'roc': roc,
                'momentum_strength': abs(short_momentum),
                'valid': True
            }

        except Exception as e:
            logger.error(f"Momentum calculation failed for {symbol}: {e}")
            return {'signal': 'NEUTRAL', 'momentum': 0, 'valid': False}

    async def calculate_volatility_regime(self, symbol: str = None) -> Dict:
        """Determine current volatility regime for position sizing"""
        try:
            vix_level = await self.get_vix_data()

            # VIX-based regime classification
            if vix_level < 16:
                regime = 'LOW_VOLATILITY'
                position_multiplier = 1.3  # Increase size in low vol
                risk_level = 'LOW'
            elif vix_level < 20:
                regime = 'NORMAL_VOLATILITY'
                position_multiplier = 1.0  # Normal size
                risk_level = 'NORMAL'
            elif vix_level < 30:
                regime = 'ELEVATED_VOLATILITY'
                position_multiplier = 0.8  # Reduce size
                risk_level = 'HIGH'
            else:
                regime = 'HIGH_VOLATILITY'
                position_multiplier = 0.6  # Significantly reduce size
                risk_level = 'EXTREME'

            return {
                'regime': regime,
                'vix_level': vix_level,
                'position_multiplier': position_multiplier,
                'risk_level': risk_level,
                'valid': True
            }

        except Exception as e:
            logger.error(f"Volatility regime calculation failed: {e}")
            return {
                'regime': 'NORMAL_VOLATILITY',
                'vix_level': 20.0,
                'position_multiplier': 1.0,
                'risk_level': 'NORMAL',
                'valid': False
            }

    async def calculate_iv_rank(self, symbol: str, current_iv: float) -> Dict:
        """Calculate IV rank for better entry timing"""
        try:
            # Get 1 year of data for IV rank calculation
            data = await self.get_stock_data(symbol, "1y")
            if data.empty:
                return {'iv_rank': 50, 'signal': 'NEUTRAL', 'valid': False}

            # Calculate historical volatility as proxy for IV history
            returns = data['Close'].pct_change().dropna()
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)

            if len(rolling_vol.dropna()) < 50:
                return {'iv_rank': 50, 'signal': 'NEUTRAL', 'valid': False}

            # Calculate IV rank (percentile)
            vol_history = rolling_vol.dropna()
            current_hv = vol_history.iloc[-1]

            # Use current IV if provided, otherwise use historical vol
            vol_to_rank = current_iv if current_iv > 0 else current_hv

            iv_rank = (vol_history < vol_to_rank).sum() / len(vol_history) * 100

            # Determine signal
            if iv_rank > 70:
                signal = 'HIGH_IV'  # Good for selling premium
                trade_preference = 'SELL_PREMIUM'
            elif iv_rank > 50:
                signal = 'ELEVATED_IV'  # Decent for selling
                trade_preference = 'NEUTRAL_SELL'
            elif iv_rank < 30:
                signal = 'LOW_IV'  # Good for buying premium
                trade_preference = 'BUY_PREMIUM'
            else:
                signal = 'NORMAL_IV'
                trade_preference = 'NEUTRAL'

            return {
                'iv_rank': iv_rank,
                'signal': signal,
                'trade_preference': trade_preference,
                'current_iv': vol_to_rank,
                'hv_percentile': iv_rank,
                'valid': True
            }

        except Exception as e:
            logger.error(f"IV rank calculation failed for {symbol}: {e}")
            return {'iv_rank': 50, 'signal': 'NEUTRAL', 'valid': False}

    def is_earnings_week(self, symbol: str) -> bool:
        """Check if stock has earnings in next 7 days"""
        try:
            # Simple heuristic - avoid major earnings weeks
            # This is a simplified implementation
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar

            if calendar is not None and not calendar.empty:
                # Check if earnings date is within next 7 days
                today = datetime.now().date()
                week_ahead = today + timedelta(days=7)

                for date in calendar.index:
                    earnings_date = pd.to_datetime(date).date()
                    if today <= earnings_date <= week_ahead:
                        return True

            return False

        except Exception:
            # If we can't determine, assume it's safe
            return False

    def is_market_hours(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()

        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    async def get_comprehensive_filters(self, symbol: str, current_iv: float = 0.25) -> Dict:
        """Get all filters for a symbol"""
        try:
            # Run all filters concurrently
            tasks = [
                self.calculate_rsi_filter(symbol),
                self.calculate_ema_filter(symbol),
                self.calculate_momentum_filter(symbol),
                self.calculate_volatility_regime(),
                self.calculate_iv_rank(symbol, current_iv)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            rsi_result = results[0] if not isinstance(results[0], Exception) else {'valid': False}
            ema_result = results[1] if not isinstance(results[1], Exception) else {'valid': False}
            momentum_result = results[2] if not isinstance(results[2], Exception) else {'valid': False}
            volatility_result = results[3] if not isinstance(results[3], Exception) else {'valid': False}
            iv_result = results[4] if not isinstance(results[4], Exception) else {'valid': False}

            # Additional filters
            earnings_risk = self.is_earnings_week(symbol)
            market_open = self.is_market_hours()

            # Compile comprehensive assessment
            assessment = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'rsi': rsi_result,
                'ema': ema_result,
                'momentum': momentum_result,
                'volatility': volatility_result,
                'iv_rank': iv_result,
                'earnings_risk': earnings_risk,
                'market_hours': market_open,
                'overall_bias': self._determine_overall_bias(rsi_result, ema_result, momentum_result),
                'risk_score': self._calculate_risk_score(volatility_result, iv_result, earnings_risk),
                'position_sizing_multiplier': self._get_position_multiplier(volatility_result, iv_result),
                'trade_recommendation': self._get_trade_recommendation(rsi_result, ema_result, momentum_result, iv_result)
            }

            return assessment

        except Exception as e:
            logger.error(f"Comprehensive filter calculation failed for {symbol}: {e}")
            return self._get_default_assessment(symbol)

    def _determine_overall_bias(self, rsi: Dict, ema: Dict, momentum: Dict) -> str:
        """Determine overall directional bias"""
        bullish_signals = 0
        bearish_signals = 0

        # RSI bias
        if rsi.get('valid') and rsi.get('trade_bias') == 'BULLISH':
            bullish_signals += 1
        elif rsi.get('valid') and rsi.get('trade_bias') == 'BEARISH':
            bearish_signals += 1

        # EMA bias
        if ema.get('valid') and 'CALLS' in ema.get('trade_bias', ''):
            bullish_signals += 1
        elif ema.get('valid') and 'PUTS' in ema.get('trade_bias', ''):
            bearish_signals += 1

        # Momentum bias
        if momentum.get('valid') and 'CALLS' in momentum.get('trade_bias', ''):
            bullish_signals += 1
        elif momentum.get('valid') and 'PUTS' in momentum.get('trade_bias', ''):
            bearish_signals += 1

        if bullish_signals > bearish_signals:
            return 'BULLISH'
        elif bearish_signals > bullish_signals:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _calculate_risk_score(self, volatility: Dict, iv_rank: Dict, earnings_risk: bool) -> float:
        """Calculate overall risk score (0-1, higher = riskier)"""
        risk_score = 0.0

        # Volatility risk
        if volatility.get('valid'):
            vix_level = volatility.get('vix_level', 20)
            risk_score += min(vix_level / 40, 0.4)  # Max 0.4 from VIX

        # IV rank risk
        if iv_rank.get('valid'):
            iv_rank_val = iv_rank.get('iv_rank', 50)
            if iv_rank_val > 70:
                risk_score += 0.2  # High IV adds risk
            elif iv_rank_val < 30:
                risk_score += 0.1  # Low IV moderate risk

        # Earnings risk
        if earnings_risk:
            risk_score += 0.3

        return min(risk_score, 1.0)

    def _get_position_multiplier(self, volatility: Dict, iv_rank: Dict) -> float:
        """Get position sizing multiplier"""
        multiplier = 1.0

        # Volatility adjustment
        if volatility.get('valid'):
            vol_multiplier = volatility.get('position_multiplier', 1.0)
            multiplier *= vol_multiplier

        # IV rank adjustment
        if iv_rank.get('valid'):
            iv_rank_val = iv_rank.get('iv_rank', 50)
            if iv_rank_val > 80:
                multiplier *= 0.8  # Reduce size in very high IV
            elif iv_rank_val < 20:
                multiplier *= 1.1  # Increase size in very low IV

        return max(0.5, min(1.5, multiplier))  # Clamp between 0.5x and 1.5x

    def _get_trade_recommendation(self, rsi: Dict, ema: Dict, momentum: Dict, iv_rank: Dict) -> str:
        """Get overall trade recommendation"""
        # Count positive signals
        positive_signals = 0
        total_signals = 0

        for signal_dict in [rsi, ema, momentum]:
            if signal_dict.get('valid'):
                total_signals += 1
                bias = signal_dict.get('trade_bias', 'NEUTRAL')
                if 'BULLISH' in bias or 'CALLS' in bias:
                    positive_signals += 0.5 if 'SLIGHT' in bias else 1
                elif 'BEARISH' in bias or 'PUTS' in bias:
                    positive_signals -= 0.5 if 'SLIGHT' in bias else -1

        if total_signals == 0:
            return 'NO_SIGNAL'

        signal_strength = positive_signals / total_signals

        if signal_strength > 0.6:
            return 'STRONG_BUY'
        elif signal_strength > 0.3:
            return 'BUY'
        elif signal_strength < -0.6:
            return 'STRONG_SELL'
        elif signal_strength < -0.3:
            return 'SELL'
        else:
            return 'NEUTRAL'

    def _get_default_assessment(self, symbol: str) -> Dict:
        """Return default assessment when calculations fail"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'rsi': {'valid': False},
            'ema': {'valid': False},
            'momentum': {'valid': False},
            'volatility': {'regime': 'NORMAL_VOLATILITY', 'position_multiplier': 1.0, 'valid': False},
            'iv_rank': {'valid': False},
            'earnings_risk': False,
            'market_hours': self.is_market_hours(),
            'overall_bias': 'NEUTRAL',
            'risk_score': 0.5,
            'position_sizing_multiplier': 1.0,
            'trade_recommendation': 'NEUTRAL'
        }

# Global instance
sharpe_enhanced_filters = SharpeEnhancedFilters()