"""
Volume/Liquidity Agent - Detects institutional flows and liquidity conditions.

Prevents trading in thin markets and catches volume-confirmed breakouts.
Analyzes spread widening, volume spikes, and liquidity dry-ups.
"""
from typing import Dict, Tuple
from .base_agent import BaseAgent
import numpy as np


class VolumeLiquidityAgent(BaseAgent):
    """
    Analyzes volume patterns and liquidity conditions to improve trade quality.

    Key Functions:
    1. Volume Confirmation - Ensures breakouts have institutional backing
    2. Spread Analysis - Detects widening spreads (low liquidity danger)
    3. Volume Profile - Identifies accumulation vs distribution
    4. Liquidity Zones - Avoids trading during thin periods
    """

    def __init__(self, initial_weight: float = 1.8):
        super().__init__(name="VolumeLiquidityAgent", initial_weight=initial_weight)

        # Historical spread baselines (will learn over time)
        self.typical_spreads = {
            "EUR_USD": 0.00010,  # 1.0 pip
            "GBP_USD": 0.00015,  # 1.5 pips
            "USD_JPY": 0.010,    # 1.0 pip
        }

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Analyze volume and liquidity conditions.

        Returns:
        - BLOCK if liquidity is dangerously low (veto trade)
        - BUY/SELL if volume confirms directional move
        - NEUTRAL if volume is normal/inconclusive
        """
        pair = market_data.get("pair", "")
        candles = market_data.get("candles", [])
        price = market_data.get("price", 0)

        # Get bid/ask from market data
        bid = market_data.get("bid", price)
        ask = market_data.get("ask", price)
        spread = ask - bid if ask and bid else 0

        if not candles or len(candles) < 20:
            return ("NEUTRAL", 0.3, {"error": "insufficient_data"})

        # Extract volume data
        volumes = np.array([c.get('volume', 0) for c in candles[-50:]])
        closes = np.array([c['close'] for c in candles[-50:]])

        if len(volumes) < 20 or np.max(volumes) == 0:
            # No volume data available (common in forex)
            return self._analyze_spread_only(pair, spread, candles)

        # Volume analysis
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1] if len(volumes) > 0 else avg_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Volume trend (increasing = accumulation, decreasing = distribution)
        volume_trend = self._calculate_volume_trend(volumes)

        # Price-Volume divergence
        price_change = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) > 10 else 0

        # Spread analysis
        spread_pips = spread * 10000 if pair.endswith("JPY") else spread * 100000
        typical_spread = self.typical_spreads.get(pair, 0.00015) * (10000 if pair.endswith("JPY") else 100000)
        spread_ratio = spread_pips / typical_spread if typical_spread > 0 else 1.0

        # Decision logic
        reasoning = {
            "volume_ratio": round(volume_ratio, 2),
            "volume_trend": volume_trend,
            "spread_pips": round(spread_pips, 2),
            "spread_ratio": round(spread_ratio, 2),
            "price_change": round(price_change, 2)
        }

        # VETO CONDITION: Dangerous spread widening (low liquidity)
        if spread_ratio > 3.0:
            return ("BLOCK", 0.95, {
                **reasoning,
                "signal": "dangerous_spread_widening",
                "message": f"Spread {spread_pips:.1f} pips is {spread_ratio:.1f}x normal - LOW LIQUIDITY"
            })

        # Warning condition: High spread
        if spread_ratio > 2.0:
            return ("NEUTRAL", 0.7, {
                **reasoning,
                "signal": "elevated_spread",
                "message": f"Spread elevated at {spread_pips:.1f} pips"
            })

        # Volume spike + price breakout = Strong signal
        if volume_ratio > 2.0 and abs(price_change) > 0.5:
            if price_change > 0:
                vote = "BUY"
                signal = "volume_breakout_bullish"
            else:
                vote = "SELL"
                signal = "volume_breakout_bearish"

            confidence = min(0.80, 0.50 + (volume_ratio - 2.0) * 0.10)

            return (vote, confidence, {
                **reasoning,
                "signal": signal,
                "message": f"Volume spike {volume_ratio:.1f}x confirms {price_change:+.2f}% move"
            })

        # Volume confirmation in direction of trend
        if volume_trend == "increasing" and abs(price_change) > 0.3:
            if price_change > 0:
                vote = "BUY"
                signal = "volume_accumulation"
            else:
                vote = "SELL"
                signal = "volume_distribution"

            confidence = min(0.70, 0.50 + abs(price_change) * 0.05)

            return (vote, confidence, {
                **reasoning,
                "signal": signal,
                "message": f"Volume {volume_trend} supports {price_change:+.2f}% move"
            })

        # Low volume = weak signal, filter trade
        if volume_ratio < 0.5:
            return ("NEUTRAL", 0.4, {
                **reasoning,
                "signal": "low_volume_warning",
                "message": f"Volume {volume_ratio:.1f}x below average - weak signal"
            })

        # Normal conditions
        return ("NEUTRAL", 0.5, {
            **reasoning,
            "signal": "normal_liquidity"
        })

    def _analyze_spread_only(self, pair: str, spread: float, candles: list) -> Tuple[str, float, Dict]:
        """Fallback analysis when volume data is not available (common in forex)."""
        spread_pips = spread * 10000 if pair.endswith("JPY") else spread * 100000
        typical_spread = self.typical_spreads.get(pair, 0.00015) * (10000 if pair.endswith("JPY") else 100000)
        spread_ratio = spread_pips / typical_spread if typical_spread > 0 else 1.0

        # Analyze candle sizes (proxy for liquidity)
        recent_candles = candles[-10:]
        candle_sizes = [(c['high'] - c['low']) for c in recent_candles]
        avg_candle_size = np.mean(candle_sizes) if candle_sizes else 0
        current_candle_size = candle_sizes[-1] if candle_sizes else 0

        reasoning = {
            "spread_pips": round(spread_pips, 2),
            "spread_ratio": round(spread_ratio, 2),
            "avg_candle_size": round(avg_candle_size * 100000, 2),
            "no_volume_data": True
        }

        # Dangerous spread
        if spread_ratio > 3.0:
            return ("BLOCK", 0.90, {
                **reasoning,
                "signal": "dangerous_spread",
                "message": f"Spread {spread_pips:.1f} pips - THIN MARKET"
            })

        # Elevated spread - reduce confidence
        if spread_ratio > 2.0:
            return ("NEUTRAL", 0.6, {
                **reasoning,
                "signal": "elevated_spread",
                "message": f"Spread {spread_pips:.1f} pips elevated"
            })

        # Abnormally small candles = low activity
        if current_candle_size < avg_candle_size * 0.4:
            return ("NEUTRAL", 0.4, {
                **reasoning,
                "signal": "low_activity",
                "message": "Small candles indicate low market activity"
            })

        # Normal spread
        return ("NEUTRAL", 0.5, {
            **reasoning,
            "signal": "normal_spread"
        })

    def _calculate_volume_trend(self, volumes: np.ndarray) -> str:
        """
        Determine if volume is increasing or decreasing.

        Returns:
        - "increasing": Accumulation phase
        - "decreasing": Distribution phase
        - "flat": No clear trend
        """
        if len(volumes) < 10:
            return "flat"

        recent_avg = np.mean(volumes[-5:])
        older_avg = np.mean(volumes[-10:-5])

        if recent_avg > older_avg * 1.2:
            return "increasing"
        elif recent_avg < older_avg * 0.8:
            return "decreasing"
        else:
            return "flat"

    def learn_spread(self, pair: str, spread: float):
        """
        Update typical spread baseline for pair.

        Called periodically to adapt to changing market conditions.
        """
        if spread <= 0:
            return

        current_typical = self.typical_spreads.get(pair, spread)

        # Exponential moving average: 90% old, 10% new
        self.typical_spreads[pair] = current_typical * 0.9 + spread * 0.1
