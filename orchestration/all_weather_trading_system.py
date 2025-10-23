#!/usr/bin/env python3
"""
ALL-WEATHER TRADING SYSTEM
Makes money in ANY market condition - Bull, Bear, Neutral, Volatile

Inspired by Ray Dalio's All Weather Portfolio
Adapted for algorithmic options trading
"""

import yfinance as yf
from datetime import datetime, timedelta
from enum import Enum


class MarketRegime(Enum):
    """All possible market conditions"""
    BULL_LOW_VOL = "bull_low_vol"        # Market up, calm → Bull Put Spreads
    BULL_HIGH_VOL = "bull_high_vol"      # Market up, volatile → Long Calls
    BEAR_LOW_VOL = "bear_low_vol"        # Market down, calm → Bear Call Spreads
    BEAR_HIGH_VOL = "bear_high_vol"      # Market down, volatile → Long Puts
    NEUTRAL_LOW_VOL = "neutral_low_vol"  # Flat, calm → Iron Condors
    NEUTRAL_HIGH_VOL = "neutral_high_vol" # Flat, volatile → Straddles/Strangles
    CRISIS = "crisis"                     # Black swan → Cash + VIX hedges
    RECOVERY = "recovery"                 # Post-crash → Aggressive bullish


class AllWeatherTradingSystem:
    """
    Detects market regime and selects optimal strategy
    Makes money in ANY condition
    """

    def __init__(self):
        self.vix_panic_level = 30  # VIX > 30 = high volatility
        self.vix_extreme_level = 40  # VIX > 40 = crisis

    def detect_market_regime(self):
        """
        Detect current market regime using S&P 500 and VIX

        Returns:
            dict: {
                'regime': MarketRegime,
                'sp500_momentum': float,
                'vix_level': float,
                'recommended_strategies': list,
                'position_sizing': float (0.5 = half size, 2.0 = double size),
                'explanation': str
            }
        """

        print("\n" + "="*70)
        print("ALL-WEATHER MARKET REGIME DETECTION")
        print("="*70)

        # Get S&P 500 data
        try:
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='30d', interval='1d')

            current_price = float(spy_hist['Close'].iloc[-1])
            price_5d_ago = float(spy_hist['Close'].iloc[-6])
            price_20d_ago = float(spy_hist['Close'].iloc[-21])

            momentum_5d = (current_price / price_5d_ago) - 1
            momentum_20d = (current_price / price_20d_ago) - 1

            # Calculate daily volatility
            returns = spy_hist['Close'].pct_change().dropna()
            daily_vol = float(returns.std())
            annualized_vol = daily_vol * (252 ** 0.5)  # Annualized volatility

        except Exception as e:
            print(f"[ERROR] Cannot fetch S&P 500 data: {e}")
            return self._default_regime()

        # Get VIX (volatility index)
        try:
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='5d', interval='1d')
            vix_level = float(vix_hist['Close'].iloc[-1])
            vix_5d_ago = float(vix_hist['Close'].iloc[-6]) if len(vix_hist) >= 6 else vix_level
            vix_change = ((vix_level / vix_5d_ago) - 1) if vix_5d_ago > 0 else 0

        except Exception as e:
            print(f"[WARNING] Cannot fetch VIX data: {e}")
            vix_level = 20  # Default assumption
            vix_change = 0

        print(f"\n[MARKET INDICATORS]")
        print(f"  S&P 500 Price: ${current_price:.2f}")
        print(f"  5-day momentum: {momentum_5d:+.1%}")
        print(f"  20-day momentum: {momentum_20d:+.1%}")
        print(f"  Annualized volatility: {annualized_vol:.1%}")
        print(f"  VIX Level: {vix_level:.2f}")
        print(f"  VIX 5-day change: {vix_change:+.1%}")

        # Determine direction (bull/bear/neutral)
        if momentum_5d > 0.02 and momentum_20d > 0.05:
            direction = "BULL"
        elif momentum_5d < -0.02 and momentum_20d < -0.05:
            direction = "BEAR"
        elif momentum_5d < -0.10:  # Sharp recent drop
            direction = "CRISIS"
        elif momentum_5d > 0.05 and momentum_20d < 0:  # Recovering from drop
            direction = "RECOVERY"
        else:
            direction = "NEUTRAL"

        # Determine volatility level
        if vix_level > self.vix_extreme_level:
            volatility = "EXTREME"
        elif vix_level > self.vix_panic_level or vix_change > 0.20:
            volatility = "HIGH"
        else:
            volatility = "LOW"

        print(f"\n[REGIME COMPONENTS]")
        print(f"  Direction: {direction}")
        print(f"  Volatility: {volatility}")

        # Determine regime and strategy
        regime_data = self._map_regime_to_strategy(direction, volatility,
                                                    momentum_5d, momentum_20d,
                                                    vix_level, vix_change)

        print(f"\n[MARKET REGIME]")
        print(f"  Regime: {regime_data['regime'].value}")
        print(f"  Position Sizing: {regime_data['position_sizing']:.1f}x")

        print(f"\n[RECOMMENDED STRATEGIES]")
        for i, strategy in enumerate(regime_data['recommended_strategies'], 1):
            print(f"  {i}. {strategy}")

        print(f"\n[EXPLANATION]")
        for line in regime_data['explanation'].split('\n'):
            print(f"  {line}")

        print("="*70 + "\n")

        return regime_data

    def _map_regime_to_strategy(self, direction, volatility, momentum_5d,
                                 momentum_20d, vix_level, vix_change):
        """Map market conditions to specific regime and strategies"""

        # CRISIS MODE (Trump tariffs, Black Swan events, etc.)
        if direction == "CRISIS" or volatility == "EXTREME":
            return {
                'regime': MarketRegime.CRISIS,
                'sp500_momentum': momentum_5d,
                'vix_level': vix_level,
                'recommended_strategies': [
                    'CASH (90% position)',
                    'Long VIX Calls (hedge)',
                    'Long SPY Puts (protection)',
                    'WAIT for stabilization'
                ],
                'position_sizing': 0.1,  # Reduce to 10% normal size
                'explanation': (
                    f"CRISIS MODE: Market chaos detected\n"
                    f"  - S&P 500: {momentum_5d:+.1%} in 5 days (sharp drop)\n"
                    f"  - VIX: {vix_level:.1f} (extreme fear)\n"
                    f"  - Action: PRESERVE CAPITAL\n"
                    f"  - DO NOT trade normal strategies\n"
                    f"  - Wait for VIX < 30 before resuming\n"
                    f"  Example: Trump tariff news = CRISIS MODE"
                )
            }

        # RECOVERY MODE (After crisis, market bouncing)
        if direction == "RECOVERY":
            return {
                'regime': MarketRegime.RECOVERY,
                'sp500_momentum': momentum_5d,
                'vix_level': vix_level,
                'recommended_strategies': [
                    'Aggressive Bull Put Spreads',
                    'ATM Call Spreads',
                    'Long Calls on dip leaders',
                    'Short VIX (volatility crush)'
                ],
                'position_sizing': 1.5,  # Increase to 1.5x size (opportunity)
                'explanation': (
                    f"RECOVERY MODE: Market recovering from sell-off\n"
                    f"  - S&P up {momentum_5d:+.1%} recently after being down\n"
                    f"  - VIX still elevated: {vix_level:.1f}\n"
                    f"  - Opportunity: Volatility will crush, premiums rich\n"
                    f"  - Action: AGGRESSIVE premium collection\n"
                    f"  - Best time for Bull Put Spreads (high premiums)"
                )
            }

        # BULL + LOW VOLATILITY (Best for Bull Put Spreads)
        if direction == "BULL" and volatility == "LOW":
            return {
                'regime': MarketRegime.BULL_LOW_VOL,
                'sp500_momentum': momentum_5d,
                'vix_level': vix_level,
                'recommended_strategies': [
                    'Bull Put Spreads (primary)',
                    'Cash-Secured Puts',
                    'Covered Calls',
                    'Wheel Strategy'
                ],
                'position_sizing': 1.0,  # Normal size
                'explanation': (
                    f"BULL + LOW VOL: Ideal for premium collection\n"
                    f"  - Market trending up steadily\n"
                    f"  - Low volatility = stable\n"
                    f"  - High probability trades\n"
                    f"  - Target: 70%+ win rate"
                )
            }

        # BULL + HIGH VOLATILITY (Directional bullish)
        if direction == "BULL" and volatility == "HIGH":
            return {
                'regime': MarketRegime.BULL_HIGH_VOL,
                'sp500_momentum': momentum_5d,
                'vix_level': vix_level,
                'recommended_strategies': [
                    'Long Calls (ride momentum)',
                    'Bull Call Spreads',
                    'Butterfly Spreads (if choppy)',
                    'Avoid Bull Put Spreads (too risky)'
                ],
                'position_sizing': 0.75,  # Reduce size (more risk)
                'explanation': (
                    f"BULL + HIGH VOL: Volatile uptrend\n"
                    f"  - Market up but choppy\n"
                    f"  - VIX elevated: {vix_level:.1f}\n"
                    f"  - Use directional strategies\n"
                    f"  - Bull Put Spreads too risky here"
                )
            }

        # BEAR + LOW VOLATILITY (Steady decline)
        if direction == "BEAR" and volatility == "LOW":
            return {
                'regime': MarketRegime.BEAR_LOW_VOL,
                'sp500_momentum': momentum_5d,
                'vix_level': vix_level,
                'recommended_strategies': [
                    'Bear Call Spreads (primary)',
                    'Cash-Secured Calls',
                    'Long Puts',
                    'Short Stock + Long Calls'
                ],
                'position_sizing': 1.0,
                'explanation': (
                    f"BEAR + LOW VOL: Steady decline\n"
                    f"  - Market trending down\n"
                    f"  - Low volatility = predictable\n"
                    f"  - Use bearish premium strategies\n"
                    f"  - Mirror of bull strategies"
                )
            }

        # BEAR + HIGH VOLATILITY (Panic selling)
        if direction == "BEAR" and volatility == "HIGH":
            return {
                'regime': MarketRegime.BEAR_HIGH_VOL,
                'sp500_momentum': momentum_5d,
                'vix_level': vix_level,
                'recommended_strategies': [
                    'Long Puts (profit from fear)',
                    'Bear Put Spreads',
                    'Long VIX Calls',
                    'MOSTLY CASH (dangerous)'
                ],
                'position_sizing': 0.5,  # Half size (very risky)
                'explanation': (
                    f"BEAR + HIGH VOL: Panic mode\n"
                    f"  - Market falling fast\n"
                    f"  - VIX: {vix_level:.1f} (fear)\n"
                    f"  - Very dangerous to trade\n"
                    f"  - Consider sitting out"
                )
            }

        # NEUTRAL + LOW VOLATILITY (Range-bound, calm)
        if direction == "NEUTRAL" and volatility == "LOW":
            return {
                'regime': MarketRegime.NEUTRAL_LOW_VOL,
                'sp500_momentum': momentum_5d,
                'vix_level': vix_level,
                'recommended_strategies': [
                    'Iron Condors (primary)',
                    'Butterfly Spreads',
                    'Calendar Spreads',
                    'Short Strangles'
                ],
                'position_sizing': 1.0,
                'explanation': (
                    f"NEUTRAL + LOW VOL: Range-bound market\n"
                    f"  - Market moving sideways\n"
                    f"  - Low volatility = staying in range\n"
                    f"  - Perfect for Iron Condors\n"
                    f"  - Collect premium both sides"
                )
            }

        # NEUTRAL + HIGH VOLATILITY (Choppy, unpredictable)
        if direction == "NEUTRAL" and volatility == "HIGH":
            return {
                'regime': MarketRegime.NEUTRAL_HIGH_VOL,
                'sp500_momentum': momentum_5d,
                'vix_level': vix_level,
                'recommended_strategies': [
                    'Long Straddles (profit from big move)',
                    'Long Strangles',
                    'Calendar Spreads',
                    'WAIT for direction (better)'
                ],
                'position_sizing': 0.75,
                'explanation': (
                    f"NEUTRAL + HIGH VOL: Choppy, uncertain\n"
                    f"  - Market going nowhere fast\n"
                    f"  - High volatility = big swings\n"
                    f"  - Hard to predict\n"
                    f"  - Consider waiting for clarity"
                )
            }

        # Default fallback
        return self._default_regime()

    def _default_regime(self):
        """Default regime when data unavailable"""
        return {
            'regime': MarketRegime.NEUTRAL_LOW_VOL,
            'sp500_momentum': 0,
            'vix_level': 20,
            'recommended_strategies': ['WAIT - insufficient data'],
            'position_sizing': 0,
            'explanation': 'Market data unavailable - wait for clear signals'
        }

    def should_trade_today(self, regime_data):
        """
        Determine if we should trade based on regime

        Returns:
            tuple: (should_trade: bool, reason: str)
        """

        regime = regime_data['regime']
        vix = regime_data['vix_level']
        position_sizing = regime_data['position_sizing']

        # NEVER trade in crisis
        if regime == MarketRegime.CRISIS:
            return (False, f"CRISIS MODE: VIX {vix:.1f} - Preserve capital, wait for stabilization")

        # Be cautious in high volatility bearish
        if regime == MarketRegime.BEAR_HIGH_VOL and vix > 35:
            return (False, f"Extreme fear: VIX {vix:.1f} - Too dangerous to trade")

        # Reduce trading in uncertain conditions
        if position_sizing < 0.5:
            return (True, f"REDUCED TRADING: Only {position_sizing:.0%} normal position size")

        # Good to trade in most other conditions
        return (True, "Market conditions favorable for trading")


def test_all_weather_system():
    """Test the all-weather detection system"""

    system = AllWeatherTradingSystem()

    # Detect current regime
    regime_data = system.detect_market_regime()

    # Check if we should trade
    should_trade, reason = system.should_trade_today(regime_data)

    print("\n" + "="*70)
    print("TRADING DECISION")
    print("="*70)
    print(f"Should trade today: {'YES' if should_trade else 'NO'}")
    print(f"Reason: {reason}")

    if should_trade:
        print(f"\nRecommended strategies:")
        for strategy in regime_data['recommended_strategies']:
            print(f"  - {strategy}")
        print(f"\nPosition sizing: {regime_data['position_sizing']:.1f}x normal")

    print("="*70)


if __name__ == "__main__":
    test_all_weather_system()
