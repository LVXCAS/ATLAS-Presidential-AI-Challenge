"""
SIMPLE ROBUST STRATEGY - Evidence-Based, 3 Parameters

Based on research (not curve-fitting):
- Regime filter (ADX)
- Trend direction (EMA 200)
- Entry timing (RSI pullback)

NO OPTIMIZATION. Using standard values from academic research.
"""

import numpy as np
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARN] TA-Lib not available")

class SimpleRobustStrategy:
    """
    Regime-Adaptive Trend Following Strategy

    Parameters (FIXED, not optimized):
    - ADX threshold: 20 (research-backed)
    - EMA period: 200 (most watched long-term MA)
    - RSI period: 14, level: 50 (standard)
    """

    def __init__(self):
        # FIXED parameters (NOT to be optimized)
        self.adx_threshold = 20  # Minimum trend strength
        self.ema_period = 200    # Long-term trend
        self.rsi_period = 14     # Standard momentum
        self.rsi_pullback_level = 50  # Midpoint (not extreme)

        # Risk management
        self.profit_target_pct = 0.02  # 2%
        self.stop_loss_pct = 0.01      # 1%
        self.daily_dd_limit = 3000     # $3k max loss per day

    def identify_regime(self, adx, bb_width, avg_bb_width):
        """
        Step 1: Determine market regime

        Returns: "TRENDING", "RANGING", "VOLATILE", or "UNCLEAR"
        """
        if adx > 25:
            return "TRENDING"  # Strong directional move

        elif adx < self.adx_threshold:
            return "RANGING"  # Choppy, no clear direction

        elif bb_width > avg_bb_width * 1.5:
            return "VOLATILE"  # High volatility, unpredictable

        else:
            return "UNCLEAR"  # Transitioning, don't trade

    def calculate_indicators(self, candles):
        """
        Calculate technical indicators

        Returns: dict with ADX, RSI, EMA200, BB_WIDTH
        """
        if len(candles) < 250:
            return None

        closes = np.array([float(c['mid']['c']) for c in candles])
        highs = np.array([float(c['mid']['h']) for c in candles])
        lows = np.array([float(c['mid']['l']) for c in candles])

        if not TALIB_AVAILABLE:
            # Simplified fallback
            return {
                'adx': 15,  # Assume ranging
                'rsi': 50,  # Neutral
                'ema_200': np.mean(closes[-200:]),
                'bb_width': 0.01,
                'avg_bb_width': 0.01
            }

        # Calculate using TA-Lib
        adx = talib.ADX(highs, lows, closes, timeperiod=14)
        rsi = talib.RSI(closes, timeperiod=self.rsi_period)
        ema_200 = talib.EMA(closes, timeperiod=self.ema_period)

        # Bollinger Band width (for regime detection)
        upper, middle, lower = talib.BBANDS(closes, timeperiod=20)
        bb_width = (upper - lower) / middle
        avg_bb_width = np.mean(bb_width[-50:])  # 50-period average

        return {
            'adx': adx[-1],
            'rsi': rsi[-1],
            'ema_200': ema_200[-1],
            'bb_width': bb_width[-1],
            'avg_bb_width': avg_bb_width,
            'price': closes[-1]
        }

    def generate_signal(self, indicators):
        """
        Step 2: Generate trading signal based on regime

        Returns: "BUY", "SELL", or None
        """
        if indicators is None:
            return None

        adx = indicators['adx']
        rsi = indicators['rsi']
        ema_200 = indicators['ema_200']
        price = indicators['price']
        bb_width = indicators['bb_width']
        avg_bb_width = indicators['avg_bb_width']

        # Step 1: Identify regime
        regime = self.identify_regime(adx, bb_width, avg_bb_width)

        print(f"[REGIME] {regime} | ADX: {adx:.1f} | RSI: {rsi:.1f} | Price vs EMA200: {((price/ema_200 - 1)*100):+.2f}%")

        # Step 2: Don't trade in unfavorable regimes
        if regime in ["RANGING", "VOLATILE", "UNCLEAR"]:
            print(f"[NO TRADE] Regime not suitable for trend following")
            return None

        # Step 3: Confirm we're in trending regime
        if regime != "TRENDING":
            return None

        # Step 4: Determine trend direction
        if price > ema_200:
            trend = "UP"
        elif price < ema_200:
            trend = "DOWN"
        else:
            return None  # Price exactly at EMA (rare)

        # Step 5: Wait for pullback (better entry)
        if trend == "UP":
            # In uptrend, buy on RSI pullback to midpoint
            if rsi < self.rsi_pullback_level:
                print(f"[BUY SIGNAL] Uptrend + RSI pullback ({rsi:.1f})")
                return "BUY"

        elif trend == "DOWN":
            # In downtrend, sell on RSI bounce to midpoint
            if rsi > self.rsi_pullback_level:
                print(f"[SELL SIGNAL] Downtrend + RSI bounce ({rsi:.1f})")
                return "SELL"

        # No signal
        print(f"[WAIT] Trend detected but no pullback yet")
        return None

    def calculate_position_size(self, balance, price, dd_cushion):
        """
        Step 3: Calculate position size with DD constraints

        Args:
            balance: Current account balance
            price: Current price
            dd_cushion: Remaining DD cushion in dollars

        Returns: Position size in units
        """
        # Risk 1.5% per trade (conservative)
        risk_amount = balance * 0.015

        # But ALSO constrain by DD cushion (the $600 lesson)
        max_safe_loss = dd_cushion * 0.60  # Only use 60% of cushion per trade

        # Take the SMALLER of the two
        risk_amount = min(risk_amount, max_safe_loss)

        # Calculate position size
        stop_distance = price * self.stop_loss_pct
        leverage = 5

        units = int((risk_amount / stop_distance) * leverage * 0.70)  # 70% of calculated (conservative)

        # Minimum 10k units (0.1 lot)
        if units < 10000:
            print(f"[SKIP] Position too small ({units} units), need >10k")
            return 0

        # Maximum 300k units (3 lots) - very conservative
        if units > 300000:
            units = 300000

        lots = units / 100000
        max_loss = (units * stop_distance) / leverage

        print(f"[POSITION SIZE] {units:,} units ({lots:.1f} lots)")
        print(f"  Risk Amount: ${risk_amount:,.2f}")
        print(f"  Max Loss at SL: ${max_loss:,.2f}")
        print(f"  DD Cushion: ${dd_cushion:,.2f}")

        return units


# ==============================================================================
# TESTING THE STRATEGY
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLE ROBUST STRATEGY - Testing Logic")
    print("=" * 70)

    strategy = SimpleRobustStrategy()

    # Test scenarios
    scenarios = [
        {
            "name": "Strong Uptrend + Pullback",
            "adx": 30,
            "rsi": 45,
            "ema_200": 1.1000,
            "price": 1.1200,
            "bb_width": 0.015,
            "avg_bb_width": 0.012,
            "expected": "BUY"
        },
        {
            "name": "Strong Downtrend + Bounce",
            "adx": 28,
            "rsi": 55,
            "ema_200": 1.1200,
            "price": 1.1000,
            "bb_width": 0.014,
            "avg_bb_width": 0.012,
            "expected": "SELL"
        },
        {
            "name": "Ranging Market (Low ADX)",
            "adx": 15,
            "rsi": 30,
            "ema_200": 1.1100,
            "price": 1.1050,
            "bb_width": 0.008,
            "avg_bb_width": 0.010,
            "expected": None
        },
        {
            "name": "Trend but No Pullback Yet",
            "adx": 32,
            "rsi": 65,  # RSI high, not pulled back
            "ema_200": 1.1000,
            "price": 1.1200,
            "bb_width": 0.016,
            "avg_bb_width": 0.012,
            "expected": None
        },
        {
            "name": "Volatile Market",
            "adx": 22,
            "rsi": 50,
            "ema_200": 1.1100,
            "price": 1.1150,
            "bb_width": 0.025,  # Very wide bands
            "avg_bb_width": 0.012,
            "expected": None
        }
    ]

    print("\nTesting Strategy Logic:\n")

    for scenario in scenarios:
        print("-" * 70)
        print(f"Scenario: {scenario['name']}")
        print(f"  ADX: {scenario['adx']} | RSI: {scenario['rsi']} | Price: {scenario['price']} | EMA200: {scenario['ema_200']}")

        indicators = {
            'adx': scenario['adx'],
            'rsi': scenario['rsi'],
            'ema_200': scenario['ema_200'],
            'price': scenario['price'],
            'bb_width': scenario['bb_width'],
            'avg_bb_width': scenario['avg_bb_width']
        }

        signal = strategy.generate_signal(indicators)
        expected = scenario['expected']

        if signal == expected:
            print(f"  Result: {signal} ✓ CORRECT")
        else:
            print(f"  Result: {signal} ✗ EXPECTED: {expected}")

        print()

    print("=" * 70)
    print("NEXT STEP: Deploy this on Match Trader Demo")
    print("=" * 70)
    print("""
This strategy has THEORETICAL edge:
1. Only trades trending markets (ADX filter)
2. Follows trend direction (EMA 200)
3. Waits for pullbacks (RSI)

It's NOT optimized on historical data.
It uses STANDARD parameter values.
It's SIMPLE (only 3 decision points).

This makes it HARDER to overfit.

Test it for 60 days on Match Trader demo:
- Track: Win rate, ROI, daily DD violations
- If works → deploy on funded
- If fails → at least we know quickly

No $600 at risk. Just forward testing.
""")
    print("=" * 70)
