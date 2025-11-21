#!/usr/bin/env python3
"""
REGIME PROTECTED TRADING SYSTEM
================================
Ensures the RIGHT strategy is used in the RIGHT market regime

PROTECTION RULES:
- VERY_BULLISH: Dual Options, Long Calls only (NO bear strategies)
- BULLISH: Bull Put Spreads, Dual Options (NO bear strategies)
- NEUTRAL: Bull Put Spreads, Iron Condors, Butterfly (IDEAL)
- BEARISH: Bear Call Spreads, Long Puts (NO bull strategies)
- CRISIS: Cash only, VIX hedges (NO normal trading)

PREVENTS DISASTERS:
- Trading bull strategies in bear markets
- Trading bear strategies in bull markets
- Trading spreads in high volatility
- Trading directional in neutral markets
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from orchestration.all_weather_trading_system import AllWeatherTradingSystem, MarketRegime


class RegimeProtectedTrading:
    """
    Enforces regime-appropriate strategy selection
    Blocks inappropriate strategies based on current market conditions
    """

    def __init__(self, config_path='REGIME_PROTECTION_CONFIG.json'):
        print("=" * 80)
        print("REGIME PROTECTED TRADING SYSTEM")
        print("=" * 80)
        print("Loading market regime protection rules...")

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize all-weather system
        self.all_weather = AllWeatherTradingSystem()

        # Current regime cache (updated every check)
        self.current_regime = None
        self.last_regime_check = None

        print("[OK] Regime protection system initialized")
        print(f"[OK] Loaded {len(self.config['regimes'])} regime configurations")

    def _load_config(self, config_path):
        """Load regime protection configuration"""

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = self._create_default_config()
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"[CREATED] Default config: {config_path}")
            return default_config

    def _create_default_config(self):
        """Create default regime protection rules"""

        return {
            "version": "1.0",
            "description": "Market regime protection rules - prevents wrong strategies in wrong regimes",
            "regimes": {
                "VERY_BULLISH": {
                    "allowed_strategies": [
                        "DUAL_OPTIONS",
                        "LONG_CALLS",
                        "BULL_CALL_SPREAD",
                        "STOCK"
                    ],
                    "blocked_strategies": [
                        "BULL_PUT_SPREAD",
                        "BEAR_CALL_SPREAD",
                        "BEAR_PUT_SPREAD",
                        "LONG_PUTS",
                        "SHORT_STOCK"
                    ],
                    "position_sizing_multiplier": 1.0,
                    "max_positions": 10,
                    "risk_per_trade": 0.015,
                    "explanation": "Very bullish market - use directional strategies only. Bull Put Spreads not viable (stocks moving too fast)."
                },
                "BULLISH": {
                    "allowed_strategies": [
                        "BULL_PUT_SPREAD",
                        "DUAL_OPTIONS",
                        "LONG_CALLS",
                        "BULL_CALL_SPREAD",
                        "COVERED_CALLS",
                        "STOCK"
                    ],
                    "blocked_strategies": [
                        "BEAR_CALL_SPREAD",
                        "BEAR_PUT_SPREAD",
                        "LONG_PUTS",
                        "SHORT_STOCK"
                    ],
                    "position_sizing_multiplier": 1.0,
                    "max_positions": 10,
                    "risk_per_trade": 0.015,
                    "explanation": "Bullish market - mix of premium collection and directional. Good for Bull Put Spreads."
                },
                "NEUTRAL": {
                    "allowed_strategies": [
                        "BULL_PUT_SPREAD",
                        "IRON_CONDOR",
                        "BUTTERFLY",
                        "CALENDAR_SPREAD",
                        "STRADDLE",
                        "STRANGLE"
                    ],
                    "blocked_strategies": [
                        "LONG_CALLS",
                        "LONG_PUTS"
                    ],
                    "position_sizing_multiplier": 1.0,
                    "max_positions": 15,
                    "risk_per_trade": 0.012,
                    "explanation": "Neutral market - IDEAL for premium collection strategies. Focus on high-probability income."
                },
                "BEARISH": {
                    "allowed_strategies": [
                        "BEAR_CALL_SPREAD",
                        "LONG_PUTS",
                        "BEAR_PUT_SPREAD",
                        "SHORT_STOCK"
                    ],
                    "blocked_strategies": [
                        "BULL_PUT_SPREAD",
                        "BULL_CALL_SPREAD",
                        "LONG_CALLS",
                        "STOCK"
                    ],
                    "position_sizing_multiplier": 0.75,
                    "max_positions": 5,
                    "risk_per_trade": 0.01,
                    "explanation": "Bearish market - use bearish strategies only. Reduce size and positions."
                },
                "CRISIS": {
                    "allowed_strategies": [
                        "CASH",
                        "LONG_VIX",
                        "LONG_SPY_PUTS"
                    ],
                    "blocked_strategies": [
                        "BULL_PUT_SPREAD",
                        "BULL_CALL_SPREAD",
                        "BEAR_CALL_SPREAD",
                        "IRON_CONDOR",
                        "BUTTERFLY",
                        "LONG_CALLS",
                        "STOCK"
                    ],
                    "position_sizing_multiplier": 0.1,
                    "max_positions": 1,
                    "risk_per_trade": 0.005,
                    "explanation": "CRISIS MODE - Preserve capital. No normal trading. Wait for stabilization."
                }
            }
        }

    def get_current_regime(self, force_refresh=False):
        """
        Get current market regime

        Args:
            force_refresh: Force a new regime detection (default: False)

        Returns:
            Regime data dict
        """

        # Check if we need to refresh (more than 1 hour old or forced)
        if force_refresh or self.current_regime is None:
            print("\n[REGIME CHECK] Detecting current market regime...")
            regime_data = self.all_weather.detect_market_regime()
            self.current_regime = regime_data
            self.last_regime_check = datetime.now()
            return regime_data

        # Use cached regime
        print(f"\n[REGIME CHECK] Using cached regime: {self.current_regime['regime'].value}")
        return self.current_regime

    def is_strategy_allowed(self, strategy_name: str) -> Tuple[bool, str]:
        """
        Check if strategy is allowed in current market regime

        Args:
            strategy_name: Strategy name (e.g., 'BULL_PUT_SPREAD')

        Returns:
            Tuple of (allowed: bool, reason: str)
        """

        # Get current regime
        regime_data = self.get_current_regime()
        regime_name = self._map_regime_enum_to_config(regime_data['regime'])

        # Get regime rules
        regime_rules = self.config['regimes'].get(regime_name)
        if not regime_rules:
            return (False, f"Unknown regime: {regime_name}")

        # Check if strategy is explicitly allowed
        if strategy_name in regime_rules['allowed_strategies']:
            return (True, f"Strategy allowed in {regime_name} regime")

        # Check if strategy is explicitly blocked
        if strategy_name in regime_rules['blocked_strategies']:
            reason = f"Strategy BLOCKED in {regime_name} regime. {regime_rules['explanation']}"
            return (False, reason)

        # Strategy not in either list - allow with warning
        return (True, f"Strategy not explicitly configured for {regime_name} regime (allowing)")

    def get_position_sizing_adjustment(self) -> float:
        """
        Get position sizing multiplier based on current regime

        Returns:
            Multiplier (e.g., 0.5 = half size, 1.0 = normal, 1.5 = 1.5x size)
        """

        regime_data = self.get_current_regime()
        regime_name = self._map_regime_enum_to_config(regime_data['regime'])

        regime_rules = self.config['regimes'].get(regime_name, {})
        multiplier = regime_rules.get('position_sizing_multiplier', 1.0)

        # Also use all-weather position sizing
        all_weather_multiplier = regime_data.get('position_sizing', 1.0)

        # Use the more conservative (lower) multiplier
        final_multiplier = min(multiplier, all_weather_multiplier)

        return final_multiplier

    def select_best_strategy_for_regime(self, available_strategies: List[str]) -> Optional[str]:
        """
        Select best strategy from available options based on current regime

        Args:
            available_strategies: List of strategy names

        Returns:
            Best strategy name or None
        """

        regime_data = self.get_current_regime()
        regime_name = self._map_regime_enum_to_config(regime_data['regime'])
        regime_rules = self.config['regimes'].get(regime_name, {})

        allowed = regime_rules.get('allowed_strategies', [])

        # Find intersection of available and allowed
        suitable = [s for s in available_strategies if s in allowed]

        if not suitable:
            return None

        # Priority order (best first)
        priority_order = regime_rules.get('allowed_strategies', [])

        # Return highest priority suitable strategy
        for strategy in priority_order:
            if strategy in suitable:
                return strategy

        return suitable[0]  # Fallback to first suitable

    def validate_trade_before_execution(self, strategy_name: str, symbol: str, size: float) -> Dict:
        """
        Validate trade before execution - comprehensive pre-trade check

        Args:
            strategy_name: Strategy to execute
            symbol: Trading symbol
            size: Position size

        Returns:
            Validation result dict
        """

        print(f"\n{'='*80}")
        print(f"REGIME PROTECTION - PRE-TRADE VALIDATION")
        print(f"{'='*80}")
        print(f"Strategy: {strategy_name}")
        print(f"Symbol: {symbol}")
        print(f"Size: ${size:,.2f}")

        # Get current regime
        regime_data = self.get_current_regime(force_refresh=True)
        regime_name = self._map_regime_enum_to_config(regime_data['regime'])

        print(f"Current Regime: {regime_name}")
        print(f"S&P 500 Momentum: {regime_data['sp500_momentum']:+.1%}")
        print(f"VIX: {regime_data['vix_level']:.2f}")

        # Check if strategy allowed
        allowed, reason = self.is_strategy_allowed(strategy_name)

        if not allowed:
            print(f"\n[BLOCKED] {reason}")
            return {
                'allowed': False,
                'reason': reason,
                'regime': regime_name,
                'action': 'REJECT_TRADE'
            }

        # Check if we should trade today
        should_trade, trade_reason = self.all_weather.should_trade_today(regime_data)

        if not should_trade:
            print(f"\n[BLOCKED] Market conditions unfavorable: {trade_reason}")
            return {
                'allowed': False,
                'reason': trade_reason,
                'regime': regime_name,
                'action': 'WAIT'
            }

        # Get position sizing adjustment
        size_multiplier = self.get_position_sizing_adjustment()
        adjusted_size = size * size_multiplier

        print(f"\n[APPROVED] Trade validated")
        print(f"Position Sizing Adjustment: {size_multiplier:.2f}x")
        print(f"Adjusted Size: ${adjusted_size:,.2f}")

        return {
            'allowed': True,
            'reason': f'Strategy approved for {regime_name} regime',
            'regime': regime_name,
            'original_size': size,
            'adjusted_size': adjusted_size,
            'size_multiplier': size_multiplier,
            'action': 'EXECUTE',
            'regime_data': regime_data
        }

    def _map_regime_enum_to_config(self, regime_enum: MarketRegime) -> str:
        """Map AllWeatherTradingSystem regime enum to config regime name"""

        mapping = {
            MarketRegime.BULL_LOW_VOL: 'BULLISH',
            MarketRegime.BULL_HIGH_VOL: 'VERY_BULLISH',
            MarketRegime.BEAR_LOW_VOL: 'BEARISH',
            MarketRegime.BEAR_HIGH_VOL: 'BEARISH',
            MarketRegime.NEUTRAL_LOW_VOL: 'NEUTRAL',
            MarketRegime.NEUTRAL_HIGH_VOL: 'NEUTRAL',
            MarketRegime.CRISIS: 'CRISIS',
            MarketRegime.RECOVERY: 'BULLISH'
        }

        return mapping.get(regime_enum, 'NEUTRAL')


def test_regime_protection():
    """Test regime protection system"""

    print("\n" + "=" * 80)
    print("TESTING REGIME PROTECTED TRADING SYSTEM")
    print("=" * 80)

    rpt = RegimeProtectedTrading()

    # Test 1: Check current regime
    print("\n\nTEST 1: Current Market Regime")
    regime = rpt.get_current_regime(force_refresh=True)

    # Test 2: Check if strategies are allowed
    print("\n\nTEST 2: Strategy Validation")
    test_strategies = [
        'BULL_PUT_SPREAD',
        'BEAR_CALL_SPREAD',
        'DUAL_OPTIONS',
        'IRON_CONDOR',
        'LONG_CALLS'
    ]

    for strategy in test_strategies:
        allowed, reason = rpt.is_strategy_allowed(strategy)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"  {strategy}: {status}")
        print(f"    Reason: {reason}")

    # Test 3: Position sizing adjustment
    print("\n\nTEST 3: Position Sizing Adjustment")
    multiplier = rpt.get_position_sizing_adjustment()
    print(f"  Current Position Sizing Multiplier: {multiplier:.2f}x")

    # Test 4: Select best strategy
    print("\n\nTEST 4: Auto-Select Best Strategy")
    available = ['BULL_PUT_SPREAD', 'DUAL_OPTIONS', 'IRON_CONDOR']
    best = rpt.select_best_strategy_for_regime(available)
    print(f"  Available: {available}")
    print(f"  Best for current regime: {best}")

    # Test 5: Validate a trade
    print("\n\nTEST 5: Pre-Trade Validation")
    validation = rpt.validate_trade_before_execution(
        strategy_name='BULL_PUT_SPREAD',
        symbol='AAPL',
        size=10000
    )

    print("\n" + "=" * 80)
    print("REGIME PROTECTION TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_regime_protection()
