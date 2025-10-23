#!/usr/bin/env python3
"""
Volatility Regime Adaptation
Adjusts position sizing and strategy based on VIX and historical volatility
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class VolatilityRegimeAdapter:
    """Adapt trading based on volatility regime"""

    def __init__(self):
        # VIX regime thresholds
        self.vix_regimes = {
            'LOW_VOL': {'max': 15, 'size_mult': 1.20, 'label': 'LOW VOLATILITY'},
            'NORMAL': {'min': 15, 'max': 25, 'size_mult': 1.00, 'label': 'NORMAL'},
            'ELEVATED': {'min': 25, 'max': 35, 'size_mult': 0.75, 'label': 'ELEVATED'},
            'HIGH_VOL': {'min': 35, 'max': 50, 'size_mult': 0.50, 'label': 'HIGH VOLATILITY'},
            'EXTREME': {'min': 50, 'size_mult': 0.30, 'label': 'EXTREME VOLATILITY'}
        }

        # Strategy adjustments per regime
        self.regime_adjustments = {
            'LOW_VOL': {
                'dte_preference': (28, 45),  # Longer DTE
                'strike_preference': 'OTM',  # More OTM strikes
                'hold_time_days': (21, 35),  # Longer holds
                'profit_target_mult': 1.2,  # Higher targets
                'stop_loss_mult': 1.0
            },
            'NORMAL': {
                'dte_preference': (21, 35),
                'strike_preference': 'BALANCED',
                'hold_time_days': (14, 28),
                'profit_target_mult': 1.0,
                'stop_loss_mult': 1.0
            },
            'ELEVATED': {
                'dte_preference': (14, 28),  # Shorter DTE
                'strike_preference': 'ATM',  # At-the-money
                'hold_time_days': (7, 14),  # Shorter holds
                'profit_target_mult': 0.8,  # Lower targets (take profits faster)
                'stop_loss_mult': 0.8  # Tighter stops
            },
            'HIGH_VOL': {
                'dte_preference': (7, 21),
                'strike_preference': 'ITM',  # In-the-money (more protection)
                'hold_time_days': (3, 10),
                'profit_target_mult': 0.6,
                'stop_loss_mult': 0.6
            },
            'EXTREME': {
                'dte_preference': (3, 14),
                'strike_preference': 'ITM',
                'hold_time_days': (1, 5),
                'profit_target_mult': 0.4,  # Take any profit
                'stop_loss_mult': 0.5
            }
        }

    def get_vix_data(self) -> Dict:
        """Fetch current VIX and recent history"""
        try:
            vix = yf.Ticker('^VIX')
            hist = vix.history(period='5d', interval='1d')

            if hist.empty:
                logger.warning("No VIX data available, using default")
                return {
                    'current': 20.0,
                    'ma_5d': 20.0,
                    'trend': 'STABLE',
                    'fetched': False
                }

            current_vix = float(hist['Close'].iloc[-1])
            ma_5d = float(hist['Close'].mean())

            # Determine trend
            if len(hist) >= 2:
                recent_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
                if recent_change > 0.10:
                    trend = 'RISING'
                elif recent_change < -0.10:
                    trend = 'FALLING'
                else:
                    trend = 'STABLE'
            else:
                trend = 'STABLE'

            return {
                'current': current_vix,
                'ma_5d': ma_5d,
                'trend': trend,
                'fetched': True
            }

        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return {
                'current': 20.0,
                'ma_5d': 20.0,
                'trend': 'STABLE',
                'fetched': False
            }

    def determine_regime(self, vix_value: float = None) -> Dict:
        """
        Determine current volatility regime

        Returns:
            {
                'regime': str,
                'vix': float,
                'size_multiplier': float,
                'adjustments': Dict,
                'description': str
            }
        """
        # Get VIX if not provided
        if vix_value is None:
            vix_data = self.get_vix_data()
            vix_value = vix_data['current']
            vix_trend = vix_data['trend']
        else:
            vix_trend = 'UNKNOWN'

        # Determine regime
        regime = None
        for regime_name, config in self.vix_regimes.items():
            if 'max' in config and 'min' in config:
                if config['min'] <= vix_value < config['max']:
                    regime = regime_name
                    break
            elif 'max' in config:
                if vix_value < config['max']:
                    regime = regime_name
                    break
            elif 'min' in config:
                if vix_value >= config['min']:
                    regime = regime_name
                    break

        if regime is None:
            regime = 'NORMAL'

        config = self.vix_regimes[regime]
        adjustments = self.regime_adjustments[regime]

        # Enhanced size multiplier if VIX is rising in elevated regimes
        size_mult = config['size_mult']
        if regime in ['ELEVATED', 'HIGH_VOL', 'EXTREME'] and vix_trend == 'RISING':
            size_mult *= 0.85  # Further reduce size if vol is spiking

        description = f"VIX {vix_value:.1f} - {config['label']}"
        if vix_trend != 'UNKNOWN':
            description += f" ({vix_trend})"

        return {
            'regime': regime,
            'vix': float(vix_value),
            'size_multiplier': float(size_mult),
            'adjustments': adjustments,
            'description': description,
            'vix_trend': vix_trend
        }

    def calculate_position_size(self, base_size: float, confidence: float,
                                vix_regime: Dict = None) -> Dict:
        """
        Calculate position size adjusted for volatility regime

        Args:
            base_size: Base position size (e.g., $1000)
            confidence: Trade confidence (0-1)
            vix_regime: Optional pre-calculated regime

        Returns:
            {
                'adjusted_size': float,
                'base_size': float,
                'vix_multiplier': float,
                'confidence_multiplier': float,
                'final_multiplier': float,
                'reasoning': str
            }
        """
        if vix_regime is None:
            vix_regime = self.determine_regime()

        vix_mult = vix_regime['size_multiplier']

        # Confidence multiplier (0.5x to 1.5x)
        # High confidence gets bigger, low confidence gets smaller
        if confidence >= 0.75:
            conf_mult = 1.3
        elif confidence >= 0.65:
            conf_mult = 1.0
        elif confidence >= 0.55:
            conf_mult = 0.7
        else:
            conf_mult = 0.5

        # Combined multiplier
        final_mult = vix_mult * conf_mult
        adjusted_size = base_size * final_mult

        reasoning = (
            f"Base ${base_size:.0f} × "
            f"VIX mult {vix_mult:.2f} ({vix_regime['regime']}) × "
            f"Conf mult {conf_mult:.2f} ({confidence:.0%}) = "
            f"${adjusted_size:.0f}"
        )

        return {
            'adjusted_size': float(adjusted_size),
            'base_size': float(base_size),
            'vix_multiplier': float(vix_mult),
            'confidence_multiplier': float(conf_mult),
            'final_multiplier': float(final_mult),
            'reasoning': reasoning
        }

    def get_strategy_params(self, vix_regime: Dict = None) -> Dict:
        """
        Get recommended strategy parameters for current regime

        Returns:
            {
                'dte_min': int,
                'dte_max': int,
                'strike_preference': str,
                'hold_days_min': int,
                'hold_days_max': int,
                'profit_target': float,
                'stop_loss': float,
                'regime': str
            }
        """
        if vix_regime is None:
            vix_regime = self.determine_regime()

        adj = vix_regime['adjustments']

        return {
            'dte_min': adj['dte_preference'][0],
            'dte_max': adj['dte_preference'][1],
            'strike_preference': adj['strike_preference'],
            'hold_days_min': adj['hold_time_days'][0],
            'hold_days_max': adj['hold_time_days'][1],
            'profit_target_mult': adj['profit_target_mult'],
            'stop_loss_mult': adj['stop_loss_mult'],
            'regime': vix_regime['regime'],
            'description': vix_regime['description']
        }

    def should_avoid_trading(self, vix_regime: Dict = None) -> Dict:
        """
        Check if trading should be avoided due to extreme volatility

        Returns:
            {
                'avoid': bool,
                'reason': str,
                'vix': float
            }
        """
        if vix_regime is None:
            vix_regime = self.determine_regime()

        regime = vix_regime['regime']
        vix = vix_regime['vix']

        # Avoid trading in extreme volatility (optional - very conservative)
        if regime == 'EXTREME' and vix > 60:
            return {
                'avoid': True,
                'reason': f"Extreme volatility (VIX {vix:.1f}) - markets unstable",
                'vix': vix
            }

        # Warning in high vol but don't block
        if regime in ['HIGH_VOL', 'EXTREME']:
            return {
                'avoid': False,
                'reason': f"High volatility (VIX {vix:.1f}) - reduced size recommended",
                'vix': vix
            }

        return {
            'avoid': False,
            'reason': f"Normal trading conditions (VIX {vix:.1f})",
            'vix': vix
        }


# Global instance
_volatility_adapter = None

def get_volatility_adapter() -> VolatilityRegimeAdapter:
    """Get singleton volatility adapter"""
    global _volatility_adapter
    if _volatility_adapter is None:
        _volatility_adapter = VolatilityRegimeAdapter()
    return _volatility_adapter


if __name__ == "__main__":
    # Test
    adapter = VolatilityRegimeAdapter()

    print("="*70)
    print("VOLATILITY REGIME ADAPTATION TEST")
    print("="*70)

    # Get current regime
    print("\n[CURRENT MARKET REGIME]")
    regime = adapter.determine_regime()
    print(f"Regime: {regime['regime']}")
    print(f"VIX: {regime['vix']:.1f}")
    print(f"Description: {regime['description']}")
    print(f"Size Multiplier: {regime['size_multiplier']:.2f}x")

    # Test position sizing
    print(f"\n[POSITION SIZING TEST]")
    base_size = 1000
    test_confidences = [0.80, 0.65, 0.55]

    for conf in test_confidences:
        size_result = adapter.calculate_position_size(
            base_size=base_size,
            confidence=conf,
            vix_regime=regime
        )
        print(f"\nConfidence {conf:.0%}:")
        print(f"  {size_result['reasoning']}")

    # Test strategy params
    print(f"\n[STRATEGY PARAMETERS]")
    params = adapter.get_strategy_params(regime)
    print(f"DTE Range: {params['dte_min']}-{params['dte_max']} days")
    print(f"Strike Preference: {params['strike_preference']}")
    print(f"Hold Time: {params['hold_days_min']}-{params['hold_days_max']} days")
    print(f"Profit Target Mult: {params['profit_target_mult']:.2f}x")
    print(f"Stop Loss Mult: {params['stop_loss_mult']:.2f}x")

    # Test different VIX levels
    print(f"\n[VIX SCENARIO TESTING]")
    test_vix_levels = [12, 20, 30, 45, 65]

    for vix in test_vix_levels:
        test_regime = adapter.determine_regime(vix)
        avoid_check = adapter.should_avoid_trading(test_regime)

        print(f"\nVIX {vix}:")
        print(f"  Regime: {test_regime['regime']}")
        print(f"  Size Mult: {test_regime['size_multiplier']:.2f}x")
        print(f"  Avoid: {avoid_check['avoid']} - {avoid_check['reason']}")
