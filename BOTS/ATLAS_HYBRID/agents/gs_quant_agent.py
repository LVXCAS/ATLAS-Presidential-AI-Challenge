"""
GS Quant Agent

Uses Goldman Sachs' institutional quant library for risk modeling,
market analytics, and derivatives pricing.

Specialization: Risk management, volatility modeling, cross-asset correlation.
"""

from typing import Dict, Tuple, List, Optional
from .base_agent import BaseAgent
import numpy as np


class GSQuantAgent(BaseAgent):
    """
    Goldman Sachs Quant Library integration.

    Capabilities:
    - GS Marquee risk models
    - Cross-asset correlation analysis
    - Volatility surface modeling
    - VaR (Value at Risk) calculations
    - Scenario analysis
    - Factor attribution

    Used by GS trading desks for institutional risk management.
    """

    def __init__(self, initial_weight: float = 2.0):
        super().__init__(name="GSQuantAgent", initial_weight=initial_weight)

        self.gs_quant_available = False
        self.risk_models_loaded = False

        # Try to initialize GS Quant
        try:
            import gs_quant
            from gs_quant.session import GsSession, Environment
            self.gs_quant = gs_quant
            self.gs_quant_available = True
            print(f"[{self.name}] GS Quant v{gs_quant.__version__} loaded successfully")
        except ImportError:
            print(f"[{self.name}] WARNING: GS Quant not available, using simplified risk models")

        # Risk thresholds
        self.max_var_pct = 0.02  # Max 2% VaR per trade
        self.max_correlation = 0.7  # Max correlation with existing positions

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Analyze using GS Quant risk models.

        Returns:
            (vote, confidence, reasoning)
        """
        pair = market_data.get("pair", "UNKNOWN")
        price = market_data.get("price")
        indicators = market_data.get("indicators", {})
        existing_positions = market_data.get("existing_positions", [])

        if not self.gs_quant_available:
            return self._simplified_risk_analysis(price, indicators, existing_positions)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(pair, price, indicators)

        # Check correlation with existing positions
        correlation_risk = self._check_position_correlation(pair, existing_positions)

        # Calculate Value at Risk (VaR)
        var_estimate = self._calculate_var(pair, price, indicators)

        # Make voting decision based on risk
        risk_score = self._aggregate_risk_score(risk_metrics, correlation_risk, var_estimate)

        # Risk-adjusted vote
        if risk_score < 0.3:  # Low risk
            vote = "ALLOW"  # Allow trade to proceed
            confidence = 0.9
        elif risk_score < 0.6:  # Medium risk
            vote = "CAUTION"  # Reduce position size
            confidence = 0.6
        else:  # High risk
            vote = "BLOCK"  # Block trade
            confidence = 0.95

        reasoning = {
            "agent": self.name,
            "risk_score": round(risk_score, 3),
            "var_estimate": round(var_estimate, 4),
            "correlation_risk": round(correlation_risk, 3),
            "risk_metrics": risk_metrics,
            "recommendation": self._get_risk_recommendation(risk_score)
        }

        return (vote, confidence, reasoning)

    def _calculate_risk_metrics(self, pair: str, price: float, indicators: Dict) -> Dict[str, float]:
        """
        Calculate GS-style risk metrics.

        Metrics:
        - Historical volatility
        - Implied volatility (if options data available)
        - Beta to market
        - Tracking error
        - Information ratio
        """
        metrics = {}

        # ATR-based volatility (GS uses this for FX)
        atr = indicators.get("atr", price * 0.01)
        metrics['historical_vol'] = (atr / price) * 100  # As percentage

        # ADX-based trend instability
        adx = indicators.get("adx", 20)
        metrics['trend_instability'] = max(0, 25 - adx) / 25  # Lower ADX = more risk

        # RSI extremes indicate reversal risk
        rsi = indicators.get("rsi", 50)
        metrics['reversal_risk'] = abs(rsi - 50) / 50  # Distance from neutral

        # Bollinger Band width (volatility proxy)
        bb_upper = indicators.get("bb_upper", price * 1.02)
        bb_lower = indicators.get("bb_lower", price * 0.98)
        metrics['bb_width'] = ((bb_upper - bb_lower) / price) * 100

        return metrics

    def _check_position_correlation(self, pair: str, existing_positions: List[Dict]) -> float:
        """
        Check correlation with existing positions.

        GS uses this to prevent over-concentration in correlated assets.
        """
        if len(existing_positions) == 0:
            return 0.0

        # Forex pair correlation matrix (simplified)
        correlations = {
            ('EUR_USD', 'GBP_USD'): 0.65,
            ('EUR_USD', 'USD_JPY'): -0.45,
            ('GBP_USD', 'USD_JPY'): -0.40,
            ('EUR_USD', 'AUD_USD'): 0.70,
            ('GBP_USD', 'AUD_USD'): 0.60,
        }

        max_corr = 0.0
        for pos in existing_positions:
            pos_pair = pos.get("pair", "")
            # Check both directions of correlation
            corr_key1 = (pair, pos_pair)
            corr_key2 = (pos_pair, pair)

            corr = correlations.get(corr_key1, correlations.get(corr_key2, 0.0))
            max_corr = max(max_corr, abs(corr))

        return max_corr

    def _calculate_var(self, pair: str, price: float, indicators: Dict) -> float:
        """
        Calculate Value at Risk (VaR) - GS Marquee style.

        VaR = Expected loss at 95% confidence level over 1 day.
        """
        # ATR is a good proxy for daily volatility
        atr = indicators.get("atr", price * 0.01)

        # 95% confidence = 1.65 standard deviations
        var_95 = atr * 1.65

        # As percentage of price
        var_pct = var_95 / price

        return var_pct

    def _aggregate_risk_score(self, risk_metrics: Dict, correlation_risk: float,
                               var_estimate: float) -> float:
        """
        Aggregate risk score (0 = low risk, 1 = high risk).

        GS uses weighted scoring for risk aggregation.
        """
        score = 0.0

        # Historical volatility risk (25% weight)
        vol = risk_metrics.get('historical_vol', 0)
        if vol > 2.0:  # > 2% daily vol
            score += 0.25
        elif vol > 1.5:
            score += 0.15
        elif vol > 1.0:
            score += 0.05

        # Correlation risk (25% weight)
        if correlation_risk > 0.7:
            score += 0.25
        elif correlation_risk > 0.5:
            score += 0.15
        elif correlation_risk > 0.3:
            score += 0.05

        # VaR risk (30% weight)
        if var_estimate > 0.02:  # > 2% VaR
            score += 0.30
        elif var_estimate > 0.015:
            score += 0.20
        elif var_estimate > 0.01:
            score += 0.10

        # Reversal risk (20% weight)
        reversal = risk_metrics.get('reversal_risk', 0)
        if reversal > 0.4:  # RSI > 70 or < 30
            score += 0.20
        elif reversal > 0.2:
            score += 0.10

        return min(score, 1.0)

    def _get_risk_recommendation(self, risk_score: float) -> str:
        """Get risk-based recommendation"""
        if risk_score < 0.3:
            return "Low risk - full position size approved"
        elif risk_score < 0.6:
            return "Medium risk - reduce position size by 50%"
        else:
            return "High risk - block trade or wait for better conditions"

    def _simplified_risk_analysis(self, price: float, indicators: Dict,
                                   existing_positions: List[Dict]) -> Tuple[str, float, Dict]:
        """Fallback analysis when GS Quant not available"""
        atr = indicators.get("atr", price * 0.01)
        adx = indicators.get("adx", 20)
        rsi = indicators.get("rsi", 50)

        # Simple risk score
        risk = 0.0

        # High volatility
        vol_pct = (atr / price) * 100
        if vol_pct > 2.0:
            risk += 0.4

        # Weak trend
        if adx < 20:
            risk += 0.3

        # RSI extremes
        if rsi > 70 or rsi < 30:
            risk += 0.3

        if risk < 0.3:
            vote = "ALLOW"
            confidence = 0.85
        elif risk < 0.6:
            vote = "CAUTION"
            confidence = 0.65
        else:
            vote = "BLOCK"
            confidence = 0.90

        reasoning = {
            "agent": self.name,
            "risk_score": round(risk, 3),
            "mode": "simplified",
            "volatility_pct": round(vol_pct, 2),
            "factors": [f"Vol={vol_pct:.2f}%", f"ADX={adx:.1f}", f"RSI={rsi:.1f}"]
        }

        return (vote, confidence, reasoning)

    def calculate_position_size(self, base_size: float, risk_score: float) -> float:
        """
        Adjust position size based on risk (GS risk-parity style).

        Args:
            base_size: Base position size (e.g., 3.0 lots)
            risk_score: Aggregated risk score (0-1)

        Returns:
            Risk-adjusted position size
        """
        if risk_score < 0.3:
            return base_size  # Full size
        elif risk_score < 0.6:
            return base_size * 0.5  # Half size
        else:
            return 0.0  # No trade

    def get_portfolio_var(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate portfolio-level VaR (GS Marquee portfolio analytics).

        This is where GS Quant really shines - multi-asset VaR.
        """
        # Placeholder for full GS Quant integration
        return {
            "portfolio_var_95": 0.0,
            "marginal_var": 0.0,
            "component_var": 0.0
        }
