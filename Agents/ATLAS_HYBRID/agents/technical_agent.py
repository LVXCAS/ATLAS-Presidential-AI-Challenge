"""
Technical Agent (Educational, Simulation-Only)

This agent looks at common technical indicators and outputs:
- a normalized risk/uncertainty score in the range 0..1
- a short explanation string for students

Important: this agent does NOT place trades and does NOT use any live data.
It only reads the historical/delayed snapshot passed in `market_data`.
"""

from __future__ import annotations

from typing import Any, Dict

from .base_agent import AgentAssessment, BaseAgent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class TechnicalAgent(BaseAgent):
    """
    Scores risk based on volatility + "stretched" conditions.

    Inputs expected (provided by `quant_team_utils.make_market_data`):
    - `market_data["price"]`
    - `market_data["indicators"]`: rsi, atr, bb_upper/bb_lower, ema50/ema200, macd_hist
    """

    def __init__(self, initial_weight: float = 1.5):
        super().__init__(name="TechnicalAgent", initial_weight=initial_weight)

        # Beginner-friendly thresholds (tunable).
        self.rsi_extreme = 70.0
        self.atr_pips_low = 10.0
        self.atr_pips_medium = 15.0
        self.atr_pips_high = 25.0

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        indicators = market_data.get("indicators", {}) or {}
        price = float(market_data.get("price", 0.0) or 0.0)
        if price <= 0:
            return AgentAssessment(
                score=0.5,
                explanation="Missing price; technical risk is uncertain.",
                details={"error": "missing_price", "data_sufficiency": "insufficient"},
            )

        rsi = float(indicators.get("rsi", 50.0) or 50.0)
        atr = float(indicators.get("atr", 0.0) or 0.0)
        ema50 = float(indicators.get("ema50", price) or price)
        ema200 = float(indicators.get("ema200", price) or price)
        bb_upper = float(indicators.get("bb_upper", price) or price)
        bb_lower = float(indicators.get("bb_lower", price) or price)
        macd_hist = float(indicators.get("macd_hist", 0.0) or 0.0)

        atr_pips = (atr / price) * 10000.0 if price else 0.0

        # --- Component risks (0..1) ---

        # 1) Volatility: map ATR pips into a risk score.
        # We treat ATR as a "how jumpy is the market?" proxy:
        # - <= low: calm (0.0)
        # - low..medium: rising risk up to 0.5
        # - medium..high: rising risk up to 1.0
        if atr_pips >= self.atr_pips_high:
            vol_risk = 1.0
        elif atr_pips <= self.atr_pips_low:
            vol_risk = 0.0
        elif atr_pips <= self.atr_pips_medium:
            vol_risk = 0.5 * ((atr_pips - self.atr_pips_low) / (self.atr_pips_medium - self.atr_pips_low))
        else:
            vol_risk = 0.5 + 0.5 * ((atr_pips - self.atr_pips_medium) / (self.atr_pips_high - self.atr_pips_medium))
        vol_risk = _clamp01(vol_risk)

        # 2) Momentum stretch: RSI far from 50 increases reversal uncertainty.
        rsi_distance = abs(rsi - 50.0)
        rsi_risk = _clamp01((max(0.0, rsi_distance - 15.0)) / 35.0)

        # 3) Bollinger "outside band" risk: price outside bands can be unstable.
        if price > bb_upper:
            band_risk = _clamp01(((price - bb_upper) / price) / 0.002)  # ~0.2% outside => high risk
        elif price < bb_lower:
            band_risk = _clamp01(((bb_lower - price) / price) / 0.002)
        else:
            band_risk = 0.0

        # 4) Trend clarity: if EMA50 is very close to EMA200, trend is unclear.
        ema_sep = abs(ema50 - ema200) / price
        trend_uncertainty = _clamp01((0.0006 - ema_sep) / 0.0006)  # <0.06% separation => more uncertainty

        # 5) Momentum clarity: very small MACD histogram implies weak momentum.
        macd_uncertainty = _clamp01((0.00005 - abs(macd_hist)) / 0.00005)

        # Weighted combination (kept simple and explainable).
        score = _clamp01(
            (0.35 * vol_risk)
            + (0.25 * rsi_risk)
            + (0.15 * band_risk)
            + (0.15 * trend_uncertainty)
            + (0.10 * macd_uncertainty)
        )

        # Pick the most important driver for the short explanation.
        drivers = [
            ("volatility", 0.35 * vol_risk),
            ("rsi", 0.25 * rsi_risk),
            ("bollinger", 0.15 * band_risk),
            ("trend", 0.15 * trend_uncertainty),
            ("macd", 0.10 * macd_uncertainty),
        ]
        top_driver = max(drivers, key=lambda d: d[1])[0]

        if top_driver == "volatility" and vol_risk > 0:
            explanation = f"Volatility is elevated (ATR ≈ {atr_pips:.1f} pips)."
        elif top_driver == "rsi" and rsi_risk > 0:
            explanation = f"Momentum looks stretched (RSI {rsi:.0f}), which can raise reversal risk."
        elif top_driver == "bollinger" and band_risk > 0:
            explanation = "Price is outside the Bollinger Bands; moves can snap back quickly."
        elif top_driver == "trend" and trend_uncertainty > 0:
            explanation = "Trend is unclear (EMA50 is close to EMA200), increasing uncertainty."
        elif top_driver == "macd" and macd_uncertainty > 0:
            explanation = "Momentum is weak (MACD histogram near zero), so signals are less clear."
        else:
            explanation = "Technical conditions look relatively stable (no major indicator warnings)."

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={
                "rsi": round(rsi, 2),
                "atr_pips": round(atr_pips, 2),
                "ema50": ema50,
                "ema200": ema200,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "macd_hist": macd_hist,
                "components": {
                    "vol_risk": round(vol_risk, 3),
                    "rsi_risk": round(rsi_risk, 3),
                    "band_risk": round(band_risk, 3),
                    "trend_uncertainty": round(trend_uncertainty, 3),
                    "macd_uncertainty": round(macd_uncertainty, 3),
                },
            },
        )
