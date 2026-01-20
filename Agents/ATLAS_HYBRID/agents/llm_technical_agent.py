"""
LLM-Assisted Technical Risk Agent (Educational)

Uses an optional LLM to summarize risk signals into a normalized 0..1 risk score.
This agent never outputs buy/sell advice and is disabled by default.
"""

from typing import Dict, Optional, Tuple
import json
import os
import urllib.request

from .base_agent import AgentAssessment, BaseAgent


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_llm_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


class LLMTechnicalAgent(BaseAgent):
    """
    Technical risk lens powered by LLM reasoning (optional).
    """

    def __init__(self, initial_weight: float = 1.0, model: str = "deepseek-chat"):
        super().__init__(name="LLMTechnicalAgent", initial_weight=initial_weight)
        self.model = model

        self.llm_enabled = _truthy(os.getenv("ENABLE_LLM_AGENTS"))
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = os.getenv("LLM_API_BASE") or ("https://api.deepseek.com" if os.getenv("DEEPSEEK_API_KEY") else "")
        self.llm_available = bool(self.llm_enabled and self.api_key and self.api_base)

    def analyze(self, market_data: Dict) -> AgentAssessment:
        indicators = market_data.get("indicators", {}) or {}
        price = float(market_data.get("price", 0.0) or 0.0)
        pair = str(market_data.get("pair", "UNKNOWN"))

        if not indicators or price <= 0:
            return AgentAssessment(
                score=0.5,
                explanation="Insufficient indicator data for LLM risk assessment.",
                details={"data_sufficiency": "insufficient"},
            )

        if not self.llm_available:
            return AgentAssessment(
                score=0.5,
                explanation="LLM agent disabled; returning NEUTRAL.",
                details={"data_sufficiency": "insufficient", "llm_status": "disabled"},
            )

        prompt, atr_pips = self._build_analysis_prompt(pair, price, indicators)
        try:
            response = self._call_llm(prompt)
            score, explanation, label = self._parse_llm_response(response)
            return AgentAssessment(
                score=score,
                explanation=explanation,
                details={"llm_status": "ok", "model": self.model, "risk_label": label, "atr_pips": atr_pips},
            )
        except Exception as exc:  # pragma: no cover - network dependent
            return self._fallback_rules_based_assessment(indicators, price, str(exc))

    def _build_analysis_prompt(self, pair: str, price: float, indicators: Dict) -> Tuple[str, float]:
        rsi = float(indicators.get("rsi", 50.0))
        macd = float(indicators.get("macd", 0.0))
        macd_hist = float(indicators.get("macd_hist", 0.0))
        adx = float(indicators.get("adx", 20.0))
        ema50 = float(indicators.get("ema50", price))
        ema200 = float(indicators.get("ema200", price))
        bb_upper = float(indicators.get("bb_upper", price * 1.02))
        bb_lower = float(indicators.get("bb_lower", price * 0.98))
        atr = float(indicators.get("atr", 0.0))
        atr_pips = (atr / price) * 10000.0 if price else 0.0

        prompt = f"""You are an educational risk-literacy assistant.
Analyze this market snapshot and return a risk score (0.0 to 1.0) plus a short explanation.
Do NOT give trading advice or predict prices.

Pair: {pair}
Price: {price:.5f}
RSI(14): {rsi:.1f}
MACD: {macd:.6f} (histogram {macd_hist:.6f})
ADX: {adx:.1f}
EMA50: {ema50:.5f}
EMA200: {ema200:.5f}
Bollinger: upper {bb_upper:.5f}, lower {bb_lower:.5f}
ATR(14) pips: {atr_pips:.1f}

Respond in JSON:
{{
  "risk_score": 0.0-1.0,
  "risk_label": "GREENLIGHT" | "WATCH" | "STAND_DOWN",
  "explanation": "1-2 sentences, student-friendly"
}}
"""
        return prompt, atr_pips

    def _call_llm(self, prompt: str) -> str:
        url = _build_llm_url(self.api_base)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You explain market risk without trading advice."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 250,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response missing choices.")
        return str(choices[0].get("message", {}).get("content", "")).strip()

    def _parse_llm_response(self, response: str) -> Tuple[float, str, str]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM response was not valid JSON.") from exc

        score = float(data.get("risk_score", 0.5))
        label = str(data.get("risk_label", "WATCH")).upper()
        explanation = str(data.get("explanation", "")).strip() or "LLM risk summary provided."

        score = max(0.0, min(1.0, score))
        if label not in {"GREENLIGHT", "WATCH", "STAND_DOWN"}:
            label = "WATCH"
        return score, explanation, label

    def _fallback_rules_based_assessment(
        self,
        indicators: Dict,
        price: float,
        error_message: str,
    ) -> AgentAssessment:
        rsi = float(indicators.get("rsi", 50.0))
        adx = float(indicators.get("adx", 20.0))
        atr = float(indicators.get("atr", 0.0))
        atr_pips = (atr / price) * 10000.0 if price else 0.0

        reasons = []
        score = 0.25

        if atr_pips >= 25:
            score = 0.90
            reasons.append("Volatility spike (ATR high)")
        elif atr_pips >= 15:
            score = max(score, 0.65)
            reasons.append("Volatility elevated (ATR medium)")

        if rsi >= 72 or rsi <= 28:
            score = max(score, 0.60)
            reasons.append("Momentum extreme (RSI)")

        if adx <= 18:
            score = max(score, 0.60)
            reasons.append("Choppy regime (low ADX)")

        if not reasons:
            explanation = "No major risk flags; conditions look relatively stable."
        else:
            explanation = "; ".join(reasons)

        return AgentAssessment(
            score=score,
            explanation=explanation,
            details={
                "llm_status": "fallback",
                "error": str(error_message)[:120],
                "atr_pips": round(atr_pips, 1),
            },
        )
