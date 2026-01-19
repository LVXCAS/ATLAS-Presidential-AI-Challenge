"""
LLM-Enhanced Technical Agent

Uses GPT-4 or Claude to reason through technical setups
instead of hardcoded rules.
"""

from typing import Dict, Tuple
from .base_agent import BaseAgent
import json
import os


class LLMTechnicalAgent(BaseAgent):
    """
    Technical agent powered by LLM reasoning.

    Instead of if/then rules, asks LLM to analyze market data
    and provide vote + reasoning.
    """

    def __init__(self, initial_weight: float = 1.5, model: str = "gpt-4"):
        super().__init__(name="LLMTechnicalAgent", initial_weight=initial_weight)

        self.model = model

        # Initialize LLM client (example with OpenAI)
        try:
            import openai
            self.openai = openai
            self.llm_available = True
        except ImportError:
            print(f"[{self.name}] WARNING: openai not installed, falling back to rules")
            self.llm_available = False

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Analyze market data using LLM reasoning.
        """
        if not self.llm_available:
            return self._fallback_rules_based_analysis(market_data)

        # Extract market data
        pair = market_data.get("pair", "UNKNOWN")
        price = market_data.get("price", 0)
        indicators = market_data.get("indicators", {})

        # Build prompt for LLM
        prompt = self._build_analysis_prompt(pair, price, indicators)

        # Get LLM response
        try:
            response = self._call_llm(prompt)
            vote, confidence, reasoning = self._parse_llm_response(response)
            return (vote, confidence, reasoning)
        except Exception as e:
            print(f"[{self.name}] LLM call failed: {e}, falling back to rules")
            return self._fallback_rules_based_analysis(market_data)

    def _build_analysis_prompt(self, pair: str, price: float, indicators: Dict) -> str:
        """
        Build detailed prompt for LLM with market context.
        """
        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd", 0)
        macd_hist = indicators.get("macd_hist", 0)
        adx = indicators.get("adx", 20)
        ema50 = indicators.get("ema50", price)
        ema200 = indicators.get("ema200", price)
        bb_upper = indicators.get("bb_upper", price * 1.02)
        bb_lower = indicators.get("bb_lower", price * 0.98)

        prompt = f"""You are a professional forex technical analyst with 20 years of experience.

Analyze this {pair} setup and provide a trading recommendation:

**Current Market State:**
- Price: {price:.5f}
- RSI (14): {rsi:.1f}
- MACD: {macd:.6f} (histogram: {macd_hist:.6f})
- ADX: {adx:.1f}
- EMA50: {ema50:.5f}
- EMA200: {ema200:.5f}
- Bollinger Bands: Upper {bb_upper:.5f}, Lower {bb_lower:.5f}

**Context:**
- Price vs EMA50: {((price/ema50 - 1)*100):+.2f}%
- Price vs EMA200: {((price/ema200 - 1)*100):+.2f}%
- BB Position: {((price - bb_lower)/(bb_upper - bb_lower)*100):.0f}% (0=lower, 100=upper)
- Trend Strength: {"Strong" if adx > 25 else "Weak" if adx > 20 else "No trend"} (ADX {adx:.1f})

**Your Task:**
1. Analyze the technical setup
2. Consider trend strength, momentum, mean reversion potential
3. Provide a clear BUY/SELL/NEUTRAL recommendation
4. Rate your confidence (0.0 to 1.0)
5. Explain your reasoning in 2-3 sentences

**Response Format (JSON):**
{{
    "vote": "BUY" | "SELL" | "NEUTRAL",
    "confidence": 0.0-1.0,
    "reasoning": "Your 2-3 sentence explanation"
}}

Be conservative - only vote BUY/SELL with high confidence when you see strong confirming signals.
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API (OpenAI example).
        """
        response = self.openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional technical analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=300
        )

        return response.choices[0].message.content

    def _parse_llm_response(self, response: str) -> Tuple[str, float, Dict]:
        """
        Parse LLM JSON response into vote, confidence, reasoning.
        """
        try:
            # Try to parse as JSON
            data = json.loads(response)
            vote = data.get("vote", "NEUTRAL").upper()
            confidence = float(data.get("confidence", 0.5))
            reasoning_text = data.get("reasoning", "LLM analysis")

            # Validate vote
            if vote not in ["BUY", "SELL", "NEUTRAL"]:
                vote = "NEUTRAL"
                confidence = 0.3

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            reasoning = {
                "agent": self.name,
                "vote": vote,
                "method": "llm_analysis",
                "model": self.model,
                "explanation": reasoning_text
            }

            return (vote, confidence, reasoning)

        except json.JSONDecodeError:
            # LLM didn't return valid JSON, parse text
            response_lower = response.lower()

            if "buy" in response_lower and "sell" not in response_lower:
                vote = "BUY"
                confidence = 0.65
            elif "sell" in response_lower and "buy" not in response_lower:
                vote = "SELL"
                confidence = 0.65
            else:
                vote = "NEUTRAL"
                confidence = 0.5

            reasoning = {
                "agent": self.name,
                "vote": vote,
                "method": "llm_text_parsing",
                "explanation": response[:200]  # First 200 chars
            }

            return (vote, confidence, reasoning)

    def _fallback_rules_based_analysis(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Fallback to traditional rule-based analysis if LLM unavailable.
        """
        indicators = market_data.get("indicators", {})
        price = market_data.get("price", 0)

        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd", 0)
        adx = indicators.get("adx", 20)
        ema50 = indicators.get("ema50", price)
        ema200 = indicators.get("ema200", price)

        # Simple rule-based logic
        score = 0

        if price > ema200 and macd > 0 and rsi < 70:
            score += 2
        elif price < ema200 and macd < 0 and rsi > 30:
            score -= 2

        if adx > 25:
            score *= 1.2  # Stronger signal in trending market

        if score >= 2:
            return ("BUY", 0.65, {"method": "rules_fallback"})
        elif score <= -2:
            return ("SELL", 0.65, {"method": "rules_fallback"})
        else:
            return ("NEUTRAL", 0.3, {"method": "rules_fallback"})
