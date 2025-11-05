"""
AI CONFIRMATION LAYER FOR MULTI-MARKET TRADING
Uses free DeepSeek V3.1 + MiniMax via OpenRouter for trade validation
Acts as SECONDARY confirmation - TA-Lib signals remain PRIMARY

Architecture:
1. TA-Lib generates trade candidates (RSI, MACD, EMA, ADX scoring)
2. AI analyzes market context for high-confidence trades (score >= 6.0)
3. Multi-model voting: Both DeepSeek + MiniMax must agree
4. AI decision: APPROVE / REDUCE_SIZE / REJECT
5. Kelly multiplier adjusted based on AI confidence

Benefits:
- Zero cost (free OpenRouter APIs)
- Filters bad trades during news/volatility spikes
- Adds intelligent context without replacing proven quant signals
- Fallback to TA-only if APIs unavailable
"""
import os
import requests
import json
from datetime import datetime
from typing import Dict, Optional, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class AIConfirmationAgent:
    """
    AI-powered trade confirmation using free LLM APIs
    Uses multi-model voting (DeepSeek V3.1 + MiniMax) for consensus
    """

    def __init__(self):
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

        if not self.openrouter_api_key:
            print("[WARN] OPENROUTER_API_KEY not set - AI confirmation disabled")
            self.enabled = False
        else:
            self.enabled = True
            print("[AI] Confirmation layer enabled (DeepSeek V3.1 + MiniMax)")

        # OpenRouter endpoint
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Free models on OpenRouter
        self.models = {
            'deepseek': 'deepseek/deepseek-chat',  # DeepSeek V3.1 (free)
            'minimax': 'minimax/minimax-01'        # MiniMax (free)
        }

        # Voting threshold: Both models must agree
        self.consensus_required = True

        # Rate limiting
        self.last_call_time = {}
        self.min_call_interval = 1.0  # 1 second between calls

    def _call_llm(self, model_name: str, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """
        Call OpenRouter API with specified model

        Args:
            model_name: 'deepseek' or 'minimax'
            prompt: The analysis prompt
            max_tokens: Max response length

        Returns:
            Model response string or None if error
        """
        if not self.enabled:
            return None

        # Rate limiting
        now = datetime.now().timestamp()
        if model_name in self.last_call_time:
            elapsed = now - self.last_call_time[model_name]
            if elapsed < self.min_call_interval:
                import time
                time.sleep(self.min_call_interval - elapsed)

        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.models[model_name],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional quantitative trader analyzing potential trades. Respond concisely with structured output."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3  # Low temperature for consistent analysis
            }

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            self.last_call_time[model_name] = datetime.now().timestamp()

            return data['choices'][0]['message']['content']

        except requests.exceptions.Timeout:
            print(f"[AI] {model_name.upper()} timeout - skipping AI confirmation")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[AI] {model_name.upper()} API error: {e}")
            return None
        except Exception as e:
            print(f"[AI] {model_name.upper()} unexpected error: {e}")
            return None

    def _parse_ai_response(self, response: str) -> Dict:
        """
        Parse AI model response into structured decision

        Expected format:
            ACTION: [APPROVE / REJECT / REDUCE_SIZE]
            CONFIDENCE: [0-100]
            REASON: [explanation]

        Returns:
            {
                'action': 'APPROVE' | 'REJECT' | 'REDUCE_SIZE',
                'confidence': 0-100,
                'reason': str
            }
        """
        try:
            lines = response.strip().split('\n')
            result = {'action': 'REJECT', 'confidence': 0, 'reason': 'Parse error'}

            for line in lines:
                line = line.strip()

                if line.startswith('ACTION:'):
                    action = line.split('ACTION:')[1].strip().upper()
                    if action in ['APPROVE', 'REJECT', 'REDUCE_SIZE']:
                        result['action'] = action

                elif line.startswith('CONFIDENCE:'):
                    try:
                        conf = line.split('CONFIDENCE:')[1].strip()
                        result['confidence'] = int(conf)
                    except ValueError:
                        result['confidence'] = 0

                elif line.startswith('REASON:'):
                    result['reason'] = line.split('REASON:')[1].strip()

            return result

        except Exception as e:
            print(f"[AI] Parse error: {e}")
            return {'action': 'REJECT', 'confidence': 0, 'reason': f'Parse error: {e}'}

    def _create_forex_prompt(self, trade_data: Dict) -> str:
        """Create analysis prompt for forex trades"""
        pair = trade_data['symbol']
        direction = trade_data['direction'].upper()
        score = trade_data['score']
        rsi = trade_data.get('rsi', 50)
        macd = trade_data.get('macd', {})
        price = trade_data.get('current_price', 0)
        trend_4h = trade_data.get('trend_4h', 'neutral')

        prompt = f"""You are analyzing a FOREX trade opportunity.

TECHNICAL ANALYSIS SIGNALS:
- Pair: {pair}
- Direction: {direction}
- Technical Score: {score:.1f}/10
- RSI: {rsi:.1f} (oversold <30, overbought >70)
- MACD: {"Bullish" if macd.get('histogram', 0) > 0 else "Bearish"}
- 4H Trend: {trend_4h}
- Current Price: {price}

YOUR TASK:
Analyze if NOW is a good time to enter this {direction} trade. Consider:
1. Major forex news events (Fed, ECB, BOJ, NFP, CPI)
2. Market session timing (London open = volatile, NY afternoon = choppy)
3. Risk events that could invalidate technical signals
4. Time of day appropriateness for {pair}

RESPOND IN THIS EXACT FORMAT:
ACTION: [APPROVE or REJECT or REDUCE_SIZE]
CONFIDENCE: [0-100]
REASON: [One sentence explaining your decision]

Example response:
ACTION: APPROVE
CONFIDENCE: 85
REASON: Strong technical setup during London session with no major news conflicts.
"""
        return prompt

    def _create_futures_prompt(self, trade_data: Dict) -> str:
        """Create analysis prompt for futures trades"""
        contract = trade_data['symbol']
        direction = trade_data['direction'].upper()
        score = trade_data['score']
        rsi = trade_data.get('rsi', 50)
        adx = trade_data.get('adx', 20)
        price = trade_data.get('current_price', 0)

        prompt = f"""You are analyzing a FUTURES trade opportunity.

TECHNICAL ANALYSIS SIGNALS:
- Contract: {contract}
- Direction: {direction}
- Technical Score: {score:.1f}/10
- RSI: {rsi:.1f}
- ADX: {adx:.1f} (trend strength, >25 = strong trend)
- Current Price: {price}

YOUR TASK:
Analyze if NOW is a good time to enter this {direction} futures trade. Consider:
1. US market hours (best: 9:30 AM - 4:00 PM ET)
2. Overnight risk and gap potential
3. Correlation between ES/NQ (do they confirm each other?)
4. Major economic releases (jobs, CPI, Fed meetings)

RESPOND IN THIS EXACT FORMAT:
ACTION: [APPROVE or REJECT or REDUCE_SIZE]
CONFIDENCE: [0-100]
REASON: [One sentence explaining your decision]
"""
        return prompt

    def _create_crypto_prompt(self, trade_data: Dict) -> str:
        """Create analysis prompt for crypto trades"""
        pair = trade_data['symbol']
        direction = trade_data['direction'].upper()
        score = trade_data['score']
        rsi = trade_data.get('rsi', 50)
        bb = trade_data.get('bollinger', {})
        price = trade_data.get('current_price', 0)

        prompt = f"""You are analyzing a CRYPTO trade opportunity.

TECHNICAL ANALYSIS SIGNALS:
- Pair: {pair}
- Direction: {direction}
- Technical Score: {score:.1f}/10
- RSI: {rsi:.1f}
- Bollinger Position: {bb.get('position', 'middle')}
- Current Price: ${price:,.2f}

YOUR TASK:
Analyze if NOW is a good time to enter this {direction} crypto trade. Consider:
1. Bitcoin correlation (most alts follow BTC)
2. 24/7 trading = weekend gaps and low liquidity periods
3. High volatility = wider stops needed
4. Whale movements and social media sentiment
5. Time of day (US hours = higher volume)

RESPOND IN THIS EXACT FORMAT:
ACTION: [APPROVE or REJECT or REDUCE_SIZE]
CONFIDENCE: [0-100]
REASON: [One sentence explaining your decision]
"""
        return prompt

    def analyze_trade(self, trade_data: Dict, market_type: str = 'forex') -> Dict:
        """
        Analyze a trade candidate using multi-model AI consensus

        Args:
            trade_data: Trade candidate from TA-Lib scoring
                {
                    'symbol': 'EUR_USD',
                    'direction': 'long',
                    'score': 7.5,
                    'rsi': 28.5,
                    'macd': {...},
                    'current_price': 1.0850,
                    'trend_4h': 'bullish'
                }
            market_type: 'forex', 'futures', or 'crypto'

        Returns:
            {
                'action': 'APPROVE' | 'REJECT' | 'REDUCE_SIZE',
                'confidence': 0-100,
                'deepseek_decision': {...},
                'minimax_decision': {...},
                'consensus': bool,
                'reason': str
            }
        """
        if not self.enabled:
            # Fallback: Auto-approve if AI disabled (TA-only mode)
            return {
                'action': 'APPROVE',
                'confidence': 100,
                'deepseek_decision': None,
                'minimax_decision': None,
                'consensus': True,
                'reason': 'AI confirmation disabled - using TA-only mode'
            }

        # Create market-specific prompt
        if market_type == 'forex':
            prompt = self._create_forex_prompt(trade_data)
        elif market_type == 'futures':
            prompt = self._create_futures_prompt(trade_data)
        elif market_type == 'crypto':
            prompt = self._create_crypto_prompt(trade_data)
        else:
            return {
                'action': 'REJECT',
                'confidence': 0,
                'deepseek_decision': None,
                'minimax_decision': None,
                'consensus': False,
                'reason': f'Unknown market type: {market_type}'
            }

        # Get responses from both models
        deepseek_response = self._call_llm('deepseek', prompt)
        minimax_response = self._call_llm('minimax', prompt)

        # Parse responses
        deepseek_decision = self._parse_ai_response(deepseek_response) if deepseek_response else None
        minimax_decision = self._parse_ai_response(minimax_response) if minimax_response else None

        # Handle API failures (fallback to TA-only)
        if not deepseek_decision and not minimax_decision:
            return {
                'action': 'APPROVE',
                'confidence': 100,
                'deepseek_decision': None,
                'minimax_decision': None,
                'consensus': True,
                'reason': 'AI APIs unavailable - fallback to TA-only mode'
            }

        # Multi-model voting logic
        if deepseek_decision and minimax_decision:
            # Both models responded - check consensus
            if deepseek_decision['action'] == minimax_decision['action']:
                # Consensus reached
                avg_confidence = (deepseek_decision['confidence'] + minimax_decision['confidence']) / 2
                return {
                    'action': deepseek_decision['action'],
                    'confidence': avg_confidence,
                    'deepseek_decision': deepseek_decision,
                    'minimax_decision': minimax_decision,
                    'consensus': True,
                    'reason': f"Consensus: {deepseek_decision['action']} ({avg_confidence:.0f}% confidence)"
                }
            else:
                # Disagreement - default to REDUCE_SIZE (conservative)
                return {
                    'action': 'REDUCE_SIZE',
                    'confidence': 50,
                    'deepseek_decision': deepseek_decision,
                    'minimax_decision': minimax_decision,
                    'consensus': False,
                    'reason': f"Models disagree (DeepSeek: {deepseek_decision['action']}, MiniMax: {minimax_decision['action']})"
                }
        elif deepseek_decision:
            # Only DeepSeek responded
            return {
                'action': deepseek_decision['action'],
                'confidence': deepseek_decision['confidence'],
                'deepseek_decision': deepseek_decision,
                'minimax_decision': None,
                'consensus': False,
                'reason': f"DeepSeek only: {deepseek_decision['reason']}"
            }
        else:
            # Only MiniMax responded
            return {
                'action': minimax_decision['action'],
                'confidence': minimax_decision['confidence'],
                'deepseek_decision': None,
                'minimax_decision': minimax_decision,
                'consensus': False,
                'reason': f"MiniMax only: {minimax_decision['reason']}"
            }

    def adjust_kelly_multiplier(self, base_multiplier: float, ai_decision: Dict) -> float:
        """
        Adjust Kelly Criterion multiplier based on AI confidence

        Args:
            base_multiplier: Original Kelly multiplier from TA scoring (0.5 - 1.5)
            ai_decision: AI analysis result

        Returns:
            Adjusted multiplier:
                - APPROVE + high confidence: base_multiplier * 1.0 (unchanged)
                - APPROVE + low confidence: base_multiplier * 0.8 (reduce 20%)
                - REDUCE_SIZE: base_multiplier * 0.5 (half size)
                - REJECT: 0.0 (skip trade)
        """
        action = ai_decision['action']
        confidence = ai_decision['confidence']

        if action == 'REJECT':
            return 0.0

        elif action == 'REDUCE_SIZE':
            return base_multiplier * 0.5

        elif action == 'APPROVE':
            if confidence >= 80:
                return base_multiplier  # High confidence - no adjustment
            elif confidence >= 60:
                return base_multiplier * 0.9  # Medium confidence - slight reduction
            else:
                return base_multiplier * 0.8  # Low confidence - reduce 20%

        else:
            # Unknown action - default to half size
            return base_multiplier * 0.5


# Singleton instance
ai_agent = AIConfirmationAgent()
