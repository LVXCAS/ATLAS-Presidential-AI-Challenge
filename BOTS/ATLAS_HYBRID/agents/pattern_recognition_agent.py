"""
Pattern Recognition Agent

Learns high-probability setups from historical trade data.

This agent gets SMARTER over time as it discovers patterns like:
- "EUR/USD + RSI 38-42 during London open = 78% win rate"
- "USD/JPY + Volume spike + ADX >30 = 2.5R average"

Specialization: Machine learning / pattern discovery
"""

from typing import Dict, Tuple, List
from collections import defaultdict
import json
from .base_agent import BaseAgent


class PatternRecognitionAgent(BaseAgent):
    """
    Discovers and exploits high-probability trading patterns.

    Learning process:
    1. After each trade, extract setup conditions
    2. Group similar setups together
    3. Calculate win rate and avg R-multiple for each pattern
    4. Vote based on pattern matches

    Example patterns:
    - "EUR_USD_RSI_38-42_LONDON" → 78% WR, 2.3R avg (42 samples)
    - "GBP_USD_VOLUME_SPIKE_ADX_30+" → 72% WR, 2.5R avg (28 samples)
    """

    def __init__(self, initial_weight: float = 1.0, min_pattern_samples: int = 10):
        super().__init__(name="PatternRecognitionAgent", initial_weight=initial_weight)

        # Pattern library
        # Key: pattern_id, Value: {win_rate, avg_r, sample_size, conditions}
        self.patterns = {}

        # Minimum samples before trusting a pattern
        self.min_pattern_samples = min_pattern_samples

        # Pattern matching threshold
        self.high_confidence_win_rate = 0.70  # 70%+ = strong pattern
        self.medium_confidence_win_rate = 0.60  # 60-70% = decent pattern

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Check if current setup matches any known high-probability patterns.

        Returns:
            (vote, confidence, reasoning)
        """
        # Extract current setup characteristics
        setup = self._extract_setup_characteristics(market_data)

        # Find matching patterns
        matches = self._find_matching_patterns(setup)

        if not matches:
            # No patterns matched
            return ("NEUTRAL", 0.0, {
                "reason": "No known patterns matched",
                "setup": setup
            })

        # Find best matching pattern
        best_match = max(matches, key=lambda m: m['confidence_score'])

        pattern = best_match['pattern']
        confidence_score = best_match['confidence_score']

        # Determine vote based on pattern
        if pattern['win_rate'] >= self.high_confidence_win_rate:
            vote = "BUY"  # High-probability pattern
            confidence = min(0.90, confidence_score)
        elif pattern['win_rate'] >= self.medium_confidence_win_rate:
            vote = "BUY"  # Decent pattern
            confidence = min(0.75, confidence_score)
        else:
            vote = "NEUTRAL"  # Pattern not strong enough
            confidence = 0.5

        reasoning = {
            "pattern_matched": best_match['pattern_id'],
            "pattern_win_rate": round(pattern['win_rate'] * 100, 1),
            "pattern_avg_r": round(pattern['avg_r'], 2),
            "pattern_samples": pattern['sample_size'],
            "confidence_score": round(confidence_score, 2),
            "setup_characteristics": setup
        }

        return (vote, confidence, reasoning)

    def _extract_setup_characteristics(self, market_data: Dict) -> Dict:
        """
        Extract key characteristics from current market setup.

        These are used to match against learned patterns.
        """
        pair = market_data.get("pair", "EUR_USD")
        indicators = market_data.get("indicators", {})
        session = market_data.get("session", "unknown")
        price = market_data.get("price", 0)

        # Extract relevant features
        rsi = indicators.get("rsi", 50)
        adx = indicators.get("adx", 20)
        volume_spike = indicators.get("volume_spike", False)
        ema50 = indicators.get("ema50", price)
        ema200 = indicators.get("ema200", price)

        # Discretize continuous values for pattern matching
        rsi_bucket = self._bucket_rsi(rsi)
        adx_bucket = self._bucket_adx(adx)
        trend = "bullish" if price > ema200 else "bearish"
        ema_alignment = "aligned" if (price > ema50 > ema200) else "misaligned"

        return {
            "pair": pair,
            "rsi_bucket": rsi_bucket,
            "adx_bucket": adx_bucket,
            "trend": trend,
            "ema_alignment": ema_alignment,
            "volume_spike": volume_spike,
            "session": session,
        }

    def _bucket_rsi(self, rsi: float) -> str:
        """Bucket RSI into ranges for pattern matching."""
        if rsi < 30:
            return "oversold"
        elif rsi < 40:
            return "35-40"
        elif rsi < 45:
            return "40-45"
        elif rsi < 55:
            return "neutral"
        elif rsi < 60:
            return "55-60"
        elif rsi < 70:
            return "60-70"
        else:
            return "overbought"

    def _bucket_adx(self, adx: float) -> str:
        """Bucket ADX into ranges."""
        if adx < 20:
            return "weak"
        elif adx < 30:
            return "moderate"
        else:
            return "strong"

    def _find_matching_patterns(self, setup: Dict) -> List[Dict]:
        """
        Find patterns that match current setup.

        Args:
            setup: Current setup characteristics

        Returns:
            List of matching patterns with confidence scores
        """
        matches = []

        for pattern_id, pattern in self.patterns.items():
            # Check if pattern has enough samples
            if pattern['sample_size'] < self.min_pattern_samples:
                continue

            # Calculate match score
            match_score = self._calculate_match_score(setup, pattern['conditions'])

            if match_score >= 0.7:  # 70%+ match required
                # Calculate confidence score (combines match quality + pattern win rate + sample size)
                confidence_score = (
                    match_score * 0.4 +
                    pattern['win_rate'] * 0.4 +
                    min(pattern['sample_size'] / 50, 1.0) * 0.2
                )

                matches.append({
                    'pattern_id': pattern_id,
                    'pattern': pattern,
                    'match_score': match_score,
                    'confidence_score': confidence_score
                })

        return matches

    def _calculate_match_score(self, setup: Dict, pattern_conditions: Dict) -> float:
        """
        Calculate how well setup matches pattern conditions.

        Returns:
            Score from 0.0 to 1.0
        """
        total_conditions = len(pattern_conditions)
        if total_conditions == 0:
            return 0.0

        matches = 0

        for key, value in pattern_conditions.items():
            if setup.get(key) == value:
                matches += 1

        return matches / total_conditions

    def learn_from_trade(self, trade_result: Dict):
        """
        Learn from completed trade by adding to pattern library.

        Args:
            trade_result: Dictionary containing:
                - outcome: "WIN" or "LOSS"
                - r_multiple: R-multiple achieved
                - entry_conditions: Setup characteristics at entry
        """
        outcome = trade_result.get("outcome")
        r_multiple = trade_result.get("r_multiple", 0)
        entry_conditions = trade_result.get("entry_conditions", {})

        # Generate pattern ID from conditions
        pattern_id = self._generate_pattern_id(entry_conditions)

        # Update or create pattern
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = {
                'conditions': entry_conditions,
                'wins': 0,
                'losses': 0,
                'total_r': 0,
                'sample_size': 0,
                'win_rate': 0.0,
                'avg_r': 0.0,
            }

        pattern = self.patterns[pattern_id]

        # Update statistics
        pattern['sample_size'] += 1

        if outcome == "WIN":
            pattern['wins'] += 1
            pattern['total_r'] += r_multiple
        else:
            pattern['losses'] += 1

        # Recalculate metrics
        total_trades = pattern['wins'] + pattern['losses']
        pattern['win_rate'] = pattern['wins'] / total_trades if total_trades > 0 else 0
        pattern['avg_r'] = pattern['total_r'] / pattern['wins'] if pattern['wins'] > 0 else 0

        # If pattern reaches significance, share with other agents
        if pattern['sample_size'] == self.min_pattern_samples and pattern['win_rate'] >= 0.70:
            print(f"[PATTERN_DISCOVERED] {pattern_id}: {pattern['win_rate']*100:.1f}% WR, {pattern['avg_r']:.2f}R avg ({pattern['sample_size']} trades)")

    def _generate_pattern_id(self, conditions: Dict) -> str:
        """
        Generate unique pattern ID from conditions.

        Example: "EUR_USD_RSI_35-40_LONDON_BULLISH"
        """
        pair = conditions.get("pair", "")
        rsi = conditions.get("rsi_bucket", "")
        session = conditions.get("session", "")
        trend = conditions.get("trend", "")
        adx = conditions.get("adx_bucket", "")

        parts = [pair, f"RSI_{rsi}", session.upper(), trend.upper()]

        if adx == "strong":
            parts.append("STRONG_TREND")

        if conditions.get("volume_spike"):
            parts.append("VOL_SPIKE")

        return "_".join(parts)

    def get_top_patterns(self, n: int = 10) -> List[Dict]:
        """
        Get top N performing patterns.

        Args:
            n: Number of patterns to return

        Returns:
            List of top patterns sorted by performance
        """
        # Filter patterns with enough samples
        valid_patterns = [
            {
                'pattern_id': pid,
                **pattern
            }
            for pid, pattern in self.patterns.items()
            if pattern['sample_size'] >= self.min_pattern_samples
        ]

        # Sort by win rate, then by sample size
        sorted_patterns = sorted(
            valid_patterns,
            key=lambda p: (p['win_rate'], p['sample_size']),
            reverse=True
        )

        return sorted_patterns[:n]

    def save_patterns(self, filepath: str):
        """Save pattern library to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.patterns, f, indent=2)

    def load_patterns(self, filepath: str):
        """Load pattern library from JSON file."""
        try:
            with open(filepath, 'r') as f:
                self.patterns = json.load(f)
            print(f"[PatternRecognitionAgent] Loaded {len(self.patterns)} patterns")
        except FileNotFoundError:
            print(f"[PatternRecognitionAgent] No saved patterns found, starting fresh")
