#!/usr/bin/env python3
"""
Multi-Strategy Ensemble Voting System
Combines multiple strategies and uses majority voting for final decision
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List
import logging
import yfinance as yf

# Import all enhancement modules
from enhancements.earnings_calendar import get_earnings_calendar
from enhancements.multi_timeframe import get_mtf_analyzer
from enhancements.price_patterns import get_pattern_detector

logger = logging.getLogger(__name__)

class EnsembleVotingSystem:
    """
    Combines multiple strategies for robust trade decisions

    Strategies:
    1. ML Models (RandomForest + XGBoost)
    2. Multi-Timeframe Analysis
    3. Price Action Patterns
    4. Earnings Calendar Check
    5. Mean Reversion
    6. Momentum
    """

    def __init__(self):
        self.earnings_calendar = get_earnings_calendar()
        self.mtf_analyzer = get_mtf_analyzer()
        self.pattern_detector = get_pattern_detector()

        # Strategy weights (must sum to 1.0)
        self.weights = {
            'ml_models': 0.35,          # ML predictions
            'multi_timeframe': 0.25,    # Timeframe alignment
            'price_patterns': 0.15,     # Candlestick patterns
            'momentum': 0.15,           # Price momentum
            'mean_reversion': 0.10      # Reversion signals
        }

    def calculate_momentum_signal(self, symbol: str) -> Dict:
        """
        Momentum strategy
        Buy if strong uptrend, sell if strong downtrend
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1mo', interval='1d')

            if data.empty or len(data) < 10:
                return {'signal': 0, 'confidence': 0, 'reason': 'Insufficient data'}

            # Calculate momentum indicators
            close = data['Close']
            sma_20 = close.rolling(20).mean()
            returns_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if len(close) >= 6 else 0
            returns_10d = (close.iloc[-1] - close.iloc[-11]) / close.iloc[-11] if len(close) >= 11 else 0

            price = close.iloc[-1]
            sma = sma_20.iloc[-1] if not sma_20.empty else price

            # Momentum signals
            above_sma = price > sma
            strong_5d = returns_5d > 0.03  # 3% gain in 5 days
            strong_10d = returns_10d > 0.05  # 5% gain in 10 days

            if above_sma and strong_5d and strong_10d:
                signal = 1  # Bullish
                confidence = 0.8
                reason = f"Strong momentum: {returns_5d:.1%} (5d), {returns_10d:.1%} (10d)"
            elif above_sma and (strong_5d or strong_10d):
                signal = 1
                confidence = 0.6
                reason = "Moderate bullish momentum"
            elif not above_sma and returns_5d < -0.03 and returns_10d < -0.05:
                signal = -1  # Bearish
                confidence = 0.8
                reason = f"Strong downward momentum: {returns_5d:.1%} (5d), {returns_10d:.1%} (10d)"
            elif not above_sma and (returns_5d < -0.03 or returns_10d < -0.05):
                signal = -1
                confidence = 0.6
                reason = "Moderate bearish momentum"
            else:
                signal = 0
                confidence = 0.3
                reason = "Weak momentum"

            return {
                'signal': signal,
                'confidence': float(confidence),
                'reason': reason,
                'returns_5d': float(returns_5d),
                'returns_10d': float(returns_10d)
            }

        except Exception as e:
            logger.error(f"Momentum calculation error for {symbol}: {e}")
            return {'signal': 0, 'confidence': 0, 'reason': f'Error: {e}'}

    def calculate_mean_reversion_signal(self, symbol: str) -> Dict:
        """
        Mean Reversion strategy
        Buy when oversold, sell when overbought
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1mo', interval='1d')

            if data.empty or len(data) < 20:
                return {'signal': 0, 'confidence': 0, 'reason': 'Insufficient data'}

            close = data['Close']

            # Bollinger Bands
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            bb_upper = sma_20 + (2 * std_20)
            bb_lower = sma_20 - (2 * std_20)

            price = close.iloc[-1]
            upper = bb_upper.iloc[-1]
            lower = bb_lower.iloc[-1]
            middle = sma_20.iloc[-1]

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # Mean reversion signals
            if price < lower and current_rsi < 30:
                signal = 1  # Oversold - buy
                confidence = 0.8
                reason = f"Oversold: Below BB lower band, RSI {current_rsi:.0f}"
            elif price < middle and current_rsi < 40:
                signal = 1
                confidence = 0.5
                reason = f"Moderately oversold: RSI {current_rsi:.0f}"
            elif price > upper and current_rsi > 70:
                signal = -1  # Overbought - sell
                confidence = 0.8
                reason = f"Overbought: Above BB upper band, RSI {current_rsi:.0f}"
            elif price > middle and current_rsi > 60:
                signal = -1
                confidence = 0.5
                reason = f"Moderately overbought: RSI {current_rsi:.0f}"
            else:
                signal = 0
                confidence = 0.3
                reason = f"Neutral: RSI {current_rsi:.0f}"

            return {
                'signal': signal,
                'confidence': float(confidence),
                'reason': reason,
                'rsi': float(current_rsi)
            }

        except Exception as e:
            logger.error(f"Mean reversion calculation error for {symbol}: {e}")
            return {'signal': 0, 'confidence': 0, 'reason': f'Error: {e}'}

    def get_ensemble_vote(self, symbol: str, ml_signal: Dict, trade_direction: str = None) -> Dict:
        """
        Get ensemble vote from all strategies

        Args:
            symbol: Stock symbol
            ml_signal: ML model prediction {'prediction': 0/1, 'confidence': float}
            trade_direction: Optional 'CALL' or 'PUT' override

        Returns:
            {
                'final_decision': str ('BUY', 'SELL', 'HOLD'),
                'confidence': float (0-1),
                'votes': {...},
                'filters': {...},
                'reasoning': List[str]
            }
        """
        votes = {}
        reasoning = []
        filters = {}

        # FILTER 1: Earnings Calendar (VETO POWER)
        earnings_check = self.earnings_calendar.is_safe_to_trade(symbol)
        filters['earnings'] = earnings_check

        if not earnings_check['safe']:
            return {
                'final_decision': 'REJECT',
                'confidence': 0,
                'votes': {},
                'filters': filters,
                'reasoning': [f"REJECTED: {earnings_check['reason']}"]
            }

        reasoning.append(f"Earnings check: {earnings_check['reason']}")

        # VOTE 1: ML Models
        if ml_signal and ml_signal.get('prediction') is not None:
            ml_vote = 1 if ml_signal['prediction'] == 1 else -1
            ml_conf = ml_signal.get('confidence', 0.5)

            votes['ml_models'] = {
                'vote': ml_vote,
                'confidence': ml_conf,
                'weight': self.weights['ml_models']
            }
            reasoning.append(f"ML Models: {'BUY' if ml_vote > 0 else 'SELL'} ({ml_conf:.0%} confidence)")

        # VOTE 2: Multi-Timeframe Analysis
        mtf_analysis = self.mtf_analyzer.analyze_all_timeframes(symbol)
        filters['multi_timeframe'] = mtf_analysis

        if mtf_analysis['score'] != 0:
            mtf_vote = 1 if mtf_analysis['score'] > 0 else -1
            mtf_conf = abs(mtf_analysis['score'])

            votes['multi_timeframe'] = {
                'vote': mtf_vote,
                'confidence': mtf_conf,
                'weight': self.weights['multi_timeframe']
            }
            reasoning.append(f"Multi-Timeframe: {mtf_analysis['consensus']} (score: {mtf_analysis['score']:.2f})")

        # VOTE 3: Price Patterns
        ticker = yf.Ticker(symbol)
        price_data = ticker.history(period='5d', interval='1d')
        pattern_analysis = self.pattern_detector.analyze_patterns(price_data)
        filters['price_patterns'] = pattern_analysis

        if pattern_analysis['signal'] != 'NEUTRAL':
            pattern_vote = 1 if pattern_analysis['signal'] == 'BULLISH' else -1
            pattern_conf = pattern_analysis['strength']

            votes['price_patterns'] = {
                'vote': pattern_vote,
                'confidence': pattern_conf,
                'weight': self.weights['price_patterns']
            }
            reasoning.append(f"Price Patterns: {pattern_analysis['description']}")

        # VOTE 4: Momentum
        momentum = self.calculate_momentum_signal(symbol)
        if momentum['signal'] != 0:
            votes['momentum'] = {
                'vote': momentum['signal'],
                'confidence': momentum['confidence'],
                'weight': self.weights['momentum']
            }
            reasoning.append(f"Momentum: {momentum['reason']}")

        # VOTE 5: Mean Reversion
        mean_rev = self.calculate_mean_reversion_signal(symbol)
        if mean_rev['signal'] != 0:
            votes['mean_reversion'] = {
                'vote': mean_rev['signal'],
                'confidence': mean_rev['confidence'],
                'weight': self.weights['mean_reversion']
            }
            reasoning.append(f"Mean Reversion: {mean_rev['reason']}")

        # WEIGHTED VOTING
        weighted_score = 0
        total_weight = 0

        for strategy, vote_data in votes.items():
            weighted_score += vote_data['vote'] * vote_data['confidence'] * vote_data['weight']
            total_weight += vote_data['weight']

        if total_weight == 0:
            return {
                'final_decision': 'HOLD',
                'confidence': 0,
                'votes': votes,
                'filters': filters,
                'reasoning': ['No strategies provided signals']
            }

        # Normalize
        final_score = weighted_score / total_weight

        # Decision thresholds
        if final_score > 0.3:
            decision = 'BUY'
            confidence = min(abs(final_score), 1.0)
        elif final_score < -0.3:
            decision = 'SELL'
            confidence = min(abs(final_score), 1.0)
        else:
            decision = 'HOLD'
            confidence = 0.5 - abs(final_score)

        # Check alignment
        buy_votes = sum(1 for v in votes.values() if v['vote'] > 0)
        sell_votes = sum(1 for v in votes.values() if v['vote'] < 0)
        total_votes = len(votes)

        if total_votes >= 3:
            if buy_votes / total_votes >= 0.7:
                reasoning.append(f"STRONG CONSENSUS: {buy_votes}/{total_votes} strategies agree on BUY")
                confidence *= 1.2
            elif sell_votes / total_votes >= 0.7:
                reasoning.append(f"STRONG CONSENSUS: {sell_votes}/{total_votes} strategies agree on SELL")
                confidence *= 1.2

        confidence = min(confidence, 1.0)

        return {
            'final_decision': decision,
            'confidence': float(confidence),
            'votes': votes,
            'filters': filters,
            'reasoning': reasoning,
            'weighted_score': float(final_score),
            'vote_count': total_votes
        }


# Global instance
_ensemble_system = None

def get_ensemble_system() -> EnsembleVotingSystem:
    """Get singleton ensemble voting system"""
    global _ensemble_system
    if _ensemble_system is None:
        _ensemble_system = EnsembleVotingSystem()
    return _ensemble_system


if __name__ == "__main__":
    # Test
    ensemble = EnsembleVotingSystem()

    symbol = 'AAPL'
    ml_signal = {'prediction': 1, 'confidence': 0.7}

    print(f"ENSEMBLE VOTING SYSTEM TEST: {symbol}")
    print("="*70)

    result = ensemble.get_ensemble_vote(symbol, ml_signal)

    print(f"\nFinal Decision: {result['final_decision']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Weighted Score: {result['weighted_score']:.3f}")
    print(f"Vote Count: {result['vote_count']}")

    print(f"\nStrategy Votes:")
    for strategy, vote_data in result['votes'].items():
        vote_str = "BUY" if vote_data['vote'] > 0 else "SELL"
        print(f"  {strategy}: {vote_str} (confidence: {vote_data['confidence']:.0%}, weight: {vote_data['weight']:.0%})")

    print(f"\nReasoning:")
    for i, reason in enumerate(result['reasoning'], 1):
        print(f"  {i}. {reason}")
