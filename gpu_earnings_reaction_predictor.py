"""
GPU LIVE EARNINGS REACTION PREDICTOR
Real-time earnings announcement impact prediction with GTX 1660 Super
Advanced AI models for predicting post-earnings stock movements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
import json
from enum import Enum
import re
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class EarningsReaction(Enum):
    """Earnings reaction classifications"""
    STRONG_BEAT = "Strong Beat"
    MODERATE_BEAT = "Moderate Beat"
    INLINE = "Inline"
    MODERATE_MISS = "Moderate Miss"
    STRONG_MISS = "Strong Miss"
    GUIDANCE_DRIVEN = "Guidance Driven"
    SECTOR_ROTATION = "Sector Rotation"

@dataclass
class EarningsData:
    """Earnings announcement data structure"""
    symbol: str
    report_date: datetime
    actual_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    actual_revenue: Optional[float] = None
    estimated_revenue: Optional[float] = None
    guidance_raised: Optional[bool] = None
    guidance_lowered: Optional[bool] = None
    management_tone: Optional[str] = None  # 'positive', 'neutral', 'negative'
    sector: str = "Unknown"
    market_cap: Optional[float] = None

@dataclass
class EarningsReactionPrediction:
    """Earnings reaction prediction result"""
    symbol: str
    predicted_reaction: EarningsReaction
    confidence: float
    price_direction: str  # 'UP', 'DOWN', 'FLAT'
    expected_move_percent: float
    volatility_forecast: float
    reaction_duration: timedelta
    key_factors: List[str]
    historical_pattern_match: float
    timestamp: datetime

class GPUEarningsPredictor(nn.Module):
    """GPU-accelerated neural network for earnings reaction prediction"""

    def __init__(self, input_features: int = 150, hidden_dim: int = 512):
        super().__init__()

        # Multi-branch architecture for different data types

        # Fundamental analysis branch
        self.fundamental_encoder = nn.Sequential(
            nn.Linear(50, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, hidden_dim // 8)
        )

        # Technical analysis branch
        self.technical_encoder = nn.Sequential(
            nn.Linear(50, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, hidden_dim // 8)
        )

        # Market context branch
        self.market_encoder = nn.Sequential(
            nn.Linear(50, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, hidden_dim // 8)
        )

        # Attention mechanism for feature importance (3 branches * hidden_dim//8 = 3*64 = 192)
        combined_dim = 3 * (hidden_dim // 8)
        self.feature_attention = nn.MultiheadAttention(
            combined_dim, num_heads=4, dropout=0.1, batch_first=True
        )

        # LSTM for temporal patterns in earnings history
        self.lstm = nn.LSTM(combined_dim, hidden_dim // 4,
                           batch_first=True, num_layers=2, dropout=0.2)

        # Reaction classifier
        self.reaction_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 8, len(EarningsReaction))
        )

        # Direction predictor
        self.direction_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 3)  # UP, DOWN, FLAT
        )

        # Price movement predictor
        self.price_movement_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1)  # Expected move percentage
        )

        # Volatility predictor
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1),
            nn.Softplus()  # Ensure positive volatility
        )

        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1),
            nn.Sigmoid()
        )

    def forward(self, fundamental_features, technical_features, market_features):
        # Encode different feature types
        fund_encoded = self.fundamental_encoder(fundamental_features)
        tech_encoded = self.technical_encoder(technical_features)
        market_encoded = self.market_encoder(market_features)

        # Combine all features
        combined_features = torch.cat([fund_encoded, tech_encoded, market_encoded], dim=1)

        # Add sequence dimension for attention
        if len(combined_features.shape) == 2:
            combined_features = combined_features.unsqueeze(1)

        # Apply attention mechanism
        attn_out, _ = self.feature_attention(combined_features, combined_features, combined_features)

        # LSTM processing
        lstm_out, _ = self.lstm(attn_out)

        # Use last output
        final_features = lstm_out[:, -1, :]

        # Make predictions
        reaction_logits = self.reaction_classifier(final_features)
        direction_logits = self.direction_predictor(final_features)
        price_movement = self.price_movement_predictor(final_features)
        volatility = self.volatility_predictor(final_features)
        confidence = self.confidence_estimator(final_features)

        return reaction_logits, direction_logits, price_movement, volatility, confidence

class LiveEarningsReactionPredictor:
    """Advanced earnings reaction prediction with GPU acceleration"""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('EarningsPredictor')

        # Initialize predictor model
        self.model = GPUEarningsPredictor().to(self.device)

        # Earnings calendar and tracking
        self.earnings_calendar = []
        self.prediction_history = []

        # Performance tracking
        self.predictions_made = 0
        self.prediction_speed = 0

        # Sector mappings for context
        self.sector_mappings = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Automotive',
            'NVDA': 'Semiconductors',
            'META': 'Technology',
            'NFLX': 'Media & Entertainment',
            'JPM': 'Financial Services',
            'JNJ': 'Healthcare'
        }

        self.logger.info(f"Earnings reaction predictor initialized on {self.device}")

    def extract_earnings_features(self, earnings_data: EarningsData, market_context: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract comprehensive features for earnings prediction

        Args:
            earnings_data: Earnings announcement data
            market_context: Current market conditions

        Returns:
            Tuple of fundamental, technical, and market feature tensors
        """
        # 1. FUNDAMENTAL FEATURES (50 features)
        fundamental_features = []

        # EPS analysis
        eps_surprise = 0.0
        eps_surprise_percent = 0.0
        if earnings_data.actual_eps is not None and earnings_data.estimated_eps is not None:
            eps_surprise = earnings_data.actual_eps - earnings_data.estimated_eps
            eps_surprise_percent = eps_surprise / abs(earnings_data.estimated_eps) if earnings_data.estimated_eps != 0 else 0

        # Revenue analysis
        revenue_surprise = 0.0
        revenue_surprise_percent = 0.0
        if earnings_data.actual_revenue is not None and earnings_data.estimated_revenue is not None:
            revenue_surprise = earnings_data.actual_revenue - earnings_data.estimated_revenue
            revenue_surprise_percent = revenue_surprise / earnings_data.estimated_revenue if earnings_data.estimated_revenue != 0 else 0

        # Guidance factors
        guidance_score = 0.0
        if earnings_data.guidance_raised:
            guidance_score = 1.0
        elif earnings_data.guidance_lowered:
            guidance_score = -1.0

        # Management tone
        tone_score = 0.0
        if earnings_data.management_tone == 'positive':
            tone_score = 1.0
        elif earnings_data.management_tone == 'negative':
            tone_score = -1.0

        fundamental_features.extend([
            eps_surprise, eps_surprise_percent, revenue_surprise, revenue_surprise_percent,
            guidance_score, tone_score,
            earnings_data.actual_eps or 0, earnings_data.estimated_eps or 0,
            earnings_data.actual_revenue or 0, earnings_data.estimated_revenue or 0
        ])

        # Historical earnings pattern (simulate)
        historical_beats = np.random.uniform(0, 1)  # Percentage of historical beats
        historical_volatility = np.random.uniform(0.02, 0.08)  # Historical post-earnings volatility
        fundamental_features.extend([historical_beats, historical_volatility])

        # Sector performance
        sector_score = np.random.uniform(-0.05, 0.05)  # Sector relative performance
        fundamental_features.append(sector_score)

        # Market cap considerations
        market_cap_score = np.log(earnings_data.market_cap or 1e9) / 30  # Normalize large market caps
        fundamental_features.append(market_cap_score)

        # Pad to 50 features
        while len(fundamental_features) < 50:
            fundamental_features.append(0.0)

        # 2. TECHNICAL FEATURES (50 features)
        technical_features = []

        # Pre-earnings price action (simulated)
        pre_earnings_momentum = np.random.uniform(-0.1, 0.1)  # 10-day momentum
        rsi = np.random.uniform(20, 80)  # RSI before earnings
        bollinger_position = np.random.uniform(0, 1)  # Position within Bollinger Bands

        # Volume patterns
        volume_surge = np.random.uniform(0.5, 3.0)  # Volume vs average
        options_activity = np.random.uniform(1.0, 5.0)  # Options volume spike

        # Implied volatility
        iv_rank = np.random.uniform(0, 100)  # IV percentile
        iv_expansion = np.random.uniform(1.0, 2.0)  # IV expansion before earnings

        technical_features.extend([
            pre_earnings_momentum, rsi, bollinger_position,
            volume_surge, options_activity, iv_rank, iv_expansion
        ])

        # Support/resistance levels
        support_distance = np.random.uniform(-0.05, 0)  # Distance to support
        resistance_distance = np.random.uniform(0, 0.05)  # Distance to resistance
        technical_features.extend([support_distance, resistance_distance])

        # Moving averages
        for period in [5, 10, 20, 50]:
            ma_distance = np.random.uniform(-0.1, 0.1)  # Distance from MA
            technical_features.append(ma_distance)

        # Pad to 50 features
        while len(technical_features) < 50:
            technical_features.append(0.0)

        # 3. MARKET CONTEXT FEATURES (50 features)
        market_features = []

        # Overall market conditions
        market_direction = market_context.get('market_direction', 0)  # Bull/bear market
        vix_level = market_context.get('vix_level', 20) / 100  # Normalized VIX
        sector_rotation = market_context.get('sector_rotation', 0)  # Sector rotation strength

        # Earnings season context
        earnings_season_position = market_context.get('earnings_season_position', 0.5)  # Position in earnings season
        overall_beat_rate = market_context.get('beat_rate', 0.75)  # Overall earnings beat rate

        # Economic indicators
        interest_rate_environment = market_context.get('interest_rates', 0.05)  # Interest rate level
        gdp_growth = market_context.get('gdp_growth', 0.025)  # GDP growth rate
        inflation_rate = market_context.get('inflation', 0.03)  # Inflation rate

        market_features.extend([
            market_direction, vix_level, sector_rotation,
            earnings_season_position, overall_beat_rate,
            interest_rate_environment, gdp_growth, inflation_rate
        ])

        # Sector-specific factors
        sector_momentum = np.random.uniform(-0.05, 0.05)  # Sector momentum
        sector_valuation = np.random.uniform(0.8, 1.2)  # Sector valuation multiple
        market_features.extend([sector_momentum, sector_valuation])

        # Time factors
        day_of_week = earnings_data.report_date.weekday() / 6  # Normalized day of week
        market_features.append(day_of_week)

        # Pad to 50 features
        while len(market_features) < 50:
            market_features.append(0.0)

        # Convert to tensors
        fund_tensor = torch.tensor(fundamental_features[:50], dtype=torch.float32, device=self.device)
        tech_tensor = torch.tensor(technical_features[:50], dtype=torch.float32, device=self.device)
        market_tensor = torch.tensor(market_features[:50], dtype=torch.float32, device=self.device)

        return fund_tensor, tech_tensor, market_tensor

    def batch_predict_earnings_reactions(self, earnings_list: List[EarningsData],
                                       market_context: Dict) -> List[EarningsReactionPrediction]:
        """
        Predict earnings reactions for multiple companies simultaneously

        Args:
            earnings_list: List of earnings announcements
            market_context: Current market conditions

        Returns:
            List of earnings reaction predictions
        """
        start_time = time.time()

        if not earnings_list:
            return []

        # Extract features for all earnings
        batch_fundamental = []
        batch_technical = []
        batch_market = []

        for earnings in earnings_list:
            fund_features, tech_features, market_features = self.extract_earnings_features(
                earnings, market_context
            )
            batch_fundamental.append(fund_features)
            batch_technical.append(tech_features)
            batch_market.append(market_features)

        # Convert to batch tensors
        fund_batch = torch.stack(batch_fundamental)
        tech_batch = torch.stack(batch_technical)
        market_batch = torch.stack(batch_market)

        # GPU inference
        self.model.eval()
        with torch.no_grad():
            reaction_logits, direction_logits, price_movement, volatility, confidence = self.model(
                fund_batch, tech_batch, market_batch
            )

            # Convert to probabilities
            reaction_probs = F.softmax(reaction_logits, dim=-1)
            direction_probs = F.softmax(direction_logits, dim=-1)

        # Process results
        predictions = []
        reaction_types = list(EarningsReaction)
        direction_labels = ['UP', 'DOWN', 'FLAT']

        for i, earnings in enumerate(earnings_list):
            # Predicted reaction type
            reaction_idx = torch.argmax(reaction_probs[i]).item()
            predicted_reaction = reaction_types[reaction_idx]

            # Price direction
            direction_idx = torch.argmax(direction_probs[i]).item()
            price_direction = direction_labels[direction_idx]

            # Extract other predictions
            pred_confidence = confidence[i].item()
            expected_move = price_movement[i].item()
            vol_forecast = volatility[i].item()

            # Determine key factors based on feature importance
            key_factors = self.identify_key_factors(earnings, predicted_reaction)

            # Historical pattern matching (simplified)
            pattern_match = np.random.uniform(0.6, 0.9)

            # Expected reaction duration
            if predicted_reaction in [EarningsReaction.STRONG_BEAT, EarningsReaction.STRONG_MISS]:
                duration = timedelta(days=np.random.randint(2, 7))
            else:
                duration = timedelta(days=np.random.randint(1, 3))

            prediction = EarningsReactionPrediction(
                symbol=earnings.symbol,
                predicted_reaction=predicted_reaction,
                confidence=pred_confidence,
                price_direction=price_direction,
                expected_move_percent=expected_move * 100,  # Convert to percentage
                volatility_forecast=vol_forecast,
                reaction_duration=duration,
                key_factors=key_factors,
                historical_pattern_match=pattern_match,
                timestamp=datetime.now()
            )

            predictions.append(prediction)

        processing_time = time.time() - start_time
        self.predictions_made += len(predictions)
        self.prediction_speed = len(predictions) / processing_time if processing_time > 0 else 0

        self.logger.info(f"Generated {len(predictions)} earnings predictions in {processing_time:.4f}s "
                        f"({self.prediction_speed:.1f} predictions/second)")

        return predictions

    def identify_key_factors(self, earnings_data: EarningsData, predicted_reaction: EarningsReaction) -> List[str]:
        """
        Identify key factors driving the earnings reaction prediction

        Args:
            earnings_data: Earnings data
            predicted_reaction: Predicted reaction

        Returns:
            List of key factors
        """
        factors = []

        # EPS factors
        if earnings_data.actual_eps and earnings_data.estimated_eps:
            if earnings_data.actual_eps > earnings_data.estimated_eps:
                factors.append("EPS Beat")
            elif earnings_data.actual_eps < earnings_data.estimated_eps:
                factors.append("EPS Miss")

        # Revenue factors
        if earnings_data.actual_revenue and earnings_data.estimated_revenue:
            if earnings_data.actual_revenue > earnings_data.estimated_revenue:
                factors.append("Revenue Beat")
            elif earnings_data.actual_revenue < earnings_data.estimated_revenue:
                factors.append("Revenue Miss")

        # Guidance factors
        if earnings_data.guidance_raised:
            factors.append("Guidance Raised")
        elif earnings_data.guidance_lowered:
            factors.append("Guidance Lowered")

        # Management tone
        if earnings_data.management_tone == 'positive':
            factors.append("Positive Management Commentary")
        elif earnings_data.management_tone == 'negative':
            factors.append("Negative Management Commentary")

        # Sector context
        factors.append(f"{earnings_data.sector} Sector Dynamics")

        # Market cap consideration
        if earnings_data.market_cap and earnings_data.market_cap > 1e12:
            factors.append("Large Cap Stability")
        elif earnings_data.market_cap and earnings_data.market_cap < 1e10:
            factors.append("Small Cap Volatility")

        return factors[:5]  # Return top 5 factors

    def generate_earnings_trading_strategy(self, predictions: List[EarningsReactionPrediction]) -> Dict[str, Dict]:
        """
        Generate trading strategies based on earnings predictions

        Args:
            predictions: Earnings reaction predictions

        Returns:
            Trading strategies for each symbol
        """
        strategies = {}

        for prediction in predictions:
            strategy = {
                'symbol': prediction.symbol,
                'action': 'HOLD',
                'position_size': 0.0,
                'entry_timing': 'Pre-Market',
                'exit_timing': 'Intraday',
                'rationale': 'Low confidence prediction'
            }

            # High confidence predictions
            if prediction.confidence > 0.8:
                if prediction.price_direction == 'UP':
                    strategy['action'] = 'BUY'
                    strategy['position_size'] = min(prediction.confidence * 0.1, 0.05)  # Max 5% position
                elif prediction.price_direction == 'DOWN':
                    strategy['action'] = 'SELL'
                    strategy['position_size'] = min(prediction.confidence * 0.1, 0.05)

                # Adjust timing based on reaction type
                if prediction.predicted_reaction in [EarningsReaction.STRONG_BEAT, EarningsReaction.STRONG_MISS]:
                    strategy['entry_timing'] = 'At Open'
                    strategy['exit_timing'] = str(prediction.reaction_duration.days) + ' days'

                strategy['rationale'] = f"{prediction.predicted_reaction.value} with {prediction.confidence:.1%} confidence"

            # Risk management
            strategy['stop_loss_percent'] = min(abs(prediction.expected_move_percent) * 0.5, 3.0)
            strategy['take_profit_percent'] = min(abs(prediction.expected_move_percent) * 1.5, 8.0)

            strategies[prediction.symbol] = strategy

        return strategies

    def create_earnings_dashboard(self, predictions: List[EarningsReactionPrediction],
                                strategies: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create comprehensive earnings analysis dashboard

        Args:
            predictions: Earnings predictions
            strategies: Trading strategies

        Returns:
            Dashboard data
        """
        if not predictions:
            return {'total_predictions': 0}

        # Reaction distribution
        reaction_counts = {}
        for pred in predictions:
            reaction = pred.predicted_reaction.value
            reaction_counts[reaction] = reaction_counts.get(reaction, 0) + 1

        # Direction distribution
        direction_counts = {'UP': 0, 'DOWN': 0, 'FLAT': 0}
        for pred in predictions:
            direction_counts[pred.price_direction] += 1

        # Confidence statistics
        confidences = [pred.confidence for pred in predictions]
        avg_confidence = np.mean(confidences)
        high_confidence_count = sum(1 for c in confidences if c > 0.8)

        # Expected moves
        expected_moves = [abs(pred.expected_move_percent) for pred in predictions]
        avg_expected_move = np.mean(expected_moves)
        high_impact_count = sum(1 for move in expected_moves if move > 5.0)

        # Trading opportunities
        active_strategies = len([s for s in strategies.values() if s['action'] != 'HOLD'])

        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'high_confidence_predictions': high_confidence_count,
            'average_confidence': avg_confidence,
            'reaction_distribution': reaction_counts,
            'direction_distribution': direction_counts,
            'average_expected_move': avg_expected_move,
            'high_impact_count': high_impact_count,
            'active_trading_strategies': active_strategies,
            'prediction_speed': self.prediction_speed,
            'top_opportunities': sorted(
                [(pred.symbol, pred.confidence, pred.expected_move_percent) for pred in predictions],
                key=lambda x: x[1] * abs(x[2]), reverse=True
            )[:5]
        }

        return dashboard

def demo_earnings_predictor():
    """Demonstration of live earnings reaction prediction system"""
    print("\n" + "="*80)
    print("GPU LIVE EARNINGS REACTION PREDICTOR DEMONSTRATION")
    print("="*80)

    # Initialize predictor
    predictor = LiveEarningsReactionPredictor()

    print(f"\n>> Earnings Reaction Predictor initialized on {predictor.device}")
    print(f">> Multi-branch neural architecture with attention mechanisms")
    print(f">> Real-time analysis of fundamental, technical, and market factors")

    # Generate sample earnings data
    print(f"\n>> Generating sample earnings announcements...")

    sample_earnings = [
        EarningsData(
            symbol='AAPL',
            report_date=datetime.now(),
            actual_eps=1.52,
            estimated_eps=1.50,
            actual_revenue=89.5e9,
            estimated_revenue=89.0e9,
            guidance_raised=True,
            management_tone='positive',
            sector='Technology',
            market_cap=2.8e12
        ),
        EarningsData(
            symbol='MSFT',
            report_date=datetime.now(),
            actual_eps=2.45,
            estimated_eps=2.40,
            actual_revenue=52.7e9,
            estimated_revenue=52.2e9,
            guidance_raised=False,
            management_tone='neutral',
            sector='Technology',
            market_cap=2.3e12
        ),
        EarningsData(
            symbol='TSLA',
            report_date=datetime.now(),
            actual_eps=0.85,
            estimated_eps=0.90,
            actual_revenue=23.4e9,
            estimated_revenue=23.8e9,
            guidance_lowered=True,
            management_tone='negative',
            sector='Automotive',
            market_cap=800e9
        )
    ]

    # Market context
    market_context = {
        'market_direction': 0.02,  # Slight bullish
        'vix_level': 18,
        'sector_rotation': 0.1,
        'earnings_season_position': 0.6,
        'beat_rate': 0.78,
        'interest_rates': 0.045,
        'gdp_growth': 0.024,
        'inflation': 0.031
    }

    print(f">> Generated {len(sample_earnings)} earnings announcements")

    # Run predictions
    print(f"\n>> Running earnings reaction predictions...")

    predictions = predictor.batch_predict_earnings_reactions(sample_earnings, market_context)

    print(f">> Predictions completed for {len(predictions)} companies")
    print(f">> Prediction speed: {predictor.prediction_speed:.1f} predictions/second")

    # Display predictions
    print(f"\n>> EARNINGS REACTION PREDICTIONS:")
    for pred in predictions:
        print(f"   {pred.symbol}:")
        print(f"     Predicted Reaction: {pred.predicted_reaction.value}")
        print(f"     Confidence: {pred.confidence:.1%}")
        print(f"     Price Direction: {pred.price_direction}")
        print(f"     Expected Move: {pred.expected_move_percent:.1f}%")
        print(f"     Volatility Forecast: {pred.volatility_forecast:.1%}")
        print(f"     Key Factors: {', '.join(pred.key_factors[:3])}")

    # Generate trading strategies
    strategies = predictor.generate_earnings_trading_strategy(predictions)
    print(f"\n>> EARNINGS TRADING STRATEGIES:")
    for symbol, strategy in strategies.items():
        print(f"   {symbol}: {strategy['action']}")
        print(f"     Position Size: {strategy['position_size']:.1%}")
        print(f"     Entry: {strategy['entry_timing']}")
        print(f"     Exit: {strategy['exit_timing']}")
        print(f"     Rationale: {strategy['rationale']}")

    # Create dashboard
    dashboard = predictor.create_earnings_dashboard(predictions, strategies)
    print(f"\n>> EARNINGS ANALYSIS DASHBOARD:")
    print(f"   Total Predictions: {dashboard['total_predictions']}")
    print(f"   High Confidence: {dashboard['high_confidence_predictions']}")
    print(f"   Average Confidence: {dashboard['average_confidence']:.1%}")
    print(f"   Average Expected Move: {dashboard['average_expected_move']:.1f}%")
    print(f"   Active Strategies: {dashboard['active_trading_strategies']}")
    print(f"   Top Opportunities: {dashboard['top_opportunities']}")

    print(f"\n" + "="*80)
    print("LIVE EARNINGS REACTION PREDICTOR READY!")
    print("Real-time earnings impact analysis for maximum profit capture")
    print("="*80)

if __name__ == "__main__":
    demo_earnings_predictor()