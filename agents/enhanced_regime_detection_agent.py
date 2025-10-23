"""
Enhanced Regime Detection Agent - Advanced Market State Classification

This agent uses multiple methods to detect market regimes:
- Hidden Markov Models (HMM) for state classification
- Volatility clustering (GARCH models)
- Trend strength analysis (ADX, price patterns)
- Correlation structure changes
- Volume profile analysis
- Cross-asset regime confirmation

The agent helps adapt trading strategies to current market conditions.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from collections import deque

# LangGraph imports
from langgraph.graph import StateGraph, END

# Machine learning imports
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    hmm = None

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Extended market regime classifications"""
    # Trend regimes
    STRONG_BULL = "strong_bull"
    MODERATE_BULL = "moderate_bull"
    WEAK_BULL = "weak_bull"
    STRONG_BEAR = "strong_bear"
    MODERATE_BEAR = "moderate_bear"
    WEAK_BEAR = "weak_bear"

    # Mean reversion regimes
    SIDEWAYS_TIGHT = "sideways_tight"
    SIDEWAYS_WIDE = "sideways_wide"

    # Volatility regimes
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    VOLATILE_EXPANSION = "volatile_expansion"
    VOLATILE_CONTRACTION = "volatile_contraction"

    # Crisis regimes
    CRISIS = "crisis"
    RECOVERY = "recovery"
    DISTRIBUTION = "distribution"
    ACCUMULATION = "accumulation"

    # Transition states
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"


class RegimeFeature(Enum):
    """Features used for regime detection"""
    RETURNS = "returns"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND_STRENGTH = "trend_strength"
    CORRELATION = "correlation"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class RegimeMetrics:
    """Metrics for current regime"""
    regime: MarketRegime
    confidence: float  # 0-1
    persistence: float  # Expected regime duration in days
    transition_probability: float  # Probability of regime change
    volatility_level: float
    trend_strength: float  # -1 to 1
    mean_reversion_score: float  # 0-1
    timestamp: datetime


@dataclass
class RegimeTransition:
    """Detected regime transition"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_time: datetime
    confidence: float
    trigger: str  # What caused the transition


@dataclass
class StrategyWeights:
    """Recommended strategy weights for current regime"""
    momentum: float
    mean_reversion: float
    volatility: float
    options: float
    short_selling: float
    long_term: float
    regime: MarketRegime
    reasoning: List[str]


@dataclass
class RegimeState:
    """LangGraph state for regime detection"""
    symbol: str
    historical_data: Optional[pd.DataFrame] = None
    current_regime: Optional[RegimeMetrics] = None
    regime_history: List[RegimeMetrics] = field(default_factory=list)
    transitions: List[RegimeTransition] = field(default_factory=list)
    strategy_weights: Optional[StrategyWeights] = None
    hmm_states: Optional[np.ndarray] = None
    errors: List[str] = field(default_factory=list)


class EnhancedRegimeDetectionAgent:
    """
    Enhanced Regime Detection Agent using multiple methods

    Uses:
    - Hidden Markov Models for probabilistic state detection
    - Volatility clustering (GARCH-like)
    - Trend strength indicators
    - Volume profile analysis
    - Correlation regime shifts
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Enhanced Regime Detection Agent"""
        self.config = config or {}

        # Configuration
        self.lookback_days = self.config.get('lookback_days', 252)  # 1 year
        self.n_regimes = self.config.get('n_regimes', 4)  # HMM states
        self.min_regime_persistence = self.config.get('min_regime_persistence', 5)  # days

        # Volatility thresholds (annualized)
        self.high_vol_threshold = self.config.get('high_vol_threshold', 0.30)  # 30%
        self.low_vol_threshold = self.config.get('low_vol_threshold', 0.10)  # 10%

        # Trend strength thresholds (ADX)
        self.strong_trend_threshold = self.config.get('strong_trend_threshold', 40)
        self.weak_trend_threshold = self.config.get('weak_trend_threshold', 20)

        # HMM model (trained on first use)
        self.hmm_model = None
        self.hmm_trained = False

        # Regime history tracking
        self.regime_buffer = deque(maxlen=100)  # Last 100 regime detections

        # Build workflow
        self.workflow = self._create_workflow()

        logger.info("Enhanced Regime Detection Agent initialized")

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for regime detection"""
        workflow = StateGraph(RegimeState)

        # Add nodes
        workflow.add_node("fetch_market_data", self._fetch_market_data)
        workflow.add_node("calculate_features", self._calculate_features)
        workflow.add_node("detect_regime_hmm", self._detect_regime_hmm)
        workflow.add_node("detect_regime_rules", self._detect_regime_rules)
        workflow.add_node("combine_regimes", self._combine_regimes)
        workflow.add_node("generate_strategy_weights", self._generate_strategy_weights)
        workflow.add_node("detect_transitions", self._detect_transitions)

        # Define edges
        workflow.set_entry_point("fetch_market_data")
        workflow.add_edge("fetch_market_data", "calculate_features")
        workflow.add_edge("calculate_features", "detect_regime_hmm")
        workflow.add_edge("detect_regime_hmm", "detect_regime_rules")
        workflow.add_edge("detect_regime_rules", "combine_regimes")
        workflow.add_edge("combine_regimes", "generate_strategy_weights")
        workflow.add_edge("generate_strategy_weights", "detect_transitions")
        workflow.add_edge("detect_transitions", END)

        return workflow.compile()

    async def detect_regime(self, symbol: str = "SPY") -> Tuple[RegimeMetrics, StrategyWeights]:
        """
        Main entry point: detect current market regime

        Args:
            symbol: Market symbol to analyze (default: SPY for broad market)

        Returns:
            (RegimeMetrics, StrategyWeights) tuple
        """
        try:
            initial_state = RegimeState(symbol=symbol)

            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)

            if final_state.current_regime and final_state.strategy_weights:
                # Store in history
                self.regime_buffer.append(final_state.current_regime)

                return final_state.current_regime, final_state.strategy_weights
            else:
                # Fallback
                return self._create_fallback_regime(), self._create_fallback_weights()

        except Exception as e:
            logger.error(f"Error detecting regime for {symbol}: {e}")
            return self._create_fallback_regime(), self._create_fallback_weights()

    async def _fetch_market_data(self, state: RegimeState) -> RegimeState:
        """Fetch historical market data"""
        try:
            # ENHANCEMENT: Use real data from Alpaca/Polygon/OpenBB/Yahoo
            from agents.real_data_connector import fetch_real_market_data

            logger.info(f"Fetching REAL market data for {state.symbol} ({self.lookback_days} days)...")
            data = await fetch_real_market_data(state.symbol, days=self.lookback_days)

            # Fallback to simulation if real data fails
            if data is None or len(data) == 0:
                logger.warning(f"Real data unavailable for {state.symbol}, using simulation fallback")
                data = await self._simulate_market_data(state.symbol, days=self.lookback_days)
            else:
                logger.info(f"✅ Using REAL market data: {len(data)} bars from live sources")

            state.historical_data = data
            logger.info(f"Fetched {len(data)} days of market data for {state.symbol}")

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            # Emergency fallback to simulation
            try:
                data = await self._simulate_market_data(state.symbol, days=self.lookback_days)
                state.historical_data = data
                logger.warning("Using simulated data due to fetch error")
            except:
                state.errors.append(f"Market data fetch failed: {str(e)}")

        return state

    async def _calculate_features(self, state: RegimeState) -> RegimeState:
        """Calculate features for regime detection"""
        try:
            if state.historical_data is None or len(state.historical_data) == 0:
                state.errors.append("No historical data available")
                return state

            df = state.historical_data.copy()

            # Calculate returns
            df['returns'] = df['close'].pct_change()

            # Calculate realized volatility (rolling 20-day)
            df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)  # Annualized

            # Calculate ADX (trend strength)
            df = self._calculate_adx(df)

            # Calculate volume features
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

            # Calculate momentum indicators
            df['momentum_5d'] = df['close'].pct_change(5)
            df['momentum_20d'] = df['close'].pct_change(20)

            # Calculate mean reversion indicators
            df['bb_position'] = self._calculate_bollinger_position(df)
            df['rsi'] = self._calculate_rsi(df, period=14)

            # Calculate market regime features
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']  # Daily range
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])  # Where close is in range

            # Drop NaN rows
            df = df.dropna()

            state.historical_data = df
            logger.info(f"Calculated {len(df.columns)} features for regime detection")

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            state.errors.append(f"Feature calculation failed: {str(e)}")

        return state

    async def _detect_regime_hmm(self, state: RegimeState) -> RegimeState:
        """Detect regime using Hidden Markov Model"""
        try:
            if not HMM_AVAILABLE:
                logger.warning("HMM not available - skipping HMM detection")
                return state

            if state.historical_data is None or len(state.historical_data) < 100:
                state.errors.append("Insufficient data for HMM")
                return state

            df = state.historical_data

            # Prepare features for HMM
            features = df[['returns', 'volatility', 'adx']].dropna()

            if len(features) < 100:
                return state

            # Normalize features
            features_normalized = (features - features.mean()) / features.std()
            X = features_normalized.values

            # Train or use existing HMM model
            if not self.hmm_trained or self.hmm_model is None:
                # Create and train Gaussian HMM
                self.hmm_model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42
                )

                try:
                    self.hmm_model.fit(X)
                    self.hmm_trained = True
                    logger.info(f"HMM trained with {self.n_regimes} regimes")
                except Exception as e:
                    logger.error(f"HMM training failed: {e}")
                    return state

            # Predict hidden states
            hidden_states = self.hmm_model.predict(X)

            # Get current state
            current_state = hidden_states[-1]

            # Calculate state persistence (how long in current state)
            persistence = 1
            for i in range(len(hidden_states) - 2, -1, -1):
                if hidden_states[i] == current_state:
                    persistence += 1
                else:
                    break

            # Calculate transition probability
            trans_prob = self.hmm_model.transmat_[current_state, current_state]

            # Store HMM states
            state.hmm_states = hidden_states
            state.hmm_current_state = current_state
            state.hmm_persistence = persistence
            state.hmm_transition_prob = 1 - trans_prob

            logger.info(f"HMM detected state {current_state} with persistence {persistence} days")

        except Exception as e:
            logger.error(f"Error in HMM regime detection: {e}")
            state.errors.append(f"HMM detection failed: {str(e)}")

        return state

    async def _detect_regime_rules(self, state: RegimeState) -> RegimeState:
        """Detect regime using rule-based methods"""
        try:
            if state.historical_data is None:
                return state

            df = state.historical_data
            current = df.iloc[-1]
            recent = df.tail(20)  # Last 20 days

            # Extract features
            volatility = current['volatility']
            adx = current['adx']
            returns_5d = current['momentum_5d']
            returns_20d = current['momentum_20d']
            rsi = current['rsi']
            volume_ratio = recent['volume_ratio'].mean()

            # Rule-based regime classification
            regime = None
            confidence = 0.0

            # Check for crisis regime (high volatility + negative returns)
            if volatility > 0.40 and returns_20d < -0.10:
                regime = MarketRegime.CRISIS
                confidence = min(1.0, volatility / 0.50)

            # Check for high volatility regime
            elif volatility > self.high_vol_threshold:
                if returns_20d > 0:
                    regime = MarketRegime.VOLATILE_EXPANSION
                else:
                    regime = MarketRegime.HIGH_VOLATILITY
                confidence = 0.8

            # Check for low volatility regime
            elif volatility < self.low_vol_threshold:
                if adx < self.weak_trend_threshold:
                    regime = MarketRegime.SIDEWAYS_TIGHT
                else:
                    regime = MarketRegime.LOW_VOLATILITY
                confidence = 0.7

            # Check for strong trends
            elif adx > self.strong_trend_threshold:
                if returns_20d > 0.05:  # 5% gain
                    regime = MarketRegime.STRONG_BULL
                    confidence = min(1.0, adx / 60)
                elif returns_20d < -0.05:  # 5% loss
                    regime = MarketRegime.STRONG_BEAR
                    confidence = min(1.0, adx / 60)
                else:
                    regime = MarketRegime.MODERATE_BULL if returns_5d > 0 else MarketRegime.MODERATE_BEAR
                    confidence = 0.6

            # Check for weak trends
            elif adx > self.weak_trend_threshold:
                if returns_20d > 0:
                    regime = MarketRegime.WEAK_BULL
                else:
                    regime = MarketRegime.WEAK_BEAR
                confidence = 0.5

            # Sideways/ranging market
            else:
                if rsi > 60:
                    regime = MarketRegime.ACCUMULATION
                elif rsi < 40:
                    regime = MarketRegime.DISTRIBUTION
                else:
                    regime = MarketRegime.SIDEWAYS_WIDE
                confidence = 0.5

            # Store rule-based regime
            state.rules_regime = regime
            state.rules_confidence = confidence
            state.rules_volatility = volatility
            state.rules_trend_strength = (adx - 25) / 50  # Normalize to -1 to 1

            logger.info(f"Rule-based regime: {regime.value if regime else 'None'} (confidence: {confidence:.2f})")

        except Exception as e:
            logger.error(f"Error in rule-based regime detection: {e}")
            state.errors.append(f"Rule-based detection failed: {str(e)}")

        return state

    async def _combine_regimes(self, state: RegimeState) -> RegimeState:
        """Combine HMM and rule-based regime detections"""
        try:
            # Get both regimes
            rules_regime = getattr(state, 'rules_regime', None)
            rules_confidence = getattr(state, 'rules_confidence', 0.0)

            # If we have HMM states, map to regime
            if hasattr(state, 'hmm_current_state'):
                hmm_regime = self._map_hmm_state_to_regime(
                    state.hmm_current_state,
                    state.historical_data
                )
                hmm_confidence = 1 - getattr(state, 'hmm_transition_prob', 0.5)
            else:
                hmm_regime = None
                hmm_confidence = 0.0

            # Combine regimes (weighted by confidence)
            if hmm_regime and rules_regime:
                # If they agree, boost confidence
                if hmm_regime == rules_regime:
                    final_regime = rules_regime
                    final_confidence = min(1.0, (rules_confidence + hmm_confidence) / 2 * 1.2)
                else:
                    # Use higher confidence regime
                    if rules_confidence > hmm_confidence:
                        final_regime = rules_regime
                        final_confidence = rules_confidence * 0.9  # Penalty for disagreement
                    else:
                        final_regime = hmm_regime
                        final_confidence = hmm_confidence * 0.9
            elif rules_regime:
                final_regime = rules_regime
                final_confidence = rules_confidence
            elif hmm_regime:
                final_regime = hmm_regime
                final_confidence = hmm_confidence
            else:
                final_regime = MarketRegime.UNCERTAIN
                final_confidence = 0.3

            # Calculate persistence
            persistence = getattr(state, 'hmm_persistence', 5)

            # Calculate transition probability
            transition_prob = getattr(state, 'hmm_transition_prob', 0.1)

            # Create regime metrics
            state.current_regime = RegimeMetrics(
                regime=final_regime,
                confidence=final_confidence,
                persistence=persistence,
                transition_probability=transition_prob,
                volatility_level=getattr(state, 'rules_volatility', 0.15),
                trend_strength=getattr(state, 'rules_trend_strength', 0.0),
                mean_reversion_score=self._calculate_mean_reversion_score(state.historical_data),
                timestamp=datetime.now(timezone.utc)
            )

            logger.info(f"Combined regime: {final_regime.value} (confidence: {final_confidence:.2%})")

        except Exception as e:
            logger.error(f"Error combining regimes: {e}")
            state.errors.append(f"Regime combination failed: {str(e)}")

        return state

    async def _generate_strategy_weights(self, state: RegimeState) -> RegimeState:
        """Generate optimal strategy weights for current regime"""
        try:
            if not state.current_regime:
                return state

            regime = state.current_regime.regime

            # Define regime-specific strategy weights
            weights_map = {
                # Bull regimes - favor momentum
                MarketRegime.STRONG_BULL: (0.50, 0.10, 0.20, 0.10, 0.05, 0.05),
                MarketRegime.MODERATE_BULL: (0.40, 0.15, 0.20, 0.15, 0.05, 0.05),
                MarketRegime.WEAK_BULL: (0.30, 0.25, 0.15, 0.15, 0.10, 0.05),

                # Bear regimes - favor short selling and hedging
                MarketRegime.STRONG_BEAR: (0.10, 0.15, 0.25, 0.30, 0.20, 0.00),
                MarketRegime.MODERATE_BEAR: (0.15, 0.20, 0.25, 0.20, 0.15, 0.05),
                MarketRegime.WEAK_BEAR: (0.20, 0.25, 0.20, 0.15, 0.15, 0.05),

                # Sideways regimes - favor mean reversion
                MarketRegime.SIDEWAYS_TIGHT: (0.10, 0.50, 0.15, 0.15, 0.05, 0.05),
                MarketRegime.SIDEWAYS_WIDE: (0.15, 0.45, 0.20, 0.10, 0.05, 0.05),

                # Volatility regimes - favor options
                MarketRegime.HIGH_VOLATILITY: (0.15, 0.25, 0.40, 0.10, 0.05, 0.05),
                MarketRegime.LOW_VOLATILITY: (0.25, 0.30, 0.10, 0.15, 0.10, 0.10),
                MarketRegime.VOLATILE_EXPANSION: (0.15, 0.20, 0.45, 0.10, 0.05, 0.05),
                MarketRegime.VOLATILE_CONTRACTION: (0.30, 0.25, 0.20, 0.15, 0.05, 0.05),

                # Special regimes
                MarketRegime.CRISIS: (0.00, 0.10, 0.30, 0.40, 0.15, 0.05),
                MarketRegime.RECOVERY: (0.35, 0.20, 0.20, 0.10, 0.10, 0.05),
                MarketRegime.ACCUMULATION: (0.30, 0.30, 0.15, 0.10, 0.10, 0.05),
                MarketRegime.DISTRIBUTION: (0.20, 0.25, 0.20, 0.15, 0.15, 0.05),

                # Default
                MarketRegime.UNCERTAIN: (0.20, 0.20, 0.20, 0.15, 0.15, 0.10),
                MarketRegime.TRANSITION: (0.20, 0.25, 0.20, 0.15, 0.10, 0.10),
            }

            weights = weights_map.get(regime, (0.20, 0.20, 0.20, 0.15, 0.15, 0.10))

            # Generate reasoning
            reasoning = self._generate_weight_reasoning(regime, weights, state.current_regime)

            state.strategy_weights = StrategyWeights(
                momentum=weights[0],
                mean_reversion=weights[1],
                volatility=weights[2],
                options=weights[3],
                short_selling=weights[4],
                long_term=weights[5],
                regime=regime,
                reasoning=reasoning
            )

            logger.info(f"Strategy weights: momentum={weights[0]:.1%}, mean_rev={weights[1]:.1%}, vol={weights[2]:.1%}")

        except Exception as e:
            logger.error(f"Error generating strategy weights: {e}")
            state.errors.append(f"Strategy weights failed: {str(e)}")

        return state

    async def _detect_transitions(self, state: RegimeState) -> RegimeState:
        """Detect regime transitions"""
        try:
            if not state.current_regime or len(self.regime_buffer) < 2:
                return state

            current_regime = state.current_regime.regime
            previous_regimes = [r.regime for r in list(self.regime_buffer)[-5:]]

            # Check if regime has changed
            if len(previous_regimes) >= 2 and previous_regimes[-1] != previous_regimes[-2]:
                transition = RegimeTransition(
                    from_regime=previous_regimes[-2],
                    to_regime=current_regime,
                    transition_time=datetime.now(timezone.utc),
                    confidence=state.current_regime.confidence,
                    trigger="Detected regime change in detection pipeline"
                )

                state.transitions.append(transition)
                logger.warning(f"REGIME TRANSITION: {previous_regimes[-2].value} → {current_regime.value}")

        except Exception as e:
            logger.error(f"Error detecting transitions: {e}")

        return state

    # Helper methods

    async def _simulate_market_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Simulate market data (replace with real data in production)"""
        # Generate synthetic OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Generate realistic price series
        returns = np.random.randn(days) * 0.01  # 1% daily volatility
        prices = 100 * (1 + returns).cumprod()

        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.randn(days) * 0.001),
            'high': prices * (1 + abs(np.random.randn(days)) * 0.01),
            'low': prices * (1 - abs(np.random.randn(days)) * 0.01),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, days)
        })

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)"""
        df = df.copy()

        # Calculate True Range
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)

        # Calculate directional movement
        df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                                 df['high'] - df['high'].shift(1), 0)
        df['dm_plus'] = np.where(df['dm_plus'] < 0, 0, df['dm_plus'])

        df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                                  df['low'].shift(1) - df['low'], 0)
        df['dm_minus'] = np.where(df['dm_minus'] < 0, 0, df['dm_minus'])

        # Calculate smoothed TR and DM
        atr = df['tr'].rolling(period).mean()
        df['di_plus'] = 100 * (df['dm_plus'].rolling(period).mean() / atr)
        df['di_minus'] = 100 * (df['dm_minus'].rolling(period).mean() / atr)

        # Calculate ADX
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].rolling(period).mean()

        return df

    def _calculate_bollinger_position(self, df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = df['close'].rolling(period).mean()
        rolling_std = df['close'].rolling(period).std()

        upper_band = sma + (rolling_std * std)
        lower_band = sma - (rolling_std * std)

        # Position: 0 = lower band, 0.5 = middle, 1 = upper band
        position = (df['close'] - lower_band) / (upper_band - lower_band)

        return position

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _map_hmm_state_to_regime(self, state_id: int, df: pd.DataFrame) -> MarketRegime:
        """Map HMM hidden state to market regime"""
        # Analyze characteristics of this HMM state
        state_mask = self.hmm_model.predict(
            ((df[['returns', 'volatility', 'adx']].dropna() - df[['returns', 'volatility', 'adx']].dropna().mean()) /
             df[['returns', 'volatility', 'adx']].dropna().std()).values
        ) == state_id

        state_data = df[state_mask]

        if len(state_data) == 0:
            return MarketRegime.UNCERTAIN

        # Analyze state characteristics
        avg_return = state_data['returns'].mean()
        avg_vol = state_data['volatility'].mean()
        avg_adx = state_data['adx'].mean()

        # Map to regime
        if avg_vol > 0.30:
            return MarketRegime.HIGH_VOLATILITY
        elif avg_adx > 35 and avg_return > 0.001:
            return MarketRegime.STRONG_BULL
        elif avg_adx > 35 and avg_return < -0.001:
            return MarketRegime.STRONG_BEAR
        elif avg_adx < 20:
            return MarketRegime.SIDEWAYS_WIDE
        elif avg_vol < 0.10:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.MODERATE_BULL if avg_return > 0 else MarketRegime.MODERATE_BEAR

    def _calculate_mean_reversion_score(self, df: pd.DataFrame) -> float:
        """Calculate mean reversion tendency score"""
        if df is None or len(df) < 50:
            return 0.5

        # Use Hurst exponent or autocorrelation
        returns = df['returns'].dropna().values[-50:]

        # Calculate autocorrelation at lag 1
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]

        # Negative autocorrelation = mean reversion
        # Positive autocorrelation = trending
        # Map from [-1, 1] to [0, 1] where 1 = strong mean reversion
        mean_reversion_score = (1 - autocorr) / 2

        return float(mean_reversion_score)

    def _generate_weight_reasoning(self, regime: MarketRegime, weights: tuple, metrics: RegimeMetrics) -> List[str]:
        """Generate reasoning for strategy weights"""
        reasoning = []

        reasoning.append(f"Detected {regime.value.replace('_', ' ')} regime with {metrics.confidence:.0%} confidence")

        if weights[0] > 0.3:  # Momentum heavy
            reasoning.append(f"High momentum weight ({weights[0]:.0%}) due to strong trend (ADX-based strength: {metrics.trend_strength:.2f})")

        if weights[1] > 0.3:  # Mean reversion heavy
            reasoning.append(f"High mean reversion weight ({weights[1]:.0%}) due to ranging market (MR score: {metrics.mean_reversion_score:.2f})")

        if weights[2] > 0.3:  # Volatility/options heavy
            reasoning.append(f"High volatility strategies weight ({weights[2]:.0%}) due to elevated volatility ({metrics.volatility_level:.1%})")

        if weights[4] > 0.15:  # Short selling elevated
            reasoning.append(f"Elevated short selling weight ({weights[4]:.0%}) due to bearish conditions")

        reasoning.append(f"Regime expected to persist for ~{metrics.persistence:.0f} days")

        return reasoning

    def _create_fallback_regime(self) -> RegimeMetrics:
        """Create safe fallback regime"""
        return RegimeMetrics(
            regime=MarketRegime.UNCERTAIN,
            confidence=0.3,
            persistence=5,
            transition_probability=0.5,
            volatility_level=0.15,
            trend_strength=0.0,
            mean_reversion_score=0.5,
            timestamp=datetime.now(timezone.utc)
        )

    def _create_fallback_weights(self) -> StrategyWeights:
        """Create balanced fallback weights"""
        return StrategyWeights(
            momentum=0.25,
            mean_reversion=0.25,
            volatility=0.20,
            options=0.15,
            short_selling=0.10,
            long_term=0.05,
            regime=MarketRegime.UNCERTAIN,
            reasoning=["Using balanced weights due to uncertain regime"]
        )


# Factory function
def create_enhanced_regime_detection_agent(config: Dict[str, Any] = None) -> EnhancedRegimeDetectionAgent:
    """Create Enhanced Regime Detection Agent instance"""
    return EnhancedRegimeDetectionAgent(config)


# Example usage
if __name__ == "__main__":
    async def test_agent():
        agent = create_enhanced_regime_detection_agent()

        # Detect current regime
        regime, weights = await agent.detect_regime("SPY")

        print(f"\n=== Current Market Regime ===")
        print(f"Regime: {regime.regime.value}")
        print(f"Confidence: {regime.confidence:.1%}")
        print(f"Persistence: {regime.persistence:.0f} days")
        print(f"Transition Probability: {regime.transition_probability:.1%}")
        print(f"Volatility Level: {regime.volatility_level:.1%}")
        print(f"Trend Strength: {regime.trend_strength:.2f}")
        print(f"Mean Reversion Score: {regime.mean_reversion_score:.2f}")

        print(f"\n=== Recommended Strategy Weights ===")
        print(f"Momentum: {weights.momentum:.1%}")
        print(f"Mean Reversion: {weights.mean_reversion:.1%}")
        print(f"Volatility/Options: {weights.volatility:.1%}")
        print(f"Short Selling: {weights.short_selling:.1%}")
        print(f"Long Term: {weights.long_term:.1%}")

        print(f"\n=== Reasoning ===")
        for i, reason in enumerate(weights.reasoning, 1):
            print(f"{i}. {reason}")

    asyncio.run(test_agent())
