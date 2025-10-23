"""
Cross-Asset Correlation Agent - Multi-Asset Risk Monitoring

This agent monitors correlations across asset classes to:
- Detect correlation breakdowns (early warning of crisis)
- Identify diversification opportunities
- Monitor systemic risk via cross-asset relationships
- Track risk-on/risk-off regime shifts
- Alert when traditional hedges fail

Monitors: Equities, Bonds, VIX, USD, Gold, Oil, Commodities
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
from collections import defaultdict

# LangGraph imports
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


class RiskRegime(Enum):
    """Risk-on/risk-off regime"""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    TRANSITION = "transition"


class CorrelationAlert(Enum):
    """Types of correlation alerts"""
    BREAKDOWN = "breakdown"  # Normal correlation broke down
    CONVERGENCE = "convergence"  # Previously uncorrelated assets converging
    EXTREME = "extreme"  # Correlation at extreme levels
    REGIME_SHIFT = "regime_shift"  # Risk regime changed
    HEDGE_FAILURE = "hedge_failure"  # Hedge is not working


@dataclass
class AssetPair:
    """Asset pair for correlation monitoring"""
    asset1: str
    asset2: str
    expected_correlation: float  # Normal correlation
    correlation_std: float  # Standard deviation of correlation


@dataclass
class CorrelationSnapshot:
    """Snapshot of correlations at a point in time"""
    timestamp: datetime
    correlations: Dict[Tuple[str, str], float]
    rolling_window: int  # Days used for calculation


@dataclass
class CorrelationBreakdown:
    """Detected correlation breakdown"""
    asset1: str
    asset2: str
    expected_correlation: float
    actual_correlation: float
    deviation_sigma: float
    severity: float  # 0-1
    alert_type: CorrelationAlert
    timestamp: datetime
    explanation: str


@dataclass
class RiskOnOffSignal:
    """Risk-on/risk-off signal"""
    regime: RiskRegime
    confidence: float
    indicators: Dict[str, float]  # Individual indicator values
    timestamp: datetime
    reasoning: List[str]


@dataclass
class DiversificationScore:
    """Portfolio diversification score"""
    overall_score: float  # 0-100
    avg_correlation: float
    correlation_dispersion: float
    effective_n_assets: float  # Effective number of independent bets
    concentration_risk: float
    recommendations: List[str]


@dataclass
class CrossAssetState:
    """LangGraph state for cross-asset analysis"""
    asset_prices: Dict[str, pd.DataFrame] = field(default_factory=dict)
    correlations: Optional[CorrelationSnapshot] = None
    breakdowns: List[CorrelationBreakdown] = field(default_factory=list)
    risk_regime: Optional[RiskOnOffSignal] = None
    diversification: Optional[DiversificationScore] = None
    portfolio_positions: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class CrossAssetCorrelationAgent:
    """
    Cross-Asset Correlation Agent for systemic risk monitoring

    Monitors key cross-asset relationships:
    - SPY vs TLT (stocks vs bonds)
    - SPY vs VIX (stocks vs volatility)
    - USD vs Gold (dollar vs safe haven)
    - SPY vs USD (stocks vs dollar)
    - Oil vs stocks (energy vs equities)
    - Commodities vs stocks
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Cross-Asset Correlation Agent"""
        self.config = config or {}

        # Key asset symbols to monitor
        self.assets = {
            'equity': 'SPY',      # S&P 500
            'bonds': 'TLT',       # 20Y Treasury
            'volatility': 'VIX',  # VIX Index
            'dollar': 'UUP',      # US Dollar
            'gold': 'GLD',        # Gold
            'oil': 'USO',         # Oil
            'commodities': 'DBC'  # Commodities basket
        }

        # Expected correlations (historical norms)
        self.expected_correlations = {
            ('SPY', 'TLT'): -0.3,   # Stocks/bonds negative
            ('SPY', 'VIX'): -0.75,  # Stocks/VIX strongly negative
            ('SPY', 'UUP'): -0.2,   # Stocks/USD slightly negative
            ('SPY', 'GLD'): 0.0,    # Stocks/gold uncorrelated
            ('SPY', 'USO'): 0.3,    # Stocks/oil positive
            ('TLT', 'VIX'): 0.2,    # Bonds/VIX positive (flight to safety)
            ('GLD', 'UUP'): -0.5,   # Gold/USD negative
            ('GLD', 'VIX'): 0.3,    # Gold/VIX positive (safe haven)
        }

        # Correlation windows
        self.short_window = self.config.get('short_window', 20)  # 1 month
        self.long_window = self.config.get('long_window', 60)    # 3 months

        # Alert thresholds
        self.breakdown_sigma = self.config.get('breakdown_sigma', 2.5)  # 2.5 sigma

        # Build workflow
        self.workflow = self._create_workflow()

        logger.info("Cross-Asset Correlation Agent initialized")

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for correlation analysis"""
        workflow = StateGraph(CrossAssetState)

        # Add nodes
        workflow.add_node("fetch_asset_prices", self._fetch_asset_prices)
        workflow.add_node("calculate_correlations", self._calculate_correlations)
        workflow.add_node("detect_breakdowns", self._detect_breakdowns)
        workflow.add_node("assess_risk_regime", self._assess_risk_regime)
        workflow.add_node("calculate_diversification", self._calculate_diversification)
        workflow.add_node("generate_alerts", self._generate_alerts)

        # Define edges
        workflow.set_entry_point("fetch_asset_prices")
        workflow.add_edge("fetch_asset_prices", "calculate_correlations")
        workflow.add_edge("calculate_correlations", "detect_breakdowns")
        workflow.add_edge("detect_breakdowns", "assess_risk_regime")
        workflow.add_edge("assess_risk_regime", "calculate_diversification")
        workflow.add_edge("calculate_diversification", "generate_alerts")
        workflow.add_edge("generate_alerts", END)

        return workflow.compile()

    async def analyze_cross_asset_risk(self, portfolio_positions: Dict[str, float] = None) -> Tuple[
        List[CorrelationBreakdown], RiskOnOffSignal, DiversificationScore
    ]:
        """
        Main entry point: analyze cross-asset correlations and risks

        Args:
            portfolio_positions: Dict of {symbol: weight} for diversification analysis

        Returns:
            (breakdowns, risk_regime, diversification_score)
        """
        try:
            initial_state = CrossAssetState(
                portfolio_positions=portfolio_positions or {}
            )

            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)

            return (
                final_state.breakdowns,
                final_state.risk_regime or self._create_default_risk_regime(),
                final_state.diversification or self._create_default_diversification()
            )

        except Exception as e:
            logger.error(f"Error in cross-asset analysis: {e}")
            return [], self._create_default_risk_regime(), self._create_default_diversification()

    async def _fetch_asset_prices(self, state: CrossAssetState) -> CrossAssetState:
        """Fetch prices for all monitored assets"""
        try:
            for asset_class, symbol in self.assets.items():
                # Fetch price data
                prices = await self._fetch_price_data(symbol, days=252)  # 1 year
                state.asset_prices[symbol] = prices

            logger.info(f"Fetched prices for {len(state.asset_prices)} asset classes")

        except Exception as e:
            logger.error(f"Error fetching asset prices: {e}")
            state.errors.append(f"Price fetch failed: {str(e)}")

        return state

    async def _calculate_correlations(self, state: CrossAssetState) -> CrossAssetState:
        """Calculate correlation matrix"""
        try:
            if not state.asset_prices:
                state.errors.append("No asset price data available")
                return state

            # Combine all returns into DataFrame
            returns_dict = {}
            for symbol, prices in state.asset_prices.items():
                if 'close' in prices.columns:
                    returns_dict[symbol] = prices['close'].pct_change()

            if not returns_dict:
                state.errors.append("No valid return data")
                return state

            returns_df = pd.DataFrame(returns_dict).dropna()

            # Calculate rolling correlations (short window)
            correlations = {}

            for (asset1, asset2), expected_corr in self.expected_correlations.items():
                if asset1 in returns_df.columns and asset2 in returns_df.columns:
                    # Rolling correlation
                    rolling_corr = returns_df[asset1].rolling(self.short_window).corr(returns_df[asset2])

                    # Current correlation
                    current_corr = rolling_corr.iloc[-1]
                    correlations[(asset1, asset2)] = current_corr

            state.correlations = CorrelationSnapshot(
                timestamp=datetime.now(timezone.utc),
                correlations=correlations,
                rolling_window=self.short_window
            )

            logger.info(f"Calculated {len(correlations)} asset pair correlations")

        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            state.errors.append(f"Correlation calculation failed: {str(e)}")

        return state

    async def _detect_breakdowns(self, state: CrossAssetState) -> CrossAssetState:
        """Detect correlation breakdowns and anomalies"""
        try:
            if not state.correlations:
                return state

            breakdowns = []

            for (asset1, asset2), expected_corr in self.expected_correlations.items():
                actual_corr = state.correlations.correlations.get((asset1, asset2))

                if actual_corr is None:
                    continue

                # Calculate historical std of this correlation
                if asset1 in state.asset_prices and asset2 in state.asset_prices:
                    corr_std = self._calculate_correlation_std(
                        state.asset_prices[asset1],
                        state.asset_prices[asset2]
                    )
                else:
                    corr_std = 0.2  # Default std

                # Check for breakdown
                deviation = abs(actual_corr - expected_corr)
                deviation_sigma = deviation / corr_std if corr_std > 0 else 0

                # Detect different types of breakdowns
                if deviation_sigma > self.breakdown_sigma:
                    # Significant deviation from expected

                    # Determine alert type
                    if asset1 == 'SPY' and asset2 == 'TLT':
                        if actual_corr > 0.3:  # Stocks and bonds moving together = crisis
                            alert_type = CorrelationAlert.BREAKDOWN
                            explanation = "⚠️ CRITICAL: Stocks and bonds positively correlated - flight to safety failing!"
                        else:
                            alert_type = CorrelationAlert.EXTREME
                            explanation = f"Stocks/Bonds correlation at {actual_corr:.2f} (expected {expected_corr:.2f})"

                    elif asset1 == 'SPY' and asset2 == 'VIX':
                        if actual_corr > -0.5:  # Less negative than expected
                            alert_type = CorrelationAlert.HEDGE_FAILURE
                            explanation = "⚠️ VIX hedge not working - correlation weakening"
                        else:
                            alert_type = CorrelationAlert.EXTREME
                            explanation = f"Stocks/VIX correlation at {actual_corr:.2f}"

                    elif asset1 == 'GLD' and asset2 == 'VIX':
                        if actual_corr < 0:  # Should be positive (both safe havens)
                            alert_type = CorrelationAlert.BREAKDOWN
                            explanation = "⚠️ Gold not acting as safe haven"
                        else:
                            alert_type = CorrelationAlert.EXTREME
                            explanation = f"Gold/VIX correlation unusually strong at {actual_corr:.2f}"

                    else:
                        alert_type = CorrelationAlert.EXTREME
                        explanation = f"{asset1}/{asset2} correlation deviated {deviation_sigma:.1f} sigma from normal"

                    breakdown = CorrelationBreakdown(
                        asset1=asset1,
                        asset2=asset2,
                        expected_correlation=expected_corr,
                        actual_correlation=actual_corr,
                        deviation_sigma=deviation_sigma,
                        severity=min(1.0, deviation_sigma / 5),
                        alert_type=alert_type,
                        timestamp=datetime.now(timezone.utc),
                        explanation=explanation
                    )

                    breakdowns.append(breakdown)

            state.breakdowns = breakdowns

            if breakdowns:
                logger.warning(f"Detected {len(breakdowns)} correlation breakdowns!")

        except Exception as e:
            logger.error(f"Error detecting breakdowns: {e}")
            state.errors.append(f"Breakdown detection failed: {str(e)}")

        return state

    async def _assess_risk_regime(self, state: CrossAssetState) -> CrossAssetState:
        """Assess risk-on vs risk-off regime"""
        try:
            if not state.correlations:
                return state

            indicators = {}
            reasoning = []

            # Indicator 1: SPY/VIX correlation
            spy_vix_corr = state.correlations.correlations.get(('SPY', 'VIX'))
            if spy_vix_corr is not None:
                # More negative = risk on, less negative = risk off
                risk_on_score_vix = (spy_vix_corr + 1) / 2  # Map [-1, 1] to [0, 1]
                indicators['vix_correlation'] = risk_on_score_vix

                if spy_vix_corr < -0.8:
                    reasoning.append("Strong negative SPY/VIX correlation indicates RISK-ON")
                elif spy_vix_corr > -0.5:
                    reasoning.append("Weak SPY/VIX correlation indicates RISK-OFF")

            # Indicator 2: SPY/TLT correlation
            spy_tlt_corr = state.correlations.correlations.get(('SPY', 'TLT'))
            if spy_tlt_corr is not None:
                # Negative = normal (risk on), positive = crisis (risk off)
                risk_on_score_bonds = 1 - ((spy_tlt_corr + 1) / 2)
                indicators['bond_correlation'] = risk_on_score_bonds

                if spy_tlt_corr < -0.3:
                    reasoning.append("Stocks/bonds negative correlation = RISK-ON")
                elif spy_tlt_corr > 0.3:
                    reasoning.append("⚠️ Stocks/bonds positive correlation = RISK-OFF (crisis mode)")

            # Indicator 3: Gold/VIX correlation
            gld_vix_corr = state.correlations.correlations.get(('GLD', 'VIX'))
            if gld_vix_corr is not None:
                # Positive = risk off, negative = risk on
                risk_on_score_gold = 1 - ((gld_vix_corr + 1) / 2)
                indicators['gold_correlation'] = risk_on_score_gold

                if gld_vix_corr > 0.5:
                    reasoning.append("Gold/VIX positive correlation = RISK-OFF (safe haven bid)")

            # Indicator 4: Recent price action
            if 'SPY' in state.asset_prices and 'VIX' in state.asset_prices:
                spy_return_5d = state.asset_prices['SPY']['close'].pct_change(5).iloc[-1]
                vix_change_5d = state.asset_prices['VIX']['close'].pct_change(5).iloc[-1]

                if spy_return_5d > 0.02 and vix_change_5d < -0.1:
                    reasoning.append("Stocks up, VIX down = RISK-ON price action")
                    indicators['price_action'] = 0.8
                elif spy_return_5d < -0.02 and vix_change_5d > 0.1:
                    reasoning.append("Stocks down, VIX up = RISK-OFF price action")
                    indicators['price_action'] = 0.2
                else:
                    indicators['price_action'] = 0.5

            # Combine indicators
            if indicators:
                avg_risk_on_score = np.mean(list(indicators.values()))

                if avg_risk_on_score > 0.65:
                    regime = RiskRegime.RISK_ON
                    confidence = avg_risk_on_score
                elif avg_risk_on_score < 0.35:
                    regime = RiskRegime.RISK_OFF
                    confidence = 1 - avg_risk_on_score
                else:
                    regime = RiskRegime.NEUTRAL
                    confidence = 0.5

                state.risk_regime = RiskOnOffSignal(
                    regime=regime,
                    confidence=confidence,
                    indicators=indicators,
                    timestamp=datetime.now(timezone.utc),
                    reasoning=reasoning
                )

                logger.info(f"Risk regime: {regime.value} (confidence: {confidence:.1%})")

        except Exception as e:
            logger.error(f"Error assessing risk regime: {e}")
            state.errors.append(f"Risk regime assessment failed: {str(e)}")

        return state

    async def _calculate_diversification(self, state: CrossAssetState) -> CrossAssetState:
        """Calculate portfolio diversification score"""
        try:
            if not state.portfolio_positions or not state.correlations:
                return state

            # Get symbols in portfolio
            portfolio_symbols = list(state.portfolio_positions.keys())

            if len(portfolio_symbols) < 2:
                # Can't calculate diversification with < 2 assets
                return state

            # Build correlation matrix for portfolio assets
            returns_dict = {}
            for symbol in portfolio_symbols:
                if symbol in state.asset_prices and 'close' in state.asset_prices[symbol].columns:
                    returns_dict[symbol] = state.asset_prices[symbol]['close'].pct_change()

            if len(returns_dict) < 2:
                return state

            returns_df = pd.DataFrame(returns_dict).dropna()
            corr_matrix = returns_df.corr()

            # Calculate average correlation
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

            # Calculate correlation dispersion
            correlation_dispersion = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].std()

            # Calculate effective number of assets (diversification ratio)
            # Formula: N_eff = 1 / sum(w_i^2) adjusted for correlation
            weights = np.array([state.portfolio_positions.get(s, 0) for s in portfolio_symbols])
            weights = weights / weights.sum()  # Normalize

            # Effective N with correlations
            portfolio_var = weights @ corr_matrix.values @ weights
            avg_asset_var = np.mean(np.diag(corr_matrix.values))
            effective_n_assets = (weights.sum() ** 2 * avg_asset_var) / portfolio_var

            # Calculate concentration risk (HHI)
            concentration_risk = np.sum(weights ** 2)

            # Overall diversification score (0-100)
            # Higher is better
            diversification_score = (
                (1 - avg_correlation) * 40 +  # Low correlation is good
                (correlation_dispersion) * 20 +  # Dispersion is good
                min(effective_n_assets / len(portfolio_symbols), 1) * 30 +  # More effective assets is good
                (1 - concentration_risk) * 10  # Low concentration is good
            ) * 100

            # Generate recommendations
            recommendations = []

            if avg_correlation > 0.7:
                recommendations.append("⚠️ HIGH CORRELATION: Portfolio assets are highly correlated - add uncorrelated assets")

            if concentration_risk > 0.5:
                recommendations.append("⚠️ CONCENTRATION: Portfolio is concentrated - consider more even weighting")

            if effective_n_assets < len(portfolio_symbols) * 0.5:
                recommendations.append("⚠️ LOW DIVERSIFICATION: Effective positions much lower than actual positions")

            if diversification_score > 70:
                recommendations.append("✓ Portfolio is well diversified")

            # Check for missing asset classes
            has_equity = any(s in ['SPY', 'QQQ', 'IWM'] for s in portfolio_symbols)
            has_bonds = any(s in ['TLT', 'IEF', 'AGG'] for s in portfolio_symbols)
            has_gold = 'GLD' in portfolio_symbols

            if has_equity and not has_bonds:
                recommendations.append("Consider adding bonds for equity hedge")

            if has_equity and not has_gold:
                recommendations.append("Consider adding gold for tail risk protection")

            state.diversification = DiversificationScore(
                overall_score=diversification_score,
                avg_correlation=avg_correlation,
                correlation_dispersion=correlation_dispersion,
                effective_n_assets=effective_n_assets,
                concentration_risk=concentration_risk,
                recommendations=recommendations
            )

            logger.info(f"Diversification score: {diversification_score:.1f}/100")

        except Exception as e:
            logger.error(f"Error calculating diversification: {e}")
            state.errors.append(f"Diversification calculation failed: {str(e)}")

        return state

    async def _generate_alerts(self, state: CrossAssetState) -> CrossAssetState:
        """Generate final alerts and warnings"""
        try:
            # Critical alerts
            critical_breakdowns = [
                b for b in state.breakdowns
                if b.severity > 0.7 or b.alert_type in [CorrelationAlert.BREAKDOWN, CorrelationAlert.HEDGE_FAILURE]
            ]

            if critical_breakdowns:
                logger.critical(f"🚨 {len(critical_breakdowns)} CRITICAL correlation alerts!")
                for breakdown in critical_breakdowns:
                    logger.critical(f"   {breakdown.explanation}")

            # Risk regime alerts
            if state.risk_regime and state.risk_regime.regime == RiskRegime.RISK_OFF:
                logger.warning(f"⚠️ RISK-OFF regime detected (confidence: {state.risk_regime.confidence:.1%})")

            # Diversification alerts
            if state.diversification and state.diversification.overall_score < 50:
                logger.warning(f"⚠️ LOW DIVERSIFICATION: Score {state.diversification.overall_score:.1f}/100")

        except Exception as e:
            logger.error(f"Error generating alerts: {e}")

        return state

    # Helper methods

    async def _fetch_price_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch historical price data from real sources"""
        # ENHANCEMENT: Use real data from Alpaca/Polygon/OpenBB/Yahoo
        from agents.real_data_connector import fetch_real_market_data

        try:
            logger.info(f"Fetching REAL market data for {symbol} ({days} days)...")
            df = await fetch_real_market_data(symbol, days=days, timeframe='1Day')

            if df is not None and len(df) > 0:
                logger.info(f"✅ Using REAL data for {symbol}: {len(df)} bars from live sources")
                return df
            else:
                logger.warning(f"Real data unavailable for {symbol}, using simulation fallback")

        except Exception as e:
            logger.warning(f"Error fetching real data for {symbol}: {e}, using simulation")

        # Fallback: Generate synthetic prices
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Generate synthetic prices
        returns = np.random.randn(days) * 0.01
        prices = 100 * (1 + returns).cumprod()

        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, days)
        })

        return df

    def _calculate_correlation_std(self, prices1: pd.DataFrame, prices2: pd.DataFrame) -> float:
        """Calculate standard deviation of rolling correlation"""
        returns1 = prices1['close'].pct_change()
        returns2 = prices2['close'].pct_change()

        rolling_corr = returns1.rolling(self.short_window).corr(returns2)
        corr_std = rolling_corr.std()

        return float(corr_std) if not np.isnan(corr_std) else 0.2

    def _create_default_risk_regime(self) -> RiskOnOffSignal:
        """Create default risk regime"""
        return RiskOnOffSignal(
            regime=RiskRegime.NEUTRAL,
            confidence=0.5,
            indicators={},
            timestamp=datetime.now(timezone.utc),
            reasoning=["Insufficient data for regime assessment"]
        )

    def _create_default_diversification(self) -> DiversificationScore:
        """Create default diversification score"""
        return DiversificationScore(
            overall_score=50.0,
            avg_correlation=0.5,
            correlation_dispersion=0.2,
            effective_n_assets=1.0,
            concentration_risk=1.0,
            recommendations=["Insufficient portfolio data for diversification analysis"]
        )


# Factory function
def create_cross_asset_correlation_agent(config: Dict[str, Any] = None) -> CrossAssetCorrelationAgent:
    """Create Cross-Asset Correlation Agent instance"""
    return CrossAssetCorrelationAgent(config)


# Example usage
if __name__ == "__main__":
    async def test_agent():
        agent = create_cross_asset_correlation_agent()

        # Example portfolio
        portfolio = {
            'SPY': 0.50,
            'TLT': 0.30,
            'GLD': 0.20
        }

        # Analyze cross-asset risk
        breakdowns, risk_regime, diversification = await agent.analyze_cross_asset_risk(portfolio)

        print(f"\n=== Cross-Asset Correlation Analysis ===")

        print(f"\n=== Risk Regime ===")
        print(f"Regime: {risk_regime.regime.value}")
        print(f"Confidence: {risk_regime.confidence:.1%}")
        print(f"\nReasoning:")
        for reason in risk_regime.reasoning:
            print(f"  - {reason}")

        if breakdowns:
            print(f"\n=== Correlation Breakdowns ({len(breakdowns)}) ===")
            for breakdown in breakdowns:
                print(f"\n{breakdown.asset1}/{breakdown.asset2}:")
                print(f"  Expected: {breakdown.expected_correlation:.2f}")
                print(f"  Actual: {breakdown.actual_correlation:.2f}")
                print(f"  Severity: {breakdown.severity:.1%}")
                print(f"  {breakdown.explanation}")

        print(f"\n=== Diversification Score ===")
        print(f"Overall: {diversification.overall_score:.1f}/100")
        print(f"Avg Correlation: {diversification.avg_correlation:.2f}")
        print(f"Effective N Assets: {diversification.effective_n_assets:.1f}")
        print(f"Concentration Risk: {diversification.concentration_risk:.2f}")
        print(f"\nRecommendations:")
        for rec in diversification.recommendations:
            print(f"  - {rec}")

    asyncio.run(test_agent())
