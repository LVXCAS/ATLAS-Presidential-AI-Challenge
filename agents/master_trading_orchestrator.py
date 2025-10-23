"""
Master Trading Orchestrator - Integrates ALL Enhanced Agents

This is the NEW central hub that coordinates:
- 3 new critical agents (Microstructure, Regime, Correlation)
- 5 upgraded existing agents (Momentum, Mean Reversion, Portfolio, Risk, Options)
- Dynamic weight adjustment based on market conditions
- Portfolio heat monitoring
- Execution optimization

USE THIS instead of your old main loop!
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

# Import all NEW agents
from agents.market_microstructure_agent import create_market_microstructure_agent
from agents.enhanced_regime_detection_agent import create_enhanced_regime_detection_agent
from agents.cross_asset_correlation_agent import create_cross_asset_correlation_agent

# Import existing agents (will be upgraded)
from agents.momentum_trading_agent import MomentumTradingAgent
from agents.mean_reversion_agent import MeanReversionTradingAgent
from agents.portfolio_allocator_agent import PortfolioAllocatorAgent
from agents.risk_manager_agent import RiskManagerAgent
# from agents.options_trading_agent import OptionsTrader  # Optional - init on demand

logger = logging.getLogger(__name__)


@dataclass
class TradingContext:
    """Current trading context with all market information"""
    # Market regime
    regime_name: str = "BALANCED"
    regime_confidence: float = 0.5
    volatility_level: float = 0.15
    trend_strength: float = 0.0

    # Strategy weights (dynamic)
    momentum_weight: float = 0.30
    mean_reversion_weight: float = 0.30
    options_weight: float = 0.20
    ml_weight: float = 0.20

    # Risk metrics
    portfolio_heat_pct: float = 0.0
    can_add_positions: bool = True
    correlation_alerts: List[str] = field(default_factory=list)

    # Execution context
    market_liquidity: str = "NORMAL"
    expected_slippage_bps: float = 10.0

    # Timestamp
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MasterTradingOrchestrator:
    """
    Master orchestrator that integrates ALL enhanced agents

    This is your NEW main trading loop coordinator!
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize all agents"""
        self.config = config or {}

        # Initialize NEW critical agents
        logger.info("Initializing NEW critical agents...")
        self.microstructure_agent = create_market_microstructure_agent()
        self.regime_agent = create_enhanced_regime_detection_agent()
        self.correlation_agent = create_cross_asset_correlation_agent()

        # Initialize EXISTING agents (will be upgraded)
        logger.info("Initializing existing agents...")
        self.momentum_agent = MomentumTradingAgent()
        self.mean_reversion_agent = MeanReversionTradingAgent()
        self.portfolio_allocator = PortfolioAllocatorAgent()
        self.risk_manager = None  # Needs db_config, will init on demand
        self.options_agent = None  # Needs config, will init on demand

        # Trading context
        self.context = TradingContext()

        # Performance tracking
        self.strategy_performance = {
            'momentum': [],
            'mean_reversion': [],
            'options': [],
            'ml_models': []
        }

        logger.info("✅ Master Trading Orchestrator initialized with ALL enhancements!")

    async def analyze_market_regime(self) -> None:
        """
        STEP 1: Analyze current market regime

        This determines which strategies to favor
        """
        try:
            logger.info("=" * 60)
            logger.info("STEP 1: ANALYZING MARKET REGIME")
            logger.info("=" * 60)

            # Detect regime using enhanced agent
            regime, weights = await self.regime_agent.detect_regime("SPY")

            # Update context
            self.context.regime_name = regime.regime.value
            self.context.regime_confidence = regime.confidence
            self.context.volatility_level = regime.volatility_level
            self.context.trend_strength = regime.trend_strength

            # Update strategy weights from regime
            self.context.momentum_weight = weights.momentum
            self.context.mean_reversion_weight = weights.mean_reversion
            self.context.options_weight = weights.options

            logger.info(f"✅ Regime: {regime.regime.value} (confidence: {regime.confidence:.1%})")
            logger.info(f"   Volatility: {regime.volatility_level:.1%}")
            logger.info(f"   Trend Strength: {regime.trend_strength:.2f}")
            logger.info(f"   Recommended Weights:")
            logger.info(f"     - Momentum: {weights.momentum:.1%}")
            logger.info(f"     - Mean Reversion: {weights.mean_reversion:.1%}")
            logger.info(f"     - Options: {weights.options:.1%}")

            for reason in weights.reasoning:
                logger.info(f"   📊 {reason}")

        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            # Use default balanced weights
            self.context.regime_name = "BALANCED"

    async def check_cross_asset_risk(self, portfolio: Dict[str, float]) -> None:
        """
        STEP 2: Check cross-asset correlations and systemic risk

        This detects correlation breakdowns and crisis conditions
        """
        try:
            logger.info("=" * 60)
            logger.info("STEP 2: CHECKING CROSS-ASSET RISK")
            logger.info("=" * 60)

            # Analyze cross-asset correlations
            breakdowns, risk_regime, diversification = await self.correlation_agent.analyze_cross_asset_risk(portfolio)

            # Check for critical alerts
            critical_alerts = [b for b in breakdowns if b.severity > 0.7]

            if critical_alerts:
                logger.critical(f"🚨 {len(critical_alerts)} CRITICAL correlation alerts!")
                for breakdown in critical_alerts:
                    logger.critical(f"   {breakdown.explanation}")
                    self.context.correlation_alerts.append(breakdown.explanation)

            # Log risk regime
            logger.info(f"✅ Risk Regime: {risk_regime.regime.value} (confidence: {risk_regime.confidence:.1%})")
            for reason in risk_regime.reasoning:
                logger.info(f"   💡 {reason}")

            # Log diversification
            logger.info(f"✅ Diversification Score: {diversification.overall_score:.1f}/100")
            logger.info(f"   Avg Correlation: {diversification.avg_correlation:.2f}")
            logger.info(f"   Effective N Assets: {diversification.effective_n_assets:.1f}")

            for rec in diversification.recommendations:
                logger.info(f"   📌 {rec}")

            # Adjust context based on risk regime
            if risk_regime.regime.value == "RISK_OFF":
                logger.warning("⚠️ RISK-OFF regime - increasing defensive positioning")
                self.context.options_weight = min(0.40, self.context.options_weight * 1.5)
                self.context.momentum_weight = max(0.10, self.context.momentum_weight * 0.7)

        except Exception as e:
            logger.error(f"Error checking cross-asset risk: {e}")

    async def check_portfolio_risk(self, positions: List[Dict], portfolio_value: float) -> bool:
        """
        STEP 3: Check portfolio heat and risk limits

        Returns True if safe to trade, False if limits exceeded
        """
        try:
            logger.info("=" * 60)
            logger.info("STEP 3: CHECKING PORTFOLIO RISK")
            logger.info("=" * 60)

            # Check portfolio heat using upgraded risk manager
            risk_check = self.risk_manager.check_portfolio_risk(positions, portfolio_value)

            heat = risk_check['heat']
            self.context.portfolio_heat_pct = heat.heat_percentage
            self.context.can_add_positions = heat.can_add_position

            logger.info(f"✅ Portfolio Heat: ${heat.total_heat:,.0f} ({heat.heat_percentage:.1f}% of portfolio)")
            logger.info(f"   Heat Usage: {heat.heat_usage:.0%} of {heat.heat_limit:.0f}% limit")
            logger.info(f"   Can Add Positions: {'YES ✅' if heat.can_add_position else 'NO ❌'}")

            # Log alerts
            for alert in risk_check['alerts']:
                if alert['severity'] == 'CRITICAL':
                    logger.critical(f"🚨 {alert['message']}")
                    return False  # STOP TRADING
                elif alert['severity'] == 'WARNING':
                    logger.warning(f"⚠️ {alert['message']}")
                else:
                    logger.info(f"ℹ️ {alert['message']}")

            # Log top risky positions
            if risk_check['top_risky_positions']:
                logger.info("   Top Risky Positions:")
                for symbol, heat_amt in risk_check['top_risky_positions']:
                    logger.info(f"     - {symbol}: ${heat_amt:,.0f}")

            return True  # Safe to trade

        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return False  # Be conservative on error

    async def generate_trading_signals(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """
        STEP 4: Generate signals from all strategy agents

        Each agent uses enhanced methods with volume, dynamic thresholds, etc.
        """
        try:
            logger.info("=" * 60)
            logger.info(f"STEP 4: GENERATING SIGNALS FOR {symbol}")
            logger.info("=" * 60)

            signals = {}

            # Momentum signal (with volume indicators)
            try:
                momentum_signal = await self.momentum_agent.generate_signal(symbol, market_data)
                signals['momentum'] = momentum_signal
                logger.info(f"✅ Momentum: {momentum_signal.signal_type.value} "
                          f"(confidence: {momentum_signal.confidence:.1%})")
                for reason in momentum_signal.top_3_reasons[:2]:  # Top 2 reasons
                    logger.info(f"   💡 {reason.explanation}")
            except Exception as e:
                logger.error(f"Momentum signal error: {e}")

            # Mean Reversion signal (with dynamic thresholds)
            try:
                mr_signal = await self.mean_reversion_agent.generate_signal(symbol, market_data)
                signals['mean_reversion'] = mr_signal
                logger.info(f"✅ Mean Reversion: {mr_signal.signal_type.value} "
                          f"(confidence: {mr_signal.confidence:.1%})")
                for reason in mr_signal.top_3_reasons[:2]:
                    logger.info(f"   💡 {reason.explanation}")
            except Exception as e:
                logger.error(f"Mean reversion signal error: {e}")

            # Options signal (with gamma exposure)
            try:
                options_signal = await self.options_agent.generate_signal(symbol, market_data)
                signals['options'] = options_signal
                logger.info(f"✅ Options: {options_signal.strategy} "
                          f"(confidence: {options_signal.confidence:.1%})")
            except Exception as e:
                logger.error(f"Options signal error: {e}")

            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {}

    async def fuse_signals_with_dynamic_weights(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        STEP 5: Fuse signals using DYNAMIC weights based on regime

        This is the UPGRADED portfolio allocator logic
        """
        try:
            logger.info("=" * 60)
            logger.info("STEP 5: FUSING SIGNALS WITH DYNAMIC WEIGHTS")
            logger.info("=" * 60)

            # Use dynamic weights from context
            weights = {
                'momentum': self.context.momentum_weight,
                'mean_reversion': self.context.mean_reversion_weight,
                'options': self.context.options_weight
            }

            logger.info("   Dynamic Weights:")
            for strategy, weight in weights.items():
                logger.info(f"     - {strategy}: {weight:.1%}")

            # Fuse signals
            total_score = 0.0
            total_confidence = 0.0
            reasons = []

            for strategy, signal in signals.items():
                if strategy in weights:
                    weight = weights[strategy]

                    # Extract signal value
                    if hasattr(signal, 'value'):
                        signal_value = signal.value
                        signal_conf = signal.confidence
                    else:
                        signal_value = 0.5  # Default
                        signal_conf = 0.5

                    contribution = signal_value * weight
                    total_score += contribution
                    total_confidence += signal_conf * weight

                    logger.info(f"   {strategy}: value={signal_value:.2f}, weight={weight:.1%}, "
                              f"contribution={contribution:.2f}")

                    # Collect reasons
                    if hasattr(signal, 'top_3_reasons'):
                        for reason in signal.top_3_reasons[:1]:  # Top reason
                            reasons.append(f"{strategy}: {reason.explanation}")

            # Determine action
            if total_score > 0.3:
                action = "BUY"
            elif total_score < -0.3:
                action = "SELL"
            else:
                action = "HOLD"

            result = {
                'action': action,
                'confidence': total_confidence,
                'score': total_score,
                'reasons': reasons,
                'weights_used': weights
            }

            logger.info(f"✅ FINAL SIGNAL: {action} (score: {total_score:.2f}, confidence: {total_confidence:.1%})")

            return result

        except Exception as e:
            logger.error(f"Error fusing signals: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'score': 0.0}

    async def optimize_execution(self, symbol: str, action: str, quantity: float) -> Dict[str, Any]:
        """
        STEP 6: Optimize trade execution to minimize slippage

        Uses market microstructure agent
        """
        try:
            logger.info("=" * 60)
            logger.info(f"STEP 6: OPTIMIZING EXECUTION FOR {symbol}")
            logger.info("=" * 60)

            # Analyze execution
            execution_rec = await self.microstructure_agent.analyze_execution(
                symbol=symbol,
                action=action,
                quantity=quantity
            )

            logger.info(f"✅ Execution Strategy: {execution_rec.execution_strategy}")
            logger.info(f"   Timing: {execution_rec.timing}")
            logger.info(f"   Expected Slippage: {execution_rec.estimated_slippage_bps:.2f} bps")
            logger.info(f"   Expected Impact: {execution_rec.estimated_impact_bps:.2f} bps")
            logger.info(f"   Confidence: {execution_rec.confidence:.1%}")

            logger.info("   Reasoning:")
            for reason in execution_rec.reasoning:
                logger.info(f"     💡 {reason}")

            if execution_rec.recommended_limit_price:
                logger.info(f"   Recommended Limit Price: ${execution_rec.recommended_limit_price:.2f}")

            if execution_rec.recommended_chunks:
                logger.info(f"   Split into {execution_rec.recommended_chunks} chunks over "
                          f"{execution_rec.recommended_interval_minutes} min intervals")

            self.context.expected_slippage_bps = execution_rec.estimated_slippage_bps

            return {
                'strategy': execution_rec.execution_strategy,
                'timing': execution_rec.timing,
                'slippage_bps': execution_rec.estimated_slippage_bps,
                'limit_price': execution_rec.recommended_limit_price,
                'chunks': execution_rec.recommended_chunks,
                'interval_minutes': execution_rec.recommended_interval_minutes
            }

        except Exception as e:
            logger.error(f"Error optimizing execution: {e}")
            return {'strategy': 'MARKET', 'timing': 'IMMEDIATE'}

    async def run_trading_cycle(
        self,
        symbols: List[str],
        portfolio: Dict[str, float],
        positions: List[Dict],
        portfolio_value: float
    ) -> List[Dict[str, Any]]:
        """
        MAIN METHOD: Run complete trading cycle with ALL enhancements

        This orchestrates all 8 agents in the optimal sequence:
        1. Regime Detection
        2. Cross-Asset Risk Check
        3. Portfolio Heat Check
        4. Signal Generation (all strategies)
        5. Signal Fusion (dynamic weights)
        6. Execution Optimization
        """

        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING ENHANCED TRADING CYCLE")
        logger.info("=" * 80)
        logger.info(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Portfolio Value: ${portfolio_value:,.0f}")
        logger.info("=" * 80 + "\n")

        try:
            # STEP 1: Analyze Market Regime
            await self.analyze_market_regime()

            # STEP 2: Check Cross-Asset Risk
            await self.check_cross_asset_risk(portfolio)

            # STEP 3: Check Portfolio Risk
            can_trade = await self.check_portfolio_risk(positions, portfolio_value)

            if not can_trade:
                logger.critical("🛑 TRADING HALTED - Portfolio risk limits exceeded!")
                return []

            if not self.context.can_add_positions:
                logger.warning("⚠️ Cannot add new positions - portfolio heat too high")
                return []

            # STEP 4-6: Process each symbol
            trade_recommendations = []

            for symbol in symbols:
                logger.info(f"\n📊 Processing {symbol}...")

                # Get market data (you'd implement this)
                market_data = await self._fetch_market_data(symbol)

                # STEP 4: Generate signals
                signals = await self.generate_trading_signals(symbol, market_data)

                if not signals:
                    logger.info(f"   ⏭️ No signals for {symbol}")
                    continue

                # STEP 5: Fuse signals
                final_signal = await self.fuse_signals_with_dynamic_weights(signals)

                if final_signal['action'] == 'HOLD':
                    logger.info(f"   ⏸️ HOLD signal for {symbol}")
                    continue

                # Check confidence threshold
                if final_signal['confidence'] < 0.6:
                    logger.info(f"   ⚠️ Confidence too low ({final_signal['confidence']:.1%}) - skipping")
                    continue

                # Calculate position size
                position_size = self._calculate_position_size(
                    portfolio_value,
                    final_signal['confidence'],
                    self.context.portfolio_heat_pct
                )

                # STEP 6: Optimize execution
                execution_plan = await self.optimize_execution(
                    symbol,
                    final_signal['action'],
                    position_size
                )

                # Create trade recommendation
                trade_rec = {
                    'symbol': symbol,
                    'action': final_signal['action'],
                    'quantity': position_size,
                    'confidence': final_signal['confidence'],
                    'score': final_signal['score'],
                    'execution_strategy': execution_plan['strategy'],
                    'execution_timing': execution_plan['timing'],
                    'limit_price': execution_plan.get('limit_price'),
                    'expected_slippage_bps': execution_plan['slippage_bps'],
                    'reasons': final_signal['reasons'],
                    'regime': self.context.regime_name,
                    'timestamp': datetime.now(timezone.utc)
                }

                trade_recommendations.append(trade_rec)

                logger.info(f"\n✅ TRADE RECOMMENDATION: {symbol}")
                logger.info(f"   Action: {final_signal['action']}")
                logger.info(f"   Quantity: {position_size}")
                logger.info(f"   Confidence: {final_signal['confidence']:.1%}")
                logger.info(f"   Execution: {execution_plan['strategy']} ({execution_plan['timing']})")
                logger.info(f"   Expected Slippage: {execution_plan['slippage_bps']:.2f} bps")

            logger.info("\n" + "=" * 80)
            logger.info(f"✅ TRADING CYCLE COMPLETE - {len(trade_recommendations)} recommendations")
            logger.info("=" * 80 + "\n")

            return trade_recommendations

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            return []

    def _calculate_position_size(self, portfolio_value: float, confidence: float, current_heat_pct: float) -> float:
        """Calculate position size based on confidence and available heat"""
        # Base size: 5% of portfolio
        base_size_pct = 0.05

        # Adjust for confidence
        confidence_multiplier = confidence  # 60% confidence = 60% of base size

        # Adjust for available heat (if heat is high, reduce size)
        heat_multiplier = max(0.5, 1 - (current_heat_pct / 15.0))  # 15% is max heat

        final_size_pct = base_size_pct * confidence_multiplier * heat_multiplier
        final_size = portfolio_value * final_size_pct

        return final_size

    async def _fetch_market_data(self, symbol: str) -> Dict:
        """Fetch market data (placeholder - implement with your data source)"""
        # TODO: Implement real data fetching
        return {
            'symbol': symbol,
            'price': 150.0,
            'volume': 1000000,
            'volatility': 0.25
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'regime': self.context.regime_name,
            'regime_confidence': self.context.regime_confidence,
            'volatility': self.context.volatility_level,
            'trend_strength': self.context.trend_strength,
            'portfolio_heat_pct': self.context.portfolio_heat_pct,
            'can_add_positions': self.context.can_add_positions,
            'strategy_weights': {
                'momentum': self.context.momentum_weight,
                'mean_reversion': self.context.mean_reversion_weight,
                'options': self.context.options_weight
            },
            'correlation_alerts': self.context.correlation_alerts,
            'updated_at': self.context.updated_at.isoformat()
        }


# Factory function
def create_master_orchestrator(config: Dict[str, Any] = None) -> MasterTradingOrchestrator:
    """Create Master Trading Orchestrator with all enhancements"""
    return MasterTradingOrchestrator(config)


# Example usage
if __name__ == "__main__":
    async def main():
        # Create orchestrator
        orchestrator = create_master_orchestrator()

        # Example portfolio
        portfolio = {
            'SPY': 0.40,
            'TLT': 0.30,
            'GLD': 0.20,
            'QQQ': 0.10
        }

        # Example positions
        positions = [
            {'symbol': 'SPY', 'quantity': 100, 'price': 450, 'volatility': 0.15, 'beta': 1.0},
            {'symbol': 'TLT', 'quantity': 200, 'price': 95, 'volatility': 0.12, 'beta': -0.3},
        ]

        portfolio_value = 100000

        # Symbols to analyze
        symbols = ['AAPL', 'MSFT', 'GOOGL']

        # Run trading cycle
        recommendations = await orchestrator.run_trading_cycle(
            symbols=symbols,
            portfolio=portfolio,
            positions=positions,
            portfolio_value=portfolio_value
        )

        # Print recommendations
        print("\n" + "=" * 80)
        print("TRADE RECOMMENDATIONS")
        print("=" * 80)
        for rec in recommendations:
            print(f"\n{rec['symbol']}:")
            print(f"  Action: {rec['action']}")
            print(f"  Quantity: {rec['quantity']:.0f}")
            print(f"  Confidence: {rec['confidence']:.1%}")
            print(f"  Execution: {rec['execution_strategy']}")
            print(f"  Reasons:")
            for reason in rec['reasons']:
                print(f"    - {reason}")

        # Print system status
        print("\n" + "=" * 80)
        print("SYSTEM STATUS")
        print("=" * 80)
        status = orchestrator.get_system_status()
        for key, value in status.items():
            print(f"{key}: {value}")

    asyncio.run(main())
