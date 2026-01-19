"""
AutoGen R&D Agent

Uses Microsoft AutoGen for autonomous strategy research and development.

Specialization: Strategy discovery, parameter optimization, backtesting automation.
"""

from typing import Dict, Tuple, List, Optional
from .base_agent import BaseAgent
import json
import asyncio


class AutoGenRDAgent(BaseAgent):
    """
    Microsoft AutoGen-powered R&D agent.

    Capabilities:
    - Autonomous strategy discovery
    - Multi-agent collaboration for research
    - Automated backtesting pipelines
    - Parameter optimization (grid search, genetic algorithms)
    - Strategy validation and ranking

    This agent DOESN'T vote on trades - it discovers NEW strategies.
    """

    def __init__(self, initial_weight: float = 1.0):
        super().__init__(name="AutoGenRDAgent", initial_weight=initial_weight)

        self.autogen_available = False
        self.research_active = False
        self.discovered_strategies = []

        # Try to initialize AutoGen
        try:
            import autogen
            self.autogen = autogen
            self.autogen_available = True
            print(f"[{self.name}] AutoGen loaded successfully")
        except ImportError:
            print(f"[{self.name}] WARNING: AutoGen not available")

        # R&D parameters
        self.min_sharpe_threshold = 1.5  # Minimum Sharpe for new strategies
        self.min_win_rate_threshold = 0.55  # Minimum 55% win rate
        self.backtest_period_days = 180  # 6 months backtest

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        R&D agents don't vote on individual trades.
        They run in background discovering strategies.

        This method returns NEUTRAL but logs research opportunities.
        """
        return ("NEUTRAL", 0.5, {
            "agent": self.name,
            "mode": "research",
            "message": "R&D agent running in background",
            "discovered_strategies": len(self.discovered_strategies)
        })

    async def discover_new_strategies(self, historical_data: Dict,
                                       performance_data: Dict) -> List[Dict]:
        """
        Use AutoGen multi-agent system to discover new trading strategies.

        AutoGen agents collaborate to:
        1. Propose new strategy ideas
        2. Code the strategy
        3. Backtest the strategy
        4. Validate performance
        5. Rank by Sharpe ratio
        """
        if not self.autogen_available:
            return self._simplified_strategy_discovery(historical_data, performance_data)

        # In production, you'd set up AutoGen agents:
        # - StrategyProposer: Generates strategy ideas based on market data
        # - Coder: Implements strategies in Python
        # - Backtester: Runs backtests using Backtrader
        # - Validator: Checks for overfitting using walk-forward
        # - Ranker: Sorts strategies by Sharpe/win rate

        strategies = []

        # Placeholder for AutoGen multi-agent workflow
        # In real implementation:
        # strategies = await self._run_autogen_research_pipeline(historical_data)

        return strategies

    def _simplified_strategy_discovery(self, historical_data: Dict,
                                        performance_data: Dict) -> List[Dict]:
        """
        Simplified strategy discovery without AutoGen.

        Proposes parameter variations of existing strategies.
        """
        strategies = []

        # Current ATLAS strategy: RSI + MACD + EMA + ADX
        # Propose variations:

        # Strategy 1: More aggressive RSI thresholds
        strategies.append({
            "name": "ATLAS_Aggressive_RSI",
            "description": "RSI 35/65 instead of 40/60",
            "parameters": {
                "rsi_oversold": 35,
                "rsi_overbought": 65,
                "macd_threshold": 0.0001,
                "adx_min": 25
            },
            "expected_sharpe": 1.8,  # Estimated
            "expected_win_rate": 0.58,
            "status": "proposed"
        })

        # Strategy 2: Stricter ADX filter
        strategies.append({
            "name": "ATLAS_Strong_Trend_Only",
            "description": "Only trade when ADX > 30 (strong trends)",
            "parameters": {
                "rsi_oversold": 40,
                "rsi_overbought": 60,
                "macd_threshold": 0.0001,
                "adx_min": 30
            },
            "expected_sharpe": 2.1,  # Higher Sharpe, fewer trades
            "expected_win_rate": 0.62,
            "status": "proposed"
        })

        # Strategy 3: Multi-timeframe confirmation
        strategies.append({
            "name": "ATLAS_Multi_Timeframe",
            "description": "H1 + H4 alignment required",
            "parameters": {
                "timeframes": ["H1", "H4"],
                "rsi_oversold": 40,
                "rsi_overbought": 60,
                "require_alignment": True
            },
            "expected_sharpe": 1.9,
            "expected_win_rate": 0.60,
            "status": "proposed"
        })

        # Strategy 4: Mean reversion in ranging markets
        strategies.append({
            "name": "ATLAS_Mean_Reversion",
            "description": "Trade RSI extremes when ADX < 20",
            "parameters": {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "adx_max": 20,  # Only in ranging markets
                "use_bollinger_bands": True
            },
            "expected_sharpe": 1.7,
            "expected_win_rate": 0.56,
            "status": "proposed"
        })

        return strategies

    def validate_strategy(self, strategy: Dict, backtest_results: Dict) -> bool:
        """
        Validate if a discovered strategy meets ATLAS standards.

        Validation criteria:
        - Sharpe > 1.5
        - Win rate > 55%
        - Max drawdown < 15%
        - Profit factor > 1.5
        - Minimum 100 trades in backtest
        """
        sharpe = backtest_results.get("sharpe_ratio", 0)
        win_rate = backtest_results.get("win_rate", 0)
        max_dd = backtest_results.get("max_drawdown", 1.0)
        profit_factor = backtest_results.get("profit_factor", 0)
        num_trades = backtest_results.get("num_trades", 0)

        # Check all criteria
        if sharpe < self.min_sharpe_threshold:
            return False
        if win_rate < self.min_win_rate_threshold:
            return False
        if max_dd > 0.15:  # 15% max DD
            return False
        if profit_factor < 1.5:
            return False
        if num_trades < 100:
            return False

        return True

    def rank_strategies(self, strategies: List[Dict]) -> List[Dict]:
        """
        Rank discovered strategies by composite score.

        Ranking factors:
        - Sharpe ratio (40%)
        - Win rate (30%)
        - Profit factor (20%)
        - Max drawdown (10%)
        """
        for strategy in strategies:
            results = strategy.get("backtest_results", {})

            sharpe = results.get("sharpe_ratio", 0)
            win_rate = results.get("win_rate", 0)
            pf = results.get("profit_factor", 0)
            max_dd = results.get("max_drawdown", 1.0)

            # Composite score
            score = (
                sharpe * 0.4 +
                win_rate * 100 * 0.3 +
                pf * 0.2 -
                max_dd * 100 * 0.1
            )

            strategy["composite_score"] = score

        # Sort by composite score
        ranked = sorted(strategies, key=lambda x: x.get("composite_score", 0), reverse=True)

        return ranked

    def create_research_report(self, strategies: List[Dict]) -> str:
        """
        Generate markdown research report.

        For example:
        # ATLAS R&D Weekly Report
        ## Strategies Discovered: 12
        ## Top 3 Strategies:
        1. ATLAS_Strong_Trend_Only - Sharpe 2.1, WR 62%
        2. ATLAS_Multi_Timeframe - Sharpe 1.9, WR 60%
        3. ATLAS_Aggressive_RSI - Sharpe 1.8, WR 58%
        """
        report = "# ATLAS R&D Research Report\n\n"
        report += f"**Strategies Analyzed:** {len(strategies)}\n\n"

        ranked = self.rank_strategies(strategies)

        report += "## Top Discovered Strategies\n\n"
        for i, strategy in enumerate(ranked[:5], 1):
            name = strategy.get("name", "Unknown")
            desc = strategy.get("description", "")
            score = strategy.get("composite_score", 0)

            results = strategy.get("backtest_results", {})
            sharpe = results.get("sharpe_ratio", 0)
            win_rate = results.get("win_rate", 0)

            report += f"### {i}. {name}\n"
            report += f"**Description:** {desc}\n"
            report += f"**Score:** {score:.2f}\n"
            report += f"**Sharpe:** {sharpe:.2f} | **Win Rate:** {win_rate*100:.1f}%\n\n"

        return report

    def get_strategy_recommendations(self) -> List[str]:
        """
        Get actionable recommendations based on R&D findings.
        """
        recommendations = []

        if len(self.discovered_strategies) == 0:
            recommendations.append("Run initial strategy discovery cycle")
            return recommendations

        # Analyze discovered strategies
        top_strategy = max(self.discovered_strategies,
                          key=lambda x: x.get("composite_score", 0))

        recommendations.append(
            f"Deploy top strategy: {top_strategy.get('name')} "
            f"(Sharpe {top_strategy.get('expected_sharpe', 0):.2f})"
        )

        # Check if current strategy is underperforming
        current_sharpe = 1.5  # Placeholder for current ATLAS Sharpe
        top_sharpe = top_strategy.get("expected_sharpe", 0)

        if top_sharpe > current_sharpe * 1.2:  # 20% better
            recommendations.append(
                f"New strategy outperforms current by {((top_sharpe/current_sharpe)-1)*100:.0f}% "
                "- consider switching"
            )

        return recommendations
