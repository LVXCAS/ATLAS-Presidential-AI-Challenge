"""
Scientific Backtest to Find Optimal ATLAS Threshold

Uses historical data to simulate ATLAS trading with different thresholds
and determines which threshold maximizes performance.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
import statistics
from pathlib import Path

from adapters.oanda_adapter import OandaAdapter
from core.coordinator import ATLASCoordinator

# Import all agents
from agents.technical_agent import TechnicalAgent
from agents.pattern_recognition_agent import PatternRecognitionAgent
from agents.news_filter_agent import NewsFilterAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.xgboost_ml_agent import XGBoostMLAgent
from agents.sentiment_agent import SentimentAgent
from agents.qlib_research_agent import QlibResearchAgent
from agents.gs_quant_agent import GSQuantAgent
from agents.market_regime_agent import MarketRegimeAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.session_timing_agent import SessionTimingAgent
from agents.correlation_agent import CorrelationAgent
from agents.multi_timeframe_agent import MultiTimeframeAgent
from agents.volume_liquidity_agent import VolumeLiquidityAgent
from agents.support_resistance_agent import SupportResistanceAgent
from agents.divergence_agent import DivergenceAgent


@dataclass
class BacktestTrade:
    """Trade result from backtest"""
    timestamp: str
    pair: str
    atlas_score: float
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str  # "stop_loss" or "take_profit"
    duration_minutes: int
    agent_votes: Dict


class ThresholdBacktester:
    """
    Backtest ATLAS decisions against historical data to find optimal threshold.

    Process:
    1. Get historical price data
    2. Run ATLAS coordinator on each candle
    3. Simulate trades with fixed stop/take levels
    4. Track outcomes by score threshold
    5. Calculate optimal threshold
    """

    def __init__(self, pairs: List[str] = None):
        """Initialize backtester"""
        if pairs is None:
            pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

        self.pairs = pairs
        self.adapter = OandaAdapter(
            os.getenv('OANDA_API_KEY'),
            os.getenv('OANDA_ACCOUNT_ID'),
            practice=True
        )

        # Load coordinator config
        config_path = Path(__file__).parent / 'config' / 'hybrid_optimized.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.coordinator = ATLASCoordinator(config)

        # Initialize all agents
        self._initialize_agents(config)

        # Backtest parameters
        self.stop_loss_pips = 25
        self.take_profit_pips = 50

        self.trades = []

    def _initialize_agents(self, config: dict):
        """Initialize all ATLAS agents from config"""
        agents_config = config.get("agents", {})
        print(f"[Backtest] Initializing {len([a for a in agents_config.values() if a.get('enabled')])} agents...")

        # Technical Agent
        if agents_config.get("TechnicalAgent", {}).get("enabled"):
            tech_config = agents_config["TechnicalAgent"]
            self.coordinator.add_agent(
                TechnicalAgent(initial_weight=tech_config["initial_weight"]),
                is_veto=tech_config.get("is_veto", False)
            )
            print(f"  + TechnicalAgent (weight={tech_config['initial_weight']})")

        # Pattern Recognition Agent
        if agents_config.get("PatternRecognitionAgent", {}).get("enabled"):
            pattern_config = agents_config["PatternRecognitionAgent"]
            self.coordinator.add_agent(
                PatternRecognitionAgent(
                    initial_weight=pattern_config["initial_weight"],
                    min_pattern_samples=pattern_config["min_pattern_samples"]
                )
            )
            print(f"  + PatternRecognitionAgent (weight={pattern_config['initial_weight']})")

        # News Filter Agent
        if agents_config.get("NewsFilterAgent", {}).get("enabled"):
            news_config = agents_config["NewsFilterAgent"]
            self.coordinator.add_agent(
                NewsFilterAgent(initial_weight=news_config["initial_weight"]),
                is_veto=True
            )
            print(f"  + NewsFilterAgent (weight={news_config['initial_weight']}, VETO)")

        # Mean Reversion Agent
        if agents_config.get("MeanReversionAgent", {}).get("enabled"):
            mr_config = agents_config.get("MeanReversionAgent", {})
            self.coordinator.add_agent(
                MeanReversionAgent(initial_weight=mr_config.get("initial_weight", 1.5))
            )
            print(f"  + MeanReversionAgent (weight={mr_config.get('initial_weight', 1.5)})")

        # XGBoost ML Agent
        if agents_config.get("XGBoostMLAgent", {}).get("enabled"):
            xgb_config = agents_config.get("XGBoostMLAgent", {})
            self.coordinator.add_agent(
                XGBoostMLAgent(initial_weight=xgb_config.get("initial_weight", 2.5))
            )
            print(f"  + XGBoostMLAgent (weight={xgb_config.get('initial_weight', 2.5)})")

        # Sentiment Agent
        if agents_config.get("SentimentAgent", {}).get("enabled"):
            sent_config = agents_config.get("SentimentAgent", {})
            self.coordinator.add_agent(
                SentimentAgent(initial_weight=sent_config.get("initial_weight", 1.5))
            )
            print(f"  + SentimentAgent (weight={sent_config.get('initial_weight', 1.5)})")

        # Qlib Research Agent
        if agents_config.get("QlibResearchAgent", {}).get("enabled"):
            qlib_config = agents_config["QlibResearchAgent"]
            self.coordinator.add_agent(
                QlibResearchAgent(initial_weight=qlib_config["initial_weight"])
            )
            print(f"  + QlibResearchAgent (weight={qlib_config['initial_weight']})")

        # GS Quant Agent
        if agents_config.get("GSQuantAgent", {}).get("enabled"):
            gs_config = agents_config["GSQuantAgent"]
            self.coordinator.add_agent(
                GSQuantAgent(initial_weight=gs_config["initial_weight"])
            )
            print(f"  + GSQuantAgent (weight={gs_config['initial_weight']})")

        # Market Regime Agent
        if agents_config.get("MarketRegimeAgent", {}).get("enabled"):
            regime_config = agents_config.get("MarketRegimeAgent", {})
            self.coordinator.add_agent(
                MarketRegimeAgent(initial_weight=regime_config.get("initial_weight", 1.2))
            )
            print(f"  + MarketRegimeAgent (weight={regime_config.get('initial_weight', 1.2)})")

        # Risk Management Agent
        if agents_config.get("RiskManagementAgent", {}).get("enabled"):
            risk_config = agents_config.get("RiskManagementAgent", {})
            self.coordinator.add_agent(
                RiskManagementAgent(initial_weight=risk_config.get("initial_weight", 1.5))
            )
            print(f"  + RiskManagementAgent (weight={risk_config.get('initial_weight', 1.5)})")

        # Session Timing Agent
        if agents_config.get("SessionTimingAgent", {}).get("enabled"):
            session_config = agents_config.get("SessionTimingAgent", {})
            self.coordinator.add_agent(
                SessionTimingAgent(initial_weight=session_config.get("initial_weight", 1.2))
            )
            print(f"  + SessionTimingAgent (weight={session_config.get('initial_weight', 1.2)})")

        # Correlation Agent
        if agents_config.get("CorrelationAgent", {}).get("enabled"):
            corr_config = agents_config.get("CorrelationAgent", {})
            self.coordinator.add_agent(
                CorrelationAgent(initial_weight=corr_config.get("initial_weight", 1.0))
            )
            print(f"  + CorrelationAgent (weight={corr_config.get('initial_weight', 1.0)})")

        # Multi-Timeframe Agent
        if agents_config.get("MultiTimeframeAgent", {}).get("enabled"):
            mtf_config = agents_config.get("MultiTimeframeAgent", {})
            self.coordinator.add_agent(
                MultiTimeframeAgent(initial_weight=mtf_config.get("initial_weight", 2.0))
            )
            print(f"  + MultiTimeframeAgent (weight={mtf_config.get('initial_weight', 2.0)})")

        # Volume/Liquidity Agent
        if agents_config.get("VolumeLiquidityAgent", {}).get("enabled"):
            vol_config = agents_config.get("VolumeLiquidityAgent", {})
            self.coordinator.add_agent(
                VolumeLiquidityAgent(initial_weight=vol_config.get("initial_weight", 1.8))
            )
            print(f"  + VolumeLiquidityAgent (weight={vol_config.get('initial_weight', 1.8)})")

        # Support/Resistance Agent
        if agents_config.get("SupportResistanceAgent", {}).get("enabled"):
            sr_config = agents_config.get("SupportResistanceAgent", {})
            self.coordinator.add_agent(
                SupportResistanceAgent(initial_weight=sr_config.get("initial_weight", 1.7))
            )
            print(f"  + SupportResistanceAgent (weight={sr_config.get('initial_weight', 1.7)})")

        # Divergence Agent
        if agents_config.get("DivergenceAgent", {}).get("enabled"):
            div_config = agents_config.get("DivergenceAgent", {})
            self.coordinator.add_agent(
                DivergenceAgent(initial_weight=div_config.get("initial_weight", 1.6))
            )
            print(f"  + DivergenceAgent (weight={div_config.get('initial_weight', 1.6)})")

        print(f"[Backtest] Agent initialization complete\n")

    def run_backtest(self, days_back: int = 30, granularity: str = 'H1'):
        """
        Run backtest on historical data.

        Args:
            days_back: Number of days to backtest
            granularity: Candle granularity (H1 = 1 hour)
        """
        print("=" * 80)
        print("ATLAS THRESHOLD BACKTEST")
        print("=" * 80)
        print(f"\nBacktesting {days_back} days on {len(self.pairs)} pairs")
        print(f"Granularity: {granularity}")
        print(f"Stop Loss: {self.stop_loss_pips} pips")
        print(f"Take Profit: {self.take_profit_pips} pips")
        print()

        for pair in self.pairs:
            print(f"\n[{pair}] Fetching historical data...")

            # Get historical candles
            candles = self.adapter.get_candles(
                pair,
                timeframe=granularity,
                count=days_back * 24  # Approximate hourly candles
            )

            if not candles:
                print(f"  No data available for {pair}")
                continue

            print(f"  Loaded {len(candles)} candles")
            print(f"  Analyzing...")

            # Analyze each candle
            for i, candle in enumerate(candles[:-1]):  # Skip last candle (no exit data)
                try:
                    # Prepare market data
                    market_data = {
                        'pair': pair,
                        'price': candle['close'],
                        'time': candle['time'],
                        'indicators': self._calculate_indicators(candles[:i+1])
                    }

                    # Get ATLAS decision
                    decision = self.coordinator.analyze_opportunity(market_data)

                    # Skip if not BUY signal
                    if decision.get('decision') != 'BUY':
                        continue

                    score = decision.get('score', 0)

                    # Simulate trade execution
                    entry_price = float(candle['close'])
                    pip_value = 0.01 if 'JPY' in pair else 0.0001

                    stop_loss = entry_price - (self.stop_loss_pips * pip_value)
                    take_profit = entry_price + (self.take_profit_pips * pip_value)

                    # Find exit
                    exit_result = self._find_exit(
                        candles[i+1:],
                        entry_price,
                        stop_loss,
                        take_profit,
                        pip_value
                    )

                    if exit_result:
                        exit_price, exit_reason, duration = exit_result

                        # Calculate P/L (simplified, doesn't account for lot size)
                        pips = (exit_price - entry_price) / pip_value
                        # Approximate P/L for 1 standard lot
                        pnl = pips * (10 if 'JPY' in pair else 100)

                        trade = BacktestTrade(
                            timestamp=candle['time'],
                            pair=pair,
                            atlas_score=score,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl=pnl,
                            exit_reason=exit_reason,
                            duration_minutes=duration,
                            agent_votes=decision.get('agent_votes', {})
                        )

                        self.trades.append(trade)

                except Exception as e:
                    print(f"  Error analyzing candle {i}: {e}")
                    continue

            print(f"  Simulated {len([t for t in self.trades if t.pair == pair])} trades")

        print(f"\n{'='*80}")
        print(f"Backtest complete: {len(self.trades)} total trades simulated")
        print(f"{'='*80}\n")

    def _calculate_indicators(self, candles: List[Dict]) -> Dict:
        """Calculate technical indicators from candle data"""
        if len(candles) < 200:
            return {
                'rsi': 50,
                'macd': 0,
                'adx': 20,
                'ema50': 0,
                'ema200': 0
            }

        # Simplified - use closing prices
        closes = [float(c['close']) for c in candles]

        # Simple RSI calculation
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [c if c > 0 else 0 for c in changes[-14:]]
        losses = [-c if c < 0 else 0 for c in changes[-14:]]

        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Simple EMAs
        ema50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]
        ema200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else closes[-1]

        # Simplified MACD
        ema12 = sum(closes[-12:]) / 12 if len(closes) >= 12 else closes[-1]
        ema26 = sum(closes[-26:]) / 26 if len(closes) >= 26 else closes[-1]
        macd = ema12 - ema26

        # Simplified ADX (just use volatility proxy)
        if len(closes) >= 14:
            highs = [float(c['high']) for c in candles[-14:]]
            lows = [float(c['low']) for c in candles[-14:]]
            tr = [highs[i] - lows[i] for i in range(len(highs))]
            atr = sum(tr) / len(tr)
            adx = min(100, (atr / closes[-1]) * 1000)
        else:
            adx = 20

        return {
            'rsi': rsi,
            'macd': macd,
            'adx': adx,
            'ema50': ema50,
            'ema200': ema200
        }

    def _find_exit(self, future_candles: List[Dict], entry_price: float,
                   stop_loss: float, take_profit: float, pip_value: float) -> Tuple:
        """
        Find when trade would exit (stop or take profit hit).

        Returns:
            (exit_price, exit_reason, duration_minutes) or None
        """
        for i, candle in enumerate(future_candles):
            low = float(candle['low'])
            high = float(candle['high'])

            # Check stop loss first (more conservative)
            if low <= stop_loss:
                return (stop_loss, 'stop_loss', i * 60)  # Assume hourly = 60 min

            # Check take profit
            if high >= take_profit:
                return (take_profit, 'take_profit', i * 60)

        # If neither hit in remaining data, assume stop loss (conservative)
        return (stop_loss, 'timeout_stop', len(future_candles) * 60)

    def analyze_thresholds(self, thresholds: List[float] = None) -> Dict:
        """
        Analyze performance at different thresholds.

        Args:
            thresholds: List of thresholds to test

        Returns:
            Dictionary of results by threshold
        """
        if thresholds is None:
            thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

        print("\n" + "=" * 80)
        print("THRESHOLD PERFORMANCE ANALYSIS")
        print("=" * 80)
        print(f"\n{'Thresh':<7} {'Trades':<7} {'Wins':<6} {'Loss':<6} {'WR%':<7} {'Net P/L':<10} {'PF':<6} {'Expect':<8}")
        print("-" * 80)

        results = {}

        for threshold in thresholds:
            # Filter trades by threshold
            qualified = [t for t in self.trades if t.atlas_score >= threshold]

            if not qualified:
                continue

            # Calculate stats
            wins = [t for t in qualified if t.pnl > 0]
            losses = [t for t in qualified if t.pnl <= 0]

            total_profit = sum(t.pnl for t in wins)
            total_loss = sum(t.pnl for t in losses)
            net_pnl = total_profit + total_loss

            win_rate = len(wins) / len(qualified) * 100 if qualified else 0
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
            expectancy = net_pnl / len(qualified) if qualified else 0

            results[threshold] = {
                'trades': len(qualified),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': win_rate,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net_pnl': net_pnl,
                'profit_factor': profit_factor,
                'expectancy': expectancy
            }

            print(f"{threshold:<7.1f} "
                  f"{len(qualified):<7} "
                  f"{len(wins):<6} "
                  f"{len(losses):<6} "
                  f"{win_rate:<7.1f} "
                  f"${net_pnl:<9,.0f} "
                  f"{profit_factor:<6.2f} "
                  f"${expectancy:<7,.0f}")

        return results

    def get_recommendation(self, results: Dict) -> Dict:
        """Get optimal threshold recommendation"""
        if not results:
            return {"error": "No results to analyze"}

        # Filter to thresholds with meaningful sample size
        valid = {k: v for k, v in results.items() if v['trades'] >= 20}

        if not valid:
            print("\nWARNING: No thresholds have 20+ trades. Results may not be statistically significant.")
            valid = results

        # Find best by different metrics
        best_pnl = max(valid.items(), key=lambda x: x[1]['net_pnl'])
        best_wr = max(valid.items(), key=lambda x: x[1]['win_rate'])

        pf_valid = {k: v for k, v in valid.items() if v['profit_factor'] != float('inf')}
        best_pf = max(pf_valid.items(), key=lambda x: x[1]['profit_factor']) if pf_valid else None

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        print(f"\n[BEST NET P/L] Threshold: {best_pnl[0]}")
        print(f"  - Net P/L: ${best_pnl[1]['net_pnl']:,.2f}")
        print(f"  - Win Rate: {best_pnl[1]['win_rate']:.1f}%")
        print(f"  - Trades: {best_pnl[1]['trades']}")

        print(f"\n[BEST WIN RATE] Threshold: {best_wr[0]}")
        print(f"  - Win Rate: {best_wr[1]['win_rate']:.1f}%")
        print(f"  - Net P/L: ${best_wr[1]['net_pnl']:,.2f}")
        print(f"  - Trades: {best_wr[1]['trades']}")

        if best_pf:
            print(f"\n[BEST PROFIT FACTOR] Threshold: {best_pf[0]}")
            print(f"  - Profit Factor: {best_pf[1]['profit_factor']:.2f}")
            print(f"  - Net P/L: ${best_pf[1]['net_pnl']:,.2f}")
            print(f"  - Win Rate: {best_pf[1]['win_rate']:.1f}%")

        # Check current threshold
        current_threshold = 1.5
        if current_threshold in results:
            current = results[current_threshold]
            print(f"\n[CURRENT THRESHOLD 1.5]:")
            print(f"  - Net P/L: ${current['net_pnl']:,.2f}")
            print(f"  - Win Rate: {current['win_rate']:.1f}%")
            print(f"  - Expectancy: ${current['expectancy']:,.2f} per trade")
            print(f"  - Trades: {current['trades']}")

            if current['net_pnl'] < 0:
                print(f"  WARNING: LOSING MONEY at threshold 1.5!")

        print("\n" + "=" * 80)

        return {
            'best_pnl_threshold': best_pnl[0],
            'best_wr_threshold': best_wr[0],
            'best_pf_threshold': best_pf[0] if best_pf else None,
            'current_performance': results.get(current_threshold, {})
        }


def main():
    """Run complete backtest and threshold analysis"""
    backtester = ThresholdBacktester()

    # Run backtest
    backtester.run_backtest(days_back=30, granularity='H1')

    if not backtester.trades:
        print("No trades generated in backtest!")
        return

    # Analyze thresholds
    results = backtester.analyze_thresholds()

    # Get recommendation
    recommendation = backtester.get_recommendation(results)

    # Save results
    output_file = 'backtest_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'backtest_params': {
                'pairs': backtester.pairs,
                'stop_loss_pips': backtester.stop_loss_pips,
                'take_profit_pips': backtester.take_profit_pips,
                'total_trades': len(backtester.trades)
            },
            'results_by_threshold': results,
            'recommendation': recommendation
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
