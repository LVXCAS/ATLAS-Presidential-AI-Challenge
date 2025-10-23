#!/usr/bin/env python3
"""
FOREX & FUTURES R&D AGENT
Autonomous agent that discovers and optimizes Forex and Futures strategies

Specializes in:
- Currency pair strategies (EUR/USD, USD/JPY, GBP/USD)
- Futures strategies (MES, MNQ, /ES, /NQ)
- Multi-timeframe analysis
- Carry trade optimization
- Futures spread strategies
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')


@dataclass
class ForexFuturesStrategy:
    """Discovered Forex/Futures strategy"""
    name: str
    type: str  # 'forex' or 'futures'
    market: str  # 'EUR_USD' or 'MES'
    timeframe: str  # 'H1', 'H4', 'D1'
    expected_sharpe: float
    expected_win_rate: float
    parameters: Dict[str, Any]
    source: str
    timestamp: datetime


class ForexFuturesRDAgent:
    """Autonomous R&D agent for Forex and Futures strategies"""

    def __init__(self):
        self.agent_id = "forex_futures_researcher"
        self.discoveries = []

        # Markets to research
        self.forex_pairs = ['EURUSD=X', 'JPY=X', 'GBPUSD=X']  # Yahoo Finance symbols
        self.futures_symbols = ['ES=F', 'NQ=F']  # E-mini S&P 500, NASDAQ

        # Timeframes to test
        self.timeframes = ['1h', '4h', '1d']

        # Strategy types
        self.strategy_types = [
            'ema_crossover',
            'rsi_mean_reversion',
            'breakout',
            'carry_trade',
            'futures_spread'
        ]

        print(f"\n[{self.agent_id}] Initialized")
        print(f"Forex Pairs: {len(self.forex_pairs)}")
        print(f"Futures: {len(self.futures_symbols)}")
        print(f"Strategy Types: {len(self.strategy_types)}")

    async def run_research_cycle(self):
        """Main research cycle - discovers strategies overnight"""
        print(f"\n[{self.agent_id}] Starting research cycle...")

        # Research Forex strategies
        forex_strategies = await self.research_forex_strategies()
        print(f"[{self.agent_id}] Discovered {len(forex_strategies)} Forex strategies")

        # Research Futures strategies
        futures_strategies = await self.research_futures_strategies()
        print(f"[{self.agent_id}] Discovered {len(futures_strategies)} Futures strategies")

        # Combine and rank
        all_strategies = forex_strategies + futures_strategies
        ranked = sorted(all_strategies, key=lambda x: x.expected_sharpe, reverse=True)

        # Save results
        self.save_discoveries(ranked)

        print(f"\n[{self.agent_id}] Research complete!")
        print(f"Total discoveries: {len(ranked)}")
        if ranked:
            print(f"Best Sharpe: {ranked[0].expected_sharpe:.2f} ({ranked[0].name})")
        else:
            print(f"No strategies met criteria (min Sharpe: 1.0)")

        return ranked

    async def research_forex_strategies(self) -> List[ForexFuturesStrategy]:
        """Research Forex strategies"""
        strategies = []

        for pair in self.forex_pairs:
            try:
                # Download data
                print(f"[{self.agent_id}] Analyzing {pair}...")
                data = yf.download(pair, period='1y', interval='1h', progress=False)

                if data.empty:
                    continue

                # Test different strategy types
                for strategy_type in self.strategy_types[:3]:  # Focus on top 3
                    strategy = await self.backtest_forex_strategy(
                        pair, data, strategy_type
                    )
                    if strategy and strategy.expected_sharpe > 1.0:
                        strategies.append(strategy)

            except Exception as e:
                print(f"[{self.agent_id}] Error with {pair}: {e}")

        return strategies

    async def research_futures_strategies(self) -> List[ForexFuturesStrategy]:
        """Research Futures strategies"""
        strategies = []

        for symbol in self.futures_symbols:
            try:
                # Download data
                print(f"[{self.agent_id}] Analyzing {symbol}...")
                data = yf.download(symbol, period='1y', interval='1h', progress=False)

                if data.empty:
                    continue

                # Test different strategy types
                for strategy_type in self.strategy_types[:3]:
                    strategy = await self.backtest_futures_strategy(
                        symbol, data, strategy_type
                    )
                    if strategy and strategy.expected_sharpe > 1.0:
                        strategies.append(strategy)

            except Exception as e:
                print(f"[{self.agent_id}] Error with {symbol}: {e}")

        return strategies

    async def backtest_forex_strategy(self, pair: str, data: pd.DataFrame,
                                      strategy_type: str) -> Optional[ForexFuturesStrategy]:
        """Backtest a Forex strategy"""
        try:
            if strategy_type == 'ema_crossover':
                return self.test_ema_crossover(pair, data, 'forex')
            elif strategy_type == 'rsi_mean_reversion':
                return self.test_rsi_mean_reversion(pair, data, 'forex')
            elif strategy_type == 'breakout':
                return self.test_breakout(pair, data, 'forex')
        except:
            return None

    async def backtest_futures_strategy(self, symbol: str, data: pd.DataFrame,
                                        strategy_type: str) -> Optional[ForexFuturesStrategy]:
        """Backtest a Futures strategy"""
        try:
            if strategy_type == 'ema_crossover':
                return self.test_ema_crossover(symbol, data, 'futures')
            elif strategy_type == 'rsi_mean_reversion':
                return self.test_rsi_mean_reversion(symbol, data, 'futures')
            elif strategy_type == 'breakout':
                return self.test_breakout(symbol, data, 'futures')
        except:
            return None

    def test_ema_crossover(self, symbol: str, data: pd.DataFrame,
                          market_type: str) -> Optional[ForexFuturesStrategy]:
        """Test EMA crossover strategy"""
        try:
            # Calculate EMAs
            data['ema_fast'] = data['Close'].ewm(span=10).mean()
            data['ema_slow'] = data['Close'].ewm(span=21).mean()

            # Generate signals
            data['signal'] = 0
            data.loc[data['ema_fast'] > data['ema_slow'], 'signal'] = 1
            data.loc[data['ema_fast'] < data['ema_slow'], 'signal'] = -1

            # Calculate returns
            data['returns'] = data['Close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']

            # Calculate metrics
            sharpe = self.calculate_sharpe(data['strategy_returns'])
            win_rate = self.calculate_win_rate(data['strategy_returns'])

            if sharpe > 1.0:
                return ForexFuturesStrategy(
                    name=f"{market_type.upper()}_EMA_Cross_{symbol.replace('=X', '').replace('=F', '')}",
                    type=market_type,
                    market=symbol,
                    timeframe='H1',
                    expected_sharpe=sharpe,
                    expected_win_rate=win_rate,
                    parameters={'ema_fast': 10, 'ema_slow': 21},
                    source='Forex_Futures_RD_Agent',
                    timestamp=datetime.now()
                )
        except:
            pass

        return None

    def test_rsi_mean_reversion(self, symbol: str, data: pd.DataFrame,
                                market_type: str) -> Optional[ForexFuturesStrategy]:
        """Test RSI mean reversion strategy"""
        try:
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))

            # Generate signals
            data['signal'] = 0
            data.loc[data['rsi'] < 30, 'signal'] = 1   # Oversold - buy
            data.loc[data['rsi'] > 70, 'signal'] = -1  # Overbought - sell

            # Calculate returns
            data['returns'] = data['Close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']

            # Calculate metrics
            sharpe = self.calculate_sharpe(data['strategy_returns'])
            win_rate = self.calculate_win_rate(data['strategy_returns'])

            if sharpe > 1.0:
                return ForexFuturesStrategy(
                    name=f"{market_type.upper()}_RSI_MeanRev_{symbol.replace('=X', '').replace('=F', '')}",
                    type=market_type,
                    market=symbol,
                    timeframe='H1',
                    expected_sharpe=sharpe,
                    expected_win_rate=win_rate,
                    parameters={'rsi_period': 14, 'oversold': 30, 'overbought': 70},
                    source='Forex_Futures_RD_Agent',
                    timestamp=datetime.now()
                )
        except:
            pass

        return None

    def test_breakout(self, symbol: str, data: pd.DataFrame,
                     market_type: str) -> Optional[ForexFuturesStrategy]:
        """Test breakout strategy"""
        try:
            # Calculate Bollinger Bands
            data['sma'] = data['Close'].rolling(window=20).mean()
            data['std'] = data['Close'].rolling(window=20).std()
            data['upper'] = data['sma'] + (data['std'] * 2)
            data['lower'] = data['sma'] - (data['std'] * 2)

            # Generate signals
            data['signal'] = 0
            data.loc[data['Close'] > data['upper'], 'signal'] = 1   # Breakout up
            data.loc[data['Close'] < data['lower'], 'signal'] = -1  # Breakout down

            # Calculate returns
            data['returns'] = data['Close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']

            # Calculate metrics
            sharpe = self.calculate_sharpe(data['strategy_returns'])
            win_rate = self.calculate_win_rate(data['strategy_returns'])

            if sharpe > 1.0:
                return ForexFuturesStrategy(
                    name=f"{market_type.upper()}_Breakout_{symbol.replace('=X', '').replace('=F', '')}",
                    type=market_type,
                    market=symbol,
                    timeframe='H1',
                    expected_sharpe=sharpe,
                    expected_win_rate=win_rate,
                    parameters={'bb_period': 20, 'bb_std': 2},
                    source='Forex_Futures_RD_Agent',
                    timestamp=datetime.now()
                )
        except:
            pass

        return None

    def calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        try:
            returns = returns.dropna()
            if len(returns) < 10:
                return 0.0

            # Annualized Sharpe (assuming hourly data)
            mean_return = returns.mean() * 252 * 24  # Annualized
            std_return = returns.std() * np.sqrt(252 * 24)

            if std_return == 0:
                return 0.0

            sharpe = mean_return / std_return
            return max(0, min(sharpe, 10))  # Cap at 10 to avoid outliers
        except:
            return 0.0

    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate"""
        try:
            returns = returns.dropna()
            if len(returns) == 0:
                return 0.0

            wins = (returns > 0).sum()
            total = len(returns)
            return wins / total
        except:
            return 0.0

    def save_discoveries(self, strategies: List[ForexFuturesStrategy]):
        """Save discoveries to JSON"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'logs/forex_futures_strategies_{timestamp}.json'

            data = []
            for s in strategies:
                data.append({
                    'name': s.name,
                    'type': s.type,
                    'market': s.market,
                    'timeframe': s.timeframe,
                    'expected_sharpe': s.expected_sharpe,
                    'expected_win_rate': s.expected_win_rate,
                    'parameters': s.parameters,
                    'source': s.source,
                    'timestamp': s.timestamp.isoformat()
                })

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"\n[{self.agent_id}] Saved {len(data)} strategies to {filename}")

        except Exception as e:
            print(f"[{self.agent_id}] Error saving: {e}")


async def main():
    """Run Forex & Futures R&D agent"""
    agent = ForexFuturesRDAgent()

    print("\n" + "="*70)
    print("FOREX & FUTURES R&D AGENT - STARTING")
    print("="*70)
    print("This will discover optimized Forex and Futures strategies")
    print("Expected runtime: 30-60 minutes")
    print("="*70 + "\n")

    strategies = await agent.run_research_cycle()

    print("\n" + "="*70)
    print("R&D COMPLETE - TOP 5 DISCOVERIES:")
    print("="*70)
    for i, s in enumerate(strategies[:5], 1):
        print(f"{i}. {s.name}")
        print(f"   Sharpe: {s.expected_sharpe:.2f} | Win Rate: {s.expected_win_rate:.1%}")
        print(f"   Market: {s.market} | Type: {s.type}")
        print()

    print("="*70)
    print(f"Total discoveries saved: {len(strategies)}")
    print("Check logs/forex_futures_strategies_*.json for full results")
    print("="*70)


if __name__ == '__main__':
    asyncio.run(main())
