"""
Hybrid R&D System - Best of Both Worlds

- Uses yfinance for historical research & backtesting (years of data)
- Uses Alpaca for real-time validation & execution (current market)
- Automatically validates yfinance discoveries against live Alpaca data before deployment

This is the production-grade approach:
1. R&D agents research with historical data (yfinance)
2. Discovered strategies validated with current market (Alpaca)
3. Only deploy strategies that pass both historical AND live validation
"""

import asyncio
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import yfinance as yf

load_dotenv('.env.paper')

class HybridDataProvider:
    """Hybrid data provider - historical from yfinance, real-time from Alpaca"""

    def __init__(self):
        self.alpaca_api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        print("[DATA] Hybrid data provider initialized")
        print("  Historical Research: yfinance (unlimited history)")
        print("  Live Validation: Alpaca (real-time data)")

    def get_historical_data(self, symbol: str, period: str = '6mo') -> pd.DataFrame:
        """Get historical data for research (yfinance)"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if not data.empty:
                print(f"[RESEARCH] Retrieved {len(data)} historical bars for {symbol}")
                return data
        except Exception as e:
            print(f"[RESEARCH] Error fetching historical {symbol}: {e}")
        return pd.DataFrame()

    def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """Get live market data for validation (Alpaca)"""
        try:
            bars = self.alpaca_api.get_bars(symbol, '1Day', limit=1).df
            if not bars.empty:
                return {
                    'price': float(bars['close'].iloc[-1]),
                    'volume': float(bars['volume'].iloc[-1]),
                    'high': float(bars['high'].iloc[-1]),
                    'low': float(bars['low'].iloc[-1]),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'ALPACA_LIVE'
                }
        except Exception as e:
            print(f"[LIVE] Error fetching {symbol}: {e}")
        return {}

class HybridStrategyResearcher:
    """Research strategies with historical data, validate with live data"""

    def __init__(self, data_provider: HybridDataProvider):
        self.data_provider = data_provider
        self.discovered_strategies = []
        print("[RESEARCHER] Hybrid strategy researcher initialized")

    async def research_and_validate_strategies(self) -> List[Dict[str, Any]]:
        """Full pipeline: research → backtest → live validation"""

        print("\n" + "="*70)
        print("STRATEGY RESEARCH & VALIDATION PIPELINE")
        print("="*70)

        validated_strategies = []

        # Step 1: Research momentum strategies with historical data
        momentum_strategies = await self.research_momentum_strategies()

        # Step 2: Validate against current market
        for strategy in momentum_strategies:
            if await self.validate_strategy_live(strategy):
                validated_strategies.append(strategy)

        # Step 3: Research volatility strategies
        vol_strategies = await self.research_volatility_strategies()

        for strategy in vol_strategies:
            if await self.validate_strategy_live(strategy):
                validated_strategies.append(strategy)

        return validated_strategies

    async def research_momentum_strategies(self) -> List[Dict[str, Any]]:
        """Research momentum using historical data"""
        print("\n[RESEARCH] Analyzing momentum strategies (historical data)...")

        strategies = []
        symbols = ['INTC', 'AMD', 'NVDA', 'AAPL', 'MSFT']

        for symbol in symbols:
            data = self.data_provider.get_historical_data(symbol, '6mo')

            if data.empty or len(data) < 60:
                continue

            # Calculate momentum metrics
            returns = data['Close'].pct_change()
            momentum_5d = float(returns.tail(5).mean())
            momentum_20d = float(returns.tail(20).mean())
            momentum_60d = float(returns.tail(60).mean())
            volatility = float(returns.std())

            # Momentum score
            momentum_score = (momentum_5d * 0.5 + momentum_20d * 0.3 + momentum_60d * 0.2) / volatility

            # Backtest simple momentum strategy
            backtest_return = self.backtest_momentum(data, momentum_score)

            strategy = {
                'symbol': symbol,
                'type': 'momentum',
                'momentum_score': momentum_score,
                'historical_return': backtest_return,
                'momentum_5d': momentum_5d,
                'momentum_20d': momentum_20d,
                'volatility': volatility,
                'research_timestamp': datetime.now().isoformat()
            }

            if backtest_return > 0.05:  # Positive historical performance
                strategies.append(strategy)
                print(f"  [DISCOVERED] {symbol}: Momentum={momentum_score:.3f}, Return={backtest_return:.1%}")

        return strategies

    def backtest_momentum(self, data: pd.DataFrame, momentum_score: float) -> float:
        """Simple backtest of momentum strategy"""
        try:
            returns = data['Close'].pct_change()
            # Simple: long if positive momentum, short if negative
            signal = 1 if momentum_score > 0 else -1
            strategy_returns = returns * signal
            cumulative_return = (1 + strategy_returns).prod() - 1
            return float(cumulative_return)
        except:
            return 0.0

    async def research_volatility_strategies(self) -> List[Dict[str, Any]]:
        """Research volatility strategies"""
        print("\n[RESEARCH] Analyzing volatility strategies (historical data)...")

        strategies = []
        symbols = ['INTC', 'AMD', 'NVDA', 'TSLA']

        for symbol in symbols:
            data = self.data_provider.get_historical_data(symbol, '6mo')

            if data.empty or len(data) < 60:
                continue

            returns = data['Close'].pct_change()
            realized_vol = float(returns.std())
            vol_20d = float(returns.tail(20).std())
            vol_60d = float(returns.tail(60).std())

            # Options-friendly: look for elevated volatility
            vol_percentile = self.calculate_vol_percentile(returns)

            strategy = {
                'symbol': symbol,
                'type': 'volatility',
                'realized_vol': realized_vol,
                'vol_20d': vol_20d,
                'vol_60d': vol_60d,
                'vol_percentile': vol_percentile,
                'research_timestamp': datetime.now().isoformat()
            }

            if vol_percentile > 60:  # Elevated volatility (good for options)
                strategies.append(strategy)
                print(f"  [DISCOVERED] {symbol}: Vol={realized_vol:.3f} ({vol_percentile:.0f}th percentile)")

        return strategies

    def calculate_vol_percentile(self, returns: pd.Series) -> float:
        """Calculate current volatility percentile"""
        rolling_vol = returns.rolling(20).std()
        current_vol = rolling_vol.iloc[-1]
        percentile = (rolling_vol < current_vol).sum() / len(rolling_vol) * 100
        return float(percentile)

    async def validate_strategy_live(self, strategy: Dict[str, Any]) -> bool:
        """Validate strategy against LIVE market data"""
        symbol = strategy['symbol']
        live_data = self.data_provider.get_live_data(symbol)

        if not live_data:
            print(f"  [VALIDATION] {symbol}: No live data - REJECTED")
            return False

        # Validate that current price is reasonable
        if live_data['price'] < 5:  # Too cheap
            print(f"  [VALIDATION] {symbol}: Price too low ${live_data['price']:.2f} - REJECTED")
            return False

        # Validate volume
        if live_data['volume'] < 100000:  # Too illiquid
            print(f"  [VALIDATION] {symbol}: Volume too low - REJECTED")
            return False

        # Strategy passed live validation
        strategy['live_validation'] = {
            'validated': True,
            'live_price': live_data['price'],
            'live_volume': live_data['volume'],
            'validation_timestamp': datetime.now().isoformat()
        }

        print(f"  [VALIDATED] {symbol}: Live ${live_data['price']:.2f}, Vol {live_data['volume']:,.0f} - APPROVED")
        return True

class HybridRDOrchestrator:
    """Orchestrates hybrid R&D system"""

    def __init__(self):
        self.data_provider = HybridDataProvider()
        self.researcher = HybridStrategyResearcher(self.data_provider)
        print("[ORCHESTRATOR] Hybrid R&D orchestrator initialized")

    async def run_full_rd_cycle(self):
        """Run complete R&D cycle"""

        print("\n" + "="*70)
        print("HYBRID R&D CYCLE - RESEARCH > VALIDATE > DEPLOY")
        print("="*70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Research and validate
        validated_strategies = await self.researcher.research_and_validate_strategies()

        # Prepare deployment package
        deployment_package = {
            'timestamp': datetime.now().isoformat(),
            'total_strategies_researched': len(self.researcher.discovered_strategies),
            'validated_strategies': len(validated_strategies),
            'strategies': validated_strategies,
            'ready_for_deployment': len(validated_strategies) > 0
        }

        # Save results
        filename = f"rd_validated_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(deployment_package, f, indent=2)

        print("\n" + "="*70)
        print("R&D CYCLE COMPLETE")
        print("="*70)
        print(f"Strategies Validated: {len(validated_strategies)}")
        print(f"Deployment Package: {filename}")

        if validated_strategies:
            print("\nREADY FOR DEPLOYMENT:")
            for strategy in validated_strategies:
                symbol = strategy['symbol']
                stype = strategy['type']
                price = strategy.get('live_validation', {}).get('live_price', 'N/A')
                print(f"  - {symbol} ({stype}) @ ${price}")

        return deployment_package

async def main():
    """Launch hybrid R&D system"""

    print("="*70)
    print("HYBRID AUTONOMOUS R&D SYSTEM")
    print("="*70)
    print("Architecture:")
    print("  1. Research with historical data (yfinance - unlimited history)")
    print("  2. Backtest discovered strategies")
    print("  3. Validate against live market (Alpaca - real-time)")
    print("  4. Generate deployment-ready strategies")
    print("="*70)

    orchestrator = HybridRDOrchestrator()
    deployment_package = await orchestrator.run_full_rd_cycle()

    print("\n" + "="*70)
    print("SYSTEM STATUS: OPERATIONAL")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
