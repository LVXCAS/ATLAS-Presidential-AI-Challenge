"""
AUTONOMOUS STOCK DISCOVERY & OPTIONS SYSTEM
Continuously discovers new stocks and deploys options strategies
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_options_discovery.log'),
        logging.StreamHandler()
    ]
)

class AutonomousOptionsDiscovery:
    """Discover new stocks and deploy options strategies autonomously"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Discovery criteria
        self.discovery_criteria = {
            'min_market_cap': 1e9,         # $1B minimum market cap
            'min_volume': 1e6,             # 1M minimum daily volume
            'max_price': 200,              # Under $200 per share
            'min_implied_vol': 0.25,       # 25% minimum implied volatility
            'max_implied_vol': 1.0,        # 100% maximum implied volatility
            'earning_days_threshold': 30,   # Within 30 days of earnings
            'momentum_threshold': 0.05,     # 5% momentum in last 5 days
        }

        # Options strategy parameters
        self.options_strategies = {
            'covered_calls': {
                'enabled': True,
                'dte_range': (7, 45),       # Days to expiration
                'delta_range': (0.15, 0.30), # OTM call delta
                'max_allocation': 0.02       # 2% of portfolio max
            },
            'cash_secured_puts': {
                'enabled': True,
                'dte_range': (7, 30),
                'delta_range': (0.15, 0.25),
                'max_allocation': 0.02
            },
            'iron_condors': {
                'enabled': True,
                'dte_range': (14, 45),
                'width': 5,                  # Strike width
                'max_allocation': 0.01
            }
        }

        self.discovered_opportunities = []

    async def discover_high_volatility_stocks(self):
        """Discover stocks with high options volatility"""

        logging.info("Starting autonomous stock discovery...")

        # Popular high-volume tickers for options trading
        candidate_tickers = [
            # Tech stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
            # Meme/high vol stocks
            'GME', 'AMC', 'PLTR', 'RIVN', 'LCID', 'COIN', 'HOOD',
            # ETFs with options
            'ARKK', 'XLF', 'XBI', 'GDX', 'USO', 'UVXY',
            # High beta stocks
            'BABA', 'NFLX', 'CRM', 'ZM', 'ROKU', 'SQ', 'UBER'
        ]

        discovered = []

        for ticker in candidate_tickers:
            try:
                # Get basic stock data
                stock = yf.Ticker(ticker)
                info = stock.info

                if not info:
                    continue

                # Basic screening criteria
                market_cap = info.get('marketCap', 0)
                price = info.get('currentPrice', 0)
                volume = info.get('averageVolume', 0)

                if (market_cap < self.discovery_criteria['min_market_cap'] or
                    price > self.discovery_criteria['max_price'] or
                    volume < self.discovery_criteria['min_volume']):
                    continue

                # Get recent price data for momentum
                hist = stock.history(period='1mo')
                if len(hist) < 5:
                    continue

                current_price = hist['Close'].iloc[-1]
                week_ago_price = hist['Close'].iloc[-5]
                momentum = (current_price / week_ago_price - 1)

                # Get options data for volatility analysis
                options_dates = stock.options
                if not options_dates:
                    continue

                # Find suitable expiration (skip 0 DTE)
                suitable_exp = None
                for exp_date in options_dates:
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_datetime - datetime.now()).days
                    if dte >= 7:  # At least 7 days
                        suitable_exp = exp_date
                        break

                if not suitable_exp:
                    continue

                calls = stock.option_chain(suitable_exp).calls

                if calls.empty:
                    continue

                # Calculate average implied volatility
                avg_iv = calls['impliedVolatility'].mean()

                # Check criteria
                if (avg_iv >= self.discovery_criteria['min_implied_vol'] and
                    avg_iv <= self.discovery_criteria['max_implied_vol'] and
                    abs(momentum) >= self.discovery_criteria['momentum_threshold']):

                    opportunity = {
                        'ticker': ticker,
                        'price': float(current_price),
                        'market_cap': market_cap,
                        'avg_volume': volume,
                        'momentum_5d': float(momentum),
                        'implied_volatility': float(avg_iv),
                        'options_available': True,
                        'discovery_time': datetime.now().isoformat(),
                        'score': self.calculate_opportunity_score(momentum, avg_iv, volume)
                    }

                    discovered.append(opportunity)
                    logging.info(f"Discovered: {ticker} - IV: {avg_iv:.1%}, Momentum: {momentum:+.1%}")

            except Exception as e:
                logging.debug(f"Error analyzing {ticker}: {e}")
                continue

        # Sort by opportunity score
        discovered.sort(key=lambda x: x['score'], reverse=True)

        logging.info(f"Discovery complete: {len(discovered)} opportunities found")
        return discovered[:10]  # Return top 10

    def calculate_opportunity_score(self, momentum, implied_vol, volume):
        """Calculate opportunity score for ranking"""

        # Score based on volatility (higher = better)
        vol_score = min(implied_vol * 100, 50)  # Cap at 50

        # Score based on momentum (absolute value)
        momentum_score = min(abs(momentum) * 100, 20)  # Cap at 20

        # Score based on volume (liquidity)
        volume_score = min(volume / 1e6, 30)  # Cap at 30

        return vol_score + momentum_score + volume_score

    async def analyze_options_strategies(self, opportunities):
        """Analyze specific options strategies for discovered stocks"""

        logging.info("Analyzing options strategies...")

        strategy_recommendations = []

        for opp in opportunities:
            ticker = opp['ticker']
            price = opp['price']

            try:
                stock = yf.Ticker(ticker)

                # Get options chain
                exp_dates = stock.options
                if not exp_dates:
                    continue

                # Focus on nearest 2 expirations
                for exp_date in exp_dates[:2]:
                    chain = stock.option_chain(exp_date)
                    calls = chain.calls
                    puts = chain.puts

                    if calls.empty or puts.empty:
                        continue

                    # Days to expiration
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_datetime - datetime.now()).days

                    # Covered call analysis
                    if self.options_strategies['covered_calls']['enabled']:
                        cc_opportunities = self.find_covered_call_opportunities(
                            calls, price, dte, ticker
                        )
                        strategy_recommendations.extend(cc_opportunities)

                    # Cash-secured put analysis
                    if self.options_strategies['cash_secured_puts']['enabled']:
                        csp_opportunities = self.find_csp_opportunities(
                            puts, price, dte, ticker
                        )
                        strategy_recommendations.extend(csp_opportunities)

            except Exception as e:
                logging.debug(f"Error analyzing options for {ticker}: {e}")
                continue

        # Sort by expected return
        strategy_recommendations.sort(key=lambda x: x.get('expected_return', 0), reverse=True)

        return strategy_recommendations[:5]  # Top 5 strategies

    def find_covered_call_opportunities(self, calls, stock_price, dte, ticker):
        """Find covered call opportunities"""

        opportunities = []

        if not (self.options_strategies['covered_calls']['dte_range'][0] <= dte <=
                self.options_strategies['covered_calls']['dte_range'][1]):
            return opportunities

        # Find OTM calls with good premium
        otm_calls = calls[calls['strike'] > stock_price * 1.02]  # At least 2% OTM

        for _, call in otm_calls.head(5).iterrows():
            try:
                strike = call['strike']
                premium = call['lastPrice']
                delta = call.get('delta', 0.2)

                if (self.options_strategies['covered_calls']['delta_range'][0] <=
                    abs(delta) <= self.options_strategies['covered_calls']['delta_range'][1]):

                    # Calculate returns
                    cost_basis = stock_price
                    max_profit = (strike - cost_basis) + premium
                    max_return = (max_profit / cost_basis) * (365 / dte)  # Annualized

                    if max_return > 0.15:  # 15% annualized minimum
                        opportunities.append({
                            'strategy': 'covered_call',
                            'ticker': ticker,
                            'stock_price': stock_price,
                            'strike': strike,
                            'premium': premium,
                            'dte': dte,
                            'expected_return': max_return,
                            'delta': delta,
                            'allocation_required': cost_basis,
                            'max_profit': max_profit
                        })

            except:
                continue

        return opportunities

    def find_csp_opportunities(self, puts, stock_price, dte, ticker):
        """Find cash-secured put opportunities"""

        opportunities = []

        if not (self.options_strategies['cash_secured_puts']['dte_range'][0] <= dte <=
                self.options_strategies['cash_secured_puts']['dte_range'][1]):
            return opportunities

        # Find OTM puts with good premium
        otm_puts = puts[puts['strike'] < stock_price * 0.98]  # At least 2% OTM

        for _, put in otm_puts.head(5).iterrows():
            try:
                strike = put['strike']
                premium = put['lastPrice']
                delta = put.get('delta', -0.2)

                if (self.options_strategies['cash_secured_puts']['delta_range'][0] <=
                    abs(delta) <= self.options_strategies['cash_secured_puts']['delta_range'][1]):

                    # Calculate returns
                    cash_required = strike
                    max_profit = premium
                    max_return = (max_profit / cash_required) * (365 / dte)  # Annualized

                    if max_return > 0.12:  # 12% annualized minimum
                        opportunities.append({
                            'strategy': 'cash_secured_put',
                            'ticker': ticker,
                            'stock_price': stock_price,
                            'strike': strike,
                            'premium': premium,
                            'dte': dte,
                            'expected_return': max_return,
                            'delta': delta,
                            'allocation_required': cash_required,
                            'max_profit': max_profit
                        })

            except:
                continue

        return opportunities

    async def execute_options_strategy(self, strategy):
        """Execute the options strategy (simulation mode)"""

        logging.info(f"Executing {strategy['strategy']} for {strategy['ticker']}")

        # For now, just log the strategy (would integrate with Alpaca options API)
        execution_log = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy['strategy'],
            'ticker': strategy['ticker'],
            'action': 'SIMULATED_EXECUTION',
            'expected_return': strategy['expected_return'],
            'allocation': strategy['allocation_required']
        }

        # Save to file
        with open('options_executions.json', 'a') as f:
            f.write(json.dumps(execution_log) + '\n')

        return True

    async def continuous_discovery_loop(self):
        """Main discovery loop"""

        logging.info("Starting continuous options discovery...")

        while True:
            try:
                # Discover opportunities
                opportunities = await self.discover_high_volatility_stocks()

                if opportunities:
                    # Analyze strategies
                    strategies = await self.analyze_options_strategies(opportunities)

                    if strategies:
                        logging.info(f"Found {len(strategies)} options strategies")

                        # Save discoveries
                        discovery_data = {
                            'timestamp': datetime.now().isoformat(),
                            'opportunities': opportunities,
                            'strategies': strategies
                        }

                        with open(f'options_discovery_{datetime.now().strftime("%Y%m%d_%H%M")}.json', 'w') as f:
                            json.dump(discovery_data, f, indent=2, default=str)

                        # Execute top strategy (simulation)
                        if strategies:
                            await self.execute_options_strategy(strategies[0])

                # Wait before next discovery cycle
                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logging.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error

async def main():
    """Start autonomous options discovery"""

    print("AUTONOMOUS OPTIONS DISCOVERY SYSTEM")
    print("=" * 50)
    print("Discovering new stocks and options opportunities...")

    discovery_system = AutonomousOptionsDiscovery()
    await discovery_system.continuous_discovery_loop()

if __name__ == "__main__":
    asyncio.run(main())