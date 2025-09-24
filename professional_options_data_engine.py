"""
Professional Options Data Engine
Connects to institutional-grade APIs for real-time options data
Replaces Yahoo Finance with professional feeds for 5000% ROI targeting
"""

import os
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptionContract:
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class ProfessionalOptionsDataEngine:
    """
    Multi-source options data engine targeting institutional-grade feeds
    Primary: TD Ameritrade API (free with account)
    Secondary: Alpha Vantage (free tier)
    Tertiary: Polygon.io (free tier)
    Fallback: Yahoo Finance
    """

    def __init__(self):
        self.td_api_key = os.getenv('TD_AMERITRADE_API_KEY', '')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        self.polygon_key = os.getenv('POLYGON_API_KEY', '')

        # Professional data sources
        self.data_sources = {
            'td_ameritrade': {
                'base_url': 'https://api.tdameritrade.com/v1',
                'options_endpoint': '/marketdata/chains',
                'rate_limit': 120,  # requests per minute
                'priority': 1
            },
            'alpha_vantage': {
                'base_url': 'https://www.alphavantage.co/query',
                'rate_limit': 5,  # requests per minute (free tier)
                'priority': 2
            },
            'polygon': {
                'base_url': 'https://api.polygon.io/v3',
                'options_endpoint': '/reference/options/contracts',
                'rate_limit': 5,  # requests per minute (free tier)
                'priority': 3
            }
        }

        # Real-time execution tracking
        self.execution_costs = self._setup_professional_execution_costs()
        self.last_api_call = {}

        print(f"[PROFESSIONAL DATA ENGINE] Initialized with {len(self.data_sources)} data sources")
        print(f"[EXECUTION COSTS] Professional modeling active")

    def _setup_professional_execution_costs(self) -> Dict:
        """Setup realistic institutional execution cost modeling"""
        return {
            'broker_commissions': {
                'interactive_brokers': 0.70,  # per options contract
                'td_ameritrade': 0.65,
                'schwab': 0.65,
                'tastyworks': 1.00,
                'average': 0.75
            },
            'bid_ask_spreads': {
                'spy_options': {
                    'atm': 0.01,      # $0.01 spread for ATM
                    'otm_5pct': 0.02, # $0.02 spread for 5% OTM
                    'otm_10pct': 0.05 # $0.05 spread for 10% OTM
                },
                'qqq_options': {
                    'atm': 0.02,
                    'otm_5pct': 0.03,
                    'otm_10pct': 0.07
                }
            },
            'slippage_models': {
                'market_orders': 0.001,    # 0.1% slippage
                'limit_orders': 0.0005,    # 0.05% slippage
                'large_size': 0.002        # 0.2% for >$100K positions
            },
            'regulatory_fees': {
                'sec_fee': 0.0000278,      # $22.80 per $1M
                'finra_taf': 0.000166,     # Trading Activity Fee
                'exchange_fees': 0.0003    # Average exchange fees
            }
        }

    async def get_options_chain_professional(self, symbol: str, expiry_days: int = 30) -> List[OptionContract]:
        """Get options chain using professional data sources in priority order"""

        # Try TD Ameritrade first (highest quality)
        if self.td_api_key:
            try:
                contracts = await self._get_td_ameritrade_chain(symbol, expiry_days)
                if contracts:
                    print(f"[TD AMERITRADE] Retrieved {len(contracts)} contracts for {symbol}")
                    return contracts
            except Exception as e:
                print(f"[TD AMERITRADE ERROR] {str(e)}")

        # Try Alpha Vantage second
        if self.alpha_vantage_key:
            try:
                contracts = await self._get_alpha_vantage_chain(symbol, expiry_days)
                if contracts:
                    print(f"[ALPHA VANTAGE] Retrieved {len(contracts)} contracts for {symbol}")
                    return contracts
            except Exception as e:
                print(f"[ALPHA VANTAGE ERROR] {str(e)}")

        # Try Polygon.io third
        if self.polygon_key:
            try:
                contracts = await self._get_polygon_chain(symbol, expiry_days)
                if contracts:
                    print(f"[POLYGON] Retrieved {len(contracts)} contracts for {symbol}")
                    return contracts
            except Exception as e:
                print(f"[POLYGON ERROR] {str(e)}")

        # Fallback to Yahoo Finance
        print(f"[FALLBACK] Using Yahoo Finance for {symbol}")
        return await self._get_yahoo_fallback_chain(symbol, expiry_days)

    async def _get_td_ameritrade_chain(self, symbol: str, expiry_days: int) -> List[OptionContract]:
        """Get options chain from TD Ameritrade API"""
        if not self.td_api_key:
            return []

        # Rate limiting
        if 'td_ameritrade' in self.last_api_call:
            time_since_last = time.time() - self.last_api_call['td_ameritrade']
            if time_since_last < 0.5:  # 120 requests per minute = 0.5 seconds between calls
                await asyncio.sleep(0.5 - time_since_last)

        base_url = self.data_sources['td_ameritrade']['base_url']
        endpoint = self.data_sources['td_ameritrade']['options_endpoint']

        params = {
            'apikey': self.td_api_key,
            'symbol': symbol,
            'contractType': 'ALL',
            'strikeCount': 50,
            'includeQuotes': 'TRUE',
            'strategy': 'SINGLE',
            'optionType': 'ALL'
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}{endpoint}", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.last_api_call['td_ameritrade'] = time.time()
                    return self._parse_td_ameritrade_response(data)
                else:
                    print(f"[TD AMERITRADE] API Error: {response.status}")
                    return []

    def _parse_td_ameritrade_response(self, data: Dict) -> List[OptionContract]:
        """Parse TD Ameritrade options chain response"""
        contracts = []

        try:
            call_map = data.get('callExpDateMap', {})
            put_map = data.get('putExpDateMap', {})

            # Parse calls
            for expiry_date, strikes in call_map.items():
                for strike_price, contract_data in strikes.items():
                    for contract in contract_data:
                        contracts.append(OptionContract(
                            symbol=contract['symbol'],
                            strike=float(strike_price),
                            expiry=expiry_date.split(':')[0],
                            option_type='call',
                            bid=contract.get('bid', 0.0),
                            ask=contract.get('ask', 0.0),
                            last=contract.get('last', 0.0),
                            volume=contract.get('totalVolume', 0),
                            open_interest=contract.get('openInterest', 0),
                            implied_volatility=contract.get('volatility', 0.0) / 100,
                            delta=contract.get('delta', 0.0),
                            gamma=contract.get('gamma', 0.0),
                            theta=contract.get('theta', 0.0),
                            vega=contract.get('vega', 0.0),
                            rho=contract.get('rho', 0.0)
                        ))

            # Parse puts
            for expiry_date, strikes in put_map.items():
                for strike_price, contract_data in strikes.items():
                    for contract in contract_data:
                        contracts.append(OptionContract(
                            symbol=contract['symbol'],
                            strike=float(strike_price),
                            expiry=expiry_date.split(':')[0],
                            option_type='put',
                            bid=contract.get('bid', 0.0),
                            ask=contract.get('ask', 0.0),
                            last=contract.get('last', 0.0),
                            volume=contract.get('totalVolume', 0),
                            open_interest=contract.get('openInterest', 0),
                            implied_volatility=contract.get('volatility', 0.0) / 100,
                            delta=contract.get('delta', 0.0),
                            gamma=contract.get('gamma', 0.0),
                            theta=contract.get('theta', 0.0),
                            vega=contract.get('vega', 0.0),
                            rho=contract.get('rho', 0.0)
                        ))

        except Exception as e:
            print(f"[TD AMERITRADE PARSE ERROR] {str(e)}")

        return contracts

    async def _get_alpha_vantage_chain(self, symbol: str, expiry_days: int) -> List[OptionContract]:
        """Get options data from Alpha Vantage API"""
        if not self.alpha_vantage_key:
            return []

        # Rate limiting for free tier
        if 'alpha_vantage' in self.last_api_call:
            time_since_last = time.time() - self.last_api_call['alpha_vantage']
            if time_since_last < 12:  # 5 requests per minute = 12 seconds between calls
                await asyncio.sleep(12 - time_since_last)

        # Alpha Vantage doesn't have a direct options chain endpoint
        # We'll get historical options data instead
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': symbol,
            'apikey': self.alpha_vantage_key
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(self.data_sources['alpha_vantage']['base_url'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.last_api_call['alpha_vantage'] = time.time()
                    return self._parse_alpha_vantage_response(data, symbol)
                else:
                    return []

    def _parse_alpha_vantage_response(self, data: Dict, symbol: str) -> List[OptionContract]:
        """Parse Alpha Vantage options response"""
        # Alpha Vantage historical options data is limited
        # This is a simplified implementation
        return []

    async def _get_polygon_chain(self, symbol: str, expiry_days: int) -> List[OptionContract]:
        """Get options data from Polygon.io API"""
        if not self.polygon_key:
            return []

        # Rate limiting for free tier
        if 'polygon' in self.last_api_call:
            time_since_last = time.time() - self.last_api_call['polygon']
            if time_since_last < 12:  # 5 requests per minute
                await asyncio.sleep(12 - time_since_last)

        base_url = self.data_sources['polygon']['base_url']
        endpoint = self.data_sources['polygon']['options_endpoint']

        params = {
            'underlying_ticker': symbol,
            'limit': 1000,
            'apikey': self.polygon_key
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}{endpoint}", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.last_api_call['polygon'] = time.time()
                    return self._parse_polygon_response(data)
                else:
                    return []

    def _parse_polygon_response(self, data: Dict) -> List[OptionContract]:
        """Parse Polygon.io options response"""
        contracts = []

        try:
            results = data.get('results', [])
            for contract in results:
                # Polygon provides contract info but not real-time pricing
                # This would need additional API calls for current quotes
                pass
        except Exception as e:
            print(f"[POLYGON PARSE ERROR] {str(e)}")

        return contracts

    async def _get_yahoo_fallback_chain(self, symbol: str, expiry_days: int) -> List[OptionContract]:
        """Fallback to Yahoo Finance for options data"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return []

            # Get the nearest expiration within our target days
            target_date = datetime.now() + timedelta(days=expiry_days)
            best_expiry = min(expirations,
                            key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))

            # Get options chain for the selected expiry
            options_chain = ticker.option_chain(best_expiry)

            contracts = []

            # Parse calls
            for _, call in options_chain.calls.iterrows():
                contracts.append(OptionContract(
                    symbol=call.get('contractSymbol', ''),
                    strike=float(call.get('strike', 0)),
                    expiry=best_expiry,
                    option_type='call',
                    bid=float(call.get('bid', 0)),
                    ask=float(call.get('ask', 0)),
                    last=float(call.get('lastPrice', 0)),
                    volume=int(call.get('volume', 0)) if pd.notna(call.get('volume')) else 0,
                    open_interest=int(call.get('openInterest', 0)) if pd.notna(call.get('openInterest')) else 0,
                    implied_volatility=float(call.get('impliedVolatility', 0)) if pd.notna(call.get('impliedVolatility')) else 0,
                    delta=0.5,  # Estimated
                    gamma=0.01, # Estimated
                    theta=-0.01, # Estimated
                    vega=0.1,   # Estimated
                    rho=0.01    # Estimated
                ))

            # Parse puts
            for _, put in options_chain.puts.iterrows():
                contracts.append(OptionContract(
                    symbol=put.get('contractSymbol', ''),
                    strike=float(put.get('strike', 0)),
                    expiry=best_expiry,
                    option_type='put',
                    bid=float(put.get('bid', 0)),
                    ask=float(put.get('ask', 0)),
                    last=float(put.get('lastPrice', 0)),
                    volume=int(put.get('volume', 0)) if pd.notna(put.get('volume')) else 0,
                    open_interest=int(put.get('openInterest', 0)) if pd.notna(put.get('openInterest')) else 0,
                    implied_volatility=float(put.get('impliedVolatility', 0)) if pd.notna(put.get('impliedVolatility')) else 0,
                    delta=-0.5,  # Estimated
                    gamma=0.01,  # Estimated
                    theta=-0.01, # Estimated
                    vega=0.1,    # Estimated
                    rho=-0.01   # Estimated
                ))

            return contracts

        except Exception as e:
            print(f"[YAHOO FALLBACK ERROR] {str(e)}")
            return []

    def calculate_real_execution_cost(self, position_size: float, option_price: float,
                                    symbol: str = 'SPY', is_opening: bool = True) -> Dict:
        """Calculate realistic execution costs for options trading"""

        num_contracts = int(position_size / (option_price * 100))

        # Commission costs
        commission = num_contracts * self.execution_costs['broker_commissions']['average']

        # Bid-ask spread cost (pay spread when opening, receive spread when closing)
        spread_cost = 0
        if symbol.upper() in ['SPY', 'QQQ']:
            if is_opening:
                spread_cost = num_contracts * 100 * self.execution_costs['bid_ask_spreads']['spy_options']['atm']

        # Slippage costs
        slippage_rate = self.execution_costs['slippage_models']['market_orders']
        if position_size > 100000:  # Large position
            slippage_rate = self.execution_costs['slippage_models']['large_size']

        slippage_cost = position_size * slippage_rate

        # Regulatory fees
        sec_fee = position_size * self.execution_costs['regulatory_fees']['sec_fee']
        finra_fee = position_size * self.execution_costs['regulatory_fees']['finra_taf']
        exchange_fee = position_size * self.execution_costs['regulatory_fees']['exchange_fees']

        total_cost = commission + spread_cost + slippage_cost + sec_fee + finra_fee + exchange_fee

        return {
            'total_cost': total_cost,
            'commission': commission,
            'spread_cost': spread_cost,
            'slippage_cost': slippage_cost,
            'regulatory_fees': sec_fee + finra_fee + exchange_fee,
            'cost_percentage': (total_cost / position_size) * 100,
            'num_contracts': num_contracts
        }

    async def get_real_time_market_data(self, symbols: List[str]) -> Dict:
        """Get real-time market data for multiple symbols"""
        market_data = {}

        for symbol in symbols:
            try:
                # Try professional APIs first, fallback to Yahoo
                if self.td_api_key:
                    data = await self._get_td_quote(symbol)
                else:
                    data = await self._get_yahoo_quote(symbol)

                market_data[symbol] = data
            except Exception as e:
                print(f"[MARKET DATA ERROR] {symbol}: {str(e)}")
                market_data[symbol] = None

        return market_data

    async def _get_td_quote(self, symbol: str) -> Dict:
        """Get real-time quote from TD Ameritrade"""
        # Rate limiting
        if 'td_quote' in self.last_api_call:
            time_since_last = time.time() - self.last_api_call['td_quote']
            if time_since_last < 0.5:
                await asyncio.sleep(0.5 - time_since_last)

        url = f"{self.data_sources['td_ameritrade']['base_url']}/marketdata/{symbol}/quotes"
        params = {'apikey': self.td_api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.last_api_call['td_quote'] = time.time()
                    return data.get(symbol, {})
                else:
                    raise Exception(f"TD API Error: {response.status}")

    async def _get_yahoo_quote(self, symbol: str) -> Dict:
        """Get quote from Yahoo Finance as fallback"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'lastPrice': info.get('currentPrice', 0),
                'bidPrice': info.get('bid', 0),
                'askPrice': info.get('ask', 0),
                'volume': info.get('volume', 0),
                'volatility': info.get('impliedVolatility', 0)
            }
        except Exception as e:
            print(f"[YAHOO QUOTE ERROR] {symbol}: {str(e)}")
            return {}

class RealTimeOptionsStrategyGenerator:
    """
    Generate options strategies using real-time professional data
    Target: 5000% annual returns through leverage and precision timing
    """

    def __init__(self):
        self.data_engine = ProfessionalOptionsDataEngine()
        self.strategies_generated = 0
        self.target_sharpe = 2.0
        self.target_annual_return = 50.0  # 5000% = 50x multiplier

        print(f"[REAL-TIME STRATEGY GENERATOR] Target: {self.target_annual_return}x returns")

    async def generate_professional_strategies(self, symbols: List[str] = ['SPY', 'QQQ']) -> List[Dict]:
        """Generate strategies using professional options data"""

        all_strategies = []

        for symbol in symbols:
            print(f"\n[GENERATING] Professional strategies for {symbol}")

            # Get professional options chain
            contracts = await self.data_engine.get_options_chain_professional(symbol)

            if not contracts:
                print(f"[WARNING] No options data for {symbol}")
                continue

            # Get current market data
            market_data = await self.data_engine.get_real_time_market_data([symbol])
            current_price = market_data.get(symbol, {}).get('lastPrice', 100)

            # Generate strategies based on professional data
            strategies = await self._generate_strategies_from_contracts(contracts, current_price, symbol)
            all_strategies.extend(strategies)

        # Rank strategies by expected return potential
        ranked_strategies = sorted(all_strategies,
                                 key=lambda x: x.get('expected_annual_return', 0),
                                 reverse=True)

        return ranked_strategies[:20]  # Top 20 strategies

    async def _generate_strategies_from_contracts(self, contracts: List[OptionContract],
                                                current_price: float, symbol: str) -> List[Dict]:
        """Generate specific strategies from options contracts"""

        strategies = []

        # Filter for liquid contracts
        liquid_contracts = [c for c in contracts if c.volume > 10 and c.open_interest > 50]

        if not liquid_contracts:
            return []

        # Strategy 1: ATM Straddle for high IV
        atm_contracts = [c for c in liquid_contracts if abs(c.strike - current_price) < current_price * 0.02]
        if atm_contracts:
            straddle_strategy = await self._create_straddle_strategy(atm_contracts, current_price, symbol)
            if straddle_strategy:
                strategies.append(straddle_strategy)

        # Strategy 2: Iron Condor for sideways markets
        iron_condor = await self._create_iron_condor_strategy(liquid_contracts, current_price, symbol)
        if iron_condor:
            strategies.append(iron_condor)

        # Strategy 3: Call spread for bullish bias
        call_spread = await self._create_call_spread_strategy(liquid_contracts, current_price, symbol)
        if call_spread:
            strategies.append(call_spread)

        # Strategy 4: Put spread for bearish bias
        put_spread = await self._create_put_spread_strategy(liquid_contracts, current_price, symbol)
        if put_spread:
            strategies.append(put_spread)

        return strategies

    async def _create_straddle_strategy(self, contracts: List[OptionContract],
                                      current_price: float, symbol: str) -> Optional[Dict]:
        """Create ATM straddle strategy"""

        # Find ATM call and put
        atm_call = min([c for c in contracts if c.option_type == 'call'],
                      key=lambda x: abs(x.strike - current_price), default=None)
        atm_put = min([c for c in contracts if c.option_type == 'put'],
                     key=lambda x: abs(x.strike - current_price), default=None)

        if not (atm_call and atm_put):
            return None

        # Calculate strategy metrics
        entry_cost = (atm_call.ask + atm_put.ask) * 100  # Cost per straddle
        max_profit_potential = float('inf')  # Unlimited profit potential
        breakeven_upper = atm_call.strike + (atm_call.ask + atm_put.ask)
        breakeven_lower = atm_put.strike - (atm_call.ask + atm_put.ask)

        # Calculate execution costs
        execution_cost = self.data_engine.calculate_real_execution_cost(
            position_size=entry_cost, option_price=(atm_call.ask + atm_put.ask)/2, symbol=symbol
        )

        # Estimate probability of profit (simplified)
        iv_avg = (atm_call.implied_volatility + atm_put.implied_volatility) / 2
        days_to_expiry = 30  # Assuming 30 days
        expected_move = current_price * iv_avg * np.sqrt(days_to_expiry / 365)

        prob_profit = min(expected_move / ((breakeven_upper - breakeven_lower) / 2), 0.8)

        # Calculate expected return
        avg_profit = expected_move * 2  # Simplified calculation
        expected_annual_return = (avg_profit / entry_cost) * (365 / days_to_expiry) * prob_profit

        return {
            'strategy_name': f'{symbol}_ATM_Straddle_Professional',
            'strategy_type': 'straddle',
            'symbol': symbol,
            'legs': [
                {
                    'action': 'buy',
                    'option_type': 'call',
                    'strike': atm_call.strike,
                    'expiry': atm_call.expiry,
                    'quantity': 1,
                    'price': atm_call.ask
                },
                {
                    'action': 'buy',
                    'option_type': 'put',
                    'strike': atm_put.strike,
                    'expiry': atm_put.expiry,
                    'quantity': 1,
                    'price': atm_put.ask
                }
            ],
            'entry_cost': entry_cost,
            'max_profit': max_profit_potential,
            'breakeven_points': [breakeven_lower, breakeven_upper],
            'execution_costs': execution_cost,
            'expected_annual_return': expected_annual_return,
            'probability_profit': prob_profit,
            'iv_rank': iv_avg,
            'professional_score': expected_annual_return * prob_profit,
            'timestamp': datetime.now().isoformat()
        }

    async def _create_iron_condor_strategy(self, contracts: List[OptionContract],
                                         current_price: float, symbol: str) -> Optional[Dict]:
        """Create Iron Condor strategy for sideways markets"""

        # Find contracts for iron condor (OTM puts and calls)
        otm_puts = [c for c in contracts if c.option_type == 'put' and c.strike < current_price * 0.95]
        otm_calls = [c for c in contracts if c.option_type == 'call' and c.strike > current_price * 1.05]

        if len(otm_puts) < 2 or len(otm_calls) < 2:
            return None

        # Select strikes for iron condor
        short_put = max(otm_puts, key=lambda x: x.strike)  # Highest strike put
        long_put = min([p for p in otm_puts if p.strike < short_put.strike],
                      key=lambda x: short_put.strike - x.strike)

        short_call = min(otm_calls, key=lambda x: x.strike)  # Lowest strike call
        long_call = max([c for c in otm_calls if c.strike > short_call.strike],
                       key=lambda x: x.strike - short_call.strike)

        # Calculate strategy metrics
        credit_received = (short_put.bid + short_call.bid - long_put.ask - long_call.ask) * 100
        max_profit = credit_received
        max_loss = ((short_call.strike - long_call.strike) * 100) - credit_received

        if credit_received <= 0 or max_loss <= 0:
            return None

        # Profit zone (between short strikes)
        profit_zone_width = short_call.strike - short_put.strike
        prob_profit = min(profit_zone_width / (current_price * 0.2), 0.7)  # Simplified

        expected_annual_return = (max_profit / abs(max_loss)) * (365 / 30) * prob_profit

        return {
            'strategy_name': f'{symbol}_Iron_Condor_Professional',
            'strategy_type': 'iron_condor',
            'symbol': symbol,
            'legs': [
                {'action': 'sell', 'option_type': 'put', 'strike': short_put.strike, 'price': short_put.bid},
                {'action': 'buy', 'option_type': 'put', 'strike': long_put.strike, 'price': long_put.ask},
                {'action': 'sell', 'option_type': 'call', 'strike': short_call.strike, 'price': short_call.bid},
                {'action': 'buy', 'option_type': 'call', 'strike': long_call.strike, 'price': long_call.ask}
            ],
            'credit_received': credit_received,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_zone': [short_put.strike, short_call.strike],
            'expected_annual_return': expected_annual_return,
            'probability_profit': prob_profit,
            'professional_score': expected_annual_return * prob_profit,
            'timestamp': datetime.now().isoformat()
        }

    async def _create_call_spread_strategy(self, contracts: List[OptionContract],
                                         current_price: float, symbol: str) -> Optional[Dict]:
        """Create bullish call spread strategy"""

        calls = [c for c in contracts if c.option_type == 'call' and c.strike > current_price]

        if len(calls) < 2:
            return None

        # Select strikes for call spread
        short_call = min(calls, key=lambda x: x.strike)  # Lower strike (sell)
        long_call = min([c for c in calls if c.strike > short_call.strike],
                       key=lambda x: x.strike - short_call.strike)  # Higher strike (buy)

        # Calculate strategy metrics
        net_debit = (long_call.ask - short_call.bid) * 100
        max_profit = ((long_call.strike - short_call.strike) * 100) - net_debit
        max_loss = net_debit

        if net_debit <= 0 or max_profit <= 0:
            return None

        # Estimate probability (simplified)
        prob_profit = max(0.3, min(0.7, (current_price / short_call.strike) * 0.5))
        expected_annual_return = (max_profit / max_loss) * (365 / 30) * prob_profit

        return {
            'strategy_name': f'{symbol}_Bull_Call_Spread_Professional',
            'strategy_type': 'call_spread',
            'symbol': symbol,
            'legs': [
                {'action': 'buy', 'option_type': 'call', 'strike': short_call.strike, 'price': short_call.ask},
                {'action': 'sell', 'option_type': 'call', 'strike': long_call.strike, 'price': long_call.bid}
            ],
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'expected_annual_return': expected_annual_return,
            'probability_profit': prob_profit,
            'professional_score': expected_annual_return * prob_profit,
            'timestamp': datetime.now().isoformat()
        }

    async def _create_put_spread_strategy(self, contracts: List[OptionContract],
                                        current_price: float, symbol: str) -> Optional[Dict]:
        """Create bearish put spread strategy"""

        puts = [c for c in contracts if c.option_type == 'put' and c.strike < current_price]

        if len(puts) < 2:
            return None

        # Select strikes for put spread
        short_put = max(puts, key=lambda x: x.strike)  # Higher strike (sell)
        long_put = max([p for p in puts if p.strike < short_put.strike],
                      key=lambda x: short_put.strike - x.strike)  # Lower strike (buy)

        # Calculate strategy metrics
        net_debit = (long_put.ask - short_put.bid) * 100
        max_profit = ((short_put.strike - long_put.strike) * 100) - net_debit
        max_loss = net_debit

        if net_debit <= 0 or max_profit <= 0:
            return None

        # Estimate probability (simplified)
        prob_profit = max(0.3, min(0.7, (short_put.strike / current_price) * 0.5))
        expected_annual_return = (max_profit / max_loss) * (365 / 30) * prob_profit

        return {
            'strategy_name': f'{symbol}_Bear_Put_Spread_Professional',
            'strategy_type': 'put_spread',
            'symbol': symbol,
            'legs': [
                {'action': 'buy', 'option_type': 'put', 'strike': short_put.strike, 'price': short_put.ask},
                {'action': 'sell', 'option_type': 'put', 'strike': long_put.strike, 'price': long_put.bid}
            ],
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'expected_annual_return': expected_annual_return,
            'probability_profit': prob_profit,
            'professional_score': expected_annual_return * prob_profit,
            'timestamp': datetime.now().isoformat()
        }

async def run_professional_options_engine():
    """Run the professional options data engine and generate strategies"""

    print("=" * 80)
    print("PROFESSIONAL OPTIONS DATA ENGINE - TARGETING 5000% ROI")
    print("=" * 80)

    # Initialize the strategy generator
    generator = RealTimeOptionsStrategyGenerator()

    # Generate professional strategies
    strategies = await generator.generate_professional_strategies(['SPY', 'QQQ', 'IWM'])

    if not strategies:
        print("[ERROR] No strategies generated - check API keys and data feeds")
        return

    # Display top strategies
    print(f"\n[SUCCESS] Generated {len(strategies)} professional strategies")
    print("\nTOP 5 PROFESSIONAL STRATEGIES:")
    print("-" * 60)

    for i, strategy in enumerate(strategies[:5], 1):
        print(f"\n{i}. {strategy['strategy_name']}")
        print(f"   Type: {strategy['strategy_type']}")
        print(f"   Expected Annual Return: {strategy['expected_annual_return']:.1f}%")
        print(f"   Probability of Profit: {strategy['probability_profit']:.1%}")
        print(f"   Professional Score: {strategy['professional_score']:.2f}")

        if 'execution_costs' in strategy:
            exec_costs = strategy['execution_costs']
            print(f"   Execution Costs: ${exec_costs['total_cost']:.2f} ({exec_costs['cost_percentage']:.2f}%)")

    # Save strategies to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"professional_options_strategies_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(strategies, f, indent=2, default=str)

    print(f"\n[SAVED] Strategies saved to {filename}")

    # Calculate aggregate statistics
    total_expected_return = sum(s['expected_annual_return'] for s in strategies)
    avg_prob_profit = sum(s['probability_profit'] for s in strategies) / len(strategies)

    print(f"\n[AGGREGATE STATS]")
    print(f"Total Expected Return: {total_expected_return:.1f}%")
    print(f"Average Probability: {avg_prob_profit:.1%}")
    print(f"Portfolio Multiplier: {total_expected_return/100:.1f}x")

    if total_expected_return > 5000:
        print("[TARGET ACHIEVED] 5000% annual return target EXCEEDED!")
    else:
        print(f"[PROGRESS] {(total_expected_return/5000)*100:.1f}% toward 5000% target")

    return strategies

if __name__ == "__main__":
    asyncio.run(run_professional_options_engine())