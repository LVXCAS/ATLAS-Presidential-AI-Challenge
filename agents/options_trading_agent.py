"""
Real Options Trading Agent
Implements actual options contract trading with proper strategies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
import numpy as np

from agents.broker_integration import AlpacaBrokerIntegration, OrderRequest, OrderSide, OrderType, TimeInForce
from agents.options_broker import OptionsBroker, OptionsOrderRequest, OptionsOrderType
from agents.quantlib_pricing import quantlib_pricer
from config.logging_config import get_logger

logger = get_logger(__name__)

class OptionsStrategy(str, Enum):
    """Options strategy types"""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"

class OptionsExpiration(str, Enum):
    """Options expiration periods"""
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

@dataclass
class OptionsContract:
    """Options contract data structure"""
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price between bid and ask"""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.ask - self.bid
    
    @property
    def days_to_expiry(self) -> int:
        """Calculate days until expiration"""
        return (self.expiration - datetime.now()).days

@dataclass
class OptionsPosition:
    """Track options positions"""
    symbol: str
    underlying: str
    strategy: OptionsStrategy
    contracts: List[OptionsContract]
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_loss: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

class OptionsTrader:
    """Real options trading implementation"""
    
    def __init__(self, broker: Optional[AlpacaBrokerIntegration] = None):
        self.broker = broker
        self.options_broker = OptionsBroker(broker, paper_trading=True)
        self.positions: Dict[str, OptionsPosition] = {}
        self.option_chains: Dict[str, List[OptionsContract]] = {}
        self.min_volume = 50  # Minimum volume for liquidity
        self.min_open_interest = 100  # Minimum open interest
        self.max_spread_ratio = 0.10  # Maximum 10% bid-ask spread
        self.min_days_to_expiry = 14  # Only trade options with > 2 weeks to expiry
        
    async def get_options_chain(self, symbol: str, expiration: Optional[datetime] = None) -> List[OptionsContract]:
        """Get options chain for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return []
            
            # Filter expirations to only those > 2 weeks out
            valid_expirations = []
            today = datetime.now()
            
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                days_to_exp = (exp_date - today).days
                if days_to_exp > self.min_days_to_expiry:
                    valid_expirations.append(exp_str)
            
            if not valid_expirations:
                logger.warning(f"No options with > {self.min_days_to_expiry} days to expiry for {symbol}")
                return []
            
            # Use specified expiration or first valid expiration
            if expiration:
                exp_str = expiration.strftime('%Y-%m-%d')
                if exp_str not in valid_expirations:
                    logger.warning(f"Expiration {exp_str} not available or < {self.min_days_to_expiry} days for {symbol}")
                    return []
                target_exp = exp_str
            else:
                # Use the nearest valid expiration (> 2 weeks)
                target_exp = valid_expirations[0]
            
            # Get options data
            opt_chain = ticker.option_chain(target_exp)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            contracts = []
            exp_date = datetime.strptime(target_exp, '%Y-%m-%d')
            
            # Process calls
            for _, row in calls.iterrows():
                if row['volume'] >= self.min_volume and row['openInterest'] >= self.min_open_interest:
                    contract = OptionsContract(
                        symbol=f"{symbol}{exp_date.strftime('%y%m%d')}C{int(row['strike']*1000):08d}",
                        underlying=symbol,
                        strike=float(row['strike']),
                        expiration=exp_date,
                        option_type='call',
                        bid=float(row['bid']),
                        ask=float(row['ask']),
                        volume=int(row['volume']),
                        open_interest=int(row['openInterest']),
                        implied_volatility=float(row['impliedVolatility']),
                        delta=0.0,  # Will be calculated with QuantLib
                        gamma=0.0,
                        theta=0.0,
                        vega=0.0
                    )
                    
                    # Enhance with QuantLib pricing if available
                    if quantlib_pricer:
                        try:
                            # Get current underlying price for Greeks calculation
                            underlying_ticker = yf.Ticker(symbol)
                            underlying_hist = underlying_ticker.history(period="1d")
                            if not underlying_hist.empty:
                                underlying_price = float(underlying_hist['Close'].iloc[-1])
                                
                                # Calculate accurate Greeks using QuantLib
                                pricing_data = quantlib_pricer.price_european_option(
                                    option_type='call',
                                    underlying_price=underlying_price,
                                    strike=contract.strike,
                                    expiry_date=contract.expiration,
                                    symbol=symbol
                                )
                                
                                # Update contract with accurate Greeks
                                contract.delta = pricing_data['delta']
                                contract.gamma = pricing_data['gamma'] 
                                contract.theta = pricing_data['theta']
                                contract.vega = pricing_data['vega']
                                
                                logger.debug(f"QuantLib Greeks for {contract.symbol}: delta={contract.delta:.3f}, gamma={contract.gamma:.4f}, theta={contract.theta:.3f}, vega={contract.vega:.3f}")
                                
                        except Exception as e:
                            logger.warning(f"QuantLib Greeks calculation failed for {contract.symbol}: {e}")
                            # Keep default values
                    
                    # Filter by spread ratio and expiration time
                    if (contract.spread / contract.mid_price <= self.max_spread_ratio and 
                        contract.days_to_expiry > self.min_days_to_expiry):
                        contracts.append(contract)
            
            # Process puts
            for _, row in puts.iterrows():
                if row['volume'] >= self.min_volume and row['openInterest'] >= self.min_open_interest:
                    contract = OptionsContract(
                        symbol=f"{symbol}{exp_date.strftime('%y%m%d')}P{int(row['strike']*1000):08d}",
                        underlying=symbol,
                        strike=float(row['strike']),
                        expiration=exp_date,
                        option_type='put',
                        bid=float(row['bid']),
                        ask=float(row['ask']),
                        volume=int(row['volume']),
                        open_interest=int(row['openInterest']),
                        implied_volatility=float(row['impliedVolatility']),
                        delta=0.0,
                        gamma=0.0,
                        theta=0.0,
                        vega=0.0
                    )
                    
                    # Enhance with QuantLib pricing if available
                    if quantlib_pricer:
                        try:
                            # Get current underlying price for Greeks calculation
                            underlying_ticker = yf.Ticker(symbol)
                            underlying_hist = underlying_ticker.history(period="1d")
                            if not underlying_hist.empty:
                                underlying_price = float(underlying_hist['Close'].iloc[-1])
                                
                                # Calculate accurate Greeks using QuantLib
                                pricing_data = quantlib_pricer.price_european_option(
                                    option_type='put',
                                    underlying_price=underlying_price,
                                    strike=contract.strike,
                                    expiry_date=contract.expiration,
                                    symbol=symbol
                                )
                                
                                # Update contract with accurate Greeks
                                contract.delta = pricing_data['delta']
                                contract.gamma = pricing_data['gamma']
                                contract.theta = pricing_data['theta']
                                contract.vega = pricing_data['vega']
                                
                                logger.debug(f"QuantLib Greeks for {contract.symbol}: delta={contract.delta:.3f}, gamma={contract.gamma:.4f}, theta={contract.theta:.3f}, vega={contract.vega:.3f}")
                                
                        except Exception as e:
                            logger.warning(f"QuantLib Greeks calculation failed for {contract.symbol}: {e}")
                            # Keep default values
                    
                    # Filter by spread ratio and expiration time
                    if (contract.spread / contract.mid_price <= self.max_spread_ratio and 
                        contract.days_to_expiry > self.min_days_to_expiry):
                        contracts.append(contract)
            
            self.option_chains[symbol] = contracts
            logger.info(f"Retrieved {len(contracts)} liquid options for {symbol} with > {self.min_days_to_expiry} days to expiry")
            return contracts
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return []
    
    def find_best_options_strategy(self, symbol: str, price: float, volatility: float, 
                                 rsi: float, price_change: float) -> Optional[Tuple[OptionsStrategy, List[OptionsContract]]]:
        """Find the best options strategy based on market conditions"""
        
        if symbol not in self.option_chains or not self.option_chains[symbol]:
            return None
        
        contracts = self.option_chains[symbol]
        
        # Separate calls and puts
        calls = [c for c in contracts if c.option_type == 'call']
        puts = [c for c in contracts if c.option_type == 'put']
        
        if not calls or not puts:
            return None
        
        # Strategy selection based on market conditions
        
        # 1. Strong bullish momentum - Long Call
        if price_change > 0.03 and rsi < 70 and volatility > 20:
            # Find slightly OTM call with good delta
            target_strike = price * 1.02
            suitable_calls = [c for c in calls if c.strike <= price * 1.05]  # Not too far OTM
            
            if suitable_calls:
                # Use QuantLib Greeks for better selection
                if quantlib_pricer:
                    # Prefer calls with delta between 0.4 and 0.7 (good directional exposure)
                    good_delta_calls = [c for c in suitable_calls if 0.4 <= abs(c.delta) <= 0.7]
                    if good_delta_calls:
                        # Select call with best delta/gamma combination
                        best_call = max(good_delta_calls, key=lambda c: c.delta * c.gamma)
                    else:
                        best_call = min(suitable_calls, key=lambda c: abs(c.strike - target_strike))
                else:
                    best_call = min(suitable_calls, key=lambda c: abs(c.strike - target_strike))
                
                return OptionsStrategy.LONG_CALL, [best_call]
        
        # 2. Strong bearish momentum - Long Put
        if price_change < -0.03 and rsi > 30 and volatility > 20:
            # Find slightly OTM put with good delta
            target_strike = price * 0.98
            suitable_puts = [p for p in puts if p.strike >= price * 0.95]  # Not too far OTM
            
            if suitable_puts:
                # Use QuantLib Greeks for better selection
                if quantlib_pricer:
                    # Prefer puts with delta between -0.7 and -0.4 (good directional exposure)
                    good_delta_puts = [p for p in suitable_puts if -0.7 <= p.delta <= -0.4]
                    if good_delta_puts:
                        # Select put with best delta/gamma combination (negative delta for puts)
                        best_put = max(good_delta_puts, key=lambda p: abs(p.delta) * p.gamma)
                    else:
                        best_put = min(suitable_puts, key=lambda p: abs(p.strike - target_strike))
                else:
                    best_put = min(suitable_puts, key=lambda p: abs(p.strike - target_strike))
                
                return OptionsStrategy.LONG_PUT, [best_put]
        
        # 3. High volatility, neutral bias - Straddle
        if volatility > 30 and abs(price_change) < 0.01 and 30 < rsi < 70:
            # Find ATM call and put for straddle
            atm_calls = [c for c in calls if abs(c.strike - price) / price < 0.02]
            atm_puts = [p for p in puts if abs(p.strike - price) / price < 0.02]
            
            if atm_calls and atm_puts:
                if quantlib_pricer:
                    # Select options with high vega for volatility plays
                    best_call = max(atm_calls, key=lambda c: c.vega)
                    best_put = max(atm_puts, key=lambda p: p.vega)
                    
                    # Ensure they're at the same strike for a proper straddle
                    if abs(best_call.strike - best_put.strike) < 0.01:
                        return OptionsStrategy.STRADDLE, [best_call, best_put]
                    else:
                        # Find matching strikes
                        for call in atm_calls:
                            matching_put = next((p for p in atm_puts if abs(p.strike - call.strike) < 0.01), None)
                            if matching_put:
                                return OptionsStrategy.STRADDLE, [call, matching_put]
                else:
                    atm_call = min(calls, key=lambda c: abs(c.strike - price))
                    atm_put = min(puts, key=lambda c: abs(c.strike - price))
                    if abs(atm_call.strike - atm_put.strike) < 0.01:
                        return OptionsStrategy.STRADDLE, [atm_call, atm_put]
        
        # 4. Moderate bullish bias - Bull Call Spread
        if 0.01 < price_change < 0.03 and rsi < 65:
            # Find ITM and OTM calls
            itm_strike = price * 0.98
            otm_strike = price * 1.05
            itm_call = min([c for c in calls if c.strike <= price], 
                          key=lambda c: abs(c.strike - itm_strike), default=None)
            otm_call = min([c for c in calls if c.strike >= price], 
                          key=lambda c: abs(c.strike - otm_strike), default=None)
            
            if itm_call and otm_call and itm_call.strike < otm_call.strike:
                return OptionsStrategy.BULL_CALL_SPREAD, [itm_call, otm_call]
        
        # 5. Moderate bearish bias - Bear Put Spread
        if -0.03 < price_change < -0.01 and rsi > 35:
            # Find ITM and OTM puts
            itm_strike = price * 1.02
            otm_strike = price * 0.95
            itm_put = min([p for p in puts if p.strike >= price], 
                         key=lambda p: abs(p.strike - itm_strike), default=None)
            otm_put = min([p for p in puts if p.strike <= price], 
                         key=lambda p: abs(p.strike - otm_strike), default=None)
            
            if itm_put and otm_put and itm_put.strike > otm_put.strike:
                return OptionsStrategy.BEAR_PUT_SPREAD, [itm_put, otm_put]
        
        # 6. Conservative income - Cash Secured Put (if we have cash)
        if rsi < 40 and price_change > -0.02 and volatility > 15:
            # Find slightly OTM put
            target_strike = price * 0.95
            best_put = min(puts, key=lambda c: abs(c.strike - target_strike))
            if best_put.strike >= price * 0.90:
                return OptionsStrategy.CASH_SECURED_PUT, [best_put]
        
        return None
    
    async def execute_options_strategy(self, strategy: OptionsStrategy, contracts: List[OptionsContract], 
                                     quantity: int = 1) -> Optional[OptionsPosition]:
        """Execute an options trading strategy with REAL options orders"""
        
        if not contracts:
            return None
        
        underlying = contracts[0].underlying
        position_id = f"{underlying}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            total_cost = 0
            executed_orders = []
            executed_contracts = []
            
            # Execute strategy legs with REAL options orders
            if strategy == OptionsStrategy.LONG_CALL:
                # Buy call option
                contract = contracts[0]
                
                order_request = OptionsOrderRequest(
                    symbol=contract.symbol,
                    underlying=underlying,
                    qty=quantity,
                    side=OrderSide.BUY,
                    type=OptionsOrderType.MARKET,
                    option_type='call',
                    strike=contract.strike,
                    expiration=contract.expiration,
                    client_order_id=f"LONG_CALL_{position_id}"
                )
                
                order_response = await self.options_broker.submit_options_order(order_request)
                executed_orders.append(order_response)
                
                if order_response.avg_fill_price:
                    total_cost = order_response.avg_fill_price * quantity * 100
                    executed_contracts = [contract]
                
            elif strategy == OptionsStrategy.LONG_PUT:
                # Buy put option
                contract = contracts[0]
                
                order_request = OptionsOrderRequest(
                    symbol=contract.symbol,
                    underlying=underlying,
                    qty=quantity,
                    side=OrderSide.BUY,
                    type=OptionsOrderType.MARKET,
                    option_type='put',
                    strike=contract.strike,
                    expiration=contract.expiration,
                    client_order_id=f"LONG_PUT_{position_id}"
                )
                
                order_response = await self.options_broker.submit_options_order(order_request)
                executed_orders.append(order_response)
                
                if order_response.avg_fill_price:
                    total_cost = order_response.avg_fill_price * quantity * 100
                    executed_contracts = [contract]
                
            elif strategy == OptionsStrategy.BULL_CALL_SPREAD:
                # Buy lower strike call, sell higher strike call
                long_call, short_call = contracts[0], contracts[1]
                if long_call.strike > short_call.strike:
                    long_call, short_call = short_call, long_call
                
                # Buy long call
                long_order = OptionsOrderRequest(
                    symbol=long_call.symbol,
                    underlying=underlying,
                    qty=quantity,
                    side=OrderSide.BUY,
                    type=OptionsOrderType.MARKET,
                    option_type='call',
                    strike=long_call.strike,
                    expiration=long_call.expiration,
                    client_order_id=f"BULL_SPREAD_LONG_{position_id}"
                )
                
                # Sell short call
                short_order = OptionsOrderRequest(
                    symbol=short_call.symbol,
                    underlying=underlying,
                    qty=quantity,
                    side=OrderSide.SELL,
                    type=OptionsOrderType.MARKET,
                    option_type='call',
                    strike=short_call.strike,
                    expiration=short_call.expiration,
                    client_order_id=f"BULL_SPREAD_SHORT_{position_id}"
                )
                
                # Execute both legs
                long_response = await self.options_broker.submit_options_order(long_order)
                short_response = await self.options_broker.submit_options_order(short_order)
                
                executed_orders = [long_response, short_response]
                
                if long_response.avg_fill_price and short_response.avg_fill_price:
                    total_cost = (long_response.avg_fill_price - short_response.avg_fill_price) * quantity * 100
                    executed_contracts = contracts
                
            elif strategy == OptionsStrategy.BEAR_PUT_SPREAD:
                # Buy higher strike put, sell lower strike put
                long_put, short_put = contracts[0], contracts[1]
                if long_put.strike < short_put.strike:
                    long_put, short_put = short_put, long_put
                
                # Buy long put
                long_order = OptionsOrderRequest(
                    symbol=long_put.symbol,
                    underlying=underlying,
                    qty=quantity,
                    side=OrderSide.BUY,
                    type=OptionsOrderType.MARKET,
                    option_type='put',
                    strike=long_put.strike,
                    expiration=long_put.expiration,
                    client_order_id=f"BEAR_SPREAD_LONG_{position_id}"
                )
                
                # Sell short put
                short_order = OptionsOrderRequest(
                    symbol=short_put.symbol,
                    underlying=underlying,
                    qty=quantity,
                    side=OrderSide.SELL,
                    type=OptionsOrderType.MARKET,
                    option_type='put',
                    strike=short_put.strike,
                    expiration=short_put.expiration,
                    client_order_id=f"BEAR_SPREAD_SHORT_{position_id}"
                )
                
                # Execute both legs
                long_response = await self.options_broker.submit_options_order(long_order)
                short_response = await self.options_broker.submit_options_order(short_order)
                
                executed_orders = [long_response, short_response]
                
                if long_response.avg_fill_price and short_response.avg_fill_price:
                    total_cost = (long_response.avg_fill_price - short_response.avg_fill_price) * quantity * 100
                    executed_contracts = contracts
                
            elif strategy == OptionsStrategy.STRADDLE:
                # Buy call and put at same strike
                call_contract = next(c for c in contracts if c.option_type == 'call')
                put_contract = next(c for c in contracts if c.option_type == 'put')
                
                # Buy call
                call_order = OptionsOrderRequest(
                    symbol=call_contract.symbol,
                    underlying=underlying,
                    qty=quantity,
                    side=OrderSide.BUY,
                    type=OptionsOrderType.MARKET,
                    option_type='call',
                    strike=call_contract.strike,
                    expiration=call_contract.expiration,
                    client_order_id=f"STRADDLE_CALL_{position_id}"
                )
                
                # Buy put
                put_order = OptionsOrderRequest(
                    symbol=put_contract.symbol,
                    underlying=underlying,
                    qty=quantity,
                    side=OrderSide.BUY,
                    type=OptionsOrderType.MARKET,
                    option_type='put',
                    strike=put_contract.strike,
                    expiration=put_contract.expiration,
                    client_order_id=f"STRADDLE_PUT_{position_id}"
                )
                
                # Execute both legs
                call_response = await self.options_broker.submit_options_order(call_order)
                put_response = await self.options_broker.submit_options_order(put_order)
                
                executed_orders = [call_response, put_response]
                
                if call_response.avg_fill_price and put_response.avg_fill_price:
                    total_cost = (call_response.avg_fill_price + put_response.avg_fill_price) * quantity * 100
                    executed_contracts = contracts
            
            # Verify at least one order executed
            if not executed_orders or not any(order.avg_fill_price for order in executed_orders):
                logger.warning(f"No options orders filled for {strategy}")
                return None
            
            # Create position tracking
            position = OptionsPosition(
                symbol=position_id,
                underlying=underlying,
                strategy=strategy,
                contracts=executed_contracts,
                quantity=quantity,
                entry_price=total_cost / (quantity * 100) if total_cost > 0 else 0,
                entry_time=datetime.now(),
                stop_loss=total_cost * 0.5,  # 50% stop loss
                take_profit=total_cost * 2.0,  # 100% profit target
                max_loss=total_cost  # Maximum loss is premium paid
            )
            
            self.positions[position_id] = position
            
            logger.info(f"REAL OPTIONS TRADE: {strategy} for {underlying} - {quantity} contracts @ ${total_cost:.2f}")
            return position
            
        except Exception as e:
            logger.error(f"Error executing options strategy {strategy}: {e}")
            return None
    
    async def monitor_options_positions(self) -> List[Dict]:
        """Monitor and manage existing options positions"""
        
        actions_taken = []
        
        for position_id, position in list(self.positions.items()):
            try:
                # Get current option prices (simplified)
                current_value = await self.get_position_value(position)
                
                if current_value is None:
                    continue
                
                # Calculate P&L
                pnl = current_value - (position.entry_price * position.quantity * 100)
                pnl_percent = pnl / (position.entry_price * position.quantity * 100)
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                # 1. Stop loss
                if position.stop_loss and current_value <= position.stop_loss:
                    should_exit = True
                    exit_reason = "Stop Loss"
                
                # 2. Take profit
                elif position.take_profit and current_value >= position.take_profit:
                    should_exit = True
                    exit_reason = "Take Profit"
                
                # 3. Time decay (close if < 7 days to expiry and losing, or < 3 days regardless)
                elif (position.contracts[0].days_to_expiry <= 7 and pnl < 0) or position.contracts[0].days_to_expiry <= 3:
                    should_exit = True
                    exit_reason = "Time Decay"
                
                # 4. Large profit (take profit at 200% gain)
                elif pnl_percent >= 2.0:
                    should_exit = True
                    exit_reason = "Large Profit"
                
                # 5. Large loss (stop at 80% loss)
                elif pnl_percent <= -0.8:
                    should_exit = True
                    exit_reason = "Large Loss"
                
                if should_exit:
                    await self.close_position(position_id, exit_reason)
                    actions_taken.append({
                        'action': 'CLOSE',
                        'position_id': position_id,
                        'reason': exit_reason,
                        'pnl': pnl,
                        'pnl_percent': pnl_percent
                    })
                
            except Exception as e:
                logger.error(f"Error monitoring position {position_id}: {e}")
        
        return actions_taken
    
    async def get_position_value(self, position: OptionsPosition) -> Optional[float]:
        """Get current value of an options position"""
        try:
            # Refresh options chain to get current prices
            contracts = await self.get_options_chain(position.underlying)
            if not contracts:
                return None
            
            total_value = 0
            for contract in position.contracts:
                # Find matching contract in current chain
                current_contract = next(
                    (c for c in contracts if c.symbol == contract.symbol), 
                    None
                )
                if current_contract:
                    total_value += current_contract.bid * position.quantity * 100
            
            return total_value if total_value > 0 else None
            
        except Exception as e:
            logger.error(f"Error getting position value: {e}")
            return None
    
    async def close_position(self, position_id: str, reason: str = "Manual"):
        """Close an options position"""
        
        if position_id not in self.positions:
            return False
        
        position = self.positions[position_id]
        
        try:
            # Get current value
            current_value = await self.get_position_value(position)
            
            # Calculate final P&L
            if current_value:
                position.pnl = current_value - (position.entry_price * position.quantity * 100)
                position.exit_price = current_value / (position.quantity * 100)
            else:
                position.pnl = -(position.entry_price * position.quantity * 100)  # Total loss
                position.exit_price = 0
            
            position.exit_time = datetime.now()
            
            # Execute REAL options closing orders
            try:
                close_orders = []
                
                if position.strategy == OptionsStrategy.LONG_CALL:
                    # Sell the call to close
                    for contract in position.contracts:
                        close_order = OptionsOrderRequest(
                            symbol=contract.symbol,
                            underlying=position.underlying,
                            qty=position.quantity,
                            side=OrderSide.SELL,
                            type=OptionsOrderType.MARKET,
                            option_type='call',
                            strike=contract.strike,
                            expiration=contract.expiration,
                            client_order_id=f"CLOSE_CALL_{position_id}"
                        )
                        close_response = await self.options_broker.submit_options_order(close_order)
                        close_orders.append(close_response)
                
                elif position.strategy == OptionsStrategy.LONG_PUT:
                    # Sell the put to close
                    for contract in position.contracts:
                        close_order = OptionsOrderRequest(
                            symbol=contract.symbol,
                            underlying=position.underlying,
                            qty=position.quantity,
                            side=OrderSide.SELL,
                            type=OptionsOrderType.MARKET,
                            option_type='put',
                            strike=contract.strike,
                            expiration=contract.expiration,
                            client_order_id=f"CLOSE_PUT_{position_id}"
                        )
                        close_response = await self.options_broker.submit_options_order(close_order)
                        close_orders.append(close_response)
                
                elif position.strategy in [OptionsStrategy.BULL_CALL_SPREAD, OptionsStrategy.BEAR_PUT_SPREAD]:
                    # Close both legs of the spread (reverse original trades)
                    for i, contract in enumerate(position.contracts):
                        # For spreads, we need to reverse the original position
                        # First contract was bought, second was sold (or vice versa)
                        close_side = OrderSide.BUY if i == 1 else OrderSide.SELL  # Reverse original
                        
                        close_order = OptionsOrderRequest(
                            symbol=contract.symbol,
                            underlying=position.underlying,
                            qty=position.quantity,
                            side=close_side,
                            type=OptionsOrderType.MARKET,
                            option_type=contract.option_type,
                            strike=contract.strike,
                            expiration=contract.expiration,
                            client_order_id=f"CLOSE_SPREAD_{i}_{position_id}"
                        )
                        close_response = await self.options_broker.submit_options_order(close_order)
                        close_orders.append(close_response)
                
                elif position.strategy == OptionsStrategy.STRADDLE:
                    # Sell both the call and put
                    for contract in position.contracts:
                        close_order = OptionsOrderRequest(
                            symbol=contract.symbol,
                            underlying=position.underlying,
                            qty=position.quantity,
                            side=OrderSide.SELL,
                            type=OptionsOrderType.MARKET,
                            option_type=contract.option_type,
                            strike=contract.strike,
                            expiration=contract.expiration,
                            client_order_id=f"CLOSE_STRADDLE_{contract.option_type.upper()}_{position_id}"
                        )
                        close_response = await self.options_broker.submit_options_order(close_order)
                        close_orders.append(close_response)
                
                # Update P&L based on actual closing prices
                if close_orders and any(order.avg_fill_price for order in close_orders):
                    total_close_value = sum(
                        (order.avg_fill_price or 0) * order.filled_qty * 100 
                        for order in close_orders
                    )
                    position.pnl = total_close_value - (position.entry_price * position.quantity * 100)
                    position.exit_price = total_close_value / (position.quantity * 100)
                        
            except Exception as e:
                logger.error(f"Error closing options position through broker: {e}")
            
            logger.info(f"Closed position {position_id}: {reason}, P&L: ${position.pnl:.2f}")
            
            # Remove from active positions
            del self.positions[position_id]
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return False
    
    def get_positions_summary(self) -> Dict:
        """Get summary of all options positions"""
        
        if not self.positions:
            return {
                'total_positions': 0,
                'total_pnl': 0,
                'open_positions': []
            }
        
        total_pnl = 0
        open_positions = []
        
        for position_id, position in self.positions.items():
            position_info = {
                'id': position_id,
                'underlying': position.underlying,
                'strategy': position.strategy,
                'quantity': position.quantity,
                'entry_time': position.entry_time,
                'entry_price': position.entry_price,
                'days_held': (datetime.now() - position.entry_time).days,
                'unrealized_pnl': position.pnl if position.pnl else 0
            }
            open_positions.append(position_info)
            
            if position.pnl:
                total_pnl += position.pnl
        
        return {
            'total_positions': len(self.positions),
            'total_pnl': total_pnl,
            'open_positions': open_positions
        }