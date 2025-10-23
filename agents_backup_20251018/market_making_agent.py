#!/usr/bin/env python3
"""
Market Making Agent - HIVE TRADE
Automated market making with dynamic spread management and inventory control
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import time
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketMakingMode(Enum):
    """Market making operation modes"""
    AGGRESSIVE = "AGGRESSIVE"       # Tight spreads, high turnover
    CONSERVATIVE = "CONSERVATIVE"   # Wide spreads, low risk
    BALANCED = "BALANCED"          # Moderate spreads and risk
    PASSIVE = "PASSIVE"            # Market following strategy
    ADAPTIVE = "ADAPTIVE"          # Dynamic mode adjustment

@dataclass
class Quote:
    """Market quote structure"""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread: float
    mid_price: float
    timestamp: datetime
    confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'spread': self.spread,
            'mid_price': self.mid_price,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence
        }

@dataclass
class InventoryPosition:
    """Inventory tracking for market making"""
    symbol: str
    net_position: float
    target_position: float
    max_position: float
    avg_cost: float
    unrealized_pnl: float
    inventory_risk: float
    rebalance_urgency: float = 0.0
    
    def calculate_skew(self) -> float:
        """Calculate inventory skew for quote adjustment"""
        if self.max_position == 0:
            return 0.0
        position_ratio = self.net_position / self.max_position
        return np.tanh(position_ratio * 2)  # Smooth skew function

@dataclass
class MarketMakingMetrics:
    """Performance metrics for market making"""
    total_volume_traded: float = 0.0
    total_spread_captured: float = 0.0
    inventory_turnover: float = 0.0
    hit_ratio: float = 0.0  # Percentage of quotes that got filled
    adverse_selection_cost: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_inventory: float = 0.0
    avg_spread: float = 0.0

class MarketMakingAgent:
    """Advanced market making agent with dynamic spread management"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 50000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Market making parameters
        self.mode = MarketMakingMode.BALANCED
        self.base_spread_bps = 10  # Base spread in basis points
        self.quote_size_usd = 1000  # Base quote size in USD
        self.max_position_pct = 20  # Max position as % of capital
        self.inventory_target_pct = 0  # Target inventory level
        self.risk_aversion = 0.5  # Risk aversion parameter (0-1)
        
        # Dynamic parameters
        self.volatility_multiplier = 1.0
        self.volume_multiplier = 1.0
        self.competition_factor = 1.0
        self.adverse_selection_protection = 0.2
        
        # State tracking
        self.positions: Dict[str, InventoryPosition] = {}
        self.active_quotes: Dict[str, Quote] = {}
        self.market_data: Dict[str, Dict] = {}
        self.metrics: MarketMakingMetrics = MarketMakingMetrics()
        self.trade_history: List[Dict] = []
        
        # Initialize positions
        for symbol in symbols:
            self.positions[symbol] = InventoryPosition(
                symbol=symbol,
                net_position=0.0,
                target_position=0.0,
                max_position=initial_capital * (self.max_position_pct / 100) / 100,  # Assuming $100 avg price
                avg_cost=0.0,
                unrealized_pnl=0.0,
                inventory_risk=0.0
            )
        
        # Market making statistics
        self.quote_updates = 0
        self.fills_count = 0
        self.last_quote_time = {}
        
        logger.info(f"Market Making Agent initialized for {len(symbols)} symbols")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Mode: {self.mode.value}")

    def update_market_data(self, symbol: str, price: float, volume: float, 
                          bid: float = None, ask: float = None, volatility: float = None):
        """Update market data for a symbol"""
        if symbol not in self.market_data:
            self.market_data[symbol] = {
                'price_history': [],
                'volume_history': [],
                'volatility_history': []
            }
        
        # Store market data
        current_time = datetime.now()
        self.market_data[symbol].update({
            'last_price': price,
            'last_volume': volume,
            'bid': bid or price * 0.999,
            'ask': ask or price * 1.001,
            'volatility': volatility or 0.02,
            'timestamp': current_time
        })
        
        # Update histories
        self.market_data[symbol]['price_history'].append((current_time, price))
        self.market_data[symbol]['volume_history'].append((current_time, volume))
        if volatility:
            self.market_data[symbol]['volatility_history'].append((current_time, volatility))
        
        # Trim histories to last 100 points
        for hist_key in ['price_history', 'volume_history', 'volatility_history']:
            if len(self.market_data[symbol][hist_key]) > 100:
                self.market_data[symbol][hist_key] = self.market_data[symbol][hist_key][-100:]

    def calculate_optimal_spread(self, symbol: str) -> float:
        """Calculate optimal spread based on market conditions"""
        if symbol not in self.market_data:
            return self.base_spread_bps / 10000  # Default spread
        
        market = self.market_data[symbol]
        base_spread = self.base_spread_bps / 10000
        
        # Volatility adjustment
        volatility = market.get('volatility', 0.02)
        vol_adjustment = min(volatility * 10, 3.0)  # Cap at 3x
        
        # Volume adjustment (less volume = wider spreads)
        recent_volume = market.get('last_volume', 1000)
        avg_volume = 10000  # Assume average volume
        volume_adjustment = max(0.5, min(2.0, avg_volume / max(recent_volume, 100)))
        
        # Inventory adjustment
        position = self.positions[symbol]
        inventory_skew = abs(position.calculate_skew())
        inventory_adjustment = 1.0 + (inventory_skew * 0.5)
        
        # Mode adjustment
        mode_multipliers = {
            MarketMakingMode.AGGRESSIVE: 0.7,
            MarketMakingMode.CONSERVATIVE: 1.5,
            MarketMakingMode.BALANCED: 1.0,
            MarketMakingMode.PASSIVE: 1.3,
            MarketMakingMode.ADAPTIVE: self._calculate_adaptive_multiplier(symbol)
        }
        
        mode_multiplier = mode_multipliers[self.mode]
        
        # Calculate final spread
        optimal_spread = (base_spread * vol_adjustment * volume_adjustment * 
                         inventory_adjustment * mode_multiplier)
        
        # Apply minimum and maximum limits
        min_spread = 0.0005  # 5 bps minimum
        max_spread = 0.01    # 100 bps maximum
        
        return max(min_spread, min(max_spread, optimal_spread))

    def _calculate_adaptive_multiplier(self, symbol: str) -> float:
        """Calculate adaptive multiplier based on recent performance"""
        # Check recent hit ratio
        recent_fills = len([t for t in self.trade_history[-20:] if t.get('symbol') == symbol])
        recent_quotes = 20  # Assume 20 recent quote updates
        
        if recent_quotes == 0:
            return 1.0
        
        hit_ratio = recent_fills / recent_quotes
        
        # If hit ratio too high, widen spreads; if too low, tighten spreads
        target_hit_ratio = 0.3
        if hit_ratio > target_hit_ratio * 1.5:
            return 1.3  # Widen spreads
        elif hit_ratio < target_hit_ratio * 0.5:
            return 0.8  # Tighten spreads
        else:
            return 1.0

    def calculate_quote_size(self, symbol: str, side: str) -> float:
        """Calculate optimal quote size for bid/ask"""
        base_size_usd = self.quote_size_usd
        current_price = self.market_data[symbol].get('last_price', 100)
        base_size = base_size_usd / current_price
        
        # Inventory adjustment
        position = self.positions[symbol]
        inventory_skew = position.calculate_skew()
        
        # Reduce size on inventory-heavy side
        if side == 'bid' and inventory_skew > 0:
            size_multiplier = max(0.3, 1.0 - inventory_skew)
        elif side == 'ask' and inventory_skew < 0:
            size_multiplier = max(0.3, 1.0 + inventory_skew)
        else:
            size_multiplier = 1.0
        
        # Volume-based adjustment
        recent_volume = self.market_data[symbol].get('last_volume', 1000)
        volume_factor = min(2.0, max(0.5, recent_volume / 5000))
        
        final_size = base_size * size_multiplier * volume_factor
        
        # Apply position limits
        max_additional = position.max_position - abs(position.net_position)
        final_size = min(final_size, max_additional)
        
        return max(0.01, final_size)  # Minimum size

    def generate_quotes(self) -> Dict[str, Quote]:
        """Generate optimal bid/ask quotes for all symbols"""
        quotes = {}
        current_time = datetime.now()
        
        for symbol in self.symbols:
            if symbol not in self.market_data:
                continue
            
            market = self.market_data[symbol]
            last_price = market.get('last_price', 100)
            
            # Calculate spread
            spread_ratio = self.calculate_optimal_spread(symbol)
            spread_amount = last_price * spread_ratio
            
            # Calculate mid price (can be skewed based on inventory)
            position = self.positions[symbol]
            inventory_skew = position.calculate_skew()
            skew_adjustment = spread_amount * inventory_skew * 0.3
            
            mid_price = last_price - skew_adjustment
            
            # Generate bid/ask
            bid_price = mid_price - (spread_amount / 2)
            ask_price = mid_price + (spread_amount / 2)
            
            # Calculate sizes
            bid_size = self.calculate_quote_size(symbol, 'bid')
            ask_size = self.calculate_quote_size(symbol, 'ask')
            
            # Create quote
            quote = Quote(
                symbol=symbol,
                bid_price=round(bid_price, 4),
                ask_price=round(ask_price, 4),
                bid_size=round(bid_size, 6),
                ask_size=round(ask_size, 6),
                spread=spread_amount,
                mid_price=mid_price,
                timestamp=current_time,
                confidence=self._calculate_quote_confidence(symbol)
            )
            
            quotes[symbol] = quote
            self.last_quote_time[symbol] = current_time
        
        self.active_quotes = quotes
        self.quote_updates += len(quotes)
        
        return quotes

    def _calculate_quote_confidence(self, symbol: str) -> float:
        """Calculate confidence in quote based on market conditions"""
        base_confidence = 0.8
        
        # Reduce confidence in volatile markets
        volatility = self.market_data[symbol].get('volatility', 0.02)
        vol_penalty = min(0.3, volatility * 5)
        
        # Reduce confidence if inventory is extreme
        position = self.positions[symbol]
        inventory_penalty = abs(position.calculate_skew()) * 0.2
        
        # Reduce confidence if data is stale
        last_update = self.market_data[symbol].get('timestamp', datetime.now() - timedelta(minutes=10))
        staleness_minutes = (datetime.now() - last_update).total_seconds() / 60
        staleness_penalty = min(0.4, staleness_minutes * 0.05)
        
        final_confidence = base_confidence - vol_penalty - inventory_penalty - staleness_penalty
        return max(0.1, final_confidence)

    def simulate_fill(self, symbol: str, side: str, price: float, size: float) -> Dict:
        """Simulate a fill and update inventory"""
        fill_time = datetime.now()
        
        # Update position
        position = self.positions[symbol]
        
        if side == 'buy':
            new_position = position.net_position + size
            cost_impact = price * size
        else:  # sell
            new_position = position.net_position - size
            cost_impact = -price * size
        
        # Update average cost
        if position.net_position == 0:
            new_avg_cost = price
        else:
            total_cost = position.avg_cost * abs(position.net_position) + cost_impact
            total_size = abs(new_position)
            new_avg_cost = total_cost / total_size if total_size > 0 else price
        
        # Update position
        position.net_position = new_position
        position.avg_cost = new_avg_cost
        
        # Calculate realized P&L for this trade
        if side == 'sell' and position.net_position >= 0:
            # Closing long position
            realized_pnl = (price - position.avg_cost) * size
        elif side == 'buy' and position.net_position <= 0:
            # Closing short position  
            realized_pnl = (position.avg_cost - price) * size
        else:
            realized_pnl = 0
        
        # Record trade
        trade = {
            'timestamp': fill_time.isoformat(),
            'symbol': symbol,
            'side': side,
            'price': price,
            'size': size,
            'realized_pnl': realized_pnl,
            'new_position': new_position,
            'avg_cost': new_avg_cost
        }
        
        self.trade_history.append(trade)
        self.fills_count += 1
        
        # Update metrics
        self.metrics.total_volume_traded += price * size
        self.metrics.realized_pnl += realized_pnl
        
        # Update spread capture (simplified)
        if symbol in self.active_quotes:
            quote = self.active_quotes[symbol]
            if side == 'buy' and price >= quote.bid_price:
                spread_captured = price - quote.mid_price
            elif side == 'sell' and price <= quote.ask_price:
                spread_captured = quote.mid_price - price
            else:
                spread_captured = 0
            
            self.metrics.total_spread_captured += spread_captured * size
        
        logger.info(f"Fill: {symbol} {side} {size:.4f} @ ${price:.4f}, "
                   f"New position: {new_position:.4f}, Realized P&L: ${realized_pnl:.2f}")
        
        return trade

    def calculate_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L across all positions"""
        total_unrealized = 0.0
        
        for symbol, position in self.positions.items():
            if abs(position.net_position) < 1e-8:  # No position
                continue
            
            current_price = self.market_data.get(symbol, {}).get('last_price', position.avg_cost)
            
            if position.net_position > 0:
                # Long position
                unrealized = (current_price - position.avg_cost) * position.net_position
            else:
                # Short position
                unrealized = (position.avg_cost - current_price) * abs(position.net_position)
            
            position.unrealized_pnl = unrealized
            total_unrealized += unrealized
        
        self.metrics.unrealized_pnl = total_unrealized
        return total_unrealized

    def calculate_inventory_risk(self) -> float:
        """Calculate inventory risk across all positions"""
        total_risk = 0.0
        
        for symbol, position in self.positions.items():
            if abs(position.net_position) < 1e-8:
                continue
            
            current_price = self.market_data.get(symbol, {}).get('last_price', position.avg_cost)
            volatility = self.market_data.get(symbol, {}).get('volatility', 0.02)
            
            # Position risk = |position_value| * volatility
            position_value = abs(position.net_position * current_price)
            risk = position_value * volatility
            
            position.inventory_risk = risk
            total_risk += risk
        
        return total_risk

    def should_rebalance_inventory(self, symbol: str) -> bool:
        """Determine if inventory rebalancing is needed"""
        position = self.positions[symbol]
        
        # Check if position is near limits
        position_ratio = abs(position.net_position) / position.max_position
        if position_ratio > 0.8:
            position.rebalance_urgency = 0.8
            return True
        
        # Check if position is far from target
        target_deviation = abs(position.net_position - position.target_position)
        if target_deviation > position.max_position * 0.3:
            position.rebalance_urgency = 0.6
            return True
        
        # Check risk-adjusted criteria
        inventory_risk_ratio = position.inventory_risk / self.current_capital
        if inventory_risk_ratio > 0.1:  # 10% risk threshold
            position.rebalance_urgency = 0.9
            return True
        
        position.rebalance_urgency = 0.0
        return False

    def get_market_making_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive market making dashboard data"""
        # Update metrics
        self.calculate_unrealized_pnl()
        inventory_risk = self.calculate_inventory_risk()
        
        # Calculate hit ratio
        recent_trades = len([t for t in self.trade_history[-50:]])
        recent_quotes = min(50, self.quote_updates)
        hit_ratio = recent_trades / recent_quotes if recent_quotes > 0 else 0
        
        # Calculate average spread
        if self.active_quotes:
            avg_spread = np.mean([q.spread / q.mid_price for q in self.active_quotes.values()]) * 10000
        else:
            avg_spread = 0
        
        # Position summary
        positions_summary = {}
        for symbol, position in self.positions.items():
            current_price = self.market_data.get(symbol, {}).get('last_price', position.avg_cost)
            positions_summary[symbol] = {
                'net_position': round(position.net_position, 6),
                'market_value': round(position.net_position * current_price, 2),
                'unrealized_pnl': round(position.unrealized_pnl, 2),
                'avg_cost': round(position.avg_cost, 4),
                'current_price': round(current_price, 4),
                'inventory_risk': round(position.inventory_risk, 2),
                'skew': round(position.calculate_skew(), 3),
                'rebalance_urgency': round(position.rebalance_urgency, 2)
            }
        
        # Active quotes summary
        quotes_summary = {}
        for symbol, quote in self.active_quotes.items():
            quotes_summary[symbol] = {
                'bid': f"${quote.bid_price:.4f} x {quote.bid_size:.2f}",
                'ask': f"${quote.ask_price:.4f} x {quote.ask_size:.2f}",
                'spread_bps': round((quote.spread / quote.mid_price) * 10000, 1),
                'confidence': round(quote.confidence, 2)
            }
        
        total_pnl = self.metrics.realized_pnl + self.metrics.unrealized_pnl
        
        return {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode.value,
            'performance_metrics': {
                'total_pnl': round(total_pnl, 2),
                'realized_pnl': round(self.metrics.realized_pnl, 2),
                'unrealized_pnl': round(self.metrics.unrealized_pnl, 2),
                'total_volume_traded': round(self.metrics.total_volume_traded, 2),
                'spread_captured': round(self.metrics.total_spread_captured, 4),
                'hit_ratio': round(hit_ratio, 3),
                'avg_spread_bps': round(avg_spread, 1),
                'inventory_risk': round(inventory_risk, 2),
                'capital_utilization': round((inventory_risk / self.current_capital) * 100, 1)
            },
            'positions': positions_summary,
            'active_quotes': quotes_summary,
            'trade_count': len(self.trade_history),
            'quote_updates': self.quote_updates,
            'symbols_active': len([s for s in self.symbols if s in self.market_data])
        }

    async def run_market_making_cycle(self):
        """Run one complete market making cycle"""
        try:
            # Generate new quotes
            quotes = self.generate_quotes()
            
            # Log quote generation
            if quotes:
                logger.info(f"Generated {len(quotes)} quotes")
                for symbol, quote in quotes.items():
                    spread_bps = (quote.spread / quote.mid_price) * 10000
                    logger.debug(f"{symbol}: ${quote.bid_price:.4f} x {quote.ask_price:.4f} "
                               f"(spread: {spread_bps:.1f}bps)")
            
            # Check for inventory rebalancing needs
            for symbol in self.symbols:
                if self.should_rebalance_inventory(symbol):
                    logger.info(f"Inventory rebalancing recommended for {symbol}")
            
            # Update inventory risk
            total_risk = self.calculate_inventory_risk()
            if total_risk > self.current_capital * 0.15:
                logger.warning(f"High inventory risk: ${total_risk:.2f} "
                             f"({(total_risk/self.current_capital)*100:.1f}% of capital)")
            
            return quotes
            
        except Exception as e:
            logger.error(f"Error in market making cycle: {e}")
            return {}

def main():
    """Demonstrate market making agent"""
    print("HIVE TRADE - Market Making Agent Demo")
    print("=" * 50)
    
    # Initialize agent
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD']
    agent = MarketMakingAgent(symbols, initial_capital=100000)
    
    # Simulate market data and trading
    print("\nSimulating market making operations...")
    
    # Add market data
    market_prices = {
        'AAPL': 180.50, 'MSFT': 340.00, 'GOOGL': 2800.00,
        'BTC-USD': 45000.00, 'ETH-USD': 4800.00
    }
    
    for symbol, price in market_prices.items():
        volatility = np.random.uniform(0.15, 0.5) if 'USD' in symbol else np.random.uniform(0.02, 0.05)
        volume = np.random.uniform(10000, 100000)
        
        agent.update_market_data(
            symbol=symbol,
            price=price,
            volume=volume,
            volatility=volatility
        )
    
    # Generate initial quotes
    quotes = agent.generate_quotes()
    print(f"\nGenerated quotes for {len(quotes)} symbols:")
    for symbol, quote in quotes.items():
        spread_bps = (quote.spread / quote.mid_price) * 10000
        print(f"  {symbol}: ${quote.bid_price:.4f} x ${quote.ask_price:.4f} "
              f"(spread: {spread_bps:.1f}bps, confidence: {quote.confidence:.2f})")
    
    # Simulate some fills
    print("\nSimulating market making fills...")
    fill_scenarios = [
        ('AAPL', 'buy', 180.48, 10),
        ('AAPL', 'sell', 180.52, 8),
        ('BTC-USD', 'buy', 44990, 0.1),
        ('MSFT', 'sell', 340.20, 5),
        ('GOOGL', 'buy', 2799.50, 2)
    ]
    
    for symbol, side, price, size in fill_scenarios:
        agent.simulate_fill(symbol, side, price, size)
    
    # Get dashboard
    dashboard = agent.get_market_making_dashboard()
    
    # Display results
    print(f"\nMARKET MAKING DASHBOARD:")
    perf = dashboard['performance_metrics']
    print(f"  Total P&L: ${perf['total_pnl']:,.2f}")
    print(f"  Realized P&L: ${perf['realized_pnl']:,.2f}")
    print(f"  Unrealized P&L: ${perf['unrealized_pnl']:,.2f}")
    print(f"  Volume Traded: ${perf['total_volume_traded']:,.2f}")
    print(f"  Spread Captured: ${perf['spread_captured']:.4f}")
    print(f"  Hit Ratio: {perf['hit_ratio']:.1%}")
    print(f"  Avg Spread: {perf['avg_spread_bps']:.1f} bps")
    print(f"  Inventory Risk: ${perf['inventory_risk']:,.2f} ({perf['capital_utilization']:.1f}%)")
    
    print(f"\nPOSITIONS:")
    for symbol, pos_data in dashboard['positions'].items():
        if abs(pos_data['net_position']) > 1e-6:
            print(f"  {symbol}: {pos_data['net_position']:+.6f} units, "
                  f"Value: ${pos_data['market_value']:+,.2f}, "
                  f"P&L: ${pos_data['unrealized_pnl']:+,.2f}")
    
    print(f"\nACTIVE QUOTES:")
    for symbol, quote_data in dashboard['active_quotes'].items():
        print(f"  {symbol}: {quote_data['bid']} | {quote_data['ask']} "
              f"({quote_data['spread_bps']:.1f}bps)")
    
    # Save results
    results_file = 'market_making_results.json'
    with open(results_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("Market Making Agent demonstration completed!")

if __name__ == "__main__":
    main()