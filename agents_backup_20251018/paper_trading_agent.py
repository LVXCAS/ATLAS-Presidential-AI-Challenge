#!/usr/bin/env python3
"""
Paper Trading Agent - Task 9.1 Implementation

This agent implements comprehensive paper trading capabilities:
- Paper trading simulation mode with realistic market simulation
- Realistic order execution simulation with slippage and commissions
- Paper trading performance tracking and analytics
- Seamless switch between paper and live trading modes
- Integration with existing trading infrastructure

Requirements: Requirement 5 (Paper Trading Validation)
Task: 9.1 Paper Trading Mode
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from decimal import Decimal
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# LangGraph imports
from langgraph.graph import StateGraph, END

# Trading system imports
from agents.broker_integration import (
    AlpacaBrokerIntegration, OrderRequest, OrderResponse, OrderSide, 
    OrderType, TimeInForce, OrderStatus
)
from agents.execution_engine_agent import ExecutionEngineAgent, ExecutionOrder, ExecutionAlgorithm
from agents.market_data_ingestor import MarketDataIngestorAgent
from agents.portfolio_allocator_agent import PortfolioAllocatorAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.performance_monitoring_agent import PerformanceMonitoringAgent
from agents.trade_logging_audit_agent import (
    TradeLoggingAuditAgent, ActionType, EntityType, LogLevel
)

# Configuration
from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


class TradingMode(Enum):
    """Trading mode enumeration"""
    PAPER = "paper"
    LIVE = "live"
    SIMULATION = "simulation"


class PaperTradingStatus(Enum):
    """Paper trading status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PaperTradingConfig:
    """Paper trading configuration"""
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_trades: int = 100
    max_daily_loss: float = 0.05  # 5% daily loss limit
    commission_rate: float = 0.001  # 0.1% commission
    slippage_model: str = "realistic"  # realistic, aggressive, conservative
    market_impact_model: str = "square_root"  # square_root, linear, none
    risk_limits_enforced: bool = True
    performance_tracking: bool = True
    trade_logging: bool = True
    backup_frequency_hours: int = 24


@dataclass
class PaperTradingPosition:
    """Paper trading position data"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    last_update: datetime
    strategy: str
    agent_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperTradingOrder:
    """Paper trading order data"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    submitted_at: datetime
    filled_at: Optional[datetime]
    filled_price: Optional[float]
    filled_quantity: float
    commission: float
    slippage: float
    strategy: str
    agent_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperTradingPortfolio:
    """Paper trading portfolio data"""
    cash: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    daily_pnl: float
    positions: Dict[str, PaperTradingPosition]
    orders: List[PaperTradingOrder]
    trades: List[Dict[str, Any]]
    last_update: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperTradingPerformance:
    """Paper trading performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    daily_returns: List[float]
    equity_curve: List[float]
    drawdown_curve: List[float]
    timestamp: datetime


class MarketSimulator:
    """Realistic market simulation for paper trading"""
    
    def __init__(self, config: PaperTradingConfig):
        self.config = config
        self.market_data_cache = {}
        self.price_impact_cache = {}
        
        # Market simulation parameters
        self.volatility_multiplier = 1.0
        self.spread_multiplier = 1.0
        self.volume_multiplier = 1.0
        
        logger.info("Market Simulator initialized")
    
    def simulate_market_data(self, symbol: str, base_price: float, 
                           volume: float, timestamp: datetime) -> Dict[str, Any]:
        """Simulate realistic market data"""
        try:
            # Generate realistic price movement
            volatility = self._calculate_volatility(symbol, base_price)
            price_change = np.random.normal(0, volatility)
            
            # Add market impact for large orders
            market_impact = self._calculate_market_impact(symbol, volume, base_price)
            price_change += market_impact
            
            # Calculate new price
            new_price = base_price * (1 + price_change)
            
            # Ensure positive price
            new_price = max(new_price, 0.01)
            
            # Generate bid/ask spread
            spread_pct = self._calculate_spread(symbol, base_price)
            bid_price = new_price * (1 - spread_pct / 2)
            ask_price = new_price * (1 + spread_pct / 2)
            
            # Simulate volume
            simulated_volume = volume * (1 + np.random.normal(0, 0.2))
            simulated_volume = max(simulated_volume, 100)
            
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'open': base_price,
                'high': max(base_price, new_price),
                'low': min(base_price, new_price),
                'close': new_price,
                'volume': simulated_volume,
                'bid': bid_price,
                'ask': ask_price,
                'spread': ask_price - bid_price,
                'volatility': volatility,
                'market_impact': market_impact
            }
            
        except Exception as e:
            logger.error(f"Error simulating market data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'open': base_price,
                'high': base_price,
                'low': base_price,
                'close': base_price,
                'volume': volume,
                'bid': base_price * 0.999,
                'ask': base_price * 1.001,
                'spread': base_price * 0.002,
                'volatility': 0.01,
                'market_impact': 0.0
            }
    
    def _calculate_volatility(self, symbol: str, price: float) -> float:
        """Calculate realistic volatility for symbol"""
        # Base volatility based on price
        base_volatility = 0.02 if price > 100 else 0.03 if price > 50 else 0.05
        
        # Adjust for symbol characteristics
        if symbol in ['AAPL', 'MSFT', 'GOOGL']:
            base_volatility *= 0.8  # Large caps are less volatile
        elif symbol in ['TSLA', 'NVDA']:
            base_volatility *= 1.5  # Tech stocks are more volatile
        
        # Add random variation
        volatility = base_volatility * (1 + np.random.normal(0, 0.2))
        
        return max(volatility, 0.005)  # Minimum 0.5% volatility
    
    def _calculate_spread(self, symbol: str, price: float) -> float:
        """Calculate realistic bid-ask spread"""
        # Base spread based on price
        if price > 100:
            base_spread = 0.001  # 0.1% for high-priced stocks
        elif price > 50:
            base_spread = 0.002  # 0.2% for medium-priced stocks
        else:
            base_spread = 0.005  # 0.5% for low-priced stocks
        
        # Adjust for symbol liquidity
        if symbol in ['SPY', 'QQQ', 'IWM']:
            base_spread *= 0.5  # ETFs have tighter spreads
        elif symbol in ['TSLA', 'NVDA']:
            base_spread *= 1.2  # High-volume stocks have tighter spreads
        
        # Add random variation
        spread = base_spread * (1 + np.random.normal(0, 0.3))
        
        return max(spread, 0.0005)  # Minimum 0.05% spread
    
    def _calculate_market_impact(self, symbol: str, volume: float, price: float) -> float:
        """Calculate market impact of order"""
        if self.config.market_impact_model == "none":
            return 0.0
        
        # Square root law for market impact
        daily_volume = self._estimate_daily_volume(symbol)
        participation_rate = volume / daily_volume if daily_volume > 0 else 0
        
        if self.config.market_impact_model == "square_root":
            impact = 0.1 * np.sqrt(participation_rate)
        else:  # linear
            impact = 0.05 * participation_rate
        
        # Adjust for price level
        impact *= (100 / price) if price > 0 else 1
        
        # Ensure reasonable bounds
        impact = max(min(impact, 0.05), -0.05)  # ±5% maximum impact
        
        return impact
    
    def _estimate_daily_volume(self, symbol: str) -> float:
        """Estimate daily volume for symbol"""
        # Base estimates for different symbols
        volume_estimates = {
            'SPY': 100000000,  # 100M shares
            'QQQ': 50000000,   # 50M shares
            'AAPL': 80000000,  # 80M shares
            'MSFT': 40000000,  # 40M shares
            'GOOGL': 20000000, # 20M shares
            'TSLA': 60000000,  # 60M shares
            'NVDA': 30000000,  # 30M shares
        }
        
        return volume_estimates.get(symbol, 10000000)  # Default 10M shares


class OrderExecutionSimulator:
    """Realistic order execution simulation for paper trading"""
    
    def __init__(self, config: PaperTradingConfig, market_simulator: MarketSimulator):
        self.config = config
        self.market_simulator = market_simulator
        self.execution_delays = {
            OrderType.MARKET: (0.1, 0.5),      # 0.1-0.5 seconds
            OrderType.LIMIT: (1.0, 5.0),       # 1-5 seconds
            OrderType.STOP: (0.5, 2.0),        # 0.5-2 seconds
            OrderType.STOP_LIMIT: (1.0, 3.0),  # 1-3 seconds
        }
        
        logger.info("Order Execution Simulator initialized")
    
    async def simulate_order_execution(self, order: PaperTradingOrder, 
                                     market_data: Dict[str, Any]) -> PaperTradingOrder:
        """Simulate realistic order execution"""
        try:
            # Calculate execution delay
            min_delay, max_delay = self.execution_delays.get(order.order_type, (1.0, 3.0))
            execution_delay = random.uniform(min_delay, max_delay)
            
            # Simulate execution time
            await asyncio.sleep(execution_delay)
            
            # Calculate fill price based on order type
            fill_price = self._calculate_fill_price(order, market_data)
            
            # Calculate slippage
            slippage = self._calculate_slippage(order, market_data)
            
            # Apply slippage
            if order.side == OrderSide.BUY:
                actual_fill_price = fill_price * (1 + slippage)
            else:
                actual_fill_price = fill_price * (1 - slippage)
            
            # Calculate commission
            commission = self._calculate_commission(order, actual_fill_price)
            
            # Update order with execution details
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            order.filled_price = actual_fill_price
            order.filled_quantity = order.quantity
            order.commission = commission
            order.slippage = slippage
            
            logger.info(f"Order {order.order_id} executed: {order.side.value} {order.quantity} {order.symbol} @ ${actual_fill_price:.4f}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
            order.status = OrderStatus.REJECTED
            return order
    
    def _calculate_fill_price(self, order: PaperTradingOrder, market_data: Dict[str, Any]) -> float:
        """Calculate fill price based on order type and market data"""
        if order.order_type == OrderType.MARKET:
            # Market orders fill at current market price
            if order.side == OrderSide.BUY:
                return market_data['ask']  # Buy at ask
            else:
                return market_data['bid']  # Sell at bid
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill at limit price if marketable
            if order.side == OrderSide.BUY and order.price >= market_data['ask']:
                return market_data['ask']  # Marketable buy
            elif order.side == OrderSide.SELL and order.price <= market_data['bid']:
                return market_data['bid']  # Marketable sell
            else:
                return order.price  # Non-marketable, fill at limit
        
        elif order.order_type == OrderType.STOP:
            # Stop orders become market orders when triggered
            if order.side == OrderSide.BUY and market_data['high'] >= order.stop_price:
                return market_data['ask']  # Stop triggered, buy at ask
            elif order.side == OrderSide.SELL and market_data['low'] <= order.stop_price:
                return market_data['bid']  # Stop triggered, sell at bid
            else:
                return order.price or market_data['close']  # Not triggered
        
        else:
            # Default to current market price
            return market_data['close']
    
    def _calculate_slippage(self, order: PaperTradingOrder, market_data: Dict[str, Any]) -> float:
        """Calculate realistic slippage based on order size and market conditions"""
        # Base slippage
        base_slippage = 0.0005  # 0.05% base slippage
        
        # Size-based slippage
        volume_impact = (order.quantity / market_data['volume']) * 0.01
        size_slippage = min(volume_impact, 0.02)  # Max 2% size impact
        
        # Volatility-based slippage
        volatility_slippage = market_data['volatility'] * 0.1
        
        # Market impact slippage
        market_impact_slippage = abs(market_data['market_impact'])
        
        # Total slippage
        total_slippage = base_slippage + size_slippage + volatility_slippage + market_impact_slippage
        
        # Ensure reasonable bounds
        total_slippage = max(min(total_slippage, 0.05), 0.0001)  # 0.01% to 5%
        
        return total_slippage
    
    def _calculate_commission(self, order: PaperTradingOrder, fill_price: float) -> float:
        """Calculate commission based on configuration"""
        trade_value = order.quantity * fill_price
        commission = trade_value * self.config.commission_rate
        
        # Minimum commission
        min_commission = 1.0  # $1 minimum
        commission = max(commission, min_commission)
        
        return commission


class PaperTradingAgent:
    """
    Comprehensive Paper Trading Agent
    
    This agent provides realistic paper trading simulation with:
    - Market data simulation
    - Order execution simulation
    - Portfolio management
    - Performance tracking
    - Risk management
    - Seamless mode switching
    """
    
    def __init__(self, config: PaperTradingConfig = None):
        self.config = config or PaperTradingConfig()
        self.trading_mode = TradingMode.PAPER
        self.status = PaperTradingStatus.STOPPED
        
        # Initialize components
        self.market_simulator = MarketSimulator(self.config)
        self.execution_simulator = OrderExecutionSimulator(self.config, self.market_simulator)
        
        # Initialize trading components (use mock versions for paper trading)
        self.broker_integration = AlpacaBrokerIntegration(paper_trading=True)
        
        # Use mock market data for paper trading to avoid API key requirements
        self.market_data_ingestor = None  # Will use market simulator instead
        self.execution_engine = None  # Not needed for paper trading
        
        # Initialize agents with required parameters
        self.portfolio_allocator = PortfolioAllocatorAgent()
        
        # Mock database config for paper trading
        mock_db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'paper_trading',
            'user': 'paper_user',
            'password': 'paper_pass'
        }
        self.risk_manager = RiskManagerAgent(db_config=mock_db_config)
        
        # Performance monitoring with default parameters
        self.performance_monitor = PerformanceMonitoringAgent(update_interval=10)
        
        # Trade logging with mock database
        self.trade_logger = TradeLoggingAuditAgent(
            db_connection_string="mock://paper_trading",
            backup_directory="paper_trading_backups"
        )
        
        # Paper trading state
        self.portfolio = self._initialize_portfolio()
        self.orders: Dict[str, PaperTradingOrder] = {}
        self.positions: Dict[str, PaperTradingPosition] = {}
        self.trades: List[Dict[str, Any]] = []
        self.performance_history: List[PaperTradingPerformance] = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.daily_stats = {}
        self.risk_metrics = {}
        
        # Configuration
        self.symbols_universe = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'IWM']
        self.active_strategies = ['momentum', 'mean_reversion', 'sentiment']
        
        logger.info("Paper Trading Agent initialized")
    
    def _initialize_portfolio(self) -> PaperTradingPortfolio:
        """Initialize paper trading portfolio"""
        return PaperTradingPortfolio(
            cash=self.config.initial_capital,
            total_value=self.config.initial_capital,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            daily_pnl=0.0,
            positions={},
            orders=[],
            trades=[],
            last_update=datetime.now(),
            metadata={'initial_capital': self.config.initial_capital}
        )
    
    async def start_paper_trading(self):
        """Start paper trading operations"""
        if self.status == PaperTradingStatus.ACTIVE:
            logger.warning("Paper trading already active")
            return
        
        try:
            self.status = PaperTradingStatus.ACTIVE
            self.start_time = datetime.now()
            
            # Start monitoring (handle errors gracefully)
            try:
                await self.performance_monitor.start_monitoring()
            except Exception as e:
                logger.warning(f"Error starting performance monitor: {e}")
            
            try:
                await self.trade_logger.start()
            except Exception as e:
                logger.warning(f"Error starting trade logger: {e}")
            
            # Log startup (handle errors gracefully)
            try:
                self.trade_logger.log_audit_event(
                    action_type=ActionType.SYSTEM_STARTUP,
                    entity_type=EntityType.SYSTEM,
                    entity_id="PAPER_TRADING",
                    description="Paper trading system started",
                    details={'initial_capital': self.config.initial_capital},
                    log_level=LogLevel.INFO,
                    agent_id="paper_trading_agent"
                )
            except Exception as e:
                logger.warning(f"Could not log startup event: {e}")
            
            logger.info("Paper trading started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start paper trading: {e}")
            self.status = PaperTradingStatus.ERROR
            # Don't raise the error to prevent system crashes
    
    async def stop_paper_trading(self):
        """Stop paper trading operations"""
        if self.status == PaperTradingStatus.STOPPED:
            logger.warning("Paper trading already stopped")
            return
        
        try:
            self.status = PaperTradingStatus.STOPPED
            
            # Stop monitoring
            try:
                await self.performance_monitor.stop_monitoring()
            except Exception as e:
                logger.warning(f"Error stopping performance monitor: {e}")
            
            try:
                self.trade_logger.stop()
            except Exception as e:
                logger.warning(f"Error stopping trade logger: {e}")
            
            # Log shutdown (handle errors gracefully)
            try:
                self.trade_logger.log_audit_event(
                    action_type=ActionType.SYSTEM_SHUTDOWN,
                    entity_type=EntityType.SYSTEM,
                    entity_id="PAPER_TRADING",
                    description="Paper trading system stopped",
                    details={'uptime_seconds': (datetime.now() - self.start_time).total_seconds()},
                    log_level=LogLevel.INFO
                )
            except Exception as e:
                logger.warning(f"Could not log shutdown event: {e}")
            
            logger.info("Paper trading stopped successfully")
            
        except Exception as e:
            logger.error(f"Failed to stop paper trading: {e}")
            # Don't raise the error to prevent system crashes
    
    async def pause_paper_trading(self):
        """Pause paper trading operations"""
        if self.status != PaperTradingStatus.ACTIVE:
            logger.warning("Paper trading not active, cannot pause")
            return
        
        self.status = PaperTradingStatus.PAUSED
        logger.info("Paper trading paused")
    
    async def resume_paper_trading(self):
        """Resume paper trading operations"""
        if self.status != PaperTradingStatus.PAUSED:
            logger.warning("Paper trading not paused, cannot resume")
            return
        
        self.status = PaperTradingStatus.ACTIVE
        logger.info("Paper trading resumed")
    
    async def submit_paper_order(self, order_request: OrderRequest, 
                                strategy: str = "paper_trading", 
                                agent_id: str = "paper_trading_agent") -> str:
        """Submit a paper trading order"""
        if self.status != PaperTradingStatus.ACTIVE:
            raise ValueError("Paper trading not active")
        
        try:
            # Generate order ID
            order_id = f"PAPER_{uuid.uuid4().hex[:8].upper()}"
            
            # Create paper trading order
            paper_order = PaperTradingOrder(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.type,
                quantity=float(order_request.qty),
                price=float(order_request.limit_price) if order_request.limit_price else None,
                stop_price=float(order_request.stop_price) if order_request.stop_price else None,
                status=OrderStatus.NEW,
                submitted_at=datetime.now(),
                filled_at=None,
                filled_price=None,
                filled_quantity=0.0,
                commission=0.0,
                slippage=0.0,
                strategy=strategy,
                agent_id=agent_id,
                metadata=asdict(order_request)
            )
            
            # Store order
            self.orders[order_id] = paper_order
            self.portfolio.orders.append(paper_order)
            
            # Log order submission
            self.trade_logger.log_audit_event(
                action_type=ActionType.ORDER_SUBMITTED,
                entity_type=EntityType.ORDER,
                entity_id=order_id,
                description=f"Paper order submitted: {order_request.side.value} {order_request.qty} {order_request.symbol}",
                details={'order_type': order_request.type.value, 'strategy': strategy},
                log_level=LogLevel.INFO,
                agent_id=agent_id
            )
            
            # Execute order asynchronously
            asyncio.create_task(self._execute_paper_order(paper_order))
            
            logger.info(f"Paper order submitted: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to submit paper order: {e}")
            raise
    
    async def _execute_paper_order(self, order: PaperTradingOrder):
        """Execute a paper trading order"""
        try:
            # Get current market data
            market_data = await self._get_market_data(order.symbol)
            
            # Simulate order execution
            executed_order = await self.execution_simulator.simulate_order_execution(order, market_data)
            
            # Update order
            self.orders[order.order_id] = executed_order
            
            # Process trade
            await self._process_paper_trade(executed_order)
            
            # Update portfolio
            await self._update_portfolio()
            
            # Log trade
            if self.config.trade_logging:
                self._log_paper_trade(executed_order)
            
        except Exception as e:
            logger.error(f"Failed to execute paper order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data for symbol (simulated or real)"""
        try:
            # For paper trading, always use simulated market data
            # This avoids API key requirements and provides consistent simulation
            base_price = 100.0  # Default price
            volume = 1000000     # Default volume
            
            # Use market simulator for realistic data
            return self.market_simulator.simulate_market_data(symbol, base_price, volume, datetime.now())
            
        except Exception as e:
            logger.warning(f"Failed to get market data for {symbol}: {e}")
            
            # Fallback to basic simulated data
            base_price = 100.0  # Default price
            volume = 1000000     # Default volume
            
            return self.market_simulator.simulate_market_data(symbol, base_price, volume, datetime.now())
    
    async def _process_paper_trade(self, order: PaperTradingOrder):
        """Process a completed paper trading order"""
        try:
            # Create trade record
            trade = {
                'trade_id': f"PAPER_TRADE_{uuid.uuid4().hex[:8].upper()}",
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.filled_quantity,
                'price': order.filled_price,
                'timestamp': order.filled_at,
                'strategy': order.strategy,
                'agent_id': order.agent_id,
                'commission': order.commission,
                'slippage': order.slippage,
                'metadata': order.metadata
            }
            
            # Add to trades list
            self.trades.append(trade)
            self.portfolio.trades.append(trade)
            
            # Update positions
            await self._update_positions(order)
            
            # Update performance metrics
            await self._update_performance_metrics()
            
            logger.info(f"Paper trade processed: {trade['trade_id']}")
            
        except Exception as e:
            logger.error(f"Failed to process paper trade: {e}")
    
    async def _update_positions(self, order: PaperTradingOrder):
        """Update positions based on executed order"""
        try:
            symbol = order.symbol
            quantity = order.filled_quantity
            price = order.filled_price
            side = order.side
            
            if symbol not in self.positions:
                self.positions[symbol] = PaperTradingPosition(
                    symbol=symbol,
                    quantity=0.0,
                    entry_price=0.0,
                    current_price=price,
                    market_value=0.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    entry_time=datetime.now(),
                    last_update=datetime.now(),
                    strategy=order.strategy,
                    agent_id=order.agent_id
                )
            
            position = self.positions[symbol]
            
            if side == OrderSide.BUY:
                # Buying
                if position.quantity == 0:
                    # New position
                    position.quantity = quantity
                    position.entry_price = price
                    position.entry_time = datetime.now()
                else:
                    # Add to existing position
                    total_cost = (position.quantity * position.entry_price) + (quantity * price)
                    total_quantity = position.quantity + quantity
                    position.entry_price = total_cost / total_quantity
                    position.quantity = total_quantity
                
                # Update cash
                self.portfolio.cash -= (quantity * price + order.commission)
                
            else:
                # Selling
                if position.quantity < quantity:
                    logger.warning(f"Insufficient position for sell order: {position.quantity} < {quantity}")
                    return
                
                # Calculate realized P&L
                realized_pnl = (price - position.entry_price) * quantity - order.commission
                position.realized_pnl += realized_pnl
                
                # Update position
                position.quantity -= quantity
                if position.quantity == 0:
                    # Position closed
                    del self.positions[symbol]
                else:
                    # Partial position remaining
                    position.last_update = datetime.now()
                
                # Update cash
                self.portfolio.cash += (quantity * price - order.commission)
            
            # Update portfolio
            self.portfolio.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    async def _update_portfolio(self):
        """Update portfolio state and calculations"""
        try:
            # Calculate unrealized P&L and market values
            total_market_value = self.portfolio.cash
            total_unrealized_pnl = 0.0
            
            for symbol, position in self.positions.items():
                # Get current market price
                market_data = await self._get_market_data(symbol)
                current_price = market_data['close']
                
                # Update position
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                position.last_update = datetime.now()
                
                # Accumulate totals
                total_market_value += position.market_value
                total_unrealized_pnl += position.unrealized_pnl
            
            # Update portfolio
            self.portfolio.total_value = total_market_value
            self.portfolio.unrealized_pnl = total_unrealized_pnl
            self.portfolio.total_pnl = self.portfolio.realized_pnl + total_unrealized_pnl
            
            # Calculate daily P&L
            today = datetime.now().date()
            if today not in self.daily_stats:
                self.daily_stats[today] = {'start_value': total_market_value, 'pnl': 0.0}
            
            daily_start_value = self.daily_stats[today]['start_value']
            self.portfolio.daily_pnl = total_market_value - daily_start_value
            self.daily_stats[today]['pnl'] = self.portfolio.daily_pnl
            
        except Exception as e:
            logger.error(f"Failed to update portfolio: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if not self.config.performance_tracking:
                return
            
            # Calculate basic metrics
            total_return = (self.portfolio.total_value / self.config.initial_capital) - 1
            volatility = self._calculate_volatility()
            sharpe_ratio = self._calculate_sharpe_ratio(total_return, volatility)
            max_drawdown = self._calculate_max_drawdown()
            win_rate = self._calculate_win_rate()
            
            # Create performance snapshot
            performance = PaperTradingPerformance(
                total_return=total_return,
                annualized_return=total_return * (252 / max(1, (datetime.now() - self.start_time).days)),
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=0.0,  # TODO: Implement Sortino ratio
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=0.0,  # TODO: Implement profit factor
                total_trades=len(self.trades),
                winning_trades=len([t for t in self.trades if t.get('pnl', 0) > 0]),
                losing_trades=len([t for t in self.trades if t.get('pnl', 0) < 0]),
                avg_win=0.0,  # TODO: Implement average win
                avg_loss=0.0,  # TODO: Implement average loss
                largest_win=0.0,  # TODO: Implement largest win
                largest_loss=0.0,  # TODO: Implement largest loss
                consecutive_wins=0,  # TODO: Implement consecutive wins
                consecutive_losses=0,  # TODO: Implement consecutive losses
                daily_returns=[],  # TODO: Implement daily returns
                equity_curve=[],  # TODO: Implement equity curve
                drawdown_curve=[],  # TODO: Implement drawdown curve
                timestamp=datetime.now()
            )
            
            # Store performance
            self.performance_history.append(performance)
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.daily_stats) < 2:
            return 0.0
        
        returns = []
        dates = sorted(self.daily_stats.keys())
        
        for i in range(1, len(dates)):
            prev_value = self.daily_stats[dates[i-1]]['start_value']
            curr_value = self.daily_stats[dates[i]]['start_value']
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
        
        if not returns:
            return 0.0
        
        return np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    def _calculate_sharpe_ratio(self, total_return: float, volatility: float) -> float:
        """Calculate Sharpe ratio"""
        if volatility == 0:
            return 0.0
        
        risk_free_rate = 0.02  # 2% risk-free rate
        excess_return = total_return - risk_free_rate
        
        return excess_return / volatility if volatility > 0 else 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.daily_stats) < 2:
            return 0.0
        
        peak_value = self.config.initial_capital
        max_drawdown = 0.0
        
        for date, stats in sorted(self.daily_stats.items()):
            current_value = stats['start_value'] + stats['pnl']
            peak_value = max(peak_value, current_value)
            drawdown = (peak_value - current_value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.trades:
            return 0.0
        
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        return winning_trades / len(self.trades)
    
    def _log_paper_trade(self, order: PaperTradingOrder):
        """Log paper trading trade"""
        try:
            trade_data = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.filled_quantity,
                'price': order.filled_price,
                'timestamp': order.filled_at,
                'strategy': order.strategy,
                'agent_id': order.agent_id,
                'signal_strength': 0.8,  # Default for paper trading
                'market_conditions': {'mode': 'paper_trading'},
                'execution_quality': {'slippage': order.slippage, 'commission': order.commission},
                'risk_metrics': {'position_size': 0.1},
                'compliance_flags': [],
                'metadata': {'paper_trading': True, 'order_type': order.order_type.value}
            }
            
            self.trade_logger.log_trade(trade_data)
            
        except Exception as e:
            logger.error(f"Failed to log paper trade: {e}")
            # Don't raise the error to prevent system crashes
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            'cash': self.portfolio.cash,
            'total_value': self.portfolio.total_value,
            'unrealized_pnl': self.portfolio.unrealized_pnl,
            'realized_pnl': self.portfolio.realized_pnl,
            'total_pnl': self.portfolio.total_pnl,
            'daily_pnl': self.portfolio.daily_pnl,
            'total_return': (self.portfolio.total_value / self.config.initial_capital) - 1,
            'position_count': len(self.positions),
            'order_count': len(self.orders),
            'trade_count': len(self.trades),
            'status': self.status.value,
            'trading_mode': self.trading_mode.value,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }
    
    def get_positions_summary(self) -> List[Dict[str, Any]]:
        """Get positions summary"""
        positions = []
        for symbol, position in self.positions.items():
            positions.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'strategy': position.strategy,
                'agent_id': position.agent_id
            })
        return positions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {}
        
        latest = self.performance_history[-1]
        return {
            'total_return': latest.total_return,
            'annualized_return': latest.annualized_return,
            'volatility': latest.volatility,
            'sharpe_ratio': latest.sharpe_ratio,
            'max_drawdown': latest.max_drawdown,
            'win_rate': latest.win_rate,
            'total_trades': latest.total_trades,
            'winning_trades': latest.winning_trades,
            'losing_trades': latest.losing_trades
        }
    
    async def switch_to_live_trading(self):
        """Switch from paper trading to live trading"""
        try:
            # Stop paper trading
            await self.stop_paper_trading()
            
            # Switch mode
            self.trading_mode = TradingMode.LIVE
            
            # Initialize live trading components
            self.broker_integration = AlpacaBrokerIntegration(paper_trading=False)
            
            # Log mode switch
            self.trade_logger.log_audit_event(
                action_type=ActionType.CONFIGURATION_CHANGED,
                entity_type=EntityType.SYSTEM,
                entity_id="TRADING_MODE",
                description="Switched from paper trading to live trading",
                details={'previous_mode': 'paper', 'new_mode': 'live'},
                log_level=LogLevel.INFO
            )
            
            logger.info("Switched to live trading mode")
            
        except Exception as e:
            logger.error(f"Failed to switch to live trading: {e}")
            raise
    
    async def switch_to_paper_trading(self):
        """Switch from live trading to paper trading"""
        try:
            # Switch mode
            self.trading_mode = TradingMode.PAPER
            
            # Initialize paper trading components
            self.broker_integration = AlpacaBrokerIntegration(paper_trading=True)
            
            # Start paper trading
            await self.start_paper_trading()
            
            # Log mode switch
            self.trade_logger.log_audit_event(
                action_type=ActionType.CONFIGURATION_CHANGED,
                entity_type=EntityType.SYSTEM,
                entity_id="TRADING_MODE",
                description="Switched from live trading to paper trading",
                details={'previous_mode': 'live', 'new_mode': 'paper'},
                log_level=LogLevel.INFO
            )
            
            logger.info("Switched to paper trading mode")
            
        except Exception as e:
            logger.error(f"Failed to switch to paper trading: {e}")
            raise


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize paper trading agent
        config = PaperTradingConfig(
            initial_capital=100000.0,
            max_position_size=0.1,
            max_daily_trades=50,
            max_daily_loss=0.03,
            commission_rate=0.001,
            slippage_model="realistic",
            market_impact_model="square_root"
        )
        
        agent = PaperTradingAgent(config)
        
        # Start paper trading
        await agent.start_paper_trading()
        
        # Submit some test orders
        order_request = OrderRequest(
            symbol="AAPL",
            qty=100,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        order_id = await agent.submit_paper_order(order_request, "test_strategy", "test_agent")
        print(f"Submitted paper order: {order_id}")
        
        # Wait for execution
        await asyncio.sleep(2)
        
        # Get portfolio summary
        summary = agent.get_portfolio_summary()
        print(f"Portfolio summary: {summary}")
        
        # Get positions
        positions = agent.get_positions_summary()
        print(f"Positions: {positions}")
        
        # Stop paper trading
        await agent.stop_paper_trading()
        
        print("Paper trading demo completed!")
    
    # Run demo
    asyncio.run(main()) 