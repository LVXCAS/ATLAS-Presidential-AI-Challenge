import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import math
from pathlib import Path

from event_bus import TradingEventBus, Event, Priority


class PositionType(Enum):
    LONG = "long"
    SHORT = "short"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    position_type: PositionType
    opened_at: datetime
    last_updated: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    current_price: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.opened_at, str):
            self.opened_at = datetime.fromisoformat(self.opened_at)
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)
    
    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.avg_price
    
    def update_price(self, new_price: float):
        self.current_price = new_price
        self.last_updated = datetime.now()
        
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (new_price - self.avg_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_price - new_price) * abs(self.quantity)
    
    def add_shares(self, quantity: float, price: float) -> float:
        if (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
            return self._close_partial_position(quantity, price)
        
        total_cost = self.cost_basis + (abs(quantity) * price)
        total_quantity = abs(self.quantity) + abs(quantity)
        
        if total_quantity > 0:
            self.avg_price = total_cost / total_quantity
        
        self.quantity += quantity
        self.last_updated = datetime.now()
        return 0.0
    
    def _close_partial_position(self, quantity: float, price: float) -> float:
        closing_quantity = min(abs(quantity), abs(self.quantity))
        
        if self.position_type == PositionType.LONG:
            realized_pnl = (price - self.avg_price) * closing_quantity
        else:
            realized_pnl = (self.avg_price - price) * closing_quantity
        
        self.realized_pnl += realized_pnl
        self.quantity += quantity
        self.last_updated = datetime.now()
        
        return realized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'position_type': self.position_type.value,
            'opened_at': self.opened_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'current_price': self.current_price
        }


@dataclass
class RiskMetrics:
    total_value: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    beta: float = 0.0
    volatility: float = 0.0
    concentration_risk: float = 0.0
    leverage_ratio: float = 0.0
    margin_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class Portfolio:
    def __init__(self, 
                 initial_cash: float = 100000.0,
                 margin_ratio: float = 2.0,
                 event_bus: Optional[TradingEventBus] = None,
                 persistence_file: Optional[str] = None):
        
        self.logger = logging.getLogger(__name__)
        
        self.cash = Decimal(str(initial_cash))
        self.initial_cash = Decimal(str(initial_cash))
        self.margin_ratio = margin_ratio
        self.positions: Dict[str, Position] = {}
        
        self.event_bus = event_bus
        self.persistence_file = persistence_file or "portfolio_state.json"
        
        self._price_history: Dict[str, List[float]] = {}
        self._pnl_history: List[float] = []
        self._portfolio_values: List[float] = [float(initial_cash)]
        self._last_portfolio_value = float(initial_cash)
        
        self.risk_metrics = RiskMetrics()
        
        self._setup_event_handlers()
        
        if Path(self.persistence_file).exists():
            self.load_state()
    
    def _setup_event_handlers(self):
        if self.event_bus:
            self.event_bus.subscribe(
                TradingEventBus.MARKET_DATA,
                self._handle_market_data
            )
            self.event_bus.subscribe(
                TradingEventBus.ORDER_FILLED,
                self._handle_order_filled
            )
    
    async def _handle_market_data(self, event: Event):
        data = event.data
        symbol = data.get('symbol')
        price = data.get('price')
        
        if symbol and price and symbol in self.positions:
            await self.update_position_price(symbol, float(price))
    
    async def _handle_order_filled(self, event: Event):
        data = event.data
        symbol = data.get('symbol')
        quantity = data.get('quantity', 0)
        price = data.get('price', 0)
        side = data.get('side', 'buy').lower()
        
        if side == 'sell':
            quantity = -abs(quantity)
        
        await self.add_trade(symbol, quantity, float(price))
    
    @property
    def buying_power(self) -> float:
        return float(self.cash) * self.margin_ratio
    
    @property
    def portfolio_value(self) -> float:
        cash_value = float(self.cash)
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return cash_value + positions_value
    
    @property
    def total_pnl(self) -> float:
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized = sum(pos.realized_pnl for pos in self.positions.values())
        return unrealized + realized
    
    @property
    def margin_used(self) -> float:
        return sum(pos.market_value / self.margin_ratio for pos in self.positions.values())
    
    @property
    def available_margin(self) -> float:
        return max(0, self.buying_power - self.margin_used)
    
    async def add_trade(self, symbol: str, quantity: float, price: float) -> bool:
        try:
            trade_value = abs(quantity) * price
            required_cash = trade_value / self.margin_ratio if quantity > 0 else 0
            
            if quantity > 0 and required_cash > float(self.cash):
                self.logger.warning(f"Insufficient cash for trade: {symbol} {quantity}@{price}")
                return False
            
            if symbol not in self.positions:
                position_type = PositionType.LONG if quantity > 0 else PositionType.SHORT
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    position_type=position_type,
                    opened_at=datetime.now(),
                    last_updated=datetime.now(),
                    current_price=price
                )
                
                self.cash -= Decimal(str(required_cash))
            else:
                position = self.positions[symbol]
                realized_pnl = position.add_shares(quantity, price)
                
                if quantity > 0:
                    self.cash -= Decimal(str(required_cash))
                else:
                    self.cash += Decimal(str(trade_value))
                
                if abs(position.quantity) < 1e-6:
                    del self.positions[symbol]
            
            await self._update_risk_metrics()
            
            if self.event_bus:
                await self.event_bus.publish(
                    TradingEventBus.POSITION_OPENED if symbol not in self.positions or quantity > 0 
                    else TradingEventBus.POSITION_CLOSED,
                    {
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'portfolio_value': self.portfolio_value,
                        'cash': float(self.cash)
                    },
                    priority=Priority.HIGH
                )
            
            await self.save_state()
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding trade {symbol}: {e}")
            return False
    
    async def update_position_price(self, symbol: str, price: float):
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        old_pnl = position.unrealized_pnl
        position.update_price(price)
        
        self._add_price_to_history(symbol, price)
        
        pnl_change = position.unrealized_pnl - old_pnl
        if abs(pnl_change) > 0.01:
            await self._update_risk_metrics()
    
    def _add_price_to_history(self, symbol: str, price: float):
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        
        self._price_history[symbol].append(price)
        
        if len(self._price_history[symbol]) > 252:
            self._price_history[symbol] = self._price_history[symbol][-252:]
    
    async def _update_risk_metrics(self):
        try:
            current_value = self.portfolio_value
            self._portfolio_values.append(current_value)
            
            if len(self._portfolio_values) > 252:
                self._portfolio_values = self._portfolio_values[-252:]
            
            self.risk_metrics.total_value = current_value
            self.risk_metrics.total_pnl = self.total_pnl
            
            if len(self._portfolio_values) > 1:
                self.risk_metrics.daily_pnl = current_value - self._portfolio_values[-2]
            
            self._calculate_var()
            self._calculate_max_drawdown()
            self._calculate_volatility()
            self._calculate_concentration_risk()
            self._calculate_leverage_metrics()
            
            if self.event_bus:
                await self.event_bus.publish(
                    "risk_metrics_updated",
                    self.risk_metrics.to_dict(),
                    priority=Priority.NORMAL
                )
                
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
    
    def _calculate_var(self):
        if len(self._portfolio_values) < 30:
            return
        
        returns = []
        for i in range(1, len(self._portfolio_values)):
            if self._portfolio_values[i-1] != 0:
                ret = (self._portfolio_values[i] - self._portfolio_values[i-1]) / self._portfolio_values[i-1]
                returns.append(ret)
        
        if returns:
            returns.sort()
            n = len(returns)
            
            var_95_index = int(n * 0.05)
            var_99_index = int(n * 0.01)
            
            if var_95_index < n:
                self.risk_metrics.var_95 = returns[var_95_index] * self.portfolio_value
            if var_99_index < n:
                self.risk_metrics.var_99 = returns[var_99_index] * self.portfolio_value
    
    def _calculate_max_drawdown(self):
        if len(self._portfolio_values) < 2:
            return
        
        peak = self._portfolio_values[0]
        max_drawdown = 0.0
        
        for value in self._portfolio_values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        self.risk_metrics.max_drawdown = max_drawdown
    
    def _calculate_volatility(self):
        if len(self._portfolio_values) < 30:
            return
        
        returns = []
        for i in range(1, len(self._portfolio_values)):
            if self._portfolio_values[i-1] != 0:
                ret = (self._portfolio_values[i] - self._portfolio_values[i-1]) / self._portfolio_values[i-1]
                returns.append(ret)
        
        if returns:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            self.risk_metrics.volatility = math.sqrt(variance * 252)
            
            if self.risk_metrics.volatility > 0:
                self.risk_metrics.sharpe_ratio = (mean_return * 252) / self.risk_metrics.volatility
    
    def _calculate_concentration_risk(self):
        if not self.positions:
            self.risk_metrics.concentration_risk = 0.0
            return
        
        total_value = sum(abs(pos.market_value) for pos in self.positions.values())
        if total_value == 0:
            self.risk_metrics.concentration_risk = 0.0
            return
        
        weights = [abs(pos.market_value) / total_value for pos in self.positions.values()]
        hhi = sum(w ** 2 for w in weights)
        self.risk_metrics.concentration_risk = hhi
    
    def _calculate_leverage_metrics(self):
        total_position_value = sum(abs(pos.market_value) for pos in self.positions.values())
        
        if self.portfolio_value > 0:
            self.risk_metrics.leverage_ratio = total_position_value / self.portfolio_value
        
        if self.buying_power > 0:
            self.risk_metrics.margin_utilization = self.margin_used / self.buying_power
    
    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        return self.positions.copy()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        return {
            'cash': float(self.cash),
            'portfolio_value': self.portfolio_value,
            'total_pnl': self.total_pnl,
            'buying_power': self.buying_power,
            'margin_used': self.margin_used,
            'available_margin': self.available_margin,
            'positions_count': len(self.positions),
            'risk_metrics': self.risk_metrics.to_dict(),
            'last_updated': datetime.now().isoformat()
        }
    
    async def save_state(self):
        try:
            state = {
                'cash': float(self.cash),
                'initial_cash': float(self.initial_cash),
                'margin_ratio': self.margin_ratio,
                'positions': {k: v.to_dict() for k, v in self.positions.items()},
                'portfolio_values': self._portfolio_values[-100:],
                'risk_metrics': self.risk_metrics.to_dict(),
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.persistence_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving portfolio state: {e}")
    
    def load_state(self):
        try:
            with open(self.persistence_file, 'r') as f:
                state = json.load(f)
            
            self.cash = Decimal(str(state.get('cash', self.cash)))
            self.initial_cash = Decimal(str(state.get('initial_cash', self.initial_cash)))
            self.margin_ratio = state.get('margin_ratio', self.margin_ratio)
            
            positions_data = state.get('positions', {})
            for symbol, pos_data in positions_data.items():
                pos_data['position_type'] = PositionType(pos_data['position_type'])
                self.positions[symbol] = Position(**pos_data)
            
            self._portfolio_values = state.get('portfolio_values', [float(self.initial_cash)])
            
            risk_data = state.get('risk_metrics', {})
            self.risk_metrics = RiskMetrics(**risk_data)
            
            self.logger.info(f"Portfolio state loaded from {self.persistence_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {e}")
    
    async def reset_portfolio(self, new_cash: Optional[float] = None):
        self.positions.clear()
        self.cash = Decimal(str(new_cash or self.initial_cash))
        self._portfolio_values = [float(self.cash)]
        self._price_history.clear()
        self.risk_metrics = RiskMetrics()
        
        await self.save_state()
        
        if self.event_bus:
            await self.event_bus.publish(
                "portfolio_reset",
                {'new_cash': float(self.cash)},
                priority=Priority.HIGH
            )
    
    async def close_position(self, symbol: str, price: Optional[float] = None) -> bool:
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        close_price = price or position.current_price
        
        return await self.add_trade(symbol, -position.quantity, close_price)
    
    async def close_all_positions(self) -> int:
        closed_count = 0
        symbols_to_close = list(self.positions.keys())
        
        for symbol in symbols_to_close:
            if await self.close_position(symbol):
                closed_count += 1
        
        return closed_count