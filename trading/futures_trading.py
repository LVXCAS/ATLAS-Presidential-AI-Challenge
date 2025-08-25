"""
Hive Trade Advanced Futures Trading with Leverage Management
Comprehensive futures trading system with sophisticated risk controls
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class FuturesPosition:
    """Futures position data structure"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    leverage: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class FuturesOrder:
    """Futures order data structure"""
    id: str
    symbol: str
    side: str
    size: float
    price: Optional[float]
    order_type: str
    leverage: float
    status: OrderStatus
    timestamp: datetime
    filled_price: Optional[float] = None
    fees: float = 0.0

class LeverageManager:
    """Advanced leverage management system"""
    
    def __init__(self, max_portfolio_leverage: float = 10.0):
        self.max_portfolio_leverage = max_portfolio_leverage
        self.position_limits = {
            'crypto': {'max_leverage': 20.0, 'max_position_size': 0.3},  # 30% of portfolio
            'commodities': {'max_leverage': 15.0, 'max_position_size': 0.25},
            'indices': {'max_leverage': 10.0, 'max_position_size': 0.4},
            'forex': {'max_leverage': 50.0, 'max_position_size': 0.2}
        }
        
    def calculate_optimal_leverage(self, symbol: str, volatility: float, 
                                 confidence: float, account_balance: float) -> float:
        """Calculate optimal leverage based on Kelly Criterion and risk factors"""
        
        # Get asset class limits
        asset_class = self._get_asset_class(symbol)
        max_leverage = self.position_limits[asset_class]['max_leverage']
        
        # Kelly Criterion: f = (bp - q) / b
        # Where b = odds, p = probability of win, q = probability of loss
        win_rate = confidence
        loss_rate = 1 - confidence
        avg_win = 0.02  # Assume 2% average win
        avg_loss = 0.015  # Assume 1.5% average loss
        
        kelly_fraction = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Adjust for volatility
        volatility_adjustment = 1 / (1 + volatility * 2)
        
        # Calculate optimal leverage
        optimal_leverage = kelly_fraction * volatility_adjustment * max_leverage
        optimal_leverage = min(optimal_leverage, max_leverage)
        optimal_leverage = max(optimal_leverage, 1.0)  # Minimum 1x
        
        return optimal_leverage
    
    def calculate_position_size(self, symbol: str, leverage: float, 
                              account_balance: float, risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on risk management rules"""
        
        # Maximum risk per trade (default 2% of account)
        max_risk_amount = account_balance * risk_per_trade
        
        # Get asset class limits
        asset_class = self._get_asset_class(symbol)
        max_position_pct = self.position_limits[asset_class]['max_position_size']
        
        # Calculate position size based on leverage
        position_value = max_risk_amount * leverage
        max_position_value = account_balance * max_position_pct
        
        # Take minimum to respect limits
        final_position_value = min(position_value, max_position_value)
        
        return final_position_value
    
    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol"""
        if any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']):
            return 'crypto'
        elif any(commodity in symbol.upper() for commodity in ['CL', 'GC', 'SI', 'NG', 'ZC']):
            return 'commodities'
        elif any(forex in symbol.upper() for forex in ['EUR', 'GBP', 'JPY', 'USD', 'CHF']):
            return 'forex'
        else:
            return 'indices'
    
    def validate_leverage(self, symbol: str, requested_leverage: float) -> Tuple[bool, str, float]:
        """Validate if requested leverage is within limits"""
        
        asset_class = self._get_asset_class(symbol)
        max_leverage = self.position_limits[asset_class]['max_leverage']
        
        if requested_leverage > max_leverage:
            return False, f"Leverage {requested_leverage} exceeds max {max_leverage} for {asset_class}", max_leverage
        elif requested_leverage < 1.0:
            return False, "Leverage cannot be less than 1.0", 1.0
        else:
            return True, "Leverage valid", requested_leverage

class RiskManager:
    """Advanced risk management for futures trading"""
    
    def __init__(self):
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.var_confidence = 0.95  # 95% VaR confidence
        self.liquidation_buffer = 0.2  # 20% buffer before liquidation
        
    def calculate_margin_requirements(self, position: FuturesPosition) -> Dict[str, float]:
        """Calculate margin requirements for position"""
        
        initial_margin = (position.size * position.entry_price) / position.leverage
        maintenance_margin = initial_margin * 0.5  # Typically 50% of initial
        
        # Calculate current margin based on unrealized PnL
        current_margin = initial_margin + position.unrealized_pnl
        
        # Margin call level (when additional margin needed)
        margin_call_level = maintenance_margin * 1.2
        
        # Liquidation level
        liquidation_level = maintenance_margin
        
        return {
            'initial_margin': initial_margin,
            'maintenance_margin': maintenance_margin,
            'current_margin': current_margin,
            'margin_call_level': margin_call_level,
            'liquidation_level': liquidation_level,
            'margin_ratio': current_margin / initial_margin if initial_margin > 0 else 0
        }
    
    def calculate_liquidation_price(self, position: FuturesPosition) -> float:
        """Calculate liquidation price for position"""
        
        margin_info = self.calculate_margin_requirements(position)
        maintenance_margin = margin_info['maintenance_margin']
        
        if position.side == PositionSide.LONG:
            # For long: liquidation when price falls to maintenance margin level
            liquidation_price = position.entry_price - (maintenance_margin / position.size)
        else:
            # For short: liquidation when price rises to maintenance margin level
            liquidation_price = position.entry_price + (maintenance_margin / position.size)
        
        return max(liquidation_price, 0)  # Price can't be negative
    
    def calculate_value_at_risk(self, positions: List[FuturesPosition], 
                               historical_returns: pd.DataFrame) -> float:
        """Calculate portfolio Value at Risk (VaR)"""
        
        if historical_returns.empty or not positions:
            return 0.0
        
        # Calculate portfolio returns
        portfolio_values = []
        
        for _, returns in historical_returns.iterrows():
            portfolio_value = 0
            
            for position in positions:
                if position.symbol in returns:
                    # Calculate position value change
                    price_change = returns[position.symbol]
                    position_change = position.size * price_change * position.leverage
                    
                    if position.side == PositionSide.SHORT:
                        position_change *= -1
                    
                    portfolio_value += position_change
            
            portfolio_values.append(portfolio_value)
        
        # Calculate VaR at specified confidence level
        portfolio_returns = pd.Series(portfolio_values)
        var = portfolio_returns.quantile(1 - self.var_confidence)
        
        return abs(var)
    
    def should_close_position(self, position: FuturesPosition, 
                            account_balance: float) -> Tuple[bool, str]:
        """Determine if position should be closed based on risk factors"""
        
        margin_info = self.calculate_margin_requirements(position)
        liquidation_price = self.calculate_liquidation_price(position)
        
        # Check margin ratio
        if margin_info['margin_ratio'] < 1 + self.liquidation_buffer:
            return True, f"Margin ratio {margin_info['margin_ratio']:.2f} below safe level"
        
        # Check unrealized loss vs account
        if abs(position.unrealized_pnl) > account_balance * self.max_drawdown_limit:
            return True, f"Unrealized loss {abs(position.unrealized_pnl):.2f} exceeds drawdown limit"
        
        # Check distance to liquidation
        price_to_liquidation = abs(position.current_price - liquidation_price) / position.current_price
        if price_to_liquidation < 0.1:  # Less than 10% from liquidation
            return True, f"Price {price_to_liquidation:.1%} from liquidation level"
        
        return False, "Position within risk limits"

class FuturesExchange:
    """Mock futures exchange for demonstration"""
    
    def __init__(self, name: str):
        self.name = name
        self.positions: Dict[str, FuturesPosition] = {}
        self.orders: Dict[str, FuturesOrder] = {}
        self.account_balance = 100000.0  # $100k starting balance
        self.leverage_manager = LeverageManager()
        self.risk_manager = RiskManager()
        
    def get_market_price(self, symbol: str) -> float:
        """Get current market price (mock)"""
        # Mock prices for different asset classes
        base_prices = {
            'BTCUSD': 45000,
            'ETHUSD': 3000,
            'EURUSD': 1.0850,
            'GBPUSD': 1.2600,
            'GOLD': 2000,
            'CRUDE': 80,
            'SP500': 4500,
            'NASDAQ': 15000
        }
        
        base_price = base_prices.get(symbol, 100)
        # Add random variation
        variation = np.random.normal(0, 0.02)  # 2% volatility
        return base_price * (1 + variation)
    
    def place_futures_order(self, symbol: str, side: str, size: float, 
                           leverage: float, order_type: str = 'market',
                           price: Optional[float] = None) -> FuturesOrder:
        """Place futures order"""
        
        # Validate leverage
        is_valid, message, adjusted_leverage = self.leverage_manager.validate_leverage(symbol, leverage)
        if not is_valid:
            leverage = adjusted_leverage
            print(f"Leverage adjusted: {message}")
        
        # Generate order ID
        order_id = f"{self.name}_{int(datetime.now().timestamp() * 1000)}"
        
        # Get current price
        current_price = self.get_market_price(symbol)
        fill_price = price if order_type == 'limit' and price else current_price
        
        # Calculate position size and margin
        position_value = self.leverage_manager.calculate_position_size(
            symbol, leverage, self.account_balance
        )
        position_size = position_value / fill_price
        
        # Create order
        order = FuturesOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            size=position_size,
            price=price,
            order_type=order_type,
            leverage=leverage,
            status=OrderStatus.FILLED,  # Mock immediate fill
            timestamp=datetime.now(),
            filled_price=fill_price,
            fees=position_value * 0.001  # 0.1% fees
        )
        
        # Update account balance for fees
        self.account_balance -= order.fees
        
        # Create or update position
        self._update_position(order)
        
        self.orders[order_id] = order
        
        print(f"Order placed: {side.upper()} {position_size:.4f} {symbol} @ {fill_price:.2f} "
              f"(Leverage: {leverage}x, Fees: ${order.fees:.2f})")
        
        return order
    
    def _update_position(self, order: FuturesOrder):
        """Update position based on order"""
        
        current_price = self.get_market_price(order.symbol)
        
        if order.symbol in self.positions:
            # Update existing position
            position = self.positions[order.symbol]
            
            # Calculate new average price and size
            if (position.side.value == order.side) or (position.size == 0):
                # Same side or new position
                total_value = position.size * position.entry_price + order.size * order.filled_price
                total_size = position.size + order.size
                new_avg_price = total_value / total_size if total_size > 0 else order.filled_price
                
                position.size = total_size
                position.entry_price = new_avg_price
                position.side = PositionSide.LONG if order.side == 'buy' else PositionSide.SHORT
            else:
                # Opposite side - reducing or reversing position
                if order.size >= position.size:
                    # Reversing position
                    remaining_size = order.size - position.size
                    position.realized_pnl += self._calculate_pnl(position, current_price)
                    
                    position.size = remaining_size
                    position.entry_price = order.filled_price
                    position.side = PositionSide.LONG if order.side == 'buy' else PositionSide.SHORT
                else:
                    # Reducing position
                    close_ratio = order.size / position.size
                    position.realized_pnl += self._calculate_pnl(position, current_price) * close_ratio
                    position.size -= order.size
        else:
            # New position
            position = FuturesPosition(
                symbol=order.symbol,
                side=PositionSide.LONG if order.side == 'buy' else PositionSide.SHORT,
                size=order.size,
                entry_price=order.filled_price,
                current_price=current_price,
                leverage=order.leverage,
                margin_used=(order.size * order.filled_price) / order.leverage,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                timestamp=datetime.now()
            )
            
            self.positions[order.symbol] = position
        
        # Update current price and unrealized PnL
        position = self.positions[order.symbol]
        position.current_price = current_price
        position.unrealized_pnl = self._calculate_pnl(position, current_price)
    
    def _calculate_pnl(self, position: FuturesPosition, current_price: float) -> float:
        """Calculate profit/loss for position"""
        
        price_diff = current_price - position.entry_price
        
        if position.side == PositionSide.SHORT:
            price_diff *= -1
        
        return position.size * price_diff * position.leverage
    
    def update_positions(self):
        """Update all positions with current market prices"""
        
        for symbol, position in self.positions.items():
            if position.size > 0:  # Only update active positions
                current_price = self.get_market_price(symbol)
                position.current_price = current_price
                position.unrealized_pnl = self._calculate_pnl(position, current_price)
                
                # Check if position should be closed due to risk
                should_close, reason = self.risk_manager.should_close_position(
                    position, self.account_balance
                )
                
                if should_close:
                    print(f"Risk Alert - {symbol}: {reason}")
                    self.close_position(symbol, reason="Risk management")
    
    def close_position(self, symbol: str, reason: str = "Manual close") -> bool:
        """Close position"""
        
        if symbol not in self.positions:
            print(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        if position.size == 0:
            print(f"Position already closed for {symbol}")
            return True
        
        # Place closing order
        close_side = 'sell' if position.side == PositionSide.LONG else 'buy'
        close_order = self.place_futures_order(
            symbol, close_side, position.size, position.leverage
        )
        
        # Realize P&L
        self.account_balance += position.unrealized_pnl
        position.realized_pnl += position.unrealized_pnl
        position.unrealized_pnl = 0.0
        position.size = 0.0
        
        print(f"Position closed - {symbol}: {reason}, P&L: ${position.realized_pnl:.2f}")
        
        return True
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_margin_used = sum(pos.margin_used for pos in self.positions.values() if pos.size > 0)
        
        # Calculate total portfolio leverage
        total_position_value = sum(
            pos.size * pos.current_price for pos in self.positions.values() if pos.size > 0
        )
        portfolio_leverage = total_position_value / self.account_balance if self.account_balance > 0 else 0
        
        # Free margin
        free_margin = self.account_balance - total_margin_used
        
        return {
            'account_balance': self.account_balance,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_margin_used': total_margin_used,
            'free_margin': free_margin,
            'portfolio_leverage': portfolio_leverage,
            'num_positions': len([pos for pos in self.positions.values() if pos.size > 0]),
            'margin_usage_pct': (total_margin_used / self.account_balance) * 100 if self.account_balance > 0 else 0
        }

class AdvancedFuturesTrader:
    """Advanced futures trading system"""
    
    def __init__(self):
        self.exchanges = {
            'futures_exchange': FuturesExchange('FuturesExchange')
        }
        self.strategy_params = {
            'crypto': {'volatility_threshold': 0.05, 'max_leverage': 10},
            'forex': {'volatility_threshold': 0.02, 'max_leverage': 20},
            'commodities': {'volatility_threshold': 0.03, 'max_leverage': 15},
            'indices': {'volatility_threshold': 0.025, 'max_leverage': 8}
        }
        
    def analyze_futures_opportunities(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze futures trading opportunities"""
        
        print("\nFUTURES TRADING ANALYSIS")
        print("="*25)
        
        opportunities = {}
        
        for symbol in symbols:
            print(f"\nAnalyzing {symbol}...")
            
            exchange = self.exchanges['futures_exchange']
            current_price = exchange.get_market_price(symbol)
            
            # Mock historical data for volatility calculation
            historical_prices = []
            for i in range(30):  # 30 days of data
                base_price = current_price * (0.95 + 0.1 * np.random.random())
                historical_prices.append(base_price)
            
            returns = pd.Series(historical_prices).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Get asset class
            asset_class = exchange.leverage_manager._get_asset_class(symbol)
            
            # Generate trading signal
            signal = self._generate_futures_signal(symbol, current_price, volatility, returns)
            
            # Calculate optimal leverage and position size
            confidence = 0.6  # Mock confidence
            optimal_leverage = exchange.leverage_manager.calculate_optimal_leverage(
                symbol, volatility, confidence, exchange.account_balance
            )
            
            position_size = exchange.leverage_manager.calculate_position_size(
                symbol, optimal_leverage, exchange.account_balance
            )
            
            opportunities[symbol] = {
                'current_price': current_price,
                'volatility': volatility,
                'asset_class': asset_class,
                'signal': signal,
                'optimal_leverage': optimal_leverage,
                'position_size': position_size,
                'confidence': confidence,
                'risk_reward_ratio': self._calculate_risk_reward(returns)
            }
            
            print(f"  Price: ${current_price:.2f}")
            print(f"  Volatility: {volatility:.1%}")
            print(f"  Signal: {signal}")
            print(f"  Optimal Leverage: {optimal_leverage:.1f}x")
        
        return opportunities
    
    def _generate_futures_signal(self, symbol: str, price: float, 
                                volatility: float, returns: pd.Series) -> str:
        """Generate trading signal for futures"""
        
        # Simple momentum + mean reversion strategy
        recent_momentum = returns.tail(5).mean()
        long_momentum = returns.tail(20).mean()
        
        # Volatility condition
        vol_threshold = self.strategy_params.get(
            symbol.split('USD')[0].lower(), self.strategy_params['indices']
        )['volatility_threshold']
        
        if recent_momentum > 0.01 and long_momentum > 0.005 and volatility < vol_threshold:
            return 'LONG'
        elif recent_momentum < -0.01 and long_momentum < -0.005 and volatility < vol_threshold:
            return 'SHORT'
        elif volatility > vol_threshold * 2:
            return 'HIGH_VOL_AVOID'
        else:
            return 'HOLD'
    
    def _calculate_risk_reward(self, returns: pd.Series) -> float:
        """Calculate risk-reward ratio"""
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 1.0
        
        avg_win = positive_returns.mean()
        avg_loss = abs(negative_returns.mean())
        
        return avg_win / avg_loss if avg_loss > 0 else 1.0
    
    def execute_futures_strategy(self, opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """Execute futures trading strategy"""
        
        print("\nEXECUTING FUTURES STRATEGY")
        print("="*27)
        
        execution_results = {}
        exchange = self.exchanges['futures_exchange']
        
        for symbol, opp in opportunities.items():
            if opp['signal'] in ['LONG', 'SHORT']:
                try:
                    # Place order
                    side = 'buy' if opp['signal'] == 'LONG' else 'sell'
                    leverage = min(opp['optimal_leverage'], 10)  # Cap at 10x for safety
                    
                    order = exchange.place_futures_order(
                        symbol=symbol,
                        side=side,
                        size=opp['position_size'],
                        leverage=leverage
                    )
                    
                    execution_results[symbol] = {
                        'order': order.__dict__,
                        'signal': opp['signal'],
                        'leverage_used': leverage,
                        'expected_return': opp['risk_reward_ratio']
                    }
                    
                except Exception as e:
                    execution_results[symbol] = {'error': str(e)}
                    print(f"Error executing {symbol}: {e}")
        
        return execution_results
    
    def monitor_positions(self) -> Dict[str, Any]:
        """Monitor all positions and risk metrics"""
        
        print("\nPOSITION MONITORING")
        print("="*20)
        
        exchange = self.exchanges['futures_exchange']
        exchange.update_positions()
        
        portfolio_summary = exchange.get_portfolio_summary()
        
        print(f"Account Balance: ${portfolio_summary['account_balance']:,.2f}")
        print(f"Unrealized P&L: ${portfolio_summary['total_unrealized_pnl']:,.2f}")
        print(f"Realized P&L: ${portfolio_summary['total_realized_pnl']:,.2f}")
        print(f"Portfolio Leverage: {portfolio_summary['portfolio_leverage']:.1f}x")
        print(f"Margin Usage: {portfolio_summary['margin_usage_pct']:.1f}%")
        
        # Individual position analysis
        position_analysis = {}
        
        for symbol, position in exchange.positions.items():
            if position.size > 0:
                margin_info = exchange.risk_manager.calculate_margin_requirements(position)
                liquidation_price = exchange.risk_manager.calculate_liquidation_price(position)
                
                position_analysis[symbol] = {
                    'side': position.side.value,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'leverage': position.leverage,
                    'unrealized_pnl': position.unrealized_pnl,
                    'margin_ratio': margin_info['margin_ratio'],
                    'liquidation_price': liquidation_price,
                    'distance_to_liquidation': abs(position.current_price - liquidation_price) / position.current_price
                }
                
                print(f"\n{symbol}:")
                print(f"  Side: {position.side.value.upper()}")
                print(f"  P&L: ${position.unrealized_pnl:.2f}")
                print(f"  Margin Ratio: {margin_info['margin_ratio']:.2f}")
                print(f"  Liquidation: ${liquidation_price:.2f}")
        
        return {
            'portfolio_summary': portfolio_summary,
            'positions': position_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_futures_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Run comprehensive futures analysis"""
        
        print("HIVE TRADE ADVANCED FUTURES TRADING")
        print("="*40)
        
        # Analyze opportunities
        opportunities = self.analyze_futures_opportunities(symbols)
        
        # Execute strategy
        execution_results = self.execute_futures_strategy(opportunities)
        
        # Monitor positions
        monitoring_results = self.monitor_positions()
        
        return {
            'opportunities': opportunities,
            'executions': execution_results,
            'monitoring': monitoring_results,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Run advanced futures trading analysis"""
    
    # Major futures symbols
    symbols = ['BTCUSD', 'ETHUSD', 'EURUSD', 'GBPUSD', 'GOLD', 'CRUDE', 'SP500', 'NASDAQ']
    
    # Initialize futures trader
    trader = AdvancedFuturesTrader()
    
    # Run comprehensive analysis
    results = trader.run_futures_analysis(symbols)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"futures_analysis_{timestamp}.json"
    
    # Clean results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if isinstance(v, (np.integer, np.floating)):
                    cleaned[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, (list, dict)):
                    cleaned[k] = clean_for_json(v)
                elif hasattr(v, '__dict__'):
                    cleaned[k] = clean_for_json(v.__dict__)
                else:
                    try:
                        json.dumps(v)  # Test if serializable
                        cleaned[k] = v
                    except:
                        cleaned[k] = str(v)
            return cleaned
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        else:
            return obj
    
    clean_results = clean_for_json(results)
    
    with open(results_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    print("\nADVANCED FUTURES TRADING ANALYSIS COMPLETE!")
    print("="*45)
    
    return results

if __name__ == "__main__":
    main()