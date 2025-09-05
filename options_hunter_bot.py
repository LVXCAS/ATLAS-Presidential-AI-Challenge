#!/usr/bin/env python3
"""
Pure Options Trading Bot - Monte Carlo Optimized
Features:
- 100% options trading (no stocks)
- Focus on highest win-rate strategies: Bear Put Spreads (86.4%) & Bull Call Spreads (71.7%)
- Optimized for Monte Carlo simulation success
- Advanced risk management and position sizing
"""

import asyncio
import sys
import os
import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf

# Add current directory to path
sys.path.append('.')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
except:
    POLYGON_API_KEY = None
    ALPACA_API_KEY = None
    ALPACA_SECRET_KEY = None

# Import components
from agents.broker_integration import AlpacaBrokerIntegration
from agents.options_trading_agent import OptionsTrader, OptionsStrategy
from agents.options_broker import OptionsBroker
from agents.risk_management import RiskManager, RiskLevel

# Create logs directory
os.makedirs('logs', exist_ok=True)

class OptionsHunterBot:
    """Pure options trading bot optimized for Monte Carlo success"""
    
    def __init__(self):
        self.broker = None
        self.options_trader = None
        self.options_broker = None
        self.risk_manager = RiskManager(RiskLevel.MODERATE)
        
        # Trading state
        self.trade_count = 0
        self.cycle_count = 0
        self.active_positions = {}
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'largest_winner': 0.0,
            'largest_loser': 0.0
        }
        
        # High-probability options trading universe
        # Focus on liquid stocks with active options markets
        self.trading_universe = [
            # Mega cap tech (highest liquidity)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # Major ETFs (very liquid options)
            'SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLV',
            # Financial leaders
            'JPM', 'BAC', 'WFC', 'GS',
            # Healthcare leaders
            'JNJ', 'UNH', 'PFE', 'MRK',
            # Other high-volume options
            'NFLX', 'CRM', 'AMD', 'INTC', 'DIS', 'V', 'MA'
        ]
        
        # Monte Carlo optimized strategy preferences
        self.strategy_weights = {
            OptionsStrategy.BEAR_PUT_SPREAD: 0.45,    # 86.4% win rate - HIGHEST WEIGHT
            OptionsStrategy.BULL_CALL_SPREAD: 0.35,   # 71.7% win rate - SECOND HIGHEST
            OptionsStrategy.LONG_PUT: 0.12,           # Conservative bearish
            OptionsStrategy.LONG_CALL: 0.08           # Conservative bullish
        }
        
    def log_trade(self, message):
        """Log trading activity"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp}: {message}"
        
        print(log_message)
        
        try:
            with open('logs/options_hunter_bot.log', 'a') as f:
                f.write(log_message + '\n')
        except:
            pass
    
    async def initialize_systems(self):
        """Initialize all trading systems"""
        try:
            # Initialize broker
            self.broker = AlpacaBrokerIntegration(paper_trading=True)
            
            # Initialize options components
            self.options_trader = OptionsTrader(self.broker)
            self.options_broker = OptionsBroker(self.broker, paper_trading=True)
            
            # Get account info and set up risk management
            account_info = await self.broker.get_account_info()
            if account_info:
                account_value = float(account_info.get('buying_power', 100000))
                if account_value <= 0:
                    account_value = 100000  # Default for paper trading
                    
                self.risk_manager.update_account_value(account_value)
                self.log_trade(f"Connected to Alpaca: {account_info.get('account_number', 'N/A')}")
                self.log_trade(f"Account Value: ${account_value:,.2f}")
                return True
            else:
                self.log_trade("Failed to get account info - using simulation mode")
                return False
                
        except Exception as e:
            self.log_trade(f"System initialization error: {e}")
            return False
    
    async def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            return None
        except Exception as e:
            self.log_trade(f"Price fetch error for {symbol}: {e}")
            return None
    
    def calculate_market_conditions(self, symbol, price, hist_data):
        """Calculate market conditions for strategy selection"""
        try:
            if len(hist_data) < 2:
                return None
                
            # Price change
            prev_close = float(hist_data['Close'].iloc[-2])
            price_change = (price - prev_close) / prev_close
            
            # Simple volatility (5-day)
            if len(hist_data) >= 5:
                returns = hist_data['Close'].pct_change().iloc[-5:].dropna()
                volatility = returns.std() * 100 * (252**0.5)  # Annualized
            else:
                volatility = 25.0  # Default
                
            # Simple RSI approximation
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            if losses > 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50.0  # Neutral
                
            # Volume analysis
            current_volume = int(hist_data['Volume'].iloc[-1])
            avg_volume = hist_data['Volume'].iloc[-10:].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                'price_change': price_change,
                'volatility': volatility,
                'rsi': rsi,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            self.log_trade(f"Market condition calculation error for {symbol}: {e}")
            return None
    
    async def find_best_options_opportunity(self, symbol):
        """Find the best options trading opportunity for a symbol"""
        try:
            # Get current price and market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="10d")
            
            if hist.empty:
                return None
                
            current_price = float(hist['Close'].iloc[-1])
            conditions = self.calculate_market_conditions(symbol, current_price, hist)
            
            if not conditions:
                return None
            
            # Get options chain
            contracts = await self.options_trader.get_options_chain(symbol)
            if not contracts or len(contracts) < 4:  # Need enough contracts for spreads
                return None
                
            # Monte Carlo optimized strategy selection
            strategy = self.select_optimal_strategy(conditions)
            
            # Find strategy-specific opportunity
            if strategy == OptionsStrategy.BEAR_PUT_SPREAD:
                opportunity = self.find_bear_put_spread(symbol, current_price, contracts, conditions)
            elif strategy == OptionsStrategy.BULL_CALL_SPREAD:
                opportunity = self.find_bull_call_spread(symbol, current_price, contracts, conditions)
            elif strategy == OptionsStrategy.LONG_PUT:
                opportunity = self.find_long_put(symbol, current_price, contracts, conditions)
            elif strategy == OptionsStrategy.LONG_CALL:
                opportunity = self.find_long_call(symbol, current_price, contracts, conditions)
            else:
                return None
            
            if opportunity:
                opportunity['symbol'] = symbol
                opportunity['current_price'] = current_price
                opportunity['conditions'] = conditions
                
            return opportunity
            
        except Exception as e:
            self.log_trade(f"Options opportunity search error for {symbol}: {e}")
            return None
    
    def select_optimal_strategy(self, conditions):
        """Select strategy based on Monte Carlo weights and market conditions"""
        
        # Adjust weights based on market conditions
        adjusted_weights = self.strategy_weights.copy()
        
        # Strong bearish signal - increase bear put spread weight
        if conditions['price_change'] < -0.01 and conditions['rsi'] < 40:
            adjusted_weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 1.5
            adjusted_weights[OptionsStrategy.LONG_PUT] *= 1.2
            
        # Strong bullish signal - increase bull call spread weight  
        elif conditions['price_change'] > 0.01 and conditions['rsi'] > 60:
            adjusted_weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.3
            adjusted_weights[OptionsStrategy.LONG_CALL] *= 1.1
            
        # High volatility - favor spreads over long options
        if conditions['volatility'] > 30:
            adjusted_weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 1.2
            adjusted_weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.2
            adjusted_weights[OptionsStrategy.LONG_PUT] *= 0.8
            adjusted_weights[OptionsStrategy.LONG_CALL] *= 0.8
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        # Weighted random selection
        rand = random.random()
        cumulative = 0
        for strategy, weight in normalized_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return strategy
        
        # Fallback to highest probability strategy
        return OptionsStrategy.BEAR_PUT_SPREAD
    
    def find_bear_put_spread(self, symbol, price, contracts, conditions):
        """Find optimal bear put spread (86.4% win rate)"""
        puts = [c for c in contracts if c.option_type == 'put']
        if len(puts) < 2:
            return None
            
        # For bear put spread: buy higher strike, sell lower strike
        # Target strikes around current price
        long_target = price * 0.98   # Slightly OTM put (higher strike)
        short_target = price * 0.92  # Further OTM put (lower strike)
        
        # Find suitable puts
        long_puts = [p for p in puts if price * 0.95 <= p.strike <= price * 1.02]
        short_puts = [p for p in puts if price * 0.88 <= p.strike < price * 0.95]
        
        if not long_puts or not short_puts:
            return None
            
        # Select best combination (maximize spread width while minimizing cost)
        long_put = max(long_puts, key=lambda p: p.delta * -1)  # Higher delta (more negative)
        short_put = min(short_puts, key=lambda p: p.strike)    # Lower strike
        
        # Calculate spread economics
        net_debit = long_put.mid_price - short_put.mid_price
        max_profit = (long_put.strike - short_put.strike) - net_debit
        max_loss = net_debit
        
        # Profitability check
        if max_profit > 0 and (max_profit / max_loss) > 0.3:  # At least 30% return potential
            confidence = min(0.95, 0.864 + (conditions['volatility'] - 25) * 0.002)  # Base 86.4% + vol adjustment
            
            return {
                'strategy': OptionsStrategy.BEAR_PUT_SPREAD,
                'contracts': [long_put, short_put],
                'max_profit': max_profit,
                'max_loss': max_loss,
                'net_debit': net_debit,
                'confidence': confidence,
                'expected_return': max_profit / max_loss,
                'reason': f"Bear put spread ${long_put.strike:.0f}/${short_put.strike:.0f} - Max profit: ${max_profit:.2f}"
            }
        
        return None
    
    def find_bull_call_spread(self, symbol, price, contracts, conditions):
        """Find optimal bull call spread (71.7% win rate)"""
        calls = [c for c in contracts if c.option_type == 'call']
        if len(calls) < 2:
            return None
            
        # For bull call spread: buy lower strike, sell higher strike
        long_target = price * 1.02   # Slightly OTM call (lower strike)
        short_target = price * 1.08  # Further OTM call (higher strike)
        
        # Find suitable calls
        long_calls = [c for c in calls if price * 0.98 <= c.strike <= price * 1.05]
        short_calls = [c for c in calls if price * 1.05 < c.strike <= price * 1.12]
        
        if not long_calls or not short_calls:
            return None
            
        # Select best combination
        long_call = max(long_calls, key=lambda c: c.delta)     # Higher delta
        short_call = min(short_calls, key=lambda c: c.delta)   # Lower delta
        
        # Calculate spread economics
        net_debit = long_call.mid_price - short_call.mid_price
        max_profit = (short_call.strike - long_call.strike) - net_debit
        max_loss = net_debit
        
        # Profitability check
        if max_profit > 0 and (max_profit / max_loss) > 0.25:  # At least 25% return potential
            confidence = min(0.90, 0.717 + (conditions['rsi'] - 50) * 0.002)  # Base 71.7% + RSI adjustment
            
            return {
                'strategy': OptionsStrategy.BULL_CALL_SPREAD,
                'contracts': [long_call, short_call],
                'max_profit': max_profit,
                'max_loss': max_loss,
                'net_debit': net_debit,
                'confidence': confidence,
                'expected_return': max_profit / max_loss,
                'reason': f"Bull call spread ${long_call.strike:.0f}/${short_call.strike:.0f} - Max profit: ${max_profit:.2f}"
            }
        
        return None
    
    def find_long_put(self, symbol, price, contracts, conditions):
        """Find optimal long put position"""
        puts = [c for c in contracts if c.option_type == 'put']
        if not puts:
            return None
            
        # Target slightly OTM puts with good delta
        target_puts = [p for p in puts if price * 0.92 <= p.strike <= price * 0.98]
        
        if not target_puts:
            return None
            
        # Select put with best delta/price ratio
        best_put = max(target_puts, key=lambda p: abs(p.delta) / max(p.mid_price, 0.1))
        
        # Simple profitability estimate
        max_loss = best_put.mid_price
        confidence = 0.45  # Conservative for long options
        
        return {
            'strategy': OptionsStrategy.LONG_PUT,
            'contracts': [best_put],
            'max_profit': best_put.strike - best_put.mid_price,  # If stock goes to 0
            'max_loss': max_loss,
            'net_debit': best_put.mid_price,
            'confidence': confidence,
            'expected_return': 1.0,  # Simplified
            'reason': f"Long put ${best_put.strike:.0f} - Delta: {best_put.delta:.2f}"
        }
    
    def find_long_call(self, symbol, price, contracts, conditions):
        """Find optimal long call position"""
        calls = [c for c in contracts if c.option_type == 'call']
        if not calls:
            return None
            
        # Target slightly OTM calls with good delta
        target_calls = [c for c in calls if price * 1.02 <= c.strike <= price * 1.08]
        
        if not target_calls:
            return None
            
        # Select call with best delta/price ratio
        best_call = max(target_calls, key=lambda c: c.delta / max(c.mid_price, 0.1))
        
        # Simple profitability estimate
        max_loss = best_call.mid_price
        confidence = 0.40  # Conservative for long options
        
        return {
            'strategy': OptionsStrategy.LONG_CALL,
            'contracts': [best_call],
            'max_profit': float('inf'),  # Unlimited upside
            'max_loss': max_loss,
            'net_debit': best_call.mid_price,
            'confidence': confidence,
            'expected_return': 2.0,  # Simplified
            'reason': f"Long call ${best_call.strike:.0f} - Delta: {best_call.delta:.2f}"
        }
    
    async def execute_options_trade(self, opportunity):
        """Execute the options trade"""
        try:
            symbol = opportunity['symbol']
            strategy = opportunity['strategy']
            contracts = opportunity['contracts']
            
            # Risk management - position sizing
            account_value = self.risk_manager.account_value
            max_risk_per_trade = account_value * 0.02  # 2% max risk per trade
            
            # Determine quantity based on max loss
            max_loss_per_contract = opportunity['max_loss']
            max_quantity = int(max_risk_per_trade / (max_loss_per_contract * 100))  # Options are $100 multiplier
            quantity = max(1, min(max_quantity, 5))  # At least 1, max 5 contracts
            
            # Execute the strategy
            position = await self.options_trader.execute_options_strategy(
                strategy, contracts, quantity=quantity
            )
            
            if position:
                # Track the position
                self.active_positions[position.symbol] = {
                    'position': position,
                    'opportunity': opportunity,
                    'entry_time': datetime.now()
                }
                
                # Update performance stats
                self.performance_stats['total_trades'] += 1
                
                # Log successful trade
                cost = opportunity['net_debit'] * quantity * 100
                self.log_trade(f"OPTIONS EXECUTED: {symbol} {strategy} - Qty: {quantity} - Cost: ${cost:.2f} - Max Profit: ${opportunity['max_profit'] * quantity * 100:.2f}")
                
                return True
            
            return False
            
        except Exception as e:
            self.log_trade(f"Trade execution error: {e}")
            return False
    
    def update_performance_stats(self, realized_pnl):
        """Update performance statistics"""
        self.performance_stats['total_profit'] += realized_pnl
        
        if realized_pnl > 0:
            self.performance_stats['winning_trades'] += 1
            self.performance_stats['largest_winner'] = max(
                self.performance_stats['largest_winner'], realized_pnl
            )
        else:
            self.performance_stats['largest_loser'] = min(
                self.performance_stats['largest_loser'], realized_pnl
            )
        
        # Calculate drawdown
        if realized_pnl < 0:
            self.performance_stats['max_drawdown'] = min(
                self.performance_stats['max_drawdown'], 
                self.performance_stats['total_profit']
            )
    
    async def scan_and_trade(self):
        """Main trading cycle - scan universe and execute best opportunities"""
        self.cycle_count += 1
        self.log_trade(f"=== OPTIONS HUNT CYCLE #{self.cycle_count} ===")
        
        opportunities = []
        
        # Scan trading universe
        for symbol in random.sample(self.trading_universe, min(15, len(self.trading_universe))):
            self.log_trade(f"Scanning {symbol}...")
            opportunity = await self.find_best_options_opportunity(symbol)
            
            if opportunity:
                opportunities.append(opportunity)
                self.log_trade(f"OPPORTUNITY: {symbol} {opportunity['strategy']} - Confidence: {opportunity['confidence']:.1%} - {opportunity['reason']}")
        
        # Sort by confidence * expected return
        opportunities.sort(key=lambda x: x['confidence'] * x['expected_return'], reverse=True)
        
        # Execute top opportunities (limit to 3 per cycle for risk management)
        executed = 0
        for opportunity in opportunities[:3]:
            success = await self.execute_options_trade(opportunity)
            if success:
                executed += 1
            
            # Small delay between trades
            await asyncio.sleep(1)
        
        self.log_trade(f"Cycle complete: {len(opportunities)} opportunities found, {executed} trades executed")
        
        # Log performance stats every 5 cycles
        if self.cycle_count % 5 == 0:
            await self.log_performance_stats()
    
    async def log_performance_stats(self):
        """Log current performance statistics"""
        stats = self.performance_stats
        win_rate = (stats['winning_trades'] / max(stats['total_trades'], 1)) * 100
        
        self.log_trade("=== PERFORMANCE STATS ===")
        self.log_trade(f"Total Trades: {stats['total_trades']}")
        self.log_trade(f"Win Rate: {win_rate:.1f}%")
        self.log_trade(f"Total P&L: ${stats['total_profit']:.2f}")
        self.log_trade(f"Max Drawdown: ${stats['max_drawdown']:.2f}")
        self.log_trade(f"Largest Winner: ${stats['largest_winner']:.2f}")
        self.log_trade(f"Largest Loser: ${stats['largest_loser']:.2f}")
        self.log_trade(f"Active Positions: {len(self.active_positions)}")
    
    async def start_options_hunting(self):
        """Main entry point - start the options trading bot"""
        print("OPTIONS HUNTER BOT - MONTE CARLO OPTIMIZED")
        print("=" * 55)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Trading Universe: {len(self.trading_universe)} symbols")
        print(f"Strategy Focus: Bear Put Spreads (86.4%) & Bull Call Spreads (71.7%)")
        print(f"Risk Management: 2% max risk per trade")
        print("-" * 55)
        
        # Initialize systems
        initialized = await self.initialize_systems()
        if not initialized:
            self.log_trade("System initialization failed - exiting")
            return
        
        self.log_trade("Options Hunter Bot started successfully")
        
        try:
            while True:
                # Check market hours (9:30 AM - 4:00 PM ET)
                import pytz
                et = pytz.timezone('US/Eastern')
                current_time = datetime.now(et)
                hour = current_time.hour
                minute = current_time.minute
                
                if (hour > 9 or (hour == 9 and minute >= 30)) and hour < 16:
                    await self.scan_and_trade()
                    # Wait 10 minutes between cycles during market hours
                    await asyncio.sleep(600)
                else:
                    # Market closed - wait and check again
                    self.log_trade(f"Market closed ({current_time.strftime('%H:%M')} ET) - waiting...")
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
        except KeyboardInterrupt:
            self.log_trade("Options Hunter Bot stopped by user")
            await self.log_performance_stats()
        except Exception as e:
            self.log_trade(f"Bot error: {e}")
            import traceback
            self.log_trade(f"Traceback: {traceback.format_exc()}")

async def main():
    """Main entry point"""
    bot = OptionsHunterBot()
    await bot.start_options_hunting()

if __name__ == "__main__":
    asyncio.run(main())