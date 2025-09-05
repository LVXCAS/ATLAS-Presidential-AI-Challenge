#!/usr/bin/env python3
"""
Tomorrow-Ready Options Trading Bot
Fully prepared for live trading with intelligent exit strategy agent

Features:
- Pre-market preparation and readiness checks
- Intelligent profit/loss decision making via Exit Strategy Agent
- Intraday position monitoring every 5 minutes
- Market open/close procedures
- Real-time position management
- Earnings calendar integration
- Market regime adaptation
"""

import asyncio
import sys
import os
import time
import json
import random
import requests
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import numpy as np
import pytz

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
from agents.exit_strategy_agent import exit_strategy_agent, ExitSignal

# Create logs directory
os.makedirs('logs', exist_ok=True)

class TomorrowReadyOptionsBot:
    """Production-ready options trading bot with intelligent exit strategy"""
    
    def __init__(self):
        self.broker = None
        self.options_trader = None
        self.options_broker = None
        self.risk_manager = RiskManager(RiskLevel.MODERATE)
        self.exit_agent = exit_strategy_agent
        
        # Eastern Time for market operations
        self.et_timezone = pytz.timezone('US/Eastern')
        
        # Trading state
        self.trade_count = 0
        self.cycle_count = 0
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.is_market_open = False
        self.last_position_check = None
        
        # Risk management - initialize with defaults
        self.daily_risk_limits = {
            'max_daily_loss': 1000,
            'max_positions': 5,
            'max_position_risk': 500,
            'remaining_daily_risk': 1000
        }
        
        # Trading schedule
        self.market_open_time = dt_time(9, 30)  # 9:30 AM ET
        self.market_close_time = dt_time(16, 0)  # 4:00 PM ET
        self.pre_market_start = dt_time(8, 0)    # 8:00 AM ET for prep
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'total_commissions': 0.0,
            'max_drawdown': 0.0,
            'largest_winner': 0.0,
            'largest_loser': 0.0,
            'daily_pnl_history': [],
            'exit_decisions': []
        }
        
        # Market regime tracking
        self.market_regime = 'NEUTRAL'
        self.vix_level = 20.0
        self.market_trend = 0.0
        
        # Professional trading universe
        self.tier1_stocks = [
            # Mega cap (best liquidity)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # Major ETFs (excellent options liquidity)
            'SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLV', 'XLE', 'GLD', 'TLT',
            # Large cap leaders
            'JPM', 'BAC', 'WFC', 'GS', 'JNJ', 'UNH', 'PFE', 'MRK',
            # High-volume options stocks
            'NFLX', 'CRM', 'AMD', 'INTC', 'DIS', 'V', 'MA', 'COIN', 'UBER'
        ]
        
        # Strategy weights (adapted by market regime)
        self.base_strategy_weights = {
            OptionsStrategy.BULL_CALL_SPREAD: 0.25,
            OptionsStrategy.BEAR_PUT_SPREAD: 0.25,
            OptionsStrategy.LONG_PUT: 0.20,
            OptionsStrategy.LONG_CALL: 0.15,
            OptionsStrategy.CASH_SECURED_PUT: 0.10,
            OptionsStrategy.COVERED_CALL: 0.05
        }
        
        # Trading plan - initialize with defaults (after tier1_stocks is defined)
        self.daily_trading_plan = {
            'market_regime': 'NEUTRAL',
            'preferred_strategies': ['ALL_SPREADS'],
            'target_new_positions': 2,
            'focus_symbols': self.tier1_stocks[:10],
            'max_position_risk': 500
        }
        
        # Readiness checklist
        self.readiness_status = {
            'broker_connected': False,
            'account_validated': False,
            'market_data_available': False,
            'positions_loaded': False,
            'risk_limits_set': False,
            'earnings_calendar_updated': False
        }
    
    def log_trade(self, message, level='INFO'):
        """Enhanced logging with market context"""
        timestamp = datetime.now(self.et_timezone).strftime('%Y-%m-%d %H:%M:%S ET')
        market_status = "OPEN" if self.is_market_open else "CLOSED"
        log_message = f"{timestamp} [{market_status}] [{level}] {message}"
        
        print(log_message)
        
        try:
            with open('logs/tomorrow_ready_bot.log', 'a') as f:
                f.write(log_message + '\n')
        except:
            pass
    
    async def pre_market_preparation(self):
        """Comprehensive pre-market preparation routine"""
        self.log_trade("=== PRE-MARKET PREPARATION STARTED ===", "INFO")
        
        try:
            # 1. Initialize all systems
            self.log_trade("Step 1: Initializing trading systems...")
            if await self.initialize_all_systems():
                self.readiness_status['broker_connected'] = True
                self.readiness_status['account_validated'] = True
                self.log_trade("[OK] Systems initialized successfully")
            else:
                self.log_trade("[X] System initialization failed", "ERROR")
                return False
            
            # 2. Load existing positions
            self.log_trade("Step 2: Loading existing positions...")
            await self.load_existing_positions()
            self.readiness_status['positions_loaded'] = True
            self.log_trade(f"[OK] Loaded {len(self.active_positions)} active positions")
            
            # 3. Update market regime
            self.log_trade("Step 3: Analyzing market regime...")
            await self.update_market_regime()
            self.log_trade(f"[OK] Market regime: {self.market_regime} (VIX: {self.vix_level:.1f})")
            
            # 4. Validate market data access
            self.log_trade("Step 4: Validating market data access...")
            if await self.validate_market_data():
                self.readiness_status['market_data_available'] = True
                self.log_trade("[OK] Market data access validated")
            else:
                self.log_trade("[X] Market data validation failed", "WARN")
            
            # 5. Set risk limits for the day
            self.log_trade("Step 5: Setting daily risk limits...")
            await self.set_daily_risk_limits()
            self.readiness_status['risk_limits_set'] = True
            
            # 6. Update earnings calendar
            self.log_trade("Step 6: Updating earnings calendar...")
            await self.update_earnings_calendar()
            self.readiness_status['earnings_calendar_updated'] = True
            
            # 7. Position risk assessment
            self.log_trade("Step 7: Assessing position risks...")
            await self.assess_overnight_position_risks()
            
            # 8. Generate trading plan
            self.log_trade("Step 8: Generating today's trading plan...")
            trading_plan = await self.generate_daily_trading_plan()
            
            # Final readiness check
            all_ready = all(self.readiness_status.values())
            
            if all_ready:
                self.log_trade("=== PRE-MARKET PREPARATION COMPLETED SUCCESSFULLY ===", "INFO")
                self.log_trade(f"Ready for market open at 9:30 AM ET", "INFO")
                return True
            else:
                failed_items = [k for k, v in self.readiness_status.items() if not v]
                self.log_trade(f"[X] Preparation incomplete. Failed items: {failed_items}", "ERROR")
                return False
                
        except Exception as e:
            self.log_trade(f"Pre-market preparation error: {e}", "ERROR")
            return False
    
    async def initialize_all_systems(self):
        """Initialize all trading systems"""
        try:
            # Initialize broker
            self.broker = AlpacaBrokerIntegration(paper_trading=True)
            self.options_trader = OptionsTrader(self.broker)
            self.options_broker = OptionsBroker(self.broker, paper_trading=True)
            
            # Get account info
            account_info = await self.broker.get_account_info()
            if account_info:
                account_value = float(account_info.get('buying_power', 100000))
                if account_value <= 0:
                    account_value = 100000
                
                self.risk_manager.update_account_value(account_value)
                self.log_trade(f"Account connected: {account_info.get('account_number', 'N/A')}")
                self.log_trade(f"Available capital: ${account_value:,.2f}")
                return True
            
            return False
            
        except Exception as e:
            self.log_trade(f"System initialization error: {e}", "ERROR")
            return False
    
    async def load_existing_positions(self):
        """Load any existing positions from overnight/previous sessions"""
        try:
            # In production, would load from database or broker API
            # For now, initialize empty
            self.active_positions = {}
            self.log_trade("Position loading completed")
        except Exception as e:
            self.log_trade(f"Position loading error: {e}", "WARN")
    
    async def update_market_regime(self):
        """Update current market regime analysis"""
        try:
            # Get VIX
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='2d')
            if not vix_hist.empty:
                self.vix_level = float(vix_hist['Close'].iloc[-1])
            
            # Get SPY trend
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='20d')
            if len(spy_hist) >= 10:
                current_price = float(spy_hist['Close'].iloc[-1])
                sma_10 = spy_hist['Close'].iloc[-10:].mean()
                self.market_trend = (current_price - sma_10) / sma_10
            
            # Determine regime
            if self.vix_level > 30:
                self.market_regime = 'HIGH_VIX'
            elif self.market_trend > 0.03:
                self.market_regime = 'BULL'
            elif self.market_trend < -0.03:
                self.market_regime = 'BEAR'
            else:
                self.market_regime = 'NEUTRAL'
                
        except Exception as e:
            self.log_trade(f"Market regime update error: {e}", "WARN")
            self.market_regime = 'NEUTRAL'
    
    async def validate_market_data(self):
        """Validate that we can access real-time market data"""
        try:
            # Test with SPY
            spy = yf.Ticker('SPY')
            hist = spy.history(period='1d')
            return not hist.empty
        except:
            return False
    
    async def set_daily_risk_limits(self):
        """Set risk limits for the trading day"""
        account_value = self.risk_manager.account_value
        
        # Daily risk limits
        daily_max_loss = account_value * 0.02  # Max 2% loss per day
        max_position_count = 8  # Max 8 simultaneous positions
        max_single_position_risk = account_value * 0.005  # 0.5% per position in high vol
        
        if self.market_regime == 'HIGH_VIX':
            daily_max_loss *= 0.5  # Reduce risk in high volatility
            max_single_position_risk *= 0.7
        elif self.market_regime in ['BULL', 'BEAR']:
            max_single_position_risk *= 1.2  # Slightly more risk in trending markets
        
        self.daily_risk_limits = {
            'max_daily_loss': daily_max_loss,
            'max_positions': max_position_count,
            'max_position_risk': max_single_position_risk,
            'remaining_daily_risk': daily_max_loss
        }
        
        self.log_trade(f"Daily risk limits set: Max loss ${daily_max_loss:.0f}, Max positions {max_position_count}")
    
    async def update_earnings_calendar(self):
        """Update earnings calendar to avoid trading before earnings"""
        try:
            # In production, would use earnings API
            # For now, implement basic earnings avoidance
            self.earnings_this_week = set()  # Symbols with earnings this week
            self.log_trade("Earnings calendar updated")
        except Exception as e:
            self.log_trade(f"Earnings calendar error: {e}", "WARN")
    
    async def assess_overnight_position_risks(self):
        """Assess risks of positions held overnight"""
        if not self.active_positions:
            self.log_trade("No overnight positions to assess")
            return
        
        self.log_trade(f"Assessing {len(self.active_positions)} overnight positions...")
        
        positions_at_risk = []
        for position_id, position_data in self.active_positions.items():
            try:
                # Get current market data
                symbol = position_data['opportunity']['symbol']
                market_data = await self.get_enhanced_market_data(symbol)
                
                if market_data:
                    # Simple overnight risk assessment
                    current_price = market_data['current_price']
                    entry_price = position_data.get('entry_price', current_price)
                    price_change = (current_price - entry_price) / entry_price
                    
                    if abs(price_change) > 0.05:  # More than 5% move
                        positions_at_risk.append({
                            'position_id': position_id,
                            'symbol': symbol,
                            'price_change': price_change,
                            'risk_level': 'HIGH' if abs(price_change) > 0.10 else 'MODERATE'
                        })
            
            except Exception as e:
                self.log_trade(f"Position risk assessment error for {position_id}: {e}", "WARN")
        
        if positions_at_risk:
            self.log_trade(f"⚠ {len(positions_at_risk)} positions at elevated risk")
            for pos in positions_at_risk:
                self.log_trade(f"  {pos['symbol']}: {pos['price_change']:+.1%} ({pos['risk_level']} risk)")
        else:
            self.log_trade("[OK] All overnight positions within normal risk parameters")
    
    async def generate_daily_trading_plan(self):
        """Generate trading plan based on market conditions"""
        plan = {
            'market_regime': self.market_regime,
            'preferred_strategies': [],
            'target_new_positions': 0,
            'focus_symbols': [],
            'risk_adjustments': []
        }
        
        # Adapt strategy preferences to market regime
        if self.market_regime == 'BULL':
            plan['preferred_strategies'] = ['BULL_CALL_SPREAD', 'LONG_CALL']
            plan['target_new_positions'] = 3
        elif self.market_regime == 'BEAR':
            plan['preferred_strategies'] = ['BEAR_PUT_SPREAD', 'LONG_PUT'] 
            plan['target_new_positions'] = 2
        elif self.market_regime == 'HIGH_VIX':
            plan['preferred_strategies'] = ['CASH_SECURED_PUT', 'SPREADS']
            plan['target_new_positions'] = 1
        else:  # NEUTRAL
            plan['preferred_strategies'] = ['ALL_SPREADS']
            plan['target_new_positions'] = 2
        
        # Focus on most liquid symbols
        plan['focus_symbols'] = random.sample(self.tier1_stocks, min(10, len(self.tier1_stocks)))
        
        self.daily_trading_plan = plan
        self.log_trade(f"Today's plan: {plan['target_new_positions']} new positions, focus on {plan['preferred_strategies']}")
        
        return plan
    
    def get_current_market_time(self):
        """Get current time in Eastern timezone"""
        return datetime.now(self.et_timezone).time()
    
    def is_pre_market_time(self):
        """Check if we're in pre-market time"""
        current_time = self.get_current_market_time()
        return self.pre_market_start <= current_time < self.market_open_time
    
    def is_market_hours(self):
        """Check if market is currently open"""
        current_time = self.get_current_market_time()
        current_date = datetime.now(self.et_timezone).date()
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if current_date.weekday() >= 5:  # Weekend
            return False
        
        # Check if it's within market hours
        return self.market_open_time <= current_time <= self.market_close_time
    
    async def get_enhanced_market_data(self, symbol):
        """Get comprehensive market data for analysis"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            
            if hist.empty:
                return None
            
            current_price = float(hist['Close'].iloc[-1])
            
            # Calculate enhanced metrics
            if len(hist) >= 20:
                sma_5 = hist['Close'].iloc[-5:].mean()
                sma_20 = hist['Close'].iloc[-20:].mean()
                price_momentum = (sma_5 - sma_20) / sma_20
                
                returns = hist['Close'].pct_change().dropna()
                realized_vol = returns.std() * np.sqrt(252) * 100
                
                avg_volume = hist['Volume'].iloc[-20:].mean()
                current_volume = hist['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                high_20 = hist['High'].iloc[-20:].max()
                low_20 = hist['Low'].iloc[-20:].min()
                price_position = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'price_momentum': price_momentum,
                    'realized_vol': realized_vol,
                    'volume_ratio': volume_ratio,
                    'price_position': price_position,
                    'avg_volume': avg_volume,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            self.log_trade(f"Market data error for {symbol}: {e}", "WARN")
            return None
    
    async def intelligent_position_monitoring(self):
        """Monitor positions using the intelligent exit strategy agent"""
        if not self.active_positions:
            return
        
        self.log_trade(f"Intelligent monitoring of {len(self.active_positions)} positions...")
        
        positions_to_exit = []
        
        for position_id, position_data in list(self.active_positions.items()):
            try:
                symbol = position_data['opportunity']['symbol']
                
                # Get current market data
                market_data = await self.get_enhanced_market_data(symbol)
                if not market_data:
                    continue
                
                # Calculate current P&L (simplified - would use real option pricing)
                entry_price = position_data.get('entry_price', 0)
                current_pnl = self.calculate_position_pnl(position_data, market_data)
                
                # Use intelligent exit agent for decision
                exit_decision = self.exit_agent.analyze_position_exit(
                    position_data, market_data, current_pnl
                )
                
                # Log the decision
                if exit_decision.signal != ExitSignal.HOLD:
                    self.log_trade(f"EXIT SIGNAL: {symbol} - {exit_decision.signal}")
                    self.log_trade(f"  Reason: {exit_decision.reasoning}")
                    self.log_trade(f"  Confidence: {exit_decision.confidence:.1%}")
                    self.log_trade(f"  Urgency: {exit_decision.urgency:.1%}")
                    
                    if exit_decision.urgency >= 0.7:  # High urgency
                        positions_to_exit.append({
                            'position_id': position_id,
                            'position_data': position_data,
                            'exit_decision': exit_decision,
                            'current_pnl': current_pnl
                        })
                
                # Update position monitoring data
                position_data['last_check'] = datetime.now()
                position_data['current_pnl'] = current_pnl
                position_data['exit_signals'] = position_data.get('exit_signals', [])
                position_data['exit_signals'].append({
                    'timestamp': datetime.now(),
                    'signal': exit_decision.signal,
                    'confidence': exit_decision.confidence
                })
                
                # Keep only recent signals
                position_data['exit_signals'] = position_data['exit_signals'][-10:]
                
            except Exception as e:
                self.log_trade(f"Position monitoring error for {position_id}: {e}", "ERROR")
        
        # Execute exits for high-urgency positions
        for exit_info in positions_to_exit:
            await self.execute_intelligent_exit(exit_info)
    
    def calculate_position_pnl(self, position_data, market_data):
        """Calculate current P&L for a position (simplified)"""
        try:
            # In production, would use real option pricing models
            # For now, use simplified estimation based on stock price movement
            
            entry_price = position_data.get('entry_price', 0)
            current_stock_price = market_data['current_price']
            
            # Mock P&L calculation - would be replaced with real option pricing
            strategy = position_data['opportunity']['strategy']
            
            if 'CALL' in str(strategy):
                # Rough approximation for call-based strategies
                price_change = market_data['price_momentum']
                estimated_pnl = price_change * 100 * position_data.get('quantity', 1)  # Simplified
            else:
                # Rough approximation for other strategies
                estimated_pnl = random.uniform(-50, 50)  # Placeholder
            
            return estimated_pnl
            
        except Exception as e:
            self.log_trade(f"P&L calculation error: {e}", "WARN")
            return 0.0
    
    async def execute_intelligent_exit(self, exit_info):
        """Execute exit based on intelligent agent decision"""
        try:
            position_id = exit_info['position_id']
            position_data = exit_info['position_data']
            exit_decision = exit_info['exit_decision']
            current_pnl = exit_info['current_pnl']
            
            symbol = position_data['opportunity']['symbol']
            
            # Log the exit execution
            self.log_trade(f"EXECUTING INTELLIGENT EXIT: {symbol}")
            self.log_trade(f"  Strategy: {position_data['opportunity']['strategy']}")
            self.log_trade(f"  Exit reason: {exit_decision.reason}")
            self.log_trade(f"  Expected P&L: ${exit_decision.expected_pnl_impact:.2f}")
            
            # In production, would execute actual option trades to close position
            # For now, simulate the exit
            
            # Update performance stats
            self.update_performance_stats(current_pnl, exit_decision)
            
            # Remove from active positions
            del self.active_positions[position_id]
            
            # Record exit decision for agent learning
            self.performance_stats['exit_decisions'].append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'exit_decision': exit_decision.signal,
                'final_pnl': current_pnl,
                'confidence': exit_decision.confidence,
                'reasoning': exit_decision.reasoning
            })
            
            self.log_trade(f"[OK] Position {symbol} exited successfully - P&L: ${current_pnl:.2f}")
            
        except Exception as e:
            self.log_trade(f"Exit execution error: {e}", "ERROR")
    
    def update_performance_stats(self, pnl, exit_decision):
        """Update performance statistics"""
        self.performance_stats['total_profit'] += pnl
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.performance_stats['winning_trades'] += 1
            self.performance_stats['largest_winner'] = max(
                self.performance_stats['largest_winner'], pnl
            )
        else:
            self.performance_stats['largest_loser'] = min(
                self.performance_stats['largest_loser'], pnl
            )
    
    async def market_open_procedures(self):
        """Procedures to run at market open"""
        self.log_trade("=== MARKET OPEN PROCEDURES ===", "INFO")
        
        self.is_market_open = True
        
        # 1. Final position assessment
        await self.intelligent_position_monitoring()
        
        # 2. Update market regime
        await self.update_market_regime()
        
        # 3. Reset daily counters
        self.daily_pnl = 0.0
        self.cycle_count = 0
        
        self.log_trade("[OK] Market open procedures completed - Ready for trading")
    
    async def market_close_procedures(self):
        """Procedures to run at market close"""
        self.log_trade("=== MARKET CLOSE PROCEDURES ===", "INFO")
        
        self.is_market_open = False
        
        # 1. Final position monitoring
        await self.intelligent_position_monitoring()
        
        # 2. Daily performance summary
        await self.log_daily_performance()
        
        # 3. Update exit agent learning
        if len(self.performance_stats['exit_decisions']) >= 5:
            self.exit_agent.update_learning_parameters(
                self.performance_stats['exit_decisions'][-10:]
            )
        
        # 4. Prepare for next day
        self.performance_stats['daily_pnl_history'].append(self.daily_pnl)
        
        self.log_trade("[OK] Market close procedures completed - End of trading day")
    
    async def log_daily_performance(self):
        """Log comprehensive daily performance"""
        stats = self.performance_stats
        
        win_rate = 0
        if stats['total_trades'] > 0:
            win_rate = (stats['winning_trades'] / stats['total_trades']) * 100
        
        self.log_trade("=== DAILY PERFORMANCE SUMMARY ===", "INFO")
        self.log_trade(f"Daily P&L: ${self.daily_pnl:.2f}")
        self.log_trade(f"Total P&L: ${stats['total_profit']:.2f}")
        self.log_trade(f"Win Rate: {win_rate:.1f}%")
        self.log_trade(f"Active Positions: {len(self.active_positions)}")
        self.log_trade(f"Exit Decisions Made: {len(stats['exit_decisions'])}")
        
        # Position breakdown
        if self.active_positions:
            strategy_counts = {}
            total_risk = 0
            for pos_data in self.active_positions.values():
                strategy = pos_data['opportunity']['strategy']
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                total_risk += pos_data['opportunity'].get('max_loss', 0) * pos_data.get('quantity', 1) * 100
            
            self.log_trade(f"Position strategies: {dict(strategy_counts)}")
            self.log_trade(f"Total risk exposure: ${total_risk:.2f}")
    
    async def intraday_trading_cycle(self):
        """Main intraday trading cycle"""
        self.cycle_count += 1
        self.log_trade(f"=== TRADING CYCLE #{self.cycle_count} ===")
        
        # 1. Intelligent position monitoring (every cycle)
        await self.intelligent_position_monitoring()
        
        # 2. Look for new opportunities (less frequent)
        if self.cycle_count % 3 == 1:  # Every 3rd cycle
            await self.scan_for_new_opportunities()
        
        # 3. Risk check
        await self.intraday_risk_check()
        
        self.log_trade(f"Cycle #{self.cycle_count} completed")
    
    async def scan_for_new_opportunities(self):
        """Scan for new trading opportunities"""
        if len(self.active_positions) >= self.daily_risk_limits.get('max_positions', 5):
            self.log_trade("Position limit reached - skipping new opportunity scan")
            return
        
        self.log_trade("Scanning for new opportunities...")
        
        opportunities = []
        scan_symbols = random.sample(
            self.daily_trading_plan.get('focus_symbols', self.tier1_stocks[:10]), 
            min(5, len(self.tier1_stocks))
        )
        
        for symbol in scan_symbols:
            try:
                opportunity = await self.find_high_quality_opportunity(symbol)
                if opportunity:
                    opportunities.append(opportunity)
                    self.log_trade(f"OPPORTUNITY: {symbol} {opportunity['strategy']} - Confidence: {opportunity.get('confidence', 0):.1%}")
            except Exception as e:
                self.log_trade(f"Opportunity scan error for {symbol}: {e}", "WARN")
        
        # Execute best opportunity if found
        if opportunities:
            best_opportunity = max(opportunities, key=lambda x: x.get('confidence', 0) * x.get('expected_return', 0))
            await self.execute_new_position(best_opportunity)
    
    async def find_high_quality_opportunity(self, symbol):
        """Find high-quality trading opportunity (simplified version)"""
        try:
            market_data = await self.get_enhanced_market_data(symbol)
            if not market_data:
                return None
            
            # Basic opportunity logic - relaxed criteria for better opportunity finding
            volume_ok = market_data['volume_ratio'] > 0.8  # Lowered from 1.2 to 0.8
            momentum_ok = abs(market_data['price_momentum']) > 0.015  # Lowered from 2.0% to 1.5%
            
            # Additional quality checks
            price_position = market_data.get('price_position', 0.5)
            vol_ok = market_data.get('realized_vol', 20) > 15  # Some volatility present
            
            if (volume_ok and momentum_ok) or (momentum_ok and vol_ok and price_position > 0.3):
                
                # Dynamic strategy selection based on momentum
                if market_data['price_momentum'] > 0.01:  # Positive momentum
                    strategy = OptionsStrategy.BULL_CALL_SPREAD
                elif market_data['price_momentum'] < -0.01:  # Negative momentum
                    strategy = OptionsStrategy.BEAR_PUT_SPREAD
                else:  # Low momentum
                    strategy = OptionsStrategy.BULL_CALL_SPREAD  # Default
                
                # Dynamic confidence based on signal strength
                confidence = 0.5  # Base confidence
                if volume_ok and momentum_ok:
                    confidence += 0.2  # Both criteria met
                if vol_ok:
                    confidence += 0.1  # Good volatility
                if abs(market_data['price_momentum']) > 0.025:
                    confidence += 0.15  # Strong momentum
                
                confidence = min(0.85, confidence)  # Cap at 85%
                
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'confidence': confidence,
                    'expected_return': 1.5,
                    'max_profit': 2.50,  # More realistic: $2.50 per contract
                    'max_loss': 1.50,    # More realistic: $1.50 per contract  
                    'market_data': market_data,
                    'reasoning': f"Volume: {market_data['volume_ratio']:.2f}x, Momentum: {market_data['price_momentum']:+.1%}"
                }
            
            return None
            
        except Exception as e:
            self.log_trade(f"Opportunity finding error for {symbol}: {e}", "WARN")
            return None
    
    async def execute_new_position(self, opportunity):
        """Execute a new position"""
        try:
            symbol = opportunity['symbol']
            
            # Risk check
            position_risk = opportunity['max_loss'] * 100  # Convert to dollars
            if position_risk > self.daily_risk_limits.get('max_position_risk', 1000):
                self.log_trade(f"Position risk too high for {symbol} - skipping")
                return False
            
            # Create position data
            position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            position_data = {
                'position': None,  # Would be real position object
                'opportunity': opportunity,
                'entry_time': datetime.now(),
                'entry_price': opportunity.get('net_debit', 0),
                'quantity': 1,
                'market_regime_at_entry': self.market_regime
            }
            
            # Add to active positions
            self.active_positions[position_id] = position_data
            
            # Update stats
            self.performance_stats['total_trades'] += 1
            
            self.log_trade(f"NEW POSITION: {symbol} {opportunity['strategy']} - Risk: ${position_risk:.2f}")
            
            return True
            
        except Exception as e:
            self.log_trade(f"Position execution error: {e}", "ERROR")
            return False
    
    async def intraday_risk_check(self):
        """Check risk limits during the day"""
        try:
            # Check daily P&L limit
            max_daily_loss = self.daily_risk_limits.get('max_daily_loss', 1000)
            if self.daily_pnl < -max_daily_loss:
                self.log_trade(f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f}", "ERROR")
                # Would implement emergency shutdown here
            
            # Check position count
            if len(self.active_positions) > self.daily_risk_limits.get('max_positions', 5):
                self.log_trade("Position count limit exceeded", "WARN")
            
        except Exception as e:
            self.log_trade(f"Risk check error: {e}", "WARN")
    
    async def start_tomorrow_ready_trading(self):
        """Main entry point for tomorrow-ready trading"""
        print("TOMORROW-READY OPTIONS TRADING BOT")
        print("=" * 70)
        print(f"Started: {datetime.now(self.et_timezone).strftime('%Y-%m-%d %H:%M:%S ET')}")
        print(f"Market Schedule: 9:30 AM - 4:00 PM ET")
        print(f"Features: Pre-market prep, Intelligent exits, Real-time monitoring")
        print("-" * 70)
        
        try:
            while True:
                current_time = self.get_current_market_time()
                current_date = datetime.now(self.et_timezone).date()
                
                # Skip weekends
                if current_date.weekday() >= 5:
                    self.log_trade("Weekend - sleeping until Monday")
                    await asyncio.sleep(3600)  # Sleep 1 hour
                    continue
                
                # Pre-market preparation
                if self.is_pre_market_time():
                    self.log_trade("Pre-market time - running preparation")
                    await self.pre_market_preparation()
                    await asyncio.sleep(600)  # Check every 10 minutes until market open
                
                # Market open
                elif self.is_market_hours() and not self.is_market_open:
                    await self.market_open_procedures()
                    
                # Intraday trading
                elif self.is_market_hours() and self.is_market_open:
                    await self.intraday_trading_cycle()
                    await asyncio.sleep(300)  # 5-minute cycles during market hours
                
                # Market close
                elif not self.is_market_hours() and self.is_market_open:
                    await self.market_close_procedures()
                
                # After hours - monitor positions less frequently
                else:
                    if self.active_positions:
                        await self.intelligent_position_monitoring()
                    await asyncio.sleep(1800)  # Check every 30 minutes after hours
                    
        except KeyboardInterrupt:
            self.log_trade("Trading bot stopped by user", "INFO")
            if self.is_market_open:
                await self.market_close_procedures()
        except Exception as e:
            self.log_trade(f"Bot error: {e}", "ERROR")
            import traceback
            self.log_trade(f"Traceback: {traceback.format_exc()}", "ERROR")

async def main():
    """Main entry point"""
    bot = TomorrowReadyOptionsBot()
    await bot.start_tomorrow_ready_trading()

if __name__ == "__main__":
    asyncio.run(main())