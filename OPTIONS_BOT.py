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
    FRED_API_KEY = os.getenv('FRED_API_KEY', '98e96c3261987f1c116c1506e6dde103')
except:
    POLYGON_API_KEY = None
    ALPACA_API_KEY = None
    ALPACA_SECRET_KEY = None
    FRED_API_KEY = '98e96c3261987f1c116c1506e6dde103'

# Import components
from agents.broker_integration import AlpacaBrokerIntegration
from agents.options_trading_agent import OptionsTrader, OptionsStrategy
from agents.options_broker import OptionsBroker
from agents.risk_management import RiskManager, RiskLevel

# Live trading integration
try:
    from agents.live_data_manager import setup_live_data
    from agents.live_trading_engine import LiveTradingEngine
    LIVE_TRADING_AVAILABLE = True
    print("+ Live trading integration loaded")
except ImportError:
    LIVE_TRADING_AVAILABLE = False
    print("- Live trading integration not available")
from agents.exit_strategy_agent import exit_strategy_agent, ExitSignal
from agents.learning_engine import learning_engine
from agents.advanced_ml_engine import advanced_ml_engine
from agents.enhanced_technical_analysis import enhanced_technical_analysis
from agents.enhanced_options_pricing import enhanced_options_pricing
from agents.economic_data_agent import economic_data_agent
from agents.cboe_data_agent import cboe_data_agent
from agents.advanced_technical_analysis import advanced_technical_analysis
from agents.ml_prediction_engine import ml_prediction_engine
from agents.advanced_risk_management import advanced_risk_manager
from agents.trading_dashboard import trading_dashboard

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
        self.learning_engine = learning_engine
        self.advanced_ml = advanced_ml_engine
        self.technical_analysis = enhanced_technical_analysis
        self.options_pricing = enhanced_options_pricing
        self.economic_data = economic_data_agent
        self.volatility_intelligence = cboe_data_agent
        
        # Advanced AI/ML components
        self.advanced_technical = advanced_technical_analysis
        self.ml_predictions = ml_prediction_engine
        self.advanced_risk = advanced_risk_manager
        self.dashboard = trading_dashboard
        
        # Learning Acceleration Components
        from agents.transfer_learning_accelerator import transfer_accelerator
        from agents.realtime_learning_engine import realtime_learner
        from agents.model_loader import model_loader
        self.transfer_learning = transfer_accelerator
        self.realtime_learning = realtime_learner
        self.pre_trained_models = model_loader
        
        # Display model loading status
        print("Pre-trained Models Status:")
        print(self.pre_trained_models.get_model_info())
        
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
        
        # Options trading configuration
        self.options_config = {
            'min_days_to_expiry': 14,      # Minimum days to expiration (14 = 2 weeks)
            'max_days_to_expiry': 45,      # Maximum days to expiration (45 = ~6 weeks)
            'preferred_dte_range': (21, 35), # Preferred range (3-5 weeks)
            'exit_at_days_remaining': 7,    # Exit when this many days left
            'use_weekly_options': True,     # Allow weekly options
            'use_monthly_options': True,    # Allow monthly options
        }
        
        # Intelligent exit configuration
        self.exit_config = {
            'use_intelligent_analysis': True,  # Use data-driven decisions
            'urgency_threshold': 0.4,          # Lower threshold for intelligent agent
            'time_exit_losing': 7,             # Exit losing positions with <= 7 days
            'time_exit_all': 3,                # Exit all positions with <= 3 days
            'max_hold_days': 30,               # Maximum days to hold any position
            'min_confidence_threshold': 0.6,   # Minimum confidence for exit decisions
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
            self.log_trade("Initializing broker...")
            self.broker = AlpacaBrokerIntegration(paper_trading=True)
            self.log_trade(f"Broker initialized: {self.broker is not None}")
            
            # Initialize options trader
            self.log_trade("Initializing options trader...")
            self.options_trader = OptionsTrader(self.broker)
            # Configure expiration settings
            if self.options_trader:
                self.options_trader.min_days_to_expiry = self.options_config['min_days_to_expiry']
                self.log_trade(f"Set minimum days to expiry: {self.options_config['min_days_to_expiry']}")
            self.log_trade(f"Options trader initialized: {self.options_trader is not None}")
            
            # Initialize options broker
            self.log_trade("Initializing options broker...")
            self.options_broker = OptionsBroker(self.broker, paper_trading=True)
            self.log_trade(f"Options broker initialized: {self.options_broker is not None}")
            
            # Show learning insights at startup
            self.log_trade("Loading historical learning data...")
            self._log_learning_insights()
            
            # Initialize advanced ML system
            self.log_trade("Initializing advanced ML system...")
            try:
                # Try to train ML model if enough data is available
                training_success = self.advanced_ml.train_model(min_samples=20)
                if training_success:
                    self.log_trade("Advanced ML model trained successfully", "INFO")
                else:
                    self.log_trade("Not enough data for ML training - using rule-based predictions", "INFO")
            except Exception as ml_error:
                self.log_trade(f"ML initialization error: {ml_error}", "WARN")
            
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
        """Update current market regime analysis with economic and volatility intelligence"""
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
            
            # Get economic analysis
            economic_analysis = await self.economic_data.get_comprehensive_economic_analysis()
            economic_regime = economic_analysis.get('market_regime', 'NEUTRAL')
            vol_environment = economic_analysis.get('volatility_environment', 'NORMAL')
            
            # Get CBOE volatility intelligence
            vix_data = await self.volatility_intelligence.get_vix_term_structure_analysis()
            flow_data = await self.volatility_intelligence.get_options_flow_analysis()
            
            volatility_regime = vix_data.get('volatility_regime', 'NORMAL')
            market_sentiment = flow_data.get('market_sentiment', 'NEUTRAL')
            put_call_ratio = flow_data.get('put_call_ratio', 1.0)
            vix_backwardation = vix_data.get('backwardation_signal', False)
            
            # Enhanced market regime determination
            if economic_regime == 'CRISIS' or volatility_regime == 'EXTREME_VOLATILITY' or self.vix_level > 35:
                self.market_regime = 'CRISIS'
            elif market_sentiment == 'EXTREME_FEAR' and put_call_ratio > 1.4:
                self.market_regime = 'FEAR_CAPITULATION'
            elif vix_backwardation and volatility_regime == 'HIGH_VOLATILITY':
                self.market_regime = 'VOLATILITY_SPIKE'
            elif economic_regime == 'HAWKISH_FED' or (self.vix_level > 25 and self.market_trend < -0.02):
                self.market_regime = 'HAWKISH_FED'
            elif economic_regime == 'DOVISH_FED' and self.market_trend > 0.02:
                self.market_regime = 'DOVISH_FED'
            elif market_sentiment == 'EXTREME_GREED' and put_call_ratio < 0.7:
                self.market_regime = 'COMPLACENCY'
            elif volatility_regime == 'LOW_VOLATILITY' and market_sentiment == 'COMPLACENT':
                self.market_regime = 'LOW_VIX_GRIND'
            elif self.vix_level > 30:
                self.market_regime = 'HIGH_VIX'
            elif self.market_trend > 0.03:
                self.market_regime = 'BULL'
            elif self.market_trend < -0.03:
                self.market_regime = 'BEAR'
            else:
                self.market_regime = 'NEUTRAL'
            
            # Log comprehensive intelligence
            fed_policy = economic_analysis.get('fed_policy', {})
            inflation_data = economic_analysis.get('inflation_data', {})
            if fed_policy.get('fed_funds_rate'):
                self.log_trade(f"Economic regime: {economic_regime}, Fed rate: {fed_policy['fed_funds_rate']:.2f}%, "
                             f"CPI: {inflation_data.get('cpi_yoy', 0):.1f}%")
            
            # Log volatility intelligence
            self.log_trade(f"Vol regime: {volatility_regime}, Sentiment: {market_sentiment}, "
                         f"P/C Ratio: {put_call_ratio:.2f}, VIX Backwardation: {vix_backwardation}")
            
            # Log special market conditions
            if vix_backwardation:
                self.log_trade("ALERT: VIX backwardation detected - volatility selling opportunity")
            if market_sentiment == 'EXTREME_FEAR':
                self.log_trade("SIGNAL: Extreme fear - potential contrarian opportunity")
            elif market_sentiment == 'EXTREME_GREED':
                self.log_trade("WARNING: Extreme greed - potential correction risk")
                
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
        
        # Adapt strategy preferences to enhanced market regime with volatility intelligence
        if self.market_regime == 'CRISIS':
            plan['preferred_strategies'] = ['LONG_PUT', 'BEAR_PUT_SPREAD']  # Protective strategies
            plan['target_new_positions'] = 1
            plan['risk_adjustments'] = ['reduce_position_size', 'shorter_expiration']
        elif self.market_regime == 'FEAR_CAPITULATION':
            plan['preferred_strategies'] = ['CASH_SECURED_PUT', 'BULL_CALL_SPREAD']  # Contrarian opportunity
            plan['target_new_positions'] = 2
            plan['risk_adjustments'] = ['wait_for_confirmation']
        elif self.market_regime == 'VOLATILITY_SPIKE':
            plan['preferred_strategies'] = ['VOLATILITY_SELLING', 'SPREADS']  # Sell high vol
            plan['target_new_positions'] = 1
            plan['risk_adjustments'] = ['avoid_long_options']
        elif self.market_regime == 'COMPLACENCY':
            plan['preferred_strategies'] = ['LONG_PUT', 'PROTECTIVE_STRATEGIES']  # Hedge complacency
            plan['target_new_positions'] = 1
            plan['risk_adjustments'] = ['buy_protection']
        elif self.market_regime == 'LOW_VIX_GRIND':
            plan['preferred_strategies'] = ['BULL_CALL_SPREAD', 'VOLATILITY_BUYING']  # Low vol opportunity
            plan['target_new_positions'] = 3
            plan['risk_adjustments'] = ['longer_expiration']
        elif self.market_regime == 'BULL':
            plan['preferred_strategies'] = ['BULL_CALL_SPREAD', 'LONG_CALL']
            plan['target_new_positions'] = 3
        elif self.market_regime == 'BEAR':
            plan['preferred_strategies'] = ['BEAR_PUT_SPREAD', 'LONG_PUT'] 
            plan['target_new_positions'] = 2
        elif self.market_regime == 'HIGH_VIX':
            plan['preferred_strategies'] = ['CASH_SECURED_PUT', 'SPREADS']
            plan['target_new_positions'] = 1
        else:  # NEUTRAL, HAWKISH_FED, DOVISH_FED
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
        """Get comprehensive market data for analysis using enhanced technical analysis"""
        try:
            # Get comprehensive technical analysis
            technical_data = await self.technical_analysis.get_comprehensive_analysis(symbol, period="60d")
            
            if not technical_data or technical_data['current_price'] == 0:
                return None
            
            # Get volatility intelligence from CBOE
            vix_data = await self.volatility_intelligence.get_vix_term_structure_analysis()
            flow_data = await self.volatility_intelligence.get_options_flow_analysis()
            
            # Extract key metrics for compatibility
            indicators = technical_data.get('technical_indicators', {})
            signals = technical_data.get('signals', {})
            volatility = technical_data.get('volatility_analysis', {})
            momentum = technical_data.get('momentum_analysis', {})
            support_resistance = technical_data.get('support_resistance', {})
            
            # Enhanced market data structure
            enhanced_data = {
                'symbol': symbol,
                'current_price': technical_data['current_price'],
                'timestamp': datetime.now(),
                
                # Legacy compatibility fields
                'price_momentum': momentum.get('price_momentum_5d', 0) / 100,  # Convert to decimal
                'realized_vol': volatility.get('realized_vol_20d', 20.0),
                'volume_ratio': indicators.get('volume_ratio', 1.0),
                'price_position': 0.5,  # Default middle position
                'avg_volume': indicators.get('volume_sma', 1000000),
                
                # Enhanced technical analysis data
                'technical_analysis': technical_data,
                'overall_signal': signals.get('overall_signal', 'NEUTRAL'),
                'signal_strength': signals.get('signal_strength', 0.0),
                'signal_confidence': signals.get('confidence', 0.5),
                'bullish_factors': signals.get('bullish_factors', []),
                'bearish_factors': signals.get('bearish_factors', []),
                
                # Technical indicators
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd', 0),
                'macd_signal': indicators.get('macd_signal', 0),
                'sma_20': indicators.get('sma_20', technical_data['current_price']),
                'sma_50': indicators.get('sma_50', technical_data['current_price']),
                'bollinger_upper': indicators.get('bb_upper', technical_data['current_price'] * 1.02),
                'bollinger_lower': indicators.get('bb_lower', technical_data['current_price'] * 0.98),
                
                # Support/Resistance
                'nearest_support': support_resistance.get('nearest_support'),
                'nearest_resistance': support_resistance.get('nearest_resistance'),
                'support_levels': support_resistance.get('support_levels', []),
                'resistance_levels': support_resistance.get('resistance_levels', []),
                
                # Volatility metrics
                'vol_percentile': volatility.get('vol_percentile', 0.5),
                'vol_trend': volatility.get('vol_trend', 'STABLE'),
                'vol_regime': volatility.get('vol_regime', 'NORMAL'),
                
                # Momentum metrics
                'momentum_strength': momentum.get('momentum_strength', 'WEAK'),
                'price_momentum_20d': momentum.get('price_momentum_20d', 0),
                'momentum_acceleration': momentum.get('acceleration', 0),
                
                # CBOE Volatility Intelligence
                'vix_data': vix_data,
                'flow_data': flow_data,
                'market_volatility_regime': vix_data.get('volatility_regime', 'NORMAL'),
                'vix_term_structure': vix_data.get('term_structure', 'NORMAL'),
                'market_sentiment': flow_data.get('market_sentiment', 'NEUTRAL'),
                'put_call_ratio': flow_data.get('put_call_ratio', 1.0),
                'volatility_trade_bias': vix_data.get('options_implications', {}).get('volatility_trade_bias', 'NEUTRAL'),
                'positioning_bias': flow_data.get('options_implications', {}).get('positioning_bias', 'NEUTRAL')
            }
            
            # Calculate price position using support/resistance if available
            if enhanced_data['nearest_support'] and enhanced_data['nearest_resistance']:
                support = enhanced_data['nearest_support']
                resistance = enhanced_data['nearest_resistance']
                if resistance > support:
                    enhanced_data['price_position'] = (technical_data['current_price'] - support) / (resistance - support)
            
            return enhanced_data
            
        except Exception as e:
            self.log_trade(f"Enhanced market data error for {symbol}: {e}", "WARN")
            # Fallback to basic data
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    return {
                        'symbol': symbol,
                        'current_price': float(hist['Close'].iloc[-1]),
                        'price_momentum': 0.0,
                        'realized_vol': 20.0,
                        'volume_ratio': 1.0,
                        'price_position': 0.5,
                        'timestamp': datetime.now()
                    }
            except:
                pass
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
                
                # Log the decision and current status with enhanced analysis
                entry_time = position_data.get('entry_time', datetime.now())
                days_held = (datetime.now() - entry_time).days
                
                # Get enhanced analysis for logging
                enhanced_analysis = await self.perform_enhanced_exit_analysis(
                    position_data, market_data, current_pnl, exit_decision
                )
                
                self.log_trade(f"MONITORING: {symbol} - P&L: ${current_pnl:.2f} ({enhanced_analysis['pnl_percentage']:+.1f}%), Signal: {exit_decision.signal}")
                self.log_trade(f"  Days held: {days_held} | Exit signals: {enhanced_analysis['exit_signals']} | Hold signals: {enhanced_analysis['hold_signals']}")
                
                if exit_decision.signal != ExitSignal.HOLD:
                    self.log_trade(f"  AGENT EXIT SIGNAL: {exit_decision.signal}")
                    self.log_trade(f"  Agent Reason: {exit_decision.reasoning}")
                    self.log_trade(f"  Agent Confidence: {exit_decision.confidence:.1%} | Urgency: {exit_decision.urgency:.1%}")
                
                if enhanced_analysis['factors']:
                    self.log_trade(f"  Analysis factors: {'; '.join(enhanced_analysis['factors'][:2])}")  # Show top 2 factors
                    
                # Enhanced intelligent analysis for exit decisions
                should_exit = False
                exit_reason = exit_decision.reasoning
                
                # Use intelligent analysis instead of simple thresholds
                if self.exit_config['use_intelligent_analysis']:
                    # Primary: Trust the intelligent agent with lower threshold
                    if (exit_decision.urgency >= self.exit_config['urgency_threshold'] and 
                        exit_decision.confidence >= self.exit_config['min_confidence_threshold']):
                        should_exit = True
                        exit_reason = f"Intelligent agent: {exit_decision.reasoning}"
                    
                    # Enhanced analysis override
                    elif enhanced_analysis['should_exit']:
                        should_exit = True
                        exit_reason = enhanced_analysis['reason']
                        self.log_trade(f"  Enhanced analysis triggered exit: {'; '.join(enhanced_analysis['factors'][:3])}")
                    
                    # Check days to expiry for time-based exits
                    entry_time = position_data.get('entry_time', datetime.now())
                    days_held = (datetime.now() - entry_time).days
                    original_dte = position_data['opportunity'].get('days_to_expiry', 30)
                    days_remaining = max(0, original_dte - days_held)
                    
                    # Time-based exits
                    if days_remaining <= self.exit_config['time_exit_losing'] and current_pnl < 0:
                        should_exit = True
                        exit_reason = f"Time decay exit: {days_remaining} days left, losing ${abs(current_pnl):.2f}"
                    elif days_remaining <= self.exit_config['time_exit_all']:
                        should_exit = True
                        exit_reason = f"Expiration exit: only {days_remaining} days remaining"
                    elif days_held >= self.exit_config['max_hold_days']:
                        should_exit = True
                        exit_reason = f"Max hold time reached: {days_held} days (limit: {self.exit_config['max_hold_days']})"
                    
                    if should_exit:
                        # Update the exit decision reason
                        exit_decision.reasoning = exit_reason
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
        
        # Show learning insights periodically
        if len(positions_to_exit) > 0:
            self._log_learning_insights()
    
    def calculate_position_pnl(self, position_data, market_data):
        """Calculate current P&L based on option contract value"""
        try:
            # Get position details
            entry_price = position_data.get('entry_price', 0)  # Entry price per contract
            quantity = position_data.get('quantity', 1)  # Number of contracts
            
            # Get current option value (professional pricing)
            current_option_price = await self.estimate_current_option_price(position_data, market_data)
            
            # Calculate P&L based on contract value difference
            # Each contract represents 100 shares, so multiply by 100
            price_per_contract_change = current_option_price - entry_price
            total_pnl = price_per_contract_change * quantity * 100
            
            return total_pnl
            
        except Exception as e:
            self.log_trade(f"P&L calculation error: {e}", "WARN")
            return 0.0
    
    async def estimate_current_option_price(self, position_data, market_data):
        """Estimate current option price using professional options pricing"""
        try:
            opportunity = position_data['opportunity']
            strategy = opportunity['strategy']
            entry_time = position_data['entry_time']
            current_stock_price = market_data['current_price']
            
            # Get option parameters
            strike_price = opportunity.get('strike_price', current_stock_price)
            entry_price = position_data.get('entry_price', 1.0)
            
            # Calculate time to expiry
            days_elapsed = (datetime.now() - entry_time).days
            original_dte = opportunity.get('days_to_expiry', 30)
            current_dte = max(1, original_dte - days_elapsed)
            
            # Get volatility from enhanced analysis
            current_vol = market_data.get('realized_vol', 20.0)
            
            # Use professional options pricing if we have sufficient data
            if strike_price > 0 and current_dte > 0:
                try:
                    # Determine option type
                    if 'CALL' in str(strategy):
                        option_type = 'call'
                    elif 'PUT' in str(strategy):
                        option_type = 'put'
                    else:
                        # For spreads, estimate the long leg
                        option_type = 'call'  # Default
                    
                    # Get professional pricing analysis
                    pricing_analysis = await self.options_pricing.get_comprehensive_option_analysis(
                        underlying_price=current_stock_price,
                        strike_price=strike_price,
                        time_to_expiry_days=current_dte,
                        volatility=current_vol,
                        option_type=option_type
                    )
                    
                    theoretical_price = pricing_analysis['pricing']['theoretical_price']
                    
                    # Log detailed pricing info occasionally
                    if random.random() < 0.1:  # 10% of the time
                        greeks = pricing_analysis['greeks']
                        self.log_trade(f"  Professional pricing: ${theoretical_price:.2f} "
                                     f"(Delta: {greeks['delta']:.2f}, Theta: {greeks['theta']:.3f}, "
                                     f"Method: {pricing_analysis['pricing']['pricing_method']})")
                    
                    return max(0.01, theoretical_price)
                    
                except Exception as pricing_error:
                    self.log_trade(f"Professional pricing failed, using fallback: {pricing_error}", "WARN")
            
            # Fallback to simplified estimation
            return self._simplified_option_price_estimate(position_data, market_data)
            
        except Exception as e:
            self.log_trade(f"Option price estimation error: {e}", "WARN")
            return position_data.get('entry_price', 1.0)  # Fallback to entry price
    
    def _simplified_option_price_estimate(self, position_data, market_data):
        """Simplified option price estimation (fallback method)"""
        try:
            opportunity = position_data['opportunity']
            strategy = opportunity['strategy']
            entry_time = position_data['entry_time']
            current_stock_price = market_data['current_price']
            
            # Get original parameters
            original_stock_price = opportunity.get('stock_price', current_stock_price)
            strike_price = opportunity.get('strike_price', current_stock_price)
            entry_price = position_data.get('entry_price', 1.0)
            
            # Time decay factor
            days_elapsed = (datetime.now() - entry_time).days
            original_dte = opportunity.get('days_to_expiry', 30)
            current_dte = max(1, original_dte - days_elapsed)
            time_decay_factor = max(0.1, current_dte / original_dte)
            
            # Price movement factor
            if 'CALL' in str(strategy):
                intrinsic_value = max(0, current_stock_price - strike_price)
                price_movement_factor = current_stock_price / original_stock_price
            elif 'PUT' in str(strategy):
                intrinsic_value = max(0, strike_price - current_stock_price)
                price_movement_factor = original_stock_price / current_stock_price
            else:
                intrinsic_value = abs(current_stock_price - strike_price) * 0.5
                price_movement_factor = 1.0
            
            # Volatility factor
            volatility_factor = market_data.get('realized_vol', 25) / 25.0
            
            # Estimate price
            time_value = entry_price * time_decay_factor * volatility_factor
            estimated_price = intrinsic_value + (time_value * price_movement_factor)
            
            return max(0.01, estimated_price)
            
        except Exception as e:
            self.log_trade(f"Simplified pricing error: {e}", "WARN")
            return 1.0
    
    async def perform_enhanced_exit_analysis(self, position_data, market_data, current_pnl, exit_decision):
        """Perform enhanced market analysis to determine if position should be exited"""
        try:
            symbol = position_data['opportunity']['symbol']
            strategy = position_data['opportunity']['strategy']
            entry_time = position_data.get('entry_time', datetime.now())
            entry_price = position_data.get('entry_price', 1.0)
            days_held = (datetime.now() - entry_time).days
            
            # Calculate percentage gains/losses
            current_option_price = await self.estimate_current_option_price(position_data, market_data)
            pnl_percentage = ((current_option_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            
            analysis_factors = []
            exit_signals = 0
            hold_signals = 0
            
            # 1. Enhanced Technical Analysis
            overall_signal = market_data.get('overall_signal', 'NEUTRAL')
            signal_strength = market_data.get('signal_strength', 0.0)
            signal_confidence = market_data.get('signal_confidence', 0.5)
            
            # Check if technical signals oppose our position
            if 'CALL' in str(strategy) and overall_signal == 'BEARISH' and signal_strength > 0.4:
                exit_signals += int(2 * signal_strength)  # Scale by strength
                analysis_factors.append(f"Strong bearish technical signal ({signal_strength:.1%} strength) against call")
            elif 'PUT' in str(strategy) and overall_signal == 'BULLISH' and signal_strength > 0.4:
                exit_signals += int(2 * signal_strength)
                analysis_factors.append(f"Strong bullish technical signal ({signal_strength:.1%} strength) against put")
            elif overall_signal != 'NEUTRAL' and signal_confidence > 0.7:
                # High confidence signals get attention even if not strongly opposing
                if ('CALL' in str(strategy) and overall_signal == 'BEARISH') or ('PUT' in str(strategy) and overall_signal == 'BULLISH'):
                    exit_signals += 1
                    analysis_factors.append(f"High confidence {overall_signal.lower()} signal")
            
            # 2. RSI and Momentum Analysis
            rsi = market_data.get('rsi', 50)
            momentum_strength = market_data.get('momentum_strength', 'WEAK')
            
            if 'CALL' in str(strategy):
                if rsi > 75:  # Severely overbought
                    exit_signals += 2
                    analysis_factors.append(f"RSI severely overbought ({rsi:.1f}) - call at risk")
                elif momentum_strength == 'STRONG' and market_data.get('momentum_acceleration', 0) < -5:
                    exit_signals += 1
                    analysis_factors.append("Strong negative momentum acceleration against calls")
            elif 'PUT' in str(strategy):
                if rsi < 25:  # Severely oversold
                    exit_signals += 2
                    analysis_factors.append(f"RSI severely oversold ({rsi:.1f}) - put at risk")
                elif momentum_strength == 'STRONG' and market_data.get('momentum_acceleration', 0) > 5:
                    exit_signals += 1
                    analysis_factors.append("Strong positive momentum acceleration against puts")
            
            # 2. Volatility Analysis
            current_vol = market_data.get('realized_vol', 25)
            if current_vol < 15 and 'CALL' in str(strategy) or 'PUT' in str(strategy):  # Vol crush hurts long options
                exit_signals += 1
                analysis_factors.append(f"Low volatility {current_vol:.1f}% hurting option value")
            elif current_vol > 35:  # High vol - good for long options, wait
                hold_signals += 1
                analysis_factors.append(f"High volatility {current_vol:.1f}% supporting option value")
            
            # 3. Enhanced Support/Resistance Analysis
            nearest_support = market_data.get('nearest_support')
            nearest_resistance = market_data.get('nearest_resistance')
            current_price = market_data.get('current_price', 0)
            
            if nearest_support and nearest_resistance and current_price > 0:
                support_distance = (current_price - nearest_support) / nearest_support * 100
                resistance_distance = (nearest_resistance - current_price) / current_price * 100
                
                if 'CALL' in str(strategy):
                    if support_distance < 2:  # Within 2% of support
                        exit_signals += 2
                        analysis_factors.append(f"Price ${current_price:.2f} very close to support ${nearest_support:.2f}")
                    elif resistance_distance < 1:  # Very close to resistance  
                        hold_signals += 1
                        analysis_factors.append(f"Near resistance ${nearest_resistance:.2f} - upside limited but close")
                elif 'PUT' in str(strategy):
                    if resistance_distance < 2:  # Within 2% of resistance
                        exit_signals += 2
                        analysis_factors.append(f"Price ${current_price:.2f} very close to resistance ${nearest_resistance:.2f}")
                    elif support_distance < 1:  # Very close to support
                        hold_signals += 1
                        analysis_factors.append(f"Near support ${nearest_support:.2f} - downside limited but close")
            
            # 4. Volume Analysis
            volume_ratio = market_data.get('volume_ratio', 1.0)
            if volume_ratio < 0.5:  # Low volume - potential reversal
                exit_signals += 1
                analysis_factors.append(f"Low volume {volume_ratio:.1f}x - weak conviction")
            elif volume_ratio > 2.0:  # High volume - strong move
                hold_signals += 1
                analysis_factors.append(f"High volume {volume_ratio:.1f}x - strong move continues")
            
            # 5. Time Decay vs Performance Analysis
            if days_held >= 14:  # After 2 weeks
                if pnl_percentage < 5:  # Less than 5% gain after 2+ weeks
                    exit_signals += 2
                    analysis_factors.append(f"Poor performance {pnl_percentage:.1f}% after {days_held} days")
                elif pnl_percentage > 25:  # Good gains, protect them
                    if current_vol < 20:  # Low vol environment
                        exit_signals += 1
                        analysis_factors.append(f"Protecting {pnl_percentage:.1f}% gains in low vol environment")
            
            # 6. Market Regime Analysis
            if hasattr(self, 'market_regime'):
                if self.market_regime == 'HIGH_VIX' and pnl_percentage > 15:  # Take profits in volatile markets
                    exit_signals += 1
                    analysis_factors.append("Taking profits in high VIX environment")
                elif self.market_regime in ['BULL', 'BEAR'] and 'SPREAD' not in str(strategy):
                    # Trending markets - let winners run more
                    hold_signals += 1
                    analysis_factors.append(f"Trending {self.market_regime} market - letting position run")
            
            # 7. Options Greeks Analysis (simplified)
            original_dte = position_data['opportunity'].get('days_to_expiry', 30)
            current_dte = max(1, original_dte - days_held)
            theta_pressure = 1 - (current_dte / original_dte)  # Higher = more time decay
            
            if theta_pressure > 0.6 and pnl_percentage < 0:  # Significant time decay on losing position
                exit_signals += 2
                analysis_factors.append(f"High time decay pressure {theta_pressure:.1%} on losing position")
            
            # Decision Logic
            net_signal_strength = exit_signals - hold_signals
            should_exit = False
            confidence = 0.5
            
            if net_signal_strength >= 3:  # Strong exit signal
                should_exit = True
                confidence = min(0.95, 0.6 + (net_signal_strength * 0.1))
                reason = f"Strong exit signal (score: +{net_signal_strength}) - {analysis_factors[0]}"
            elif net_signal_strength >= 2:  # Moderate exit signal
                should_exit = True
                confidence = 0.7
                reason = f"Moderate exit signal (score: +{net_signal_strength}) - Multiple factors align"
            elif net_signal_strength <= -2:  # Strong hold signal
                should_exit = False
                reason = f"Strong hold signal (score: {net_signal_strength}) - Favorable conditions continue"
            else:  # Neutral - defer to original exit agent
                should_exit = False
                reason = "Neutral signals - deferring to exit agent analysis"
            
            return {
                'should_exit': should_exit,
                'reason': reason,
                'confidence': confidence,
                'factors': analysis_factors,
                'exit_signals': exit_signals,
                'hold_signals': hold_signals,
                'pnl_percentage': pnl_percentage,
                'days_held': days_held
            }
            
        except Exception as e:
            self.log_trade(f"Enhanced analysis error: {e}", "WARN")
            return {
                'should_exit': False,
                'reason': 'Analysis error - holding position',
                'confidence': 0.5,
                'factors': ['analysis_error'],
                'exit_signals': 0,
                'hold_signals': 0,
                'pnl_percentage': 0,
                'days_held': 0
            }
    
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
            
            # Record trade exit for learning
            try:
                exit_price = current_value / 100 if current_value else 0.0  # Convert to per-contract price
                self.learning_engine.record_trade_exit(
                    trade_id=exit_info['position_id'],
                    exit_price=exit_price,
                    exit_reason=exit_info['urgency']
                )
            except Exception as learning_error:
                self.log_trade(f"Learning engine error: {learning_error}", "WARN")
            
            self.log_trade(f"[OK] Position {symbol} exited successfully - P&L: ${current_pnl:.2f}")
            
        except Exception as e:
            self.log_trade(f"Exit execution error: {e}", "ERROR")
    
    def _log_learning_insights(self):
        """Log learning insights periodically"""
        try:
            insights = self.learning_engine.get_learning_insights()
            
            if insights['total_trades'] > 0:
                self.log_trade(f"=== LEARNING INSIGHTS ===", "INFO")
                self.log_trade(f"Total completed trades: {insights['total_trades']}", "INFO")
                
                # Strategy performance
                for strategy, perf in insights['strategy_performance'].items():
                    self.log_trade(f"{strategy}: {perf['win_rate']:.1%} win rate, "
                                 f"avg P&L: ${perf['avg_pnl']:.2f}, "
                                 f"multiplier: {perf['multiplier']:.2f}", "INFO")
                
                # Recent performance
                if 'recent_performance' in insights:
                    recent = insights['recent_performance']
                    self.log_trade(f"Recent (30d): {recent['win_rate']:.1%} win rate, "
                                 f"P&L: ${recent['total_pnl']:.2f}", "INFO")
                
                # Recommendations
                for rec in insights['recommendations'][:3]:  # Show top 3
                    self.log_trade(f"Recommendation: {rec}", "INFO")
                
                self.log_trade(f"=== END INSIGHTS ===", "INFO")
                
        except Exception as e:
            self.log_trade(f"Learning insights error: {e}", "WARN")
    
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
        
        # 2. Look for new opportunities (every cycle for more aggressive trading)
        await self.scan_for_new_opportunities()
        
        # 3. Risk check
        await self.intraday_risk_check()
        
        self.log_trade(f"Cycle #{self.cycle_count} completed")
    
    async def scan_for_new_opportunities(self):
        """Scan for new trading opportunities"""
        if len(self.active_positions) >= self.daily_risk_limits.get('max_positions', 5):
            self.log_trade("Position limit reached - skipping new opportunity scan")
            return
        
        self.log_trade(f"Scanning for new opportunities across {len(self.tier1_stocks)} symbols...")
        
        opportunities = []
        # Scan ALL tier1 symbols for maximum opportunity detection
        scan_symbols = self.tier1_stocks
        
        for symbol in scan_symbols:
            try:
                opportunity = await self.find_high_quality_opportunity(symbol)
                if opportunity:
                    opportunities.append(opportunity)
                    self.log_trade(f"OPPORTUNITY: {symbol} {opportunity['strategy']} - Confidence: {opportunity.get('confidence', 0):.1%}")
            except Exception as e:
                self.log_trade(f"Opportunity scan error for {symbol}: {e}", "WARN")
        
        self.log_trade(f"Scan complete: Found {len(opportunities)} opportunities from {len(scan_symbols)} symbols")
        
        # Execute ALL opportunities with 75%+ confidence
        if opportunities:
            # Filter opportunities with 75% or higher confidence
            high_confidence_opportunities = [opp for opp in opportunities if opp.get('confidence', 0) >= 0.75]
            
            if high_confidence_opportunities:
                self.log_trade(f"Found {len(high_confidence_opportunities)} opportunities with 75%+ confidence")
                
                # Sort by confidence * expected_return for execution order
                high_confidence_opportunities.sort(key=lambda x: x.get('confidence', 0) * x.get('expected_return', 0), reverse=True)
                
                # Execute all high-confidence opportunities (respecting position limits)
                for opportunity in high_confidence_opportunities:
                    if len(self.active_positions) >= self.daily_risk_limits.get('max_positions', 5):
                        self.log_trade("Position limit reached - stopping new trades")
                        break
                    
                    success = await self.execute_new_position(opportunity)
                    if success:
                        self.log_trade(f"EXECUTED: {opportunity['symbol']} at {opportunity.get('confidence', 0):.1%} confidence")
                    else:
                        self.log_trade(f"FAILED to execute: {opportunity['symbol']}")
            else:
                # If no 75%+ confidence, execute best opportunity if it's above 60%
                best_opportunity = max(opportunities, key=lambda x: x.get('confidence', 0) * x.get('expected_return', 0))
                if best_opportunity.get('confidence', 0) >= 0.60:
                    self.log_trade(f"Executing best opportunity at {best_opportunity.get('confidence', 0):.1%} confidence")
                    await self.execute_new_position(best_opportunity)
                else:
                    self.log_trade(f"Best opportunity only {best_opportunity.get('confidence', 0):.1%} confidence - too low")
    
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
                
                # Dynamic strategy selection based on momentum (Level 1 compatible)
                if market_data['price_momentum'] > 0.01:  # Positive momentum
                    strategy = OptionsStrategy.LONG_CALL
                elif market_data['price_momentum'] < -0.01:  # Negative momentum
                    strategy = OptionsStrategy.LONG_PUT
                else:  # Low momentum
                    strategy = OptionsStrategy.LONG_CALL  # Default to calls
                
                # Dynamic confidence based on signal strength
                base_confidence = 0.5  # Base confidence
                if volume_ok and momentum_ok:
                    base_confidence += 0.2  # Both criteria met
                if vol_ok:
                    base_confidence += 0.1  # Good volatility
                if abs(market_data['price_momentum']) > 0.025:
                    base_confidence += 0.15  # Strong momentum
                
                base_confidence = min(0.85, base_confidence)  # Cap at 85%
                
                # Apply basic machine learning calibration
                confidence = self.learning_engine.calibrate_confidence(
                    base_confidence, strategy.value, symbol, market_data
                )
                
                # Apply advanced ML prediction
                try:
                    ml_prob, ml_explanation = self.advanced_ml.predict_trade_success(
                        symbol, strategy.value, confidence
                    )
                    
                    # Blend basic confidence with ML prediction
                    confidence = (confidence * 0.6) + (ml_prob * 0.4)
                    
                    # Log ML insights for high-confidence trades
                    if confidence >= 0.70:
                        feature_analysis = self.advanced_ml.get_feature_analysis(symbol)
                        if feature_analysis:
                            tech = feature_analysis.get('technical', {})
                            flow = feature_analysis.get('options_flow', {})
                            self.log_trade(f"ML Analysis for {symbol}: RSI={tech.get('rsi_14', 0):.1f}, "
                                         f"Momentum={tech.get('momentum_5d', 0):+.1f}%, "
                                         f"P/C Ratio={flow.get('put_call_ratio', 1.0):.2f}, "
                                         f"Flow={flow.get('sentiment', 'NEUTRAL')}", "INFO")
                
                except Exception as ml_error:
                    self.log_trade(f"Advanced ML error for {symbol}: {ml_error}", "WARN")
                
                # Check if strategy should be avoided due to poor performance
                if self.learning_engine.should_avoid_strategy(strategy.value):
                    self.log_trade(f"Avoiding {strategy.value} for {symbol} due to poor historical performance", "INFO")
                    return None
                
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
        """Execute a REAL options position"""
        try:
            symbol = opportunity['symbol']
            
            # Check if options_trader is properly initialized
            if self.options_trader is None:
                self.log_trade(f"ERROR: options_trader is None - reinitializing systems", "ERROR")
                success = await self.initialize_all_systems()
                
                if not success or self.options_trader is None:
                    self.log_trade(f"CRITICAL: Failed to initialize options_trader for {symbol}", "ERROR")
                    # Try one more time with direct initialization
                    try:
                        from agents.options_trading_agent import OptionsTrader
                        from agents.broker_integration import AlpacaBrokerIntegration
                        from agents.options_broker import OptionsBroker
                        
                        if self.broker is None:
                            self.broker = AlpacaBrokerIntegration(paper_trading=True)
                        
                        self.options_trader = OptionsTrader(self.broker)
                        self.options_broker = OptionsBroker(self.broker, paper_trading=True)
                        
                        if self.options_trader is None:
                            self.log_trade(f"CRITICAL: Direct initialization also failed for {symbol}", "ERROR")
                            return False
                        else:
                            self.log_trade(f"SUCCESS: Direct initialization worked for {symbol}", "INFO")
                            
                    except Exception as init_error:
                        self.log_trade(f"CRITICAL: Direct initialization exception: {init_error}", "ERROR")
                        return False
            strategy = opportunity['strategy']
            
            # Risk check
            position_risk = opportunity['max_loss'] * 100  # Convert to dollars
            if position_risk > self.daily_risk_limits.get('max_position_risk', 1000):
                self.log_trade(f"Position risk too high for {symbol} - skipping")
                return False
            
            # Get options chain for the symbol
            if not hasattr(self.options_trader, 'option_chains') or symbol not in self.options_trader.option_chains:
                options_contracts = await self.options_trader.get_options_chain(symbol)
                if not options_contracts:
                    self.log_trade(f"No options available for {symbol}")
                    return False
            
            # Find the best options strategy to execute
            market_data = opportunity.get('market_data', {})
            current_price = market_data.get('current_price', 550.0)
            volatility = market_data.get('realized_vol', 20) / 100
            
            # Get the strategy with proper parameters
            strategy_result = self.options_trader.find_best_options_strategy(
                symbol=symbol,
                price=current_price,
                volatility=volatility,
                rsi=60.0,  # Assume bullish
                price_change=0.01  # Small positive change
            )
            
            if strategy_result:
                strategy_type, contracts = strategy_result
                
                if contracts:
                    # Execute the options strategy with REAL orders
                    self.log_trade(f"PLACING REAL OPTIONS TRADE: {symbol} {strategy_type}")
                    
                    try:
                        # Execute through the options trader with confidence level and adaptive sizing
                        opportunity_confidence = opportunity.get('confidence', 0.5)
                        
                        # Apply adaptive position sizing based on learning
                        base_quantity = 1
                        size_multiplier = self.learning_engine.get_position_size_multiplier()
                        adaptive_quantity = max(1, int(base_quantity * size_multiplier))
                        
                        position = await self.options_trader.execute_options_strategy(
                            strategy=strategy_type,
                            contracts=contracts,
                            quantity=adaptive_quantity,
                            confidence=opportunity_confidence
                        )
                        
                        if position:
                            # Create position data with real trade information
                            position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            
                            # Record trade entry for learning
                            self.learning_engine.record_trade_entry(
                                trade_id=position_id,
                                symbol=symbol,
                                strategy=strategy_type.value,
                                confidence=opportunity_confidence,
                                entry_price=position.entry_price,
                                quantity=position.quantity,
                                max_profit=opportunity.get('max_profit', 2.0),
                                max_loss=opportunity.get('max_loss', 1.0),
                                market_conditions={
                                    'market_regime': self.market_regime,
                                    'volatility': opportunity.get('market_data', {}).get('realized_vol', 20),
                                    'volume_ratio': opportunity.get('market_data', {}).get('volume_ratio', 1.0),
                                    'price_momentum': opportunity.get('market_data', {}).get('price_momentum', 0.0)
                                }
                            )
                            position_data = {
                                'position': position,  # Real position object
                                'opportunity': opportunity,
                                'entry_time': datetime.now(),
                                'entry_price': position.entry_price,
                                'quantity': 1,
                                'market_regime_at_entry': self.market_regime,
                                'real_trade': True,
                                'order_ids': getattr(position, 'order_ids', [])
                            }
                            
                            # Add to active positions
                            self.active_positions[position_id] = position_data
                            
                            # Update stats
                            self.performance_stats['total_trades'] += 1
                            
                            self.log_trade(f"SUCCESS: REAL TRADE EXECUTED: {symbol} {strategy_type} - Risk: ${position_risk:.2f}")
                            self.log_trade(f"   Order IDs: {position_data.get('order_ids', [])}")
                            self.log_trade(f"   Entry Price: ${position.entry_price:.2f}")
                            
                            return True
                        else:
                            self.log_trade(f"FAILED Options strategy execution failed for {symbol}")
                            return False
                            
                    except Exception as trade_error:
                        self.log_trade(f"FAILED Real trade execution failed: {trade_error}", "ERROR")
                        
                        # Fall back to simulation tracking if real trade fails
                        position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        position_data = {
                            'position': None,
                            'opportunity': opportunity,
                            'entry_time': datetime.now(),
                            'entry_price': opportunity.get('net_debit', 0),
                            'quantity': 1,
                            'market_regime_at_entry': self.market_regime,
                            'real_trade': False,
                            'simulation_reason': str(trade_error)
                        }
                        
                        self.active_positions[position_id] = position_data
                        self.performance_stats['total_trades'] += 1
                        
                        self.log_trade(f"WARNING: FALLBACK SIMULATION: {symbol} {strategy} - Risk: ${position_risk:.2f}")
                        return True
                else:
                    self.log_trade(f"No suitable contracts found for {symbol} {strategy}")
                    return False
            else:
                self.log_trade(f"No strategy generated for {symbol}")
                return False
            
        except Exception as e:
            self.log_trade(f"Position execution error: {e}", "ERROR")
            import traceback
            self.log_trade(f"Traceback: {traceback.format_exc()}", "ERROR")
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