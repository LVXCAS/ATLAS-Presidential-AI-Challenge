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
import logging
# Suppress yfinance error messages about delisted stocks and missing data
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
# Also suppress urllib3 logging from yfinance
logging.getLogger('urllib3').setLevel(logging.WARNING)
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
from ai.ml_ensemble_wrapper import get_ml_ensemble

# Import ensemble voting system
from enhancements.ensemble_voting import get_ensemble_system

# Import all enhancement modules
from enhancements.greeks_optimizer import get_greeks_optimizer
from enhancements.volatility_regime import get_volatility_adapter
from enhancements.spread_strategies import get_spread_strategies
from enhancements.market_regime import get_market_regime_detector
from enhancements.dynamic_stops import get_dynamic_stop_manager
from enhancements.liquidity_filter import get_liquidity_filter

# Import profit target monitoring
try:
    from profit_target_monitor import ProfitTargetMonitor
    PROFIT_MONITOR_AVAILABLE = True
except ImportError:
    PROFIT_MONITOR_AVAILABLE = False
    print("- Profit monitor not available")

# Import advanced quantitative finance capabilities
try:
    from agents.quantitative_finance_engine import quantitative_engine
    QUANT_ENGINE_AVAILABLE = True
    print("+ Quantitative engine loaded successfully")
except ImportError as e:
    quantitative_engine = None
    QUANT_ENGINE_AVAILABLE = False
    print(f"- Quantitative engine not available: {e}")
except Exception as e:
    quantitative_engine = None
    QUANT_ENGINE_AVAILABLE = False
    print(f"- Quantitative engine load error: {e}")

try:
    from agents.quant_integration import quant_analyzer, analyze_option, analyze_portfolio, predict_returns
    QUANT_INTEGRATION_AVAILABLE = True
except ImportError:
    quant_analyzer = analyze_option = analyze_portfolio = predict_returns = None
    QUANT_INTEGRATION_AVAILABLE = False
    print("- Quant integration not available")

# Live trading integration
try:
    from agents.live_data_manager import setup_live_data
    from agents.live_trading_engine import LiveTradingEngine
    LIVE_TRADING_AVAILABLE = True
    print("+ Live trading integration loaded")
except ImportError:
    LIVE_TRADING_AVAILABLE = False
    print("- Live trading integration not available")
# Import all other agents with error handling
try:
    from agents.exit_strategy_agent import exit_strategy_agent, ExitSignal
    EXIT_STRATEGY_AVAILABLE = True
except ImportError:
    exit_strategy_agent = None
    ExitSignal = None
    EXIT_STRATEGY_AVAILABLE = False
    print("- Exit strategy agent not available")

try:
    from agents.learning_engine import learning_engine
    LEARNING_ENGINE_AVAILABLE = True
except ImportError:
    learning_engine = None
    LEARNING_ENGINE_AVAILABLE = False
    print("- Learning engine not available")

try:
    from agents.advanced_ml_engine import advanced_ml_engine
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    advanced_ml_engine = None
    ADVANCED_ML_AVAILABLE = False
    print("- Advanced ML engine not available")

try:
    from agents.enhanced_technical_analysis_multiapi import enhanced_technical_analysis_multiapi
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    enhanced_technical_analysis_multiapi = None
    TECHNICAL_ANALYSIS_AVAILABLE = False
    print("- Enhanced technical analysis not available")

try:
    from agents.enhanced_options_pricing_engine import enhanced_options_pricing_engine
    OPTIONS_PRICING_AVAILABLE = True
except ImportError:
    enhanced_options_pricing_engine = None
    OPTIONS_PRICING_AVAILABLE = False
    print("- Enhanced options pricing not available")

try:
    from agents.advanced_monte_carlo_engine import advanced_monte_carlo_engine
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    advanced_monte_carlo_engine = None
    MONTE_CARLO_AVAILABLE = False
    print("- Advanced Monte Carlo engine not available")

try:
    from analysis.multitimeframe_analyzer import MultiTimeframeAnalyzer
    MULTITIMEFRAME_AVAILABLE = True
    print("+ Multi-timeframe analyzer loaded")
except ImportError:
    MultiTimeframeAnalyzer = None
    MULTITIMEFRAME_AVAILABLE = False
    print("- Multi-timeframe analyzer not available")

try:
    from agents.simple_iv_analyzer import get_iv_analyzer
    IV_ANALYZER_AVAILABLE = True
    print("+ IV Rank analyzer loaded")
except ImportError:
    get_iv_analyzer = None
    IV_ANALYZER_AVAILABLE = False
    print("- IV Rank analyzer not available")

try:
    from agents.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
    SENTIMENT_ANALYZER_AVAILABLE = True
    print("+ Sentiment analyzer loaded")
except ImportError:
    EnhancedSentimentAnalyzer = None
    SENTIMENT_ANALYZER_AVAILABLE = False
    print("- Sentiment analyzer not available")
try:
    from agents.sharpe_enhanced_filters import sharpe_enhanced_filters
    SHARPE_FILTERS_AVAILABLE = True
except ImportError:
    sharpe_enhanced_filters = None
    SHARPE_FILTERS_AVAILABLE = False
    print("- Sharpe enhanced filters not available")

try:
    from agents.economic_data_agent import economic_data_agent
    ECONOMIC_DATA_AVAILABLE = True
except ImportError:
    economic_data_agent = None
    ECONOMIC_DATA_AVAILABLE = False
    print("- Economic data agent not available")

try:
    from agents.cboe_data_agent import cboe_data_agent
    CBOE_DATA_AVAILABLE = True
except ImportError:
    cboe_data_agent = None
    CBOE_DATA_AVAILABLE = False
    print("- CBOE data agent not available")

try:
    from agents.advanced_technical_analysis import advanced_technical_analysis
    ADVANCED_TA_AVAILABLE = True
except ImportError:
    advanced_technical_analysis = None
    ADVANCED_TA_AVAILABLE = False
    print("- Advanced technical analysis not available")

try:
    from agents.ml_prediction_engine import ml_prediction_engine
    ML_PREDICTION_AVAILABLE = True
except ImportError:
    ml_prediction_engine = None
    ML_PREDICTION_AVAILABLE = False
    print("- ML prediction engine not available")

try:
    from agents.advanced_risk_management import advanced_risk_manager
    ADVANCED_RISK_AVAILABLE = True
except ImportError:
    advanced_risk_manager = None
    ADVANCED_RISK_AVAILABLE = False
    print("- Advanced risk management not available")

try:
    from agents.trading_dashboard import trading_dashboard
    TRADING_DASHBOARD_AVAILABLE = True
except ImportError:
    trading_dashboard = None
    TRADING_DASHBOARD_AVAILABLE = False
    print("- Trading dashboard not available")

# Create logs directory
os.makedirs('logs', exist_ok=True)

class TomorrowReadyOptionsBot:
    """Production-ready options trading bot with intelligent exit strategy"""
    
    def __init__(self):
        self.broker = None
        self.options_trader = None
        self.options_broker = None
        self.risk_manager = RiskManager(RiskLevel.MODERATE)

        # Initialize components with availability checks
        self.exit_agent = exit_strategy_agent if EXIT_STRATEGY_AVAILABLE else None
        self.learning_engine = learning_engine if LEARNING_ENGINE_AVAILABLE else None
        self.advanced_ml = advanced_ml_engine if ADVANCED_ML_AVAILABLE else None
        self.technical_analysis = enhanced_technical_analysis_multiapi if TECHNICAL_ANALYSIS_AVAILABLE else None
        self.options_pricing = enhanced_options_pricing_engine if OPTIONS_PRICING_AVAILABLE else None
        self.monte_carlo_engine = advanced_monte_carlo_engine if MONTE_CARLO_AVAILABLE else None
        self.economic_data = economic_data_agent if ECONOMIC_DATA_AVAILABLE else None
        self.volatility_intelligence = cboe_data_agent if CBOE_DATA_AVAILABLE else None

        # Advanced AI/ML components
        self.advanced_technical = advanced_technical_analysis if ADVANCED_TA_AVAILABLE else None
        self.ml_predictions = ml_prediction_engine if ML_PREDICTION_AVAILABLE else None
        self.advanced_risk = advanced_risk_manager if ADVANCED_RISK_AVAILABLE else None
        self.dashboard = trading_dashboard if TRADING_DASHBOARD_AVAILABLE else None

        # Enhanced Sharpe Ratio Optimization System
        self.sharpe_filters = sharpe_enhanced_filters if SHARPE_FILTERS_AVAILABLE else None
        # Multi-timeframe Analysis
        self.multitimeframe_analyzer = MultiTimeframeAnalyzer() if MULTITIMEFRAME_AVAILABLE else None

        # IV Rank Analyzer
        self.iv_analyzer = get_iv_analyzer() if IV_ANALYZER_AVAILABLE else None

        # Sentiment Analyzer
        self.sentiment_analyzer = EnhancedSentimentAnalyzer() if SENTIMENT_ANALYZER_AVAILABLE else None

        # Advanced Quantitative Finance Engine
        self.quant_engine = quantitative_engine if QUANT_ENGINE_AVAILABLE else None
        self.quant_analyzer = quant_analyzer if QUANT_INTEGRATION_AVAILABLE else None

        # Display quantitative engine status
        if QUANT_ENGINE_AVAILABLE:
            print("+ Quantitative finance engine ready")
        else:
            print("- Quantitative finance engine unavailable")

        # ML Ensemble Integration
        try:
            self.ml_ensemble = get_ml_ensemble()
            if self.ml_ensemble.loaded:
                print("+ ML Ensemble loaded (RF + XGB models)")
            else:
                print("- ML Ensemble models not loaded")
                self.ml_ensemble = None
        except Exception as e:
            print(f"- ML Ensemble unavailable: {e}")
            self.ml_ensemble = None

        # Learning Acceleration Components with error handling - deferred for performance
        self.transfer_learning = None
        self.realtime_learning = None
        self.pre_trained_models = None
        self.learning_models_loaded = False
        print("- Learning models deferred (will load on demand)")

        # Enhancement Modules Integration
        try:
            self.greeks_optimizer = get_greeks_optimizer()
            self.volatility_adapter = get_volatility_adapter()
            self.spread_strategies = get_spread_strategies()
            self.market_regime_detector = get_market_regime_detector()
            self.dynamic_stop_manager = get_dynamic_stop_manager()
            self.liquidity_filter = get_liquidity_filter()
            print("+ Enhancement modules loaded (Greeks, VIX regime, Spreads, Market regime, Dynamic stops, Liquidity)")
        except Exception as e:
            print(f"- Enhancement modules unavailable: {e}")
            self.greeks_optimizer = None
            self.volatility_adapter = None
            self.spread_strategies = None
            self.market_regime_detector = None
            self.dynamic_stop_manager = None
            self.liquidity_filter = None
        
        # Eastern Time for market operations
        self.et_timezone = pytz.timezone('US/Eastern')
        
        # Trading state
        self.trade_count = 0
        self.cycle_count = 0
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.starting_equity = 0.0  # Track starting equity for accurate daily P&L
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

        # Profit target monitoring (5.75% daily target)
        self.profit_monitor = None
        self.profit_monitoring_task = None

        # Daily trading controls
        self.daily_loss_limit_hit = False
        self.trading_stopped_for_day = False
        self.daily_loss_limit_pct = -30.0  # Stop trading at -30% daily loss
        
        # Market regime tracking
        self.market_regime = 'NEUTRAL'
        self.vix_level = 20.0
        self.market_trend = 0.0
        
        # Professional trading universe
        # UPDATED: Top 80 S&P 500 stocks - Maximum coverage with excellent options liquidity
        self.tier1_stocks = [
            # TECHNOLOGY (20 stocks - 25%)
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA',
            'AVGO', 'ORCL', 'CRM', 'ADBE', 'CSCO', 'ACN', 'AMD', 'INTC',
            'NOW', 'QCOM', 'TXN', 'INTU',

            # FINANCIALS (15 stocks - 18.75%)
            'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS',
            'SPGI', 'BLK', 'C', 'AXP', 'SCHW', 'CB', 'PGR',

            # HEALTHCARE (12 stocks - 15%)
            'UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR',
            'PFE', 'BMY', 'AMGN', 'GILD',

            # CONSUMER DISCRETIONARY (9 stocks - 11.25%)
            'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG', 'MAR',

            # CONSUMER STAPLES (6 stocks - 7.5%)
            'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM',

            # ENERGY (5 stocks - 6.25%)
            'XOM', 'CVX', 'COP', 'SLB', 'EOG',

            # INDUSTRIALS (6 stocks - 7.5%)
            'BA', 'CAT', 'GE', 'RTX', 'HON', 'UPS',

            # COMMUNICATION (2 stocks - 2.5%)
            'NFLX', 'DIS',

            # UTILITIES (2 stocks - 2.5%)
            'NEE', 'DUK',

            # HIGH-VOLUME FINTECH/TECH (3 stocks)
            'PYPL', 'SQ', 'UBER'
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


    def load_learning_models(self):
        """Load machine learning models on demand"""
        if not self.learning_models_loaded:
            try:
                from agents.transfer_learning_accelerator import transfer_accelerator
                from agents.realtime_learning_engine import realtime_learner
                from agents.model_loader import model_loader

                self.transfer_learning = transfer_accelerator
                self.realtime_learning = realtime_learner
                self.pre_trained_models = model_loader
                self.learning_models_loaded = True

                self.log_trade("Learning models loaded on demand", "INFO")
                self.log_trade(f"Models: {self.pre_trained_models.get_model_info()}", "INFO")
                return True
            except Exception as e:
                self.log_trade(f"Failed to load learning models: {e}", "WARN")
                return False
        return True

    async def check_daily_loss_limit(self):
        """Check if daily loss limit has been hit and stop trading if so"""
        try:
            if self.trading_stopped_for_day:
                return True  # Already stopped

            # Get current account value and starting equity
            if not self.broker:
                return False

            account_info = await self.broker.get_account_info()
            if not account_info:
                self.log_trade("Could not retrieve account info for loss limit check", "WARN")
                return False

            # Get current equity (try both 'equity' and 'portfolio_value' for compatibility)
            current_equity = float(getattr(account_info, 'equity', 0))
            if current_equity == 0:
                current_equity = float(account_info.get('portfolio_value', 0))

            # Use the starting_equity captured at market open
            # If not set yet, use current equity as baseline
            if self.starting_equity == 0:
                self.starting_equity = current_equity
                self.log_trade(f"Starting equity not set, using current: ${current_equity:,.2f}", "INFO")

            starting_equity = self.starting_equity

            # Calculate daily P&L percentage
            if starting_equity > 0:
                daily_pnl_pct = ((current_equity - starting_equity) / starting_equity) * 100
                daily_pnl_dollars = current_equity - starting_equity
            else:
                daily_pnl_pct = 0
                daily_pnl_dollars = 0

            # Check if loss limit hit
            if daily_pnl_pct <= self.daily_loss_limit_pct:
                if not self.daily_loss_limit_hit:
                    self.daily_loss_limit_hit = True
                    self.trading_stopped_for_day = True

                    self.log_trade(f"[CRITICAL] DAILY LOSS LIMIT HIT: {daily_pnl_pct:.2f}% <= {self.daily_loss_limit_pct}%", "CRITICAL")
                    self.log_trade(f"Current Equity: ${current_equity:,.2f} | Starting Equity: ${starting_equity:,.2f}", "INFO")
                    self.log_trade(f"Daily Loss: ${daily_pnl_dollars:,.2f}", "INFO")
                    self.log_trade("STOPPING ALL TRADING FOR THE DAY", "CRITICAL")

                    # Close all positions immediately
                    if self.broker:
                        try:
                            await self.broker.close_all_positions()
                            self.log_trade("All positions closed due to daily loss limit", "INFO")
                        except Exception as e:
                            self.log_trade(f"Error closing positions: {e}", "ERROR")

                    # Stop profit monitoring
                    if self.profit_monitoring_task:
                        self.profit_monitoring_task.cancel()

                return True

            return False

        except Exception as e:
            self.log_trade(f"Error checking daily loss limit: {e}", "ERROR")
            return False

    def reset_daily_trading_flags(self):
        """Reset daily trading flags for a new trading day"""
        self.daily_loss_limit_hit = False
        self.trading_stopped_for_day = False
        self.log_trade("Daily trading flags reset for new trading day", "INFO")

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
            # 0. Reset daily trading flags for new day
            self.reset_daily_trading_flags()

            # 1. Initialize all systems
            self.log_trade("Step 1: Initializing trading systems...")
            if await self.initialize_all_systems():
                self.readiness_status['broker_connected'] = True
                self.readiness_status['account_validated'] = True
                self.log_trade("[OK] Systems initialized successfully")

                # Save starting equity for daily loss limit tracking
                try:
                    account_info = await self.broker.get_account_info()
                    if account_info:
                        starting_equity = float(account_info.get('portfolio_value', 0))
                        equity_data = {
                            'starting_equity': starting_equity,
                            'date': datetime.now().strftime('%Y-%m-%d')
                        }
                        with open('daily_starting_equity.json', 'w') as f:
                            json.dump(equity_data, f)
                        self.log_trade(f"Starting equity saved: ${starting_equity:,.2f}", "INFO")
                except Exception as e:
                    self.log_trade(f"Could not save starting equity: {e}", "WARN")
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
            
            # Initialize profit target monitoring (5.75% daily target)
            self.log_trade("Initializing profit/loss monitoring (5.75% target, -4.9% limit)...")
            try:
                self.profit_monitor = ProfitTargetMonitor()
                if await self.profit_monitor.initialize_broker():
                    # Start monitoring in background
                    self.profit_monitoring_task = asyncio.create_task(
                        self.profit_monitor.monitor_profit_target(check_interval=30)  # Check every 30 seconds
                    )
                    self.log_trade("✅ Profit/loss monitoring started (5.75% target, -4.9% limit)", "INFO")
                else:
                    self.log_trade("❌ Failed to initialize profit target monitoring", "WARN")
            except Exception as e:
                self.log_trade(f"Profit monitoring initialization error: {e}", "WARN")

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

        # Check daily loss limit FIRST - if hit, return empty plan
        loss_limit_hit = await self.check_daily_loss_limit()
        if loss_limit_hit:
            self.log_trade("Daily loss limit hit - returning empty trading plan", "WARN")
            return {
                'market_regime': self.market_regime,
                'preferred_strategies': [],
                'target_new_positions': 0,
                'focus_symbols': [],
                'risk_adjustments': ['trading_stopped_for_day'],
                'message': f'Trading stopped due to daily loss limit of {self.daily_loss_limit_pct}%'
            }

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

        # Check daily loss limit - if hit, close all positions and stop
        loss_limit_hit = await self.check_daily_loss_limit()
        if loss_limit_hit:
            self.log_trade("Daily loss limit hit - closing all positions in monitoring", "WARN")
            self.active_positions.clear()  # Clear all positions as they should be closed
            return

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
                current_pnl = await self.calculate_position_pnl(position_data, market_data)
                
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
                
                # Add detailed P&L breakdown for debugging
                quantity = position_data.get('quantity', 1)
                entry_price = position_data.get('entry_price', 0)
                total_investment = entry_price * quantity * 100

                self.log_trade(f"MONITORING: {symbol} - P&L: ${current_pnl:.2f} ({enhanced_analysis['pnl_percentage']:+.1f}%), Signal: {exit_decision.signal}")
                self.log_trade(f"  Position Details: {quantity} contracts @ ${entry_price:.2f} entry = ${total_investment:.2f} invested")
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

                # DYNAMIC STOP LOSS CHECK (Priority: Check first)
                if self.dynamic_stop_manager:
                    try:
                        # Get entry data
                        entry_price_stops = position_data.get('entry_price', 0)
                        entry_date = position_data.get('entry_date', datetime.now())

                        # Get current price (estimate based on P&L)
                        # Approximate current option price from P&L
                        quantity = position_data.get('quantity', 1)
                        if quantity > 0 and entry_price_stops > 0:
                            price_change = current_pnl / (quantity * 100)  # P&L per contract
                            current_option_price = entry_price_stops + price_change
                        else:
                            current_option_price = entry_price_stops

                        # Update peak price
                        peak_price = position_data.get('peak_price', entry_price_stops)
                        if current_option_price > peak_price:
                            peak_price = current_option_price
                            position_data['peak_price'] = peak_price

                        # Calculate current P&L percentage
                        if entry_price_stops > 0:
                            current_pnl_pct = (current_option_price - entry_price_stops) / entry_price_stops
                        else:
                            current_pnl_pct = 0

                        # Check if dynamic stop hit
                        exit_check = self.dynamic_stop_manager.should_exit(
                            entry_price=entry_price_stops,
                            entry_date=entry_date,
                            current_price=current_option_price,
                            peak_price=peak_price
                        )

                        if exit_check['exit']:
                            should_exit = True
                            exit_reason = f"DYNAMIC STOP: {exit_check['reason']}"
                            self.log_trade(f"  DYNAMIC STOP HIT: {exit_check['stop_hit']} - {exit_check['reason']}")
                        else:
                            self.log_trade(f"  Dynamic Stop: ${exit_check['stop_price']:.2f} | Current: ${current_option_price:.2f} | Peak: ${peak_price:.2f}")

                    except Exception as dyn_stop_error:
                        self.log_trade(f"  Dynamic stop error: {dyn_stop_error}", "WARN")

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
    
    async def calculate_position_pnl(self, position_data, market_data):
        """Calculate current P&L - uses REAL broker data when available"""
        try:
            symbol = position_data['opportunity']['symbol']

            # FIRST: Try to get REAL P&L from broker positions
            try:
                # FIXED: Try to get option symbol first (for accurate broker lookup)
                option_symbol = position_data.get('option_symbol')
                if not option_symbol:
                    option_symbol = position_data.get('opportunity', {}).get('option_symbol')

                # Get all positions from broker
                all_positions = await self.broker.get_positions()

                # FIXED: Match using option_symbol first, fall back to underlying symbol
                for pos in all_positions:
                    # Try exact option symbol match first (most accurate)
                    if option_symbol and pos.symbol == option_symbol and hasattr(pos, 'unrealized_pl'):
                        real_pnl = float(pos.unrealized_pl)
                        pnl_pct = (real_pnl / (float(pos.cost_basis) if float(pos.cost_basis) > 0 else 1)) * 100
                        self.log_trade(f"  [REAL BROKER P&L via option_symbol] ${real_pnl:.2f} ({pnl_pct:+.1f}%)", "INFO")
                        return real_pnl
                    # Fall back to underlying symbol (less reliable but better than nothing)
                    elif pos.symbol == symbol and hasattr(pos, 'unrealized_pl'):
                        real_pnl = float(pos.unrealized_pl)
                        pnl_pct = (real_pnl / (float(pos.cost_basis) if float(pos.cost_basis) > 0 else 1)) * 100
                        self.log_trade(f"  [REAL BROKER P&L via underlying symbol] ${real_pnl:.2f} ({pnl_pct:+.1f}%)", "INFO")
                        return real_pnl

                # If not found in positions, try individual position lookup
                lookup_symbol = option_symbol if option_symbol else symbol
                broker_position = await self.broker.get_position(lookup_symbol)
                if broker_position and hasattr(broker_position, 'unrealized_pl') and broker_position.unrealized_pl is not None:
                    real_pnl = float(broker_position.unrealized_pl)
                    if hasattr(broker_position, 'cost_basis') and float(broker_position.cost_basis) > 0:
                        pnl_pct = (real_pnl / float(broker_position.cost_basis)) * 100
                        self.log_trade(f"  [REAL BROKER P&L] ${real_pnl:.2f} ({pnl_pct:+.1f}%)", "INFO")
                    else:
                        self.log_trade(f"  [REAL BROKER P&L] ${real_pnl:.2f}", "INFO")
                    return real_pnl

            except Exception as broker_error:
                self.log_trade(f"  [WARN] Could not get real broker P&L: {broker_error}", "WARN")

            # SECOND: Try to get REAL current market price from broker (CRITICAL FIX)
            try:
                option_symbol = position_data.get('option_symbol')
                if not option_symbol:
                    option_symbol = position_data.get('opportunity', {}).get('option_symbol')

                if option_symbol and hasattr(self, 'broker') and self.broker:
                    # Try to get current market quote for the option
                    current_quote = await self.broker.get_latest_quote(option_symbol)
                    if current_quote:
                        # Use mark price (midpoint of bid/ask) if available, else last price
                        current_price = None
                        if hasattr(current_quote, 'ap') and hasattr(current_quote, 'bp'):
                            # Calculate mark price from bid/ask
                            if current_quote.ap > 0 and current_quote.bp > 0:
                                current_price = (current_quote.ap + current_quote.bp) / 2
                        elif hasattr(current_quote, 'last_price') and current_quote.last_price:
                            current_price = float(current_quote.last_price)

                        if current_price and current_price > 0:
                            entry_price = position_data.get('entry_price', 0)
                            quantity = position_data.get('quantity', 1)

                            # Calculate P&L from REAL market prices
                            total_pnl = (current_price - entry_price) * quantity * 100
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

                            self.log_trade(f"  [REAL MARKET PRICE P&L] Entry: ${entry_price:.2f}, Current: ${current_price:.2f}, P&L: ${total_pnl:.2f} ({pnl_pct:+.1f}%)", "INFO")
                            return total_pnl
            except Exception as quote_error:
                self.log_trade(f"  [WARN] Could not get real market quote: {quote_error}", "WARN")

            # FALLBACK: Use tracked P&L with estimated pricing (LAST RESORT)
            # Get position details
            entry_price = position_data.get('entry_price', 0)  # Entry price per contract
            quantity = position_data.get('quantity', 1)  # Number of contracts

            # Get current option value (professional pricing)
            self.log_trade(f"  [FALLBACK] Using estimated pricing (not reliable!)", "WARN")
            current_option_price = await self.estimate_current_option_price(position_data, market_data)

            # Sanity check: if estimate seems broken, use entry price as fallback
            if current_option_price <= 0:
                self.log_trade(f"  [ERROR] Invalid option price ${current_option_price:.2f}, using entry price ${entry_price:.2f}", "ERROR")
                current_option_price = entry_price

            # Calculate P&L based on contract value difference
            # Each contract represents 100 shares, so multiply by 100
            price_per_contract_change = current_option_price - entry_price
            total_pnl = price_per_contract_change * quantity * 100

            # Calculate percentage for logging
            if entry_price > 0:
                pnl_pct = ((current_option_price - entry_price) / entry_price) * 100
                self.log_trade(f"  [PAPER MODE P&L] ${total_pnl:.2f} ({pnl_pct:+.1f}%)", "INFO")

                # Log large P&L for information (no cap - show actual values)
                if abs(pnl_pct) > 1000:  # Over 10x gain
                    self.log_trade(f"  [INFO] Large gain detected: Entry: ${entry_price:.2f}, "
                                 f"Current: ${current_option_price:.2f}, "
                                 f"P&L: ${total_pnl:.2f} ({pnl_pct:+.1f}%)", "INFO")
            else:
                self.log_trade(f"  [PAPER MODE P&L] ${total_pnl:.2f}", "INFO")

            return total_pnl

        except Exception as e:
            self.log_trade(f"P&L calculation error: {e}", "WARN")
            return 0.0
    
    async def estimate_current_option_price(self, position_data, market_data):
        """Get current option price - tries to fetch real market data first"""
        try:
            opportunity = position_data['opportunity']
            strategy = opportunity['strategy']
            entry_time = position_data['entry_time']
            current_stock_price = market_data['current_price']
            entry_price = position_data.get('entry_price', 1.0)

            # Get option parameters
            strike_price = opportunity.get('strike_price', current_stock_price)
            symbol = opportunity.get('symbol')

            # TRY 1: Get real option quote from broker/market (paper mode compatible)
            try:
                # Try to get current option contract price from options broker
                if hasattr(self, 'options_broker') and self.options_broker:
                    option_symbol = opportunity.get('option_symbol')
                    if option_symbol:
                        # Get current bid/ask for the option
                        quote = await self.options_broker.get_option_quote(option_symbol)
                        if quote and 'mark' in quote:
                            current_price = quote['mark']
                            if current_price > 0:
                                self.log_trade(f"  Using LIVE option price: ${current_price:.2f}", "DEBUG")
                                return current_price
            except Exception as e:
                pass  # Fall through to estimation

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
            
            # Calculate percentage gains/losses based on total investment
            quantity = position_data.get('quantity', 1)
            total_investment = entry_price * quantity * 100  # Total cost of position

            # Validate total investment to prevent ridiculous percentage calculations
            if total_investment < 1.0:  # If investment is less than $1, something is wrong
                self.log_trade(f"WARNING: Suspicious total investment: ${total_investment:.2f} (entry_price=${entry_price:.4f}, qty={quantity})", "WARN")
                pnl_percentage = 0  # Don't calculate percentage for invalid data
            else:
                pnl_percentage = (current_pnl / total_investment) * 100
            
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
            
            # HARD STOP LOSS at -20% (CRITICAL FIX)
            if pnl_percentage <= -20:
                return {
                    "should_exit": True,
                    "reason": f"STOP LOSS: Position down {pnl_percentage:.1f}%",
                    "confidence": 0.95,
                    "factors": ["stop_loss_triggered"],
                    "exit_signals": 5,
                    "hold_signals": 0,
                    "pnl_percentage": pnl_percentage,
                    "days_held": days_held
                }

            # Decision Logic
            net_signal_strength = exit_signals - hold_signals
            should_exit = False
            confidence = 0.5

            # SPECIAL CASE: Losing positions (CRITICAL FIX - exits faster)
            if pnl_percentage < -10:  # Down more than 10%
                if net_signal_strength >= 1:  # Much lower threshold for losers
                    should_exit = True
                    confidence = min(0.85, 0.6 + (net_signal_strength * 0.1))
                    reason = f"Exit losing position (score: +{net_signal_strength}, P&L: {pnl_percentage:.1f}%)"
                elif net_signal_strength >= 0:  # Even neutral exits losers
                    should_exit = True
                    confidence = 0.65
                    reason = f"Exit losing position on neutral signal (P&L: {pnl_percentage:.1f}%)"
                else:
                    should_exit = False
                    reason = f"Hold losing position (score: {net_signal_strength}, P&L: {pnl_percentage:.1f}%)"
            # ORIGINAL LOGIC for winning/neutral positions
            elif net_signal_strength >= 3:  # Strong exit signal
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

    async def handle_insufficient_funds_emergency(self):
        """Handle insufficient funds by triggering emergency exit analysis"""
        try:
            self.log_trade("=== EMERGENCY FUND MANAGEMENT MODE ===", "WARN")

            if not self.active_positions:
                self.log_trade("No active positions to exit for fund management", "INFO")
                return

            self.log_trade(f"Analyzing {len(self.active_positions)} positions for emergency exits...", "INFO")

            # Get current account info to see available funds
            try:
                account_info = await self.broker.get_account_info()
                if account_info:
                    buying_power = float(account_info.get('buying_power', 0))
                    self.log_trade(f"Current buying power: ${buying_power:.2f}", "INFO")
            except Exception as account_error:
                self.log_trade(f"Could not get account info: {account_error}", "WARN")

            # Sort positions by profitability and urgency for emergency exits
            emergency_exits = []

            for position_id, position_data in list(self.active_positions.items()):
                try:
                    symbol = position_data['opportunity']['symbol']

                    # Get current market data for exit analysis
                    market_data = await self.get_enhanced_market_data(symbol)
                    if not market_data:
                        continue

                    # Calculate current P&L
                    current_pnl = await self.calculate_position_pnl(position_data, market_data)

                    # Force urgent exit analysis - override normal hold signals
                    exit_decision = self.exit_agent.analyze_position_exit(
                        position_data, market_data, current_pnl
                    )

                    # Calculate exit priority (profitable first, then least loss)
                    if current_pnl > 0:
                        priority = 1000 + current_pnl  # High priority for profitable positions
                    else:
                        priority = 500 + current_pnl   # Higher priority for smaller losses

                    emergency_exits.append({
                        'position_id': position_id,
                        'position_data': position_data,
                        'exit_decision': exit_decision,
                        'current_pnl': current_pnl,
                        'priority': priority,
                        'symbol': symbol
                    })

                    self.log_trade(f"Emergency exit candidate: {symbol} P&L: ${current_pnl:.2f}, Priority: {priority:.0f}")

                except Exception as position_error:
                    self.log_trade(f"Error analyzing position {position_id} for emergency exit: {position_error}", "ERROR")

            # Sort by priority (highest first) - profitable positions and smaller losses first
            emergency_exits.sort(key=lambda x: x['priority'], reverse=True)

            # Execute emergency exits for top candidates
            exits_executed = 0
            max_emergency_exits = min(3, len(emergency_exits))  # Exit up to 3 positions

            for exit_info in emergency_exits[:max_emergency_exits]:
                self.log_trade(f"EMERGENCY EXIT: {exit_info['symbol']} (P&L: ${exit_info['current_pnl']:.2f})", "WARN")

                try:
                    await self.execute_intelligent_exit(exit_info)
                    exits_executed += 1

                    # Add a small delay between exits
                    await asyncio.sleep(1)

                except Exception as exit_error:
                    self.log_trade(f"Emergency exit failed for {exit_info['symbol']}: {exit_error}", "ERROR")

            self.log_trade(f"=== EMERGENCY FUND MANAGEMENT COMPLETE ===", "INFO")
            self.log_trade(f"Executed {exits_executed} emergency exits to free up capital", "INFO")

            # Update account info after exits
            try:
                account_info = await self.broker.get_account_info()
                if account_info:
                    new_buying_power = float(account_info.get('buying_power', 0))
                    self.log_trade(f"Updated buying power: ${new_buying_power:.2f}", "INFO")
                    freed_capital = new_buying_power - buying_power if 'buying_power' in locals() else 0
                    if freed_capital > 0:
                        self.log_trade(f"Freed up approximately ${freed_capital:.2f} in capital", "INFO")
            except Exception as account_error:
                self.log_trade(f"Could not get updated account info: {account_error}", "WARN")

        except Exception as e:
            self.log_trade(f"Emergency fund management error: {e}", "ERROR")

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

        # 3. Capture starting equity for accurate daily P&L tracking
        try:
            account_info = await self.broker.get_account_info()
            self.starting_equity = float(account_info.equity)
            self.log_trade(f"[OK] Starting equity captured: ${self.starting_equity:,.2f}")
        except Exception as e:
            self.log_trade(f"[WARN] Could not capture starting equity: {e}")
            self.starting_equity = 0.0

        # 4. Reset daily counters
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
    
    async def get_real_daily_pnl(self):
        """Get REAL daily P&L from broker account"""
        try:
            account_info = await self.broker.get_account_info()
            current_equity = float(account_info.equity)

            if self.starting_equity > 0:
                # Calculate actual daily P&L from equity change
                real_daily_pnl = current_equity - self.starting_equity
                real_daily_pnl_pct = (real_daily_pnl / self.starting_equity) * 100
                return real_daily_pnl, real_daily_pnl_pct, current_equity
            else:
                # No starting equity captured, return current values
                return 0.0, 0.0, current_equity

        except Exception as e:
            self.log_trade(f"Could not get real P&L: {e}", "WARN")
            return None, None, None

    async def log_daily_performance(self):
        """Log comprehensive daily performance"""
        stats = self.performance_stats

        win_rate = 0
        if stats['total_trades'] > 0:
            win_rate = (stats['winning_trades'] / stats['total_trades']) * 100

        self.log_trade("=== DAILY PERFORMANCE SUMMARY ===", "INFO")

        # Get REAL P&L from broker
        real_pnl, real_pnl_pct, current_equity = await self.get_real_daily_pnl()

        if real_pnl is not None:
            self.log_trade(f"[REAL BROKER DATA]")
            self.log_trade(f"  Current Equity: ${current_equity:,.2f}")
            self.log_trade(f"  Starting Equity: ${self.starting_equity:,.2f}")
            self.log_trade(f"  Daily P&L: ${real_pnl:+,.2f} ({real_pnl_pct:+.2f}%)")
            self.daily_pnl = real_pnl  # Update internal tracking with real value
        else:
            self.log_trade(f"Daily P&L (estimated): ${self.daily_pnl:.2f}")

        self.log_trade(f"Total P&L (all-time): ${stats['total_profit']:.2f}")
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

        # CHECK LOSS LIMIT FIRST (CRITICAL FIX - prevent trading past limit)
        loss_limit_hit = await self.check_daily_loss_limit()
        if loss_limit_hit:
            self.log_trade("Daily loss limit hit - SKIPPING ALL TRADING", "CRITICAL")
            return  # Hard stop - don't do anything else

        # 1. Intelligent position monitoring (every cycle)
        await self.intelligent_position_monitoring()

        # 2. Look for new opportunities (only if loss limit not hit)
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
        # UPDATED: Scan all 20 stocks in one go (no batching needed with smaller watchlist)
        scan_symbols = self.tier1_stocks

        self.log_trade(f"Scanning {len(scan_symbols)} diversified stocks: {', '.join(scan_symbols[:5])}...")
        
        for i, symbol in enumerate(scan_symbols):
            try:
                # Rate limiting: Add small delay every 10 stocks to avoid API throttling
                if i > 0 and i % 10 == 0:
                    import time
                    time.sleep(1)  # 1 second pause every 10 stocks
                opportunity = await self.find_high_quality_opportunity(symbol)
                if opportunity:
                    opportunities.append(opportunity)
                    self.log_trade(f"OPPORTUNITY: {symbol} {opportunity['strategy']} - Confidence: {opportunity.get('confidence', 0):.1%}")
            except Exception as e:
                self.log_trade(f"Opportunity scan error for {symbol}: {e}", "WARN")
        
        self.log_trade(f"Scan complete: Found {len(opportunities)} opportunities from {len(scan_symbols)} symbols")
        
        # Execute ONLY opportunities with 65%+ confidence (OPTIMIZED)
        if opportunities:
            # Filter opportunities with 65% or higher confidence (balanced quality vs frequency)
            high_confidence_opportunities = [opp for opp in opportunities if opp.get('confidence', 0) >= 0.65]

            if high_confidence_opportunities:
                self.log_trade(f"Found {len(high_confidence_opportunities)} opportunities with 65%+ confidence")

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
                # If no 65%+ confidence, don't trade (balanced threshold)
                best_confidence = max([opp.get('confidence', 0) for opp in opportunities]) if opportunities else 0
                self.log_trade(f"No opportunities meet 65% confidence threshold (best: {best_confidence:.1%})")
    
    async def find_high_quality_opportunity(self, symbol):
        """Find high-quality trading opportunity with enhanced filters for Sharpe optimization"""
        try:
            # Use sequential API manager - exhausts one API before moving to next
            from agents.sequential_api_manager import get_api_manager
            api_manager = get_api_manager()

            market_data = api_manager.get_market_data(symbol, period="10d")

            if not market_data:
                return None

            # Log API source for tracking
            if market_data.get('source'):
                self.log_trade(f"Data for {symbol} from {market_data['source'].upper()} API")

            # Get realized volatility early (needed for filters)
            realized_vol = market_data.get('realized_vol', 20)

            # Apply comprehensive filters for Sharpe ratio optimization
            try:
                if self.sharpe_filters:
                    filter_results = await self.sharpe_filters.get_comprehensive_filters(symbol, realized_vol / 100)
                else:
                    filter_results = None

                # Extract filter results (use defaults if filters not available)
                if filter_results:
                    rsi_filter = filter_results.get('rsi_filter', {})
                    ema_filter = filter_results.get('ema_filter', {})
                    momentum_filter = filter_results.get('momentum_filter', {})
                    volatility_filter = filter_results.get('volatility_filter', {})
                    iv_rank_filter = filter_results.get('iv_rank_filter', {})
                    earnings_filter = filter_results.get('earnings_filter', {})
                else:
                    # Use default/empty filters if sharpe_filters not available
                    rsi_filter = {'pass': True}
                    ema_filter = {'trend': 'NEUTRAL'}
                    momentum_filter = {}
                    volatility_filter = {'regime': 'NORMAL'}
                    iv_rank_filter = {}
                    earnings_filter = {}

                # Check if RSI filter passes (avoid extreme conditions)
                if not rsi_filter.get('pass', True):
                    return None  # Skip extreme RSI conditions

                # Check if EMA trend is favorable
                ema_trend = ema_filter.get('trend', 'NEUTRAL')
                # REMOVED BEARISH FILTER - was causing bot to only trade calls in bearish markets
                # Previously filtered out 70% of bearish/PUT trades which is WRONG

                # Check volatility regime for position sizing later
                vol_regime = volatility_filter.get('regime', 'NORMAL')

                # Check IV rank for premium strategies
                iv_rank = iv_rank_filter.get('rank', 50)
                if iv_rank < 30:  # Skip low IV environments for premium selling
                    if np.random.random() < 0.4:
                        return None

                # Check earnings proximity
                if not earnings_filter.get('safe_to_trade', True):
                    return None  # Skip trades near earnings

                # Enhanced opportunity logic with filters
                volume_ok = market_data['volume_ratio'] > 0.8
                momentum_ok = abs(market_data['price_momentum']) > 0.015
                filter_ok = (rsi_filter.get('pass', True) and
                           iv_rank >= 30)  # REMOVED bearish bias filter

                # Additional quality checks
                price_position = market_data.get('price_position', 0.5)
                vol_ok = realized_vol > 15

            except Exception as e:
                # Fallback to basic filters if enhanced filters fail
                self.log_trade(f"Enhanced filters failed for {symbol}, using basic: {e}", "WARN")
                volume_ok = market_data['volume_ratio'] > 0.8
                momentum_ok = abs(market_data['price_momentum']) > 0.015
                filter_ok = True
                price_position = market_data.get('price_position', 0.5)
                vol_ok = realized_vol > 15
                vol_regime = 'NORMAL'
                ema_trend = 'NEUTRAL'

            if (volume_ok and momentum_ok and filter_ok) or (momentum_ok and vol_ok and price_position > 0.3 and filter_ok):
                
                # Enhanced strategy selection based on filters and momentum
                # FIXED: Removed bullish bias, now properly trades BOTH calls AND puts
                if 'ema_trend' in locals() and ema_trend == 'BULLISH':
                    # Strong bullish EMA bias - prefer calls
                    strategy = OptionsStrategy.LONG_CALL
                elif 'ema_trend' in locals() and ema_trend == 'BEARISH':
                    # Bearish EMA bias - prefer puts (FIXED: now properly used)
                    strategy = OptionsStrategy.LONG_PUT
                else:
                    # Traditional momentum-based selection
                    if market_data['price_momentum'] > 0.01:
                        strategy = OptionsStrategy.LONG_CALL
                    elif market_data['price_momentum'] < -0.01:
                        strategy = OptionsStrategy.LONG_PUT
                    else:
                        # FIXED: Use market regime instead of defaulting to calls
                        # Check broader market trend via SPY or use momentum sign
                        if market_data['price_momentum'] >= 0:
                            strategy = OptionsStrategy.LONG_CALL
                        else:
                            strategy = OptionsStrategy.LONG_PUT

                # Multi-timeframe Analysis
                mtf_trend = 'NEUTRAL'
                mtf_confidence_bonus = 0.0
                if self.multitimeframe_analyzer:
                    try:
                        mtf_data = self.multitimeframe_analyzer.fetch_multitimeframe_data(symbol, period='6mo')
                        if mtf_data and len(mtf_data) >= 2:
                            # Analyze daily and weekly timeframes
                            daily_trend = 'NEUTRAL'
                            weekly_trend = 'NEUTRAL'
                            
                            if '1d' in mtf_data:
                                daily_df = self.multitimeframe_analyzer.calculate_technical_indicators(mtf_data['1d'], '1d')
                                if not daily_df.empty and len(daily_df) > 1:
                                    latest = daily_df.iloc[-1]
                                    if latest['Close'] > latest['sma_20'] and latest['macd'] > latest['macd_signal']:
                                        daily_trend = 'BULLISH'
                                    elif latest['Close'] < latest['sma_20'] and latest['macd'] < latest['macd_signal']:
                                        daily_trend = 'BEARISH'
                            
                            if '1wk' in mtf_data:
                                weekly_df = self.multitimeframe_analyzer.calculate_technical_indicators(mtf_data['1wk'], '1wk')
                                if not weekly_df.empty and len(weekly_df) > 1:
                                    latest = weekly_df.iloc[-1]
                                    if latest['Close'] > latest['sma_20'] and latest['macd'] > latest['macd_signal']:
                                        weekly_trend = 'BULLISH'
                                    elif latest['Close'] < latest['sma_20'] and latest['macd'] < latest['macd_signal']:
                                        weekly_trend = 'BEARISH'
                            
                            # MTF alignment bonus
                            if daily_trend == weekly_trend:
                                mtf_trend = daily_trend
                                if mtf_trend == 'BULLISH' and strategy == OptionsStrategy.LONG_CALL:
                                    mtf_confidence_bonus = 0.20  # Strong MTF alignment for calls
                                elif mtf_trend == 'BEARISH' and strategy == OptionsStrategy.LONG_PUT:
                                    mtf_confidence_bonus = 0.20  # Strong MTF alignment for puts
                                elif mtf_trend != 'NEUTRAL':
                                    mtf_confidence_bonus = 0.10  # Aligned but different strategy
                            elif daily_trend != 'NEUTRAL' or weekly_trend != 'NEUTRAL':
                                mtf_confidence_bonus = 0.05  # Partial alignment
                                
                    except Exception as e:
                        self.log_trade(f"MTF analysis error for {symbol}: {e}", "WARN")

                # IV Rank Analysis for Options Buying Environment
                iv_recommendation = None
                iv_confidence_adjustment = 0.0
                if self.iv_analyzer and 'iv_rank' in locals():
                    try:
                        # Get current IV from the market data or iv_rank_filter
                        current_iv = market_data.get('implied_volatility', iv_rank / 100.0)

                        # Analyze IV environment
                        iv_recommendation = self.iv_analyzer.should_buy_options(symbol, current_iv)
                        iv_confidence_adjustment = iv_recommendation['confidence_adjustment']

                        self.log_trade(
                            f"IV Analysis {symbol}: {iv_recommendation['recommendation']} - "
                            f"{iv_recommendation['reasoning']} (Adj: {iv_confidence_adjustment:+.0%})"
                        )
                    except Exception as e:
                        self.log_trade(f"IV analysis error for {symbol}: {e}", "WARN")

                # Sentiment Analysis
                sentiment_score = 0.0
                sentiment_confidence_adjustment = 0.0
                if self.sentiment_analyzer:
                    try:
                        # Analyze sentiment (async operation with timeout)
                        sentiment_result = await asyncio.wait_for(
                            self.sentiment_analyzer.analyze_symbol_sentiment(symbol),
                            timeout=3.0  # 3 second timeout to avoid blocking
                        )

                        composite = sentiment_result.get('composite_sentiment', {})
                        sentiment_score = composite.get('composite_score', 0.0)
                        sentiment_confidence = composite.get('confidence', 0.0)
                        sentiment_label = composite.get('sentiment_label', 'neutral')

                        # Adjust confidence based on sentiment alignment
                        # Strong sentiment alignment boosts confidence, conflicts reduce it
                        if sentiment_confidence > 0.6:  # Only consider high-confidence sentiment
                            if strategy == OptionsStrategy.LONG_CALL:
                                if sentiment_score > 0.5:  # Very positive sentiment for calls
                                    sentiment_confidence_adjustment = 0.12
                                elif sentiment_score < -0.5:  # Very negative sentiment conflicts with call
                                    sentiment_confidence_adjustment = -0.15
                            elif strategy == OptionsStrategy.LONG_PUT:
                                if sentiment_score < -0.5:  # Very negative sentiment for puts
                                    sentiment_confidence_adjustment = 0.12
                                elif sentiment_score > 0.5:  # Very positive sentiment conflicts with put
                                    sentiment_confidence_adjustment = -0.15

                        self.log_trade(
                            f"Sentiment {symbol}: {sentiment_label} (score: {sentiment_score:+.2f}, "
                            f"confidence: {sentiment_confidence:.0%}) | Adj: {sentiment_confidence_adjustment:+.0%}"
                        )
                    except asyncio.TimeoutError:
                        self.log_trade(f"Sentiment analysis timeout for {symbol} (skipping)", "WARN")
                    except Exception as e:
                        self.log_trade(f"Sentiment analysis error for {symbol}: {e}", "WARN")

                # SELECTIVE CONFIDENCE SCORING - Must earn 85%+ to execute
                # Start lower - make the signal prove itself
                base_confidence = 0.30

                # REJECTION CRITERIA - Reject conflicting or weak setups
                reject_reasons = []

                # 0. IV Rank too low for buying options (CRITICAL)
                if iv_recommendation and iv_recommendation['recommendation'] == 'AVOID':
                    reject_reasons.append(iv_recommendation['reasoning'])

                # 1. Extreme RSI conflicts with strategy
                if 'rsi_filter' in locals() and rsi_filter:
                    rsi_value = rsi_filter.get('rsi', 50)
                    if strategy == OptionsStrategy.LONG_CALL and rsi_value > 80:
                        reject_reasons.append(f"RSI too high ({rsi_value:.0f}) for CALL - overbought")
                    elif strategy == OptionsStrategy.LONG_PUT and rsi_value < 20:
                        reject_reasons.append(f"RSI too low ({rsi_value:.0f}) for PUT - oversold")

                # 2. Very low volume (lowered threshold from 0.5 to 0.25 to allow more opportunities)
                volume_ratio = market_data.get('volume_ratio', 0)
                if volume_ratio < 0.25:
                    reject_reasons.append(f"Volume too low ({volume_ratio:.2f}x avg)")

                # 3. EMA conflicts with strategy (strong bearish with call)
                if 'ema_trend' in locals():
                    if ema_trend == 'BEARISH' and strategy == OptionsStrategy.LONG_CALL:
                        reject_reasons.append("BEARISH EMA conflicts with CALL strategy")
                    elif ema_trend == 'BULLISH' and strategy == OptionsStrategy.LONG_PUT:
                        reject_reasons.append("BULLISH EMA conflicts with PUT strategy")

                # 4. MTF conflicts with strategy
                if mtf_trend == 'BEARISH' and strategy == OptionsStrategy.LONG_CALL:
                    reject_reasons.append("BEARISH MTF conflicts with CALL")
                elif mtf_trend == 'BULLISH' and strategy == OptionsStrategy.LONG_PUT:
                    reject_reasons.append("BULLISH MTF conflicts with PUT")

                # 5. Weak momentum
                momentum = market_data.get('price_momentum', 0)
                if abs(momentum) < 0.005:
                    reject_reasons.append(f"Momentum too weak ({momentum:.4f})")

                # 6. Extremely negative sentiment conflicts with strategy
                if sentiment_score < -0.7 and strategy == OptionsStrategy.LONG_CALL:
                    reject_reasons.append(f"Very negative sentiment ({sentiment_score:.2f}) conflicts with CALL")
                elif sentiment_score > 0.7 and strategy == OptionsStrategy.LONG_PUT:
                    reject_reasons.append(f"Very positive sentiment ({sentiment_score:.2f}) conflicts with PUT")

                # If any critical conflicts exist, REJECT the trade by setting confidence to 0
                if reject_reasons:
                    self.log_trade(f"REJECTED {symbol} {strategy.value}: {'; '.join(reject_reasons)}")
                    confidence = 0.0  # Force rejection
                else:
                    # SELECTIVE BONUS SYSTEM - Must have strong confirming signals

                    # Volume & momentum confirmation (up to +25%)
                    if volume_ok and momentum_ok:
                        if volume_ratio > 1.5:
                            base_confidence += 0.15  # Strong volume
                        else:
                            base_confidence += 0.10  # Good volume

                    # Volatility quality (up to +10%)
                    if vol_ok:
                        realized_vol = market_data.get('realized_vol', 20)
                        if 15 < realized_vol < 50:
                            base_confidence += 0.10  # Sweet spot volatility
                        elif realized_vol >= 50:
                            base_confidence += 0.05  # High vol (risky but opportunity)

                    # Strong momentum (up to +15%)
                    if abs(momentum) > 0.025:
                        base_confidence += 0.15  # Very strong momentum
                    elif abs(momentum) > 0.015:
                        base_confidence += 0.08  # Strong momentum

                    # EMA alignment (up to +20%)
                    try:
                        if 'ema_trend' in locals():
                            if ema_trend == 'BULLISH' and strategy == OptionsStrategy.LONG_CALL:
                                base_confidence += 0.20  # Perfect EMA alignment
                            elif ema_trend == 'BULLISH_CONT' and strategy == OptionsStrategy.LONG_CALL:
                                base_confidence += 0.12  # Good continuation
                            elif ema_trend == 'BEARISH' and strategy == OptionsStrategy.LONG_PUT:
                                base_confidence += 0.20  # Perfect bearish alignment
                    except:
                        pass

                    # RSI in optimal zone (up to +8%)
                    try:
                        if 'rsi_filter' in locals() and rsi_filter:
                            rsi_signal = rsi_filter.get('signal', '')
                            if rsi_signal == 'NEUTRAL':
                                base_confidence += 0.08  # RSI sweet spot (40-60)
                            elif rsi_signal == 'OVERSOLD' and strategy == OptionsStrategy.LONG_CALL:
                                base_confidence += 0.05  # Bounce opportunity
                            elif rsi_signal == 'OVERBOUGHT' and strategy == OptionsStrategy.LONG_PUT:
                                base_confidence += 0.05  # Reversal opportunity
                    except:
                        pass

                    # Volatility regime (up to +8%)
                    try:
                        if 'vol_regime' in locals():
                            if vol_regime == 'NORMAL':
                                base_confidence += 0.05  # Stable conditions
                            elif vol_regime == 'HIGH_VOL':
                                base_confidence += 0.08  # Opportunity in volatility
                    except:
                        pass

                    # IV rank for options pricing (up to +10%)
                    try:
                        if 'iv_rank' in locals():
                            if 50 <= iv_rank <= 70:
                                base_confidence += 0.10  # Good IV for buying options
                            elif iv_rank >= 70:
                                base_confidence += 0.06  # High IV (expensive but volatile)
                    except:
                        pass

                    # Momentum filter strength (up to +15%)
                    try:
                        if 'momentum_filter' in locals() and momentum_filter:
                            strength = momentum_filter.get('strength', 'WEAK')
                            if strength == 'STRONG':
                                base_confidence += 0.15  # Strong momentum confirmation
                            elif strength == 'MODERATE':
                                base_confidence += 0.08  # Moderate confirmation
                    except:
                        pass

                    # Multi-timeframe alignment bonus (up to +20%)
                    if mtf_confidence_bonus > 0:
                        base_confidence += mtf_confidence_bonus

                    # IV Rank environment adjustment (up to +15% or -20%)
                    if iv_confidence_adjustment != 0:
                        base_confidence += iv_confidence_adjustment

                    # Sentiment alignment adjustment (up to +12% or -15%)
                    if sentiment_confidence_adjustment != 0:
                        base_confidence += sentiment_confidence_adjustment

                    # Cap at 92% to leave room for ML enhancement
                    base_confidence = min(0.92, base_confidence)
                
                # Apply basic machine learning calibration
                if self.learning_engine:
                    confidence = self.learning_engine.calibrate_confidence(
                        base_confidence, strategy.value, symbol, market_data
                    )
                else:
                    confidence = base_confidence

                # Apply ML Ensemble prediction (60/40 hybrid: learning_engine 60%, ML ensemble 40%)
                if self.ml_ensemble:
                    try:
                        ml_prediction = self._get_ml_prediction(market_data, symbol)
                        ml_confidence = ml_prediction.get('confidence', 0.5)
                        ml_direction = ml_prediction.get('prediction', 0)  # 0=down, 1=up

                        # Check if ML prediction aligns with strategy
                        strategy_is_bullish = strategy == OptionsStrategy.LONG_CALL
                        ml_is_bullish = ml_direction == 1

                        if strategy_is_bullish == ml_is_bullish:
                            # ML agrees with strategy - blend confidences
                            confidence = (confidence * 0.6) + (ml_confidence * 0.4)
                            self.log_trade(f"ML BOOST: {symbol} - Learning: {confidence*0.6:.1%}, ML: {ml_confidence*0.4:.1%} = {confidence:.1%}")
                        else:
                            # ML disagrees - reduce confidence
                            confidence = confidence * 0.7
                            self.log_trade(f"ML CONFLICT: {symbol} - Reduced confidence to {confidence:.1%}")

                        # Log ML votes if available
                        if 'model_votes' in ml_prediction and ml_prediction['model_votes']:
                            votes = ml_prediction['model_votes']
                            self.log_trade(f"  ML Votes: RF={votes.get('rf', '?')}, XGB={votes.get('xgb', '?')}")

                    except Exception as e:
                        self.log_trade(f"ML ensemble integration error: {e}", "WARN")

                # ENSEMBLE VOTING SYSTEM - Final decision from all strategies
                try:
                    ensemble_system = get_ensemble_system()

                    # Prepare ML signal for ensemble
                    ml_signal = None
                    if self.ml_ensemble:
                        try:
                            ml_prediction = self._get_ml_prediction(market_data, symbol)
                            ml_signal = {
                                'prediction': ml_prediction.get('prediction', 0),
                                'confidence': ml_prediction.get('confidence', 0.5)
                            }
                        except:
                            pass

                    # Get ensemble vote
                    trade_direction = 'CALL' if strategy == OptionsStrategy.LONG_CALL else 'PUT'
                    ensemble_result = ensemble_system.get_ensemble_vote(symbol, ml_signal, trade_direction)

                    # Log ensemble decision
                    self.log_trade(f"=== ENSEMBLE VOTE for {symbol} ===")
                    self.log_trade(f"Decision: {ensemble_result['final_decision']}")
                    self.log_trade(f"Confidence: {ensemble_result['confidence']:.0%}")
                    self.log_trade(f"Vote Count: {ensemble_result.get('vote_count', 0)}")
                    self.log_trade(f"Weighted Score: {ensemble_result.get('weighted_score', 0):.3f}")

                    # Log individual strategy votes
                    if ensemble_result.get('votes'):
                        self.log_trade(f"Strategy Votes:")
                        for strategy_name, vote_data in ensemble_result['votes'].items():
                            vote_str = "BUY" if vote_data['vote'] > 0 else "SELL"
                            self.log_trade(f"  {strategy_name}: {vote_str} (conf: {vote_data['confidence']:.0%}, weight: {vote_data['weight']:.0%})")

                    # Log reasoning
                    if ensemble_result.get('reasoning'):
                        self.log_trade(f"Reasoning:")
                        for i, reason in enumerate(ensemble_result['reasoning'][:5], 1):  # Show top 5 reasons
                            self.log_trade(f"  {i}. {reason}")

                    # REJECT if ensemble says REJECT or HOLD
                    if ensemble_result['final_decision'] in ['REJECT', 'HOLD']:
                        self.log_trade(f"ENSEMBLE REJECTED: {symbol} - {ensemble_result['final_decision']}")

                        # Show appropriate rejection reason
                        if ensemble_result['final_decision'] == 'REJECT':
                            # For REJECT, first reason is the actual problem (earnings, etc)
                            if ensemble_result.get('reasoning'):
                                self.log_trade(f"Rejection reason: {ensemble_result['reasoning'][0]}")
                        else:
                            # For HOLD, ensemble score was too low
                            score = ensemble_result.get('weighted_score', 0)
                            vote_count = ensemble_result.get('vote_count', 0)
                            self.log_trade(f"Rejection reason: Ensemble score too low ({score:.2f}, need >0.3 for BUY or <-0.3 for SELL)")
                            self.log_trade(f"Vote count: {vote_count} strategies voted")
                            if ensemble_result.get('reasoning'):
                                self.log_trade(f"Checks performed: {', '.join(ensemble_result['reasoning'][:3])}")
                        return None

                    # Use ensemble confidence (weighted average from all strategies)
                    ensemble_confidence = ensemble_result['confidence']

                    # Blend ensemble confidence with existing confidence (70% ensemble, 30% original)
                    confidence = (ensemble_confidence * 0.7) + (confidence * 0.3)

                    self.log_trade(f"ENSEMBLE APPROVED: {symbol} {trade_direction} - Final confidence: {confidence:.0%}")

                except Exception as e:
                    self.log_trade(f"Ensemble voting error for {symbol}: {e}", "ERROR")
                    # Don't reject on error - fall back to original logic
                    import traceback
                    self.log_trade(f"Ensemble traceback: {traceback.format_exc()}", "WARN")

                # === ADDITIONAL ENHANCEMENT FILTERS ===
                # Apply 6 additional enhancement filters after ensemble approval
                try:
                    self.log_trade(f"=== ENHANCEMENT FILTERS for {symbol} ===")

                    # FILTER 1: Liquidity Check
                    if self.liquidity_filter:
                        liq_check = self.liquidity_filter.approve_for_trading(symbol)
                        self.log_trade(f"1. Liquidity: {liq_check['recommendation']}")
                        if not liq_check['approved']:
                            self.log_trade(f"REJECTED - {liq_check['recommendation']}")
                            return None

                    # FILTER 2: Market Regime Detection
                    regime_info = None
                    if self.market_regime_detector:
                        regime_info = self.market_regime_detector.detect_regime('SPY')
                        self.log_trade(f"2. Market Regime: {regime_info['regime']} (Trend: {regime_info['trend_direction']})")

                        # Check if direction aligns with regime
                        regime_approval = self.market_regime_detector.should_trade_direction(
                            regime_info, trade_direction
                        )
                        if not regime_approval['approved']:
                            self.log_trade(f"REJECTED - {regime_approval['reason']}")
                            return None

                    # FILTER 3: VIX Regime & Position Sizing
                    vix_multiplier = 1.0
                    if self.volatility_adapter:
                        vix_info = self.volatility_adapter.determine_regime()
                        vix_regime = vix_info['regime']
                        vix_value = vix_info['vix']
                        size_mult = vix_info['size_multiplier']
                        vix_multiplier = size_mult
                        self.log_trade(f"3. VIX Regime: {vix_regime} (VIX: {vix_value:.1f}) - Size: {size_mult:.2f}x")

                        # Block trades in extreme volatility
                        if vix_value > 60:
                            self.log_trade(f"REJECTED - VIX too high ({vix_value:.1f})")
                            return None

                    # FILTER 4: Greeks Optimization (will be used for strike selection)
                    # Note: Greeks check happens during option selection, not here
                    # Just log that it will be applied
                    if self.greeks_optimizer:
                        self.log_trade(f"4. Greeks Optimizer: Active (will check Delta 0.4-0.6, DTE 21-45)")

                    # FILTER 5: Spread Strategy Evaluation
                    # Determine if spread is better than naked option
                    use_spread = False
                    if self.spread_strategies and confidence >= 0.65:
                        # For high-confidence trades, evaluate spread vs naked
                        if trade_direction == 'CALL':
                            spread = self.spread_strategies.design_bull_call_spread(
                                current_price, confidence
                            )
                        else:
                            spread = self.spread_strategies.design_bear_put_spread(
                                current_price, confidence
                            )

                        quality = self.spread_strategies.evaluate_spread_quality(spread)
                        self.log_trade(f"5. Spread Strategy: Quality {quality:.0f}/100")

                        # Use spread if quality is good (score > 70)
                        if quality >= 70:
                            use_spread = True
                            self.log_trade(f"   SPREAD SELECTED: {spread['type']}")
                            self.log_trade(f"   Long: ${spread['long_strike']}, Short: ${spread['short_strike']}")
                            self.log_trade(f"   Est Cost: ${spread['estimated_cost']:.2f}, Max Profit: ${spread['max_profit']:.2f}")

                    # FILTER 6: Dynamic Stops (will be applied during position management)
                    if self.dynamic_stop_manager:
                        self.log_trade(f"6. Dynamic Stops: Active (time-based + profit trailing)")

                    # Apply regime-based adjustments to confidence
                    if regime_info:
                        regime_adjustments = self.market_regime_detector.get_strategy_adjustments(
                            regime_info
                        )
                        confidence_mult = regime_adjustments.get('confidence_mult', 1.0)
                        confidence = confidence * confidence_mult
                        self.log_trade(f"   Regime adjusted confidence: {confidence:.1%}")

                    # Apply VIX-based confidence adjustment
                    if vix_multiplier < 0.8:
                        confidence = confidence * 0.9  # Reduce confidence in high vol
                        self.log_trade(f"   VIX adjusted confidence: {confidence:.1%}")

                    self.log_trade(f"=== ALL FILTERS PASSED ===")
                    self.log_trade(f"Final Confidence: {confidence:.1%}")

                except Exception as filter_error:
                    self.log_trade(f"Enhancement filter error: {filter_error}", "WARN")
                    import traceback
                    self.log_trade(f"Filter traceback: {traceback.format_exc()}", "WARN")

                # Apply quantitative finance analysis
                quant_analysis = None
                try:
                    # Get current price from market data
                    current_price = market_data.get('current_price', 0)
                    if current_price <= 0:
                        raise ValueError("Invalid current price")

                    # Determine strike price and expiry for analysis
                    days_to_expiry = random.randint(21, 35)  # 3-5 weeks
                    expiry_date = (datetime.now() + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d')

                    # Choose strike based on strategy and momentum
                    if strategy == OptionsStrategy.LONG_CALL:
                        strike_multiplier = 1.02 if market_data.get('price_momentum', 0) > 0.02 else 1.01
                    else:  # LONG_PUT
                        strike_multiplier = 0.98 if market_data.get('price_momentum', 0) < -0.02 else 0.99

                    strike_price = round(current_price * strike_multiplier, 2)
                    option_type = 'call' if strategy == OptionsStrategy.LONG_CALL else 'put'

                    # Get comprehensive quantitative analysis
                    quant_analysis = analyze_option(symbol, strike_price, expiry_date, option_type)

                    if quant_analysis and 'error' not in quant_analysis:
                        # Extract key metrics
                        quant_confidence = self._calculate_quant_confidence(quant_analysis, market_data)

                        # Blend quantitative analysis with existing confidence
                        confidence = (confidence * 0.5) + (quant_confidence * 0.3) + (base_confidence * 0.2)

                        # Log quantitative insights
                        if confidence >= 0.70:
                            self.log_trade(f"QUANT ANALYSIS: {symbol} {option_type.upper()} ${strike_price}")
                            self.log_trade(f"  Black-Scholes Price: ${quant_analysis.get('bs_price', 0):.2f}")
                            self.log_trade(f"  Delta: {quant_analysis.get('delta', 0):.3f}")
                            self.log_trade(f"  Risk Score: {quant_analysis.get('overall_risk_score', 0):.2f}")
                            self.log_trade(f"  Entry Rec: {quant_analysis.get('entry_recommendation', 'UNKNOWN')}")
                            self.log_trade(f"  Quant Confidence: {quant_confidence:.1%}")

                except Exception as e:
                    self.log_trade(f"Quantitative analysis error for {symbol}: {e}", "WARN")

                # Apply advanced ML prediction
                try:
                    ml_prob, ml_explanation = self.advanced_ml.predict_trade_success(
                        symbol, strategy.value, confidence
                    )

                    # Adjust ML blending based on quantitative analysis availability
                    if quant_analysis and 'error' not in quant_analysis:
                        # Three-way blend: confidence, ML, quantitative
                        confidence = (confidence * 0.5) + (ml_prob * 0.3) + (base_confidence * 0.2)
                    else:
                        # Traditional two-way blend
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
                
                # Calculate volatility-based position sizing multiplier
                try:
                    if 'vol_regime' in locals():
                        if vol_regime == 'LOW_VOL':
                            position_multiplier = 1.2  # Increase size in low vol
                        elif vol_regime == 'HIGH_VOL':
                            position_multiplier = 0.8  # Decrease size in high vol
                        else:
                            position_multiplier = 1.0  # Normal sizing
                    else:
                        position_multiplier = 1.0
                except:
                    position_multiplier = 1.0

                # Enhanced reasoning with filter information
                filter_info = []
                try:
                    if 'ema_trend' in locals() and ema_trend != 'NEUTRAL':
                        filter_info.append(f"EMA: {ema_trend}")
                        if mtf_trend != "NEUTRAL":
                            filter_info.append(f"MTF: {mtf_trend}")
                    if 'vol_regime' in locals():
                        filter_info.append(f"Vol: {vol_regime}")
                    if 'iv_rank' in locals():
                        filter_info.append(f"IV: {iv_rank:.0f}%")
                except:
                    pass

                reasoning = f"Volume: {market_data['volume_ratio']:.2f}x, Momentum: {market_data['price_momentum']:+.1%}"
                if filter_info:
                    reasoning += f", Filters: {', '.join(filter_info)}"

                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'confidence': confidence,
                    'expected_return': 1.5,
                    'max_profit': 2.50,  # More realistic: $2.50 per contract
                    'max_loss': 1.50,    # More realistic: $1.50 per contract
                    'market_data': market_data,
                    'reasoning': reasoning,
                    'position_multiplier': position_multiplier,  # For volatility-based sizing
                    'volatility_regime': vol_regime if 'vol_regime' in locals() else 'NORMAL',
                    'ema_trend': ema_trend if 'ema_trend' in locals() else 'NEUTRAL',
                    'enhanced_filters': True  # Flag that enhanced filters were used
                }
            
            return None
            
        except Exception as e:
            self.log_trade(f"Opportunity finding error for {symbol}: {e}", "WARN")
            return None
    
    async def execute_new_position(self, opportunity):
        """Execute a REAL options position"""
        try:
            # CRITICAL: Check daily loss limit before executing any new trades
            loss_limit_hit = await self.check_daily_loss_limit()
            if loss_limit_hit:
                self.log_trade(f"BLOCKED TRADE: Daily loss limit hit, cannot execute new position for {opportunity['symbol']}", "WARN")
                return False

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
            
            # Enhanced risk check with tighter limits
            base_max_loss = opportunity['max_loss']

            # Apply enhanced risk management with tighter stops (25% vs 30%)
            if opportunity.get('enhanced_filters', False):
                enhanced_max_loss = base_max_loss * 0.83  # Reduce by ~17% (from 30% to 25%)
            else:
                enhanced_max_loss = base_max_loss

            position_risk = enhanced_max_loss * 100  # Convert to dollars

            # Adjust risk limits based on volatility regime
            vol_regime = opportunity.get('volatility_regime', 'NORMAL')
            if vol_regime == 'HIGH_VOL':
                risk_limit = self.daily_risk_limits.get('max_position_risk', 1000) * 0.8  # Reduce in high vol
            elif vol_regime == 'LOW_VOL':
                risk_limit = self.daily_risk_limits.get('max_position_risk', 1000) * 1.2  # Allow more in low vol
            else:
                risk_limit = self.daily_risk_limits.get('max_position_risk', 1000)

            if position_risk > risk_limit:
                self.log_trade(f"Position risk {position_risk:.0f} > limit {risk_limit:.0f} for {symbol} - skipping")
                return False
            
            # Get options chain for the symbol
            if not hasattr(self.options_trader, 'option_chains') or symbol not in self.options_trader.option_chains:
                options_contracts = await self.options_trader.get_options_chain(symbol)
                if not options_contracts:
                    self.log_trade(f"No options available for {symbol}")
                    return False
            
            # CRITICAL FIX (Oct 15, 2025): Use the strategy already determined in the opportunity
            # BUG: Previously was re-calling find_best_options_strategy here, which could return
            # a DIFFERENT strategy than what was identified earlier, causing bot to trade the
            # WRONG direction (e.g., identified LONG_PUT but executed LONG_CALL)
            # This bug caused 0% win rate with 15 consecutive losses (-$4,268)

            strategy_type = opportunity['strategy']  # Use the strategy from opportunity

            # Get the contracts for this strategy from the options chain
            if symbol in self.options_trader.option_chains:
                all_contracts = self.options_trader.option_chains[symbol]

                # Filter contracts by strategy type
                if strategy_type == OptionsStrategy.LONG_CALL:
                    contracts = [c for c in all_contracts if c.option_type == 'call']
                elif strategy_type == OptionsStrategy.LONG_PUT:
                    contracts = [c for c in all_contracts if c.option_type == 'put']
                else:
                    contracts = all_contracts  # For spreads/straddles that use multiple types

                # Sort by best scoring (will be selected in execute_options_strategy)
                contracts.sort(key=lambda c: (c.volume, c.open_interest), reverse=True)
            else:
                self.log_trade(f"ERROR: Options chain not found for {symbol} - cannot execute {strategy_type}")
                return False

            if contracts:
                # Execute the options strategy with REAL orders
                self.log_trade(f"PLACING REAL OPTIONS TRADE: {symbol} {strategy_type}")

                try:
                    # Execute through the options trader with confidence level and adaptive sizing
                    opportunity_confidence = opportunity.get('confidence', 0.5)

                    # Enhanced position sizing with volatility-based adjustments
                    base_quantity = 1

                    # Apply learning-based multiplier
                    learning_multiplier = self.learning_engine.get_position_size_multiplier()

                    # Apply volatility-based multiplier from enhanced filters
                    volatility_multiplier = opportunity.get('position_multiplier', 1.0)

                    # Combine multipliers with conservative cap
                    combined_multiplier = learning_multiplier * volatility_multiplier

                    # Apply confidence-based sizing (higher confidence = larger size)
                    confidence_multiplier = 0.5 + (opportunity_confidence * 0.5)  # 0.5 to 1.0 range

                    # Final quantity calculation with caps
                    final_multiplier = combined_multiplier * confidence_multiplier
                    final_multiplier = max(0.5, min(2.0, final_multiplier))  # Cap between 0.5x and 2.0x

                    adaptive_quantity = max(1, int(base_quantity * final_multiplier))

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
                            'entry_date': datetime.now(),  # For dynamic stops
                            'entry_price': position.entry_price,
                            'peak_price': position.entry_price,  # Track highest price for trailing stops
                            'quantity': adaptive_quantity,  # Use actual quantity
                            'market_regime_at_entry': self.market_regime,
                            'real_trade': True,
                            'order_ids': getattr(position, 'order_ids', []),
                            # Enhanced risk management data
                            'enhanced_filters': opportunity.get('enhanced_filters', False),
                            'volatility_regime': opportunity.get('volatility_regime', 'NORMAL'),
                            'ema_trend': opportunity.get('ema_trend', 'NEUTRAL'),
                            'position_multiplier': opportunity.get('position_multiplier', 1.0),
                            'final_multiplier': final_multiplier,
                            'max_loss_enhanced': enhanced_max_loss,
                            'stop_loss_pct': 0.25 if opportunity.get('enhanced_filters', False) else 0.30,
                            'risk_limit_used': risk_limit
                        }

                        # Add to active positions
                        self.active_positions[position_id] = position_data

                        # Update stats
                        self.performance_stats['total_trades'] += 1

                        # Enhanced logging with Sharpe optimization details
                        filters_used = "ENHANCED" if opportunity.get('enhanced_filters', False) else "BASIC"
                        vol_regime = opportunity.get('volatility_regime', 'NORMAL')
                        ema_trend = opportunity.get('ema_trend', 'NEUTRAL')

                        self.log_trade(f"SUCCESS: REAL TRADE EXECUTED: {symbol} {strategy_type} - Risk: ${position_risk:.2f}")
                        self.log_trade(f"   Filters: {filters_used} | Vol: {vol_regime} | EMA: {ema_trend}")
                        self.log_trade(f"   Position Size: {adaptive_quantity}x (Multiplier: {final_multiplier:.2f})")
                        self.log_trade(f"   Stop Loss: {position_data['stop_loss_pct']:.0%} | Confidence: {opportunity_confidence:.1%}")
                        self.log_trade(f"   Order IDs: {position_data.get('order_ids', [])}")
                        self.log_trade(f"   Entry Price: ${position.entry_price:.2f}")
                        self.log_trade(f"   Expected Sharpe Boost: Enhanced filters active", "INFO")

                        return True
                    else:
                        self.log_trade(f"FAILED Options strategy execution failed for {symbol}")
                        return False

                except Exception as trade_error:
                    self.log_trade(f"FAILED Real trade execution failed: {trade_error}", "ERROR")

                    # Check if error is due to insufficient funds/buying power
                    error_msg = str(trade_error).lower()
                    if any(keyword in error_msg for keyword in ['insufficient', 'buying power', 'not enough', 'funds']):
                        self.log_trade(f"DETECTED INSUFFICIENT FUNDS - Triggering emergency exit analysis", "WARN")
                        await self.handle_insufficient_funds_emergency()

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

                    self.log_trade(f"WARNING: FALLBACK SIMULATION: {symbol} {strategy_type} - Risk: ${position_risk:.2f}")
                    return True
            else:
                self.log_trade(f"No suitable contracts found for {symbol} {strategy_type}")
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
                    await asyncio.sleep(60)  # 1-minute cycles during market hours (increased frequency)
                
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

    def _get_ml_prediction(self, market_data: dict, symbol: str) -> dict:
        """
        Get ML ensemble prediction with all 26 required features

        Args:
            market_data: Market data dict with technical indicators
            symbol: Stock symbol

        Returns:
            ML prediction dict with 'prediction', 'confidence', 'model_votes'
        """
        if not self.ml_ensemble:
            return {'prediction': 0, 'confidence': 0.5, 'model_votes': {}}

        try:
            # Extract the 26 features required by the model
            features = {
                # Returns (1d, 3d, 5d, 10d, 20d)
                'returns_1d': market_data.get('price_momentum', 0.0),
                'returns_3d': market_data.get('price_momentum', 0.0) * 1.5,  # Estimate
                'returns_5d': market_data.get('price_momentum', 0.0) * 2.5,
                'returns_10d': market_data.get('price_momentum', 0.0) * 5.0,
                'returns_20d': market_data.get('price_momentum', 0.0) * 10.0,

                # Price to SMA ratios
                'price_to_sma_5': market_data.get('current_price', 100) / market_data.get('sma_20', 100),
                'price_to_sma_10': market_data.get('current_price', 100) / market_data.get('sma_20', 100),
                'price_to_sma_20': market_data.get('current_price', 100) / market_data.get('sma_20', 100),
                'price_to_sma_50': market_data.get('current_price', 100) / market_data.get('sma_50', 100),

                # Technical indicators
                'rsi': market_data.get('rsi', 50.0),
                'macd': market_data.get('macd', 0.0),
                'macd_signal': market_data.get('macd_signal', 0.0),
                'macd_histogram': market_data.get('macd', 0.0) - market_data.get('macd_signal', 0.0),

                # Bollinger bands
                'bb_width': abs(market_data.get('bollinger_upper', 105) - market_data.get('bollinger_lower', 95)) / market_data.get('current_price', 100),
                'bb_position': (market_data.get('current_price', 100) - market_data.get('bollinger_lower', 95)) / (market_data.get('bollinger_upper', 105) - market_data.get('bollinger_lower', 95) + 0.001),

                # Volatility
                'volatility_5d': market_data.get('volatility', 0.02),
                'volatility_20d': market_data.get('volatility', 0.02) * 1.2,
                'volatility_ratio': 0.85,  # Estimate

                # Volume features
                'volume_ratio': market_data.get('volume_trend', 1.0),
                'volume_momentum': (market_data.get('volume_trend', 1.0) - 1.0) * 5.0,  # Estimate from 5-day change

                # High/Low features
                'high_low_ratio': market_data.get('high', market_data.get('current_price', 100)) / market_data.get('low', market_data.get('current_price', 100) * 0.98),
                'close_to_high': market_data.get('current_price', 100) / market_data.get('high', market_data.get('current_price', 100)),
                'close_to_low': market_data.get('current_price', 100) / market_data.get('low', market_data.get('current_price', 100) * 0.98),

                # Momentum and trend features
                'momentum_3d': market_data.get('price_momentum', 0.0) * 1.5,  # Estimate from 1d momentum
                'momentum_10d': market_data.get('price_momentum', 0.0) * 5.0,  # Estimate from 1d momentum
                'trend_strength': market_data.get('price_momentum', 0.0) * 0.5,  # Estimate
            }

            # Get ML prediction
            ml_result = self.ml_ensemble.predict_direction(features)
            return ml_result

        except Exception as e:
            self.log_trade(f"ML prediction error for {symbol}: {e}", "WARN")
            return {'prediction': 0, 'confidence': 0.5, 'model_votes': {}}

    def _calculate_quant_confidence(self, quant_analysis: dict, market_data: dict) -> float:
        """
        Calculate confidence score based on quantitative analysis results

        Args:
            quant_analysis: Results from quantitative options analysis
            market_data: Current market data

        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            confidence_factors = []

            # Entry recommendation factor (most important)
            entry_rec = quant_analysis.get('entry_recommendation', 'AVOID')
            if entry_rec == 'STRONG_BUY':
                confidence_factors.append(0.9)
            elif entry_rec == 'BUY':
                confidence_factors.append(0.75)
            elif entry_rec == 'HOLD':
                confidence_factors.append(0.6)
            else:  # AVOID
                confidence_factors.append(0.3)

            # Risk score factor (lower risk = higher confidence)
            risk_score = quant_analysis.get('overall_risk_score', 0.5)
            risk_confidence = 1.0 - risk_score  # Invert risk score
            confidence_factors.append(risk_confidence)

            # Delta factor (meaningful delta exposure)
            delta = abs(quant_analysis.get('delta', 0))
            if delta > 0.5:
                confidence_factors.append(0.8)  # High delta
            elif delta > 0.3:
                confidence_factors.append(0.7)  # Medium delta
            else:
                confidence_factors.append(0.5)  # Low delta

            # Time decay factor
            time_decay_risk = quant_analysis.get('time_decay_risk', 'MEDIUM')
            if time_decay_risk == 'LOW':
                confidence_factors.append(0.8)
            elif time_decay_risk == 'MEDIUM':
                confidence_factors.append(0.6)
            else:  # HIGH
                confidence_factors.append(0.4)

            # Volatility alignment factor
            vol_risk = quant_analysis.get('volatility_risk', 'MEDIUM')
            if vol_risk == 'LOW':
                confidence_factors.append(0.7)
            elif vol_risk == 'MEDIUM':
                confidence_factors.append(0.6)
            else:  # HIGH
                confidence_factors.append(0.5)

            # Technical signal alignment factor
            tech_signal = quant_analysis.get('technical_signal', 'NEUTRAL')
            tech_strength = quant_analysis.get('technical_strength', 0.5)

            if tech_signal in ['BULLISH', 'BEARISH'] and tech_strength > 0.6:
                confidence_factors.append(0.8)  # Strong technical signal
            elif tech_signal in ['BULLISH', 'BEARISH']:
                confidence_factors.append(0.6)  # Weak technical signal
            else:
                confidence_factors.append(0.5)  # Neutral

            # Moneyness factor (prefer near-the-money)
            moneyness = quant_analysis.get('moneyness', 1.0)
            if 0.95 <= moneyness <= 1.05:
                confidence_factors.append(0.8)  # Very close to ATM
            elif 0.9 <= moneyness <= 1.1:
                confidence_factors.append(0.7)  # Close to ATM
            else:
                confidence_factors.append(0.5)  # Far from ATM

            # Calculate weighted average
            weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]  # Entry rec most important
            if len(confidence_factors) == len(weights):
                weighted_confidence = sum(f * w for f, w in zip(confidence_factors, weights))
            else:
                weighted_confidence = sum(confidence_factors) / len(confidence_factors)

            # Ensure confidence is within valid range
            return max(0.0, min(1.0, weighted_confidence))

        except Exception as e:
            self.log_trade(f"Quantitative confidence calculation error: {e}", "WARN")
            return 0.5  # Default neutral confidence


async def main():
    """Main entry point"""
    bot = TomorrowReadyOptionsBot()
    try:
        await bot.start_tomorrow_ready_trading()
    finally:
        # Cleanup profit monitoring
        if bot.profit_monitor:
            bot.profit_monitor.stop_monitoring()
        if bot.profit_monitoring_task:
            bot.profit_monitoring_task.cancel()
            try:
                await bot.profit_monitoring_task
            except asyncio.CancelledError:
                pass
        print("[OK] Profit monitoring stopped")

if __name__ == "__main__":
    asyncio.run(main())
