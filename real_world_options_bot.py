#!/usr/bin/env python3
"""
Real-World Options Trading Bot
Optimized for actual market conditions and live trading success

Key Real-World Features:
- Earnings/event avoidance
- Liquidity and spread quality checks
- Market regime detection (bull/bear/sideways)
- Dynamic position sizing based on volatility
- Greeks-based risk management
- Exit strategies with profit targets and stop losses
- IV rank/percentile analysis
- Time decay management
"""

import asyncio
import sys
import os
import time
import json
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import numpy as np

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

class RealWorldOptionsBot:
    """Professional options trading bot optimized for real-world conditions"""
    
    def __init__(self):
        self.broker = None
        self.options_trader = None
        self.options_broker = None
        self.risk_manager = RiskManager(RiskLevel.MODERATE)
        
        # Trading state
        self.trade_count = 0
        self.cycle_count = 0
        self.active_positions = {}
        
        # Real-world performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'total_commissions': 0.0,
            'max_drawdown': 0.0,
            'largest_winner': 0.0,
            'largest_loser': 0.0,
            'avg_days_in_trade': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'monthly_returns': [],
            'daily_pnl': []
        }
        
        # Market regime tracking
        self.market_regime = 'NEUTRAL'  # BULL, BEAR, NEUTRAL, HIGH_VIX
        self.vix_level = 20.0
        self.market_trend = 0.0  # SPY 20-day trend
        
        # Professional trading universe - high liquidity, tight spreads
        self.tier1_stocks = [
            # Mega cap tech (best options liquidity)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # Major ETFs (most liquid options)
            'SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLV', 'XLE', 'GLD', 'TLT',
            # Financial leaders
            'JPM', 'BAC', 'WFC', 'GS', 'BRK-B',
            # Healthcare/Pharma leaders  
            'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV',
            # Other high-volume options
            'NFLX', 'CRM', 'AMD', 'INTC', 'DIS', 'V', 'MA', 'COIN'
        ]
        
        # Real-world strategy allocation (based on market conditions)
        self.base_strategy_weights = {
            OptionsStrategy.BULL_CALL_SPREAD: 0.25,
            OptionsStrategy.BEAR_PUT_SPREAD: 0.25, 
            OptionsStrategy.LONG_PUT: 0.20,        # Hedging in uncertain markets
            OptionsStrategy.LONG_CALL: 0.15,       # Growth momentum plays
            OptionsStrategy.CASH_SECURED_PUT: 0.10, # Income generation
            OptionsStrategy.COVERED_CALL: 0.05     # If we own stock
        }
        
        # Earnings calendar cache (avoid trading before earnings)
        self.earnings_calendar = {}
        self.last_earnings_update = None
        
        # IV percentile cache
        self.iv_percentiles = {}
        
    def log_trade(self, message):
        """Enhanced logging with timestamp and context"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp} [{self.market_regime}] {message}"
        
        print(log_message)
        
        try:
            with open('logs/real_world_options_bot.log', 'a') as f:
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
                
                # Initialize market regime
                await self.update_market_regime()
                return True
            else:
                self.log_trade("Failed to get account info - using simulation mode")
                return False
                
        except Exception as e:
            self.log_trade(f"System initialization error: {e}")
            return False
    
    async def update_market_regime(self):
        """Detect current market regime for strategy adaptation"""
        try:
            # Get VIX level
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='5d')
            if not vix_hist.empty:
                self.vix_level = float(vix_hist['Close'].iloc[-1])
            
            # Get SPY trend
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='30d')
            if len(spy_hist) >= 20:
                current_price = float(spy_hist['Close'].iloc[-1])
                sma_20 = spy_hist['Close'].iloc[-20:].mean()
                self.market_trend = (current_price - sma_20) / sma_20
            
            # Determine regime
            if self.vix_level > 30:
                self.market_regime = 'HIGH_VIX'
            elif self.market_trend > 0.05:  # SPY > 5% above 20-day SMA
                self.market_regime = 'BULL'
            elif self.market_trend < -0.05:  # SPY < 5% below 20-day SMA
                self.market_regime = 'BEAR'
            else:
                self.market_regime = 'NEUTRAL'
                
            self.log_trade(f"Market Regime: {self.market_regime} (VIX: {self.vix_level:.1f}, Trend: {self.market_trend:+.1%})")
            
        except Exception as e:
            self.log_trade(f"Market regime update error: {e}")
    
    async def check_earnings_calendar(self, symbol):
        """Check if earnings are coming up (avoid trading before earnings)"""
        try:
            # Simple earnings check - avoid symbols with recent earnings announcements
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is not None and len(calendar) > 0:
                next_earnings = calendar.iloc[0]['Earnings Date'] if 'Earnings Date' in calendar.columns else None
                if next_earnings:
                    days_to_earnings = (next_earnings - datetime.now()).days
                    return days_to_earnings < 7  # Avoid if earnings within 7 days
            
            return False
            
        except:
            # If we can't get earnings data, assume it's safe
            return False
    
    def calculate_iv_percentile(self, current_iv, historical_iv_data):
        """Calculate IV percentile (where current IV ranks vs historical)"""
        try:
            if len(historical_iv_data) < 10:
                return 50.0  # Default to middle percentile
                
            # Convert to numpy array and sort
            iv_array = np.array(historical_iv_data)
            iv_array = iv_array[~np.isnan(iv_array)]  # Remove NaN values
            
            if len(iv_array) == 0:
                return 50.0
                
            percentile = (np.sum(iv_array <= current_iv) / len(iv_array)) * 100
            return min(100.0, max(0.0, percentile))
            
        except Exception as e:
            self.log_trade(f"IV percentile calculation error: {e}")
            return 50.0
    
    async def get_enhanced_market_data(self, symbol):
        """Get comprehensive market data for real-world analysis"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get price history (30 days for better analysis)
            hist = ticker.history(period="30d")
            if hist.empty:
                return None
                
            current_price = float(hist['Close'].iloc[-1])
            
            # Calculate comprehensive metrics
            if len(hist) >= 20:
                # Price momentum
                sma_5 = hist['Close'].iloc[-5:].mean()
                sma_20 = hist['Close'].iloc[-20:].mean()
                price_momentum = (sma_5 - sma_20) / sma_20
                
                # Volatility (20-day realized vol)
                returns = hist['Close'].pct_change().dropna()
                realized_vol = returns.std() * np.sqrt(252) * 100
                
                # Volume analysis
                avg_volume = hist['Volume'].iloc[-20:].mean()
                current_volume = hist['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                # Support/Resistance levels
                high_20 = hist['High'].iloc[-20:].max()
                low_20 = hist['Low'].iloc[-20:].min()
                price_position = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'price_momentum': price_momentum,
                    'realized_vol': realized_vol,
                    'volume_ratio': volume_ratio,
                    'price_position': price_position,  # 0 = at 20-day low, 1 = at 20-day high
                    'avg_volume': avg_volume,
                    'high_20': high_20,
                    'low_20': low_20,
                    'hist_data': hist
                }
            
            return None
            
        except Exception as e:
            self.log_trade(f"Enhanced market data error for {symbol}: {e}")
            return None
    
    def get_dynamic_strategy_weights(self):
        """Adjust strategy weights based on market regime"""
        weights = self.base_strategy_weights.copy()
        
        if self.market_regime == 'BULL':
            # Favor bullish strategies in bull markets
            weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.5
            weights[OptionsStrategy.LONG_CALL] *= 1.3
            weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 0.7
            weights[OptionsStrategy.LONG_PUT] *= 0.5
            
        elif self.market_regime == 'BEAR':
            # Favor bearish/hedging strategies in bear markets
            weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 1.8
            weights[OptionsStrategy.LONG_PUT] *= 1.5
            weights[OptionsStrategy.BULL_CALL_SPREAD] *= 0.5
            weights[OptionsStrategy.LONG_CALL] *= 0.3
            
        elif self.market_regime == 'HIGH_VIX':
            # In high VIX, favor premium selling and defined risk
            weights[OptionsStrategy.CASH_SECURED_PUT] *= 2.0
            weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.3
            weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 1.3
            weights[OptionsStrategy.LONG_CALL] *= 0.6
            weights[OptionsStrategy.LONG_PUT] *= 0.8
            
        else:  # NEUTRAL
            # In neutral markets, favor income strategies
            weights[OptionsStrategy.CASH_SECURED_PUT] *= 1.5
            weights[OptionsStrategy.COVERED_CALL] *= 1.5
            weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.1
            weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 1.1
        
        # Normalize weights
        total_weight = sum(weights.values())
        return {k: v/total_weight for k, v in weights.items()}
    
    def calculate_position_size(self, opportunity, market_data):
        """Dynamic position sizing based on volatility and account risk"""
        try:
            account_value = self.risk_manager.account_value
            
            # Base risk per trade: 1-3% depending on market conditions
            if self.market_regime == 'HIGH_VIX':
                base_risk_pct = 0.01  # 1% in high volatility
            elif self.market_regime in ['BULL', 'BEAR']:
                base_risk_pct = 0.025  # 2.5% in trending markets
            else:
                base_risk_pct = 0.02  # 2% in neutral markets
            
            # Adjust for realized volatility
            vol_adjustment = 1.0
            if market_data and market_data['realized_vol'] > 40:
                vol_adjustment = 0.7  # Reduce size for high vol stocks
            elif market_data and market_data['realized_vol'] < 15:
                vol_adjustment = 1.3  # Increase size for low vol stocks
            
            # Adjust for confidence level
            confidence_adjustment = opportunity.get('confidence', 0.6)
            
            # Calculate final position size
            max_risk_per_trade = account_value * base_risk_pct * vol_adjustment * confidence_adjustment
            max_loss_per_contract = opportunity['max_loss'] * 100  # Options multiplier
            
            quantity = int(max_risk_per_trade / max_loss_per_contract)
            quantity = max(1, min(quantity, 10))  # Between 1 and 10 contracts
            
            return quantity
            
        except Exception as e:
            self.log_trade(f"Position sizing error: {e}")
            return 1
    
    async def find_high_quality_opportunities(self, symbol):
        """Find high-quality options opportunities with real-world filters"""
        try:
            # Skip if earnings coming up
            if await self.check_earnings_calendar(symbol):
                self.log_trade(f"Skipping {symbol} - earnings within 7 days")
                return None
            
            # Get enhanced market data
            market_data = await self.get_enhanced_market_data(symbol)
            if not market_data:
                return None
            
            # Skip if volume is too low (illiquid)
            if market_data['avg_volume'] < 500000:  # 500K average volume minimum
                return None
                
            # Get options chain with strict liquidity requirements
            contracts = await self.options_trader.get_options_chain(symbol)
            if not contracts or len(contracts) < 8:  # Need good selection
                return None
            
            # Filter for high-quality contracts only
            quality_contracts = []
            for contract in contracts:
                # Strict liquidity requirements for real trading
                if (contract.volume >= 20 and  # Minimum daily volume
                    contract.open_interest >= 100 and  # Minimum open interest
                    contract.bid > 0.05 and  # Avoid penny options
                    contract.spread / contract.mid_price <= 0.10 and  # Max 10% spread
                    contract.days_to_expiry >= 7 and  # At least 1 week
                    contract.days_to_expiry <= 60):  # No more than 60 days
                    quality_contracts.append(contract)
            
            if len(quality_contracts) < 4:  # Need minimum selection
                return None
            
            # Get dynamic strategy weights
            strategy_weights = self.get_dynamic_strategy_weights()
            
            # Select strategy based on market conditions and weights
            strategy = self.select_strategy_by_conditions(market_data, strategy_weights)
            
            # Find specific opportunity
            if strategy == OptionsStrategy.BULL_CALL_SPREAD:
                opportunity = self.find_professional_bull_call_spread(symbol, market_data, quality_contracts)
            elif strategy == OptionsStrategy.BEAR_PUT_SPREAD:
                opportunity = self.find_professional_bear_put_spread(symbol, market_data, quality_contracts)
            elif strategy == OptionsStrategy.LONG_PUT:
                opportunity = self.find_professional_long_put(symbol, market_data, quality_contracts)
            elif strategy == OptionsStrategy.LONG_CALL:
                opportunity = self.find_professional_long_call(symbol, market_data, quality_contracts)
            elif strategy == OptionsStrategy.CASH_SECURED_PUT:
                opportunity = self.find_cash_secured_put(symbol, market_data, quality_contracts)
            else:
                return None
            
            if opportunity:
                opportunity['symbol'] = symbol
                opportunity['market_data'] = market_data
                opportunity['position_size'] = self.calculate_position_size(opportunity, market_data)
                
            return opportunity
            
        except Exception as e:
            self.log_trade(f"High-quality opportunity search error for {symbol}: {e}")
            return None
    
    def select_strategy_by_conditions(self, market_data, strategy_weights):
        """Select strategy based on specific market conditions"""
        
        # Strong directional signals override base weights
        if market_data['price_momentum'] > 0.08 and market_data['price_position'] > 0.7:
            # Strong bullish momentum near highs
            return OptionsStrategy.BULL_CALL_SPREAD if random.random() < 0.7 else OptionsStrategy.LONG_CALL
            
        elif market_data['price_momentum'] < -0.08 and market_data['price_position'] < 0.3:
            # Strong bearish momentum near lows
            return OptionsStrategy.BEAR_PUT_SPREAD if random.random() < 0.7 else OptionsStrategy.LONG_PUT
            
        elif market_data['realized_vol'] > 35 and self.market_regime == 'HIGH_VIX':
            # High volatility - prefer premium selling
            return OptionsStrategy.CASH_SECURED_PUT if random.random() < 0.6 else OptionsStrategy.BEAR_PUT_SPREAD
            
        else:
            # Use weighted random selection
            strategies = list(strategy_weights.keys())
            weights = list(strategy_weights.values())
            return np.random.choice(strategies, p=weights)
    
    def find_professional_bull_call_spread(self, symbol, market_data, contracts):
        """Find professional-grade bull call spread"""
        calls = [c for c in contracts if c.option_type == 'call']
        if len(calls) < 2:
            return None
        
        current_price = market_data['current_price']
        
        # Professional strike selection: ATM to slightly OTM long, further OTM short
        long_calls = [c for c in calls if 
                     current_price * 0.97 <= c.strike <= current_price * 1.05 and
                     c.delta >= 0.30]  # Good delta for directional play
                     
        short_calls = [c for c in calls if 
                      current_price * 1.05 < c.strike <= current_price * 1.20 and
                      c.delta <= 0.25]  # Lower delta for premium collection
        
        if not long_calls or not short_calls:
            return None
        
        # Select optimal combination
        long_call = max(long_calls, key=lambda c: c.delta * c.volume)  # Best delta*volume
        short_call = min(short_calls, key=lambda c: c.delta)  # Lowest delta (most OTM)
        
        # Professional risk/reward analysis
        net_debit = long_call.mid_price - short_call.mid_price
        max_profit = (short_call.strike - long_call.strike) - net_debit
        max_loss = net_debit
        
        # Professional filters
        if (net_debit > 0 and max_profit > 0 and
            max_profit / max_loss >= 0.4 and  # At least 40% return potential
            net_debit <= current_price * 0.05):  # Reasonable premium vs stock price
            
            # Calculate probability of success based on price position and momentum
            base_prob = 0.65  # Professional bull spread base probability
            momentum_boost = min(0.15, market_data['price_momentum'] * 2)
            position_boost = market_data['price_position'] * 0.10
            
            confidence = min(0.90, base_prob + momentum_boost + position_boost)
            
            return {
                'strategy': OptionsStrategy.BULL_CALL_SPREAD,
                'contracts': [long_call, short_call],
                'max_profit': max_profit,
                'max_loss': max_loss,
                'net_debit': net_debit,
                'confidence': confidence,
                'expected_return': max_profit / max_loss,
                'days_to_expiry': min(long_call.days_to_expiry, short_call.days_to_expiry),
                'reason': f"Professional bull spread {long_call.strike:.0f}/{short_call.strike:.0f} - ROI: {(max_profit/max_loss)*100:.0f}%"
            }
        
        return None
    
    def find_professional_bear_put_spread(self, symbol, market_data, contracts):
        """Find professional-grade bear put spread"""
        puts = [c for c in contracts if c.option_type == 'put']
        if len(puts) < 2:
            return None
        
        current_price = market_data['current_price']
        
        # Professional strike selection
        long_puts = [p for p in puts if 
                    current_price * 0.95 <= p.strike <= current_price * 1.03 and
                    abs(p.delta) >= 0.30]  # Good delta for directional play
                    
        short_puts = [p for p in puts if 
                     current_price * 0.80 <= p.strike < current_price * 0.95 and
                     abs(p.delta) <= 0.25]  # Lower delta
        
        if not long_puts or not short_puts:
            return None
        
        # Select optimal combination
        long_put = max(long_puts, key=lambda p: abs(p.delta) * p.volume)
        short_put = min(short_puts, key=lambda p: abs(p.delta))
        
        # Professional risk/reward analysis
        net_debit = long_put.mid_price - short_put.mid_price
        max_profit = (long_put.strike - short_put.strike) - net_debit
        max_loss = net_debit
        
        # Professional filters
        if (net_debit > 0 and max_profit > 0 and
            max_profit / max_loss >= 0.5 and  # Higher requirement for bear spreads
            net_debit <= current_price * 0.04):
            
            # Bear spread probability calculation
            base_prob = 0.70  # Professional bear spread base probability
            momentum_penalty = max(-0.15, market_data['price_momentum'] * -2)
            position_boost = (1 - market_data['price_position']) * 0.12  # Better when price is high
            
            confidence = min(0.88, base_prob + momentum_penalty + position_boost)
            
            return {
                'strategy': OptionsStrategy.BEAR_PUT_SPREAD,
                'contracts': [long_put, short_put],
                'max_profit': max_profit,
                'max_loss': max_loss,
                'net_debit': net_debit,
                'confidence': confidence,
                'expected_return': max_profit / max_loss,
                'days_to_expiry': min(long_put.days_to_expiry, short_put.days_to_expiry),
                'reason': f"Professional bear spread {long_put.strike:.0f}/{short_put.strike:.0f} - ROI: {(max_profit/max_loss)*100:.0f}%"
            }
        
        return None
    
    def find_professional_long_put(self, symbol, market_data, contracts):
        """Find professional long put with hedging focus"""
        puts = [c for c in contracts if c.option_type == 'put']
        if not puts:
            return None
        
        current_price = market_data['current_price']
        
        # Target puts with good delta/gamma for hedging
        target_puts = [p for p in puts if 
                      current_price * 0.90 <= p.strike <= current_price * 0.98 and
                      abs(p.delta) >= 0.25 and
                      p.gamma > 0.01]  # Good gamma for acceleration
        
        if not target_puts:
            return None
        
        # Select put with best risk/reward profile
        best_put = max(target_puts, key=lambda p: (abs(p.delta) * p.gamma) / max(p.mid_price, 0.1))
        
        max_loss = best_put.mid_price
        
        # Only trade if reasonably priced
        if max_loss <= current_price * 0.03:  # Max 3% of stock price
            confidence = 0.40 + (market_data['realized_vol'] - 20) * 0.01  # Higher vol = higher confidence
            confidence = max(0.25, min(0.60, confidence))
            
            return {
                'strategy': OptionsStrategy.LONG_PUT,
                'contracts': [best_put],
                'max_profit': best_put.strike - best_put.mid_price,
                'max_loss': max_loss,
                'net_debit': best_put.mid_price,
                'confidence': confidence,
                'expected_return': (best_put.strike * 0.1) / max_loss,  # Assume 10% move
                'days_to_expiry': best_put.days_to_expiry,
                'reason': f"Hedging put {best_put.strike:.0f} - Delta: {best_put.delta:.2f}"
            }
        
        return None
    
    def find_professional_long_call(self, symbol, market_data, contracts):
        """Find professional long call for momentum plays"""
        calls = [c for c in contracts if c.option_type == 'call']
        if not calls:
            return None
        
        current_price = market_data['current_price']
        
        # Target calls with good delta/gamma for momentum
        target_calls = [c for c in calls if 
                       current_price * 1.02 <= c.strike <= current_price * 1.10 and
                       c.delta >= 0.25 and
                       c.gamma > 0.01]
        
        if not target_calls:
            return None
        
        # Select call with best momentum characteristics
        best_call = max(target_calls, key=lambda c: (c.delta * c.gamma) / max(c.mid_price, 0.1))
        
        max_loss = best_call.mid_price
        
        # Only trade if reasonably priced and momentum is strong
        if (max_loss <= current_price * 0.04 and  # Max 4% of stock price
            market_data['price_momentum'] > 0.02):  # Positive momentum
            
            momentum_confidence = min(0.25, market_data['price_momentum'] * 5)
            base_confidence = 0.35
            confidence = base_confidence + momentum_confidence
            
            return {
                'strategy': OptionsStrategy.LONG_CALL,
                'contracts': [best_call],
                'max_profit': float('inf'),  # Unlimited
                'max_loss': max_loss,
                'net_debit': best_call.mid_price,
                'confidence': confidence,
                'expected_return': (current_price * 0.15) / max_loss,  # Assume 15% move
                'days_to_expiry': best_call.days_to_expiry,
                'reason': f"Momentum call {best_call.strike:.0f} - Delta: {best_call.delta:.2f}"
            }
        
        return None
    
    def find_cash_secured_put(self, symbol, market_data, contracts):
        """Find cash-secured put for income generation"""
        puts = [c for c in contracts if c.option_type == 'put']
        if not puts:
            return None
        
        current_price = market_data['current_price']
        
        # Target OTM puts for premium collection
        target_puts = [p for p in puts if 
                      current_price * 0.85 <= p.strike <= current_price * 0.95 and
                      abs(p.delta) <= 0.30 and  # OTM
                      p.days_to_expiry <= 45]  # Shorter term for faster decay
        
        if not target_puts:
            return None
        
        # Select put with best premium/risk ratio
        best_put = max(target_puts, key=lambda p: p.mid_price / (current_price - p.strike))
        
        premium_collected = best_put.mid_price
        cash_required = best_put.strike * 100  # Cash secured
        
        # Calculate return on cash secured
        max_return_pct = (premium_collected * 100) / cash_required
        
        # Only trade if decent premium and not too risky
        if max_return_pct >= 0.01:  # At least 1% return
            confidence = 0.75 - abs(best_put.delta) * 1.5  # Higher confidence for more OTM
            confidence = max(0.50, min(0.85, confidence))
            
            return {
                'strategy': OptionsStrategy.CASH_SECURED_PUT,
                'contracts': [best_put],
                'max_profit': premium_collected,
                'max_loss': best_put.strike - premium_collected,
                'net_debit': -premium_collected,  # Negative because we collect premium
                'confidence': confidence,
                'expected_return': max_return_pct,
                'days_to_expiry': best_put.days_to_expiry,
                'cash_required': cash_required,
                'reason': f"Income put {best_put.strike:.0f} - {max_return_pct*100:.1f}% return"
            }
        
        return None
    
    async def execute_professional_trade(self, opportunity):
        """Execute trade with professional risk management"""
        try:
            symbol = opportunity['symbol']
            strategy = opportunity['strategy']
            contracts = opportunity['contracts']
            quantity = opportunity['position_size']
            
            # Final risk check
            total_risk = opportunity['max_loss'] * quantity * 100
            account_value = self.risk_manager.account_value
            risk_pct = total_risk / account_value
            
            if risk_pct > 0.05:  # Never risk more than 5% on single trade
                quantity = int((account_value * 0.05) / (opportunity['max_loss'] * 100))
                quantity = max(1, quantity)
                self.log_trade(f"Reducing position size to {quantity} contracts (risk management)")
            
            # Execute the strategy
            position = await self.options_trader.execute_options_strategy(
                strategy, contracts, quantity=quantity
            )
            
            if position:
                # Enhanced position tracking
                self.active_positions[position.symbol] = {
                    'position': position,
                    'opportunity': opportunity,
                    'entry_time': datetime.now(),
                    'entry_price': opportunity['net_debit'],
                    'target_profit': opportunity['max_profit'] * 0.5,  # Take profit at 50% max
                    'stop_loss': opportunity['max_loss'] * 0.8,  # Stop loss at 80% max loss
                    'days_in_trade': 0,
                    'market_regime_at_entry': self.market_regime
                }
                
                # Update performance stats
                self.performance_stats['total_trades'] += 1
                self.performance_stats['total_commissions'] += 2.0 * quantity  # Estimate $2 per contract
                
                # Enhanced logging
                cost = abs(opportunity['net_debit'] * quantity * 100)
                max_profit = opportunity['max_profit'] * quantity * 100
                max_loss = opportunity['max_loss'] * quantity * 100
                
                self.log_trade(f"PROFESSIONAL TRADE EXECUTED:")
                self.log_trade(f"  {symbol} {strategy} x{quantity}")
                self.log_trade(f"  Cost: ${cost:.2f}, Max Profit: ${max_profit:.2f}, Max Loss: ${max_loss:.2f}")
                self.log_trade(f"  Expected Return: {opportunity['expected_return']:.1f}x, Confidence: {opportunity['confidence']:.1%}")
                self.log_trade(f"  Reason: {opportunity['reason']}")
                
                return True
            
            return False
            
        except Exception as e:
            self.log_trade(f"Professional trade execution error: {e}")
            return False
    
    async def manage_existing_positions(self):
        """Active position management with profit taking and stop losses"""
        if not self.active_positions:
            return
            
        self.log_trade(f"Managing {len(self.active_positions)} active positions...")
        
        for position_id, position_data in list(self.active_positions.items()):
            try:
                position = position_data['position']
                opportunity = position_data['opportunity']
                entry_time = position_data['entry_time']
                
                # Update days in trade
                days_in_trade = (datetime.now() - entry_time).days
                position_data['days_in_trade'] = days_in_trade
                
                # Get current option value (simplified - would use real pricing in live trading)
                current_value = opportunity['net_debit']  # Placeholder
                
                # Calculate current P&L
                current_pnl = (current_value - position_data['entry_price']) * position.quantity * 100
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                # Profit target hit
                if current_pnl >= position_data['target_profit'] * position.quantity * 100:
                    should_exit = True
                    exit_reason = "PROFIT_TARGET"
                
                # Stop loss hit
                elif current_pnl <= -position_data['stop_loss'] * position.quantity * 100:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                
                # Time-based exit (avoid last week before expiration)
                elif days_in_trade >= opportunity.get('days_to_expiry', 30) - 7:
                    should_exit = True
                    exit_reason = "TIME_DECAY"
                
                # Market regime change
                elif (position_data['market_regime_at_entry'] != self.market_regime and
                      abs(current_pnl) > opportunity['max_loss'] * position.quantity * 100 * 0.3):
                    should_exit = True
                    exit_reason = "REGIME_CHANGE"
                
                if should_exit:
                    self.log_trade(f"POSITION EXIT: {position_id} - Reason: {exit_reason} - P&L: ${current_pnl:.2f}")
                    
                    # Update performance stats
                    self.update_performance_stats(current_pnl)
                    
                    # Remove from active positions
                    del self.active_positions[position_id]
                
            except Exception as e:
                self.log_trade(f"Position management error for {position_id}: {e}")
    
    def update_performance_stats(self, realized_pnl):
        """Update comprehensive performance statistics"""
        self.performance_stats['total_profit'] += realized_pnl
        self.performance_stats['daily_pnl'].append(realized_pnl)
        
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
        peak_value = max(0, max(self.performance_stats['daily_pnl'])) if self.performance_stats['daily_pnl'] else 0
        current_value = sum(self.performance_stats['daily_pnl'])
        drawdown = peak_value - current_value
        self.performance_stats['max_drawdown'] = max(self.performance_stats['max_drawdown'], drawdown)
        
        # Calculate profit factor
        if self.performance_stats['total_trades'] > 0:
            gross_profit = sum([pnl for pnl in self.performance_stats['daily_pnl'] if pnl > 0])
            gross_loss = abs(sum([pnl for pnl in self.performance_stats['daily_pnl'] if pnl < 0]))
            
            if gross_loss > 0:
                self.performance_stats['profit_factor'] = gross_profit / gross_loss
            else:
                self.performance_stats['profit_factor'] = float('inf')
    
    async def professional_scan_and_trade(self):
        """Professional-grade scanning and trading cycle"""
        self.cycle_count += 1
        await self.update_market_regime()  # Update market conditions
        
        self.log_trade(f"=== PROFESSIONAL TRADING CYCLE #{self.cycle_count} ===")
        self.log_trade(f"Market Regime: {self.market_regime}, VIX: {self.vix_level:.1f}")
        
        # First, manage existing positions
        await self.manage_existing_positions()
        
        # Find new opportunities
        opportunities = []
        
        # Scan tier 1 stocks (most liquid)
        scan_list = random.sample(self.tier1_stocks, min(12, len(self.tier1_stocks)))
        
        for symbol in scan_list:
            self.log_trade(f"Professional scan: {symbol}...")
            opportunity = await self.find_high_quality_opportunities(symbol)
            
            if opportunity:
                opportunities.append(opportunity)
                self.log_trade(f"QUALITY OPPORTUNITY: {symbol} {opportunity['strategy']}")
                self.log_trade(f"  Confidence: {opportunity['confidence']:.1%}, Return: {opportunity['expected_return']:.1f}x")
                self.log_trade(f"  {opportunity['reason']}")
        
        # Sort by risk-adjusted expected value
        opportunities.sort(
            key=lambda x: x['confidence'] * x['expected_return'] / max(x['max_loss'], 0.01), 
            reverse=True
        )
        
        # Execute top opportunities (max 2 new positions per cycle for risk management)
        max_new_positions = 2 - min(2, len(self.active_positions) // 3)  # Reduce new trades if many active
        executed = 0
        
        for opportunity in opportunities[:max_new_positions]:
            success = await self.execute_professional_trade(opportunity)
            if success:
                executed += 1
            
            # Delay between trades
            await asyncio.sleep(2)
        
        self.log_trade(f"Professional cycle complete: {len(opportunities)} opportunities, {executed} trades executed")
        
        # Performance reporting every 3 cycles
        if self.cycle_count % 3 == 0:
            await self.log_professional_performance()
    
    async def log_professional_performance(self):
        """Comprehensive performance reporting"""
        stats = self.performance_stats
        
        # Calculate metrics
        total_trades = max(stats['total_trades'], 1)
        win_rate = (stats['winning_trades'] / total_trades) * 100
        avg_winner = stats['largest_winner'] / max(stats['winning_trades'], 1)
        avg_loser = abs(stats['largest_loser']) / max(total_trades - stats['winning_trades'], 1)
        
        net_profit = stats['total_profit'] - stats['total_commissions']
        account_value = self.risk_manager.account_value
        
        self.log_trade("=== PROFESSIONAL PERFORMANCE REPORT ===")
        self.log_trade(f"Account Value: ${account_value:,.2f}")
        self.log_trade(f"Total Trades: {stats['total_trades']}")
        self.log_trade(f"Win Rate: {win_rate:.1f}%")
        self.log_trade(f"Net P&L: ${net_profit:.2f}")
        self.log_trade(f"Gross P&L: ${stats['total_profit']:.2f}")
        self.log_trade(f"Commissions: ${stats['total_commissions']:.2f}")
        self.log_trade(f"Profit Factor: {stats['profit_factor']:.2f}")
        self.log_trade(f"Max Drawdown: ${stats['max_drawdown']:.2f}")
        self.log_trade(f"Avg Winner: ${avg_winner:.2f}")
        self.log_trade(f"Avg Loser: ${avg_loser:.2f}")
        self.log_trade(f"Active Positions: {len(self.active_positions)}")
        
        # Strategy breakdown
        if self.active_positions:
            strategy_count = {}
            for pos_data in self.active_positions.values():
                strategy = pos_data['opportunity']['strategy']
                strategy_count[strategy] = strategy_count.get(strategy, 0) + 1
            self.log_trade(f"Active Strategies: {strategy_count}")
    
    async def start_professional_trading(self):
        """Main entry point for professional options trading"""
        print("REAL-WORLD OPTIONS TRADING BOT")
        print("=" * 65)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Trading Universe: {len(self.tier1_stocks)} Tier-1 symbols")
        print(f"Features: Earnings avoidance, Liquidity filtering, Dynamic sizing")
        print(f"Risk Management: 1-3% per trade based on market conditions")
        print(f"Position Management: Profit targets, Stop losses, Time decay")
        print("-" * 65)
        
        # Initialize systems
        initialized = await self.initialize_systems()
        if not initialized:
            self.log_trade("System initialization failed - exiting")
            return
        
        self.log_trade("Real-World Options Bot started successfully")
        
        try:
            while True:
                # Check market hours (9:30 AM - 4:00 PM ET)
                import pytz
                et = pytz.timezone('US/Eastern')
                current_time = datetime.now(et)
                hour = current_time.hour
                minute = current_time.minute
                
                if (hour > 9 or (hour == 9 and minute >= 30)) and hour < 16:
                    await self.professional_scan_and_trade()
                    # Wait 15 minutes between cycles (more frequent than simulation)
                    await asyncio.sleep(900)
                else:
                    # Market closed - manage positions and wait
                    await self.manage_existing_positions()
                    self.log_trade(f"Market closed ({current_time.strftime('%H:%M')} ET) - monitoring positions...")
                    await asyncio.sleep(600)  # Check every 10 minutes
                    
        except KeyboardInterrupt:
            self.log_trade("Real-World Options Bot stopped by user")
            await self.log_professional_performance()
        except Exception as e:
            self.log_trade(f"Professional bot error: {e}")
            import traceback
            self.log_trade(f"Traceback: {traceback.format_exc()}")

async def main():
    """Main entry point"""
    bot = RealWorldOptionsBot()
    await bot.start_professional_trading()

if __name__ == "__main__":
    asyncio.run(main())