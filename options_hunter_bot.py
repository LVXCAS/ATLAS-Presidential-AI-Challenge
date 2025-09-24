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

# Advanced AI/ML components
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

class OptionsHunterBot:
    """Pure options trading bot optimized for Monte Carlo success"""
    
    def __init__(self):
        self.broker = None
        self.options_trader = None
        self.options_broker = None
        self.risk_manager = RiskManager(RiskLevel.MODERATE)
        
        # Advanced AI/ML Intelligence Components
        self.technical_analysis = enhanced_technical_analysis
        self.options_pricing = enhanced_options_pricing
        self.economic_data = economic_data_agent
        self.volatility_intelligence = cboe_data_agent
        self.advanced_technical = advanced_technical_analysis
        self.ml_predictions = ml_prediction_engine
        self.advanced_risk = advanced_risk_manager
        self.dashboard = trading_dashboard
        
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
    
    async def get_enhanced_market_intelligence(self, symbol):
        """Get comprehensive market intelligence from all advanced systems"""
        try:
            # Get current market data
            stock_data = yf.download(symbol, period="30d", interval="1d")
            if stock_data.empty:
                return None
            
            current_price = float(stock_data['Close'].iloc[-1])
            
            # Get enhanced technical analysis with advanced indicators
            technical_analysis = await self.technical_analysis.get_comprehensive_analysis(symbol, period="60d")
            
            # Get ML predictions for price and volatility
            ml_price_pred = await self.ml_predictions.get_price_prediction(symbol, horizon_days=5)
            ml_vol_pred = await self.ml_predictions.get_volatility_prediction(symbol, horizon_days=10)
            
            # Get economic and volatility intelligence
            economic_data = await self.economic_data.get_comprehensive_economic_analysis()
            vix_data = await self.volatility_intelligence.get_vix_term_structure_analysis()
            
            # Get professional options pricing
            options_pricing = await self.options_pricing.get_comprehensive_pricing_analysis(symbol)
            
            # Create comprehensive intelligence analysis
            intelligence_analysis = {
                'overall_confidence': 0.0,
                'signal_strength': 0.0,
                'market_regime': 'NEUTRAL',
                'volatility_regime': 'NORMAL',
                'economic_regime': economic_data.get('market_regime', 'NEUTRAL'),
                'recommendation': 'HOLD',
                'risk_level': 'MEDIUM',
                'position_sizing_factor': 1.0
            }
            
            # Technical analysis contribution
            if technical_analysis and technical_analysis.get('signals'):
                tech_signals = technical_analysis['signals']
                intelligence_analysis['signal_strength'] += tech_signals.get('signal_strength', 0) * 0.3
                
                # Add volatility regime
                vol_analysis = technical_analysis.get('volatility_analysis', {})
                intelligence_analysis['volatility_regime'] = vol_analysis.get('vol_regime', 'NORMAL')
                
                # Market regime from technical
                regime_detection = technical_analysis.get('regime_detection', {})
                tech_trend = regime_detection.get('trend', 'NEUTRAL')
                if tech_trend != 'NEUTRAL':
                    intelligence_analysis['market_regime'] = tech_trend
            
            # ML predictions contribution
            if ml_price_pred and ml_vol_pred:
                ml_signals = ml_price_pred.get('trading_signals', {})
                model_confidence = ml_price_pred.get('model_confidence', 0)
                
                intelligence_analysis['signal_strength'] += model_confidence * 0.4
                
                # Get ML signal direction
                ml_signal = ml_signals.get('ml_signal', 'HOLD')
                if ml_signal != 'HOLD':
                    intelligence_analysis['recommendation'] = ml_signal
                
                # Volatility regime from ML
                ml_vol_regime = ml_vol_pred.get('volatility_regime_prediction', 'NORMAL')
                if ml_vol_regime == 'HIGH_VOL':
                    intelligence_analysis['volatility_regime'] = 'HIGH'
                elif ml_vol_regime == 'LOW_VOL':
                    intelligence_analysis['volatility_regime'] = 'LOW'
            
            # Economic conditions contribution  
            if economic_data.get('market_regime') == 'CRISIS':
                intelligence_analysis['risk_level'] = 'HIGH'
                intelligence_analysis['position_sizing_factor'] = 0.5  # Reduce position sizes
            elif economic_data.get('market_regime') == 'EXPANSION':
                intelligence_analysis['risk_level'] = 'LOW'
                intelligence_analysis['position_sizing_factor'] = 1.2  # Increase position sizes
            
            # VIX conditions contribution
            if vix_data.get('volatility_regime') == 'HIGH_VOL':
                intelligence_analysis['volatility_regime'] = 'HIGH'
                if intelligence_analysis['risk_level'] == 'MEDIUM':
                    intelligence_analysis['risk_level'] = 'HIGH'
            
            # Options pricing contribution
            if options_pricing and options_pricing.get('implied_volatility_analysis'):
                iv_analysis = options_pricing['implied_volatility_analysis']
                iv_regime = iv_analysis.get('iv_regime', 'NORMAL')
                
                if iv_regime == 'HIGH_IV' and intelligence_analysis['volatility_regime'] != 'HIGH':
                    intelligence_analysis['volatility_regime'] = 'ELEVATED'
            
            # Calculate overall confidence
            base_confidence = intelligence_analysis['signal_strength'] / 100.0
            
            # Adjust confidence based on regime consistency
            if (intelligence_analysis['market_regime'] != 'NEUTRAL' and 
                intelligence_analysis['recommendation'] != 'HOLD'):
                base_confidence *= 1.2  # Boost confidence when signals align
            
            # Adjust for risk conditions
            if intelligence_analysis['risk_level'] == 'HIGH':
                base_confidence *= 0.8  # Reduce confidence in high risk conditions
            elif intelligence_analysis['risk_level'] == 'LOW':
                base_confidence *= 1.1  # Boost confidence in low risk conditions
            
            intelligence_analysis['overall_confidence'] = min(base_confidence, 0.95)  # Cap at 95%
            
            return {
                'current_price': current_price,
                'intelligence_analysis': intelligence_analysis,
                'technical_analysis': technical_analysis,
                'ml_predictions': {
                    'price_prediction': ml_price_pred,
                    'volatility_prediction': ml_vol_pred
                },
                'economic_intelligence': economic_data,
                'volatility_intelligence': vix_data,
                'options_pricing': options_pricing
            }
            
        except Exception as e:
            self.log_trade(f"Enhanced market intelligence error for {symbol}: {e}")
            # Fallback to basic market analysis
            basic_conditions = await self.get_market_conditions(symbol)
            if basic_conditions:
                return {
                    'current_price': stock_data['Close'].iloc[-1],
                    'intelligence_analysis': {
                        'overall_confidence': 0.3,  # Lower confidence for basic analysis
                        'signal_strength': 30.0,
                        'market_regime': 'NEUTRAL',
                        'volatility_regime': 'NORMAL',
                        'recommendation': 'HOLD',
                        'risk_level': 'MEDIUM',
                        'position_sizing_factor': 1.0
                    }
                }
            return None
    
    async def find_best_options_opportunity(self, symbol):
        """Find the best options trading opportunity for a symbol using advanced intelligence"""
        try:
            # Get comprehensive market intelligence
            market_data = await self.get_enhanced_market_intelligence(symbol)
            
            if not market_data:
                return None
                
            current_price = market_data['current_price']
            
            # Use advanced conditions from multiple intelligence sources
            conditions = market_data['intelligence_analysis']
            
            if not conditions or conditions.get('overall_confidence', 0) < 0.3:
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
        """Select strategy based on advanced intelligence and Monte Carlo weights"""
        
        # Start with Monte Carlo optimized weights
        adjusted_weights = self.strategy_weights.copy()
        
        # Advanced intelligence-based adjustments
        market_regime = conditions.get('market_regime', 'NEUTRAL')
        volatility_regime = conditions.get('volatility_regime', 'NORMAL')
        risk_level = conditions.get('risk_level', 'MEDIUM')
        recommendation = conditions.get('recommendation', 'HOLD')
        signal_strength = conditions.get('signal_strength', 0)
        
        # Market regime adjustments
        if market_regime == 'BEARISH' or recommendation == 'SELL':
            adjusted_weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 2.0
            adjusted_weights[OptionsStrategy.LONG_PUT] *= 1.5
            adjusted_weights[OptionsStrategy.BULL_CALL_SPREAD] *= 0.3
            adjusted_weights[OptionsStrategy.LONG_CALL] *= 0.2
            
        elif market_regime == 'BULLISH' or recommendation == 'BUY':
            adjusted_weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.8
            adjusted_weights[OptionsStrategy.LONG_CALL] *= 1.4
            adjusted_weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 0.4
            adjusted_weights[OptionsStrategy.LONG_PUT] *= 0.3
        
        # Volatility regime adjustments
        if volatility_regime in ['HIGH', 'ELEVATED']:
            # High volatility - favor spreads (limited risk)
            adjusted_weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 1.4
            adjusted_weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.4
            adjusted_weights[OptionsStrategy.LONG_PUT] *= 0.7
            adjusted_weights[OptionsStrategy.LONG_CALL] *= 0.7
            
        elif volatility_regime == 'LOW':
            # Low volatility - favor long options (cheaper premiums)
            adjusted_weights[OptionsStrategy.LONG_PUT] *= 1.3
            adjusted_weights[OptionsStrategy.LONG_CALL] *= 1.3
            adjusted_weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 0.8
            adjusted_weights[OptionsStrategy.BULL_CALL_SPREAD] *= 0.8
        
        # Risk level adjustments
        if risk_level == 'HIGH':
            # High risk - strongly favor spreads over long options
            adjusted_weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 1.6
            adjusted_weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.5
            adjusted_weights[OptionsStrategy.LONG_PUT] *= 0.5
            adjusted_weights[OptionsStrategy.LONG_CALL] *= 0.5
            
        elif risk_level == 'LOW':
            # Low risk - can take more directional risk
            adjusted_weights[OptionsStrategy.LONG_PUT] *= 1.2
            adjusted_weights[OptionsStrategy.LONG_CALL] *= 1.2
        
        # Signal strength adjustments
        if signal_strength > 70:
            # Very strong signals - increase directional plays
            if recommendation == 'BUY':
                adjusted_weights[OptionsStrategy.LONG_CALL] *= 1.5
                adjusted_weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.3
            elif recommendation == 'SELL':
                adjusted_weights[OptionsStrategy.LONG_PUT] *= 1.5
                adjusted_weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 1.3
        
        # Fallback to basic conditions if advanced intelligence unavailable
        if not any([market_regime != 'NEUTRAL', recommendation != 'HOLD']):
            # Use basic technical conditions as fallback
            price_change = conditions.get('price_change', 0)
            rsi = conditions.get('rsi', 50)
            volatility = conditions.get('volatility', 20)
            
            # Strong bearish signal
            if price_change < -0.01 and rsi < 40:
                adjusted_weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 1.5
                adjusted_weights[OptionsStrategy.LONG_PUT] *= 1.2
                
            # Strong bullish signal
            elif price_change > 0.01 and rsi > 60:
                adjusted_weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.3
                adjusted_weights[OptionsStrategy.LONG_CALL] *= 1.1
                
            # High volatility
            if volatility > 30:
                adjusted_weights[OptionsStrategy.BEAR_PUT_SPREAD] *= 1.2
                adjusted_weights[OptionsStrategy.BULL_CALL_SPREAD] *= 1.2
                adjusted_weights[OptionsStrategy.LONG_PUT] *= 0.8
                adjusted_weights[OptionsStrategy.LONG_CALL] *= 0.8
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight <= 0:
            return OptionsStrategy.BEAR_PUT_SPREAD  # Safe fallback
            
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
        """Execute the options trade with advanced risk management"""
        try:
            symbol = opportunity['symbol']
            strategy = opportunity['strategy']
            contracts = opportunity['contracts']
            conditions = opportunity.get('conditions', {})
            
            # Advanced risk management - position sizing
            account_value = self.risk_manager.account_value
            base_risk_per_trade = account_value * 0.02  # 2% base risk per trade
            
            # Apply position sizing factor from intelligence analysis
            position_sizing_factor = conditions.get('position_sizing_factor', 1.0)
            risk_level = conditions.get('risk_level', 'MEDIUM')
            signal_strength = conditions.get('signal_strength', 50)
            overall_confidence = conditions.get('overall_confidence', 0.5)
            
            # Adjust risk based on multiple factors
            risk_multiplier = 1.0
            
            # Risk level adjustments
            if risk_level == 'HIGH':
                risk_multiplier *= 0.5  # Reduce position size in high risk conditions
            elif risk_level == 'LOW':
                risk_multiplier *= 1.3  # Increase position size in low risk conditions
            
            # Signal strength adjustments
            if signal_strength > 70 and overall_confidence > 0.7:
                risk_multiplier *= 1.2  # Increase for high confidence signals
            elif signal_strength < 40 or overall_confidence < 0.4:
                risk_multiplier *= 0.7  # Reduce for low confidence signals
            
            # Apply position sizing factor from intelligence
            risk_multiplier *= position_sizing_factor
            
            # Calculate final risk amount
            max_risk_per_trade = base_risk_per_trade * risk_multiplier
            
            # Use advanced risk management for position sizing if available
            try:
                advanced_sizing = await self.advanced_risk.calculate_position_sizing(
                    signal_strength=signal_strength / 100.0,
                    confidence=overall_confidence,
                    volatility=conditions.get('volatility', 25) / 100.0,
                    portfolio_value=account_value,
                    max_position_risk=max_risk_per_trade / account_value
                )
                
                if advanced_sizing and advanced_sizing.get('recommended_size'):
                    recommended_size = advanced_sizing['recommended_size']
                    max_loss_per_contract = opportunity['max_loss'] * 100  # Options multiplier
                    
                    if max_loss_per_contract > 0:
                        quantity = max(1, min(int(recommended_size / max_loss_per_contract), 10))
                    else:
                        quantity = 1
                else:
                    # Fallback to basic calculation
                    max_loss_per_contract = opportunity['max_loss']
                    max_quantity = int(max_risk_per_trade / (max_loss_per_contract * 100))
                    quantity = max(1, min(max_quantity, 5))
                    
            except Exception as e:
                self.log_trade(f"Advanced position sizing failed, using basic: {e}")
                # Fallback to basic calculation
                max_loss_per_contract = opportunity['max_loss']
                max_quantity = int(max_risk_per_trade / (max_loss_per_contract * 100))
                quantity = max(1, min(max_quantity, 5))
            
            # Log risk assessment
            actual_risk = opportunity['max_loss'] * quantity * 100
            risk_percentage = (actual_risk / account_value) * 100
            
            self.log_trade(f"RISK ASSESSMENT: {symbol}")
            self.log_trade(f"  Signal Strength: {signal_strength:.1f}%")
            self.log_trade(f"  Confidence: {overall_confidence:.1%}")
            self.log_trade(f"  Risk Level: {risk_level}")
            self.log_trade(f"  Position Size: {quantity} contracts")
            self.log_trade(f"  Max Risk: ${actual_risk:.2f} ({risk_percentage:.2f}% of account)")
            
            # Execute the strategy
            position = await self.options_trader.execute_options_strategy(
                strategy, contracts, quantity=quantity
            )
            
            if position:
                # Track the position with enhanced data
                self.active_positions[position.symbol] = {
                    'position': position,
                    'opportunity': opportunity,
                    'entry_time': datetime.now(),
                    'risk_assessment': {
                        'signal_strength': signal_strength,
                        'confidence': overall_confidence,
                        'risk_level': risk_level,
                        'max_risk': actual_risk,
                        'risk_percentage': risk_percentage
                    }
                }
                
                # Update performance stats
                self.performance_stats['total_trades'] += 1
                
                # Log successful trade with enhanced information
                cost = opportunity['net_debit'] * quantity * 100
                max_profit = opportunity['max_profit'] * quantity * 100
                
                self.log_trade(f"OPTIONS EXECUTED: {symbol} {strategy}")
                self.log_trade(f"  Quantity: {quantity} contracts")
                self.log_trade(f"  Entry Cost: ${cost:.2f}")
                self.log_trade(f"  Max Profit: ${max_profit:.2f}")
                self.log_trade(f"  Max Loss: ${actual_risk:.2f}")
                self.log_trade(f"  Confidence: {overall_confidence:.1%}")
                self.log_trade(f"  Reason: {opportunity.get('reason', 'N/A')}")
                
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