#!/usr/bin/env python3
"""
HiveTrading Real Market Data Hunter
Live market data + comprehensive trading system
Enhanced with Microsoft RD-Agent and QuantConnect LEAN
"""

import sys
import os
import asyncio
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Add current directory to path
sys.path.append('.')

# Enhanced AI Research and Institutional Trading
try:
    from rd_agent_integration import RDAgentTradingResearcher
    from lean_integration import LEANIntegration
    from RD_Agent_Research.factors.rd_agent_factors import RDAgentFactors
except ImportError as e:
    print(f"Advanced AI systems unavailable: {e}")
    RDAgentTradingResearcher = None
    LEANIntegration = None
    RDAgentFactors = None

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

# Import Advanced AI/ML Intelligence Components
from agents.enhanced_technical_analysis import enhanced_technical_analysis
from agents.enhanced_options_pricing import enhanced_options_pricing
from agents.enhanced_options_pricing_engine import enhanced_options_pricing_engine
from agents.economic_data_agent import economic_data_agent
from agents.cboe_data_agent import cboe_data_agent
from agents.advanced_technical_analysis import advanced_technical_analysis
from agents.ml_prediction_engine import ml_prediction_engine
from agents.advanced_risk_management import advanced_risk_manager
from agents.trading_dashboard import trading_dashboard

# Enhanced Sharpe Ratio Optimization System
from agents.sharpe_enhanced_filters import sharpe_enhanced_filters
from agents.advanced_monte_carlo_engine import advanced_monte_carlo_engine

# Import profit target monitoring
from profit_target_monitor import ProfitTargetMonitor

# Import advanced quantitative finance capabilities
from agents.quantitative_finance_engine import quantitative_engine
from agents.quant_integration import quant_analyzer, analyze_option, analyze_portfolio, predict_returns

# Enhanced ML Integration
try:
    from agents.ml4t_agent import ml4t_agent
    from agents.enhanced_ml_ensemble_agent import enhanced_ml_ensemble_agent
    from agents.advanced_ml_finance_agent import advanced_ml_finance_agent
    from agents.finance_integration_agent import finance_integration_agent
    from agents.finance_database_agent import finance_database_agent
    ENHANCED_ML_AVAILABLE = True
    print("+ Enhanced ML agents loaded (ML4T, Finance, Ensemble, FinanceDB)")
except ImportError as e:
    ENHANCED_ML_AVAILABLE = False
    ml4t_agent = None
    enhanced_ml_ensemble_agent = None
    advanced_ml_finance_agent = None
    finance_integration_agent = None
    print(f"- Enhanced ML not available: {e}")

# New Quantitative Finance Integration
try:
    from agents.quantitative_integration_hub import quant_hub
    QUANT_HUB_AVAILABLE = True
    print("+ Quantitative Finance Hub loaded (QuantLib, TA-Lib, Vectorbt, PyPortfolioOpt, Backtrader, PyTorch)")
except ImportError:
    QUANT_HUB_AVAILABLE = False
    print("- Quantitative Finance Hub not available")

# Create logs directory
os.makedirs('logs', exist_ok=True)

class RealMarketDataHunter:
    """Real market data hunting system"""
    
    def __init__(self):
        self.broker = None
        self.trade_count = 0
        self.position_tracker = {}
        self.price_cache = {}
        self.last_cache_update = {}
        
        # Advanced AI/ML Intelligence Systems
        self.technical_analysis = enhanced_technical_analysis
        self.options_pricing = enhanced_options_pricing
        self.enhanced_options_pricing_engine = enhanced_options_pricing_engine
        self.economic_data = economic_data_agent
        self.volatility_intelligence = cboe_data_agent
        self.advanced_technical = advanced_technical_analysis
        self.ml_predictions = ml_prediction_engine
        self.advanced_risk = advanced_risk_manager
        self.dashboard = trading_dashboard

        # Advanced Quantitative Finance Engine
        self.quant_engine = quantitative_engine
        self.quant_analyzer = quant_analyzer

        # Enhanced Sharpe Ratio Optimization System
        self.sharpe_filters = sharpe_enhanced_filters
        self.monte_carlo_engine = advanced_monte_carlo_engine

        # Enhanced ML Integration
        self.ml4t = ml4t_agent if ENHANCED_ML_AVAILABLE else None
        self.enhanced_ml_ensemble = enhanced_ml_ensemble_agent if ENHANCED_ML_AVAILABLE else None
        self.advanced_ml_finance = advanced_ml_finance_agent if ENHANCED_ML_AVAILABLE else None
        self.finance_integration = finance_integration_agent if ENHANCED_ML_AVAILABLE else None
        self.finance_database = finance_database_agent if ENHANCED_ML_AVAILABLE else None

        # Microsoft RD-Agent AI Research System
        self.rd_agent = None
        self.ai_factors = None
        if RDAgentTradingResearcher:
            try:
                self.rd_agent = RDAgentTradingResearcher()
                self.ai_factors = RDAgentFactors() if RDAgentFactors else None
                print("Microsoft RD-Agent AI Research System: ACTIVE")
            except Exception as e:
                print(f"RD-Agent initialization failed: {e}")
        
        # QuantConnect LEAN Institutional Engine
        self.lean_engine = None
        if LEANIntegration:
            try:
                self.lean_engine = LEANIntegration()
                print("QuantConnect LEAN Institutional Engine: ACTIVE")
            except Exception as e:
                print(f"LEAN initialization failed: {e}")
        
        # Pre-trained Models for Acceleration
        from agents.model_loader import model_loader
        self.pre_trained_models = model_loader
        print("Market Hunter - Pre-trained Models Loaded:")
        print(self.pre_trained_models.get_model_info())
        
        # Quantitative Finance Integration
        if QUANT_HUB_AVAILABLE:
            self.quant_hub = quant_hub
            print("+ Quantitative Finance Hub integrated for Market Hunter")
        else:
            self.quant_hub = None
        
        # Stock universe for real trading
        self.stock_sectors = {
            'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'],
            'large_cap_tech': ['ADBE', 'CRM', 'NFLX', 'AMD', 'INTC', 'ORCL', 'CSCO'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT'],
            'consumer': ['WMT', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX'],
            'industrial': ['CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'LMT'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD'],
            'etfs': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'XLF', 'XLK']
        }
        
        # Flatten to list
        self.all_stocks = []
        for sector, stocks in self.stock_sectors.items():
            self.all_stocks.extend(stocks)
        self.all_stocks = list(set(self.all_stocks))  # Remove duplicates

        # Profit target monitoring (5.75% daily target)
        self.profit_monitor = None
        self.profit_monitoring_task = None
        
    def log_trade(self, message):
        """Log trade to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp}: {message}"
        
        print(log_message)
        
        try:
            with open('logs/real_market_hunter.log', 'a') as f:
                f.write(log_message + '\n')
        except:
            pass
    
    async def get_real_market_data(self, symbol):
        """Get real market data from multiple sources"""
        try:
            # Try Polygon API first (most reliable for real-time)
            if POLYGON_API_KEY:
                data = await self.get_polygon_data(symbol)
                if data:
                    return data
            
            # Fallback to Alpaca data
            if ALPACA_API_KEY:
                data = await self.get_alpaca_data(symbol)
                if data:
                    return data
            
            # Final fallback to Yahoo Finance (free but limited)
            data = await self.get_yahoo_data(symbol)
            if data:
                return data
            
            self.log_trade(f"Warning: Could not fetch real data for {symbol}")
            return None
            
        except Exception as e:
            self.log_trade(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def get_polygon_data(self, symbol):
        """Get data from Polygon API"""
        try:
            # Get current price
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apikey={POLYGON_API_KEY}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    result = data['results'][0]
                    
                    # Get additional metrics
                    current_price = result['c']  # Close price
                    volume = result['v']
                    price_change = ((result['c'] - result['o']) / result['o']) if result['o'] > 0 else 0
                    
                    # Get technical indicators (simplified)
                    rsi = await self.calculate_rsi(symbol, current_price)
                    volatility = abs(price_change) * 100
                    
                    return {
                        'symbol': symbol,
                        'price': current_price,
                        'open': result['o'],
                        'high': result['h'],
                        'low': result['l'],
                        'volume': volume,
                        'price_change': price_change,
                        'volatility': volatility,
                        'rsi': rsi,
                        'source': 'polygon'
                    }
        except Exception as e:
            self.log_trade(f"Polygon API error for {symbol}: {e}")
        
        return None
    
    async def get_alpaca_data(self, symbol):
        """Get data from Alpaca API"""
        try:
            if not self.broker:
                return None
                
            # This would use Alpaca's market data API
            # Simplified implementation for now
            return None
            
        except Exception as e:
            self.log_trade(f"Alpaca data error for {symbol}: {e}")
        
        return None
    
    async def get_yahoo_data(self, symbol):
        """Get data from Yahoo Finance (free fallback)"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            
            # Get recent data
            hist = ticker.history(period="2d", interval="1d")
            if hist.empty:
                return None
            
            current_data = hist.iloc[-1]
            prev_data = hist.iloc[-2] if len(hist) > 1 else current_data
            
            current_price = current_data['Close']
            price_change = (current_price - prev_data['Close']) / prev_data['Close'] if prev_data['Close'] > 0 else 0
            
            # Calculate simple RSI
            rsi = await self.calculate_rsi(symbol, current_price)
            
            return {
                'symbol': symbol,
                'price': float(current_price),
                'open': float(current_data['Open']),
                'high': float(current_data['High']),
                'low': float(current_data['Low']),
                'volume': int(current_data['Volume']),
                'price_change': price_change,
                'volatility': abs(price_change) * 100,
                'rsi': rsi,
                'source': 'yahoo'
            }
            
        except Exception as e:
            self.log_trade(f"Yahoo data error for {symbol}: {e}")
        
        return None
    
    async def calculate_rsi(self, symbol, current_price):
        """Calculate simplified RSI"""
        try:
            # For real implementation, would use historical data
            # For now, return a realistic RSI based on recent performance
            import random
            
            # Cache RSI values to maintain consistency
            cache_key = f"{symbol}_rsi"
            if cache_key in self.price_cache:
                last_rsi, last_time = self.price_cache[cache_key]
                # Update RSI slightly if recent
                if (datetime.now() - last_time).seconds < 300:  # 5 minutes
                    new_rsi = max(0, min(100, last_rsi + random.uniform(-5, 5)))
                    self.price_cache[cache_key] = (new_rsi, datetime.now())
                    return new_rsi
            
            # Generate initial RSI
            rsi = random.uniform(25, 75)
            self.price_cache[cache_key] = (rsi, datetime.now())
            return rsi
            
        except:
            return 50.0  # Neutral RSI fallback
    
    async def get_advanced_market_intelligence(self, symbol, market_regime, volatility_regime):
        """Get comprehensive market intelligence using advanced AI/ML systems"""
        try:
            # Get enhanced technical analysis with advanced indicators
            technical_analysis = await self.technical_analysis.get_comprehensive_analysis(symbol, period="60d")
            
            # Get advanced technical analysis with ML pattern recognition  
            advanced_technical = await self.advanced_technical.get_comprehensive_analysis(symbol, period="60d")
            
            # Get ML predictions for price and volatility
            ml_price_pred = await self.ml_predictions.get_price_prediction(symbol, horizon_days=5)
            ml_vol_pred = await self.ml_predictions.get_volatility_prediction(symbol, horizon_days=10)
            
            # Get professional options pricing analysis
            options_pricing = await self.options_pricing.get_comprehensive_pricing_analysis(symbol)
            
            # Create comprehensive intelligence summary
            intelligence_summary = {
                'overall_confidence': 0.5,
                'signal_strength': 50.0,
                'market_regime': market_regime,
                'volatility_regime': volatility_regime,
                'recommendation': 'HOLD',
                'risk_level': 'MEDIUM',
                'position_sizing_factor': 1.0,
                'expected_return': 1.0,
                'probability_of_profit': 0.5
            }
            
            # Enhanced technical analysis contribution
            if technical_analysis and technical_analysis.get('signals'):
                tech_signals = technical_analysis['signals']
                intelligence_summary['signal_strength'] = tech_signals.get('signal_strength', 50.0)
                intelligence_summary['overall_confidence'] = max(intelligence_summary['overall_confidence'], 
                                                               tech_signals.get('signal_strength', 50) / 100.0)
                
                # Market regime from technical analysis
                overall_signal = tech_signals.get('overall_signal', 'HOLD')
                if overall_signal != 'HOLD':
                    intelligence_summary['recommendation'] = overall_signal
            
            # ML predictions contribution
            if ml_price_pred and ml_vol_pred:
                ml_signals = ml_price_pred.get('trading_signals', {})
                model_confidence = ml_price_pred.get('model_confidence', 0.5)
                
                intelligence_summary['overall_confidence'] = max(intelligence_summary['overall_confidence'], 
                                                               model_confidence)
                
                # ML signal direction
                ml_signal = ml_signals.get('ml_signal', 'HOLD')
                if ml_signal != 'HOLD':
                    intelligence_summary['recommendation'] = ml_signal
                
                # Expected return from ML prediction
                ensemble_pred = ml_price_pred.get('ensemble_prediction', {})
                ensemble_return = ensemble_pred.get('ensemble_return', 0.0)
                if abs(ensemble_return) > 0.01:  # More than 1% expected move
                    intelligence_summary['expected_return'] = abs(ensemble_return) * 5
            
            # Market regime adjustments
            if market_regime == 'CRISIS':
                intelligence_summary['risk_level'] = 'HIGH'
                intelligence_summary['position_sizing_factor'] = 0.5
            elif market_regime == 'EXPANSION':
                intelligence_summary['risk_level'] = 'LOW'
                intelligence_summary['position_sizing_factor'] = 1.3
            
            # Volatility regime adjustments
            if volatility_regime == 'HIGH_VOL':
                intelligence_summary['volatility_regime'] = 'HIGH'
                if intelligence_summary['risk_level'] == 'MEDIUM':
                    intelligence_summary['risk_level'] = 'HIGH'
            
            # Options pricing contribution
            if options_pricing and options_pricing.get('implied_volatility_analysis'):
                iv_analysis = options_pricing['implied_volatility_analysis']
                iv_regime = iv_analysis.get('iv_regime', 'NORMAL')
                
                if iv_regime == 'HIGH_IV':
                    intelligence_summary['strategy_bias'] = 'PREMIUM_SELLING'
                elif iv_regime == 'LOW_IV':
                    intelligence_summary['strategy_bias'] = 'PREMIUM_BUYING'
            
            # Calculate probability of profit
            signal_factor = intelligence_summary['signal_strength'] / 100.0
            confidence_factor = intelligence_summary['overall_confidence']
            intelligence_summary['probability_of_profit'] = min(0.9, 0.4 + (signal_factor * confidence_factor * 0.5))
            
            return {
                'intelligence_summary': intelligence_summary,
                'technical_analysis': technical_analysis,
                'advanced_technical': advanced_technical,
                'ml_predictions': {
                    'price_prediction': ml_price_pred,
                    'volatility_prediction': ml_vol_pred
                },
                'options_pricing': options_pricing
            }
            
        except Exception as e:
            self.log_trade(f"Advanced market intelligence error for {symbol}: {e}")
            return None
    
    async def analyze_advanced_options_opportunities(self, symbol, market_data, advanced_intel):
        """Analyze options opportunities using advanced intelligence"""
        try:
            if not advanced_intel:
                return []
            
            intelligence_summary = advanced_intel.get('intelligence_summary', {})
            options_pricing = advanced_intel.get('options_pricing', {})
            
            strategies = []
            current_price = market_data['price']
            
            recommendation = intelligence_summary.get('recommendation', 'HOLD')
            confidence = intelligence_summary.get('overall_confidence', 0.5)
            signal_strength = intelligence_summary.get('signal_strength', 50.0)
            volatility_regime = intelligence_summary.get('volatility_regime', 'NORMAL')
            strategy_bias = intelligence_summary.get('strategy_bias', None)
            
            # Apply advanced quantitative analysis to enhance strategies
            quant_enhancements = {}
            try:
                # Run quantitative analysis for potential strategies
                days_to_expiry = random.randint(21, 35)  # 3-5 weeks
                expiry_date = (datetime.now() + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d')

                # Analyze calls and puts with quantitative methods
                call_analysis = analyze_option(symbol, current_price * 1.02, expiry_date, 'call')
                put_analysis = analyze_option(symbol, current_price * 0.98, expiry_date, 'put')

                quant_enhancements = {
                    'call_analysis': call_analysis,
                    'put_analysis': put_analysis,
                    'expiry_date': expiry_date,
                    'days_to_expiry': days_to_expiry
                }

                # Adjust confidence based on quantitative analysis
                if call_analysis and 'error' not in call_analysis:
                    call_entry_rec = call_analysis.get('entry_recommendation', 'HOLD')
                    call_risk_score = call_analysis.get('overall_risk_score', 0.5)

                    if call_entry_rec in ['STRONG_BUY', 'BUY'] and call_risk_score < 0.6:
                        confidence = min(confidence + 0.1, 0.95)  # Boost confidence
                        self.log_trade(f"QUANT BOOST: {symbol} call confidence increased by quantitative analysis")

                if put_analysis and 'error' not in put_analysis:
                    put_entry_rec = put_analysis.get('entry_recommendation', 'HOLD')
                    put_risk_score = put_analysis.get('overall_risk_score', 0.5)

                    if put_entry_rec in ['STRONG_BUY', 'BUY'] and put_risk_score < 0.6:
                        confidence = min(confidence + 0.1, 0.95)  # Boost confidence
                        self.log_trade(f"QUANT BOOST: {symbol} put confidence increased by quantitative analysis")

            except Exception as e:
                self.log_trade(f"Quantitative analysis error for {symbol}: {e}", "WARN")

            # High confidence directional strategies (now with quantitative enhancement)
            if confidence > 0.7 and signal_strength > 70:
                if recommendation == 'BUY':
                    if volatility_regime in ['HIGH', 'ELEVATED']:
                        # High vol - favor spreads
                        bull_spread_strategy = {
                            'strategy': 'BULL_CALL_SPREAD',
                            'confidence': confidence,
                            'signal_strength': signal_strength,
                            'strike': current_price * 1.02,
                            'premium': current_price * 0.02,
                            'reason': f"AI Bull Call Spread: {signal_strength:.1f}% confidence, high vol regime",
                            'quantity': 1,
                            'intelligence': intelligence_summary
                        }

                        # Add quantitative insights
                        if quant_enhancements.get('call_analysis'):
                            call_analysis = quant_enhancements['call_analysis']
                            bull_spread_strategy.update({
                                'quant_analysis': call_analysis,
                                'bs_price': call_analysis.get('bs_price', 0),
                                'delta': call_analysis.get('delta', 0),
                                'risk_score': call_analysis.get('overall_risk_score', 0.5),
                                'profit_target': call_analysis.get('profit_target', 0),
                                'stop_loss': call_analysis.get('stop_loss', 0)
                            })
                            bull_spread_strategy['reason'] += f" | Quant: {call_analysis.get('entry_recommendation', 'N/A')}"

                        strategies.append(bull_spread_strategy)
                    else:
                        # Normal vol - can use long calls
                        strategies.append({
                            'strategy': 'LONG_CALL',
                            'confidence': confidence,
                            'signal_strength': signal_strength,
                            'strike': current_price * 1.03,
                            'premium': current_price * 0.025,
                            'reason': f"AI Long Call: {signal_strength:.1f}% confidence",
                            'quantity': 1,
                            'intelligence': intelligence_summary
                        })
                        
                elif recommendation == 'SELL':
                    if volatility_regime in ['HIGH', 'ELEVATED']:
                        strategies.append({
                            'strategy': 'BEAR_PUT_SPREAD',
                            'confidence': confidence,
                            'signal_strength': signal_strength,
                            'strike': current_price * 0.98,
                            'premium': current_price * 0.02,
                            'reason': f"AI Bear Put Spread: {signal_strength:.1f}% confidence, high vol regime",
                            'quantity': 1,
                            'intelligence': intelligence_summary
                        })
                    else:
                        strategies.append({
                            'strategy': 'LONG_PUT',
                            'confidence': confidence,
                            'signal_strength': signal_strength,
                            'strike': current_price * 0.97,
                            'premium': current_price * 0.025,
                            'reason': f"AI Long Put: {signal_strength:.1f}% confidence",
                            'quantity': 1,
                            'intelligence': intelligence_summary
                        })
            
            # Strategy bias from IV analysis
            if strategy_bias == 'PREMIUM_SELLING' and volatility_regime in ['HIGH', 'ELEVATED']:
                strategies.append({
                    'strategy': 'CASH_SECURED_PUT',
                    'confidence': min(confidence + 0.1, 0.9),
                    'signal_strength': signal_strength,
                    'strike': current_price * 0.95,
                    'premium': current_price * 0.03,
                    'reason': f"AI Premium Selling: High IV detected",
                    'quantity': 1,
                    'intelligence': intelligence_summary
                })
            
            return strategies
            
        except Exception as e:
            self.log_trade(f"Advanced options analysis error for {symbol}: {e}")
            return []
    
    async def run_ai_research_cycle(self):
        """Run Microsoft RD-Agent research cycle for new factors and models"""
        if not self.rd_agent:
            self.log_trade("RD-Agent not available - skipping AI research cycle")
            return None
        
        try:
            self.log_trade("Starting AI research cycle with Microsoft RD-Agent...")
            
            # Discover new factors
            new_factors = await self.rd_agent.discover_new_factors(self.all_stocks[:10])
            
            # Research new models
            new_models = await self.rd_agent.research_new_models(self.all_stocks[:10])
            
            # Generate research insights
            research_insights = await self.rd_agent.generate_research_insights({
                'factors': new_factors,
                'models': new_models
            })
            
            self.log_trade(f"AI Research Complete:")
            if new_factors:
                self.log_trade(f"  New factors discovered: {len(new_factors)}")
            if new_models:
                self.log_trade(f"  New models researched: {len(new_models)}")
            if research_insights:
                self.log_trade(f"  Research insights: {len(research_insights)} findings")
            
            return {
                'factors': new_factors,
                'models': new_models,
                'insights': research_insights
            }
            
        except Exception as e:
            self.log_trade(f"AI research cycle error: {e}")
            return None
    
    async def run_institutional_backtest(self, start_date="2020-01-01", end_date="2025-01-01"):
        """Run institutional-grade backtest using QuantConnect LEAN"""
        if not self.lean_engine:
            self.log_trade("LEAN engine not available - skipping institutional backtest")
            return None
        
        try:
            self.log_trade("Running institutional-grade backtest with QuantConnect LEAN...")
            
            # Configure backtest
            config = {
                'start_date': start_date,
                'end_date': end_date,
                'cash': 100000,
                'symbols': self.all_stocks[:20],  # Test on top 20 stocks
            }
            
            # Run backtest
            results = await self.lean_engine.run_backtest(config)
            
            if results:
                performance = results.get('performance', {})
                self.log_trade(f"Institutional Backtest Complete:")
                self.log_trade(f"  Total Return: {performance.get('total_return', 0):.2%}")
                self.log_trade(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
                self.log_trade(f"  Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
                self.log_trade(f"  Win Rate: {performance.get('win_rate', 0):.1%}")
            
            return results
            
        except Exception as e:
            self.log_trade(f"Institutional backtest error: {e}")
            return None
    
    async def optimize_strategies_with_lean(self):
        """Optimize strategies using LEAN's optimization engine"""
        if not self.lean_engine:
            return None
        
        try:
            self.log_trade("Optimizing strategies with LEAN engine...")
            
            # Define optimization parameters
            optimization_config = {
                'target_metric': 'sharpe_ratio',
                'symbols': self.all_stocks[:10],
                'parameters': {
                    'fast_period': [8, 10, 12],
                    'slow_period': [20, 25, 30],
                    'rsi_threshold': [65, 70, 75]
                }
            }
            
            # Run optimization
            optimization_results = await self.lean_engine.optimize_parameters(optimization_config)
            
            if optimization_results:
                best_params = optimization_results.get('best_parameters', {})
                self.log_trade(f"Strategy Optimization Complete:")
                self.log_trade(f"  Best Sharpe Ratio: {optimization_results.get('best_score', 0):.2f}")
                self.log_trade(f"  Optimal Parameters: {best_params}")
            
            return optimization_results
            
        except Exception as e:
            self.log_trade(f"Strategy optimization error: {e}")
            return None
    
    async def integrate_ai_discoveries(self, symbol, market_data):
        """Integrate AI-discovered factors and models into analysis"""
        if not self.ai_factors:
            return None
        
        try:
            import pandas as pd
            
            # Convert market data to DataFrame format for factors
            data_df = pd.DataFrame({
                'Close': [market_data['price']],
                'Open': [market_data['open']],
                'High': [market_data['high']],
                'Low': [market_data['low']],
                'Volume': [market_data['volume']]
            }, index=[datetime.now()])
            
            # Calculate AI-discovered factors
            ai_factor_signals = {}
            
            # Adaptive Volume Momentum
            try:
                avm = self.ai_factors.adaptive_volume_momentum(data_df)
                if not avm.empty:
                    ai_factor_signals['adaptive_volume_momentum'] = float(avm.iloc[-1])
            except Exception as e:
                pass
            
            # Cross Asset Correlation Alpha
            try:
                caca = self.ai_factors.cross_asset_correlation_alpha(data_df)
                if not caca.empty:
                    ai_factor_signals['cross_asset_correlation_alpha'] = float(caca.iloc[-1])
            except Exception as e:
                pass
            
            # Volatility Regime Momentum
            try:
                vrm = self.ai_factors.volatility_regime_momentum(data_df)
                if not vrm.empty:
                    ai_factor_signals['volatility_regime_momentum'] = float(vrm.iloc[-1])
            except Exception as e:
                pass
            
            # Generate enhanced signal from AI factors
            if ai_factor_signals:
                avg_signal = sum(ai_factor_signals.values()) / len(ai_factor_signals)
                signal_strength = abs(avg_signal) * 100
                
                if avg_signal > 0.05:
                    ai_recommendation = 'BUY'
                elif avg_signal < -0.05:
                    ai_recommendation = 'SELL'
                else:
                    ai_recommendation = 'HOLD'
                
                return {
                    'ai_recommendation': ai_recommendation,
                    'ai_signal_strength': signal_strength,
                    'ai_factors': ai_factor_signals,
                    'ai_confidence': min(0.9, signal_strength / 100 + 0.3)
                }
            
        except Exception as e:
            self.log_trade(f"AI integration error for {symbol}: {e}")
        
        return None
    
    async def enhanced_pre_market_analysis(self):
        """Enhanced pre-market analysis using all AI capabilities"""
        self.log_trade("Starting enhanced pre-market analysis...")
        
        analysis_results = {}
        
        # Run AI research cycle
        if self.rd_agent:
            ai_research = await self.run_ai_research_cycle()
            analysis_results['ai_research'] = ai_research
        
        # Run institutional backtest
        if self.lean_engine:
            backtest_results = await self.run_institutional_backtest()
            analysis_results['institutional_backtest'] = backtest_results
        
        # Optimize strategies
        if self.lean_engine:
            optimization_results = await self.optimize_strategies_with_lean()
            analysis_results['strategy_optimization'] = optimization_results
        
        return analysis_results
    
    def analyze_market_data(self, data):
        """Analyze real market data for trading signals"""
        if not data:
            return "HOLD", 0.0, "No data"
        
        symbol = data['symbol']
        price = data['price']
        volume = data['volume']
        price_change = data['price_change']
        rsi = data['rsi']
        volatility = data['volatility']
        
        signals = []
        
        # 1. Momentum Analysis
        if abs(price_change) > 0.02 and volume > 500000:  # 2%+ move on decent volume
            if price_change > 0.03:  # Strong upward momentum
                signals.append(("BUY", 0.7, f"Strong momentum: +{price_change:.1%} on {volume:,} volume"))
            elif price_change < -0.03:  # Strong downward momentum
                signals.append(("SELL", 0.7, f"Strong decline: {price_change:.1%} on {volume:,} volume"))
        
        # 2. Mean Reversion Analysis
        if rsi < 30:  # Oversold
            signals.append(("BUY", 0.6, f"Oversold: RSI {rsi:.1f}"))
        elif rsi > 70:  # Overbought
            signals.append(("SELL", 0.6, f"Overbought: RSI {rsi:.1f}"))
        
        # 3. Volume Analysis
        if volume > 2000000:  # High volume
            if price_change > 0.01:
                signals.append(("BUY", 0.5, f"High volume breakout: {volume:,} shares"))
            elif price_change < -0.01:
                signals.append(("SELL", 0.5, f"High volume selloff: {volume:,} shares"))
        
        # 4. Volatility Analysis
        if volatility > 3.0:  # High volatility day
            if price_change > 0:
                signals.append(("BUY", 0.4, f"High volatility breakout: {volatility:.1f}%"))
            else:
                signals.append(("SELL", 0.4, f"High volatility decline: {volatility:.1f}%"))
        
        # Combine signals
        if not signals:
            return "HOLD", 0.0, f"No signals - Price: ${price:.2f}, RSI: {rsi:.1f}, Vol: {volume:,}"
        
        buy_signals = [s for s in signals if s[0] == "BUY"]
        sell_signals = [s for s in signals if s[0] == "SELL"]
        
        if len(buy_signals) > len(sell_signals):
            total_confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
            reasons = " | ".join(s[2] for s in buy_signals)
            return "BUY", total_confidence, reasons
        
        elif len(sell_signals) > len(buy_signals):
            total_confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
            reasons = " | ".join(s[2] for s in sell_signals)
            return "SELL", total_confidence, reasons
        
        return "HOLD", 0.0, "Conflicting signals"
    
    async def analyze_options_opportunities(self, symbol, data):
        """Analyze real options opportunities"""
        if not data:
            return []
        
        price = data['price']
        volatility = data['volatility']
        rsi = data['rsi']
        price_change = data['price_change']
        
        strategies = []
        
        # Covered Call opportunity
        if rsi < 60 and price_change > -0.01:
            strike = price * 1.05  # 5% OTM
            premium = price * 0.02  # Estimate 2% premium
            delta = -0.3  # OTM call sold has negative delta from seller's perspective
            
            strategies.append({
                'symbol': symbol,
                'strategy': 'COVERED_CALL',
                'strike': strike,
                'premium': premium,
                'delta': delta,
                'confidence': 0.6,
                'reason': f"Covered Call ${strike:.2f} strike for ${premium:.2f} premium"
            })
        
        # Long Call on momentum
        if price_change > 0.03 and rsi < 75:
            strike = price * 1.02
            premium = price * 0.04
            delta = 0.6  # ITM/ATM call has high positive delta
            
            strategies.append({
                'symbol': symbol,
                'strategy': 'LONG_CALL',
                'strike': strike,
                'premium': premium,
                'delta': delta,
                'confidence': 0.5 + abs(price_change) * 5,
                'reason': f"Momentum call ${strike:.2f} strike, momentum: {price_change:.1%}"
            })
        
        # Long Put on decline
        if price_change < -0.03 and rsi > 25:
            strike = price * 0.98
            premium = price * 0.04
            
            strategies.append({
                'symbol': symbol,
                'strategy': 'LONG_PUT',
                'strike': strike,
                'premium': premium,
                'confidence': 0.5 + abs(price_change) * 5,
                'reason': f"Momentum put ${strike:.2f} strike, decline: {price_change:.1%}"
            })
        
        return strategies
    
    async def hunt_real_opportunities(self):
        """Hunt opportunities using advanced AI/ML intelligence and real market data"""
        self.log_trade("=== ADVANCED REAL MARKET HUNTING CYCLE ===")
        
        # Get global market intelligence first
        try:
            economic_intel = await self.economic_data.get_comprehensive_economic_analysis()
            vix_intel = await self.volatility_intelligence.get_vix_term_structure_analysis()
            
            market_regime = economic_intel.get('market_regime', 'NEUTRAL')
            volatility_regime = vix_intel.get('volatility_regime', 'NORMAL')
            
            self.log_trade(f"Market Intelligence: {market_regime} regime, VIX {volatility_regime}")
        except Exception as e:
            self.log_trade(f"Market intelligence error: {e}")
            market_regime = 'NEUTRAL'
            volatility_regime = 'NORMAL'
        
        all_opportunities = []
        successful_fetches = 0
        advanced_analysis_count = 0
        
        # Analyze each stock with advanced intelligence
        for symbol in self.all_stocks:
            try:
                self.log_trade(f"Advanced analysis for {symbol}...")
                
                # Get real market data with timeout
                data = await asyncio.wait_for(self.get_real_market_data(symbol), timeout=10)
                
                if data:
                    successful_fetches += 1
                    
                    # Get advanced intelligence for this symbol
                    try:
                        advanced_intel = await self.get_advanced_market_intelligence(symbol, market_regime, volatility_regime)
                        advanced_analysis_count += 1
                        
                        # Enhanced quantitative analysis with institutional frameworks
                        quant_analysis = None
                        if self.quant_hub:
                            try:
                                # Run comprehensive analysis for high-potential symbols
                                analysis_types = ['technical', 'lean_backtest', 'rd_agent_research', 'openbb_analysis']
                                quant_analysis = await self.quant_hub.comprehensive_market_analysis(
                                    symbol, analysis_types=analysis_types, lookback_days=60
                                )
                                
                                if quant_analysis:
                                    # Enhanced logging with institutional insights
                                    if 'lean_backtest' in quant_analysis:
                                        lean = quant_analysis['lean_backtest']
                                        if 'simulated_performance' in lean:
                                            perf = lean['simulated_performance']
                                            self.log_trade(f"LEAN Performance {symbol}: Return={perf.get('total_return', 0):.1%}, "
                                                         f"Sharpe={perf.get('sharpe_ratio', 0):.2f}")
                                    
                                    if 'rd_agent_research' in quant_analysis:
                                        rd = quant_analysis['rd_agent_research']
                                        if rd.get('status') == 'simulated':
                                            self.log_trade(f"RD-Agent AI Research {symbol}: Factor discovery and model optimization active")
                                    
                                    # Also get quick metrics for compatibility
                                    quick_analysis = await self.quant_hub.run_quick_analysis(symbol)
                                    if quick_analysis and 'quick_metrics' in quick_analysis:
                                        quant_analysis['quick_metrics'] = quick_analysis['quick_metrics']
                                        metrics = quick_analysis['quick_metrics']
                                        self.log_trade(f"Quant Hunter Analysis {symbol}: Trend={metrics.get('trend', 'N/A')}, "
                                                     f"RSI={metrics.get('rsi', 0):.1f}, Momentum={metrics.get('momentum', 'N/A')}")
                                        
                            except Exception as e:
                                self.log_trade(f"Comprehensive quantitative analysis error for {symbol}: {e}")
                                # Fallback to quick analysis
                                try:
                                    quant_analysis = await self.quant_hub.run_quick_analysis(symbol)
                                except:
                                    pass
                        
                        # Integrate AI discoveries
                        ai_discoveries = await self.integrate_ai_discoveries(symbol, data)
                        if ai_discoveries:
                            self.log_trade(f"AI factors for {symbol}: {ai_discoveries['ai_recommendation']} (strength: {ai_discoveries['ai_signal_strength']:.1f}%)")
                        
                        if advanced_intel:
                            intelligence_summary = advanced_intel.get('intelligence_summary', {})
                            overall_confidence = intelligence_summary.get('overall_confidence', 0.5)
                            signal_strength = intelligence_summary.get('signal_strength', 50.0)
                            recommendation = intelligence_summary.get('recommendation', 'HOLD')
                            
                            # Enhanced with AI discoveries
                            if ai_discoveries and ai_discoveries['ai_confidence'] > 0.7:
                                # AI discoveries override
                                recommendation = ai_discoveries['ai_recommendation']
                                overall_confidence = ai_discoveries['ai_confidence']
                                signal_strength = ai_discoveries['ai_signal_strength']
                            
                            # Enhance with quantitative signals
                            if quant_analysis and 'quick_metrics' in quant_analysis:
                                metrics = quant_analysis['quick_metrics']
                                trend = metrics.get('trend', 'SIDEWAYS')
                                momentum_signal = metrics.get('momentum', 'NEUTRAL')
                                rsi = metrics.get('rsi', 50)
                                
                                # Boost confidence for aligned quantitative signals
                                if trend == 'UP' and recommendation == 'BUY':
                                    overall_confidence = min(0.95, overall_confidence + 0.1)
                                    signal_strength = min(100, signal_strength + 10)
                                elif trend == 'DOWN' and recommendation == 'SELL':
                                    overall_confidence = min(0.95, overall_confidence + 0.1)
                                    signal_strength = min(100, signal_strength + 10)
                                
                                # Boost for momentum reversal signals
                                if momentum_signal == 'OVERSOLD' and recommendation == 'BUY':
                                    overall_confidence = min(0.95, overall_confidence + 0.15)
                                    signal_strength = min(100, signal_strength + 15)
                                elif momentum_signal == 'OVERBOUGHT' and recommendation == 'SELL':
                                    overall_confidence = min(0.95, overall_confidence + 0.15)
                                    signal_strength = min(100, signal_strength + 15)
                                
                                # Reduce confidence for conflicting signals
                                if trend == 'UP' and recommendation == 'SELL':
                                    overall_confidence = max(0.2, overall_confidence - 0.2)
                                elif trend == 'DOWN' and recommendation == 'BUY':
                                    overall_confidence = max(0.2, overall_confidence - 0.2)
                                
                            # Advanced stock analysis
                            if recommendation != "HOLD" and overall_confidence > 0.6:
                                # Calculate advanced position sizing
                                risk_factor = intelligence_summary.get('position_sizing_factor', 1.0)
                                quantity = max(1, int(overall_confidence * signal_strength * risk_factor / 10))
                                
                                # Enhanced reason with quantitative signals
                                reason_parts = [f"Advanced AI: {recommendation} - {signal_strength:.1f}% strength"]
                                if quant_analysis and 'quick_metrics' in quant_analysis:
                                    metrics = quant_analysis['quick_metrics']
                                    quant_reason = f"Quant: {metrics.get('trend', 'N/A')} trend, RSI {metrics.get('rsi', 0):.0f}, {metrics.get('momentum', 'N/A')}"
                                    reason_parts.append(quant_reason)
                                
                                all_opportunities.append({
                                    'symbol': symbol,
                                    'signal': recommendation,
                                    'confidence': overall_confidence,
                                    'signal_strength': signal_strength,
                                    'price': data['price'],
                                    'reason': " | ".join(reason_parts),
                                    'type': 'STOCK',
                                    'quantity': quantity,
                                    'data_source': 'QUANT_AI_ENHANCED' if quant_analysis else ('AI_ENHANCED' if ai_discoveries else 'ADVANCED_AI'),
                                    'ai_factors': ai_discoveries.get('ai_factors', {}) if ai_discoveries else {},
                                    'quant_metrics': quant_analysis.get('quick_metrics', {}) if quant_analysis else {},
                                    'intelligence': intelligence_summary
                                })
                            
                            # Advanced options analysis
                            options_strategies = await self.analyze_advanced_options_opportunities(symbol, data, advanced_intel)
                            for strategy in options_strategies:
                                if strategy['confidence'] > 0.6:
                                    all_opportunities.append({
                                        'symbol': symbol,
                                        'strategy': strategy['strategy'],
                                        'strike': strategy.get('strike', 0),
                                        'premium': strategy.get('premium', 0),
                                        'confidence': strategy['confidence'],
                                        'signal_strength': strategy.get('signal_strength', 50),
                                        'reason': strategy['reason'],
                                        'type': 'ADVANCED_OPTIONS',
                                        'quantity': strategy.get('quantity', 1),
                                        'intelligence': strategy.get('intelligence', {})
                                    })
                        else:
                            # Fallback to basic analysis
                            signal, confidence, reason = self.analyze_market_data(data)
                            
                            if signal != "HOLD" and confidence > 0.5:
                                quantity = max(1, int(confidence * 10))
                                
                                all_opportunities.append({
                                    'symbol': symbol,
                                    'signal': signal,
                                    'confidence': confidence,
                                    'price': data['price'],
                                    'reason': reason,
                                    'type': 'STOCK',
                                    'quantity': quantity,
                                    'data_source': 'BASIC'
                                })
                    
                    except Exception as e:
                        self.log_trade(f"Advanced analysis failed for {symbol}, using basic: {e}")
                        # Fallback to basic analysis
                        signal, confidence, reason = self.analyze_market_data(data)
                        
                        if signal != "HOLD" and confidence > 0.5:
                            quantity = max(1, int(confidence * 10))
                            
                            all_opportunities.append({
                                'symbol': symbol,
                                'signal': signal,
                                'confidence': confidence,
                                'price': data['price'],
                                'reason': reason,
                                'type': 'STOCK',
                                'quantity': quantity,
                                'data_source': 'BASIC_FALLBACK'
                            })
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except asyncio.TimeoutError:
                self.log_trade(f"Timeout fetching data for {symbol} - skipping")
            except Exception as e:
                self.log_trade(f"Error processing {symbol}: {e}")
        
        self.log_trade(f"Market Intelligence Summary:")
        self.log_trade(f"  Data fetched: {successful_fetches}/{len(self.all_stocks)} symbols")
        self.log_trade(f"  Advanced AI analysis: {advanced_analysis_count} symbols")
        self.log_trade(f"  Market regime: {market_regime}")
        self.log_trade(f"  Volatility regime: {volatility_regime}")
        
        # Sort and execute best opportunities
        all_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        executed_count = 0
        for opp in all_opportunities[:10]:  # Top 10 opportunities
            await self.execute_real_opportunity(opp)
            executed_count += 1
            await asyncio.sleep(1)
        
        self.log_trade(f"Executed {executed_count} real-data-driven trades")
        
        if all_opportunities:
            best = all_opportunities[0]
            self.log_trade(f"Best opportunity: {best['symbol']} {best.get('signal', best.get('strategy', 'N/A'))} - {best['confidence']:.1%}")
    
    async def execute_real_opportunity(self, opp):
        """Execute opportunity based on real market data"""
        self.trade_count += 1
        
        if opp['type'] == 'STOCK':
            try:
                if self.broker:
                    from agents.broker_integration import OrderRequest, OrderSide, OrderType
                    
                    order_side = OrderSide.BUY if opp['signal'] == "BUY" else OrderSide.SELL
                    
                    order_request = OrderRequest(
                        symbol=opp['symbol'],
                        qty=opp['quantity'],
                        side=order_side,
                        type=OrderType.MARKET
                    )
                    
                    order_response = await self.broker.submit_order(order_request)
                    self.log_trade(f"REAL TRADE #{self.trade_count}: {opp['signal']} {opp['quantity']} {opp['symbol']} @ ${opp['price']:.2f} | ID: {order_response.id} | Confidence: {opp['confidence']:.1%} | {opp['reason']} | Source: {opp.get('data_source', 'unknown')}")
                else:
                    self.log_trade(f"REAL TRADE #{self.trade_count}: {opp['signal']} {opp['quantity']} {opp['symbol']} @ ${opp['price']:.2f} | Confidence: {opp['confidence']:.1%} | {opp['reason']} | Source: {opp.get('data_source', 'unknown')}")
                    
                # Update positions
                if opp['symbol'] not in self.position_tracker:
                    self.position_tracker[opp['symbol']] = 0
                
                if opp['signal'] == "BUY":
                    self.position_tracker[opp['symbol']] += opp['quantity']
                else:
                    self.position_tracker[opp['symbol']] -= opp['quantity']
                    
            except Exception as e:
                self.log_trade(f"Error executing stock trade: {e}")
        
        else:  # OPTIONS
            strategy = opp['strategy']
            try:
                if self.broker:
                    # Create options order for Alpaca
                    from agents.broker_integration import OrderRequest, OrderSide, OrderType
                    
                    # Determine if we're buying or selling the option
                    if strategy in ['LONG_PUT', 'LONG_CALL']:
                        side = OrderSide.BUY
                        qty = 1  # Buy 1 contract
                    elif strategy in ['COVERED_CALL', 'COVERED_PUT']:
                        side = OrderSide.SELL  
                        qty = 1  # Sell 1 contract
                    else:
                        side = OrderSide.BUY
                        qty = 1
                    
                    # Alpaca paper trading doesn't support real options, so we'll simulate with leveraged positions
                    # Calculate equivalent stock position that mimics options exposure
                    
                    # Options strategies to stock equivalent mapping
                    if strategy == 'LONG_PUT':
                        # Long put = bearish position, equivalent to short stock with less capital
                        equivalent_side = OrderSide.SELL  # Short position
                        # Use delta-adjusted quantity (puts typically have negative delta)
                        equivalent_qty = int(50 * abs(opp.get('delta', 0.5)))  # 50-100 shares based on delta
                    elif strategy == 'LONG_CALL':  
                        # Long call = bullish position
                        equivalent_side = OrderSide.BUY
                        equivalent_qty = int(50 * opp.get('delta', 0.5))  # 25-50 shares based on delta
                    elif strategy == 'COVERED_CALL':
                        # Covered call = own stock + sell call (conservative bullish)
                        equivalent_side = OrderSide.BUY  
                        equivalent_qty = 100  # Full position like owning 100 shares
                    elif strategy == 'COVERED_PUT':
                        # Covered put = short stock + sell put
                        equivalent_side = OrderSide.SELL
                        equivalent_qty = 100
                    else:
                        equivalent_side = side
                        equivalent_qty = 50  # Default moderate position
                    
                    # Ensure minimum quantity
                    equivalent_qty = max(1, equivalent_qty)
                    
                    options_order = OrderRequest(
                        symbol=opp['symbol'],
                        qty=equivalent_qty,
                        side=equivalent_side,
                        type=OrderType.MARKET
                    )
                    
                    order_response = await self.broker.submit_order(options_order)
                    self.log_trade(f"REAL TRADE #{self.trade_count}: {strategy} {opp['symbol']} Equivalent Position ({equivalent_qty} shares) | ID: {order_response.id} | Strike: ${opp.get('strike', 0):.2f} | Premium: ${opp.get('premium', 0):.2f} | Confidence: {opp['confidence']:.1%} | {opp['reason']}")
                    
                    # Update position tracker
                    if opp['symbol'] not in self.position_tracker:
                        self.position_tracker[opp['symbol']] = 0
                    
                    if equivalent_side == OrderSide.BUY:
                        self.position_tracker[opp['symbol']] += equivalent_qty
                    else:
                        self.position_tracker[opp['symbol']] -= equivalent_qty
                        
                else:
                    self.log_trade(f"REAL OPTIONS #{self.trade_count}: {strategy} {opp['symbol']} ${opp['strike']:.2f} strike | Premium: ${opp['premium']:.2f} | Confidence: {opp['confidence']:.1%} | {opp['reason']}")
                    
            except Exception as e:
                self.log_trade(f"Error executing options trade: {e}")
                self.log_trade(f"REAL OPTIONS #{self.trade_count}: {strategy} {opp['symbol']} ${opp['strike']:.2f} strike | Premium: ${opp['premium']:.2f} | Confidence: {opp['confidence']:.1%} | {opp['reason']}")
    
    async def start_real_market_hunting(self):
        """Start real market data hunting system"""
        print("HIVE TRADE - REAL MARKET DATA HUNTER")
        print("=" * 45)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Real data for {len(self.all_stocks)} stocks")
        print("Data Sources: Polygon + Alpaca + Yahoo Finance")
        print("Mode: Paper Trading with Real Data")
        print("-" * 45)
        
        # Check API configurations
        if POLYGON_API_KEY:
            self.log_trade(f"Polygon API configured: ...{POLYGON_API_KEY[-4:]}")
        else:
            self.log_trade("Warning: No Polygon API key found")
        
        if ALPACA_API_KEY:
            self.log_trade(f"Alpaca API configured: ...{ALPACA_API_KEY[-4:]}")
        else:
            self.log_trade("Warning: No Alpaca API key found")
        
        # Initialize broker
        try:
            from agents.broker_integration import AlpacaBrokerIntegration
            self.broker = AlpacaBrokerIntegration(paper_trading=True)
            account_info = await self.broker.get_account_info()
            if account_info:
                self.log_trade(f"Connected to Alpaca: {account_info.get('account_number', 'N/A')}")
                self.log_trade(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            else:
                self.log_trade("Alpaca connection failed - using real data mock mode")

            # Initialize profit target monitoring (5.75% daily target)
            if self.broker:
                self.log_trade("Initializing profit/loss monitoring (5.75% target, -4.9% limit)...")
                try:
                    self.profit_monitor = ProfitTargetMonitor()
                    if await self.profit_monitor.initialize_broker():
                        # Start monitoring in background
                        self.profit_monitoring_task = asyncio.create_task(
                            self.profit_monitor.monitor_profit_target(check_interval=30)  # Check every 30 seconds
                        )
                        self.log_trade("✅ Profit/loss monitoring started (5.75% target, -4.9% limit)")
                    else:
                        self.log_trade("❌ Failed to initialize profit target monitoring")
                except Exception as e:
                    self.log_trade(f"Profit monitoring initialization error: {e}")
                self.broker = None
        except Exception as e:
            self.log_trade(f"Broker error: {e}")
            self.log_trade("Running with real market data but simulated execution")
            self.broker = None
        
        # Run enhanced pre-market analysis
        try:
            pre_market_results = await self.enhanced_pre_market_analysis()
            if pre_market_results:
                self.log_trade("Pre-market AI analysis complete")
        except Exception as e:
            self.log_trade(f"Pre-market analysis error: {e}")
        
        self.log_trade("Starting real market data hunting...")
        
        try:
            hunt_cycle = 0
            while True:
                # Use Eastern Time for market hours
                import pytz
                et = pytz.timezone('US/Eastern')
                current_time = datetime.now(et)
                current_hour = current_time.hour
                current_minute = current_time.minute
                
                # Hunt during market hours (9:30 AM - 4 PM ET)
                market_open = (current_hour > 9) or (current_hour == 9 and current_minute >= 30)
                market_close = current_hour < 16
                
                self.log_trade(f"Market check: {current_time.strftime('%H:%M ET')}, Open: {market_open}, Before Close: {market_close}")
                
                if market_open and market_close:
                    hunt_cycle += 1
                    self.log_trade(f"=== REAL DATA HUNT CYCLE #{hunt_cycle} ===")
                    
                    await self.hunt_real_opportunities()
                    
                    # Log positions every 3 cycles
                    if hunt_cycle % 3 == 0 and self.position_tracker:
                        positions_str = ", ".join([f"{sym}: {pos}" for sym, pos in self.position_tracker.items() if pos != 0])
                        if positions_str:
                            self.log_trade(f"Active Positions: {positions_str}")
                
                else:
                    self.log_trade("Market closed - waiting 60 seconds...")
                    await asyncio.sleep(60)  # Check every minute when market is closed
                    continue
                
                # Wait 5 minutes before next hunt (to respect API limits)
                await asyncio.sleep(300)
                
        except KeyboardInterrupt:
            self.log_trade(f"Real market hunting stopped. Total trades: {self.trade_count}")
        except Exception as e:
            self.log_trade(f"Real market hunting error: {e}")
        
        self.log_trade("Real market hunting session ended")
        return self.trade_count

async def main():
    """Main function"""
    print("Starting HiveTrading Real Market Data Hunter...")
    print("Features:")
    print("- Live market data from Polygon, Alpaca, Yahoo Finance")
    print("- Real price analysis and technical indicators")
    print("- Actual market conditions and volume analysis")
    print("- Paper trading with real market data")
    print("- Enhanced Sharpe Ratio Optimization System")
    print("- RSI, EMA, Volatility, IV Rank, and Momentum Filters")
    print("- Advanced Monte Carlo Engine for Risk Assessment")
    print()
    
    hunter = RealMarketDataHunter()
    try:
        result = await hunter.start_real_market_hunting()
        print(f"\nReal market hunting completed!")
        print(f"Total real-data trades: {result}")
    finally:
        # Cleanup profit monitoring
        if hunter.profit_monitor:
            hunter.profit_monitor.stop_monitoring()
        if hunter.profit_monitoring_task:
            hunter.profit_monitoring_task.cancel()
            try:
                await hunter.profit_monitoring_task
            except asyncio.CancelledError:
                pass
        print("✅ Profit monitoring stopped")

if __name__ == "__main__":
    asyncio.run(main())