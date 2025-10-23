#!/usr/bin/env python3
"""
Enhanced OPTIONS_BOT with Advanced Financial Libraries
Integrates FinancePy, QuantLib, QFin, Statsmodels for superior performance

Key Improvements:
- Advanced volatility forecasting using GARCH models
- Professional options pricing with QuantLib Greeks
- Portfolio optimization using Modern Portfolio Theory
- Market regime detection with statistical models
- Kelly Criterion position sizing
"""

import asyncio
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append('.')

from OPTIONS_BOT import TomorrowReadyOptionsBot
from agents.enhanced_financial_analytics import enhanced_analytics
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf

class EnhancedOptionsBot(TomorrowReadyOptionsBot):
    """
    Enhanced version of OPTIONS_BOT with advanced analytics
    """
    
    def __init__(self):
        super().__init__()
        self.analytics = enhanced_analytics
        self.volatility_cache = {}
        self.regime_cache = {}
        self.last_regime_update = None
        
        # Enhanced performance tracking
        self.advanced_stats = {
            'volatility_predictions': [],
            'pricing_accuracy': [],
            'regime_accuracy': [],
            'kelly_sizing': [],
            'sharpe_ratio_history': []
        }
    
    async def enhanced_market_regime_detection(self) -> Dict[str, Any]:
        """
        Use advanced analytics for market regime detection
        """
        try:
            # Cache regime data for 1 hour
            now = datetime.now()
            if (self.last_regime_update is None or 
                (now - self.last_regime_update).seconds > 3600):
                
                regime_data = await self.analytics.market_regime_detection(['SPY', 'QQQ'])
                self.regime_cache = regime_data
                self.last_regime_update = now
                
                # Update bot's regime variables
                self.market_regime = regime_data['regime']
                self.vix_level = regime_data['vix_level']
                self.market_trend = regime_data.get('trend_signal', 0) * 0.02
                
                self.log_trade(f"Enhanced regime detection: {self.market_regime} "
                             f"(VIX: {self.vix_level:.1f}, Confidence: {regime_data['confidence']:.1%})")
                
                return regime_data
            else:
                return self.regime_cache
                
        except Exception as e:
            self.log_trade(f"Enhanced regime detection error: {e}", "WARN")
            return await super().update_market_regime()
    
    async def enhanced_volatility_forecasting(self, symbol: str) -> float:
        """
        Advanced volatility forecasting for better options pricing
        """
        try:
            # Cache volatility for 30 minutes
            cache_key = f"{symbol}_{datetime.now().strftime('%H:%M')[:4]}"
            
            if cache_key not in self.volatility_cache:
                vol_data = await self.analytics.enhanced_volatility_forecasting(symbol, days_ahead=30)
                predicted_vol = vol_data['predicted_vol']
                confidence = vol_data['confidence']
                
                # Store for performance tracking
                self.advanced_stats['volatility_predictions'].append({
                    'symbol': symbol,
                    'predicted_vol': predicted_vol,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                })
                
                self.volatility_cache[cache_key] = predicted_vol
                
                self.log_trade(f"Enhanced vol forecast {symbol}: {predicted_vol:.1f}% "
                             f"(confidence: {confidence:.1%})")
                
                return predicted_vol
            else:
                return self.volatility_cache[cache_key]
                
        except Exception as e:
            self.log_trade(f"Enhanced volatility forecasting error for {symbol}: {e}", "WARN")
            # Fallback to simple calculation
            return await super().get_enhanced_market_data(symbol).get('realized_vol', 25.0)
    
    async def enhanced_options_pricing(self, symbol: str, strike: float, 
                                     expiry_days: int, option_type: str = 'call') -> Dict[str, float]:
        """
        Professional options pricing with QuantLib
        """
        try:
            # Get current price and volatility
            market_data = await self.get_enhanced_market_data(symbol)
            if not market_data:
                return {}
            
            current_price = market_data['current_price']
            
            # Use enhanced volatility forecasting
            volatility = await self.enhanced_volatility_forecasting(symbol)
            volatility = volatility / 100.0  # Convert to decimal
            
            # Time to expiry
            time_to_expiry = expiry_days / 365.0
            
            # Risk-free rate (could be dynamic)
            risk_free_rate = 0.05
            
            # Get professional pricing
            pricing_data = self.analytics.enhanced_options_pricing(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type=option_type
            )
            
            # Enhanced analysis
            pricing_data['symbol'] = symbol
            pricing_data['implied_volatility'] = volatility * 100
            pricing_data['days_to_expiry'] = expiry_days
            pricing_data['probability_itm'] = self._calculate_probability_itm(
                current_price, strike, volatility, time_to_expiry, option_type
            )
            
            return pricing_data
            
        except Exception as e:
            self.log_trade(f"Enhanced options pricing error for {symbol}: {e}", "WARN")
            return {}
    
    def _calculate_probability_itm(self, S: float, K: float, sigma: float, 
                                 T: float, option_type: str) -> float:
        """Calculate probability of finishing in-the-money"""
        try:
            from scipy.stats import norm
            import math
            
            # Log-normal distribution parameters
            mu = math.log(S) + (0.05 - 0.5 * sigma**2) * T  # Risk-neutral drift
            std_dev = sigma * math.sqrt(T)
            
            if option_type.lower() == 'call':
                # P(S_T > K)
                prob_itm = 1 - norm.cdf(math.log(K), mu, std_dev)
            else:
                # P(S_T < K)
                prob_itm = norm.cdf(math.log(K), mu, std_dev)
            
            return prob_itm
            
        except Exception:
            return 0.5  # Default 50% if calculation fails
    
    async def enhanced_opportunity_analysis(self, symbol: str) -> Optional[Dict]:
        """
        Enhanced opportunity detection with advanced analytics
        """
        try:
            # Get enhanced market data
            market_data = await self.get_enhanced_market_data(symbol)
            if not market_data:
                return None
            
            # Get volatility forecast
            predicted_vol = await self.enhanced_volatility_forecasting(symbol)
            current_vol = market_data.get('realized_vol', 25.0)
            
            # Enhanced criteria with volatility predictions
            volume_ok = market_data['volume_ratio'] > 0.8
            momentum_ok = abs(market_data['price_momentum']) > 0.015
            vol_opportunity = abs(predicted_vol - current_vol) > 3.0  # Vol mismatch opportunity
            
            # Market regime factor
            regime_data = await self.enhanced_market_regime_detection()
            regime_factor = self._get_regime_multiplier(regime_data['regime'])
            
            # Enhanced opportunity scoring
            base_score = 0.5
            if volume_ok:
                base_score += 0.15
            if momentum_ok:
                base_score += 0.20
            if vol_opportunity:
                base_score += 0.25  # Volatility mismatch is valuable for options
            
            # Regime adjustment
            base_score *= regime_factor
            
            # Probability-based confidence
            if market_data['price_momentum'] > 0.015:  # Bullish
                strategy = 'BULL_CALL_SPREAD'
                strike_price = market_data['current_price'] * 1.05  # 5% OTM
            elif market_data['price_momentum'] < -0.015:  # Bearish
                strategy = 'BEAR_PUT_SPREAD'
                strike_price = market_data['current_price'] * 0.95  # 5% OTM
            else:
                strategy = 'BULL_CALL_SPREAD'  # Default
                strike_price = market_data['current_price'] * 1.03  # 3% OTM
            
            # Get enhanced pricing
            pricing_data = await self.enhanced_options_pricing(
                symbol, strike_price, 30, 'call' if 'BULL' in strategy else 'put'
            )
            
            if pricing_data:
                # Probability-adjusted returns
                prob_itm = pricing_data.get('probability_itm', 0.5)
                expected_return = prob_itm * 2.0 + (1 - prob_itm) * (-1.0)  # Simplified
                
                confidence = min(0.95, base_score + prob_itm * 0.2)
                
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'confidence': confidence,
                    'expected_return': expected_return,
                    'max_profit': pricing_data.get('time_value', 2.50),
                    'max_loss': pricing_data.get('time_value', 1.50),
                    'market_data': market_data,
                    'pricing_data': pricing_data,
                    'probability_itm': prob_itm,
                    'volatility_edge': predicted_vol - current_vol,
                    'reasoning': f"Enhanced: Vol edge {predicted_vol-current_vol:+.1f}%, "
                               f"Prob ITM {prob_itm:.1%}, Regime {regime_data['regime']}"
                }
            
            return None
            
        except Exception as e:
            self.log_trade(f"Enhanced opportunity analysis error for {symbol}: {e}", "WARN")
            return await super().find_high_quality_opportunity(symbol)
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """Get confidence multiplier based on market regime"""
        multipliers = {
            'BULL_MARKET': 1.3,
            'HIGH_VOLATILITY': 1.4,
            'BEAR_MARKET': 1.1,
            'NEUTRAL': 1.0,
            'CRISIS': 0.7
        }
        return multipliers.get(regime, 1.0)
    
    def enhanced_position_sizing(self, opportunity: Dict, account_value: float) -> int:
        """
        Kelly Criterion-based position sizing
        """
        try:
            # Get historical performance for Kelly calculation
            win_rate = self.performance_stats.get('winning_trades', 1) / max(
                self.performance_stats.get('total_trades', 1), 1
            )
            
            # Average win/loss from performance history
            total_profit = self.performance_stats.get('total_profit', 0)
            total_trades = max(self.performance_stats.get('total_trades', 1), 1)
            avg_trade = total_profit / total_trades
            
            # Estimate win/loss amounts
            avg_win = max(opportunity.get('max_profit', 2.5), 1.0)
            avg_loss = max(abs(opportunity.get('max_loss', 1.5)), 0.5)
            
            # Kelly fraction
            kelly_fraction = self.analytics.calculate_kelly_criterion(
                win_rate, avg_win, avg_loss
            )
            
            # Adjust for confidence
            confidence = opportunity.get('confidence', 0.6)
            adjusted_kelly = kelly_fraction * confidence
            
            # Position size
            max_risk = account_value * adjusted_kelly
            position_size = max(1, int(max_risk / (avg_loss * 100)))
            
            # Cap at risk limits
            max_allowed = int(self.daily_risk_limits['max_position_risk'] / (avg_loss * 100))
            final_size = min(position_size, max_allowed)
            
            self.advanced_stats['kelly_sizing'].append({
                'kelly_fraction': kelly_fraction,
                'confidence': confidence,
                'position_size': final_size,
                'timestamp': datetime.now()
            })
            
            self.log_trade(f"Enhanced sizing: Kelly {kelly_fraction:.1%}, "
                         f"Confidence {confidence:.1%}, Size {final_size} contracts")
            
            return final_size
            
        except Exception as e:
            self.log_trade(f"Enhanced position sizing error: {e}", "WARN")
            return super().calculate_position_size(opportunity, {})
    
    async def enhanced_performance_analysis(self) -> Dict[str, Any]:
        """
        Advanced performance analytics
        """
        try:
            stats = self.performance_stats.copy()
            
            # Calculate Sharpe ratio
            if len(self.advanced_stats['sharpe_ratio_history']) > 10:
                returns = self.advanced_stats['sharpe_ratio_history']
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                stats['sharpe_ratio'] = sharpe_ratio
            
            # Volatility prediction accuracy
            if len(self.advanced_stats['volatility_predictions']) > 5:
                vol_preds = self.advanced_stats['volatility_predictions']
                avg_confidence = np.mean([p['confidence'] for p in vol_preds])
                stats['vol_prediction_confidence'] = avg_confidence
            
            # Kelly sizing efficiency
            if len(self.advanced_stats['kelly_sizing']) > 5:
                kelly_data = self.advanced_stats['kelly_sizing']
                avg_kelly = np.mean([k['kelly_fraction'] for k in kelly_data])
                stats['average_kelly_fraction'] = avg_kelly
            
            return stats
            
        except Exception as e:
            self.log_trade(f"Enhanced performance analysis error: {e}", "WARN")
            return self.performance_stats
    
    async def scan_for_enhanced_opportunities(self):
        """
        Enhanced opportunity scanning with advanced analytics
        """
        if len(self.active_positions) >= self.daily_risk_limits.get('max_positions', 5):
            self.log_trade("Position limit reached - skipping enhanced scan")
            return
        
        self.log_trade("Enhanced opportunity scanning with advanced analytics...")
        
        opportunities = []
        # UPDATED: Top 80 S&P 500 stocks - Maximum coverage with excellent options liquidity
        scan_symbols = [
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

        for symbol in scan_symbols:  # Scan all 80 stocks
            try:
                opportunity = await self.enhanced_opportunity_analysis(symbol)
                if opportunity and opportunity['confidence'] > 0.6:
                    opportunities.append(opportunity)
                    self.log_trade(f"ENHANCED OPPORTUNITY: {symbol} {opportunity['strategy']} "
                                 f"- {opportunity['confidence']:.0%} confidence, "
                                 f"Vol edge: {opportunity.get('volatility_edge', 0):+.1f}%")
            except Exception as e:
                self.log_trade(f"Enhanced scan error for {symbol}: {e}", "WARN")
        
        # Execute best opportunity with enhanced position sizing
        if opportunities:
            best = max(opportunities, key=lambda x: x['confidence'] * x.get('expected_return', 1))
            await self.execute_enhanced_position(best)
    
    async def execute_enhanced_position(self, opportunity: Dict):
        """
        Execute position with enhanced analytics
        """
        try:
            symbol = opportunity['symbol']
            
            # Enhanced position sizing
            position_size = self.enhanced_position_sizing(opportunity, self.risk_manager.account_value)
            
            if position_size <= 0:
                self.log_trade(f"Enhanced position sizing returned 0 for {symbol} - skipping")
                return False
            
            # Create enhanced position data
            position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            position_data = {
                'entry_time': datetime.now(),
                'opportunity': opportunity,
                'entry_price': opportunity.get('max_profit', 2.50),
                'quantity': position_size,
                'market_regime_at_entry': self.market_regime,
                'volatility_forecast': opportunity.get('volatility_edge', 0),
                'probability_itm': opportunity.get('probability_itm', 0.5),
                'kelly_fraction': self.advanced_stats['kelly_sizing'][-1]['kelly_fraction'] 
                                if self.advanced_stats['kelly_sizing'] else 0.02
            }
            
            # Add to positions
            self.active_positions[position_id] = position_data
            self.performance_stats['total_trades'] += 1
            
            self.log_trade(f"ENHANCED POSITION EXECUTED: {symbol} {opportunity['strategy']} "
                         f"- Size: {position_size}, Kelly: {position_data['kelly_fraction']:.1%}")
            
            return True
            
        except Exception as e:
            self.log_trade(f"Enhanced position execution error: {e}", "ERROR")
            return False

# Test the enhanced bot
async def test_enhanced_bot():
    """Test the enhanced OPTIONS_BOT"""
    print("TESTING ENHANCED OPTIONS_BOT")
    print("=" * 60)
    
    bot = EnhancedOptionsBot()
    
    # Initialize
    await bot.initialize_all_systems()
    print(f"[OK] Enhanced bot initialized - Account: ${bot.risk_manager.account_value:,.2f}")
    
    # Test enhanced regime detection
    regime = await bot.enhanced_market_regime_detection()
    print(f"[OK] Enhanced regime: {regime['regime']} (confidence: {regime.get('confidence', 0):.1%})")
    
    # Test enhanced volatility forecasting
    vol = await bot.enhanced_volatility_forecasting('AAPL')
    print(f"[OK] Enhanced AAPL vol forecast: {vol:.1f}%")
    
    # Test enhanced opportunity analysis
    opportunity = await bot.enhanced_opportunity_analysis('AAPL')
    if opportunity:
        print(f"[OK] Enhanced AAPL opportunity: {opportunity['strategy']} "
              f"({opportunity['confidence']:.0%} confidence)")
        print(f"     Volatility edge: {opportunity.get('volatility_edge', 0):+.1f}%, "
              f"Prob ITM: {opportunity.get('probability_itm', 0):.1%}")
    else:
        print(f"[INFO] No enhanced opportunity found for AAPL")
    
    # Test enhanced opportunity scanning
    await bot.scan_for_enhanced_opportunities()
    
    print(f"\nEnhanced bot active with {len(bot.active_positions)} positions")
    print("ENHANCED OPTIONS_BOT READY!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_bot())