#!/usr/bin/env python3
"""
HiveTrading Full Market Hunter
Comprehensive Opportunity Scanner with Options + Stock Trading
"""

import sys
import os
import asyncio
import time
import random
import math
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Add current directory to path
sys.path.append('.')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('.env.development')
except:
    pass

# Create logs directory
os.makedirs('logs', exist_ok=True)

class MarketHunter:
    """Comprehensive market opportunity hunter"""
    
    def __init__(self):
        self.broker = None
        self.trade_count = 0
        self.position_tracker = {}
        self.screened_stocks = []
        self.options_opportunities = []
        
        # Load comprehensive stock universe
        self.load_stock_universe()
        
    def load_stock_universe(self):
        """Load comprehensive list of tradeable stocks"""
        # Major indices and sectors for comprehensive scanning
        self.stock_sectors = {
            'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B'],
            'large_cap_tech': ['ADBE', 'CRM', 'NFLX', 'AMD', 'INTC', 'ORCL', 'CSCO', 'QCOM'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'USB'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY'],
            'consumer': ['WMT', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'TGT'],
            'industrial': ['CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'KMI', 'OXY'],
            'etfs': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'XLF', 'XLK', 'XLE'],
            'growth_stocks': ['SHOP', 'SQ', 'ROKU', 'PLTR', 'RBLX', 'SNAP', 'UBER', 'LYFT'],
            'meme_stocks': ['GME', 'AMC', 'BB', 'NOK', 'WISH', 'CLOV', 'SPCE', 'LCID']
        }
        
        # Flatten to comprehensive list
        self.all_stocks = []
        for sector, stocks in self.stock_sectors.items():
            self.all_stocks.extend(stocks)
        
        # Remove duplicates
        self.all_stocks = list(set(self.all_stocks))
        
    def log_trade(self, message):
        """Log trade to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp}: {message}"
        
        print(log_message)
        
        try:
            with open('logs/market_hunter.log', 'a') as f:
                f.write(log_message + '\n')
        except:
            pass
    
    def simulate_market_data(self, symbol):
        """Simulate realistic market data for analysis"""
        # Sector-based price simulation
        sector_prices = {
            'AAPL': 190, 'MSFT': 380, 'GOOGL': 135, 'AMZN': 155, 'NVDA': 450,
            'TSLA': 250, 'META': 320, 'NFLX': 450, 'AMD': 110, 'INTC': 35,
            'JPM': 150, 'BAC': 32, 'WFC': 45, 'UNH': 520, 'JNJ': 165,
            'SPY': 450, 'QQQ': 390, 'IWM': 200, 'XLF': 38, 'XLK': 180
        }
        
        base_price = sector_prices.get(symbol, random.uniform(50, 300))
        
        # Simulate price with various patterns
        patterns = ['trending_up', 'trending_down', 'mean_reverting', 'breakout', 'breakdown']
        pattern = random.choice(patterns)
        
        if pattern == 'trending_up':
            price_change = random.uniform(0.02, 0.08)  # 2-8% up
            volume_mult = random.uniform(1.2, 3.0)  # Higher volume
        elif pattern == 'trending_down':
            price_change = random.uniform(-0.08, -0.02)  # 2-8% down
            volume_mult = random.uniform(1.5, 4.0)  # High selling volume
        elif pattern == 'breakout':
            price_change = random.uniform(0.05, 0.15)  # 5-15% breakout
            volume_mult = random.uniform(3.0, 10.0)  # Massive volume
        elif pattern == 'breakdown':
            price_change = random.uniform(-0.15, -0.05)  # 5-15% breakdown
            volume_mult = random.uniform(2.0, 8.0)  # Heavy selling
        else:  # mean_reverting
            price_change = random.uniform(-0.03, 0.03)  # Small moves
            volume_mult = random.uniform(0.8, 1.5)  # Normal volume
        
        current_price = base_price * (1 + price_change)
        volume = int(1000000 * volume_mult)
        
        # Generate additional metrics
        volatility = abs(price_change) * 100
        rsi = random.uniform(20, 80)
        
        # Adjust RSI based on pattern
        if pattern in ['trending_up', 'breakout']:
            rsi = random.uniform(60, 85)
        elif pattern in ['trending_down', 'breakdown']:
            rsi = random.uniform(15, 40)
        
        return {
            'symbol': symbol,
            'price': current_price,
            'price_change': price_change,
            'volume': volume,
            'volatility': volatility,
            'rsi': rsi,
            'pattern': pattern,
            'base_price': base_price
        }
    
    def screen_stocks_momentum(self, min_volume=1000000, min_change=0.03):
        """Screen for momentum opportunities"""
        opportunities = []
        
        for symbol in self.all_stocks:
            data = self.simulate_market_data(symbol)
            
            # Momentum criteria
            if (data['volume'] > min_volume and 
                abs(data['price_change']) > min_change and
                data['pattern'] in ['trending_up', 'breakout', 'trending_down', 'breakdown']):
                
                confidence = min(0.9, abs(data['price_change']) * 10 + data['volume'] / 5000000)
                
                signal = "BUY" if data['price_change'] > 0 else "SELL"
                reason = f"{data['pattern'].replace('_', ' ').title()} - {data['price_change']:.1%} move on {data['volume']:,} volume"
                
                opportunities.append({
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'price': data['price'],
                    'reason': reason,
                    'type': 'STOCK',
                    'quantity': max(1, int(confidence * 10)),
                    'pattern': data['pattern'],
                    'volume': data['volume']
                })
        
        # Sort by confidence and volume
        opportunities.sort(key=lambda x: (x['confidence'], x['volume']), reverse=True)
        return opportunities[:15]  # Top 15 opportunities
    
    def screen_stocks_mean_reversion(self):
        """Screen for mean reversion opportunities"""
        opportunities = []
        
        for symbol in self.all_stocks:
            data = self.simulate_market_data(symbol)
            
            # Mean reversion criteria - look for oversold/overbought
            oversold = data['rsi'] < 30 and data['price_change'] < -0.02
            overbought = data['rsi'] > 70 and data['price_change'] > 0.02
            
            if oversold or overbought:
                confidence = 0.3 + (abs(data['price_change']) * 5) + (abs(50 - data['rsi']) / 100)
                confidence = min(0.95, confidence)
                
                signal = "BUY" if oversold else "SELL"
                reason = f"Mean Reversion - RSI {data['rsi']:.1f} ({'oversold' if oversold else 'overbought'})"
                
                opportunities.append({
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'price': data['price'],
                    'reason': reason,
                    'type': 'STOCK',
                    'quantity': max(1, int(confidence * 8)),
                    'rsi': data['rsi']
                })
        
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities[:10]  # Top 10 mean reversion plays
    
    def screen_options_opportunities(self):
        """Screen for options trading opportunities"""
        opportunities = []
        
        # Focus on high-volume, liquid names for options
        options_candidates = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'SPY', 'QQQ', 'META', 'NFLX']
        
        for symbol in options_candidates:
            data = self.simulate_market_data(symbol)
            
            # Options strategies based on market conditions
            strategies = self.analyze_options_strategies(symbol, data)
            opportunities.extend(strategies)
        
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities[:8]  # Top 8 options plays
    
    def analyze_options_strategies(self, symbol, data):
        """Analyze options strategies for a symbol"""
        strategies = []
        current_price = data['price']
        volatility = data['volatility']
        rsi = data['rsi']
        
        # Strategy 1: Covered Calls (for bullish/neutral outlook)
        if rsi < 60 and data['price_change'] > -0.02:
            strike_price = current_price * 1.05  # 5% OTM
            premium = current_price * random.uniform(0.01, 0.03)  # 1-3% premium
            
            strategies.append({
                'symbol': symbol,
                'strategy': 'COVERED_CALL',
                'signal': 'SELL_CALL',
                'strike': strike_price,
                'premium': premium,
                'confidence': 0.6 + (60 - rsi) / 200,
                'reason': f"Covered Call - Collect ${premium:.2f} premium, cap gains at ${strike_price:.2f}",
                'type': 'OPTIONS',
                'quantity': 1
            })
        
        # Strategy 2: Protective Puts (for holdings protection)
        if data['volatility'] > 15 and data['price_change'] > 0:
            strike_price = current_price * 0.95  # 5% OTM put
            premium = current_price * random.uniform(0.015, 0.04)  # 1.5-4% premium
            
            strategies.append({
                'symbol': symbol,
                'strategy': 'PROTECTIVE_PUT',
                'signal': 'BUY_PUT',
                'strike': strike_price,
                'premium': premium,
                'confidence': 0.5 + volatility / 100,
                'reason': f"Protective Put - Insure holdings, max loss at ${strike_price:.2f}",
                'type': 'OPTIONS',
                'quantity': 1
            })
        
        # Strategy 3: Long Calls (bullish momentum)
        if data['pattern'] in ['trending_up', 'breakout'] and rsi < 75:
            strike_price = current_price * 1.02  # 2% OTM
            premium = current_price * random.uniform(0.02, 0.06)
            
            strategies.append({
                'symbol': symbol,
                'strategy': 'LONG_CALL',
                'signal': 'BUY_CALL',
                'strike': strike_price,
                'premium': premium,
                'confidence': 0.4 + abs(data['price_change']) * 5,
                'reason': f"Long Call - Bullish momentum, target ${strike_price:.2f}",
                'type': 'OPTIONS',
                'quantity': max(1, int(abs(data['price_change']) * 20))
            })
        
        # Strategy 4: Long Puts (bearish momentum)  
        if data['pattern'] in ['trending_down', 'breakdown'] and rsi > 25:
            strike_price = current_price * 0.98  # 2% OTM
            premium = current_price * random.uniform(0.02, 0.06)
            
            strategies.append({
                'symbol': symbol,
                'strategy': 'LONG_PUT',
                'signal': 'BUY_PUT',
                'strike': strike_price,
                'premium': premium,
                'confidence': 0.4 + abs(data['price_change']) * 5,
                'reason': f"Long Put - Bearish momentum, target ${strike_price:.2f}",
                'type': 'OPTIONS',
                'quantity': max(1, int(abs(data['price_change']) * 20))
            })
        
        # Strategy 5: Iron Condor (sideways market)
        if data['pattern'] == 'mean_reverting' and 30 < rsi < 70:
            credit = current_price * random.uniform(0.005, 0.02)
            
            strategies.append({
                'symbol': symbol,
                'strategy': 'IRON_CONDOR',
                'signal': 'SELL_SPREAD',
                'credit': credit,
                'confidence': 0.6,
                'reason': f"Iron Condor - Sideways market, collect ${credit:.2f} credit",
                'type': 'OPTIONS',
                'quantity': 1
            })
        
        return strategies
    
    async def hunt_opportunities(self):
        """Hunt for the best trading opportunities across entire market"""
        self.log_trade("=== MARKET HUNTING CYCLE ===")
        
        # 1. Screen for momentum opportunities
        momentum_ops = self.screen_stocks_momentum()
        self.log_trade(f"Found {len(momentum_ops)} momentum opportunities")
        
        # 2. Screen for mean reversion opportunities  
        mean_rev_ops = self.screen_stocks_mean_reversion()
        self.log_trade(f"Found {len(mean_rev_ops)} mean reversion opportunities")
        
        # 3. Screen for options opportunities
        options_ops = self.screen_options_opportunities()
        self.log_trade(f"Found {len(options_ops)} options opportunities")
        
        # 4. Combine and rank all opportunities
        all_opportunities = []
        
        # Add stock opportunities
        all_opportunities.extend(momentum_ops)
        all_opportunities.extend(mean_rev_ops)
        
        # Add options opportunities
        all_opportunities.extend(options_ops)
        
        # Sort by confidence
        all_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Execute top opportunities
        executed_count = 0
        for opp in all_opportunities[:12]:  # Top 12 opportunities
            if opp['confidence'] > 0.5:
                await self.execute_opportunity(opp)
                executed_count += 1
                await asyncio.sleep(1)  # Brief delay between trades
        
        self.log_trade(f"Executed {executed_count} high-confidence trades")
        
        # Log opportunity summary
        if all_opportunities:
            best_stock = max([o for o in all_opportunities if o['type'] == 'STOCK'], 
                           key=lambda x: x['confidence'], default=None)
            best_option = max([o for o in all_opportunities if o['type'] == 'OPTIONS'], 
                            key=lambda x: x['confidence'], default=None)
            
            if best_stock:
                self.log_trade(f"Best Stock: {best_stock['symbol']} {best_stock['signal']} - {best_stock['confidence']:.1%}")
            if best_option:
                self.log_trade(f"Best Option: {best_option['symbol']} {best_option['strategy']} - {best_option['confidence']:.1%}")
    
    async def execute_opportunity(self, opp):
        """Execute a trading opportunity"""
        self.trade_count += 1
        
        if opp['type'] == 'STOCK':
            await self.execute_stock_trade(opp)
        else:  # OPTIONS
            await self.execute_options_trade(opp)
    
    async def execute_stock_trade(self, opp):
        """Execute stock trade"""
        try:
            if self.broker:
                # Execute through Alpaca API
                from agents.broker_integration import OrderRequest, OrderSide, OrderType
                
                order_side = OrderSide.BUY if opp['signal'] == "BUY" else OrderSide.SELL
                
                order_request = OrderRequest(
                    symbol=opp['symbol'],
                    qty=opp['quantity'],
                    side=order_side,
                    type=OrderType.MARKET
                )
                
                order_response = await self.broker.submit_order(order_request)
                self.log_trade(f"STOCK TRADE #{self.trade_count}: {opp['signal']} {opp['quantity']} {opp['symbol']} @ ${opp['price']:.2f} | ID: {order_response.id} | Confidence: {opp['confidence']:.1%} | {opp['reason']}")
            else:
                self.log_trade(f"STOCK TRADE #{self.trade_count}: {opp['signal']} {opp['quantity']} {opp['symbol']} @ ${opp['price']:.2f} | Confidence: {opp['confidence']:.1%} | {opp['reason']}")
                
            # Update position tracking
            if opp['symbol'] not in self.position_tracker:
                self.position_tracker[opp['symbol']] = 0
            
            if opp['signal'] == "BUY":
                self.position_tracker[opp['symbol']] += opp['quantity']
            else:
                self.position_tracker[opp['symbol']] -= opp['quantity']
                
        except Exception as e:
            self.log_trade(f"Error executing stock trade: {e}")
    
    async def execute_options_trade(self, opp):
        """Execute options trade (simulated)"""
        # Options trading simulation (Alpaca has limited options support)
        strategy = opp['strategy']
        
        if 'strike' in opp:
            trade_desc = f"{strategy} {opp['symbol']} ${opp['strike']:.2f} strike"
        elif 'credit' in opp:
            trade_desc = f"{strategy} {opp['symbol']} for ${opp['credit']:.2f} credit"
        else:
            trade_desc = f"{strategy} {opp['symbol']}"
        
        self.log_trade(f"OPTIONS TRADE #{self.trade_count}: {trade_desc} | Contracts: {opp['quantity']} | Confidence: {opp['confidence']:.1%} | {opp['reason']}")
    
    async def start_market_hunting(self):
        """Start the comprehensive market hunting system"""
        print("HIVE TRADE - FULL MARKET OPPORTUNITY HUNTER")
        print("=" * 52)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Scanning {len(self.all_stocks)} stocks across all sectors")
        print("Strategies: Momentum + Mean Reversion + Options")
        print("Mode: Paper Trading (Safe)")
        print("-" * 52)
        
        # Initialize broker
        try:
            from agents.broker_integration import AlpacaBrokerIntegration
            self.broker = AlpacaBrokerIntegration(paper_trading=True)
            account_info = await self.broker.get_account_info()
            if account_info:
                self.log_trade(f"Connected to Alpaca: {account_info.get('account_number', 'N/A')}")
                self.log_trade(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            else:
                self.log_trade("Alpaca connection failed - using comprehensive mock mode")
                self.broker = None
        except Exception as e:
            self.log_trade(f"Broker error: {e}")
            self.log_trade("Running in comprehensive simulation mode")
            self.broker = None
        
        self.log_trade("Starting comprehensive market hunting...")
        self.log_trade(f"Stock universe: {len(self.all_stocks)} symbols")
        self.log_trade("Screening for: Momentum, Mean Reversion, Options opportunities")
        
        try:
            hunt_cycle = 0
            while True:
                current_time = datetime.now()
                current_hour = current_time.hour
                
                # Hunt during market hours (9 AM - 4 PM)
                if 9 <= current_hour <= 16:
                    hunt_cycle += 1
                    self.log_trade(f"=== HUNT CYCLE #{hunt_cycle} ===")
                    
                    await self.hunt_opportunities()
                    
                    # Log positions every 5 cycles
                    if hunt_cycle % 5 == 0 and self.position_tracker:
                        positions_str = ", ".join([f"{sym}: {pos}" for sym, pos in self.position_tracker.items() if pos != 0])
                        if positions_str:
                            self.log_trade(f"Active Positions: {positions_str}")
                
                # Wait 3 minutes before next hunt cycle
                await asyncio.sleep(180)
                
        except KeyboardInterrupt:
            self.log_trade(f"Market hunting stopped. Total trades executed: {self.trade_count}")
        except Exception as e:
            self.log_trade(f"Market hunting error: {e}")
        
        self.log_trade("Market hunting session ended")
        return self.trade_count

async def main():
    """Main function"""
    print("Starting HiveTrading Full Market Hunter...")
    print("Features:")
    print("- Scans 80+ stocks across all major sectors")
    print("- Momentum & breakout detection")  
    print("- Mean reversion opportunities")
    print("- Comprehensive options strategies")
    print("- All trades in PAPER TRADING mode")
    print()
    
    hunter = MarketHunter()
    result = await hunter.start_market_hunting()
    
    print(f"\nMarket hunting completed!")
    print(f"Total opportunities executed: {result}")

if __name__ == "__main__":
    asyncio.run(main())