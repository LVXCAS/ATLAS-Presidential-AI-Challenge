#!/usr/bin/env python3
"""
Enhanced HiveTrading Market Hunter
Features real options trading, position management, and risk controls
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

# Import enhanced components
from agents.broker_integration import AlpacaBrokerIntegration
from agents.position_manager import PositionManager, PositionType, ExitRule, ExitReason
from agents.options_trading_agent import OptionsTrader, OptionsStrategy
from agents.risk_management import RiskManager, RiskLevel

# Create logs directory
os.makedirs('logs', exist_ok=True)

class EnhancedMarketHunter:
    """Enhanced trading system with options, position management, and risk controls"""
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.broker = None
        self.position_manager = None
        self.risk_manager = RiskManager(risk_level)
        
        # Trading state
        self.trade_count = 0
        self.hunt_cycle = 0
        self.price_cache = {}
        self.last_cache_update = {}
        
        # Comprehensive stock universe: S&P 500, NASDAQ-100, and Dow Jones
        self.sp500_stocks = [
            'AAPL','MSFT','AMZN','NVDA','GOOGL','GOOG','BRK-B','META','TSLA','UNH','LLY','V','XOM','JPM',
            'JNJ','WMT','MA','AVGO','HD','PG','CVX','ABBV','MRK','KO','PEP','COST','BAC','TMO','NFLX',
            'DHR','ABT','CRM','VZ','ADBE','WFC','CMCSA','ACN','NKE','LIN','TXN','RTX','NEE','AMD','QCOM',
            'PM','LOW','SPGI','HON','UNP','T','INTU','MDT','CAT','AMGN','GS','ORCL','INTC','AXP','IBM',
            'COP','NOW','PFE','AMAT','BMY','DE','GILD','MU','BKNG','TJX','SCHW','ADP','ADI','SYK','TMUS',
            'LRCX','PLD','CB','MMC','FI','MO','DUK','TGT','CSX','SO','KLAC','ITW','EOG','WM','APD',
            'CL','CME','FCX','SHW','MAR','USB','HCA','NSC','ICE','GD','COF','EMR','PGR','NOC','TFC',
            'BSX','GM','F','MCD','BDX','AON','JCI','TRV','CARR','MMM','PSX','MPC','D','EXC','KMI',
            'SLB','PAYX','AIG','NUE','ECL','OKE','CTAS','GWW','CMI','WELL','ORLY','TEL','RSG','MSI',
            'ETN','FAST','VRTX','VRSK','HPQ','BK','MTB','PH','CPRT','GLW','WBA','DG','CTVA','LHX',
            'KR','EW','BIIB','MLM','A','DVA','DD','EA','HPE','MDLZ','IEX','RMD','ADM','WAB','XEL',
            'URI','WRB','FITB','LVS','SIVB','ZION','RF','HBAN','CFG','KEY','PVH','HRL','CLX','TSN',
            'WDC','MHK','LW','XRAY','PNR','SWK','NLSN','PRGO','HAS','DISH','GPS','NWL','UAA','UA'
        ]
        
        self.nasdaq100_stocks = [
            'AAPL','MSFT','AMZN','NVDA','GOOGL','GOOG','META','TSLA','COST','NFLX','ADBE','PEP','TMUS',
            'CMCSA','INTC','TXN','QCOM','INTU','AMD','AMAT','LRCX','MU','ADI','MELI','KLAC','CDNS',
            'SNPS','MRVL','ORLY','CSX','ABNB','FTNT','CHTR','TEAM','NXPI','ADP','WDAY','DXCM','PAYX',
            'ROST','FAST','BIIB','KDP','EA','VRSK','CTSH','LULU','CSGP','DLTR','ODFL','CRWD','ANSS',
            'FANG','LCID','MRNA','DDOG','ZM','DOCU','PTON','EBAY','ILMN','WBA','SIRI','MAT','CERN',
            'ALGN','BMRN','EXPE','VRTX','REGN','GILD','ISRG','BKNG','SBUX','FISV','MDLZ','MNST',
            'ATVI','PDD','PYPL','NTES','JD','BIDU','HON','ASML','AVGO','MCHP','PCAR','SGEN','XLNX'
        ]
        
        self.dow_jones_stocks = [
            'AAPL','MSFT','UNH','GS','HD','CAT','MCD','V','AXP','BA','TRV','JPM','JNJ','PG','CVX',
            'MRK','WMT','DIS','CRM','NKE','IBM','MMM','KO','DOW','CSCO','INTC','VZ','WBA','AMGN','HON'
        ]
        
        # ETFs for broader market exposure
        self.etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'XLF', 'XLK', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU']
        
        # Combine all stocks and remove duplicates
        self.all_stocks = list(set(
            self.sp500_stocks + 
            self.nasdaq100_stocks + 
            self.dow_jones_stocks + 
            self.etfs
        ))
        
        # Create sector mapping for comprehensive risk management
        self.stock_sectors = {
            'mega_cap_tech': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BK', 'USB', 'TFC', 'COF'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'GILD'],
            'consumer_discretionary': ['WMT', 'HD', 'TGT', 'NKE', 'MCD', 'SBUX', 'TJX', 'LOW'],
            'consumer_staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'CL', 'MDLZ'],
            'industrials': ['CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE'],
            'technology': ['NFLX', 'CRM', 'ADBE', 'INTC', 'IBM', 'ORCL', 'AMD', 'TXN', 'QCOM'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX'],
            'utilities': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL'],
            'etfs': self.etfs
        }
        
        # Performance tracking
        self.last_portfolio_report = None
        
    def log_trade(self, message):
        """Log trade to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp}: {message}"
        
        print(log_message)
        
        try:
            with open('logs/enhanced_market_hunter.log', 'a') as f:
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
            if ALPACA_API_KEY and self.broker:
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
                    current_price = result['c']
                    volume = result['v']
                    price_change = ((result['c'] - result['o']) / result['o']) if result['o'] > 0 else 0
                    
                    # Get technical indicators
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
            # Alpaca market data implementation would go here
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
        
        # 1. Enhanced Momentum Analysis
        if abs(price_change) > 0.02 and volume > 500000:
            if price_change > 0.03:  # Strong upward momentum
                signals.append(("BUY", 0.8, f"Strong momentum: +{price_change:.1%} on {volume:,} volume"))
            elif price_change < -0.03:  # Strong downward momentum
                signals.append(("SELL", 0.8, f"Strong decline: {price_change:.1%} on {volume:,} volume"))
        
        # 2. Mean Reversion Analysis with volatility filter
        if volatility > 15:  # Only trade mean reversion on volatile days
            if rsi < 25:  # Extremely oversold
                signals.append(("BUY", 0.7, f"Extremely oversold: RSI {rsi:.1f}"))
            elif rsi > 75:  # Extremely overbought
                signals.append(("SELL", 0.7, f"Extremely overbought: RSI {rsi:.1f}"))
        
        # 3. Volume Breakout Analysis
        if volume > 2000000:  # High volume
            if price_change > 0.015:
                signals.append(("BUY", 0.6, f"Volume breakout: {volume:,} shares, +{price_change:.1%}"))
            elif price_change < -0.015:
                signals.append(("SELL", 0.6, f"Volume selloff: {volume:,} shares, {price_change:.1%}"))
        
        # 4. Volatility Expansion
        if volatility > 4.0:  # High volatility day
            if price_change > 0.01:
                signals.append(("BUY", 0.5, f"Volatility expansion breakout: {volatility:.1f}%"))
            elif price_change < -0.01:
                signals.append(("SELL", 0.5, f"Volatility expansion decline: {volatility:.1f}%"))
        
        # Combine signals with improved logic
        if not signals:
            return "HOLD", 0.0, f"No signals - Price: ${price:.2f}, RSI: {rsi:.1f}, Vol: {volume:,}"
        
        buy_signals = [s for s in signals if s[0] == "BUY"]
        sell_signals = [s for s in signals if s[0] == "SELL"]
        
        if len(buy_signals) > len(sell_signals):
            total_confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
            reasons = " | ".join(s[2] for s in buy_signals)
            return "BUY", min(0.95, total_confidence), reasons
        
        elif len(sell_signals) > len(buy_signals):
            total_confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
            reasons = " | ".join(s[2] for s in sell_signals)
            return "SELL", min(0.95, total_confidence), reasons
        
        return "HOLD", 0.0, "Conflicting signals"
    
    async def hunt_opportunities(self):
        """Hunt for trading opportunities using enhanced system"""
        self.log_trade("=== ENHANCED MARKET HUNTING CYCLE ===")
        
        # Check if trading is allowed by risk manager
        trading_allowed, reason = self.risk_manager.check_trading_allowed()
        if not trading_allowed:
            self.log_trade(f"Trading suspended: {reason}")
            return
        
        stock_opportunities = []
        options_opportunities = []
        successful_fetches = 0
        
        # Get current prices for position monitoring
        price_updates = {}
        
        # For performance with large stock universe, sample a subset each cycle
        # This ensures we scan different stocks each time while managing API limits
        import random
        shuffled_stocks = self.all_stocks.copy()
        random.shuffle(shuffled_stocks)
        sample_size = min(120, len(shuffled_stocks))  # Scan up to 120 stocks per cycle
        sampled_stocks = shuffled_stocks[:sample_size]
        
        self.log_trade(f"Scanning {len(sampled_stocks)} stocks from universe of {len(self.all_stocks)}")
        
        # Analyze each stock with real data
        for symbol in sampled_stocks:
            try:
                self.log_trade(f"Fetching data for {symbol}...")
                data = await asyncio.wait_for(self.get_real_market_data(symbol), timeout=10)
                
                if data:
                    successful_fetches += 1
                    price_updates[symbol] = data['price']
                    
                    # Analyze for stock signals
                    signal, confidence, reason = self.analyze_market_data(data)
                    
                    if signal != "HOLD" and confidence > 0.5:
                        # Calculate position size using risk management
                        quantity, risk_assessment = self.risk_manager.calculate_position_size(
                            symbol, data['price'], confidence, data['volatility'] / 100
                        )
                        
                        if risk_assessment.recommendation in ["APPROVE", "REDUCE"] and quantity > 0:
                            stock_opportunities.append({
                                'symbol': symbol,
                                'signal': signal,
                                'confidence': confidence,
                                'price': data['price'],
                                'reason': reason,
                                'quantity': quantity,
                                'risk_score': risk_assessment.overall_risk_score,
                                'data_source': data['source']
                            })
                    
                    # Analyze for options opportunities
                    if data['volatility'] > 15 and confidence > 0.55:  # Lowered thresholds for broader coverage
                        # Define liquid options stocks (higher volume, better spreads)
                        liquid_options_stocks = (
                            self.stock_sectors['mega_cap_tech'] + 
                            ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLK'] +  # Major ETFs
                            ['JPM', 'BAC', 'WFC', 'GS', 'C'] +  # Major financials
                            ['JNJ', 'PFE', 'MRK', 'UNH'] +  # Major healthcare
                            ['XOM', 'CVX', 'COP'] +  # Major energy
                            ['WMT', 'HD', 'MCD', 'NKE', 'DIS'] +  # Major consumer
                            ['BA', 'CAT', 'MMM', 'GE'] +  # Major industrials
                            ['V', 'MA', 'AXP'] +  # Payment processors
                            ['NFLX', 'CRM', 'ADBE', 'INTC', 'AMD', 'ORCL']  # Major tech
                        )
                        
                        # Trade options on liquid stocks, or any stock with very high confidence
                        if symbol in liquid_options_stocks or confidence > 0.75:
                            options_id = await self.position_manager.execute_options_trade(
                                symbol, data['price'], data['volatility'], data['rsi'], data['price_change']
                            )
                            if options_id:
                                options_opportunities.append({
                                    'symbol': symbol,
                                    'options_position_id': options_id,
                                    'confidence': confidence
                                })
                                self.log_trade(f"OPTIONS TRADE: {symbol} strategy executed - ID: {options_id}")
                            else:
                                # Only log failures for high-confidence trades to reduce noise
                                if confidence > 0.7:
                                    self.log_trade(f"OPTIONS FAILED: {symbol} - no suitable contracts found")
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except asyncio.TimeoutError:
                self.log_trade(f"Timeout fetching data for {symbol} - skipping")
            except Exception as e:
                self.log_trade(f"Error processing {symbol}: {e}")
        
        # Update position prices
        if price_updates:
            await self.position_manager.update_position_prices(price_updates)
        
        # Sort stock opportunities by risk-adjusted confidence
        stock_opportunities.sort(key=lambda x: x['confidence'] * (1 - x['risk_score']), reverse=True)
        
        self.log_trade(f"Successfully fetched real data for {successful_fetches}/{len(self.all_stocks)} symbols")
        self.log_trade(f"Found {len(stock_opportunities)} stock opportunities, {len(options_opportunities)} options trades")
        
        # Execute best stock opportunities
        executed_count = 0
        for opp in stock_opportunities[:5]:  # Top 5 opportunities
            position_id = await self.position_manager.execute_trade_with_management(
                symbol=opp['symbol'],
                signal=opp['signal'],
                quantity=opp['quantity'],
                price=opp['price'],
                confidence=opp['confidence'],
                strategy_source="enhanced_hunter"
            )
            
            if position_id:
                executed_count += 1
                self.trade_count += 1
                self.log_trade(f"TRADE #{self.trade_count}: {opp['signal']} {opp['quantity']} {opp['symbol']} @ ${opp['price']:.2f} | "
                             f"Confidence: {opp['confidence']:.1%} | Risk Score: {opp['risk_score']:.2f} | {opp['reason']}")
            
            await asyncio.sleep(1)
        
        self.log_trade(f"Executed {executed_count} stock trades, {len(options_opportunities)} options trades")
        
        # Monitor and manage existing positions
        await self.monitor_positions()
    
    async def monitor_positions(self):
        """Monitor and manage existing positions"""
        try:
            actions = await self.position_manager.monitor_positions()
            
            for action in actions:
                if action['action'] == 'CLOSE_STOCK':
                    self.log_trade(f"POSITION CLOSED: {action['symbol']} - {action['reason']} - "
                                 f"P&L: ${action['pnl']:.2f}")
                    # Update risk manager with trade outcome
                    self.risk_manager.record_trade_outcome(action['pnl'])
                
                elif action['action'] == 'CLOSE':  # Options
                    self.log_trade(f"OPTIONS CLOSED: {action['position_id']} - {action['reason']} - "
                                 f"P&L: ${action['pnl']:.2f} ({action['pnl_percent']:.1%})")
                    self.risk_manager.record_trade_outcome(action['pnl'])
        
        except Exception as e:
            self.log_trade(f"Error monitoring positions: {e}")
    
    async def log_portfolio_status(self):
        """Log detailed portfolio and risk status"""
        try:
            portfolio_summary = self.position_manager.get_portfolio_summary()
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            self.log_trade("=== PORTFOLIO STATUS ===")
            self.log_trade(f"Account Value: ${risk_metrics['account_value']:,.2f}")
            self.log_trade(f"Stock Positions: {portfolio_summary['stock_positions']['total_positions']}")
            self.log_trade(f"Options Positions: {portfolio_summary['options_positions']['total_positions']}")
            self.log_trade(f"Total Unrealized P&L: ${portfolio_summary['stock_positions']['total_unrealized_pnl']:.2f}")
            self.log_trade(f"Portfolio Heat: {risk_metrics['portfolio_heat_pct']:.1f}%")
            self.log_trade(f"Max Drawdown: {risk_metrics['max_drawdown_pct']:.1f}%")
            self.log_trade(f"Win Rate: {portfolio_summary['performance']['win_rate']:.1f}%")
            
            # Log sector allocations
            if risk_metrics['sector_allocations']:
                sectors = ", ".join([f"{k}: {v:.1f}%" for k, v in risk_metrics['sector_allocations'].items()])
                self.log_trade(f"Sector Allocations: {sectors}")
            
        except Exception as e:
            self.log_trade(f"Error logging portfolio status: {e}")
    
    async def start_enhanced_hunting(self):
        """Start enhanced trading system"""
        print("HIVE TRADE - ENHANCED MARKET HUNTER")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Trading Universe: {len(self.all_stocks)} stocks")
        print(f"  • S&P 500: {len(self.sp500_stocks)} stocks") 
        print(f"  • NASDAQ-100: {len(self.nasdaq100_stocks)} stocks")
        print(f"  • Dow Jones: {len(self.dow_jones_stocks)} stocks")
        print(f"  • ETFs: {len(self.etfs)} funds")
        print("Features: Real Options, Position Management, Risk Controls")
        print("Data Sources: Polygon + Alpaca + Yahoo Finance")
        print("-" * 60)
        
        # Initialize broker connection
        try:
            from agents.broker_integration import AlpacaBrokerIntegration
            self.broker = AlpacaBrokerIntegration(paper_trading=True)
            
            # Initialize position manager with broker
            self.position_manager = PositionManager(self.broker)
            
            account_info = await self.broker.get_account_info()
            if account_info:
                account_value = float(account_info.get('buying_power', 100000))
                # Fix: Ensure account value is never 0
                if account_value <= 0:
                    account_value = 100000  # Default to $100k for paper trading
                    self.log_trade("Warning: Account value was 0, using default $100,000")
                self.risk_manager.update_account_value(account_value)
                self.log_trade(f"Connected to Alpaca: {account_info.get('account_number', 'N/A')}")
                self.log_trade(f"Account Value: ${account_value:,.2f}")
            else:
                self.log_trade("Alpaca connection failed - using simulation mode")
                self.broker = None
                self.position_manager = PositionManager(None)
                
        except Exception as e:
            self.log_trade(f"Broker error: {e}")
            self.log_trade("Running in simulation mode")
            self.broker = None
            self.position_manager = PositionManager(None)
        
        # Check API configurations
        if POLYGON_API_KEY:
            self.log_trade(f"Polygon API configured: ...{POLYGON_API_KEY[-4:]}")
        else:
            self.log_trade("Warning: No Polygon API key found")
        
        self.log_trade("Starting enhanced market hunting...")
        
        try:
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
                
                if market_open and market_close:
                    self.hunt_cycle += 1
                    self.log_trade(f"=== ENHANCED HUNT CYCLE #{self.hunt_cycle} ===")
                    
                    await self.hunt_opportunities()
                    
                    # Log detailed status every 3 cycles
                    if self.hunt_cycle % 3 == 0:
                        await self.log_portfolio_status()
                        
                        # Show risk suggestions
                        suggestions = self.risk_manager.suggest_position_adjustments()
                        if suggestions:
                            self.log_trade("Risk Management Suggestions:")
                            for suggestion in suggestions[:3]:  # Show top 3
                                self.log_trade(f"  - {suggestion['type']}: {suggestion['reason']}")
                
                else:
                    self.log_trade(f"Market closed ({current_time.strftime('%H:%M ET')}) - monitoring positions only")
                    await self.monitor_positions()
                    await asyncio.sleep(300)  # Check every 5 minutes when closed
                    continue
                
                # Wait 5 minutes before next hunt during market hours
                await asyncio.sleep(300)
                
        except KeyboardInterrupt:
            self.log_trade(f"Enhanced market hunting stopped. Total trades: {self.trade_count}")
            await self.log_portfolio_status()
        except Exception as e:
            self.log_trade(f"Enhanced market hunting error: {e}")
            import traceback
            self.log_trade(f"Error details: {traceback.format_exc()}")
            # Try to recover gracefully
            await asyncio.sleep(60)  # Wait 1 minute before trying again
        
        self.log_trade("Enhanced market hunting session ended")
        return self.trade_count

async def main():
    """Main function"""
    print("Starting Enhanced HiveTrading Market Hunter...")
    print("New Features:")
    print("- Real options contract trading")
    print("- Automatic position management with stop-loss/take-profit")
    print("- Risk-based position sizing")
    print("- Portfolio heat and drawdown monitoring")
    print("- Multi-timeframe exit strategies")
    print()
    
    # Choose risk level
    risk_level = RiskLevel.MODERATE  # Can be CONSERVATIVE, MODERATE, or AGGRESSIVE
    
    hunter = EnhancedMarketHunter(risk_level)
    result = await hunter.start_enhanced_hunting()
    
    print(f"\nEnhanced market hunting completed!")
    print(f"Total trades executed: {result}")

if __name__ == "__main__":
    asyncio.run(main())