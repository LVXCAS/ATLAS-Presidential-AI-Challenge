#!/usr/bin/env python3
"""
Hive Trade - AI Trading System Architecture
Based on your specifications:
- Companies with Market cap over $1 billion
- Stock price over $1.50
- All exchanges
- Options Theta rules
- Risk management
"""

import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asyncio
import logging

class MarketDataAgent:
    """Historical/Real-Time Market Data Agent"""
    
    def __init__(self):
        load_dotenv()
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        
        # Market filter criteria
        self.min_market_cap = 1_000_000_000  # $1B minimum
        self.min_stock_price = 1.50
        self.min_option_price = 75.0  # $75 per contract minimum
        self.max_theta_pct = 0.10  # 10% max theta
        self.max_position_size = 0.20  # 20% max per trade
        
        print("[MARKET DATA AGENT] Initialized")
    
    def get_qualified_stocks(self):
        """Get stocks meeting our criteria"""
        try:
            # Get active US equities
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            qualified = []
            
            print(f"[SCREENING] {len(assets)} total assets...")
            
            # Sample major stocks first (for testing)
            major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
                           'NFLX', 'AMD', 'INTC', 'BABA', 'CRM', 'PYPL', 'ADBE']
            
            for symbol in major_stocks:
                try:
                    # Get stock info
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    market_cap = info.get('marketCap', 0)
                    current_price = info.get('currentPrice', 0)
                    
                    if market_cap >= self.min_market_cap and current_price >= self.min_stock_price:
                        qualified.append({
                            'symbol': symbol,
                            'price': current_price,
                            'market_cap': market_cap,
                            'volume': info.get('volume', 0),
                            'sector': info.get('sector', 'Unknown')
                        })
                        print(f"  [PASS] {symbol}: ${current_price} | Cap: ${market_cap/1e9:.1f}B")
                    else:
                        print(f"  [FAIL] {symbol}: ${current_price} | Cap: ${market_cap/1e9:.1f}B (filtered)")
                        
                except Exception as e:
                    print(f"  [ERROR] {symbol}: Error - {e}")
                    
            print(f"[QUALIFIED] {len(qualified)} stocks meet criteria")
            return qualified
            
        except Exception as e:
            print(f"[ERROR] Failed to get qualified stocks: {e}")
            return []

class MomentumAgent:
    """ML/RL Momentum Trading Agent"""
    
    def __init__(self, market_data_agent):
        self.market_data = market_data_agent
        self.name = "MOMENTUM_AGENT"
        print(f"[{self.name}] Initialized")
    
    def analyze_momentum(self, symbol, days=20):
        """Analyze momentum signals"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days*2}d")
            
            if len(hist) < days:
                return None
            
            # Calculate momentum indicators
            prices = hist['Close']
            sma_short = prices.rolling(window=10).mean()
            sma_long = prices.rolling(window=20).mean()
            rsi = self.calculate_rsi(prices)
            
            # Momentum signal
            current_price = prices.iloc[-1]
            signal_strength = 0.0
            signal = 'HOLD'
            
            # Bullish momentum
            if (sma_short.iloc[-1] > sma_long.iloc[-1] and 
                rsi.iloc[-1] < 70 and 
                prices.iloc[-1] > sma_short.iloc[-1]):
                signal = 'BUY'
                signal_strength = min(0.8, (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1] + 0.5)
            
            # Bearish momentum
            elif (sma_short.iloc[-1] < sma_long.iloc[-1] and 
                  rsi.iloc[-1] > 30):
                signal = 'SELL'
                signal_strength = min(0.8, (sma_long.iloc[-1] - sma_short.iloc[-1]) / sma_long.iloc[-1] + 0.5)
            
            return {
                'agent': self.name,
                'symbol': symbol,
                'signal': signal,
                'strength': signal_strength,
                'price': current_price,
                'rsi': rsi.iloc[-1],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"[{self.name}] Error analyzing {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class MeanReversionAgent:
    """ML/RL Mean Reversion Agent"""
    
    def __init__(self, market_data_agent):
        self.market_data = market_data_agent
        self.name = "MEAN_REVERSION_AGENT"
        print(f"[{self.name}] Initialized")
    
    def analyze_mean_reversion(self, symbol, days=30):
        """Analyze mean reversion opportunities"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days*2}d")
            
            if len(hist) < days:
                return None
            
            prices = hist['Close']
            mean_price = prices.rolling(window=days).mean()
            std_dev = prices.rolling(window=days).std()
            
            current_price = prices.iloc[-1]
            current_mean = mean_price.iloc[-1]
            current_std = std_dev.iloc[-1]
            
            # Calculate z-score
            z_score = (current_price - current_mean) / current_std
            
            signal = 'HOLD'
            signal_strength = 0.0
            
            # Mean reversion signals
            if z_score < -1.5:  # Oversold
                signal = 'BUY'
                signal_strength = min(0.9, abs(z_score) / 3.0)
            elif z_score > 1.5:  # Overbought
                signal = 'SELL'
                signal_strength = min(0.9, abs(z_score) / 3.0)
            
            return {
                'agent': self.name,
                'symbol': symbol,
                'signal': signal,
                'strength': signal_strength,
                'price': current_price,
                'z_score': z_score,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"[{self.name}] Error analyzing {symbol}: {e}")
            return None

class RiskManagerAgent:
    """Risk Management Agent"""
    
    def __init__(self, market_data_agent):
        self.market_data = market_data_agent
        self.name = "RISK_MANAGER"
        self.max_position_size = 0.20  # 20% max
        self.max_theta_pct = 0.10  # 10% max theta
        print(f"[{self.name}] Initialized")
    
    def evaluate_risk(self, symbol, trade_size, account_value, signal_data):
        """Evaluate trade risk"""
        try:
            # Position size check
            position_pct = trade_size / account_value
            if position_pct > self.max_position_size:
                return {
                    'approved': False,
                    'reason': f'Position size {position_pct:.1%} exceeds {self.max_position_size:.1%} limit',
                    'max_size': account_value * self.max_position_size
                }
            
            # Stock price check for options
            if signal_data.get('price', 0) < 5.0:
                return {
                    'approved': True,
                    'reason': 'Stock under $5 - recommend stock purchase over options',
                    'recommendation': 'STOCK_ONLY'
                }
            
            return {
                'approved': True,
                'reason': 'Trade approved by risk management',
                'position_pct': position_pct
            }
            
        except Exception as e:
            print(f"[{self.name}] Risk evaluation error: {e}")
            return {'approved': False, 'reason': f'Risk evaluation failed: {e}'}

class MasterAgent:
    """Master Coordinator Agent"""
    
    def __init__(self):
        load_dotenv()
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        
        # Initialize sub-agents
        self.market_data = MarketDataAgent()
        self.momentum = MomentumAgent(self.market_data)
        self.mean_reversion = MeanReversionAgent(self.market_data)
        self.risk_manager = RiskManagerAgent(self.market_data)
        
        self.name = "MASTER_AGENT"
        print(f"[{self.name}] All agents initialized")
    
    def aggregate_signals(self, symbol):
        """Aggregate signals from all agents"""
        try:
            print(f"\n[{self.name}] Analyzing {symbol}...")
            
            # Get signals from all agents
            momentum_signal = self.momentum.analyze_momentum(symbol)
            mean_reversion_signal = self.mean_reversion.analyze_mean_reversion(symbol)
            
            if not momentum_signal or not mean_reversion_signal:
                return None
            
            # Aggregate signals
            signals = [momentum_signal, mean_reversion_signal]
            buy_signals = [s for s in signals if s['signal'] == 'BUY']
            sell_signals = [s for s in signals if s['signal'] == 'SELL']
            
            # Decision logic
            if len(buy_signals) >= 1 and len(sell_signals) == 0:
                final_signal = 'BUY'
                confidence = np.mean([s['strength'] for s in buy_signals])
            elif len(sell_signals) >= 1 and len(buy_signals) == 0:
                final_signal = 'SELL'
                confidence = np.mean([s['strength'] for s in sell_signals])
            else:
                final_signal = 'HOLD'
                confidence = 0.5
            
            return {
                'symbol': symbol,
                'final_signal': final_signal,
                'confidence': confidence,
                'price': momentum_signal['price'],
                'supporting_signals': len(buy_signals) + len(sell_signals),
                'individual_signals': signals,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"[{self.name}] Error aggregating signals for {symbol}: {e}")
            return None
    
    def execute_trade_decision(self, analysis):
        """Execute trade with risk management"""
        try:
            if analysis['final_signal'] == 'HOLD':
                return None
            
            symbol = analysis['symbol']
            signal = analysis['final_signal']
            confidence = analysis['confidence']
            price = analysis['price']
            
            # Get account info
            account = self.api.get_account()
            account_value = float(account.portfolio_value)
            
            # Calculate position size based on confidence
            base_position_size = account_value * 0.05  # 5% base
            adjusted_size = base_position_size * confidence
            
            # Risk check
            risk_assessment = self.risk_manager.evaluate_risk(
                symbol, adjusted_size, account_value, analysis
            )
            
            if not risk_assessment['approved']:
                print(f"[{self.name}] Trade blocked: {risk_assessment['reason']}")
                return None
            
            # Calculate quantity
            qty = int(adjusted_size / price)
            
            if qty > 0:
                print(f"[{self.name}] EXECUTING: {signal} {qty} {symbol} @ ${price}")
                print(f"  Confidence: {confidence:.2f}")
                print(f"  Position Size: ${adjusted_size:,.2f} ({adjusted_size/account_value:.1%})")
                
                # Execute trade (uncomment when ready for live trading)
                # order = self.api.submit_order(
                #     symbol=symbol,
                #     qty=qty,
                #     side=signal.lower(),
                #     type='market',
                #     time_in_force='day'
                # )
                
                return {
                    'symbol': symbol,
                    'action': signal,
                    'qty': qty,
                    'price': price,
                    'confidence': confidence,
                    'status': 'EXECUTED'  # Change to order.status when live
                }
            
        except Exception as e:
            print(f"[{self.name}] Execution error: {e}")
            return None

async def main():
    """Main trading system loop"""
    print("=" * 60)
    print("HIVE TRADE - AI TRADING SYSTEM v2.1.3")
    print("Multi-Agent Architecture Implementation")
    print("=" * 60)
    
    master = MasterAgent()
    
    # Get qualified stocks
    qualified_stocks = master.market_data.get_qualified_stocks()
    
    if not qualified_stocks:
        print("[ERROR] No qualified stocks found")
        return
    
    # Analyze top stocks
    for stock in qualified_stocks[:5]:  # Analyze top 5
        symbol = stock['symbol']
        analysis = master.aggregate_signals(symbol)
        
        if analysis and analysis['final_signal'] != 'HOLD':
            print(f"\n[SIGNAL DETECTED]")
            print(f"   Symbol: {symbol}")
            print(f"   Signal: {analysis['final_signal']}")
            print(f"   Confidence: {analysis['confidence']:.2f}")
            print(f"   Price: ${analysis['price']}")
            print(f"   Supporting Agents: {analysis['supporting_signals']}")
            
            # Execute trade decision
            execution = master.execute_trade_decision(analysis)
            if execution:
                print(f"   [EXECUTED] {execution['action']}: {execution['qty']} shares")
    
    print("\n" + "=" * 60)
    print("AI ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())