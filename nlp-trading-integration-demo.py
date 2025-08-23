#!/usr/bin/env python3
"""
Hive Trade - NLP-Driven Trading Decision Demo
Shows how sentiment analysis directly influences trading algorithms and portfolio decisions
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

# Import our previous NLP results
def load_nlp_results():
    """Load the NLP analysis results from previous demo"""
    try:
        with open("nlp_analysis_results.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: nlp_analysis_results.json not found. Using mock data.")
        return None

class NLPTradingEngine:
    """Trading engine that incorporates NLP sentiment analysis"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}  # symbol -> {'shares': int, 'avg_cost': float, 'sentiment_score': float}
        self.trades = []
        self.sentiment_threshold_buy = 0.3
        self.sentiment_threshold_sell = -0.3
        self.max_position_size = 0.25  # Max 25% of portfolio per position
        self.transaction_cost = 0.001  # 0.1% transaction cost
        
        print("NLP Trading Engine Initialized")
        print(f"Starting Capital: ${self.capital:,.2f}")
        print(f"Sentiment Thresholds: Buy > {self.sentiment_threshold_buy}, Sell < {self.sentiment_threshold_sell}")
        print(f"Max Position Size: {self.max_position_size*100}% of portfolio")
        print()
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        cash_value = self.capital
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_value += position['shares'] * current_prices[symbol]
        
        return cash_value + positions_value
    
    def calculate_position_size(self, sentiment_score: float, confidence: float, current_price: float, portfolio_value: float) -> int:
        """Calculate position size based on sentiment and confidence"""
        
        # Base position size as percentage of portfolio
        base_allocation = min(self.max_position_size, abs(sentiment_score) * confidence * 0.4)
        
        # Dollar amount to allocate
        dollar_allocation = portfolio_value * base_allocation
        
        # Convert to shares (rounded down)
        shares = int(dollar_allocation / current_price)
        
        # Ensure we have enough capital for the trade
        required_capital = shares * current_price * (1 + self.transaction_cost)
        if required_capital > self.capital:
            shares = int(self.capital / (current_price * (1 + self.transaction_cost)))
        
        return max(0, shares)
    
    async def process_nlp_signal(self, symbol: str, sentiment_score: float, confidence: float, 
                                signals: Dict, current_price: float, timestamp: str) -> Dict[str, Any]:
        """Process a single NLP signal and make trading decisions"""
        
        portfolio_value = self.get_portfolio_value({symbol: current_price})
        current_position = self.positions.get(symbol, {'shares': 0, 'avg_cost': 0.0, 'sentiment_score': 0.0})
        
        decision = {
            'symbol': symbol,
            'timestamp': timestamp,
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'current_price': current_price,
            'action': 'HOLD',
            'shares': 0,
            'rationale': '',
            'portfolio_value_before': portfolio_value
        }
        
        # Decision logic based on sentiment and existing position
        if sentiment_score > self.sentiment_threshold_buy and confidence > 0.6:
            # Strong positive sentiment - consider buying
            if current_position['shares'] == 0:
                # No existing position - open new long position
                shares_to_buy = self.calculate_position_size(sentiment_score, confidence, current_price, portfolio_value)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    
                    self.capital -= cost
                    self.positions[symbol] = {
                        'shares': shares_to_buy,
                        'avg_cost': current_price,
                        'sentiment_score': sentiment_score
                    }
                    
                    trade = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost,
                        'sentiment_score': sentiment_score,
                        'confidence': confidence
                    }
                    self.trades.append(trade)
                    
                    decision.update({
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'rationale': f'Strong positive sentiment ({sentiment_score:.3f}) with high confidence ({confidence:.2f})'
                    })
            
            elif current_position['sentiment_score'] < sentiment_score - 0.2:
                # Existing position but sentiment improved significantly - add to position
                additional_shares = self.calculate_position_size(sentiment_score * 0.5, confidence, current_price, portfolio_value)
                if additional_shares > 0:
                    cost = additional_shares * current_price * (1 + self.transaction_cost)
                    
                    if self.capital >= cost:
                        total_shares = current_position['shares'] + additional_shares
                        new_avg_cost = ((current_position['shares'] * current_position['avg_cost']) + 
                                      (additional_shares * current_price)) / total_shares
                        
                        self.capital -= cost
                        self.positions[symbol] = {
                            'shares': total_shares,
                            'avg_cost': new_avg_cost,
                            'sentiment_score': sentiment_score
                        }
                        
                        trade = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'action': 'BUY_ADD',
                            'shares': additional_shares,
                            'price': current_price,
                            'cost': cost,
                            'sentiment_score': sentiment_score,
                            'confidence': confidence
                        }
                        self.trades.append(trade)
                        
                        decision.update({
                            'action': 'BUY_ADD',
                            'shares': additional_shares,
                            'rationale': f'Sentiment improved from {current_position["sentiment_score"]:.3f} to {sentiment_score:.3f}'
                        })
        
        elif sentiment_score < self.sentiment_threshold_sell and confidence > 0.6:
            # Strong negative sentiment - consider selling
            if current_position['shares'] > 0:
                # Existing long position - sell based on sentiment strength
                sell_ratio = min(1.0, abs(sentiment_score) * confidence)
                shares_to_sell = int(current_position['shares'] * sell_ratio)
                
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                    
                    self.capital += proceeds
                    remaining_shares = current_position['shares'] - shares_to_sell
                    
                    if remaining_shares > 0:
                        self.positions[symbol] = {
                            'shares': remaining_shares,
                            'avg_cost': current_position['avg_cost'],
                            'sentiment_score': sentiment_score
                        }
                    else:
                        del self.positions[symbol]
                    
                    pnl = (current_price - current_position['avg_cost']) * shares_to_sell
                    
                    trade = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'proceeds': proceeds,
                        'pnl': pnl,
                        'sentiment_score': sentiment_score,
                        'confidence': confidence
                    }
                    self.trades.append(trade)
                    
                    decision.update({
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'rationale': f'Strong negative sentiment ({sentiment_score:.3f}) with high confidence ({confidence:.2f})'
                    })
        
        # Update decision with final portfolio value
        decision['portfolio_value_after'] = self.get_portfolio_value({symbol: current_price})
        decision['pnl'] = decision['portfolio_value_after'] - decision['portfolio_value_before']
        
        return decision
    
    def calculate_performance_metrics(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        final_portfolio_value = self.get_portfolio_value(current_prices)
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        
        # Calculate trade statistics
        profitable_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        win_rate = len(profitable_trades) / len(self.trades) if self.trades else 0
        avg_win = sum(t.get('pnl', 0) for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0
        avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Calculate sentiment-based metrics
        sentiment_scores = [t['sentiment_score'] for t in self.trades]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': len(self.trades),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'avg_trade_sentiment': avg_sentiment,
            'cash_remaining': self.capital,
            'positions_value': final_portfolio_value - self.capital,
            'active_positions': len(self.positions)
        }

async def demonstrate_nlp_trading_integration():
    """Main demonstration of NLP-driven trading decisions"""
    
    print("=" * 80)
    print("HIVE TRADE - NLP-DRIVEN TRADING DECISION DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Load NLP results from previous analysis
    nlp_data = load_nlp_results()
    
    # Initialize trading engine
    trading_engine = NLPTradingEngine(initial_capital=100000.0)
    
    # Mock current stock prices (would be real-time in production)
    current_prices = {
        'AAPL': 185.50,
        'TSLA': 220.30,
        'MSFT': 375.80,
        'SPY': 485.20
    }
    
    print("MARKET DATA:")
    print("-" * 15)
    for symbol, price in current_prices.items():
        print(f"{symbol}: ${price:.2f}")
    print()
    
    # Process each NLP signal
    trading_decisions = []
    
    if nlp_data and 'analysis_results' in nlp_data:
        analysis_results = nlp_data['analysis_results']
    else:
        # Use mock data if file not found
        analysis_results = [
            {
                'article': {'symbol': 'AAPL', 'timestamp': '2024-01-15 16:30:00'},
                'sentiment': {'composite_sentiment': 0.800},
                'trading_impact': {'confidence': 0.81, 'total_impact_score': 0.597},
                'signals': {'earnings_signals': [{'type': 'earnings_beat'}]}
            },
            {
                'article': {'symbol': 'TSLA', 'timestamp': '2024-01-15 14:22:00'},
                'sentiment': {'composite_sentiment': -0.750},
                'trading_impact': {'confidence': 0.86, 'total_impact_score': -0.738},
                'signals': {'rating_changes': [{'type': 'downgrade'}]}
            },
            {
                'article': {'symbol': 'MSFT', 'timestamp': '2024-01-15 15:45:00'},
                'sentiment': {'composite_sentiment': 0.600},
                'trading_impact': {'confidence': 0.73, 'total_impact_score': 0.360},
                'signals': {}
            },
            {
                'article': {'symbol': 'SPY', 'timestamp': '2024-01-15 18:15:00'},
                'sentiment': {'composite_sentiment': 0.250},
                'trading_impact': {'confidence': 0.53, 'total_impact_score': 0.150},
                'signals': {}
            }
        ]
    
    print("PROCESSING NLP SIGNALS AND MAKING TRADING DECISIONS:")
    print("=" * 60)
    
    for i, result in enumerate(analysis_results, 1):
        symbol = result['article']['symbol']
        timestamp = result['article']['timestamp']
        sentiment_score = result['sentiment']['composite_sentiment']
        confidence = result['trading_impact']['confidence']
        signals = result['signals']
        current_price = current_prices.get(symbol, 100.0)
        
        print(f"\nSIGNAL {i}/4: {symbol} at {timestamp}")
        print("-" * 40)
        print(f"Sentiment Score: {sentiment_score:+.3f}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Key Signals: {', '.join([f'{k}: {len(v)}' for k, v in signals.items() if v])}")
        
        # Process the signal
        decision = await trading_engine.process_nlp_signal(
            symbol, sentiment_score, confidence, signals, current_price, timestamp
        )
        
        trading_decisions.append(decision)
        
        # Display decision
        print(f"\nTRADING DECISION:")
        print(f"Action: {decision['action']}")
        if decision['shares'] > 0:
            print(f"Shares: {decision['shares']:,}")
            trade_value = decision['shares'] * current_price
            print(f"Trade Value: ${trade_value:,.2f}")
        print(f"Rationale: {decision['rationale']}")
        print(f"Portfolio Impact: ${decision['pnl']:+,.2f}")
        print(f"New Portfolio Value: ${decision['portfolio_value_after']:,.2f}")
    
    print("\n" + "=" * 80)
    print("FINAL TRADING RESULTS AND PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Calculate performance metrics
    performance = trading_engine.calculate_performance_metrics(current_prices)
    
    print("\nPORTFOLIO SUMMARY:")
    print("-" * 20)
    print(f"Initial Capital: ${performance['initial_capital']:,.2f}")
    print(f"Final Portfolio Value: ${performance['final_portfolio_value']:,.2f}")
    print(f"Total Return: ${performance['final_portfolio_value'] - performance['initial_capital']:+,.2f}")
    print(f"Total Return %: {performance['total_return_pct']:+.2f}%")
    print(f"Cash Remaining: ${performance['cash_remaining']:,.2f}")
    print(f"Positions Value: ${performance['positions_value']:,.2f}")
    print(f"Active Positions: {performance['active_positions']}")
    
    print("\nTRADING STATISTICS:")
    print("-" * 20)
    print(f"Total Trades Executed: {performance['total_trades']}")
    print(f"Profitable Trades: {performance['profitable_trades']}")
    print(f"Losing Trades: {performance['losing_trades']}")
    print(f"Win Rate: {performance['win_rate']:.1%}")
    if performance['avg_win'] > 0:
        print(f"Average Win: ${performance['avg_win']:+,.2f}")
    if performance['avg_loss'] < 0:
        print(f"Average Loss: ${performance['avg_loss']:+,.2f}")
    if performance['profit_factor'] != float('inf'):
        print(f"Profit Factor: {performance['profit_factor']:.2f}")
    print(f"Average Trade Sentiment: {performance['avg_trade_sentiment']:+.3f}")
    
    print("\nCURRENT POSITIONS:")
    print("-" * 18)
    if trading_engine.positions:
        for symbol, position in trading_engine.positions.items():
            current_price = current_prices[symbol]
            position_value = position['shares'] * current_price
            unrealized_pnl = (current_price - position['avg_cost']) * position['shares']
            pnl_pct = (unrealized_pnl / (position['avg_cost'] * position['shares'])) * 100
            
            print(f"{symbol}: {position['shares']:,} shares @ ${position['avg_cost']:.2f} avg")
            print(f"  Current Value: ${position_value:,.2f}")
            print(f"  Unrealized P&L: ${unrealized_pnl:+,.2f} ({pnl_pct:+.1f}%)")
            print(f"  Entry Sentiment: {position['sentiment_score']:+.3f}")
    else:
        print("No active positions")
    
    print("\nTRADE HISTORY:")
    print("-" * 15)
    for trade in trading_engine.trades:
        action_symbol = "BUY" if trade['action'].startswith('BUY') else "SELL"
        trade_value = trade.get('cost', trade.get('proceeds', 0))
        pnl_info = f" | P&L: ${trade['pnl']:+.2f}" if 'pnl' in trade else ""
        
        print(f"{trade['timestamp']} | {action_symbol} {trade['shares']:,} {trade['symbol']} @ ${trade['price']:.2f}")
        print(f"  Value: ${trade_value:,.2f} | Sentiment: {trade['sentiment_score']:+.3f}{pnl_info}")
    
    # NLP Integration Insights
    print("\nNLP INTEGRATION INSIGHTS:")
    print("-" * 30)
    
    # Analyze sentiment vs performance correlation
    sentiment_trades = [(t['sentiment_score'], t.get('pnl', 0)) for t in trading_engine.trades if 'pnl' in t]
    if sentiment_trades:
        positive_sentiment_trades = [(s, p) for s, p in sentiment_trades if s > 0]
        negative_sentiment_trades = [(s, p) for s, p in sentiment_trades if s < 0]
        
        if positive_sentiment_trades:
            avg_positive_pnl = sum(p for _, p in positive_sentiment_trades) / len(positive_sentiment_trades)
            print(f"* Average P&L from positive sentiment trades: ${avg_positive_pnl:+,.2f}")
        
        if negative_sentiment_trades:
            avg_negative_pnl = sum(p for _, p in negative_sentiment_trades) / len(negative_sentiment_trades)
            print(f"* Average P&L from negative sentiment trades: ${avg_negative_pnl:+,.2f}")
    
    # Success rate by sentiment strength
    strong_sentiment_trades = [t for t in trading_engine.trades if abs(t['sentiment_score']) > 0.5]
    weak_sentiment_trades = [t for t in trading_engine.trades if abs(t['sentiment_score']) <= 0.5]
    
    print(f"* Strong sentiment signals (|score| > 0.5): {len(strong_sentiment_trades)} trades")
    print(f"* Weak sentiment signals (|score| <= 0.5): {len(weak_sentiment_trades)} trades")
    
    # Confidence impact
    high_confidence_trades = [t for t in trading_engine.trades if t['confidence'] > 0.7]
    print(f"* High confidence trades (>70%): {len(high_confidence_trades)}")
    
    print("\nSYSTEM RECOMMENDATIONS:")
    print("-" * 25)
    
    if performance['total_return_pct'] > 0:
        print("* NLP-driven strategy showed positive returns")
        print("* Consider increasing position sizes for high-confidence signals")
        print("* Monitor sentiment threshold effectiveness")
    else:
        print("* Strategy needs optimization")
        print("* Consider adjusting sentiment thresholds")
        print("* Review risk management parameters")
    
    if performance['win_rate'] > 0.6:
        print("* Strong win rate indicates effective signal filtering")
    else:
        print("* Consider improving signal quality filters")
    
    print("\nNEXT STEPS FOR PRODUCTION:")
    print("-" * 30)
    print("* Implement real-time news feed processing")
    print("* Add position sizing based on volatility")
    print("* Integrate with live trading APIs (Alpaca, IEX)")
    print("* Set up sentiment threshold optimization")
    print("* Add risk management stop-losses")
    print("* Implement portfolio rebalancing logic")
    
    # Save detailed results
    detailed_results = {
        'timestamp': datetime.now().isoformat(),
        'performance_metrics': performance,
        'trading_decisions': trading_decisions,
        'trade_history': trading_engine.trades,
        'final_positions': trading_engine.positions,
        'nlp_integration_insights': {
            'sentiment_threshold_buy': trading_engine.sentiment_threshold_buy,
            'sentiment_threshold_sell': trading_engine.sentiment_threshold_sell,
            'signals_processed': len(analysis_results),
            'trades_executed': len(trading_engine.trades),
            'avg_sentiment_per_trade': performance['avg_trade_sentiment']
        }
    }
    
    with open('nlp_trading_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n[SUCCESS] Detailed results saved to nlp_trading_results.json")
    print()

if __name__ == "__main__":
    print("Starting NLP-driven trading integration demo...")
    print()
    asyncio.run(demonstrate_nlp_trading_integration())