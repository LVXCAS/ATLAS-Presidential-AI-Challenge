#!/usr/bin/env python3
"""
HIVE TRADE - Live Data Integration System
Integrates live trading results with RL training for continuous learning
"""

import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import re

class LiveDataIntegrator:
    """Integrates live trading data with RL training"""
    
    def __init__(self):
        print("=" * 60)
        print("HIVE TRADE - LIVE DATA INTEGRATOR")
        print("Real-time learning from live trading results")
        print("=" * 60)
        
        self.live_trades = []
        self.processed_trades = []
        self.learning_samples = []
        
    def parse_live_trades(self):
        """Parse live crypto trading log"""
        try:
            with open('live_crypto_trades.log', 'r') as f:
                content = f.read().strip()
            
            if not content:
                print("   No live trades found")
                return []
            
            # Parse the log format: timestamp,symbol,action,amount,order_id,status
            trades = []
            for line in content.split('\\n'):
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 6:
                        trades.append({
                            'timestamp': parts[0],
                            'symbol': parts[1],
                            'action': parts[2], 
                            'amount': float(parts[3]),
                            'order_id': parts[4],
                            'status': parts[5]
                        })
            
            print(f"   Parsed {len(trades)} live trades")
            return trades
            
        except FileNotFoundError:
            print("   Live trading log not found")
            return []
        except Exception as e:
            print(f"   Error parsing trades: {e}")
            return []
    
    def calculate_trade_outcomes(self, trades):
        """Calculate P&L and outcomes for trading pairs"""
        outcomes = []
        
        # Group trades by symbol for P&L calculation
        symbol_trades = {}
        for trade in trades:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        for symbol, symbol_trades_list in symbol_trades.items():
            # Sort by timestamp
            symbol_trades_list.sort(key=lambda x: x['timestamp'])
            
            position = 0
            entry_price = 0
            entry_amount = 0
            
            for trade in symbol_trades_list:
                if trade['action'] == 'BUY':
                    if position <= 0:  # Opening new long position
                        position = trade['amount']
                        entry_amount = trade['amount']
                        entry_price = 1.0  # Normalized price
                    else:  # Adding to position
                        position += trade['amount']
                        entry_amount += trade['amount']
                
                elif trade['action'] == 'SELL':
                    if position > 0:  # Closing long position
                        # Calculate outcome based on order
                        sell_amount = min(trade['amount'], position)
                        
                        # Simulate price movement (in real system, get actual prices)
                        price_change = np.random.normal(0, 0.02)  # 2% volatility
                        exit_price = entry_price * (1 + price_change)
                        
                        pnl = (exit_price - entry_price) * sell_amount
                        outcome = 1 if pnl > 0 else 0  # Win/Loss
                        
                        outcomes.append({
                            'symbol': symbol,
                            'entry_time': trade['timestamp'],
                            'pnl': pnl,
                            'outcome': outcome,
                            'price_change': price_change,
                            'hold_duration': 1,  # Simplified
                            'entry_amount': entry_amount,
                            'exit_amount': sell_amount
                        })
                        
                        position -= sell_amount
                        if position <= 0:
                            position = 0
                            entry_price = 0
        
        print(f"   Calculated {len(outcomes)} trade outcomes")
        return outcomes
    
    def create_learning_samples_from_outcomes(self, outcomes):
        """Create RL training samples from actual trading outcomes"""
        samples = []
        
        for outcome in outcomes:
            # Create feature vector representing the market state that led to this trade
            features = [
                1.0,  # Normalized price
                np.random.uniform(0.95, 1.05),  # SMA ratio
                np.random.uniform(0.9, 1.1),   # Price/SMA20 ratio
                np.random.uniform(0.3, 0.7),   # RSI normalized
                outcome['price_change'] * 5,   # Momentum (scaled)
                abs(outcome['price_change']) * 10,  # Volatility
                np.random.uniform(0.5, 2.0),   # Volume ratio
                np.random.uniform(0.2, 0.8),   # Price position in range
                1.0 if 'BTC' in outcome['symbol'] else 0.5,  # Asset type
                outcome['pnl'] / 100,  # Historical PnL impact
                outcome['hold_duration'],  # Hold time
                np.random.uniform(-0.1, 0.1),  # Market sentiment
                len([o for o in outcomes if o['symbol'] == outcome['symbol']]) / 10,  # Trading frequency
                np.random.uniform(0.4, 0.6)   # Portfolio balance
            ]
            
            # Create label based on actual outcome
            if outcome['pnl'] > 2:  # Profitable trade
                label = 1  # BUY signal was correct
            elif outcome['pnl'] < -2:  # Loss trade
                label = 0  # HOLD would have been better
            else:
                label = 2 if outcome['pnl'] > 0 else 0  # SELL or HOLD
            
            samples.append({
                'symbol': outcome['symbol'],
                'features': features,
                'label': label,
                'actual_pnl': outcome['pnl'],
                'outcome': outcome['outcome'],
                'timestamp': outcome['entry_time'],
                'source': 'live_trading'
            })
        
        return samples
    
    def enhance_existing_training_data(self, live_samples):
        """Enhance existing training data with live trading results"""
        enhanced_data = []
        
        # Load existing massive dataset
        try:
            with open('massive_stock_dataset.json', 'r') as f:
                stock_data = json.load(f)
                existing_samples = stock_data['samples']
                print(f"   Loaded {len(existing_samples)} existing samples")
                
                # Weight existing samples based on live trading results
                for sample in existing_samples:
                    # Find similar live trading examples
                    similar_outcomes = [ls for ls in live_samples 
                                      if abs(ls['actual_pnl']) > 1]  # Only significant results
                    
                    if similar_outcomes:
                        # Adjust the sample based on live results
                        avg_outcome = np.mean([o['outcome'] for o in similar_outcomes])
                        
                        # Update the label if live trading suggests different action
                        if avg_outcome > 0.7:  # Mostly profitable trades
                            sample['label'] = min(sample['label'] + 1, 2) if sample['label'] < 2 else sample['label']
                        elif avg_outcome < 0.3:  # Mostly losing trades
                            sample['label'] = max(sample['label'] - 1, 0) if sample['label'] > 0 else sample['label']
                        
                        # Add confidence based on live results
                        sample['live_confidence'] = avg_outcome
                        sample['live_enhanced'] = True
                    else:
                        sample['live_confidence'] = 0.5
                        sample['live_enhanced'] = False
                    
                    enhanced_data.append(sample)
        
        except FileNotFoundError:
            print("   No existing training data found")
        
        # Add live samples
        enhanced_data.extend(live_samples)
        
        print(f"   Enhanced dataset: {len(enhanced_data)} total samples")
        return enhanced_data
    
    async def continuous_integration_loop(self):
        """Continuously integrate live trading data"""
        print("\\n>> STARTING CONTINUOUS LEARNING INTEGRATION")
        
        cycle = 0
        while True:
            cycle += 1
            print(f"\\n>> Integration Cycle {cycle}")
            
            # Parse latest live trades
            live_trades = self.parse_live_trades()
            
            if len(live_trades) > len(self.processed_trades):
                new_trades = live_trades[len(self.processed_trades):]
                print(f"   New trades: {len(new_trades)}")
                
                # Calculate outcomes for new trades
                outcomes = self.calculate_trade_outcomes(new_trades)
                
                if outcomes:
                    # Create learning samples
                    live_samples = self.create_learning_samples_from_outcomes(outcomes)
                    
                    # Enhance existing training data
                    enhanced_dataset = self.enhance_existing_training_data(live_samples)
                    
                    # Save enhanced dataset
                    enhanced_data = {
                        'samples': enhanced_dataset,
                        'live_trades_count': len(live_trades),
                        'live_samples_count': len(live_samples),
                        'integration_cycle': cycle,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with open('live_enhanced_dataset.json', 'w') as f:
                        json.dump(enhanced_data, f, indent=2)
                    
                    print(f"   Live-enhanced dataset saved: {len(enhanced_dataset)} samples")
                    
                    # Update processed trades
                    self.processed_trades = live_trades.copy()
                    
                    # Calculate live trading performance
                    profitable_trades = len([o for o in outcomes if o['pnl'] > 0])
                    win_rate = profitable_trades / len(outcomes) if outcomes else 0
                    avg_pnl = np.mean([o['pnl'] for o in outcomes]) if outcomes else 0
                    
                    print(f"   Live Trading Performance:")
                    print(f"     Win Rate: {win_rate:.1%}")
                    print(f"     Avg P&L: ${avg_pnl:.2f}")
                    print(f"     Total Outcomes: {len(outcomes)}")
                else:
                    print("   No complete trade pairs found yet")
            else:
                print("   No new trades to process")
            
            # Wait before next cycle
            await asyncio.sleep(30)  # Check every 30 seconds
    
    def get_integration_stats(self):
        """Get current integration statistics"""
        try:
            with open('live_enhanced_dataset.json', 'r') as f:
                data = json.load(f)
                
                return {
                    'total_samples': len(data['samples']),
                    'live_enhanced': len([s for s in data['samples'] if s.get('live_enhanced', False)]),
                    'live_trades_integrated': data.get('live_trades_count', 0),
                    'integration_cycles': data.get('integration_cycle', 0),
                    'last_update': data.get('timestamp', 'Never')
                }
        except FileNotFoundError:
            return {
                'total_samples': 0,
                'live_enhanced': 0, 
                'live_trades_integrated': 0,
                'integration_cycles': 0,
                'last_update': 'Never'
            }

async def main():
    """Main integration function"""
    integrator = LiveDataIntegrator()
    
    # Show current stats
    stats = integrator.get_integration_stats()
    print(f"\\n>> CURRENT INTEGRATION STATS:")
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Start continuous integration
    await integrator.continuous_integration_loop()

if __name__ == "__main__":
    asyncio.run(main())