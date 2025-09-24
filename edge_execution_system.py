"""
HIGH-EDGE OPTIONS EXECUTION SYSTEM
==================================
Automated execution of high-edge opportunities for 20%+ monthly returns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from hybrid_edge_finder import HybridEdgeFinder

class EdgeExecutionSystem:
    """Automated execution system for high-edge opportunities."""
    
    def __init__(self, account_size=100000):
        """Initialize execution system."""
        self.account_size = account_size
        self.max_risk_per_trade = 0.03  # 3% max risk per trade
        self.max_daily_risk = 0.10      # 10% max daily risk
        self.daily_trades = 0
        self.daily_risk_used = 0.0
        
        # Initialize edge finder
        self.finder = HybridEdgeFinder()
        
        print("EDGE EXECUTION SYSTEM INITIALIZED")
        print("=" * 50)
        print(f"Account size: ${account_size:,}")
        print(f"Max risk per trade: {self.max_risk_per_trade*100:.1f}%")
        print(f"Max daily risk: {self.max_daily_risk*100:.1f}%")
    
    def calculate_position_size(self, edge_score, volatility):
        """Calculate optimal position size based on edge and volatility."""
        
        # Base position size (3% max)
        base_risk = self.account_size * self.max_risk_per_trade
        
        # Adjust for edge strength (higher edge = larger size)
        edge_multiplier = min(edge_score / 1000, 2.0)  # Cap at 2x
        
        # Adjust for volatility (higher vol = smaller size)
        vol_adjustment = max(0.5, 1.0 - (volatility - 20) / 100)
        
        # Calculate final position size
        position_size = base_risk * edge_multiplier * vol_adjustment
        
        # Ensure we don't exceed daily risk limit
        remaining_daily_risk = (self.max_daily_risk - self.daily_risk_used) * self.account_size
        position_size = min(position_size, remaining_daily_risk)
        
        return max(position_size, 0)
    
    def generate_trade_signals(self):
        """Generate specific trade signals from edge opportunities."""
        
        print("\nGENERATING TRADE SIGNALS...")
        print("-" * 40)
        
        # Get fresh opportunities
        opportunities = self.finder.generate_comprehensive_report()
        
        trade_signals = []
        
        # Process momentum plays
        for play in opportunities['momentum'][:2]:  # Top 2 momentum plays
            
            position_size = self.calculate_position_size(
                play['edge_score'], 
                play['volatility']
            )
            
            if position_size > 1000:  # Minimum $1000 trade
                
                signal = {
                    'type': 'DIRECTIONAL',
                    'symbol': play['symbol'],
                    'direction': play['direction'],
                    'current_price': play['current_price'],
                    'strategy': 'LONG_CALLS' if play['direction'] == 'BULLISH' else 'LONG_PUTS',
                    'target_dte': 45,
                    'position_size': position_size,
                    'max_loss': position_size,
                    'target_profit': position_size * 1.5,  # 150% target
                    'edge_score': play['edge_score'],
                    'confidence': 'HIGH' if play['edge_score'] > 800 else 'MEDIUM',
                    'entry_conditions': {
                        'momentum_20d': play['momentum_20d'],
                        'volume_surge': play['volume_surge'],
                        'volatility': play['volatility']
                    },
                    'risk_level': position_size / self.account_size,
                    'timestamp': datetime.now().isoformat()
                }
                
                trade_signals.append(signal)
                
                print(f"SIGNAL: {signal['strategy']} {signal['symbol']}")
                print(f"  Size: ${signal['position_size']:,.0f}")
                print(f"  Confidence: {signal['confidence']}")
        
        # Process volatility breakout plays  
        for setup in opportunities['breakouts'][:2]:  # Top 2 breakout setups
            
            position_size = self.calculate_position_size(
                setup['edge_score'],
                setup['recent_vol']
            )
            
            if position_size > 1000:
                
                signal = {
                    'type': 'VOLATILITY',
                    'symbol': setup['symbol'],
                    'direction': 'NEUTRAL',
                    'current_price': setup['current_price'],
                    'strategy': 'LONG_STRADDLE',
                    'target_dte': 35,
                    'position_size': position_size,
                    'max_loss': position_size,
                    'target_profit': position_size * 2.0,  # 200% target for vol plays
                    'edge_score': setup['edge_score'],
                    'confidence': 'HIGH' if setup['edge_score'] > 500 else 'MEDIUM',
                    'entry_conditions': {
                        'vol_contraction': setup['vol_contraction'],
                        'breakout_potential': setup['breakout_potential'],
                        'range_compression': setup['range_compression']
                    },
                    'risk_level': position_size / self.account_size,
                    'timestamp': datetime.now().isoformat()
                }
                
                trade_signals.append(signal)
                
                print(f"SIGNAL: {signal['strategy']} {signal['symbol']}")
                print(f"  Size: ${signal['position_size']:,.0f}")
                print(f"  Confidence: {signal['confidence']}")
        
        return trade_signals
    
    def create_execution_plan(self, signals):
        """Create detailed execution plan."""
        
        print(f"\nCREATING EXECUTION PLAN FOR {len(signals)} SIGNALS...")
        print("=" * 50)
        
        execution_plan = {
            'timestamp': datetime.now().isoformat(),
            'account_size': self.account_size,
            'total_signals': len(signals),
            'total_risk': sum(s['risk_level'] for s in signals),
            'trades': []
        }
        
        for i, signal in enumerate(signals, 1):
            
            # Calculate specific options parameters
            if signal['strategy'] in ['LONG_CALLS', 'LONG_PUTS']:
                strike_price = signal['current_price'] * (1.05 if signal['direction'] == 'BULLISH' else 0.95)
                contract_type = 'CALL' if signal['direction'] == 'BULLISH' else 'PUT'
            else:  # LONG_STRADDLE
                strike_price = signal['current_price']
                contract_type = 'STRADDLE'
            
            trade_plan = {
                'trade_id': f"EDGE_{signal['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                'priority': i,
                'symbol': signal['symbol'],
                'strategy': signal['strategy'],
                'contract_type': contract_type,
                'strike_price': round(strike_price, 2),
                'target_dte': signal['target_dte'],
                'position_size': signal['position_size'],
                'max_loss': signal['max_loss'],
                'target_profit': signal['target_profit'],
                'stop_loss': signal['position_size'] * 0.3,  # 30% stop loss
                'take_profit_1': signal['position_size'] * 0.5,  # 50% first target
                'take_profit_2': signal['target_profit'],  # Full target
                'confidence': signal['confidence'],
                'edge_score': signal['edge_score'],
                'execution_time': 'MARKET_OPEN',
                'notes': f"High-edge {signal['type'].lower()} play targeting 20%+ monthly returns"
            }
            
            execution_plan['trades'].append(trade_plan)
            
            print(f"\nTRADE #{i}: {trade_plan['trade_id']}")
            print(f"  Strategy: {trade_plan['strategy']}")
            print(f"  Strike: ${trade_plan['strike_price']}")
            print(f"  Size: ${trade_plan['position_size']:,.0f}")
            print(f"  Max Loss: ${trade_plan['max_loss']:,.0f}")
            print(f"  Target: ${trade_plan['target_profit']:,.0f}")
            print(f"  Edge Score: {trade_plan['edge_score']:.0f}")
        
        # Save execution plan
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"execution_plan_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(execution_plan, f, indent=2)
        
        print(f"\nExecution plan saved: {filename}")
        print(f"Total capital at risk: ${sum(t['max_loss'] for t in execution_plan['trades']):,.0f}")
        print(f"Risk as % of account: {execution_plan['total_risk']*100:.1f}%")
        
        return execution_plan
    
    def generate_trading_checklist(self, execution_plan):
        """Generate pre-market trading checklist."""
        
        checklist = f"""
DAILY HIGH-EDGE OPTIONS TRADING CHECKLIST
========================================
Date: {datetime.now().strftime('%Y-%m-%d')}
Trades Planned: {len(execution_plan['trades'])}

PRE-MARKET CHECKLIST:
[ ] Check overall market sentiment (VIX, futures)
[ ] Verify no major news on target symbols
[ ] Confirm options liquidity and spreads
[ ] Set stop losses and profit targets
[ ] Review position sizing (max 3% risk per trade)

TRADE EXECUTION PLAN:
"""
        
        for trade in execution_plan['trades']:
            checklist += f"""
[ ] {trade['trade_id']}
   Symbol: {trade['symbol']}
   Strategy: {trade['strategy']}
   Strike: ${trade['strike_price']}
   Max Risk: ${trade['max_loss']:,.0f}
   Target: ${trade['target_profit']:,.0f}
   
"""
        
        checklist += """
POST-TRADE CHECKLIST:
[ ] Confirm all orders filled at acceptable prices
[ ] Set GTC stop loss orders
[ ] Set profit target alerts
[ ] Update trade journal with entry reasons
[ ] Monitor positions throughout day

RISK MANAGEMENT REMINDERS:
[ ] Never risk more than 3% per trade
[ ] Never risk more than 10% per day
[ ] Cut losses at 25-30%
[ ] Take profits at 50-100% gains
[ ] Review and adjust position sizes daily

TARGET: 20%+ MONTHLY RETURNS WITH DISCIPLINED RISK MANAGEMENT
"""
        
        # Save checklist
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"trading_checklist_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(checklist)
        
        print(f"\nTrading checklist saved: {filename}")
        
        return checklist

def main():
    """Run the edge execution system."""
    
    # Initialize system
    executor = EdgeExecutionSystem(account_size=100000)  # Adjust your account size
    
    # Generate trade signals
    signals = executor.generate_trade_signals()
    
    if signals:
        # Create execution plan
        execution_plan = executor.create_execution_plan(signals)
        
        # Generate trading checklist
        checklist = executor.generate_trading_checklist(execution_plan)
        
        print(f"\n🎯 READY TO EXECUTE {len(signals)} HIGH-EDGE TRADES")
        print("Next steps:")
        print("1. Review execution plan JSON file")
        print("2. Follow pre-market checklist")
        print("3. Execute trades at market open")
        print("4. Monitor positions and manage risk")
        
    else:
        print("\nNo high-edge opportunities found at this time.")
        print("System will continue monitoring for new setups.")

if __name__ == "__main__":
    main()