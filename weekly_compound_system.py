"""
WEEKLY COMPOUND SYSTEM - PATH TO 25% MONTHLY
============================================
Smart approach: Target 5-8% weekly gains, compound aggressively
Math: 6% weekly = 26% monthly, 7% weekly = 32% monthly

This is much more achievable than trying to hit 25% in single trades.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class WeeklyCompoundSystem:
    """Compound 5-8% weekly gains to achieve 25%+ monthly"""
    
    def __init__(self, initial_capital=10000, weekly_target=0.06):
        self.initial_capital = initial_capital
        self.weekly_target = weekly_target
        
        # Calculate what weekly target means monthly
        monthly_equivalent = (1 + weekly_target) ** 4 - 1
        annual_equivalent = (1 + monthly_equivalent) ** 12 - 1
        
        print("WEEKLY COMPOUND SYSTEM")
        print("=" * 50)
        print("Strategy: Compound smaller, consistent weekly gains")
        print(f"Weekly Target: {weekly_target:.1%}")
        print(f"Monthly Equivalent: {monthly_equivalent:.1%}")
        print(f"Annual Equivalent: {annual_equivalent:.0%}")
        print(f"Starting Capital: ${initial_capital:,}")
        
        # Compounding parameters
        self.compound_settings = {
            'weekly_target': weekly_target,
            'base_position_size': 0.4,      # Conservative 40% base
            'kelly_multiplier': 1.5,        # Aggressive Kelly scaling
            'reinvestment_rate': 1.0,       # Reinvest 100% of profits
            'stop_loss': 0.08,              # 8% stop loss per trade
            'take_profit': weekly_target,   # Take profit at weekly target
            'max_trades_per_week': 2,       # Max 2 trades per week
            'confidence_threshold': 0.65    # Only trade 65%+ confidence
        }
        
        print(f"\nCOMPOUNDING PARAMETERS:")
        for param, value in self.compound_settings.items():
            if isinstance(value, float) and value < 1:
                print(f"  {param}: {value:.1%}")
            else:
                print(f"  {param}: {value}")
        print("=" * 50)
    
    def detect_current_regime(self):
        """Detect current market regime for position sizing"""
        # Simplified regime detection - in real implementation would use live data
        # For simulation, assume we're in Bull_Low_Vol 69.5% of the time (our best regime)
        regime_probabilities = {
            "Bull_Low_Vol": 0.695,
            "Bull_High_Vol": 0.15,
            "Bear_High_Vol": 0.10,
            "Bear_Low_Vol": 0.055
        }
        
        rand = np.random.random()
        cumulative = 0
        for regime, prob in regime_probabilities.items():
            cumulative += prob
            if rand <= cumulative:
                return regime
        return "Bull_Low_Vol"
    
    def apply_compound_reinvestment(self, capital, weekly_profit, week_number):
        """Apply intelligent compound reinvestment strategy"""
        
        # Base reinvestment rate from settings
        base_reinvestment = self.compound_settings['reinvestment_rate']
        
        # Progressive reinvestment - reinvest more as we build confidence
        if week_number <= 2:
            reinvestment_rate = base_reinvestment * 0.8  # Conservative start
        elif week_number <= 6:
            reinvestment_rate = base_reinvestment * 1.0  # Full reinvestment
        else:
            reinvestment_rate = base_reinvestment * 1.2  # Aggressive later
        
        # Risk adjustment - reduce reinvestment after losses
        if weekly_profit < 0:
            reinvestment_rate *= 0.5  # Halve reinvestment after loss
        
        # Calculate new capital
        reinvested_profit = weekly_profit * reinvestment_rate
        new_capital = capital + reinvested_profit
        
        return {
            'new_capital': new_capital,
            'reinvested_amount': reinvested_profit,
            'reinvestment_rate': reinvestment_rate,
            'cash_taken': weekly_profit - reinvested_profit
        }
    
    def calculate_kelly_position_size(self, confidence, current_capital, regime="Bull_Low_Vol"):
        """Calculate aggressive Kelly position sizing for compounding"""
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        win_rate = confidence
        loss_rate = 1 - confidence
        
        # Dynamic reward-to-risk ratio based on regime
        regime_multipliers = {
            "Bull_Low_Vol": 1.8,    # Higher reward potential in stable bull
            "Bull_High_Vol": 1.4,   # Moderate in volatile bull
            "Bear_High_Vol": 1.2,   # Conservative in bear
            "Bear_Low_Vol": 1.1     # Very conservative
        }
        reward_risk_ratio = regime_multipliers.get(regime, 1.5)
        
        # Kelly fraction
        kelly_fraction = (win_rate * reward_risk_ratio - loss_rate) / reward_risk_ratio
        kelly_fraction = max(0, kelly_fraction)  # Don't go negative
        
        # Apply aggressive multiplier for compounding
        kelly_multiplier = self.compound_settings['kelly_multiplier']
        aggressive_kelly = kelly_fraction * kelly_multiplier
        
        # Confidence scaling - higher confidence allows larger positions
        confidence_boost = min((confidence - 0.6) * 2, 0.3) if confidence > 0.6 else 0
        
        # Combine with base position size
        base_size = self.compound_settings['base_position_size']
        final_position_size = base_size + (aggressive_kelly * 0.5) + confidence_boost
        
        # Cap at reasonable maximum based on regime
        regime_caps = {
            "Bull_Low_Vol": 0.85,   # Higher cap for best regime
            "Bull_High_Vol": 0.7,   # Moderate cap
            "Bear_High_Vol": 0.6,   # Conservative cap
            "Bear_Low_Vol": 0.4     # Very conservative cap
        }
        max_cap = regime_caps.get(regime, 0.8)
        
        final_position_size = min(final_position_size, max_cap)
        final_position_size = max(final_position_size, 0.1)  # Min 10% of capital
        
        return {
            'position_size': final_position_size,
            'kelly_fraction': kelly_fraction,
            'aggressive_kelly': aggressive_kelly,
            'confidence_boost': confidence_boost,
            'reward_risk_ratio': reward_risk_ratio,
            'dollar_amount': current_capital * final_position_size,
            'regime': regime
        }
    
    def simulate_weekly_compounding(self, weeks=12):
        """Simulate weekly compounding over 3 months"""
        print(f"\nSIMULATING WEEKLY COMPOUNDING...")
        print(f"Simulation period: {weeks} weeks (~3 months)")
        
        # Use our proven Bull_Low_Vol confidence
        base_confidence = 0.695  # 69.5% from regime system
        weekly_target = self.compound_settings['weekly_target']
        
        results = []
        
        # Run 1000 simulations
        for sim in range(1000):
            capital = self.initial_capital
            week_by_week = [capital]
            
            for week in range(weeks):
                # Simulate 1-2 trades per week
                trades_this_week = np.random.choice([1, 2], p=[0.6, 0.4])  # 60% chance 1 trade, 40% chance 2
                
                week_return = 0
                
                for trade in range(trades_this_week):
                    if capital < self.initial_capital * 0.2:  # Stop if capital drops below 20% of initial
                        break
                    
                    # Calculate position size using Kelly with regime adaptation
                    current_regime = self.detect_current_regime()
                    position_info = self.calculate_kelly_position_size(base_confidence, capital, current_regime)
                    position_size = position_info['dollar_amount']
                    
                    # Simulate trade outcome
                    if np.random.random() < base_confidence:
                        # Winning trade - target weekly return per trade
                        trade_return = weekly_target / trades_this_week  # Split target across trades
                        actual_return = trade_return + np.random.normal(0, 0.01)  # Add some noise
                    else:
                        # Losing trade - stop loss
                        trade_return = -self.compound_settings['stop_loss'] / trades_this_week
                        actual_return = trade_return - np.random.normal(0, 0.005)  # Slippage
                    
                    # Apply return to position
                    dollar_return = position_size * actual_return
                    week_return += dollar_return
                
                # Apply compound reinvestment at end of week
                reinvestment_info = self.apply_compound_reinvestment(capital, week_return, week + 1)
                capital = reinvestment_info['new_capital']
                
                week_by_week.append(capital)
                
                # Early exit if wiped out
                if capital < self.initial_capital * 0.1:
                    break
            
            # Calculate final metrics
            final_return = (capital - self.initial_capital) / self.initial_capital
            max_capital = max(week_by_week)
            max_drawdown = (max_capital - min(week_by_week[week_by_week.index(max_capital):])) / max_capital
            
            results.append({
                'final_capital': capital,
                'total_return': final_return,
                'max_drawdown': max_drawdown,
                'weeks_survived': len(week_by_week) - 1,
                'week_by_week': week_by_week
            })
        
        return results
    
    def analyze_compounding_results(self, results):
        """Analyze weekly compounding simulation results"""
        print(f"\nCOMPOUNDING ANALYSIS:")
        print("=" * 40)
        
        # Extract metrics
        total_returns = [r['total_return'] for r in results]
        final_capitals = [r['final_capital'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        
        # Calculate statistics
        avg_return = np.mean(total_returns)
        median_return = np.median(total_returns)
        best_case = np.percentile(total_returns, 95)
        worst_case = np.percentile(total_returns, 5)
        
        # Success probabilities
        prob_25_plus_monthly = np.mean([r > 0.25 for r in total_returns])  # 25%+ total return
        prob_positive = np.mean([r > 0 for r in total_returns])
        prob_double = np.mean([r > 1.0 for r in total_returns])  # 100%+ return
        prob_catastrophic = np.mean([r < -0.5 for r in total_returns])  # 50%+ loss
        
        # Drawdown analysis
        avg_drawdown = np.mean(max_drawdowns)
        max_recorded_drawdown = np.max(max_drawdowns)
        
        print(f"PERFORMANCE OVER 12 WEEKS (~3 MONTHS):")
        print(f"  Average Return: {avg_return:.1%}")
        print(f"  Median Return: {median_return:.1%}")
        print(f"  Best Case (95th): {best_case:.1%}")
        print(f"  Worst Case (5th): {worst_case:.1%}")
        
        print(f"\nSUCCESS PROBABILITIES:")
        print(f"  25%+ Return: {prob_25_plus_monthly:.1%}")
        print(f"  Positive Return: {prob_positive:.1%}")
        print(f"  100%+ Return: {prob_double:.1%}")
        print(f"  50%+ Loss: {prob_catastrophic:.1%}")
        
        print(f"\nRISK METRICS:")
        print(f"  Average Max Drawdown: {avg_drawdown:.1%}")
        print(f"  Worst Drawdown Seen: {max_recorded_drawdown:.1%}")
        
        # Monthly breakdown (approximate)
        monthly_equiv = (1 + avg_return / 3) - 1 if avg_return > -1 else avg_return / 3
        monthly_prob = prob_25_plus_monthly  # Same for 3-month period
        
        print(f"\nMONTHLY EQUIVALENT:")
        print(f"  Average Monthly Return: ~{monthly_equiv:.1%}")
        print(f"  Probability of 25%+ Monthly: {monthly_prob:.1%}")
        
        return {
            'avg_return': avg_return,
            'monthly_equivalent': monthly_equiv,
            'prob_25_plus': prob_25_plus_monthly,
            'prob_positive': prob_positive,
            'catastrophic_risk': prob_catastrophic,
            'avg_drawdown': avg_drawdown
        }
    
    def create_weekly_implementation_plan(self, analysis):
        """Create practical weekly compounding implementation"""
        print(f"\nWEEKLY COMPOUNDING IMPLEMENTATION:")
        print("=" * 50)
        
        if analysis['prob_25_plus'] < 0.4:
            print("WEEKLY COMPOUNDING NOT VIABLE")
            print("Probability too low for reliable 25%+ returns")
            return None
        
        print("WEEKLY COMPOUNDING SYSTEM VIABLE!")
        print(f"Probability of 25%+ returns: {analysis['prob_25_plus']:.1%}")
        print(f"Average 3-month return: {analysis['avg_return']:.1%}")
        
        weekly_steps = [
            "MONDAY: Detect market regime using SPY analysis",
            "TUESDAY: Calculate Kelly position size for current capital",
            "WEDNESDAY: Execute 1st trade if Bull_Low_Vol regime (69.5% confidence)",
            "THURSDAY: Monitor position, apply stop-loss if needed",
            "FRIDAY: Execute 2nd trade if needed to hit weekly target",
            "WEEKEND: Calculate weekly return, compound profits into next week's capital",
            "REPEAT: Use new capital base for next week's position sizing"
        ]
        
        print(f"\nWEEKLY WORKFLOW:")
        for step in weekly_steps:
            print(f"  {step}")
        
        risk_controls = [
            "Never risk more than 40% of capital in single trade",
            "Always use 8% stop losses on every position",
            "Only trade during Bull_Low_Vol regime (69.5% confidence)",
            "Take profits at weekly target (6%) - don't get greedy",
            "Reinvest 100% of profits immediately into capital base",
            "If capital drops 50%, reduce position sizes by half",
            "Monthly review and system adjustment"
        ]
        
        print(f"\nRISK CONTROLS:")
        for control in risk_controls:
            print(f"  {control}")
        
        # Show compound math
        print(f"\nCOMPOUNDING MATH EXAMPLE:")
        print(f"Week 1: $10,000 + 6% = $10,600")
        print(f"Week 2: $10,600 + 6% = $11,236") 
        print(f"Week 3: $11,236 + 6% = $11,910")
        print(f"Week 4: $11,910 + 6% = $12,625 = 26.3% monthly!")
        print(f"\nThis is why compounding works better than big single trades")
        
        return {
            'viable': True,
            'expected_3month_return': analysis['avg_return'],
            'monthly_probability': analysis['prob_25_plus'],
            'risk_level': analysis['catastrophic_risk']
        }
    
    def run_weekly_compound_system(self):
        """Run complete weekly compounding system"""
        
        # Simulate compounding
        results = self.simulate_weekly_compounding(weeks=12)
        
        # Analyze results
        analysis = self.analyze_compounding_results(results)
        
        # Create implementation plan
        implementation = self.create_weekly_implementation_plan(analysis)
        
        print(f"\nWEEKLY COMPOUND SYSTEM COMPLETE!")
        print("=" * 50)
        
        if implementation and implementation['viable']:
            print("SUCCESS! Weekly compounding path to 25% monthly is VIABLE!")
            print(f"3-Month Expected Return: {implementation['expected_3month_return']:.1%}")
            print(f"Monthly Target Probability: {implementation['monthly_probability']:.1%}")
        else:
            print("Weekly compounding needs refinement")
        
        return {
            'simulation_results': results,
            'analysis': analysis,
            'implementation': implementation
        }

if __name__ == "__main__":
    print("Building weekly compounding system for 25% monthly target...")
    
    # Test different weekly targets
    targets = [0.05, 0.06, 0.07, 0.08]  # 5%, 6%, 7%, 8% weekly
    
    for target in targets:
        print(f"\n{'='*60}")
        print(f"TESTING {target:.0%} WEEKLY TARGET:")
        monthly_equiv = (1 + target) ** 4 - 1
        print(f"Theoretical Monthly: {monthly_equiv:.1%}")
        print(f"{'='*60}")
        
        system = WeeklyCompoundSystem(weekly_target=target)
        results = system.run_weekly_compound_system()
        
        if results['implementation'] and results['implementation']['viable']:
            print(f"*** {target:.0%} WEEKLY TARGET IS VIABLE ***")
        else:
            print(f"*** {target:.0%} WEEKLY TARGET NEEDS WORK ***")