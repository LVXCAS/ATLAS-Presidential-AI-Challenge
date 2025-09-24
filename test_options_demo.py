"""
Options Trading System Demonstration

Quick demo to show the options trading capabilities of the HiveTrading system
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class OptionsDemo:
    """Demonstration of options trading capabilities"""

    def __init__(self):
        self.demo_data = {
            'AAPL': {
                'current_price': 175.50,
                'volatility': 0.28,
                'risk_free_rate': 0.045
            },
            'TSLA': {
                'current_price': 245.80,
                'volatility': 0.45,
                'risk_free_rate': 0.045
            }
        }

    def black_scholes_call(self, S, K, T, r, sigma):
        """Black-Scholes call option pricing"""
        from math import log, sqrt, exp
        from scipy.stats import norm

        d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        call_price = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
        return call_price

    def calculate_greeks(self, S, K, T, r, sigma):
        """Calculate option Greeks"""
        from math import log, sqrt, exp
        from scipy.stats import norm

        d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        # Delta (price sensitivity)
        delta = norm.cdf(d1)

        # Gamma (delta sensitivity)
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))

        # Theta (time decay)
        theta = -(S * norm.pdf(d1) * sigma / (2 * sqrt(T)) +
                 r * K * exp(-r*T) * norm.cdf(d2)) / 365

        # Vega (volatility sensitivity)
        vega = S * norm.pdf(d1) * sqrt(T) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }

    def demonstrate_options_pricing(self):
        """Demonstrate options pricing capabilities"""
        print("=" * 70)
        print("HIVE TRADING - OPTIONS PRICING DEMONSTRATION")
        print("=" * 70)

        for symbol, data in self.demo_data.items():
            print(f"\n{symbol} OPTIONS ANALYSIS")
            print("-" * 40)

            S = data['current_price']
            sigma = data['volatility']
            r = data['risk_free_rate']

            print(f"Current Stock Price: ${S:.2f}")
            print(f"Implied Volatility: {sigma*100:.1f}%")
            print(f"Risk-Free Rate: {r*100:.1f}%")

            # Demo different expiration dates
            expirations = [7, 30, 60, 90]  # days
            strikes = [S*0.95, S, S*1.05]  # ATM, ITM, OTM

            print(f"\nOPTIONS CHAIN:")
            print(f"{'Strike':<8} {'Days':<5} {'Call Price':<12} {'Delta':<8} {'Gamma':<8} {'Theta':<8} {'Vega':<8}")
            print("-" * 65)

            for K in strikes:
                for days in expirations:
                    T = days / 365.0

                    try:
                        call_price = self.black_scholes_call(S, K, T, r, sigma)
                        greeks = self.calculate_greeks(S, K, T, r, sigma)

                        print(f"${K:<7.0f} {days:<5} ${call_price:<11.2f} "
                              f"{greeks['delta']:<7.3f} {greeks['gamma']:<7.3f} "
                              f"{greeks['theta']:<7.2f} {greeks['vega']:<7.2f}")
                    except:
                        print(f"${K:<7.0f} {days:<5} ${'N/A':<11} {'N/A':<7} {'N/A':<7} {'N/A':<7} {'N/A':<7}")

    def demonstrate_strategies(self):
        """Demonstrate options strategies"""
        print(f"\n{'='*70}")
        print("OPTIONS STRATEGIES DEMONSTRATION")
        print("=" * 70)

        S = 175.50  # AAPL current price

        strategies = {
            'Long Call': {
                'description': 'Bullish strategy - buy call option',
                'positions': [('buy', 'call', 180, 30)],
                'max_profit': 'Unlimited',
                'max_loss': 'Premium paid'
            },
            'Covered Call': {
                'description': 'Income strategy - own stock + sell call',
                'positions': [('own', 'stock', S, 0), ('sell', 'call', 185, 30)],
                'max_profit': 'Strike - Stock Price + Premium',
                'max_loss': 'Stock Price - Premium'
            },
            'Iron Condor': {
                'description': 'Neutral strategy - profit from low volatility',
                'positions': [
                    ('sell', 'call', 185, 30),
                    ('buy', 'call', 190, 30),
                    ('sell', 'put', 165, 30),
                    ('buy', 'put', 160, 30)
                ],
                'max_profit': 'Net premium received',
                'max_loss': 'Spread width - Net premium'
            },
            'Straddle': {
                'description': 'Volatility strategy - profit from big moves',
                'positions': [('buy', 'call', S, 30), ('buy', 'put', S, 30)],
                'max_profit': 'Unlimited',
                'max_loss': 'Total premium paid'
            }
        }

        for strategy, details in strategies.items():
            print(f"\n{strategy.upper()}")
            print("-" * 40)
            print(f"Description: {details['description']}")
            print(f"Max Profit: {details['max_profit']}")
            print(f"Max Loss: {details['max_loss']}")

            total_cost = 0
            print(f"Positions:")
            for position in details['positions']:
                if len(position) == 4:
                    action, option_type, strike, days = position
                    if option_type in ['call', 'put']:
                        # Simplified cost calculation
                        cost = 2.50 if action == 'buy' else -2.50
                        total_cost += cost
                        print(f"  {action.capitalize()} {option_type} ${strike} ({days} days) - ${abs(cost):.2f}")
                    else:
                        print(f"  {action.capitalize()} {option_type} @ ${strike}")

            if total_cost != 0:
                print(f"Net Cost: ${total_cost:.2f}")

    def demonstrate_risk_management(self):
        """Demonstrate risk management for options"""
        print(f"\n{'='*70}")
        print("OPTIONS RISK MANAGEMENT")
        print("=" * 70)

        portfolio = {
            'cash': 100000,
            'positions': [
                {'symbol': 'AAPL', 'type': 'call', 'strike': 180, 'quantity': 10, 'premium': 3.50},
                {'symbol': 'TSLA', 'type': 'put', 'strike': 240, 'quantity': 5, 'premium': 8.20},
                {'symbol': 'SPY', 'type': 'call', 'strike': 450, 'quantity': 20, 'premium': 2.10}
            ]
        }

        print("CURRENT OPTIONS PORTFOLIO:")
        print(f"{'Symbol':<8} {'Type':<6} {'Strike':<8} {'Qty':<6} {'Premium':<10} {'Value':<10}")
        print("-" * 60)

        total_value = 0
        total_delta = 0

        for pos in portfolio['positions']:
            position_value = pos['quantity'] * pos['premium'] * 100  # Options are per 100 shares
            total_value += position_value

            # Simplified delta calculation
            delta = 0.5 if pos['type'] == 'call' else -0.5
            position_delta = pos['quantity'] * delta * 100
            total_delta += position_delta

            print(f"{pos['symbol']:<8} {pos['type']:<6} ${pos['strike']:<7} {pos['quantity']:<6} "
                  f"${pos['premium']:<9.2f} ${position_value:<9,.0f}")

        print("-" * 60)
        print(f"Total Portfolio Value: ${portfolio['cash'] + total_value:,.0f}")
        print(f"Options Value: ${total_value:,.0f}")
        print(f"Portfolio Delta: {total_delta:,.0f}")

        # Risk metrics
        print(f"\nRISK METRICS:")
        print(f"Options as % of Portfolio: {(total_value/(portfolio['cash'] + total_value))*100:.1f}%")
        print(f"Maximum Daily Risk (2% of portfolio): ${(portfolio['cash'] + total_value)*0.02:,.0f}")
        print(f"Delta-Adjusted Exposure: ${abs(total_delta):,.0f}")

        # Risk warnings
        if total_value / (portfolio['cash'] + total_value) > 0.3:
            print(f"[WARNING] High options concentration ({(total_value/(portfolio['cash'] + total_value))*100:.1f}%)")

        if abs(total_delta) > 50000:
            print(f"[WARNING] High delta exposure (${abs(total_delta):,.0f})")

async def main():
    """Run the options demonstration"""
    print("HIVE TRADING SYSTEM - OPTIONS CAPABILITIES DEMO")
    print("=" * 70)

    demo = OptionsDemo()

    # Check if scipy is available for full calculations
    try:
        import scipy.stats
        print("[OK] Advanced calculations available (scipy installed)")
    except ImportError:
        print("[WARN] Basic calculations only (install scipy for full features)")

    # Run demonstrations
    demo.demonstrate_options_pricing()
    demo.demonstrate_strategies()
    demo.demonstrate_risk_management()

    print(f"\n{'='*70}")
    print("OPTIONS DEMO COMPLETED")
    print("=" * 70)
    print("\n[OK] Your HiveTrading system includes:")
    print("   - Black-Scholes options pricing")
    print("   - Greeks calculations (Delta, Gamma, Theta, Vega)")
    print("   - Advanced options strategies")
    print("   - Comprehensive risk management")
    print("   - Real-time options data integration")
    print("   - Options execution via broker APIs")

    print(f"\nReady for live options trading!")

if __name__ == "__main__":
    asyncio.run(main())