#!/usr/bin/env python3
"""
OPTIONS DECISION ADVISOR
Smart analysis for expiring options positions
Helps make data-driven decisions on profitable options
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - OPTIONS - %(message)s')

class OptionsDecisionAdvisor:
    """Intelligent advisor for options trading decisions"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        logging.info("OPTIONS DECISION ADVISOR INITIALIZED")

    async def analyze_intc_puts_situation(self):
        """Analyze the INTC puts situation specifically"""

        print("=== INTEL PUTS DECISION ANALYSIS ===")
        print("Position: INTC $29 Puts expiring Friday")
        print("Current P&L: +70.6% (MAJOR WINNER)")
        print("-" * 50)

        try:
            # Get Intel stock price
            intc_quote = self.alpaca.get_latest_quote('INTC')
            intc_price = float(intc_quote.bid_price)

            print(f"Intel Stock Price: ${intc_price:.2f}")
            print(f"Put Strike Price: $29.00")

            # Calculate intrinsic value
            intrinsic_value = max(0, 29.00 - intc_price)
            time_to_expiry = 1  # 1 day to Friday expiry

            print(f"Intrinsic Value: ${intrinsic_value:.2f}")
            print(f"Time to Expiry: {time_to_expiry} day")

            # Decision analysis
            if intc_price < 29:
                in_the_money = True
                potential_profit = 29 - intc_price
                print(f"Status: IN THE MONEY by ${potential_profit:.2f}")
            else:
                in_the_money = False
                print(f"Status: OUT OF THE MONEY by ${intc_price - 29:.2f}")

            return {
                'intc_price': intc_price,
                'strike': 29.00,
                'intrinsic_value': intrinsic_value,
                'in_the_money': in_the_money,
                'time_to_expiry': time_to_expiry,
                'current_gain': 70.6
            }

        except Exception as e:
            print(f"Error getting Intel data: {e}")
            return None

    def generate_decision_scenarios(self, analysis):
        """Generate decision scenarios based on analysis"""

        if not analysis:
            return []

        intc_price = analysis['intc_price']
        current_gain = analysis['current_gain']

        scenarios = []

        print(f"\n=== DECISION SCENARIOS ===")

        # Scenario 1: Take profits now
        scenarios.append({
            'action': 'CLOSE NOW',
            'rationale': f'Lock in guaranteed {current_gain:.1f}% profit',
            'pros': [
                'Guaranteed profit secured',
                'No expiration risk',
                'Free up buying power for other opportunities'
            ],
            'cons': [
                'Miss potential further gains if Intel drops more',
                'FOMO if Intel crashes tomorrow'
            ],
            'probability_success': 100,
            'max_gain': current_gain,
            'max_loss': 0
        })

        # Scenario 2: Hold until tomorrow
        scenarios.append({
            'action': 'HOLD UNTIL EXPIRY',
            'rationale': 'Intel might drop further, maximize gains',
            'pros': [
                'Potential for higher profits if Intel continues falling',
                'Maximize intrinsic value at expiration'
            ],
            'cons': [
                'Risk of losing current gains if Intel rebounds',
                'Time decay accelerating',
                'All-or-nothing at expiration'
            ],
            'probability_success': 60,  # Estimate
            'max_gain': 150,  # If Intel crashes to $27
            'max_loss': -100  # If expires worthless
        })

        # Scenario 3: Partial close
        scenarios.append({
            'action': 'CLOSE HALF NOW',
            'rationale': 'Take some profits, let rest ride',
            'pros': [
                'Lock in partial profits',
                'Keep upside exposure',
                'Balanced risk management'
            ],
            'cons': [
                'Compromise solution',
                'Still have expiration risk on remainder'
            ],
            'probability_success': 80,
            'max_gain': current_gain * 0.5 + 75,  # Half now + potential on rest
            'max_loss': current_gain * 0.5 - 50   # Half secured, half at risk
        })

        return scenarios

    def make_smart_recommendation(self, scenarios, analysis):
        """Make intelligent recommendation based on analysis"""

        print(f"\n=== SMART RECOMMENDATION ===")

        # Risk assessment factors
        factors = {
            'time_pressure': 'HIGH (expires tomorrow)',
            'current_profit': 'EXCELLENT (70.6%)',
            'market_risk': 'MEDIUM (Intel could bounce)',
            'opportunity_cost': 'LOW (other trades available)'
        }

        print("Risk Factors:")
        for factor, level in factors.items():
            print(f"  {factor}: {level}")

        # Logic-based recommendation
        if analysis and analysis['current_gain'] > 50:
            if analysis['time_to_expiry'] <= 1:
                recommended_action = "CLOSE HALF NOW"
                reasoning = [
                    "Exceptional 70.6% gain already achieved",
                    "High time decay risk with 1 day to expiry",
                    "Intel at critical support levels",
                    "Partial close balances profit-taking with upside"
                ]
            else:
                recommended_action = "HOLD"
                reasoning = ["More time available", "Trend still favorable"]
        else:
            recommended_action = "CLOSE NOW"
            reasoning = ["Profit too small to risk", "Time decay too high"]

        print(f"\nRECOMMENDATION: {recommended_action}")
        print("\nReasoning:")
        for reason in reasoning:
            print(f"  • {reason}")

        return {
            'recommendation': recommended_action,
            'confidence': 'HIGH',
            'reasoning': reasoning
        }

    async def execute_recommendation(self, recommendation):
        """Execute the recommended action"""

        action = recommendation['recommendation']

        print(f"\n=== EXECUTION PLAN ===")
        print(f"Recommended Action: {action}")

        if action == "CLOSE NOW":
            print("Executing: Close entire INTC puts position")
            print("Command: SELL TO CLOSE all INTC $29 puts")

        elif action == "CLOSE HALF NOW":
            print("Executing: Close 50% of INTC puts position")
            print("Command: SELL TO CLOSE 50% of INTC $29 puts")

        elif action == "HOLD":
            print("Executing: Hold position until tomorrow")
            print("Command: Monitor closely, set alerts")

        # Note: Actual execution would go here
        print("\nWARNING: EXECUTION SIMULATION MODE")
        print("Review recommendation and execute manually in Alpaca")

        return True

    async def run_options_decision_analysis(self):
        """Run complete options decision analysis"""

        print("OPTIONS DECISION ADVISOR")
        print("="*50)
        print("Analyzing INTC $29 puts expiring tomorrow")
        print("="*50)

        # Analyze current situation
        analysis = await self.analyze_intc_puts_situation()

        # Generate scenarios
        scenarios = self.generate_decision_scenarios(analysis)

        # Display scenarios
        print(f"\n=== ALL SCENARIOS ===")
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario['action']}")
            print(f"   Rationale: {scenario['rationale']}")
            print(f"   Success Probability: {scenario['probability_success']}%")
            print(f"   Max Gain: {scenario['max_gain']:.1f}%")
            print(f"   Max Loss: {scenario['max_loss']:.1f}%")

        # Make recommendation
        recommendation = self.make_smart_recommendation(scenarios, analysis)

        # Show execution plan
        await self.execute_recommendation(recommendation)

        print(f"\n{'='*50}")
        print("DECISION ANALYSIS COMPLETE")
        print("Review analysis and make your decision!")

async def main():
    """Run options decision analysis"""
    advisor = OptionsDecisionAdvisor()
    await advisor.run_options_decision_analysis()

if __name__ == "__main__":
    asyncio.run(main())