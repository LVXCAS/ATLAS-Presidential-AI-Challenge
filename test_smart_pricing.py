#!/usr/bin/env python3
"""
Test Smart Pricing Agent - Verify Intelligent Price Optimization
"""

import sys
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('.')

from agents.smart_pricing_agent import smart_pricing_agent, PricingContext
from agents.broker_integration import OrderSide

def test_pricing_scenarios():
    """Test various pricing scenarios"""
    
    print("TESTING SMART PRICING AGENT")
    print("=" * 60)
    
    # Test scenarios with different market conditions
    scenarios = [
        {
            'name': 'Liquid ATM Call - Tight Spread',
            'context': PricingContext(
                symbol='AAPL250919C00225000',
                underlying_price=225.0,
                bid=8.50,
                ask=8.80,  # 30 cent spread = 3.5%
                volume=500,
                open_interest=2000,
                spread_pct=3.5,
                time_to_expiry_days=20,
                volatility=0.25,
                delta=0.55,
                gamma=0.03,
                theta=-0.08,
                vega=0.25
            ),
            'side': OrderSide.BUY,
            'confidence': 0.7
        },
        {
            'name': 'Wide Spread OTM Put',
            'context': PricingContext(
                symbol='TSLA250919P00180000',
                underlying_price=200.0,
                bid=2.00,
                ask=3.50,  # $1.50 spread = 55%
                volume=50,
                open_interest=200,
                spread_pct=55.0,
                time_to_expiry_days=15,
                volatility=0.45,
                delta=-0.25,
                gamma=0.02,
                theta=-0.12,
                vega=0.20
            ),
            'side': OrderSide.BUY,
            'confidence': 0.5
        },
        {
            'name': 'Close to Expiry - High Theta',
            'context': PricingContext(
                symbol='SPY250905C00450000',
                underlying_price=450.0,
                bid=1.20,
                ask=1.35,  # 15 cent spread = 11.8%
                volume=300,
                open_interest=800,
                spread_pct=11.8,
                time_to_expiry_days=2,  # Very close to expiry
                volatility=0.18,
                delta=0.45,
                gamma=0.08,  # High gamma
                theta=-0.25,  # High theta decay
                vega=0.05
            ),
            'side': OrderSide.BUY,
            'confidence': 0.8
        },
        {
            'name': 'High Confidence Long-Dated',
            'context': PricingContext(
                symbol='NVDA251219C00400000',
                underlying_price=450.0,
                bid=45.00,
                ask=47.00,  # $2 spread = 4.3%
                volume=150,
                open_interest=1500,
                spread_pct=4.3,
                time_to_expiry_days=45,
                volatility=0.35,
                delta=0.65,
                gamma=0.015,
                theta=-0.05,
                vega=0.40
            ),
            'side': OrderSide.BUY,
            'confidence': 0.9  # Very high confidence
        },
        {
            'name': 'Exit Pricing - Profitable Position',
            'context': PricingContext(
                symbol='AAPL250919C00220000',
                underlying_price=235.0,  # Stock moved up $10
                bid=12.50,
                ask=13.00,
                volume=400,
                open_interest=1200,
                spread_pct=3.8,
                time_to_expiry_days=15,
                volatility=0.28,
                delta=0.75,
                gamma=0.025,
                theta=-0.10,
                vega=0.18
            ),
            'side': OrderSide.SELL,  # Selling to close
            'confidence': 0.6,
            'is_exit': True,
            'position_pnl_pct': 60.0,  # 60% profit
            'time_held_hours': 48.0
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\nTesting: {scenario['name']}")
        print(f"  Spread: {scenario['context'].spread_pct:.1f}% | "
              f"Volume: {scenario['context'].volume} | "
              f"DTE: {scenario['context'].time_to_expiry_days} | "
              f"Confidence: {scenario['confidence']:.1%}")
        
        if scenario.get('is_exit'):
            # Test exit pricing
            smart_price = smart_pricing_agent.determine_optimal_exit_price(
                scenario['context'],
                scenario['side'],
                scenario['position_pnl_pct'],
                scenario['time_held_hours']
            )
        else:
            # Test entry pricing
            smart_price = smart_pricing_agent.determine_optimal_entry_price(
                scenario['context'],
                scenario['side'],
                scenario['confidence']
            )
        
        # Calculate potential savings vs market order
        market_price = scenario['context'].ask if scenario['side'] == OrderSide.BUY else scenario['context'].bid
        potential_savings = abs(market_price - smart_price.target_price) * 100  # Per contract
        
        print(f"  Market Price: ${market_price:.2f}")
        print(f"  Smart Target: ${smart_price.target_price:.2f}")
        print(f"  Order Type: {smart_price.order_type}")
        print(f"  Fill Probability: {smart_price.expected_fill_probability:.0%}")
        print(f"  Savings per Contract: ${potential_savings:.0f}")
        print(f"  Reasoning: {smart_price.reasoning}")
        
        results.append({
            'scenario': scenario['name'],
            'order_type': smart_price.order_type,
            'savings_per_contract': potential_savings,
            'fill_probability': smart_price.expected_fill_probability,
            'reasoning': smart_price.reasoning
        })
    
    # Summary analysis
    print(f"\n" + "=" * 60)
    print(f"SMART PRICING ANALYSIS")
    print(f"=" * 60)
    
    total_savings = sum(r['savings_per_contract'] for r in results)
    avg_savings = total_savings / len(results)
    
    limit_orders = len([r for r in results if r['order_type'] == 'LIMIT'])
    market_orders = len([r for r in results if r['order_type'] == 'MARKET'])
    
    avg_fill_prob = sum(r['fill_probability'] for r in results) / len(results)
    
    print(f"Total Scenarios: {len(results)}")
    print(f"Average Savings per Contract: ${avg_savings:.0f}")
    print(f"Total Savings (all scenarios): ${total_savings:.0f}")
    print(f"Limit Orders: {limit_orders}/{len(results)} ({limit_orders/len(results)*100:.1f}%)")
    print(f"Market Orders: {market_orders}/{len(results)} ({market_orders/len(results)*100:.1f}%)")
    print(f"Average Fill Probability: {avg_fill_prob:.0%}")
    
    # Monthly impact calculation
    monthly_trades = 22  # 22 trading days
    monthly_savings = avg_savings * monthly_trades
    
    print(f"\nMONTHLY IMPACT PROJECTION:")
    print(f"  Estimated Monthly Trades: {monthly_trades}")
    print(f"  Monthly Savings per Contract: ${monthly_savings:.0f}")
    print(f"  Annual Savings per Contract: ${monthly_savings * 12:.0f}")
    
    # ROI improvement calculation
    avg_trade_size = 5000  # $5000 average trade
    roi_improvement = (monthly_savings / avg_trade_size) * 100
    
    print(f"\nROI IMPROVEMENT:")
    print(f"  Average Trade Size: ${avg_trade_size:,.0f}")
    print(f"  Monthly ROI Improvement: {roi_improvement:.2f}%")
    print(f"  Annual ROI Improvement: {roi_improvement * 12:.1f}%")
    
    # Assessment
    if avg_savings > 30 and limit_orders > len(results) * 0.6:
        assessment = "EXCELLENT - Smart pricing provides significant edge"
        grade = "A+"
    elif avg_savings > 20 and limit_orders > len(results) * 0.4:
        assessment = "VERY GOOD - Meaningful cost savings"
        grade = "A"
    elif avg_savings > 10:
        assessment = "GOOD - Modest improvements"
        grade = "B+"
    else:
        assessment = "NEEDS WORK - Limited benefits"
        grade = "C"
    
    print(f"\nOVERALL ASSESSMENT:")
    print(f"  Grade: {grade}")
    print(f"  Assessment: {assessment}")
    
    if grade in ['A+', 'A']:
        print(f"\n[SUCCESS] Smart pricing significantly improves your returns!")
        print(f"  Your bot now makes intelligent pricing decisions")
        print(f"  Expected additional return: +{roi_improvement * 12:.1f}% annually")
    else:
        print(f"\n[NEEDS IMPROVEMENT] Consider more aggressive pricing strategies")
    
    print(f"=" * 60)

def test_dynamic_stops_and_targets():
    """Test dynamic stop loss and take profit calculations"""
    
    print(f"\nTESTING DYNAMIC STOPS & TARGETS")
    print("=" * 60)
    
    test_context = PricingContext(
        symbol='AAPL250919C00225000',
        underlying_price=225.0,
        bid=8.50,
        ask=8.80,
        volume=500,
        open_interest=2000,
        spread_pct=3.5,
        time_to_expiry_days=20,
        volatility=0.25,
        delta=0.55,
        gamma=0.03,
        theta=-0.08,
        vega=0.25
    )
    
    entry_price = 8.65
    position_side = OrderSide.BUY
    
    # Test dynamic stop loss
    stop_price, stop_reasoning = smart_pricing_agent.create_dynamic_stop_loss(
        test_context, entry_price, position_side, base_stop_pct=0.30
    )
    
    # Test dynamic take profit
    target_price, target_reasoning = smart_pricing_agent.create_dynamic_take_profit(
        test_context, entry_price, position_side, base_target_pct=0.50
    )
    
    stop_loss_pct = (entry_price - stop_price) / entry_price * 100
    take_profit_pct = (target_price - entry_price) / entry_price * 100
    
    print(f"Entry Price: ${entry_price:.2f}")
    print(f"Stop Loss: ${stop_price:.2f} ({stop_loss_pct:.1f}% loss)")
    print(f"Stop Reasoning: {stop_reasoning}")
    print(f"Take Profit: ${target_price:.2f} ({take_profit_pct:.1f}% gain)")
    print(f"Target Reasoning: {target_reasoning}")
    
    # Risk/Reward ratio
    risk_reward_ratio = take_profit_pct / stop_loss_pct
    print(f"Risk/Reward Ratio: 1:{risk_reward_ratio:.1f}")
    
    if risk_reward_ratio >= 2.0:
        print("[EXCELLENT] Great risk/reward ratio!")
    elif risk_reward_ratio >= 1.5:
        print("[GOOD] Acceptable risk/reward ratio")
    else:
        print("[CAUTION] Low risk/reward ratio")
    
    print("=" * 60)

if __name__ == "__main__":
    test_pricing_scenarios()
    test_dynamic_stops_and_targets()