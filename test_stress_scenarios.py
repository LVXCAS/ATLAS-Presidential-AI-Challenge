#!/usr/bin/env python3
"""
Stress Test Scenarios for OPTIONS_BOT
Tests bot performance under extreme market conditions
"""

import asyncio
import sys
import random
import numpy as np
from datetime import datetime, timedelta
sys.path.append('.')

from OPTIONS_BOT import TomorrowReadyOptionsBot
from agents.exit_strategy_agent import ExitSignal, ExitReason, ExitDecision

async def stress_test_scenarios():
    """Run stress tests on OPTIONS_BOT under extreme conditions"""
    print("STRESS TEST SCENARIOS - OPTIONS_BOT")
    print("=" * 60)
    
    bot = TomorrowReadyOptionsBot()
    
    # Initialize bot
    try:
        await bot.initialize_all_systems()
        print(f"[OK] Bot initialized - Account: ${bot.risk_manager.account_value:,.2f}")
    except Exception as e:
        print(f"[FAIL] Bot initialization failed: {e}")
        return
    
    # Define extreme stress scenarios
    stress_scenarios = [
        {
            "name": "Market Crash (-30% day)",
            "momentum": -0.30,
            "volatility": 80.0,
            "volume_ratio": 5.0,
            "description": "Black Monday style crash"
        },
        {
            "name": "Flash Crash (-10% in minutes)",
            "momentum": -0.10,
            "volatility": 150.0,
            "volume_ratio": 10.0,
            "description": "Rapid algorithmic selloff"
        },
        {
            "name": "Melt-Up (+15% day)",
            "momentum": 0.15,
            "volatility": 60.0,
            "volume_ratio": 3.0,
            "description": "Euphoric buying frenzy"
        },
        {
            "name": "Volatility Crush",
            "momentum": 0.005,
            "volatility": 5.0,
            "volume_ratio": 0.3,
            "description": "Post-earnings vol collapse"
        },
        {
            "name": "Circuit Breaker",
            "momentum": -0.20,
            "volatility": 100.0,
            "volume_ratio": 8.0,
            "description": "Trading halt scenario"
        },
        {
            "name": "Liquidity Crisis",
            "momentum": -0.05,
            "volatility": 45.0,
            "volume_ratio": 0.1,
            "description": "No buyers, wide spreads"
        },
        {
            "name": "FOMC Surprise",
            "momentum": 0.08,
            "volatility": 35.0,
            "volume_ratio": 4.0,
            "description": "Unexpected rate decision"
        }
    ]
    
    print(f"\nTesting {len(stress_scenarios)} extreme market scenarios...\n")
    
    stress_results = []
    
    for i, scenario in enumerate(stress_scenarios, 1):
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Conditions: {scenario['momentum']:+.1%} momentum, {scenario['volatility']:.1f}% vol, {scenario['volume_ratio']:.1f}x volume")
        
        # Create market data for this scenario
        market_data = {
            'symbol': 'SPY',  # Use SPY as representative
            'current_price': 400.0,
            'price_momentum': scenario['momentum'],
            'realized_vol': scenario['volatility'],
            'volume_ratio': scenario['volume_ratio'],
            'price_position': 0.5,
            'avg_volume': 50000000,
            'timestamp': datetime.now()
        }
        
        scenario_result = {
            'name': scenario['name'],
            'conditions': scenario,
            'opportunity_found': False,
            'opportunity_details': None,
            'exit_decisions': [],
            'risk_assessment': 'UNKNOWN'
        }
        
        try:
            # Test opportunity detection
            print("  Testing opportunity detection...")
            opportunity = await bot.find_high_quality_opportunity('SPY')
            
            if opportunity:
                scenario_result['opportunity_found'] = True
                scenario_result['opportunity_details'] = opportunity
                print(f"    [OK] Opportunity found: {opportunity['strategy']} ({opportunity['confidence']:.0%} confidence)")
                print(f"    Max profit: ${opportunity['max_profit']:.2f}, Max loss: ${opportunity['max_loss']:.2f}")
                
                # Test position risk under stress
                position_risk = opportunity['max_loss'] * 100
                risk_limit = bot.daily_risk_limits.get('max_position_risk', 500)
                
                if position_risk <= risk_limit:
                    scenario_result['risk_assessment'] = 'ACCEPTABLE'
                    print(f"    [OK] Risk acceptable: ${position_risk:.2f} <= ${risk_limit:.2f}")
                else:
                    scenario_result['risk_assessment'] = 'TOO_HIGH'
                    print(f"    [FAIL] Risk too high: ${position_risk:.2f} > ${risk_limit:.2f}")
                
                # Test exit strategy under stress conditions
                print("  Testing exit strategy under stress...")
                exit_scenarios = [
                    (150.0, "Major profit"),
                    (-75.0, "Large loss"),
                    (25.0, "Small profit"),
                    (-125.0, "Catastrophic loss")
                ]
                
                for pnl, desc in exit_scenarios:
                    mock_position = {
                        'entry_time': datetime.now() - timedelta(hours=2),
                        'opportunity': opportunity,
                        'entry_price': 2.50,
                        'quantity': 1,
                        'market_regime_at_entry': 'BULL'
                    }
                    
                    try:
                        exit_decision = bot.exit_agent.analyze_position_exit(
                            mock_position, market_data, pnl
                        )
                        
                        scenario_result['exit_decisions'].append({
                            'pnl': pnl,
                            'description': desc,
                            'signal': exit_decision.signal,
                            'reason': exit_decision.reason,
                            'confidence': exit_decision.confidence,
                            'urgency': exit_decision.urgency
                        })
                        
                        print(f"    {desc} (${pnl:+.0f}): {exit_decision.signal} ({exit_decision.confidence:.0%} confidence)")
                    
                    except Exception as e:
                        print(f"    {desc}: Exit analysis failed - {e}")
            
            else:
                print("    [FAIL] No opportunity found")
                scenario_result['risk_assessment'] = 'NO_TRADE'
        
        except Exception as e:
            print(f"    [FAIL] Scenario test failed: {e}")
            scenario_result['risk_assessment'] = 'ERROR'
        
        stress_results.append(scenario_result)
        print()  # Empty line between scenarios
    
    # Analyze stress test results
    print("=" * 60)
    print("STRESS TEST ANALYSIS")
    print("=" * 60)
    
    opportunities_found = sum(1 for r in stress_results if r['opportunity_found'])
    acceptable_risk = sum(1 for r in stress_results if r['risk_assessment'] == 'ACCEPTABLE')
    
    print(f"\nOPPORTUNITY DETECTION:")
    print(f"  Scenarios tested: {len(stress_scenarios)}")
    print(f"  Opportunities found: {opportunities_found}")
    print(f"  Detection rate: {opportunities_found/len(stress_scenarios):.1%}")
    
    print(f"\nRISK MANAGEMENT:")
    print(f"  Acceptable risk trades: {acceptable_risk}")
    print(f"  Risk control rate: {acceptable_risk/len(stress_scenarios):.1%}")
    
    print(f"\nEXIT STRATEGY ANALYSIS:")
    all_exits = []
    for result in stress_results:
        all_exits.extend(result['exit_decisions'])
    
    if all_exits:
        # Analyze exit signals by P&L scenario
        profit_exits = [e for e in all_exits if e['pnl'] > 0]
        loss_exits = [e for e in all_exits if e['pnl'] < 0]
        
        print(f"  Total exit decisions analyzed: {len(all_exits)}")
        
        if profit_exits:
            take_profit_rate = sum(1 for e in profit_exits if e['signal'] == ExitSignal.TAKE_PROFIT) / len(profit_exits)
            print(f"  Profit scenarios - Take profit rate: {take_profit_rate:.1%}")
        
        if loss_exits:
            stop_loss_rate = sum(1 for e in loss_exits if e['signal'] == ExitSignal.STOP_LOSS) / len(loss_exits)
            print(f"  Loss scenarios - Stop loss rate: {stop_loss_rate:.1%}")
    
    print(f"\nSCENARIO-SPECIFIC RESULTS:")
    for result in stress_results:
        status = "FOUND" if result['opportunity_found'] else "NO OPPORTUNITY"
        risk = result['risk_assessment']
        print(f"  {result['name']}: {status}, Risk: {risk}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if opportunities_found / len(stress_scenarios) > 0.7:
        print("  [OK] Bot shows good opportunity detection even under stress")
    elif opportunities_found / len(stress_scenarios) > 0.3:
        print("  ⚠ Bot has moderate opportunity detection under stress")
    else:
        print("  [FAIL] Bot may be too conservative under extreme conditions")
    
    if acceptable_risk / len(stress_scenarios) > 0.6:
        print("  [OK] Risk management appears robust under stress")
    else:
        print("  ⚠ Consider reviewing risk limits for extreme conditions")
    
    print(f"\n" + "=" * 60)
    print("STRESS TEST COMPLETE")
    
    return stress_results

if __name__ == "__main__":
    asyncio.run(stress_test_scenarios())