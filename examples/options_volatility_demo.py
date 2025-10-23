#!/usr/bin/env python3
"""
Options Volatility Agent Demo

This demo showcases the Options Volatility Agent's capabilities:
1. IV surface analysis and skew detection
2. Earnings calendar integration
3. Greeks calculation and risk management
4. Volatility regime detection
5. Comprehensive signal generation with explainability

The demo uses realistic market scenarios to demonstrate how the agent
identifies options trading opportunities and manages risk.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.options_volatility_agent import (
    OptionsVolatilityAgent,
    OptionsData,
    VolatilityRegime,
    OptionsStrategy,
    options_volatility_agent_node
)

def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"[TARGET] {title}")
    print("=" * 60)

def print_subheader(title: str):
    """Print formatted subheader"""
    print(f"\n[CHART] {title}")
    print("-" * 40)

def create_realistic_options_chain(symbol: str, underlying_price: float, scenario: str = "normal") -> list:
    """Create realistic options chain based on market scenario"""
    options_data = []
    
    # Scenario-based parameters
    if scenario == "high_iv":
        base_iv = 0.45
        skew_factor = 0.8
    elif scenario == "low_iv":
        base_iv = 0.15
        skew_factor = 0.3
    elif scenario == "earnings":
        base_iv = 0.35
        skew_factor = 0.6
    else:  # normal
        base_iv = 0.25
        skew_factor = 0.5
    
    # Create strikes
    strikes = np.arange(underlying_price * 0.85, underlying_price * 1.15, underlying_price * 0.025)
    
    # Create expirations
    expirations = [
        datetime.now() + timedelta(days=7),   # Weekly
        datetime.now() + timedelta(days=21),  # Monthly
        datetime.now() + timedelta(days=45),  # Next month
    ]
    
    for exp in expirations:
        tte = (exp - datetime.now()).days / 365.0
        
        for strike in strikes:
            moneyness = strike / underlying_price
            
            # Create volatility smile/skew
            if moneyness < 0.95:  # OTM puts
                iv = base_iv + (0.95 - moneyness) * skew_factor
            elif moneyness > 1.05:  # OTM calls
                iv = base_iv + (moneyness - 1.05) * skew_factor * 0.6
            else:  # ATM
                iv = base_iv
            
            # Add term structure
            iv = iv + (tte * 0.03)
            iv = max(0.1, min(iv, 0.8))
            
            for option_type in ['call', 'put']:
                # Calculate Greeks
                if option_type == 'call':
                    delta = 0.5 if abs(moneyness - 1.0) < 0.05 else (0.8 if moneyness < 1.0 else 0.2)
                else:
                    delta = -0.5 if abs(moneyness - 1.0) < 0.05 else (-0.2 if moneyness < 1.0 else -0.8)
                
                gamma = 0.03 / (abs(moneyness - 1.0) + 0.1)
                theta = -0.05 * iv * np.sqrt(tte)
                vega = 0.15 * np.sqrt(tte)
                rho = 0.08 * tte * (1 if option_type == 'call' else -1)
                
                # Market data
                theoretical_price = max(0.05, iv * underlying_price * 0.1)
                volume = np.random.randint(50, 2000)
                open_interest = np.random.randint(500, 10000)
                
                option = OptionsData(
                    symbol=symbol,
                    expiration=exp,
                    strike=strike,
                    option_type=option_type,
                    bid=theoretical_price * 0.98,
                    ask=theoretical_price * 1.02,
                    last_price=theoretical_price,
                    volume=volume,
                    open_interest=open_interest,
                    implied_volatility=iv,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                    rho=rho,
                    underlying_price=underlying_price,
                    time_to_expiration=tte
                )
                
                options_data.append(option)
    
    return options_data

async def demo_iv_surface_analysis():
    """Demo IV surface analysis"""
    print_header("IV SURFACE ANALYSIS DEMO")
    
    agent = OptionsVolatilityAgent()
    
    # Test different scenarios
    scenarios = [
        ("AAPL", 150.0, "normal", "Normal Market Conditions"),
        ("TSLA", 200.0, "high_iv", "High Volatility Environment"),
        ("MSFT", 300.0, "low_iv", "Low Volatility Environment"),
    ]
    
    for symbol, price, scenario, description in scenarios:
        print_subheader(f"{symbol} - {description}")
        
        options_data = create_realistic_options_chain(symbol, price, scenario)
        iv_analysis = await agent.analyze_iv_surface(symbol, options_data)
        
        if 'error' not in iv_analysis:
            metrics = iv_analysis['surface_metrics']
            skew_analysis = iv_analysis['skew_analysis']
            arbitrage_opps = iv_analysis['arbitrage_opportunities']
            
            print(f"[UP] Surface Points: {iv_analysis['surface_points']}")
            print(f"[CHART] Average IV: {metrics['average_iv']:.1%}")
            print(f"[INFO] IV Range: {metrics['min_iv']:.1%} - {metrics['max_iv']:.1%}")
            print(f"[SEARCH] Skew Anomalies: {len([s for s in skew_analysis if s.is_anomalous])}")
            print(f"[FAST] Arbitrage Opportunities: {len(arbitrage_opps)}")
            
            # Show skew details
            if skew_analysis:
                print("\n[INFO] Volatility Skew Analysis:")
                for skew in skew_analysis[:2]:  # Show first 2
                    print(f"  Expiration: {skew.expiration.strftime('%Y-%m-%d')}")
                    print(f"  Skew Slope: {skew.skew_slope:.3f}")
                    print(f"  Put-Call Skew: {skew.put_call_skew:.3f}")
                    print(f"  Anomalous: {'Yes' if skew.is_anomalous else 'No'}")
            
            # Show arbitrage opportunities
            if arbitrage_opps:
                print("\n[MONEY] Arbitrage Opportunities:")
                for opp in arbitrage_opps[:2]:  # Show first 2
                    print(f"  Type: {opp['type']}")
                    print(f"  Strike: ${opp['strike']:.2f}")
                    print(f"  IV Difference: {opp['iv_difference']:.1%}")
                    print(f"  Confidence: {opp['confidence']:.1%}")
        else:
            print(f"[X] Error: {iv_analysis['error']}")

async def demo_earnings_integration():
    """Demo earnings calendar integration"""
    print_header("EARNINGS CALENDAR INTEGRATION DEMO")
    
    agent = OptionsVolatilityAgent()
    
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    for symbol in symbols:
        print_subheader(f"{symbol} Earnings Analysis")
        
        price = np.random.uniform(150, 300)
        options_data = create_realistic_options_chain(symbol, price, "earnings")
        
        earnings_event = await agent.integrate_earnings_calendar(symbol, options_data)
        
        if earnings_event:
            print(f"[CAL] Earnings Date: {earnings_event.earnings_date.strftime('%Y-%m-%d')}")
            print(f"[CLOCK] Days to Earnings: {earnings_event.days_to_earnings}")
            print(f"[CHART] Expected Move: {earnings_event.expected_move:.1%}")
            print(f"[UP] IV Rank: {earnings_event.iv_rank:.1%}")
            print(f"[CHART] IV Percentile: {earnings_event.iv_percentile:.0f}th")
            print(f"[TARGET] Recommended Strategy: {earnings_event.strategy_recommendation.value}")
            print(f"[INFO] Historical Moves: {[f'{m:.1%}' for m in earnings_event.historical_earnings_moves]}")
        else:
            print("[CAL] No upcoming earnings events detected")

async def demo_greeks_calculation():
    """Demo Greeks calculation and risk management"""
    print_header("GREEKS CALCULATION & RISK MANAGEMENT DEMO")
    
    agent = OptionsVolatilityAgent()
    
    # Create a sample portfolio
    print_subheader("Sample Options Portfolio")
    
    portfolio_positions = []
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        price = np.random.uniform(150, 300)
        options_data = create_realistic_options_chain(symbol, price, "normal")
        
        # Take some positions from the chain
        portfolio_positions.extend(options_data[:5])  # 5 positions per symbol
    
    print(f"[CHART] Portfolio Size: {len(portfolio_positions)} positions")
    print(f"[INFO] Symbols: {', '.join(symbols)}")
    
    # Calculate portfolio Greeks
    greeks_risk = await agent.calculate_greeks_risk(portfolio_positions)
    
    print_subheader("Portfolio Greeks Analysis")
    print(f"[INFO] Total Delta: {greeks_risk.total_delta:.2f}")
    print(f"[INFO] Total Gamma: {greeks_risk.total_gamma:.4f}")
    print(f"[CLOCK] Total Theta: ${greeks_risk.total_theta:.2f}/day")
    print(f"[CHART] Total Vega: ${greeks_risk.total_vega:.2f}")
    print(f"[MONEY] Total Rho: ${greeks_risk.total_rho:.2f}")
    
    print_subheader("Risk Assessment")
    print(f"[BALANCE] Delta Neutral: {'Yes' if greeks_risk.delta_neutral else 'No'}")
    print(f"[WARN] Gamma Risk Level: {greeks_risk.gamma_risk_level.upper()}")
    print(f"[UP] Vega Exposure: ${greeks_risk.vega_exposure:.2f}")
    print(f"⏳ Daily Theta Decay: ${greeks_risk.theta_decay_daily:.2f}")
    
    # Risk recommendations
    print_subheader("Risk Management Recommendations")
    if not greeks_risk.delta_neutral:
        print("[INFO] Consider delta hedging to neutralize directional risk")
    if greeks_risk.gamma_risk_level == "high":
        print("[WARN] High gamma exposure - monitor for large price moves")
    if greeks_risk.vega_exposure > 500:
        print("[CHART] High vega exposure - vulnerable to volatility changes")
    if greeks_risk.theta_decay_daily < -100:
        print("[CLOCK] Significant time decay - consider rolling positions")

async def demo_volatility_regime_detection():
    """Demo volatility regime detection"""
    print_header("VOLATILITY REGIME DETECTION DEMO")
    
    agent = OptionsVolatilityAgent()
    
    symbols = ["AAPL", "TSLA", "SPY", "QQQ", "IWM"]
    
    print_subheader("Current Market Volatility Regimes")
    
    regime_counts = {regime: 0 for regime in VolatilityRegime}
    
    for symbol in symbols:
        regime = await agent.detect_volatility_regime(symbol)
        regime_counts[regime] += 1
        
        # Get regime description
        regime_descriptions = {
            VolatilityRegime.LOW_VOL: "[DOWN] Low volatility - consider buying premium",
            VolatilityRegime.NORMAL_VOL: "[CHART] Normal volatility - balanced strategies",
            VolatilityRegime.HIGH_VOL: "[UP] High volatility - consider selling premium",
            VolatilityRegime.EXTREME_VOL: "[ALERT] Extreme volatility - defensive strategies"
        }
        
        print(f"{symbol:>6}: {regime.value.replace('_', ' ').title()}")
        print(f"        {regime_descriptions[regime]}")
    
    print_subheader("Market Overview")
    for regime, count in regime_counts.items():
        if count > 0:
            percentage = (count / len(symbols)) * 100
            print(f"{regime.value.replace('_', ' ').title():>20}: {count} symbols ({percentage:.0f}%)")

async def demo_signal_generation():
    """Demo comprehensive signal generation"""
    print_header("OPTIONS SIGNAL GENERATION DEMO")
    
    agent = OptionsVolatilityAgent()
    
    # Test different market scenarios
    scenarios = [
        ("AAPL", 150.0, "normal", "Normal Market"),
        ("TSLA", 200.0, "high_iv", "High Volatility"),
        ("MSFT", 300.0, "earnings", "Pre-Earnings"),
    ]
    
    for symbol, price, scenario, description in scenarios:
        print_subheader(f"{symbol} - {description}")
        
        market_data = {
            'current_price': price,
            'volume': np.random.randint(1000000, 5000000)
        }
        
        signals = await agent.generate_options_signals(symbol, market_data)
        
        if signals:
            print(f"[TARGET] Generated {len(signals)} trading signals")
            
            for i, signal in enumerate(signals, 1):
                print(f"\n[CHART] Signal #{i}: {signal.signal_type.upper()}")
                print(f"   Strategy: {signal.strategy.value.replace('_', ' ').title()}")
                print(f"   Strength: {signal.value:.2f} ({signal.value * 100:+.0f}%)")
                print(f"   Confidence: {signal.confidence:.1%}")
                print(f"   Volatility Regime: {signal.volatility_regime.value.replace('_', ' ').title()}")
                
                if signal.expiration:
                    print(f"   Expiration: {signal.expiration.strftime('%Y-%m-%d')}")
                if signal.strike:
                    print(f"   Strike: ${signal.strike:.2f}")
                
                print(f"   Expected Return: {signal.expected_return:.1%}")
                print(f"   Max Risk: {signal.max_risk:.1%}")
                
                print("   [NOTE] Top Reasons:")
                for reason in signal.top_3_reasons:
                    print(f"      {reason['rank']}. {reason['factor']}: {reason['explanation']}")
                    print(f"         Confidence: {reason['confidence']:.1%}")
        else:
            print("[INFO] No signals generated for current market conditions")

async def demo_langgraph_integration():
    """Demo LangGraph integration"""
    print_header("LANGGRAPH INTEGRATION DEMO")
    
    print_subheader("LangGraph State Processing")
    
    # Create mock LangGraph state
    state = {
        'market_data': {
            'AAPL': {'current_price': 150.0, 'volume': 2000000},
            'MSFT': {'current_price': 300.0, 'volume': 1500000},
            'GOOGL': {'current_price': 2500.0, 'volume': 800000},
        },
        'signals': {},
        'portfolio_state': {'cash': 100000, 'positions': []},
        'risk_metrics': {'var': 5000, 'max_drawdown': 0.05}
    }
    
    print("[INFO] Input State:")
    print(f"   Market Data: {len(state['market_data'])} symbols")
    print(f"   Existing Signals: {len(state['signals'])} agent types")
    
    # Process through LangGraph node
    result_state = await options_volatility_agent_node(state)
    
    print("\n[INFO] Output State:")
    print(f"   Options Signals: {len(result_state['signals']['options_volatility'])} signals")
    print(f"   Analysis Timestamp: {result_state['options_analysis']['timestamp']}")
    
    # Show signal summary
    if result_state['signals']['options_volatility']:
        print("\n[CHART] Generated Signals Summary:")
        signal_types = {}
        strategies = {}
        
        for signal_dict in result_state['signals']['options_volatility']:
            signal_type = signal_dict['signal_type']
            strategy = signal_dict['strategy']
            
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        print("   Signal Types:")
        for sig_type, count in signal_types.items():
            print(f"     {sig_type}: {count}")
        
        print("   Strategies:")
        for strategy, count in strategies.items():
            print(f"     {strategy.replace('_', ' ').title()}: {count}")

async def main():
    """Main demo function"""
    print("[LAUNCH] OPTIONS VOLATILITY AGENT COMPREHENSIVE DEMO")
    print("This demo showcases all major capabilities of the Options Volatility Agent")
    
    try:
        # Run all demos
        await demo_iv_surface_analysis()
        await demo_earnings_integration()
        await demo_greeks_calculation()
        await demo_volatility_regime_detection()
        await demo_signal_generation()
        await demo_langgraph_integration()
        
        print_header("DEMO COMPLETED SUCCESSFULLY")
        print("[OK] All Options Volatility Agent features demonstrated")
        print("[TARGET] The agent is ready for integration into the trading system")
        
    except Exception as e:
        print(f"\n[X] Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())