"""
Portfolio Allocator Agent Demo

This demo showcases the signal fusion, conflict resolution, explainability engine,
and regime-based strategy weighting capabilities of the Portfolio Allocator Agent.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.portfolio_allocator_agent import (
    PortfolioAllocatorAgent,
    Signal,
    SignalType,
    MarketRegime
)


def create_sample_signals() -> Dict[str, List[Signal]]:
    """Create comprehensive sample signals for demonstration"""
    
    # AAPL signals - Multiple strategies with some conflicts
    aapl_signals = [
        # Strong momentum signal
        Signal(
            symbol="AAPL",
            signal_type=SignalType.MOMENTUM,
            value=0.8,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            agent_name="momentum_agent",
            model_version="1.0.0",
            metadata={
                'technical_indicators': {
                    'indicators': {
                        'ema_crossover': {'signal': 0.9},
                        'rsi_breakout': {'signal': 0.8},
                        'macd': {'signal': 0.7},
                        'stochastic': {'signal': 0.6}
                    }
                },
                'sentiment_score': 0.7,
                'fibonacci_levels': {
                    'levels': {
                        'fib_236': 145.0,
                        'fib_382': 148.0,
                        'fib_500': 150.0,
                        'fib_618': 152.0
                    }
                },
                'volume_data': {
                    'current_volume': 2500000,
                    'average_volume': 1500000
                },
                'risk_reward': {
                    'ratio': 3.2,
                    'stop_loss': 145.0,
                    'target': 165.0
                }
            }
        ),
        
        # Conflicting mean reversion signal
        Signal(
            symbol="AAPL",
            signal_type=SignalType.MEAN_REVERSION,
            value=-0.4,
            confidence=0.6,
            timestamp=datetime.now(timezone.utc),
            agent_name="mean_reversion_agent",
            model_version="1.0.0",
            metadata={
                'bollinger_bands': {'signal': -0.5, 'position': 'upper_band'},
                'z_score': -1.8,
                'rsi_overbought': True,
                'volume_data': {
                    'current_volume': 1200000,
                    'average_volume': 1500000
                }
            }
        ),
        
        # Supporting sentiment signal
        Signal(
            symbol="AAPL",
            signal_type=SignalType.SENTIMENT,
            value=0.6,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            agent_name="sentiment_agent",
            model_version="1.0.0",
            metadata={
                'sentiment_score': 0.6,
                'news_count': 25,
                'social_sentiment': 0.7,
                'analyst_upgrades': 3,
                'earnings_sentiment': 0.8
            }
        ),
        
        # Options volatility signal
        Signal(
            symbol="AAPL",
            signal_type=SignalType.OPTIONS_VOLATILITY,
            value=0.3,
            confidence=0.7,
            timestamp=datetime.now(timezone.utc),
            agent_name="options_volatility_agent",
            model_version="1.0.0",
            metadata={
                'iv_rank': 0.3,
                'iv_percentile': 0.25,
                'skew': 0.15,
                'earnings_in_days': 12,
                'put_call_ratio': 0.8
            }
        )
    ]
    
    # MSFT signals - Clean mean reversion setup
    msft_signals = [
        Signal(
            symbol="MSFT",
            signal_type=SignalType.MEAN_REVERSION,
            value=-0.7,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            agent_name="mean_reversion_agent",
            model_version="1.0.0",
            metadata={
                'bollinger_bands': {'signal': -0.8, 'position': 'upper_band'},
                'z_score': -2.1,
                'pairs_trading': {'correlation': 0.85, 'spread_zscore': -2.3},
                'fibonacci_levels': {
                    'levels': {
                        'fib_618': 285.0,
                        'fib_786': 290.0
                    }
                },
                'volume_data': {
                    'current_volume': 800000,
                    'average_volume': 1200000
                }
            }
        ),
        
        Signal(
            symbol="MSFT",
            signal_type=SignalType.LONG_TERM_CORE,
            value=0.5,
            confidence=0.6,
            timestamp=datetime.now(timezone.utc),
            agent_name="long_term_core_agent",
            model_version="1.0.0",
            metadata={
                'fundamental_score': 0.7,
                'pe_ratio': 28.5,
                'growth_rate': 0.15,
                'dividend_yield': 0.011,
                'debt_to_equity': 0.35
            }
        )
    ]
    
    # TSLA signals - High volatility with mixed signals
    tsla_signals = [
        Signal(
            symbol="TSLA",
            signal_type=SignalType.MOMENTUM,
            value=0.9,
            confidence=0.7,
            timestamp=datetime.now(timezone.utc),
            agent_name="momentum_agent",
            model_version="1.0.0",
            metadata={
                'technical_indicators': {
                    'indicators': {
                        'ema_crossover': {'signal': 0.95},
                        'rsi_breakout': {'signal': 0.85},
                        'macd': {'signal': 0.9}
                    }
                },
                'volume_data': {
                    'current_volume': 15000000,
                    'average_volume': 8000000
                }
            }
        ),
        
        Signal(
            symbol="TSLA",
            signal_type=SignalType.SHORT_SELLING,
            value=-0.6,
            confidence=0.5,
            timestamp=datetime.now(timezone.utc),
            agent_name="short_selling_agent",
            model_version="1.0.0",
            metadata={
                'valuation_concern': True,
                'short_interest': 0.12,
                'borrow_cost': 0.08,
                'fundamental_weakness': 0.6
            }
        ),
        
        Signal(
            symbol="TSLA",
            signal_type=SignalType.OPTIONS_VOLATILITY,
            value=0.8,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            agent_name="options_volatility_agent",
            model_version="1.0.0",
            metadata={
                'iv_rank': 0.85,
                'iv_percentile': 0.92,
                'skew': 0.25,
                'gamma_exposure': 'high',
                'vix_correlation': 0.7
            }
        )
    ]
    
    return {
        "AAPL": aapl_signals,
        "MSFT": msft_signals,
        "TSLA": tsla_signals
    }


def create_market_scenarios() -> List[Dict[str, any]]:
    """Create different market scenarios for testing"""
    
    scenarios = [
        {
            'name': 'Bull Market Trending',
            'data': {
                'volatility': 0.15,
                'trend_strength': 0.8,
                'volume_ratio': 1.4,
                'market_sentiment': 0.7,
                'vix_level': 18.0
            }
        },
        {
            'name': 'Bear Market Trending',
            'data': {
                'volatility': 0.28,
                'trend_strength': -0.7,
                'volume_ratio': 1.6,
                'market_sentiment': -0.6,
                'vix_level': 32.0
            }
        },
        {
            'name': 'High Volatility Crisis',
            'data': {
                'volatility': 0.45,
                'trend_strength': -0.3,
                'volume_ratio': 2.1,
                'market_sentiment': -0.8,
                'vix_level': 55.0
            }
        },
        {
            'name': 'Low Volatility Grind',
            'data': {
                'volatility': 0.08,
                'trend_strength': 0.2,
                'volume_ratio': 0.8,
                'market_sentiment': 0.3,
                'vix_level': 12.0
            }
        },
        {
            'name': 'Mean Reverting Market',
            'data': {
                'volatility': 0.18,
                'trend_strength': 0.1,
                'volume_ratio': 1.0,
                'market_sentiment': 0.0,
                'vix_level': 22.0
            }
        }
    ]
    
    return scenarios


def print_signal_summary(symbol: str, signals: List[Signal]):
    """Print a summary of input signals"""
    print(f"\n📊 Input Signals for {symbol}:")
    print("=" * 50)
    
    for signal in signals:
        direction = "🟢 BUY" if signal.value > 0 else "🔴 SELL"
        strength = "Strong" if abs(signal.value) > 0.6 else "Moderate" if abs(signal.value) > 0.3 else "Weak"
        
        print(f"  {direction} | {signal.signal_type.value.upper():<20} | "
              f"Value: {signal.value:+.2f} | Confidence: {signal.confidence:.2f} | "
              f"Strength: {strength}")


def print_fused_signal_details(symbol: str, fused_signal):
    """Print detailed analysis of fused signal"""
    print(f"\n🎯 Fused Signal for {symbol}:")
    print("=" * 60)
    
    direction = "🟢 BUY" if fused_signal.value > 0 else "🔴 SELL"
    strength = "Strong" if abs(fused_signal.value) > 0.6 else "Moderate" if abs(fused_signal.value) > 0.3 else "Weak"
    
    print(f"  Final Decision: {direction}")
    print(f"  Signal Strength: {strength} ({fused_signal.value:+.3f})")
    print(f"  Confidence: {fused_signal.confidence:.1%}")
    print(f"  Contributing Agents: {', '.join(fused_signal.contributing_agents)}")
    
    if fused_signal.conflict_resolution:
        print(f"  ⚠️  Conflict Resolved: {fused_signal.conflict_resolution}")
    
    print(f"\n  🧠 Top 3 Reasons:")
    for reason in fused_signal.top_3_reasons:
        print(f"    {reason.rank}. {reason.factor.replace('_', ' ').title()}")
        print(f"       {reason.explanation}")
        print(f"       Contribution: {reason.contribution:.1%} | Confidence: {reason.confidence:.1%}")
        print()


async def run_scenario_analysis():
    """Run comprehensive scenario analysis"""
    print("🚀 Portfolio Allocator Agent - Comprehensive Demo")
    print("=" * 80)
    
    # Initialize agent
    agent = PortfolioAllocatorAgent()
    
    # Get sample signals
    raw_signals = create_sample_signals()
    
    # Print input signals summary
    print("\n📈 INPUT SIGNALS SUMMARY")
    print("=" * 80)
    for symbol, signals in raw_signals.items():
        print_signal_summary(symbol, signals)
    
    # Test different market scenarios
    scenarios = create_market_scenarios()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n\n🌍 SCENARIO {i}: {scenario['name'].upper()}")
        print("=" * 80)
        
        # Print market conditions
        market_data = scenario['data']
        print(f"Market Conditions:")
        print(f"  Volatility: {market_data['volatility']:.1%}")
        print(f"  Trend Strength: {market_data['trend_strength']:+.2f}")
        print(f"  Volume Ratio: {market_data['volume_ratio']:.1f}x")
        print(f"  Market Sentiment: {market_data.get('market_sentiment', 0):+.2f}")
        print(f"  VIX Level: {market_data.get('vix_level', 20):.1f}")
        
        # Process signals
        try:
            fused_signals = await agent.process_signals(raw_signals, market_data)
            
            # Print results for each symbol
            for symbol in sorted(fused_signals.keys()):
                print_fused_signal_details(symbol, fused_signals[symbol])
            
            # Print scenario summary
            print(f"\n📋 Scenario Summary:")
            total_signals = len(fused_signals)
            buy_signals = sum(1 for s in fused_signals.values() if s.value > 0)
            sell_signals = total_signals - buy_signals
            avg_confidence = sum(s.confidence for s in fused_signals.values()) / max(total_signals, 1)
            
            print(f"  Total Signals: {total_signals}")
            print(f"  Buy Signals: {buy_signals} | Sell Signals: {sell_signals}")
            print(f"  Average Confidence: {avg_confidence:.1%}")
            
            conflicts_resolved = sum(1 for s in fused_signals.values() if s.conflict_resolution)
            if conflicts_resolved > 0:
                print(f"  Conflicts Resolved: {conflicts_resolved}")
            
        except Exception as e:
            print(f"❌ Error processing scenario: {str(e)}")
        
        print("\n" + "-" * 80)


async def run_explainability_showcase():
    """Showcase the explainability engine in detail"""
    print("\n\n🧠 EXPLAINABILITY ENGINE SHOWCASE")
    print("=" * 80)
    
    agent = PortfolioAllocatorAgent()
    
    # Create a signal with rich metadata for explainability
    rich_signal = {
        "NVDA": [
            Signal(
                symbol="NVDA",
                signal_type=SignalType.MOMENTUM,
                value=0.85,
                confidence=0.92,
                timestamp=datetime.now(timezone.utc),
                agent_name="momentum_agent",
                model_version="1.0.0",
                metadata={
                    'technical_indicators': {
                        'indicators': {
                            'ema_crossover': {'signal': 0.95},
                            'rsi_breakout': {'signal': 0.88},
                            'macd': {'signal': 0.82},
                            'stochastic': {'signal': 0.79},
                            'williams_r': {'signal': 0.73}
                        }
                    },
                    'sentiment_score': 0.78,
                    'fibonacci_levels': {
                        'levels': {
                            'fib_382': 420.0,
                            'fib_500': 435.0,
                            'fib_618': 450.0
                        }
                    },
                    'volume_data': {
                        'current_volume': 45000000,
                        'average_volume': 25000000
                    },
                    'risk_reward': {
                        'ratio': 4.2,
                        'stop_loss': 410.0,
                        'target': 485.0
                    }
                }
            )
        ]
    }
    
    market_data = {
        'volatility': 0.22,
        'trend_strength': 0.7,
        'volume_ratio': 1.8,
        'ai_sector_momentum': 0.85
    }
    
    print("🎯 Processing NVDA signal with rich metadata...")
    fused_signals = await agent.process_signals(rich_signal, market_data)
    
    if "NVDA" in fused_signals:
        nvda_signal = fused_signals["NVDA"]
        
        print(f"\n📊 Detailed Explainability Analysis:")
        print(f"Final Signal: {nvda_signal.value:+.3f} (Confidence: {nvda_signal.confidence:.1%})")
        
        print(f"\n🔍 Detailed Reasoning Breakdown:")
        for i, reason in enumerate(nvda_signal.top_3_reasons, 1):
            print(f"\n  Reason #{i}: {reason.factor.replace('_', ' ').title()}")
            print(f"  Contribution: {reason.contribution:.1%}")
            print(f"  Confidence: {reason.confidence:.1%}")
            print(f"  Explanation: {reason.explanation}")
            
            if reason.supporting_data:
                print(f"  Supporting Data:")
                for key, value in reason.supporting_data.items():
                    if isinstance(value, dict):
                        print(f"    {key}: {value}")
                    else:
                        print(f"    {key}: {value}")


async def run_conflict_resolution_demo():
    """Demonstrate conflict resolution capabilities"""
    print("\n\n⚔️ CONFLICT RESOLUTION DEMO")
    print("=" * 80)
    
    agent = PortfolioAllocatorAgent()
    
    # Create highly conflicting signals
    conflicting_signals = {
        "AMZN": [
            # Strong bullish momentum
            Signal(
                symbol="AMZN",
                signal_type=SignalType.MOMENTUM,
                value=0.9,
                confidence=0.95,
                timestamp=datetime.now(timezone.utc),
                agent_name="momentum_agent",
                model_version="1.0.0",
                metadata={'breakout_strength': 0.9}
            ),
            
            # Strong bearish mean reversion
            Signal(
                symbol="AMZN",
                signal_type=SignalType.MEAN_REVERSION,
                value=-0.8,
                confidence=0.85,
                timestamp=datetime.now(timezone.utc),
                agent_name="mean_reversion_agent",
                model_version="1.0.0",
                metadata={'overbought_level': 0.9}
            ),
            
            # Bearish sentiment
            Signal(
                symbol="AMZN",
                signal_type=SignalType.SENTIMENT,
                value=-0.6,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                agent_name="sentiment_agent",
                model_version="1.0.0",
                metadata={'negative_news_count': 12}
            ),
            
            # Bullish short selling (covering)
            Signal(
                symbol="AMZN",
                signal_type=SignalType.SHORT_SELLING,
                value=0.4,
                confidence=0.6,
                timestamp=datetime.now(timezone.utc),
                agent_name="short_selling_agent",
                model_version="1.0.0",
                metadata={'short_squeeze_risk': 0.7}
            )
        ]
    }
    
    market_data = {
        'volatility': 0.25,
        'trend_strength': 0.3,
        'volume_ratio': 1.5
    }
    
    print("⚡ Processing highly conflicting signals for AMZN...")
    print("\nInput Signals:")
    for signal in conflicting_signals["AMZN"]:
        direction = "🟢" if signal.value > 0 else "🔴"
        print(f"  {direction} {signal.signal_type.value}: {signal.value:+.2f} "
              f"(confidence: {signal.confidence:.1%})")
    
    fused_signals = await agent.process_signals(conflicting_signals, market_data)
    
    if "AMZN" in fused_signals:
        amzn_signal = fused_signals["AMZN"]
        
        print(f"\n🎯 Conflict Resolution Result:")
        direction = "🟢 BUY" if amzn_signal.value > 0 else "🔴 SELL"
        print(f"  Final Decision: {direction}")
        print(f"  Resolved Value: {amzn_signal.value:+.3f}")
        print(f"  Final Confidence: {amzn_signal.confidence:.1%}")
        
        if amzn_signal.conflict_resolution:
            print(f"  Resolution Method: {amzn_signal.conflict_resolution}")
        
        print(f"\n  Resolution Reasoning:")
        for reason in amzn_signal.top_3_reasons:
            print(f"    • {reason.explanation}")


async def main():
    """Run the complete demo"""
    try:
        # Run scenario analysis
        await run_scenario_analysis()
        
        # Run explainability showcase
        await run_explainability_showcase()
        
        # Run conflict resolution demo
        await run_conflict_resolution_demo()
        
        print("\n\n✅ Demo completed successfully!")
        print("=" * 80)
        print("The Portfolio Allocator Agent demonstrated:")
        print("  ✓ Multi-strategy signal fusion")
        print("  ✓ Regime-based strategy weighting")
        print("  ✓ Intelligent conflict resolution")
        print("  ✓ Comprehensive explainability")
        print("  ✓ Robust error handling")
        print("  ✓ LangGraph workflow orchestration")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())