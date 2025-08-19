"""
Visualization Demo Script

This script demonstrates the visualization capabilities of the trading system,
including performance dashboards, agent-specific charts, and unified monitoring.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import visualization components
from agents.performance_dashboard import PerformanceDashboard
from agents.agent_visualizers import (
    MomentumAgentVisualizer, 
    MeanReversionAgentVisualizer, 
    PortfolioAllocatorVisualizer,
    create_momentum_dashboard,
    create_mean_reversion_dashboard,
    create_portfolio_dashboard
)
from agents.unified_dashboard import UnifiedTradingDashboard

def generate_sample_data():
    """Generate sample data for demonstration"""
    # Sample market data
    market_data = [
        {
            'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
            'open': 150 + np.random.normal(0, 1),
            'high': 152 + np.random.normal(0, 1),
            'low': 148 + np.random.normal(0, 1),
            'close': 150 + np.random.normal(0, 1),
            'volume': 1000000 + np.random.randint(-100000, 100000),
            'signals': []  # Placeholder for signals
        }
        for i in range(30, 0, -1)
    ]
    
    # Add some signals to sample data
    for i in range(0, len(market_data), 5):
        market_data[i]['signals'] = [{
            'signal_type': 'buy' if np.random.random() > 0.5 else 'sell',
            'value': np.random.uniform(-1, 1),
            'confidence': np.random.uniform(0.5, 1.0)
        }]
    
    # Sample signals
    signals = [
        {
            'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
            'signal_type': 'buy' if np.random.random() > 0.5 else 'sell',
            'value': np.random.uniform(-1, 1),
            'confidence': np.random.uniform(0.5, 1.0)
        }
        for i in range(30, 0, -1)
    ][::5]  # Every 5th day
    
    # Sample indicator data
    indicator_data = {
        'ema_fast': [150 + np.random.normal(0, 0.5) for _ in range(30)],
        'ema_slow': [150 + np.random.normal(0, 0.3) for _ in range(30)],
        'rsi': [50 + np.random.normal(0, 15) for _ in range(30)],
        'macd_line': [0 + np.random.normal(0, 0.1) for _ in range(30)],
        'signal_line': [0 + np.random.normal(0, 0.05) for _ in range(30)],
        'histogram': [0 + np.random.normal(0, 0.1) for _ in range(30)],
        'upper_band': [155 + np.random.normal(0, 1) for _ in range(30)],
        'middle_band': [150 + np.random.normal(0, 0.5) for _ in range(30)],
        'lower_band': [145 + np.random.normal(0, 1) for _ in range(30)],
        'z_score': [0 + np.random.normal(0, 1) for _ in range(30)],
        'spread': [0 + np.random.normal(0, 0.5) for _ in range(30)],
        'spread_std': [1.0 for _ in range(30)]
    }
    
    # Sample portfolio data
    portfolio_data = {
        'positions': {
            'AAPL': {'quantity': 100, 'market_value': 15000, 'unrealized_pnl': 500},
            'GOOGL': {'quantity': 50, 'market_value': 75000, 'unrealized_pnl': -200},
            'TSLA': {'quantity': 200, 'market_value': 40000, 'unrealized_pnl': 1200}
        },
        'total_value': 130000,
        'cash': 20000,
        'unrealized_pnl': 1500
    }
    
    # Sample allocation history
    allocation_history = [
        {
            'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
            'total_value': 130000 - i * 100,
            'cash': 20000 - i * 50,
            'positions_value': 110000 - i * 50
        }
        for i in range(30, 0, -1)
    ]
    
    return {
        'market_data': market_data,
        'signals': signals,
        'indicator_data': indicator_data,
        'portfolio_data': portfolio_data,
        'allocation_history': allocation_history
    }

def demo_individual_visualizers(sample_data):
    """Demonstrate individual agent visualizers"""
    print("=== Individual Agent Visualizer Demo ===")
    print("Creating sample charts for trading agents...")
    
    # Create momentum agent chart
    print("\n1. Creating Momentum Agent Dashboard...")
    try:
        fig1 = create_momentum_dashboard(
            sample_data['market_data'], 
            sample_data['signals'], 
            sample_data['indicator_data']
        )
        print("   Momentum dashboard created successfully")
    except Exception as e:
        print(f"   Error creating momentum dashboard: {e}")
    
    # Create mean reversion agent chart
    print("\n2. Creating Mean Reversion Agent Dashboard...")
    try:
        fig2 = create_mean_reversion_dashboard(
            sample_data['market_data'], 
            sample_data['signals'], 
            sample_data['indicator_data']
        )
        print("   Mean reversion dashboard created successfully")
    except Exception as e:
        print(f"   Error creating mean reversion dashboard: {e}")
    
    # Create portfolio allocator chart
    print("\n3. Creating Portfolio Allocator Dashboard...")
    try:
        fig3 = create_portfolio_dashboard(
            sample_data['portfolio_data'], 
            sample_data['allocation_history']
        )
        print("   Portfolio dashboard created successfully")
    except Exception as e:
        print(f"   Error creating portfolio dashboard: {e}")

def demo_unified_dashboard():
    """Demonstrate unified dashboard"""
    print("\n=== Unified Dashboard Demo ===")
    print("Creating unified trading system dashboard...")
    
    try:
        # Create unified dashboard
        dashboard = UnifiedTradingDashboard()
        print("   Unified dashboard initialized successfully")
        print("   Dashboard would launch in GUI mode when run")
        return dashboard
    except Exception as e:
        print(f"   Error creating unified dashboard: {e}")
        return None

def demo_performance_dashboard():
    """Demonstrate performance monitoring dashboard"""
    print("\n=== Performance Monitoring Dashboard Demo ===")
    print("Creating performance monitoring dashboard...")
    
    try:
        # Create performance dashboard
        perf_dashboard = PerformanceDashboard()
        print("   Performance dashboard initialized successfully")
        print("   Performance dashboard would launch in GUI mode when run")
        return perf_dashboard
    except Exception as e:
        print(f"   Error creating performance dashboard: {e}")
        return None

async def main():
    """Main demo function"""
    print("TRADING SYSTEM VISUALIZATION DEMO")
    print("=" * 50)
    
    # Generate sample data
    print("\nGenerating sample data for visualization...")
    sample_data = generate_sample_data()
    print("   Sample data generated successfully")
    
    # Demonstrate individual visualizers
    demo_individual_visualizers(sample_data)
    
    # Demonstrate unified dashboard
    unified_dashboard = demo_unified_dashboard()
    
    # Demonstrate performance dashboard
    perf_dashboard = demo_performance_dashboard()
    
    # Summary
    print("\n" + "=" * 50)
    print("DEMO SUMMARY")
    print("=" * 50)
    print("Individual agent visualizers created")
    print("Unified trading system dashboard initialized")
    print("Performance monitoring dashboard initialized")
    print("\nTo run the interactive dashboards:")
    print("   - Run 'python agents/unified_dashboard.py' for the main dashboard")
    print("   - Run 'python agents/performance_dashboard.py' for performance monitoring")
    print("\nVisualization components are ready for integration with live data!")

if __name__ == "__main__":
    asyncio.run(main())