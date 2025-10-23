"""
Agent Visualization Panels

This module provides visualization capabilities for individual trading agents,
including technical indicator charts, signal strength monitoring, and performance tracking.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MomentumAgentVisualizer:
    """Visualization panel for Momentum Trading Agent"""
    
    def __init__(self):
        self.fig = None
        self.axes = {}
        
    def create_technical_chart(self, market_data: List[Dict], signals: List[Dict], 
                             indicator_data: Dict[str, List[float]]) -> plt.Figure:
        """
        Create a comprehensive technical analysis chart for momentum signals
        
        Args:
            market_data: List of market data points with OHLCV data
            signals: List of generated signals
            indicator_data: Dictionary of indicator values (EMA, RSI, MACD, etc.)
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        # Price chart with indicators
        ax1 = fig.add_subplot(gs[0])
        self._plot_price_chart(ax1, market_data, indicator_data)
        
        # RSI chart
        ax2 = fig.add_subplot(gs[1])
        self._plot_rsi_chart(ax2, indicator_data)
        
        # MACD chart
        ax3 = fig.add_subplot(gs[2])
        self._plot_macd_chart(ax3, indicator_data)
        
        # Signal strength chart
        ax4 = fig.add_subplot(gs[3])
        self._plot_signal_strength(ax4, signals)
        
        # Add title
        fig.suptitle('Momentum Trading Agent - Technical Analysis', fontsize=16, fontweight='bold')
        
        return fig
        
    def _plot_price_chart(self, ax, market_data: List[Dict], indicator_data: Dict[str, List[float]]):
        """Plot price chart with technical indicators"""
        # Extract data
        timestamps = [datetime.fromisoformat(data['timestamp']) for data in market_data]
        closes = [data['close'] for data in market_data]
        highs = [data['high'] for data in market_data]
        lows = [data['low'] for data in market_data]
        
        # Plot candlestick-like chart
        ax.plot(timestamps, closes, linewidth=1.5, color='black', label='Close Price')
        ax.fill_between(timestamps, highs, lows, alpha=0.3, color='gray', label='High-Low Range')
        
        # Plot EMAs if available
        if 'ema_fast' in indicator_data and len(indicator_data['ema_fast']) == len(timestamps):
            ax.plot(timestamps, indicator_data['ema_fast'], 
                   linewidth=1, color='blue', label='Fast EMA')
        if 'ema_slow' in indicator_data and len(indicator_data['ema_slow']) == len(timestamps):
            ax.plot(timestamps, indicator_data['ema_slow'], 
                   linewidth=1, color='red', label='Slow EMA')
        
        # Add buy/sell signals
        buy_signals = [i for i, data in enumerate(market_data) 
                      if any(s.get('signal_type') == 'buy' for s in data.get('signals', []))]
        sell_signals = [i for i, data in enumerate(market_data) 
                       if any(s.get('signal_type') == 'sell' for s in data.get('signals', []))]
        
        if buy_signals:
            buy_prices = [closes[i] for i in buy_signals]
            buy_times = [timestamps[i] for i in buy_signals]
            ax.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy Signal')
            
        if sell_signals:
            sell_prices = [closes[i] for i in sell_signals]
            sell_times = [timestamps[i] for i in sell_signals]
            ax.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell Signal')
        
        ax.set_title('Price Chart with Technical Indicators')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_rsi_chart(self, ax, indicator_data: Dict[str, List[float]]):
        """Plot RSI chart"""
        if 'rsi' not in indicator_data or not indicator_data['rsi']:
            ax.text(0.5, 0.5, 'No RSI data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('RSI')
            return
            
        rsi_values = indicator_data['rsi']
        timestamps = [datetime.now() - timedelta(minutes=i) 
                     for i in range(len(rsi_values)-1, -1, -1)]
        
        ax.plot(timestamps, rsi_values, color='purple', linewidth=1.5)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        
        ax.set_title('Relative Strength Index (RSI)')
        ax.set_ylabel('RSI')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_macd_chart(self, ax, indicator_data: Dict[str, List[float]]):
        """Plot MACD chart"""
        if ('macd_line' not in indicator_data or 'signal_line' not in indicator_data or 
            'histogram' not in indicator_data):
            ax.text(0.5, 0.5, 'No MACD data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('MACD')
            return
            
        macd_line = indicator_data['macd_line']
        signal_line = indicator_data['signal_line']
        histogram = indicator_data['histogram']
        
        timestamps = [datetime.now() - timedelta(minutes=i) 
                     for i in range(len(macd_line)-1, -1, -1)]
        
        ax.plot(timestamps, macd_line, color='blue', linewidth=1.5, label='MACD Line')
        ax.plot(timestamps, signal_line, color='red', linewidth=1.5, label='Signal Line')
        
        # Plot histogram as bar chart
        colors = ['red' if x < 0 else 'green' for x in histogram]
        ax.bar(timestamps, histogram, color=colors, alpha=0.6, width=0.01)
        
        ax.set_title('Moving Average Convergence Divergence (MACD)')
        ax.set_ylabel('MACD')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_signal_strength(self, ax, signals: List[Dict]):
        """Plot signal strength over time"""
        if not signals:
            ax.text(0.5, 0.5, 'No signal data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Signal Strength')
            return
            
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in signals]
        strengths = [s['value'] for s in signals]
        confidences = [s['confidence'] for s in signals]
        
        # Plot signal strength
        ax.plot(timestamps, strengths, color='blue', linewidth=1.5, label='Signal Strength')
        ax.fill_between(timestamps, strengths, alpha=0.3, color='blue')
        
        # Plot confidence as secondary axis
        ax2 = ax.twinx()
        ax2.plot(timestamps, confidences, color='orange', linewidth=1.5, label='Confidence')
        ax2.fill_between(timestamps, confidences, alpha=0.3, color='orange')
        ax2.set_ylabel('Confidence', color='orange')
        
        ax.set_title('Signal Strength and Confidence')
        ax.set_ylabel('Signal Strength', color='blue')
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

class MeanReversionAgentVisualizer:
    """Visualization panel for Mean Reversion Trading Agent"""
    
    def __init__(self):
        self.fig = None
        self.axes = {}
        
    def create_mean_reversion_chart(self, market_data: List[Dict], signals: List[Dict], 
                                  indicator_data: Dict[str, List[float]]) -> plt.Figure:
        """
        Create a comprehensive mean reversion analysis chart
        
        Args:
            market_data: List of market data points with OHLCV data
            signals: List of generated signals
            indicator_data: Dictionary of indicator values (Bollinger Bands, Z-score, etc.)
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        # Price chart with Bollinger Bands
        ax1 = fig.add_subplot(gs[0])
        self._plot_bollinger_chart(ax1, market_data, indicator_data)
        
        # Z-score chart
        ax2 = fig.add_subplot(gs[1])
        self._plot_zscore_chart(ax2, indicator_data)
        
        # Pairs trading chart (if available)
        ax3 = fig.add_subplot(gs[2])
        self._plot_pairs_chart(ax3, indicator_data)
        
        # Signal strength chart
        ax4 = fig.add_subplot(gs[3])
        self._plot_signal_strength(ax4, signals)
        
        # Add title
        fig.suptitle('Mean Reversion Trading Agent - Analysis', fontsize=16, fontweight='bold')
        
        return fig
        
    def _plot_bollinger_chart(self, ax, market_data: List[Dict], indicator_data: Dict[str, List[float]]):
        """Plot price chart with Bollinger Bands"""
        # Extract data
        timestamps = [datetime.fromisoformat(data['timestamp']) for data in market_data]
        closes = [data['close'] for data in market_data]
        
        # Plot price
        ax.plot(timestamps, closes, linewidth=1.5, color='black', label='Close Price')
        
        # Plot Bollinger Bands if available
        if ('upper_band' in indicator_data and 'middle_band' in indicator_data and 
            'lower_band' in indicator_data):
            upper_band = indicator_data['upper_band']
            middle_band = indicator_data['middle_band']
            lower_band = indicator_data['lower_band']
            
            if (len(upper_band) == len(timestamps) and len(middle_band) == len(timestamps) and 
                len(lower_band) == len(timestamps)):
                ax.plot(timestamps, upper_band, color='red', linewidth=1, alpha=0.7, label='Upper Band')
                ax.plot(timestamps, middle_band, color='blue', linewidth=1, alpha=0.7, label='Middle Band')
                ax.plot(timestamps, lower_band, color='red', linewidth=1, alpha=0.7, label='Lower Band')
                
                # Fill between bands
                ax.fill_between(timestamps, upper_band, lower_band, alpha=0.1, color='blue')
        
        ax.set_title('Price Chart with Bollinger Bands')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_zscore_chart(self, ax, indicator_data: Dict[str, List[float]]):
        """Plot Z-score chart"""
        if 'z_score' not in indicator_data or not indicator_data['z_score']:
            ax.text(0.5, 0.5, 'No Z-score data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Z-Score Analysis')
            return
            
        z_scores = indicator_data['z_score']
        timestamps = [datetime.now() - timedelta(minutes=i) 
                     for i in range(len(z_scores)-1, -1, -1)]
        
        ax.plot(timestamps, z_scores, color='purple', linewidth=1.5)
        ax.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Overbought (2)')
        ax.axhline(y=-2, color='green', linestyle='--', alpha=0.7, label='Oversold (-2)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax.set_title('Z-Score Analysis')
        ax.set_ylabel('Z-Score')
        ax.set_ylim(-4, 4)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_pairs_chart(self, ax, indicator_data: Dict[str, List[float]]):
        """Plot pairs trading chart"""
        if 'spread' not in indicator_data or not indicator_data['spread']:
            ax.text(0.5, 0.5, 'No pairs data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Pairs Trading Analysis')
            return
            
        spreads = indicator_data['spread']
        timestamps = [datetime.now() - timedelta(minutes=i) 
                     for i in range(len(spreads)-1, -1, -1)]
        
        ax.plot(timestamps, spreads, color='blue', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Add standard deviation bands if available
        if 'spread_std' in indicator_data and indicator_data['spread_std']:
            std = indicator_data['spread_std'][-1] if indicator_data['spread_std'] else 1
            ax.axhline(y=2*std, color='red', linestyle='--', alpha=0.7, label='+2σ')
            ax.axhline(y=-2*std, color='green', linestyle='--', alpha=0.7, label='-2σ')
        
        ax.set_title('Pairs Trading Spread')
        ax.set_ylabel('Spread')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_signal_strength(self, ax, signals: List[Dict]):
        """Plot signal strength over time"""
        if not signals:
            ax.text(0.5, 0.5, 'No signal data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Signal Strength')
            return
            
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in signals]
        strengths = [s['value'] for s in signals]
        confidences = [s['confidence'] for s in signals]
        
        # Plot signal strength
        ax.plot(timestamps, strengths, color='blue', linewidth=1.5, label='Signal Strength')
        ax.fill_between(timestamps, strengths, alpha=0.3, color='blue')
        
        # Plot confidence as secondary axis
        ax2 = ax.twinx()
        ax2.plot(timestamps, confidences, color='orange', linewidth=1.5, label='Confidence')
        ax2.fill_between(timestamps, confidences, alpha=0.3, color='orange')
        ax2.set_ylabel('Confidence', color='orange')
        
        ax.set_title('Signal Strength and Confidence')
        ax.set_ylabel('Signal Strength', color='blue')
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

class PortfolioAllocatorVisualizer:
    """Visualization panel for Portfolio Allocator Agent"""
    
    def __init__(self):
        self.fig = None
        self.axes = {}
        
    def create_portfolio_chart(self, portfolio_data: Dict[str, Any], 
                             allocation_history: List[Dict]) -> plt.Figure:
        """
        Create a comprehensive portfolio allocation chart
        
        Args:
            portfolio_data: Current portfolio data
            allocation_history: Historical allocation data
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Portfolio allocation pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_allocation_pie(ax1, portfolio_data)
        
        # Portfolio value over time
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_portfolio_value(ax2, allocation_history)
        
        # Risk metrics heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_risk_heatmap(ax3, portfolio_data)
        
        # Sector allocation
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_sector_allocation(ax4, portfolio_data)
        
        # Add title
        fig.suptitle('Portfolio Allocator Agent - Portfolio Analysis', fontsize=16, fontweight='bold')
        
        return fig
        
    def _plot_allocation_pie(self, ax, portfolio_data: Dict[str, Any]):
        """Plot portfolio allocation pie chart"""
        positions = portfolio_data.get('positions', {})
        if not positions:
            ax.text(0.5, 0.5, 'No positions available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Portfolio Allocation')
            return
            
        # Calculate position values
        symbols = list(positions.keys())
        values = [abs(pos.get('market_value', 0)) for pos in positions.values()]
        total_value = sum(values)
        
        if total_value == 0:
            ax.text(0.5, 0.5, 'No portfolio value', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Portfolio Allocation')
            return
            
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
        wedges, texts, autotexts = ax.pie(values, labels=symbols, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        ax.set_title('Portfolio Allocation')
        
    def _plot_portfolio_value(self, ax, allocation_history: List[Dict]):
        """Plot portfolio value over time"""
        if not allocation_history:
            ax.text(0.5, 0.5, 'No historical data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Portfolio Value Over Time')
            return
            
        timestamps = [datetime.fromisoformat(data['timestamp']) for data in allocation_history]
        values = [data.get('total_value', 0) for data in allocation_history]
        
        ax.plot(timestamps, values, color='blue', linewidth=2, marker='o')
        ax.set_title('Portfolio Value Over Time')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_risk_heatmap(self, ax, portfolio_data: Dict[str, Any]):
        """Plot risk metrics heatmap"""
        # Mock risk metrics data
        risk_metrics = {
            'Volatility': 0.15,
            'Max Drawdown': 0.08,
            'VaR 95%': 0.05,
            'Beta': 1.2,
            'Sharpe Ratio': 1.8
        }
        
        # Convert to matrix format for heatmap
        metrics_names = list(risk_metrics.keys())
        metrics_values = list(risk_metrics.values())
        
        # Reshape for heatmap
        data = np.array(metrics_values).reshape(-1, 1)
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
        ax.set_yticks(range(len(metrics_names)))
        ax.set_yticklabels(metrics_names)
        ax.set_xticks([])
        
        # Add value annotations
        for i in range(len(metrics_names)):
            ax.text(0, i, f'{metrics_values[i]:.2f}', 
                   ha='center', va='center', color='white' if data[i, 0] > 0.5 else 'black')
        
        ax.set_title('Risk Metrics')
        
    def _plot_sector_allocation(self, ax, portfolio_data: Dict[str, Any]):
        """Plot sector allocation bar chart"""
        positions = portfolio_data.get('positions', {})
        if not positions:
            ax.text(0.5, 0.5, 'No positions available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Sector Allocation')
            return
            
        # Mock sector data (in real implementation, this would come from position data)
        sectors = ['Technology', 'Financial', 'Healthcare', 'Energy', 'Consumer']
        allocations = [0.35, 0.20, 0.15, 0.15, 0.15]  # Mock allocations
        
        bars = ax.bar(sectors, allocations, color=plt.cm.Set3(np.linspace(0, 1, len(sectors))))
        ax.set_title('Sector Allocation')
        ax.set_ylabel('Allocation (%)')
        ax.set_ylim(0, 0.5)
        
        # Add value labels
        for bar, value in zip(bars, allocations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.1%}', ha='center', va='bottom')
        
        # Format x-axis
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Example usage functions
def create_momentum_dashboard(market_data: List[Dict], signals: List[Dict], 
                            indicator_data: Dict[str, List[float]]):
    """Create and display momentum agent dashboard"""
    visualizer = MomentumAgentVisualizer()
    fig = visualizer.create_technical_chart(market_data, signals, indicator_data)
    plt.tight_layout()
    plt.show()
    return fig

def create_mean_reversion_dashboard(market_data: List[Dict], signals: List[Dict], 
                                  indicator_data: Dict[str, List[float]]):
    """Create and display mean reversion agent dashboard"""
    visualizer = MeanReversionAgentVisualizer()
    fig = visualizer.create_mean_reversion_chart(market_data, signals, indicator_data)
    plt.tight_layout()
    plt.show()
    return fig

def create_portfolio_dashboard(portfolio_data: Dict[str, Any], 
                             allocation_history: List[Dict]):
    """Create and display portfolio allocator dashboard"""
    visualizer = PortfolioAllocatorVisualizer()
    fig = visualizer.create_portfolio_chart(portfolio_data, allocation_history)
    plt.tight_layout()
    plt.show()
    return fig

# Example data for testing
if __name__ == "__main__":
    # Example usage - create sample data and charts
    print("Creating sample visualizations...")
    
    # Sample market data
    sample_market_data = [
        {
            'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
            'open': 150 + np.random.normal(0, 1),
            'high': 152 + np.random.normal(0, 1),
            'low': 148 + np.random.normal(0, 1),
            'close': 150 + np.random.normal(0, 1),
            'volume': 1000000 + np.random.randint(-100000, 100000)
        }
        for i in range(30, 0, -1)
    ]
    
    # Sample signals
    sample_signals = [
        {
            'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
            'signal_type': 'buy' if np.random.random() > 0.5 else 'sell',
            'value': np.random.uniform(-1, 1),
            'confidence': np.random.uniform(0.5, 1.0)
        }
        for i in range(30, 0, -1)
    ][::5]  # Every 5th day
    
    # Sample indicator data
    sample_indicators = {
        'ema_fast': [150 + np.random.normal(0, 0.5) for _ in range(30)],
        'ema_slow': [150 + np.random.normal(0, 0.3) for _ in range(30)],
        'rsi': [50 + np.random.normal(0, 15) for _ in range(30)],
        'macd_line': [0 + np.random.normal(0, 0.1) for _ in range(30)],
        'signal_line': [0 + np.random.normal(0, 0.05) for _ in range(30)],
        'histogram': [0 + np.random.normal(0, 0.1) for _ in range(30)]
    }
    
    # Create sample charts (uncomment to test)
    # fig1 = create_momentum_dashboard(sample_market_data, sample_signals, sample_indicators)
    # fig2 = create_mean_reversion_dashboard(sample_market_data, sample_signals, sample_indicators)
    
    print("Sample visualization modules created successfully!")