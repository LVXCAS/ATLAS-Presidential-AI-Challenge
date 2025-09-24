"""Report generation for backtesting results."""

from typing import Dict, Any
import pandas as pd


class ReportGenerator:
    """Generate reports for backtesting results."""
    
    def __init__(self):
        """Initialize report generator."""
        pass
        
    def generate_report(self, results) -> str:
        """Generate a text report of backtest results."""
        
        report = []
        report.append("=" * 60)
        report.append("BACKTEST RESULTS REPORT")
        report.append("=" * 60)
        
        if hasattr(results, 'total_return'):
            report.append(f"Total Return: {results.total_return:.2%}")
            report.append(f"Annualized Return: {results.annual_return:.2%}")
            report.append(f"Volatility: {results.volatility:.2%}")
            report.append(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            report.append(f"Max Drawdown: {results.max_drawdown:.2%}")
            report.append("")
            report.append(f"Total Trades: {results.total_trades}")
            report.append(f"Win Rate: {results.win_rate:.2%}")
        
        return "\n".join(report)