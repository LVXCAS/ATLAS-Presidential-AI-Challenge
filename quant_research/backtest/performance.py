"""Performance analysis for backtesting."""

import pandas as pd
import numpy as np
from typing import Dict, Any


class PerformanceAnalyzer:
    """Performance analysis for backtesting results."""
    
    def __init__(self, config):
        """Initialize performance analyzer."""
        self.config = config
        self.risk_free_rate = getattr(config, 'risk_free_rate', 0.02)
        
    async def calculate_metrics(
        self,
        portfolio_history: pd.DataFrame,
        trades_history: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.
        
        Args:
            portfolio_history: Portfolio value history
            trades_history: Trade history
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        if portfolio_history.empty:
            return self._empty_metrics()
        
        # Calculate returns
        if 'total_value' in portfolio_history.columns:
            values = portfolio_history['total_value']
            returns = values.pct_change().dropna()
        else:
            returns = pd.Series()
        
        # Basic metrics
        if not returns.empty:
            total_return = (values.iloc[-1] / values.iloc[0]) - 1 if len(values) > 0 else 0
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe ratio
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = (annual_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
            
            # Calmar ratio
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            metrics.update({
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio
            })
        
        # Trade statistics
        if not trades_history.empty:
            total_trades = len(trades_history)
            
            # Count winning/losing trades (simplified)
            if 'realized_pnl' in trades_history.columns:
                winning_trades = len(trades_history[trades_history['realized_pnl'] > 0])
                losing_trades = len(trades_history[trades_history['realized_pnl'] < 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                wins = trades_history[trades_history['realized_pnl'] > 0]['realized_pnl']
                losses = trades_history[trades_history['realized_pnl'] < 0]['realized_pnl']
                
                avg_win = wins.mean() if len(wins) > 0 else 0
                avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
                profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 0
            else:
                winning_trades = 0
                losing_trades = 0
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            metrics.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            })
        else:
            metrics.update({
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            })
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dictionary."""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }