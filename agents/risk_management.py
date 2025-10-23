"""
Risk Management System
Handles position sizing, portfolio risk, and trading limits
"""

import sys
import os
from pathlib import Path

# Add project root to Python path to ensure local config is imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import logging
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config.logging_config import get_logger

logger = get_logger(__name__)

class RiskLevel(str, Enum):
    """Risk level classifications"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    
class RiskMetric(str, Enum):
    """Risk metrics to track"""
    POSITION_SIZE = "position_size"
    PORTFOLIO_HEAT = "portfolio_heat"  # Total risk exposure
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    SECTOR_CONCENTRATION = "sector_concentration"

@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size_pct: float = 5.0      # Max 5% per position
    max_sector_allocation_pct: float = 20.0  # Max 20% per sector
    max_portfolio_heat_pct: float = 15.0     # Max 15% portfolio at risk
    max_correlation: float = 0.7             # Max correlation between positions
    max_drawdown_pct: float = 10.0           # Stop trading at 10% drawdown
    max_daily_loss_pct: float = 3.0          # Max 3% daily loss
    max_consecutive_losses: int = 5          # Max 5 consecutive losing trades
    
    # Position-specific limits
    min_liquidity_volume: int = 100000       # Min daily volume
    max_volatility_pct: float = 50.0         # Max 50% volatility
    min_market_cap: float = 1e9              # Min $1B market cap

@dataclass 
class PositionRisk:
    """Risk assessment for a position"""
    symbol: str
    proposed_size: int
    proposed_value: float
    risk_pct: float              # Percentage of portfolio at risk
    volatility: float            # Expected volatility
    liquidity_score: float       # Liquidity rating 0-1
    sector: str
    correlation_risk: float      # Risk from correlation with existing positions
    overall_risk_score: float    # Combined risk score 0-1
    recommendation: str          # APPROVE, REDUCE, REJECT
    sizing_adjustment: float = 1.0  # Multiplier for position size

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.risk_level = risk_level
        self.risk_limits = self._get_risk_limits(risk_level)
        
        # Portfolio state
        self.account_value = 100000  # Default account value
        self.current_positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        
        # Sector classifications
        self.sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
            'NVDA': 'Technology', 'TSLA': 'Technology', 'META': 'Technology', 'NFLX': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
            'WMT': 'Consumer', 'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer',
            'CAT': 'Industrial', 'BA': 'Industrial', 'GE': 'Industrial', 'MMM': 'Industrial'
        }
        
        # Performance tracking
        self.risk_violations = []
        self.rejected_trades = []
        
    def _get_risk_limits(self, risk_level: RiskLevel) -> RiskLimits:
        """Get risk limits based on risk level"""
        
        if risk_level == RiskLevel.CONSERVATIVE:
            return RiskLimits(
                max_position_size_pct=3.0,
                max_sector_allocation_pct=15.0,
                max_portfolio_heat_pct=10.0,
                max_drawdown_pct=5.0,
                max_daily_loss_pct=2.0,
                max_consecutive_losses=3
            )
        elif risk_level == RiskLevel.AGGRESSIVE:
            return RiskLimits(
                max_position_size_pct=10.0,
                max_sector_allocation_pct=30.0,
                max_portfolio_heat_pct=25.0,
                max_drawdown_pct=15.0,
                max_daily_loss_pct=5.0,
                max_consecutive_losses=8
            )
        else:  # MODERATE
            return RiskLimits()
    
    def update_account_value(self, value: float):
        """Update account value for position sizing"""
        self.account_value = value
        logger.info(f"Updated account value to ${value:,.2f}")
    
    def update_position(self, symbol: str, quantity: int, price: float, pnl: float = 0):
        """Update position in portfolio"""
        self.current_positions[symbol] = {
            'quantity': quantity,
            'price': price,
            'value': abs(quantity) * price,
            'pnl': pnl,
            'sector': self.sector_map.get(symbol, 'Other')
        }
    
    def remove_position(self, symbol: str):
        """Remove position from portfolio"""
        if symbol in self.current_positions:
            del self.current_positions[symbol]
    
    def calculate_position_size(self, symbol: str, price: float, confidence: float, 
                              volatility: float = 0.20) -> Tuple[int, PositionRisk]:
        """Calculate optimal position size based on risk management"""
        
        # Base position size calculation using Kelly criterion
        win_rate = 0.55 + (confidence - 0.5) * 0.3  # Adjust win rate by confidence
        avg_win = 0.15  # Expected average win
        avg_loss = 0.08  # Expected average loss
        
        # Kelly fraction
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0.01, min(kelly_fraction, 0.20))  # Cap between 1% and 20%
        
        # Position size based on Kelly and risk limits
        max_position_value = self.account_value * self.risk_limits.max_position_size_pct / 100
        kelly_position_value = self.account_value * kelly_fraction
        
        target_position_value = min(max_position_value, kelly_position_value)
        base_quantity = int(target_position_value / price)
        
        # Risk assessment
        risk_assessment = self._assess_position_risk(symbol, base_quantity, price, volatility)
        
        # Adjust quantity based on risk
        adjusted_quantity = int(base_quantity * risk_assessment.sizing_adjustment)
        risk_assessment.proposed_size = adjusted_quantity
        risk_assessment.proposed_value = adjusted_quantity * price
        
        logger.info(f"Position sizing for {symbol}: Base={base_quantity}, Adjusted={adjusted_quantity}, "
                   f"Risk Score={risk_assessment.overall_risk_score:.2f}")
        
        return adjusted_quantity, risk_assessment
    
    def _assess_position_risk(self, symbol: str, quantity: int, price: float, 
                            volatility: float) -> PositionRisk:
        """Comprehensive position risk assessment"""
        
        position_value = quantity * price
        position_risk_pct = position_value / self.account_value * 100
        
        # Liquidity assessment (simplified)
        liquidity_score = 0.8  # Default good liquidity
        if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
            liquidity_score = 1.0
        elif symbol not in self.sector_map:
            liquidity_score = 0.5
        
        # Sector concentration risk
        sector = self.sector_map.get(symbol, 'Other')
        current_sector_exposure = sum(
            pos['value'] for pos in self.current_positions.values() 
            if pos['sector'] == sector
        )
        new_sector_exposure_pct = (current_sector_exposure + position_value) / self.account_value * 100
        
        # Correlation risk (simplified)
        correlation_risk = 0.0
        if sector == 'Technology':
            tech_positions = [s for s, p in self.current_positions.items() if p['sector'] == 'Technology']
            correlation_risk = len(tech_positions) * 0.1  # Higher risk with more tech positions
        
        # Overall risk score (0-1, higher = riskier)
        risk_factors = [
            position_risk_pct / self.risk_limits.max_position_size_pct,  # Position size risk
            volatility / 0.5,  # Volatility risk (normalized to 50% max)
            1 - liquidity_score,  # Liquidity risk
            new_sector_exposure_pct / self.risk_limits.max_sector_allocation_pct,  # Sector concentration
            correlation_risk  # Correlation risk
        ]
        
        overall_risk_score = sum(risk_factors) / len(risk_factors)
        overall_risk_score = min(1.0, max(0.0, overall_risk_score))
        
        # Risk-based recommendations
        sizing_adjustment = 1.0
        recommendation = "APPROVE"
        
        if overall_risk_score > 0.8:
            recommendation = "REJECT"
            sizing_adjustment = 0.0
        elif overall_risk_score > 0.6:
            recommendation = "REDUCE"
            sizing_adjustment = 0.5
        elif overall_risk_score > 0.4:
            sizing_adjustment = 0.75
        
        # Additional checks
        if position_risk_pct > self.risk_limits.max_position_size_pct:
            recommendation = "REDUCE"
            sizing_adjustment = min(sizing_adjustment, self.risk_limits.max_position_size_pct / position_risk_pct)
        
        if new_sector_exposure_pct > self.risk_limits.max_sector_allocation_pct:
            recommendation = "REDUCE" if recommendation != "REJECT" else "REJECT"
            sector_adjustment = (self.risk_limits.max_sector_allocation_pct - 
                               (new_sector_exposure_pct - position_value / self.account_value * 100)) / (position_value / self.account_value * 100)
            sizing_adjustment = min(sizing_adjustment, max(0, sector_adjustment))
        
        return PositionRisk(
            symbol=symbol,
            proposed_size=quantity,
            proposed_value=position_value,
            risk_pct=position_risk_pct,
            volatility=volatility,
            liquidity_score=liquidity_score,
            sector=sector,
            correlation_risk=correlation_risk,
            overall_risk_score=overall_risk_score,
            recommendation=recommendation,
            sizing_adjustment=sizing_adjustment
        )
    
    def check_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is currently allowed based on risk limits"""
        
        # Check drawdown limit
        if self.max_drawdown > self.risk_limits.max_drawdown_pct:
            return False, f"Max drawdown exceeded: {self.max_drawdown:.1f}%"
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.account_value * 100
        if self.daily_pnl < 0 and daily_loss_pct > self.risk_limits.max_daily_loss_pct:
            return False, f"Daily loss limit exceeded: {daily_loss_pct:.1f}%"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.risk_limits.max_consecutive_losses:
            return False, f"Too many consecutive losses: {self.consecutive_losses}"
        
        # Check portfolio heat
        total_risk = self.calculate_portfolio_heat()
        if total_risk > self.risk_limits.max_portfolio_heat_pct:
            return False, f"Portfolio heat too high: {total_risk:.1f}%"
        
        return True, "Trading allowed"
    
    def calculate_portfolio_heat(self) -> float:
        """Calculate total portfolio risk exposure"""
        
        if not self.current_positions:
            return 0.0
        
        # Estimate risk as percentage of portfolio potentially at risk
        total_value = sum(pos['value'] for pos in self.current_positions.values())
        
        # Assume average position could lose 10% (simplified)
        estimated_risk_per_position = 0.10
        total_risk = total_value * estimated_risk_per_position / self.account_value * 100
        
        return total_risk
    
    def record_trade_outcome(self, pnl: float):
        """Record trade outcome for risk tracking"""
        
        self.daily_pnl += pnl
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update max drawdown
        current_drawdown = abs(min(0, self.daily_pnl)) / self.account_value * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        
        portfolio_value = sum(pos['value'] for pos in self.current_positions.values())
        portfolio_heat = self.calculate_portfolio_heat()
        
        # Sector allocation
        sector_allocations = {}
        for pos in self.current_positions.values():
            sector = pos['sector']
            sector_allocations[sector] = sector_allocations.get(sector, 0) + pos['value']
        
        sector_allocations = {k: v / self.account_value * 100 for k, v in sector_allocations.items()}
        
        return {
            'account_value': self.account_value,
            'portfolio_value': portfolio_value,
            'portfolio_allocation_pct': portfolio_value / self.account_value * 100,
            'portfolio_heat_pct': portfolio_heat,
            'max_drawdown_pct': self.max_drawdown,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'sector_allocations': sector_allocations,
            'positions_count': len(self.current_positions),
            'risk_limits': {
                'max_position_size_pct': self.risk_limits.max_position_size_pct,
                'max_sector_allocation_pct': self.risk_limits.max_sector_allocation_pct,
                'max_portfolio_heat_pct': self.risk_limits.max_portfolio_heat_pct,
                'max_drawdown_pct': self.risk_limits.max_drawdown_pct
            }
        }
    
    def suggest_position_adjustments(self) -> List[Dict]:
        """Suggest adjustments to current positions for better risk management"""
        
        suggestions = []
        
        # Check for oversized positions
        for symbol, pos in self.current_positions.items():
            position_pct = pos['value'] / self.account_value * 100
            if position_pct > self.risk_limits.max_position_size_pct:
                reduction = position_pct - self.risk_limits.max_position_size_pct
                suggestions.append({
                    'type': 'REDUCE_POSITION',
                    'symbol': symbol,
                    'reason': f'Position too large: {position_pct:.1f}%',
                    'suggested_reduction_pct': reduction / position_pct * 100
                })
        
        # Check for sector concentration
        sector_allocations = {}
        for pos in self.current_positions.values():
            sector = pos['sector']
            sector_allocations[sector] = sector_allocations.get(sector, 0) + pos['value']
        
        for sector, value in sector_allocations.items():
            sector_pct = value / self.account_value * 100
            if sector_pct > self.risk_limits.max_sector_allocation_pct:
                suggestions.append({
                    'type': 'REDUCE_SECTOR',
                    'sector': sector,
                    'reason': f'Sector overweight: {sector_pct:.1f}%',
                    'target_allocation': self.risk_limits.max_sector_allocation_pct
                })
        
        # Check portfolio heat
        current_heat = self.calculate_portfolio_heat()
        if current_heat > self.risk_limits.max_portfolio_heat_pct:
            suggestions.append({
                'type': 'REDUCE_OVERALL_RISK',
                'reason': f'Portfolio heat too high: {current_heat:.1f}%',
                'target_heat': self.risk_limits.max_portfolio_heat_pct
            })
        
        return suggestions
    
    def reset_daily_metrics(self):
        """Reset daily tracking metrics"""
        self.daily_pnl = 0.0
        logger.info("Reset daily risk metrics")