"""
Risk Manager Agent - LangGraph-based risk monitoring and safety controls

This agent implements comprehensive risk management including real-time position monitoring,
VaR calculation, dynamic position limits, emergency circuit breakers, and correlation risk management.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from langgraph.graph import StateGraph, END
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import math
from scipy import stats
from sklearn.covariance import LedoitWolf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAlertSeverity(Enum):
    """Risk alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskAlertType(Enum):
    """Types of risk alerts"""
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    POSITION_LIMIT = "POSITION_LIMIT"
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"
    VAR_EXCEEDED = "VAR_EXCEEDED"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    CORRELATION_RISK = "CORRELATION_RISK"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
    SYSTEM_ANOMALY = "SYSTEM_ANOMALY"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class EmergencyAction(Enum):
    """Emergency actions that can be taken"""
    HALT_ALL_TRADING = "HALT_ALL_TRADING"
    REDUCE_POSITIONS = "REDUCE_POSITIONS"
    CLOSE_RISKY_POSITIONS = "CLOSE_RISKY_POSITIONS"
    INCREASE_CASH = "INCREASE_CASH"
    ACTIVATE_HEDGES = "ACTIVATE_HEDGES"


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_daily_loss_pct: float = 10.0  # Maximum daily loss percentage
    max_position_size_pct: float = 5.0  # Maximum single position size
    max_leverage: float = 2.0  # Maximum portfolio leverage
    max_var_95_pct: float = 3.0  # Maximum 1-day VaR at 95% confidence
    max_correlation: float = 0.8  # Maximum correlation between positions
    min_liquidity_days: int = 5  # Minimum days to liquidate position
    max_sector_concentration_pct: float = 20.0  # Maximum sector concentration
    volatility_spike_threshold: float = 2.0  # Volatility spike multiplier


@dataclass
class RiskAlert:
    """Risk alert data structure"""
    timestamp: datetime
    alert_type: RiskAlertType
    severity: RiskAlertSeverity
    symbol: Optional[str]
    strategy: Optional[str]
    agent_name: Optional[str]
    current_value: float
    limit_value: float
    breach_percentage: float
    description: str
    action_taken: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics"""
    timestamp: datetime
    portfolio_value: float
    cash: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    var_1d_95: float
    var_1d_99: float
    var_5d_95: float
    var_5d_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    max_position_size: float
    max_position_pct: float
    sector_concentration: float
    correlation_risk: float
    liquidity_risk: float


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    exchange: str
    strategy: str
    agent_name: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    weight_pct: float


class RiskManagerAgent:
    """
    LangGraph-based Risk Manager Agent for comprehensive risk monitoring and safety controls.
    
    This agent provides:
    - Real-time position monitoring and VaR calculation
    - Dynamic position limits and exposure controls
    - Emergency circuit breakers and kill switch
    - Correlation risk management
    - Automated risk alerts and actions
    """
    
    def __init__(self, db_config: Dict[str, Any], risk_limits: Optional[RiskLimits] = None):
        """
        Initialize the Risk Manager Agent.
        
        Args:
            db_config: Database configuration
            risk_limits: Risk limits configuration
        """
        self.db_config = db_config
        self.risk_limits = risk_limits or RiskLimits()
        self.emergency_stop_active = False
        self.last_risk_check = None
        
        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
        
        logger.info("Risk Manager Agent initialized")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for risk management"""
        
        # Use TypedDict for state instead of dataclass for better LangGraph compatibility
        from typing import TypedDict
        
        class RiskState(TypedDict):
            positions: List[Position]
            portfolio_metrics: Optional[PortfolioRiskMetrics]
            risk_alerts: List[RiskAlert]
            emergency_actions: List[EmergencyAction]
            market_data: Dict[str, Any]
            
        workflow = StateGraph(RiskState)
        
        # Add nodes
        workflow.add_node("load_positions", self._load_positions)
        workflow.add_node("calculate_risk_metrics", self._calculate_risk_metrics)
        workflow.add_node("check_risk_limits", self._check_risk_limits)
        workflow.add_node("generate_alerts", self._generate_alerts)
        workflow.add_node("execute_emergency_actions", self._execute_emergency_actions)
        workflow.add_node("update_risk_database", self._update_risk_database)
        
        # Define workflow edges
        workflow.add_edge("load_positions", "calculate_risk_metrics")
        workflow.add_edge("calculate_risk_metrics", "check_risk_limits")
        workflow.add_edge("check_risk_limits", "generate_alerts")
        workflow.add_edge("generate_alerts", "execute_emergency_actions")
        workflow.add_edge("execute_emergency_actions", "update_risk_database")
        workflow.add_edge("update_risk_database", END)
        
        workflow.set_entry_point("load_positions")
        
        return workflow.compile()
    
    async def monitor_portfolio_risk(self) -> PortfolioRiskMetrics:
        """
        Main entry point for portfolio risk monitoring.
        
        Returns:
            PortfolioRiskMetrics: Current portfolio risk metrics
        """
        try:
            # Execute the risk monitoring workflow
            initial_state = {
                'positions': [],
                'portfolio_metrics': None,
                'risk_alerts': [],
                'emergency_actions': [],
                'market_data': {}
            }
            result = await self.workflow.ainvoke(initial_state)
            
            self.last_risk_check = datetime.now(timezone.utc)
            
            if result.get('portfolio_metrics'):
                return result['portfolio_metrics']
            else:
                raise Exception("Failed to calculate portfolio risk metrics")
                
        except Exception as e:
            logger.error(f"Error in portfolio risk monitoring: {e}")
            raise
    
    async def _load_positions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Load current positions from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Load current positions
            cursor.execute("""
                SELECT 
                    symbol, exchange, strategy, agent_name,
                    quantity, avg_cost, market_value, 
                    unrealized_pnl, realized_pnl
                FROM positions 
                WHERE quantity != 0
                ORDER BY ABS(market_value) DESC
            """)
            
            positions_data = cursor.fetchall()
            
            # Calculate portfolio total value for position weights
            total_value = sum(abs(pos['market_value']) for pos in positions_data)
            
            positions = []
            for pos_data in positions_data:
                weight_pct = (abs(pos_data['market_value']) / total_value * 100) if total_value > 0 else 0
                
                position = Position(
                    symbol=pos_data['symbol'],
                    exchange=pos_data['exchange'],
                    strategy=pos_data['strategy'],
                    agent_name=pos_data['agent_name'],
                    quantity=float(pos_data['quantity']),
                    avg_cost=float(pos_data['avg_cost']),
                    market_value=float(pos_data['market_value']),
                    unrealized_pnl=float(pos_data['unrealized_pnl']),
                    realized_pnl=float(pos_data['realized_pnl']),
                    weight_pct=weight_pct
                )
                positions.append(position)
            
            cursor.close()
            conn.close()
            
            state['positions'] = positions
            logger.info(f"Loaded {len(positions)} positions for risk monitoring")
            
            return state
            
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
            state['positions'] = []
            return state
    
    async def _calculate_risk_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            positions = state.get('positions', [])
            
            if not positions:
                logger.warning("No positions found for risk calculation")
                return state
            
            # Calculate basic portfolio metrics
            portfolio_value = sum(pos.market_value for pos in positions)
            long_value = sum(pos.market_value for pos in positions if pos.market_value > 0)
            short_value = sum(abs(pos.market_value) for pos in positions if pos.market_value < 0)
            gross_exposure = long_value + short_value
            net_exposure = long_value - short_value
            
            # Get cash position from portfolio snapshots
            cash = await self._get_current_cash()
            total_value = portfolio_value + cash
            leverage = gross_exposure / total_value if total_value > 0 else 0
            
            # Calculate VaR using historical simulation
            var_metrics = await self._calculate_var(positions)
            
            # Calculate position concentration
            max_position_size = max((abs(pos.market_value) for pos in positions), default=0)
            max_position_pct = (max_position_size / total_value * 100) if total_value > 0 else 0
            
            # Calculate sector concentration
            sector_concentration = await self._calculate_sector_concentration(positions)
            
            # Calculate correlation risk
            correlation_risk = await self._calculate_correlation_risk(positions)
            
            # Calculate liquidity risk
            liquidity_risk = await self._calculate_liquidity_risk(positions)
            
            # Create risk metrics object
            risk_metrics = PortfolioRiskMetrics(
                timestamp=datetime.now(timezone.utc),
                portfolio_value=portfolio_value,
                cash=cash,
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
                leverage=leverage,
                var_1d_95=var_metrics.get('var_1d_95', 0),
                var_1d_99=var_metrics.get('var_1d_99', 0),
                var_5d_95=var_metrics.get('var_5d_95', 0),
                var_5d_99=var_metrics.get('var_5d_99', 0),
                expected_shortfall_95=var_metrics.get('expected_shortfall_95', 0),
                expected_shortfall_99=var_metrics.get('expected_shortfall_99', 0),
                max_position_size=max_position_size,
                max_position_pct=max_position_pct,
                sector_concentration=sector_concentration,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk
            )
            
            state['portfolio_metrics'] = risk_metrics
            logger.info(f"Calculated risk metrics - Leverage: {leverage:.2f}, VaR 95%: {var_metrics.get('var_1d_95', 0):.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return state    

    async def _calculate_var(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate Value at Risk using historical simulation"""
        try:
            if not positions:
                return {'var_1d_95': 0, 'var_1d_99': 0, 'var_5d_95': 0, 'var_5d_99': 0, 
                       'expected_shortfall_95': 0, 'expected_shortfall_99': 0}
            
            # Get historical returns for all positions
            symbols = [pos.symbol for pos in positions]
            returns_data = await self._get_historical_returns(symbols, days=252)  # 1 year of data
            
            if returns_data.empty:
                logger.warning("No historical data available for VaR calculation")
                return {'var_1d_95': 0, 'var_1d_99': 0, 'var_5d_95': 0, 'var_5d_99': 0,
                       'expected_shortfall_95': 0, 'expected_shortfall_99': 0}
            
            # Calculate portfolio returns
            weights = np.array([pos.market_value for pos in positions])
            total_value = np.sum(np.abs(weights))
            weights = weights / total_value if total_value > 0 else weights
            
            # Align returns data with positions
            portfolio_returns = []
            for date in returns_data.index:
                daily_return = 0
                for i, pos in enumerate(positions):
                    if pos.symbol in returns_data.columns:
                        daily_return += weights[i] * returns_data.loc[date, pos.symbol]
                portfolio_returns.append(daily_return)
            
            portfolio_returns = np.array(portfolio_returns)
            portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]
            
            if len(portfolio_returns) < 30:
                logger.warning("Insufficient data for reliable VaR calculation")
                return {'var_1d_95': 0, 'var_1d_99': 0, 'var_5d_95': 0, 'var_5d_99': 0,
                       'expected_shortfall_95': 0, 'expected_shortfall_99': 0}
            
            # Calculate VaR at different confidence levels
            var_1d_95 = np.percentile(portfolio_returns, 5) * total_value
            var_1d_99 = np.percentile(portfolio_returns, 1) * total_value
            
            # Scale to 5-day VaR (assuming sqrt of time scaling)
            var_5d_95 = var_1d_95 * np.sqrt(5)
            var_5d_99 = var_1d_99 * np.sqrt(5)
            
            # Calculate Expected Shortfall (Conditional VaR)
            tail_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]
            tail_99 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)]
            
            expected_shortfall_95 = np.mean(tail_95) * total_value if len(tail_95) > 0 else var_1d_95
            expected_shortfall_99 = np.mean(tail_99) * total_value if len(tail_99) > 0 else var_1d_99
            
            return {
                'var_1d_95': abs(var_1d_95),
                'var_1d_99': abs(var_1d_99),
                'var_5d_95': abs(var_5d_95),
                'var_5d_99': abs(var_5d_99),
                'expected_shortfall_95': abs(expected_shortfall_95),
                'expected_shortfall_99': abs(expected_shortfall_99)
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {'var_1d_95': 0, 'var_1d_99': 0, 'var_5d_95': 0, 'var_5d_99': 0,
                   'expected_shortfall_95': 0, 'expected_shortfall_99': 0}
    
    async def _get_historical_returns(self, symbols: List[str], days: int = 252) -> pd.DataFrame:
        """Get historical returns for symbols"""
        try:
            conn = psycopg2.connect(**self.db_config)
            
            # Get historical price data
            query = """
                SELECT symbol, date, close, 
                       LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close
                FROM market_data_daily 
                WHERE symbol = ANY(%s) 
                AND date >= %s
                ORDER BY symbol, date
            """
            
            start_date = datetime.now().date() - timedelta(days=days)
            df = pd.read_sql_query(query, conn, params=(symbols, start_date))
            conn.close()
            
            # Calculate returns
            df['return'] = (df['close'] - df['prev_close']) / df['prev_close']
            df = df.dropna()
            
            # Pivot to get returns by symbol
            returns_df = df.pivot(index='date', columns='symbol', values='return')
            returns_df = returns_df.fillna(0)  # Fill missing values with 0 return
            
            return returns_df
            
        except Exception as e:
            logger.error(f"Error getting historical returns: {e}")
            return pd.DataFrame()
    
    async def _get_current_cash(self) -> float:
        """Get current cash position from latest portfolio snapshot"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT cash FROM portfolio_snapshots 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return float(result[0]) if result else 0.0
            
        except Exception as e:
            logger.error(f"Error getting current cash: {e}")
            return 0.0
    
    async def _calculate_sector_concentration(self, positions: List[Position]) -> float:
        """Calculate sector concentration risk"""
        try:
            # Group positions by sector (simplified - using first 3 chars of symbol as sector proxy)
            sector_exposure = {}
            total_exposure = sum(abs(pos.market_value) for pos in positions)
            
            for pos in positions:
                # Simplified sector classification - in production, use proper sector mapping
                sector = pos.symbol[:3] if len(pos.symbol) >= 3 else pos.symbol
                sector_exposure[sector] = sector_exposure.get(sector, 0) + abs(pos.market_value)
            
            # Calculate maximum sector concentration
            max_sector_exposure = max(sector_exposure.values()) if sector_exposure else 0
            concentration_pct = (max_sector_exposure / total_exposure * 100) if total_exposure > 0 else 0
            
            return concentration_pct
            
        except Exception as e:
            logger.error(f"Error calculating sector concentration: {e}")
            return 0.0
    
    async def _calculate_correlation_risk(self, positions: List[Position]) -> float:
        """Calculate correlation risk between positions"""
        try:
            if len(positions) < 2:
                return 0.0
            
            symbols = [pos.symbol for pos in positions]
            returns_data = await self._get_historical_returns(symbols, days=60)  # 3 months
            
            if returns_data.empty or len(returns_data.columns) < 2:
                return 0.0
            
            # Calculate correlation matrix
            correlation_matrix = returns_data.corr()
            
            # Calculate average correlation (excluding diagonal)
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if not np.isnan(corr_value):
                        correlations.append(abs(corr_value))
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
            return avg_correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    async def _calculate_liquidity_risk(self, positions: List[Position]) -> float:
        """Calculate liquidity risk based on position sizes and average volumes"""
        try:
            if not positions:
                return 0.0
            
            conn = psycopg2.connect(**self.db_config)
            
            # Get average volumes for positions
            symbols = [pos.symbol for pos in positions]
            query = """
                SELECT symbol, AVG(volume) as avg_volume
                FROM market_data_daily 
                WHERE symbol = ANY(%s) 
                AND date >= %s
                GROUP BY symbol
            """
            
            start_date = datetime.now().date() - timedelta(days=30)
            df = pd.read_sql_query(query, conn, params=(symbols, start_date))
            conn.close()
            
            # Calculate days to liquidate for each position
            liquidity_scores = []
            for pos in positions:
                symbol_volume = df[df['symbol'] == pos.symbol]['avg_volume'].iloc[0] if pos.symbol in df['symbol'].values else 1
                days_to_liquidate = abs(pos.quantity) / (symbol_volume * 0.1)  # Assume 10% of volume participation
                liquidity_scores.append(min(days_to_liquidate, 30))  # Cap at 30 days
            
            # Return weighted average liquidity risk
            weights = [abs(pos.market_value) for pos in positions]
            total_weight = sum(weights)
            
            if total_weight > 0:
                weighted_liquidity = sum(score * weight for score, weight in zip(liquidity_scores, weights)) / total_weight
                return weighted_liquidity
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return 0.0
    
    async def _check_risk_limits(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check all risk limits and identify breaches"""
        try:
            portfolio_metrics = state.get('portfolio_metrics')
            positions = state.get('positions', [])
            
            if not portfolio_metrics:
                return state
            
            risk_alerts = []
            
            # Check daily loss limit
            daily_pnl = sum(pos.unrealized_pnl for pos in positions)
            daily_loss_pct = abs(daily_pnl / portfolio_metrics.portfolio_value * 100) if portfolio_metrics.portfolio_value > 0 else 0
            
            if daily_pnl < 0 and daily_loss_pct > self.risk_limits.max_daily_loss_pct:
                alert = RiskAlert(
                    timestamp=datetime.now(timezone.utc),
                    alert_type=RiskAlertType.DAILY_LOSS_LIMIT,
                    severity=RiskAlertSeverity.CRITICAL,
                    symbol=None,
                    strategy=None,
                    agent_name=None,
                    current_value=daily_loss_pct,
                    limit_value=self.risk_limits.max_daily_loss_pct,
                    breach_percentage=(daily_loss_pct - self.risk_limits.max_daily_loss_pct) / self.risk_limits.max_daily_loss_pct * 100,
                    description=f"Daily loss of {daily_loss_pct:.2f}% exceeds limit of {self.risk_limits.max_daily_loss_pct}%"
                )
                risk_alerts.append(alert)
            
            # Check position size limits
            for pos in positions:
                if pos.weight_pct > self.risk_limits.max_position_size_pct:
                    alert = RiskAlert(
                        timestamp=datetime.now(timezone.utc),
                        alert_type=RiskAlertType.POSITION_LIMIT,
                        severity=RiskAlertSeverity.HIGH,
                        symbol=pos.symbol,
                        strategy=pos.strategy,
                        agent_name=pos.agent_name,
                        current_value=pos.weight_pct,
                        limit_value=self.risk_limits.max_position_size_pct,
                        breach_percentage=(pos.weight_pct - self.risk_limits.max_position_size_pct) / self.risk_limits.max_position_size_pct * 100,
                        description=f"Position {pos.symbol} size {pos.weight_pct:.2f}% exceeds limit of {self.risk_limits.max_position_size_pct}%"
                    )
                    risk_alerts.append(alert)
            
            # Check leverage limit
            if portfolio_metrics.leverage > self.risk_limits.max_leverage:
                alert = RiskAlert(
                    timestamp=datetime.now(timezone.utc),
                    alert_type=RiskAlertType.LEVERAGE_LIMIT,
                    severity=RiskAlertSeverity.HIGH,
                    symbol=None,
                    strategy=None,
                    agent_name=None,
                    current_value=portfolio_metrics.leverage,
                    limit_value=self.risk_limits.max_leverage,
                    breach_percentage=(portfolio_metrics.leverage - self.risk_limits.max_leverage) / self.risk_limits.max_leverage * 100,
                    description=f"Portfolio leverage {portfolio_metrics.leverage:.2f} exceeds limit of {self.risk_limits.max_leverage}"
                )
                risk_alerts.append(alert)
            
            # Check VaR limit
            var_limit = portfolio_metrics.portfolio_value * self.risk_limits.max_var_95_pct / 100
            if portfolio_metrics.var_1d_95 > var_limit:
                alert = RiskAlert(
                    timestamp=datetime.now(timezone.utc),
                    alert_type=RiskAlertType.VAR_EXCEEDED,
                    severity=RiskAlertSeverity.HIGH,
                    symbol=None,
                    strategy=None,
                    agent_name=None,
                    current_value=portfolio_metrics.var_1d_95,
                    limit_value=var_limit,
                    breach_percentage=(portfolio_metrics.var_1d_95 - var_limit) / var_limit * 100,
                    description=f"VaR 95% ${portfolio_metrics.var_1d_95:.2f} exceeds limit of ${var_limit:.2f}"
                )
                risk_alerts.append(alert)
            
            # Check correlation risk
            if portfolio_metrics.correlation_risk > self.risk_limits.max_correlation:
                alert = RiskAlert(
                    timestamp=datetime.now(timezone.utc),
                    alert_type=RiskAlertType.CORRELATION_RISK,
                    severity=RiskAlertSeverity.MEDIUM,
                    symbol=None,
                    strategy=None,
                    agent_name=None,
                    current_value=portfolio_metrics.correlation_risk,
                    limit_value=self.risk_limits.max_correlation,
                    breach_percentage=(portfolio_metrics.correlation_risk - self.risk_limits.max_correlation) / self.risk_limits.max_correlation * 100,
                    description=f"Average correlation {portfolio_metrics.correlation_risk:.3f} exceeds limit of {self.risk_limits.max_correlation}"
                )
                risk_alerts.append(alert)
            
            # Check sector concentration
            if portfolio_metrics.sector_concentration > self.risk_limits.max_sector_concentration_pct:
                alert = RiskAlert(
                    timestamp=datetime.now(timezone.utc),
                    alert_type=RiskAlertType.POSITION_LIMIT,
                    severity=RiskAlertSeverity.MEDIUM,
                    symbol=None,
                    strategy=None,
                    agent_name=None,
                    current_value=portfolio_metrics.sector_concentration,
                    limit_value=self.risk_limits.max_sector_concentration_pct,
                    breach_percentage=(portfolio_metrics.sector_concentration - self.risk_limits.max_sector_concentration_pct) / self.risk_limits.max_sector_concentration_pct * 100,
                    description=f"Sector concentration {portfolio_metrics.sector_concentration:.2f}% exceeds limit of {self.risk_limits.max_sector_concentration_pct}%"
                )
                risk_alerts.append(alert)
            
            state['risk_alerts'] = risk_alerts
            logger.info(f"Risk limit check completed - {len(risk_alerts)} alerts generated")
            
            return state
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            state['risk_alerts'] = []
            return state
    
    async def _generate_alerts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and process risk alerts"""
        try:
            risk_alerts = state.get('risk_alerts', [])
            emergency_actions = []
            
            # Determine emergency actions based on alert severity
            critical_alerts = [alert for alert in risk_alerts if alert.severity == RiskAlertSeverity.CRITICAL]
            high_alerts = [alert for alert in risk_alerts if alert.severity == RiskAlertSeverity.HIGH]
            
            if critical_alerts:
                # Critical alerts trigger emergency stop
                emergency_actions.append(EmergencyAction.HALT_ALL_TRADING)
                logger.critical(f"CRITICAL RISK ALERT: Emergency stop triggered due to {len(critical_alerts)} critical alerts")
                
                for alert in critical_alerts:
                    alert.action_taken = "EMERGENCY_STOP_TRIGGERED"
            
            elif len(high_alerts) >= 2:
                # Multiple high alerts trigger position reduction
                emergency_actions.append(EmergencyAction.REDUCE_POSITIONS)
                logger.warning(f"HIGH RISK ALERT: Position reduction triggered due to {len(high_alerts)} high alerts")
                
                for alert in high_alerts:
                    alert.action_taken = "POSITION_REDUCTION_TRIGGERED"
            
            elif high_alerts:
                # Single high alert triggers increased monitoring
                logger.warning(f"HIGH RISK ALERT: Increased monitoring triggered")
                
                for alert in high_alerts:
                    alert.action_taken = "INCREASED_MONITORING"
            
            # Log all alerts
            for alert in risk_alerts:
                logger.warning(f"Risk Alert: {alert.alert_type.value} - {alert.description}")
            
            state['emergency_actions'] = emergency_actions
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return state
    
    async def _execute_emergency_actions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute emergency actions based on risk alerts"""
        try:
            emergency_actions = state.get('emergency_actions', [])
            
            for action in emergency_actions:
                if action == EmergencyAction.HALT_ALL_TRADING:
                    await self._halt_all_trading()
                elif action == EmergencyAction.REDUCE_POSITIONS:
                    await self._reduce_positions(state.get('positions', []))
                elif action == EmergencyAction.CLOSE_RISKY_POSITIONS:
                    await self._close_risky_positions(state.get('positions', []))
                elif action == EmergencyAction.INCREASE_CASH:
                    await self._increase_cash_position()
                elif action == EmergencyAction.ACTIVATE_HEDGES:
                    await self._activate_hedges()
            
            return state
            
        except Exception as e:
            logger.error(f"Error executing emergency actions: {e}")
            return state
    
    async def _halt_all_trading(self):
        """Halt all trading activities - emergency stop"""
        try:
            self.emergency_stop_active = True
            
            # In a real implementation, this would:
            # 1. Cancel all pending orders
            # 2. Notify all trading agents to stop
            # 3. Close broker connections
            # 4. Send emergency notifications
            
            logger.critical("EMERGENCY STOP ACTIVATED - All trading halted")
            
            # Store emergency stop in database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO risk_alerts 
                (timestamp, alert_type, severity, description, action_taken)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                datetime.now(timezone.utc),
                RiskAlertType.EMERGENCY_STOP.value,
                RiskAlertSeverity.CRITICAL.value,
                "Emergency stop activated due to critical risk limit breach",
                "ALL_TRADING_HALTED"
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error halting trading: {e}")
    
    async def _reduce_positions(self, positions: List[Position]):
        """Reduce position sizes to manage risk"""
        try:
            # Identify largest positions for reduction
            sorted_positions = sorted(positions, key=lambda x: abs(x.market_value), reverse=True)
            positions_to_reduce = sorted_positions[:min(5, len(sorted_positions))]  # Top 5 positions
            
            for pos in positions_to_reduce:
                reduction_pct = 0.25  # Reduce by 25%
                
                # In a real implementation, this would submit orders to reduce positions
                logger.warning(f"Reducing position {pos.symbol} by {reduction_pct*100}%")
                
                # Store action in database
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO risk_alerts 
                    (timestamp, alert_type, severity, symbol, strategy, agent_name, description, action_taken)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    datetime.now(timezone.utc),
                    RiskAlertType.POSITION_LIMIT.value,
                    RiskAlertSeverity.HIGH.value,
                    pos.symbol,
                    pos.strategy,
                    pos.agent_name,
                    f"Position reduction triggered for {pos.symbol}",
                    f"REDUCE_POSITION_{reduction_pct*100}%"
                ))
                
                conn.commit()
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error reducing positions: {e}")
    
    async def _close_risky_positions(self, positions: List[Position]):
        """Close positions that exceed risk thresholds"""
        # Implementation would identify and close risky positions
        pass
    
    async def _increase_cash_position(self):
        """Increase cash position by selling some holdings"""
        # Implementation would sell positions to increase cash
        pass
    
    async def _activate_hedges(self):
        """Activate hedging strategies"""
        # Implementation would activate hedging positions
        pass
    
    async def _update_risk_database(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update risk metrics and alerts in database"""
        try:
            portfolio_metrics = state.get('portfolio_metrics')
            risk_alerts = state.get('risk_alerts', [])
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Insert risk metrics
            if portfolio_metrics:
                cursor.execute("""
                    INSERT INTO risk_metrics (
                        timestamp, portfolio_value, cash, gross_exposure, net_exposure,
                        leverage, var_1d_95, var_1d_99, var_5d_95, var_5d_99,
                        expected_shortfall_95, expected_shortfall_99, max_position_size,
                        max_position_pct, sector_concentration, correlation_risk, liquidity_risk
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    portfolio_metrics.timestamp,
                    portfolio_metrics.portfolio_value,
                    portfolio_metrics.cash,
                    portfolio_metrics.gross_exposure,
                    portfolio_metrics.net_exposure,
                    portfolio_metrics.leverage,
                    portfolio_metrics.var_1d_95,
                    portfolio_metrics.var_1d_99,
                    portfolio_metrics.var_5d_95,
                    portfolio_metrics.var_5d_99,
                    portfolio_metrics.expected_shortfall_95,
                    portfolio_metrics.expected_shortfall_99,
                    portfolio_metrics.max_position_size,
                    portfolio_metrics.max_position_pct,
                    portfolio_metrics.sector_concentration,
                    portfolio_metrics.correlation_risk,
                    portfolio_metrics.liquidity_risk
                ))
            
            # Insert risk alerts
            for alert in risk_alerts:
                cursor.execute("""
                    INSERT INTO risk_alerts (
                        timestamp, alert_type, severity, symbol, strategy, agent_name,
                        current_value, limit_value, breach_percentage, description, action_taken
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    alert.timestamp,
                    alert.alert_type.value,
                    alert.severity.value,
                    alert.symbol,
                    alert.strategy,
                    alert.agent_name,
                    alert.current_value,
                    alert.limit_value,
                    alert.breach_percentage,
                    alert.description,
                    alert.action_taken
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Updated risk database - {len(risk_alerts)} alerts stored")
            
            return state
            
        except Exception as e:
            logger.error(f"Error updating risk database: {e}")
            return state
    
    async def check_position_limits(self, new_order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if a new order would breach position limits.
        
        Args:
            new_order: Order details including symbol, quantity, price
            
        Returns:
            Dict with 'approved' boolean and 'reason' string
        """
        try:
            # Check if emergency stop is active FIRST
            if self.emergency_stop_active:
                return {
                    'approved': False,
                    'reason': 'Emergency stop is active - all trading halted'
                }
            
            # Load current positions
            positions = await self._load_current_positions()
            
            # Calculate impact of new order
            symbol = new_order['symbol']
            quantity = new_order['quantity']
            price = new_order.get('price', 0)
            order_value = quantity * price
            
            # Find existing position
            existing_position = None
            for pos in positions:
                if pos.symbol == symbol:
                    existing_position = pos
                    break
            
            # Calculate new position size
            if existing_position:
                new_quantity = existing_position.quantity + quantity
                new_value = existing_position.market_value + order_value
            else:
                new_quantity = quantity
                new_value = order_value
            
            # Calculate portfolio impact
            total_portfolio_value = sum(abs(pos.market_value) for pos in positions) + abs(order_value)
            new_position_pct = abs(new_value) / total_portfolio_value * 100 if total_portfolio_value > 0 else 0
            
            # Check position size limit
            if new_position_pct > self.risk_limits.max_position_size_pct:
                return {
                    'approved': False,
                    'reason': f'Position size {new_position_pct:.2f}% would exceed limit of {self.risk_limits.max_position_size_pct}%'
                }
            
            return {
                'approved': True,
                'reason': 'Order approved - within risk limits'
            }
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return {
                'approved': False,
                'reason': f'Error checking limits: {str(e)}'
            }
    
    async def _load_current_positions(self) -> List[Position]:
        """Load current positions from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT symbol, exchange, strategy, agent_name,
                       quantity, avg_cost, market_value, unrealized_pnl, realized_pnl
                FROM positions 
                WHERE quantity != 0
            """)
            
            positions_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            positions = []
            for pos_data in positions_data:
                position = Position(
                    symbol=pos_data['symbol'],
                    exchange=pos_data['exchange'],
                    strategy=pos_data['strategy'],
                    agent_name=pos_data['agent_name'],
                    quantity=float(pos_data['quantity']),
                    avg_cost=float(pos_data['avg_cost']),
                    market_value=float(pos_data['market_value']),
                    unrealized_pnl=float(pos_data['unrealized_pnl']),
                    realized_pnl=float(pos_data['realized_pnl']),
                    weight_pct=0  # Will be calculated when needed
                )
                positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
            return []
    
    def trigger_emergency_stop(self, reason: str) -> bool:
        """
        Trigger emergency stop manually.
        
        Args:
            reason: Reason for emergency stop
            
        Returns:
            bool: True if successful
        """
        try:
            self.emergency_stop_active = True
            logger.critical(f"MANUAL EMERGENCY STOP: {reason}")
            
            # This would trigger the emergency stop workflow
            asyncio.create_task(self._halt_all_trading())
            
            return True
            
        except Exception as e:
            logger.error(f"Error triggering emergency stop: {e}")
            return False
    
    def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is currently active"""
        return self.emergency_stop_active
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop (manual intervention required)"""
        try:
            self.emergency_stop_active = False
            logger.info("Emergency stop reset - trading can resume")
            return True
        except Exception as e:
            logger.error(f"Error resetting emergency stop: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from config.database import get_database_config
    
    async def test_risk_manager():
        """Test the Risk Manager Agent"""
        try:
            # Initialize with test configuration
            db_config = get_database_config()
            risk_limits = RiskLimits(
                max_daily_loss_pct=5.0,  # 5% daily loss limit for testing
                max_position_size_pct=3.0,  # 3% position size limit
                max_leverage=1.5  # 1.5x leverage limit
            )
            
            risk_manager = RiskManagerAgent(db_config, risk_limits)
            
            # Test portfolio risk monitoring
            print("Testing portfolio risk monitoring...")
            risk_metrics = await risk_manager.monitor_portfolio_risk()
            
            print(f"Portfolio Value: ${risk_metrics.portfolio_value:,.2f}")
            print(f"Leverage: {risk_metrics.leverage:.2f}x")
            print(f"VaR 95% (1d): ${risk_metrics.var_1d_95:,.2f}")
            print(f"Max Position: {risk_metrics.max_position_pct:.2f}%")
            
            # Test position limit check
            print("\nTesting position limit check...")
            test_order = {
                'symbol': 'AAPL',
                'quantity': 1000,
                'price': 150.0
            }
            
            limit_check = await risk_manager.check_position_limits(test_order)
            print(f"Order approved: {limit_check['approved']}")
            print(f"Reason: {limit_check['reason']}")
            
            print("Risk Manager Agent test completed successfully!")
            
        except Exception as e:
            print(f"Error in risk manager test: {e}")
    
    # Run the test
    asyncio.run(test_risk_manager())