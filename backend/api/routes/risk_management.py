"""
Risk Management API routes for Bloomberg Terminal.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class RiskMetricsResponse(BaseModel):
    """Risk metrics response model."""
    portfolio_value: float
    daily_var: float
    max_drawdown: float
    leverage: float
    risk_score: float
    var_utilization: float
    leverage_utilization: float
    emergency_mode: bool
    circuit_breaker_active: bool
    last_updated: Optional[str]


class PositionRiskResponse(BaseModel):
    """Position risk response model."""
    symbol: str
    market_value: float
    weight: float
    daily_var: float
    beta: float
    volatility: float
    liquidity_score: float
    sector: str
    concentration_risk: float


class RiskAlertResponse(BaseModel):
    """Risk alert response model."""
    id: str
    risk_type: str
    risk_level: str
    symbol: Optional[str]
    message: str
    current_value: float
    threshold: float
    timestamp: str
    action_required: str


class PortfolioSummaryResponse(BaseModel):
    """Portfolio summary response model."""
    total_positions: int
    total_value: float
    total_pnl: float
    unrealized_pnl: float
    daily_pnl: float
    positions_by_sector: Dict[str, Any]
    top_positions: List[Dict[str, Any]]
    worst_positions: List[Dict[str, Any]]


class PositionDetailsResponse(BaseModel):
    """Position details response model."""
    id: str
    symbol: str
    position_type: str
    status: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    pnl_pct: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    holding_period_hours: float
    sector: str
    trades_count: int


class TradeRiskCheckRequest(BaseModel):
    """Trade risk check request model."""
    symbol: str
    signal_type: str
    quantity: float
    price: float
    confidence: float = Field(ge=0, le=1)


class TradeRiskCheckResponse(BaseModel):
    """Trade risk check response model."""
    approved: bool
    warnings: List[str]
    recommended_position_size: float
    risk_score: float
    max_position_size: float


class EmergencyStopRequest(BaseModel):
    """Emergency stop request model."""
    reason: str
    force_close_positions: bool = True


@router.get("/portfolio/risk-metrics", response_model=RiskMetricsResponse)
async def get_portfolio_risk_metrics():
    """Get current portfolio risk metrics."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        risk_metrics = await risk_monitoring_service.risk_engine.get_portfolio_risk_metrics()
        
        return RiskMetricsResponse(**risk_metrics)
        
    except Exception as e:
        logger.error(f"Error getting portfolio risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/positions", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary():
    """Get portfolio summary with risk information."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        summary = await risk_monitoring_service.position_manager.get_portfolio_summary()
        
        return PortfolioSummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions/{position_id}", response_model=PositionDetailsResponse)
async def get_position_details(position_id: str):
    """Get detailed information about a specific position."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        position_details = await risk_monitoring_service.position_manager.get_position_details(position_id)
        
        if not position_details:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
        
        return PositionDetailsResponse(**position_details)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions/by-symbol/{symbol}")
async def get_positions_by_symbol(symbol: str) -> List[PositionDetailsResponse]:
    """Get all positions for a specific symbol."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        positions = await risk_monitoring_service.position_manager.get_positions_by_symbol(symbol)
        
        return [PositionDetailsResponse(**pos) for pos in positions]
        
    except Exception as e:
        logger.error(f"Error getting positions by symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/position-risks")
async def get_position_risks() -> Dict[str, PositionRiskResponse]:
    """Get risk metrics for all positions."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        position_risks = await risk_monitoring_service.risk_engine.get_position_risks()
        
        return {
            symbol: PositionRiskResponse(**risk_data)
            for symbol, risk_data in position_risks.items()
        }
        
    except Exception as e:
        logger.error(f"Error getting position risks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[RiskAlertResponse])
async def get_active_risk_alerts():
    """Get all active risk alerts."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        alerts = await risk_monitoring_service.risk_engine.get_active_alerts()
        
        return [RiskAlertResponse(**alert) for alert in alerts]
        
    except Exception as e:
        logger.error(f"Error getting risk alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trade/risk-check", response_model=TradeRiskCheckResponse)
async def check_trade_risk(request: TradeRiskCheckRequest):
    """Check risk for a proposed trade."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        from agents.base_agent import TradingSignal, SignalType
        from datetime import datetime, timezone
        import uuid
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        # Create mock trading signal
        signal_type_map = {
            'buy': SignalType.BUY,
            'sell': SignalType.SELL,
            'strong_buy': SignalType.STRONG_BUY,
            'strong_sell': SignalType.STRONG_SELL,
            'hold': SignalType.HOLD
        }
        
        signal = TradingSignal(
            id=str(uuid.uuid4()),
            agent_name="RiskAPI",
            symbol=request.symbol,
            timestamp=datetime.now(timezone.utc),
            signal_type=signal_type_map.get(request.signal_type.lower(), SignalType.BUY),
            confidence=request.confidence,
            strength=0.5,
            reasoning={},
            features_used={},
            prediction_horizon=60,
            target_price=request.price * 1.05,
            stop_loss=request.price * 0.95,
            risk_score=0.5,
            expected_return=0.05
        )
        
        # Perform risk check
        approved, warnings, position_size = await risk_monitoring_service.pre_trade_risk_check(signal)
        
        # Calculate additional metrics
        portfolio_metrics = await risk_monitoring_service.risk_engine.get_portfolio_risk_metrics()
        max_position = portfolio_metrics.get('portfolio_value', 1000000) * 0.1  # 10% max
        
        return TradeRiskCheckResponse(
            approved=approved,
            warnings=warnings,
            recommended_position_size=position_size,
            risk_score=min(len(warnings) / 5.0, 1.0),
            max_position_size=max_position
        )
        
    except Exception as e:
        logger.error(f"Error checking trade risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_risk_dashboard() -> Dict[str, Any]:
    """Get comprehensive risk dashboard data."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        dashboard = await risk_monitoring_service.get_risk_dashboard()
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Error getting risk dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency-stop")
async def trigger_emergency_stop(request: EmergencyStopRequest) -> Dict[str, Any]:
    """Trigger emergency stop procedures."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        results = await risk_monitoring_service.trigger_emergency_procedures(request.reason)
        
        return results
        
    except Exception as e:
        logger.error(f"Error triggering emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/positions/{position_id}/close")
async def close_position(position_id: str, close_price: Optional[float] = None) -> Dict[str, Any]:
    """Close a specific position."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        success = await risk_monitoring_service.position_manager.close_position(position_id, close_price)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to close position {position_id}")
        
        return {
            'status': 'success',
            'message': f'Position {position_id} closed successfully',
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance/report")
async def get_compliance_report() -> Dict[str, Any]:
    """Get latest compliance report."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        report = await risk_monitoring_service.generate_compliance_report()
        
        return report
        
    except Exception as e:
        logger.error(f"Error getting compliance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stress-test")
async def run_stress_test(
    scenario: Optional[str] = Query(None, description="Stress test scenario"),
    shock_pct: Optional[float] = Query(-0.1, description="Market shock percentage")
) -> Dict[str, Any]:
    """Run portfolio stress test."""
    try:
        # Mock stress test results
        portfolio_value = 1000000  # Would get actual value
        stressed_value = portfolio_value * (1 + shock_pct)
        loss = portfolio_value - stressed_value
        
        return {
            'scenario': scenario or 'Market Crash',
            'shock_percentage': shock_pct * 100,
            'current_portfolio_value': portfolio_value,
            'stressed_portfolio_value': stressed_value,
            'potential_loss': loss,
            'loss_percentage': (loss / portfolio_value) * 100,
            'var_impact': loss * 1.5,  # Simplified VaR impact
            'positions_at_risk': 12,  # Would calculate actual positions at risk
            'recommendations': [
                'Reduce technology sector exposure',
                'Increase hedge positions',
                'Monitor liquidity closely'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/limits")
async def get_risk_limits() -> Dict[str, Any]:
    """Get current risk limits and utilization."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        if not risk_monitoring_service:
            raise HTTPException(status_code=503, detail="Risk monitoring service not available")
        
        risk_metrics = await risk_monitoring_service.risk_engine.get_portfolio_risk_metrics()
        config = risk_monitoring_service.risk_engine.config
        
        return {
            'portfolio_limits': {
                'max_var': {
                    'limit': config['max_portfolio_var'],
                    'current': risk_metrics.get('daily_var', 0),
                    'utilization': risk_metrics.get('var_utilization', 0),
                    'status': 'OK' if risk_metrics.get('var_utilization', 0) < 0.8 else 'WARNING'
                },
                'max_leverage': {
                    'limit': config['max_portfolio_leverage'],
                    'current': risk_metrics.get('leverage', 0),
                    'utilization': risk_metrics.get('leverage_utilization', 0),
                    'status': 'OK' if risk_metrics.get('leverage_utilization', 0) < 0.8 else 'WARNING'
                },
                'max_drawdown': {
                    'limit': config['max_drawdown'],
                    'current': risk_metrics.get('max_drawdown', 0),
                    'utilization': risk_metrics.get('drawdown_utilization', 0),
                    'status': 'OK' if risk_metrics.get('drawdown_utilization', 0) < 0.8 else 'WARNING'
                }
            },
            'position_limits': {
                'max_position_weight': config['max_position_weight'],
                'max_sector_weight': config['max_sector_weight'],
                'min_liquidity_score': config['min_liquidity_score']
            },
            'risk_controls': {
                'emergency_mode': risk_metrics.get('emergency_mode', False),
                'circuit_breaker': risk_metrics.get('circuit_breaker_active', False),
                'auto_hedge_enabled': True,
                'position_timeout_hours': 48
            },
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def risk_health_check() -> Dict[str, Any]:
    """Health check for risk management system."""
    try:
        from services.risk_monitoring_service import risk_monitoring_service
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        if not risk_monitoring_service:
            health_data['status'] = 'unhealthy'
            health_data['error'] = 'Risk monitoring service not available'
            return health_data
        
        # Check service status
        if risk_monitoring_service.is_running:
            health_data['components']['monitoring_service'] = 'healthy'
        else:
            health_data['components']['monitoring_service'] = 'unhealthy'
            health_data['status'] = 'degraded'
        
        # Check risk engine
        if risk_monitoring_service.risk_engine.is_running:
            health_data['components']['risk_engine'] = 'healthy'
        else:
            health_data['components']['risk_engine'] = 'unhealthy'
            health_data['status'] = 'degraded'
        
        # Check position manager
        if risk_monitoring_service.position_manager.is_running:
            health_data['components']['position_manager'] = 'healthy'
        else:
            health_data['components']['position_manager'] = 'unhealthy'
            health_data['status'] = 'degraded'
        
        # Add service metrics
        health_data['metrics'] = risk_monitoring_service.service_metrics
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error in risk health check: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }