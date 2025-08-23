"""
Agent management API routes for Bloomberg Terminal.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class AgentStatusResponse(BaseModel):
    """Agent status response model."""
    agent_name: str
    status: str
    last_updated: Optional[str]
    performance_metrics: Dict[str, Any]


class SignalResponse(BaseModel):
    """Trading signal response model."""
    id: str
    agent_name: str
    symbol: str
    signal_type: str
    confidence: float
    strength: float
    timestamp: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    prediction_horizon: int
    risk_score: float
    expected_return: float


class SystemStatusResponse(BaseModel):
    """System status response model."""
    is_running: bool
    startup_time: Optional[str]
    uptime_seconds: float
    environment: str
    symbols_count: int
    system_metrics: Dict[str, Any]
    event_bus: Optional[Dict[str, Any]]
    agents: Optional[Dict[str, Any]]
    signals: Optional[Dict[str, Any]]


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status."""
    try:
        # Import here to avoid circular imports
        from api.main import orchestration_service
        
        if not orchestration_service:
            raise HTTPException(status_code=503, detail="Orchestration service not available")
        
        status = await orchestration_service.get_system_status()
        return SystemStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/status")
async def get_agents_status() -> Dict[str, AgentStatusResponse]:
    """Get status of all trading agents."""
    try:
        from api.main import orchestration_service
        
        if not orchestration_service or not orchestration_service.agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        summary = await orchestration_service.agent_orchestrator.get_agent_performance_summary()
        
        agents_status = {}
        for agent_name in summary.get('agent_status', {}):
            agent_status = summary['agent_status'][agent_name]
            agent_perf = summary.get('agent_performance', {}).get(agent_name, {})
            
            agents_status[agent_name] = AgentStatusResponse(
                agent_name=agent_name,
                status=agent_status.get('status', 'unknown'),
                last_updated=agent_status.get('last_updated'),
                performance_metrics=agent_perf
            )
        
        return agents_status
        
    except Exception as e:
        logger.error(f"Error getting agents status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_name}/performance")
async def get_agent_performance(agent_name: str) -> Dict[str, Any]:
    """Get performance metrics for a specific agent."""
    try:
        from api.main import orchestration_service
        
        if not orchestration_service or not orchestration_service.agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        if agent_name not in orchestration_service.agent_orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        summary = await orchestration_service.agent_orchestrator.get_agent_performance_summary()
        agent_perf = summary.get('agent_performance', {}).get(agent_name, {})
        
        if not agent_perf:
            raise HTTPException(status_code=404, detail=f"Performance data for {agent_name} not found")
        
        return {
            'agent_name': agent_name,
            'weight': orchestration_service.agent_orchestrator.agent_weights.get(agent_name, 0.0),
            'performance': agent_perf
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/active")
async def get_active_signals(
    symbol: Optional[str] = Query(None, description="Filter by symbol")
) -> Dict[str, Any]:
    """Get active trading signals."""
    try:
        from api.main import orchestration_service
        
        if not orchestration_service:
            raise HTTPException(status_code=503, detail="Orchestration service not available")
        
        signals_data = await orchestration_service.get_active_signals(symbol)
        
        if 'error' in signals_data:
            raise HTTPException(status_code=500, detail=signals_data['error'])
        
        return signals_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting active signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/{symbol}")
async def get_signals_for_symbol(symbol: str) -> List[SignalResponse]:
    """Get active signals for a specific symbol."""
    try:
        from api.main import orchestration_service
        
        if not orchestration_service or not orchestration_service.signal_coordinator:
            raise HTTPException(status_code=503, detail="Signal coordinator not available")
        
        signals = await orchestration_service.signal_coordinator.get_active_signals_for_symbol(symbol)
        
        response_signals = []
        for signal in signals:
            response_signals.append(SignalResponse(
                id=signal.id,
                agent_name=signal.agent_name,
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                confidence=signal.confidence,
                strength=signal.strength,
                timestamp=signal.timestamp.isoformat(),
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                prediction_horizon=signal.prediction_horizon,
                risk_score=signal.risk_score,
                expected_return=signal.expected_return
            ))
        
        return response_signals
        
    except Exception as e:
        logger.error(f"Error getting signals for symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals/generate")
async def generate_signals() -> Dict[str, Any]:
    """Manually trigger signal generation."""
    try:
        from api.main import orchestration_service
        
        if not orchestration_service or not orchestration_service.signal_coordinator:
            raise HTTPException(status_code=503, detail="Signal coordinator not available")
        
        results = await orchestration_service.signal_coordinator.generate_and_publish_signals()
        
        return {
            'status': 'success',
            'results': results,
            'timestamp': orchestration_service.startup_time.isoformat() if orchestration_service.startup_time else None
        }
        
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_orchestration_metrics() -> Dict[str, Any]:
    """Get orchestration and agent metrics."""
    try:
        from api.main import orchestration_service
        
        if not orchestration_service:
            raise HTTPException(status_code=503, detail="Orchestration service not available")
        
        # Get system status (includes metrics)
        status = await orchestration_service.get_system_status()
        
        # Get signal coordinator metrics
        signal_metrics = {}
        if orchestration_service.signal_coordinator:
            signal_metrics = await orchestration_service.signal_coordinator.get_performance_metrics()
        
        # Get event bus metrics
        event_metrics = {}
        if orchestration_service.event_bus:
            event_metrics = await orchestration_service.event_bus.get_metrics()
        
        return {
            'system_metrics': status.get('system_metrics', {}),
            'signal_metrics': signal_metrics,
            'event_metrics': event_metrics,
            'agent_metrics': status.get('agents', {}),
            'timestamp': status.get('startup_time')
        }
        
    except Exception as e:
        logger.error(f"Error getting orchestration metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency-stop")
async def emergency_stop() -> Dict[str, str]:
    """Trigger emergency stop of all trading activities."""
    try:
        from api.main import orchestration_service
        
        if not orchestration_service or not orchestration_service.event_bus:
            raise HTTPException(status_code=503, detail="Orchestration service not available")
        
        # Publish emergency stop event
        emergency_data = {
            'trigger': 'manual_api_call',
            'timestamp': orchestration_service.startup_time.isoformat() if orchestration_service.startup_time else None,
            'message': 'Emergency stop triggered via API'
        }
        
        from events.event_bus import EventType, Event
        from datetime import datetime, timezone
        import uuid
        
        event = Event(
            id=str(uuid.uuid4()),
            event_type=EventType.EMERGENCY_STOP,
            timestamp=datetime.now(timezone.utc),
            source='API',
            data=emergency_data,
            priority=2  # Critical priority
        )
        
        success = await orchestration_service.event_bus.publish(event)
        
        if success:
            return {'status': 'emergency_stop_triggered', 'message': 'Emergency stop event published successfully'}
        else:
            raise HTTPException(status_code=500, detail="Failed to publish emergency stop event")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for orchestration system."""
    try:
        from api.main import orchestration_service
        
        if not orchestration_service:
            return {'status': 'unhealthy', 'error': 'Orchestration service not available'}
        
        if not orchestration_service.is_running:
            return {'status': 'unhealthy', 'error': 'Orchestration service not running'}
        
        # Get basic health metrics
        health_data = {
            'status': 'healthy',
            'is_running': orchestration_service.is_running,
            'uptime_seconds': orchestration_service.system_metrics['uptime_seconds'],
            'components': {}
        }
        
        # Check component health
        if orchestration_service.event_bus and orchestration_service.event_bus.is_running:
            health_data['components']['event_bus'] = 'healthy'
        else:
            health_data['components']['event_bus'] = 'unhealthy'
        
        if orchestration_service.agent_orchestrator and orchestration_service.agent_orchestrator.is_running:
            health_data['components']['agents'] = 'healthy'
        else:
            health_data['components']['agents'] = 'unhealthy'
        
        if orchestration_service.signal_coordinator and orchestration_service.signal_coordinator.is_running:
            health_data['components']['signals'] = 'healthy'
        else:
            health_data['components']['signals'] = 'unhealthy'
        
        # Overall health status
        unhealthy_components = [k for k, v in health_data['components'].items() if v != 'healthy']
        if unhealthy_components:
            health_data['status'] = 'degraded'
            health_data['unhealthy_components'] = unhealthy_components
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }