"""
Bloomberg Terminal API - Main FastAPI Application
High-performance backend for the Bloomberg Terminal interface.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from core.config import get_settings
from core.database import init_database, get_database_health
from core.redis_manager import get_redis_manager
from services.market_data_service import MarketDataService
from services.order_service import OrderService
from services.portfolio_service import PortfolioService
from services.risk_service import RiskService
from services.websocket_manager import WebSocketManager
from services.orchestration_service import OrchestrationService
from services.risk_monitoring_service import RiskMonitoringService
from api.routes import market_data, orders, portfolio, risk, system, agents, risk_management, backtesting, training, monitoring

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

settings = get_settings()
logger = logging.getLogger(__name__)

# Global services
market_data_service: MarketDataService = None
order_service: OrderService = None
portfolio_service: PortfolioService = None
risk_service: RiskService = None
websocket_manager: WebSocketManager = None
orchestration_service: OrchestrationService = None
risk_monitoring_service: RiskMonitoringService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Bloomberg Terminal API...")
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized")
        
        # Initialize Redis
        redis_manager = get_redis_manager()
        await redis_manager.initialize()
        logger.info("Redis initialized")
        
        # Initialize services
        global market_data_service, order_service, portfolio_service, risk_service, websocket_manager, orchestration_service, risk_monitoring_service
        
        # Initialize orchestration service (this will initialize all other components)
        orchestration_service = OrchestrationService()
        await orchestration_service.initialize()
        await orchestration_service.start()
        
        # Get references to individual services from orchestration
        market_data_service = orchestration_service.market_data_service
        order_service = orchestration_service.order_service
        
        # Initialize risk monitoring service
        risk_monitoring_service = RiskMonitoringService()
        await risk_monitoring_service.initialize()
        await risk_monitoring_service.start()
        
        # Initialize legacy services for API compatibility
        portfolio_service = PortfolioService()
        risk_service = RiskService()
        websocket_manager = WebSocketManager()
        
        # Start additional background tasks
        asyncio.create_task(risk_service.start_monitoring())
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Bloomberg Terminal API...")
    
    # Stop orchestration service (this will stop all managed components)
    if orchestration_service:
        await orchestration_service.stop()
    
    # Stop risk monitoring service
    if risk_monitoring_service:
        await risk_monitoring_service.stop()
    
    # Stop remaining services
    if websocket_manager:
        await websocket_manager.shutdown()
    
    # Close Redis connections
    redis_manager = get_redis_manager()
    await redis_manager.close()
    
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Bloomberg Terminal API",
    description="High-performance trading system backend",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect request metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    # Add latency header
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    
    return response


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Global error handling."""
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled error in {request.url.path}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e) if settings.debug else None}
        )


# Health check endpoints
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """System health check."""
    try:
        # Check database
        db_health = await get_database_health()
        
        # Check Redis
        redis_manager = get_redis_manager()
        redis_health = await redis_manager.health_check()
        
        # Check services
        services_health = {
            "market_data": market_data_service.is_healthy() if market_data_service else False,
            "orders": order_service.is_healthy() if order_service else False,
            "portfolio": portfolio_service.is_healthy() if portfolio_service else False,
            "risk": risk_service.is_healthy() if risk_service else False
        }
        
        overall_health = all([
            db_health,
            redis_health,
            all(services_health.values())
        ])
        
        return {
            "status": "healthy" if overall_health else "unhealthy",
            "timestamp": time.time(),
            "services": {
                "database": db_health,
                "redis": redis_health,
                **services_health
            },
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Bloomberg Terminal API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


# WebSocket endpoint for real-time data
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket connection for real-time data streaming."""
    await websocket_manager.connect(client_id, websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Handle client commands
            if data.startswith("SUBSCRIBE:"):
                symbols = data.replace("SUBSCRIBE:", "").split(",")
                await websocket_manager.subscribe_to_symbols(client_id, symbols)
            elif data.startswith("UNSUBSCRIBE:"):
                symbols = data.replace("UNSUBSCRIBE:", "").split(",")
                await websocket_manager.unsubscribe_from_symbols(client_id, symbols)
            elif data == "PING":
                await websocket.send_text("PONG")
                
    except WebSocketDisconnect:
        await websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await websocket_manager.disconnect(client_id)


# Emergency stop endpoint
@app.post("/emergency/stop")
async def emergency_stop():
    """Emergency stop all trading operations."""
    logger.critical("EMERGENCY STOP TRIGGERED")
    
    try:
        # Cancel all pending orders
        if order_service:
            await order_service.cancel_all_orders()
        
        # Flatten all positions
        if portfolio_service:
            await portfolio_service.flatten_all_positions()
        
        # Stop market data streaming
        if market_data_service:
            await market_data_service.stop_streaming()
        
        # Stop risk monitoring
        if risk_service:
            await risk_service.stop_monitoring()
        
        return {"status": "EMERGENCY_STOP_EXECUTED", "timestamp": time.time()}
        
    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")
        raise HTTPException(status_code=500, detail=f"Emergency stop failed: {e}")


# Include API routes
app.include_router(market_data.router, prefix="/api/v1/market-data", tags=["market-data"])
app.include_router(orders.router, prefix="/api/v1/orders", tags=["orders"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["risk"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(risk_management.router, prefix="/api/v1/risk-management", tags=["risk-management"])
app.include_router(backtesting.router, prefix="/api/v1/backtesting", tags=["backtesting"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["monitoring"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
        access_log=settings.debug,
        log_level="info" if settings.debug else "warning"
    )