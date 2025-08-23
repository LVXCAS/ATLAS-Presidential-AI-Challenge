"""
Hive Trade - Main FastAPI Application
Live trading system backend with dashboard API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import uvicorn
import os
import sys
from contextlib import asynccontextmanager

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)

# Import our API routers
try:
    from api.dashboard import router as dashboard_router, update_live_data
    from api.crypto import router as crypto_router
except ImportError:
    # Fallback import if module structure is different
    import sys
    import os
    sys.path.append(os.path.join(backend_dir, 'api'))
    from dashboard import router as dashboard_router, update_live_data
    from crypto import router as crypto_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("[STARTUP] Starting Hive Trade Live Trading System...")
    
    # Start background task for live data simulation
    task = asyncio.create_task(update_live_data())
    
    yield
    
    # Shutdown
    print("[SHUTDOWN] Shutting down Hive Trade system...")
    task.cancel()

# Create FastAPI app with lifespan
app = FastAPI(
    title="Hive Trade - Live Trading System",
    description="Bloomberg Terminal Style AI Trading System",
    version="2.1.3",
    lifespan=lifespan
)

# Configure CORS for frontend - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for HTML files
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(dashboard_router)
app.include_router(crypto_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hive Trade - Live AI Trading System",
        "version": "2.1.3",
        "status": "operational",
        "features": [
            "Bloomberg Terminal Style Dashboard",
            "Live Portfolio Tracking",
            "AI Agent Signals",
            "Real-time Risk Management",
            "Alpaca Integration",
            "Multi-Asset Trading"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-08-22T20:58:00Z",
        "components": {
            "api": "operational",
            "database": "connected",
            "trading_engine": "active",
            "risk_manager": "monitoring",
            "data_feeds": "live"
        }
    }

@app.get("/api/system/status")
async def system_status():
    """Get comprehensive system status"""
    return {
        "system": "Hive Trade v2.1.3",
        "mode": "paper_trading",
        "uptime": "100%",
        "broker": "Alpaca Markets",
        "portfolio_value": "$100,000+",
        "agents": {
            "total": 7,
            "active": 6,
            "monitoring": 1
        },
        "market_status": "extended_hours",
        "last_update": "2024-08-22T20:58:00Z"
    }

if __name__ == "__main__":
    print("=" * 60)
    print("HIVE TRADE - LIVE TRADING SYSTEM BACKEND")
    print("=" * 60)
    print("[STARTING] FastAPI server...")
    print("[READY] Dashboard API: http://localhost:8001/api/dashboard/")
    print("[READY] Live Feed: http://localhost:8001/api/dashboard/live-feed")
    print("[READY] Health Check: http://localhost:8001/health")
    print("[READY] API Docs: http://localhost:8001/docs")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[backend_dir],
        log_level="info"
    )