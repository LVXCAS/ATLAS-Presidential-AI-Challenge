"""
Trading API Endpoints
Provides access to live trading operations, profit/loss monitoring, and quantitative analysis
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import asyncio
import json
import sys
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import trading components
try:
    from profit_target_monitor import ProfitTargetMonitor
    from agents.quantitative_finance_engine import quantitative_engine, OptionParameters
    from agents.quant_integration import analyze_option, analyze_portfolio, predict_returns
    from OPTIONS_BOT import TomorrowReadyOptionsBot
    from start_real_market_hunter import RealMarketDataHunter
except ImportError as e:
    print(f"Trading components not available: {e}")
    ProfitTargetMonitor = None
    quantitative_engine = None
    OptionParameters = None

router = APIRouter(prefix="/api/trading", tags=["trading"])

# Pydantic models for API requests
class OptionAnalysisRequest(BaseModel):
    symbol: str
    strike_price: float
    expiry_date: str
    option_type: str = "call"

class PortfolioAnalysisRequest(BaseModel):
    positions: List[Dict[str, Any]]

class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"

# Global trading instances
profit_monitor = None
options_bot = None
market_hunter = None

async def initialize_trading_systems():
    """Initialize trading systems"""
    global profit_monitor, options_bot, market_hunter

    try:
        if ProfitTargetMonitor:
            profit_monitor = ProfitTargetMonitor()
            await profit_monitor.initialize_broker()

        if TomorrowReadyOptionsBot:
            options_bot = TomorrowReadyOptionsBot()

        if RealMarketDataHunter:
            market_hunter = RealMarketDataHunter()

    except Exception as e:
        print(f"Error initializing trading systems: {e}")

@router.get("/status")
async def get_trading_status():
    """Get current trading system status"""
    global profit_monitor

    if not profit_monitor:
        await initialize_trading_systems()

    if profit_monitor:
        try:
            status = profit_monitor.get_status()
            return {
                "system_status": "operational",
                "profit_monitoring": {
                    "active": status["monitoring_active"],
                    "profit_target": f"{profit_monitor.profit_target_pct}%",
                    "loss_limit": f"{profit_monitor.loss_limit_pct}%",
                    "target_hit": status["target_hit"],
                    "loss_limit_hit": status["loss_limit_hit"],
                    "daily_profit_pct": status["daily_profit_pct"]
                },
                "equity": {
                    "initial": status.get("initial_equity", 0),
                    "current": status.get("current_equity", 0)
                },
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    else:
        return {
            "system_status": "not_initialized",
            "message": "Trading systems not available",
            "timestamp": datetime.now().isoformat()
        }

@router.get("/profit-loss")
async def get_profit_loss_metrics():
    """Get detailed profit/loss metrics"""
    global profit_monitor

    if not profit_monitor:
        await initialize_trading_systems()

    if profit_monitor:
        try:
            current_equity, profit_pct, profit_target_hit, loss_limit_hit = await profit_monitor.check_daily_profit()
            status = profit_monitor.get_status()

            return {
                "daily_metrics": {
                    "current_equity": current_equity,
                    "daily_profit_pct": profit_pct,
                    "profit_amount": current_equity - status.get("initial_equity", 0) if status.get("initial_equity") else 0,
                    "profit_target_hit": profit_target_hit,
                    "loss_limit_hit": loss_limit_hit
                },
                "limits": {
                    "profit_target_pct": profit_monitor.profit_target_pct,
                    "loss_limit_pct": profit_monitor.loss_limit_pct
                },
                "status": {
                    "monitoring_active": status["monitoring_active"],
                    "should_sell_all": profit_target_hit or loss_limit_hit
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting P&L metrics: {e}")
    else:
        raise HTTPException(status_code=503, detail="Profit/loss monitoring not available")

@router.post("/analyze-option")
async def analyze_option_endpoint(request: OptionAnalysisRequest):
    """Analyze an options contract using quantitative methods"""
    if not quantitative_engine:
        raise HTTPException(status_code=503, detail="Quantitative engine not available")

    try:
        analysis = analyze_option(
            request.symbol,
            request.strike_price,
            request.expiry_date,
            request.option_type
        )

        return {
            "symbol": request.symbol,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing option: {e}")

@router.post("/analyze-portfolio")
async def analyze_portfolio_endpoint(request: PortfolioAnalysisRequest):
    """Analyze portfolio risk and performance"""
    if not quantitative_engine:
        raise HTTPException(status_code=503, detail="Quantitative engine not available")

    try:
        analysis = analyze_portfolio(request.positions)

        return {
            "portfolio_analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing portfolio: {e}")

@router.post("/predict-returns")
async def predict_returns_endpoint(request: PredictionRequest):
    """Get machine learning predictions for returns"""
    try:
        prediction = predict_returns(request.symbol)

        return {
            "symbol": request.symbol,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting returns: {e}")

@router.get("/opportunities")
async def get_trading_opportunities():
    """Get current trading opportunities from both bots"""
    opportunities = []

    try:
        # Get opportunities from OPTIONS_BOT
        if options_bot:
            # This would require modifying the bot to expose opportunities
            opportunities.append({
                "source": "OPTIONS_BOT",
                "status": "active",
                "message": "Scanning for high-quality options opportunities"
            })

        # Get opportunities from Market Hunter
        if market_hunter:
            opportunities.append({
                "source": "RealMarketDataHunter",
                "status": "active",
                "message": "Hunting for market inefficiencies"
            })

        return {
            "opportunities": opportunities,
            "count": len(opportunities),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting opportunities: {e}")

@router.post("/emergency-sell")
async def emergency_sell_all():
    """Emergency sell all positions"""
    global profit_monitor

    if not profit_monitor:
        await initialize_trading_systems()

    if profit_monitor:
        try:
            result = await profit_monitor.sell_all_positions("Emergency sell via API")
            return {
                "action": "emergency_sell",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error executing emergency sell: {e}")
    else:
        raise HTTPException(status_code=503, detail="Profit monitor not available")

@router.get("/system-health")
async def get_system_health():
    """Get comprehensive system health metrics"""
    health = {
        "components": {},
        "overall_status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

    # Check profit monitor
    health["components"]["profit_monitor"] = {
        "status": "available" if profit_monitor else "unavailable",
        "features": ["5.75% profit target", "4.9% loss limit", "real-time monitoring"]
    }

    # Check quantitative engine
    health["components"]["quantitative_engine"] = {
        "status": "available" if quantitative_engine else "unavailable",
        "features": ["Black-Scholes pricing", "Monte Carlo simulations", "Greeks calculation"]
    }

    # Check trading bots
    health["components"]["options_bot"] = {
        "status": "available" if options_bot else "unavailable",
        "features": ["options trading", "quantitative analysis", "ML predictions"]
    }

    health["components"]["market_hunter"] = {
        "status": "available" if market_hunter else "unavailable",
        "features": ["market scanning", "opportunity detection", "advanced strategies"]
    }

    # Calculate overall health
    available_components = sum(1 for comp in health["components"].values() if comp["status"] == "available")
    total_components = len(health["components"])
    health_percentage = (available_components / total_components) * 100

    if health_percentage >= 75:
        health["overall_status"] = "healthy"
    elif health_percentage >= 50:
        health["overall_status"] = "degraded"
    else:
        health["overall_status"] = "unhealthy"

    health["health_percentage"] = health_percentage

    return health

# Initialize on import
asyncio.create_task(initialize_trading_systems())