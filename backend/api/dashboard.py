"""
Live Trading Dashboard API Endpoints
Provides real-time data for the Bloomberg Terminal-style dashboard
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import asyncio
import json
import random
from typing import List, Dict, Any
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Global state for live data
live_data = {
    "portfolio_value": 100000.0,
    "total_pnl": 0.0,
    "day_pnl": 0.0,
    "positions": [],
    "performance_history": [],
    "agent_signals": [],
    "market_data": [],
    "system_stats": {},
    "risk_metrics": {},
    "last_update": datetime.now()
}

def simulate_live_data():
    """Simulate live trading data"""
    global live_data
    
    # Update portfolio with realistic changes
    pnl_change = random.uniform(-100, 200)  # Bias towards positive
    live_data["total_pnl"] += pnl_change
    live_data["day_pnl"] = live_data["total_pnl"] * 0.4  # Day P&L as portion
    live_data["portfolio_value"] = 100000 + live_data["total_pnl"]
    
    # Update performance history
    now = datetime.now()
    time_str = now.strftime("%H:%M")
    
    live_data["performance_history"].append({
        "time": time_str,
        "portfolio": live_data["portfolio_value"],
        "benchmark": 100000 + (live_data["total_pnl"] * 0.6),  # Benchmark performance
        "volume": random.uniform(1.0, 3.0)
    })
    
    # Keep only last 50 data points
    if len(live_data["performance_history"]) > 50:
        live_data["performance_history"] = live_data["performance_history"][-50:]
    
    # Update positions with live P&L
    live_data["positions"] = [
        {
            "symbol": "AAPL",
            "qty": 1250,
            "price": round(178.25 + random.uniform(-2, 2), 2),
            "mktVal": 222812,
            "pnl": round(8745 + random.uniform(-1000, 2000), 2),
            "pnlPct": round(4.08 + random.uniform(-2, 3), 2),
            "side": "LONG"
        },
        {
            "symbol": "MSFT",
            "qty": 850,
            "price": round(342.18 + random.uniform(-3, 3), 2),
            "mktVal": 290853,
            "pnl": round(-2134 + random.uniform(-1000, 3000), 2),
            "pnlPct": round(-0.73 + random.uniform(-1.5, 2.5), 2),
            "side": "LONG"
        },
        {
            "symbol": "GOOGL",
            "qty": 450,
            "price": round(128.76 + random.uniform(-1.5, 1.5), 2),
            "mktVal": 57942,
            "pnl": round(1876 + random.uniform(-500, 1000), 2),
            "pnlPct": round(3.34 + random.uniform(-1, 2), 2),
            "side": "LONG"
        },
        {
            "symbol": "TSLA",
            "qty": -200,
            "price": round(245.80 + random.uniform(-5, 5), 2),
            "mktVal": -49160,
            "pnl": round(3421 + random.uniform(-800, 1500), 2),
            "pnlPct": round(7.48 + random.uniform(-3, 4), 2),
            "side": "SHORT"
        },
        {
            "symbol": "SPY",
            "qty": 2100,
            "price": round(432.50 + random.uniform(-2, 2), 2),
            "mktVal": 908250,
            "pnl": round(12456 + random.uniform(-2000, 3000), 2),
            "pnlPct": round(1.39 + random.uniform(-1, 2), 2),
            "side": "LONG"
        },
        {
            "symbol": "NVDA",
            "qty": 63,
            "price": round(450.72 + random.uniform(-8, 8), 2),
            "mktVal": round(63 * 450.72),
            "pnl": round(random.uniform(-500, 1200), 2),
            "pnlPct": round(random.uniform(-2, 4), 2),
            "side": "LONG"
        }
    ]
    
    # Update agent signals
    agent_names = ["MOMENTUM_01", "SENTIMENT_02", "MEAN_REV_03", "NEWS_NLP_04", "RISK_MGR_05", "ARBIT_06"]
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ", "NVDA", "AMZN", "META"]
    signals = ["BUY", "SELL", "HOLD"]
    
    live_data["agent_signals"] = []
    for agent in agent_names:
        signal = random.choice(signals)
        symbol = random.choice(symbols)
        
        live_data["agent_signals"].append({
            "agent": agent,
            "signal": signal,
            "strength": round(random.uniform(0.6, 1.0), 2),
            "symbol": symbol,
            "price": round(random.uniform(100, 500), 2),
            "size": 0 if signal == "HOLD" else random.randint(50, 500),
            "confidence": round(random.uniform(0.7, 0.95), 2),
            "timestamp": now.isoformat()
        })
    
    # Update market data
    live_data["market_data"] = [
        {
            "symbol": "SPY",
            "bid": 432.48,
            "ask": 432.52,
            "last": round(432.50 + random.uniform(-2, 2), 2),
            "chg": round(random.uniform(-5, 8), 2),
            "chgPct": round(random.uniform(-2, 3), 2),
            "vol": f"{random.uniform(20, 60):.1f}M",
            "high": 434.12,
            "low": 428.34
        },
        {
            "symbol": "QQQ",
            "bid": 368.72,
            "ask": 368.78,
            "last": round(368.75 + random.uniform(-3, 3), 2),
            "chg": round(random.uniform(-4, 6), 2),
            "chgPct": round(random.uniform(-1.5, 2.5), 2),
            "vol": f"{random.uniform(15, 45):.1f}M",
            "high": 371.45,
            "low": 367.23
        },
        {
            "symbol": "IWM",
            "bid": 198.45,
            "ask": 198.51,
            "last": round(198.48 + random.uniform(-1.5, 1.5), 2),
            "chg": round(random.uniform(-2, 3), 2),
            "chgPct": round(random.uniform(-1, 2), 2),
            "vol": f"{random.uniform(10, 35):.1f}M",
            "high": 199.23,
            "low": 196.78
        },
        {
            "symbol": "VIX",
            "bid": 16.23,
            "ask": 16.28,
            "last": round(16.25 + random.uniform(-1, 1), 2),
            "chg": round(random.uniform(-1, 1), 2),
            "chgPct": round(random.uniform(-5, 5), 2),
            "vol": f"{random.uniform(5, 25):.1f}M",
            "high": 17.12,
            "low": 15.89
        },
        {
            "symbol": "AAPL",
            "bid": 178.23,
            "ask": 178.27,
            "last": round(178.25 + random.uniform(-2, 2), 2),
            "chg": round(random.uniform(-3, 4), 2),
            "chgPct": round(random.uniform(-2, 3), 2),
            "vol": f"{random.uniform(25, 80):.1f}M",
            "high": 180.12,
            "low": 176.45
        },
        {
            "symbol": "MSFT",
            "bid": 342.16,
            "ask": 342.20,
            "last": round(342.18 + random.uniform(-3, 3), 2),
            "chg": round(random.uniform(-4, 5), 2),
            "chgPct": round(random.uniform(-1.5, 2), 2),
            "vol": f"{random.uniform(20, 60):.1f}M",
            "high": 345.67,
            "low": 339.23
        }
    ]
    
    # Update system stats
    live_data["system_stats"] = {
        "TOTAL_PNL": live_data["total_pnl"],
        "DAY_PNL": live_data["day_pnl"],
        "POSITIONS": len([p for p in live_data["positions"] if p["qty"] != 0]),
        "EXEC_LAT": round(random.uniform(1.2, 3.5), 2),
        "SYS_UPTIME": round(99.95 + random.uniform(0, 0.04), 2),
        "DATA_FEEDS": 8,
        "ACTIVE_AGENTS": len([s for s in live_data["agent_signals"] if s["signal"] != "HOLD"]),
        "ORDERS_TODAY": random.randint(15, 45)
    }
    
    # Update risk metrics
    live_data["risk_metrics"] = [
        {
            "metric": "PORT_VAR_95",
            "value": round(-23450.67 + random.uniform(-5000, 5000), 2),
            "limit": -50000,
            "status": "OK"
        },
        {
            "metric": "MAX_DD",
            "value": round(-41234.89 + random.uniform(-8000, 8000), 2),
            "limit": -100000,
            "status": "OK"
        },
        {
            "metric": "SHARPE_RTD",
            "value": round(2.84 + random.uniform(-0.5, 0.5), 2),
            "limit": 2.0,
            "status": "OK"
        },
        {
            "metric": "BETA_SPX",
            "value": round(0.67 + random.uniform(-0.2, 0.2), 2),
            "limit": 1.0,
            "status": "OK"
        },
        {
            "metric": "GAMMA_EXP",
            "value": round(12456.78 + random.uniform(-2000, 2000), 2),
            "limit": 50000,
            "status": "OK"
        },
        {
            "metric": "THETA_DECAY",
            "value": round(-234.56 - random.uniform(0, 50), 2),
            "limit": -1000,
            "status": "OK"
        }
    ]
    
    live_data["last_update"] = now

# Background task to simulate live data updates
async def update_live_data():
    """Background task to continuously update live data"""
    while True:
        simulate_live_data()
        await asyncio.sleep(2)  # Update every 2 seconds

# API Endpoints

@router.get("/status")
async def get_dashboard_status():
    """Get dashboard connection status"""
    return {
        "status": "connected",
        "timestamp": datetime.now().isoformat(),
        "last_update": live_data["last_update"].isoformat(),
        "data_age_seconds": (datetime.now() - live_data["last_update"]).seconds
    }

@router.get("/portfolio")
async def get_portfolio_data():
    """Get portfolio performance and summary data"""
    return {
        "totalValue": live_data["portfolio_value"],
        "totalPnL": live_data["total_pnl"],
        "dayPnL": live_data["day_pnl"],
        "performance": live_data["performance_history"],
        "timestamp": datetime.now().isoformat()
    }

@router.get("/positions")
async def get_positions():
    """Get current portfolio positions"""
    return {
        "positions": live_data["positions"],
        "count": len(live_data["positions"]),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/signals")
async def get_agent_signals():
    """Get current agent signals"""
    return {
        "signals": live_data["agent_signals"],
        "active_signals": len([s for s in live_data["agent_signals"] if s["signal"] != "HOLD"]),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/market")
async def get_market_data():
    """Get live market data"""
    return {
        "market": live_data["market_data"],
        "symbols": len(live_data["market_data"]),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    stats_list = [
        {"label": key, "value": value, "format": get_stat_format(key)}
        for key, value in live_data["system_stats"].items()
    ]
    
    return {
        "stats": stats_list,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/risk")
async def get_risk_metrics():
    """Get risk management metrics"""
    return {
        "metrics": live_data["risk_metrics"],
        "timestamp": datetime.now().isoformat()
    }

@router.get("/live-feed")
async def get_live_feed():
    """Get complete live data feed for dashboard"""
    return {
        "portfolio": {
            "totalValue": live_data["portfolio_value"],
            "totalPnL": live_data["total_pnl"],
            "dayPnL": live_data["day_pnl"],
            "performance": live_data["performance_history"][-20:]  # Last 20 points
        },
        "positions": live_data["positions"],
        "signals": live_data["agent_signals"],
        "market": live_data["market_data"],
        "stats": live_data["system_stats"],
        "risk": live_data["risk_metrics"],
        "timestamp": datetime.now().isoformat(),
        "status": "live"
    }

def get_stat_format(key: str) -> str:
    """Get format type for system statistics"""
    if "PNL" in key:
        return "currency"
    elif "LAT" in key:
        return "ms"
    elif "UPTIME" in key:
        return "percent"
    else:
        return "number"

# Initialize live data on startup
simulate_live_data()